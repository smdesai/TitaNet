"""TitaNet-Large on Core AI: 16 kHz mono audio in, NeMo's 192-d speaker embedding out. numpy only, no torch.

    python titanet_coreai.py a.wav b.wav --repo DIR [--compute gpu|ane|cpu|default]

    from titanet_coreai import TitaNet
    tn = await TitaNet.load("path/to/TitaNet-Large-CoreAI")
    emb = await tn.embed_samples(samples)        # float32 [N] at 16 kHz -> float32 [192]

The bundle compiles ONE 512-frame (5.12 s) window. Utterances of any length are handled by sweeping that window and
rebuilding TitaNet's three global reductions on the host, exactly:

  * five SqueezeExcite gates: a triangular fixed point, made exact by five sweeps over the windows
    (sweep k makes gate k-1 exact; fewer sweeps look converged but are not);
  * the pooling mean / std of the encoder output: one pass of running sums;
  * the attention softmax over time: one pass, combined across windows by running maximum.

Each window carries 46 frames of real context on each side (the conv stack's receptive field is 93 frames) and only
its interior is kept, so the interiors tile the utterance exactly once. Host sums are float64.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
from pathlib import Path

import numpy as np

# ---- the mel front end: NeMo FilterbankFeatures, eval mode ------------------------------------------------------

N_FFT, HOP, WIN, N_MELS = 512, 160, 400, 80
PREEMPH = np.float32(0.97)
LOG_GUARD = np.float32(2.0 ** -24)
MEL_STD_EPS = np.float32(1e-5)


class MelFrontend:
    """samples float32 [N] (16 kHz) -> (mel float32 [80, N // 160 + 1], valid frames N // 160).

    preemphasis 0.97; centred STFT (n_fft 512, hop 160, Hann(400) centred in 512), DFT summed in float64; power;
    slaney mel [80, 257]; log(x + 2^-24); per-feature normalization over the valid frames (unbiased std + 1e-5).
    The filterbank and window are the checkpoint's own buffers (frontend/*.f32). Dither is training-only.
    """

    def __init__(self, fb: np.ndarray, window: np.ndarray):
        assert fb.shape == (N_MELS, N_FFT // 2 + 1) and window.shape == (WIN,)
        self.fb = fb.astype(np.float64)
        w = np.zeros(N_FFT, np.float64)
        off = (N_FFT - WIN) // 2
        w[off:off + WIN] = window.astype(np.float64)
        self.window = w
        k = np.arange(N_FFT // 2 + 1)[:, None]
        j = np.arange(N_FFT)[None, :]
        ang = 2.0 * np.pi * ((k * j) % N_FFT) / N_FFT      # the angle reduced exactly before the trig call
        self.cos, self.sin = np.cos(ang), -np.sin(ang)

    @classmethod
    def load(cls, frontend_dir: Path) -> "MelFrontend":
        fb = np.fromfile(frontend_dir / "mel_fb.f32", "<f4").reshape(N_MELS, N_FFT // 2 + 1)
        return cls(fb, np.fromfile(frontend_dir / "mel_window.f32", "<f4"))

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        x = np.asarray(x, np.float32)
        n_valid = len(x) // HOP
        if len(x) == 0:
            return np.zeros((N_MELS, 1), np.float32), 0
        y = np.empty_like(x)
        y[0] = x[0]
        y[1:] = x[1:] - PREEMPH * x[:-1]
        padded = np.concatenate([np.zeros(N_FFT // 2, np.float32), y, np.zeros(N_FFT // 2, np.float32)])
        n_frames = 1 + (len(padded) - N_FFT) // HOP
        idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
        frames = padded[idx].astype(np.float64) * self.window            # [frames, 512]
        re = (frames @ self.cos.T).astype(np.float32)
        im = (frames @ self.sin.T).astype(np.float32)
        mag = np.sqrt(re * re + im * im)
        power = mag * mag
        mel = (self.fb @ power.T.astype(np.float64)).astype(np.float32)  # [80, frames]
        logmel = np.log(mel.astype(np.float64) + float(LOG_GUARD)).astype(np.float32)
        return _normalize(logmel, n_valid), n_valid


def _normalize(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(x)
    if n == 0:
        return out
    v = x[:, :n].astype(np.float64)
    ref = x[:, 0].astype(np.float64)
    centered = (v - ref[:, None]).astype(np.float32).astype(np.float64)
    mean = (ref + centered.sum(1) / n).astype(np.float32)
    dev = (v - mean[:, None].astype(np.float64)).astype(np.float32).astype(np.float64)
    std = np.sqrt((dev * dev).sum(1) / (n - 1.0)).astype(np.float32) if n > 1 else np.zeros(x.shape[0], np.float32)
    out[:, :n] = (x[:, :n] - mean[:, None]) / (std + MEL_STD_EPS)[:, None]
    return out


# ---- the windowed host ------------------------------------------------------------------------------------------

C_OUT = 3072                                   # encoder output channels
SE_CHANNELS = (1024, 1024, 1024, 1024, 3072)   # the five SqueezeExcite blocks
STD_EPS = 1e-10
SWEEPS = 5
GC_EVERY = 32


def windows(n: int, F: int, margin: int) -> list[tuple[int, int, int]]:
    """[(start, keep_lo, keep_hi)] in frame coordinates: the kept interiors tile [0, n) exactly once."""
    if n <= F:
        return [(0, 0, n)]
    step = F - 2 * margin
    out, s = [], 0
    while True:
        end = s + F
        out.append((s, 0 if s == 0 else s + margin, n if end >= n else end - margin))
        if end >= n:
            return out
        s += step


def options(name: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    return {"default": SpecializationOptions.default(), "cpu": SpecializationOptions.cpu_only(),
            "gpu": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()),
            "ane": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())}[name]


class TitaNet:
    @classmethod
    async def load(cls, repo: str | Path, compute: str = "gpu") -> "TitaNet":
        from coreai.runtime import AIModel
        self = cls()
        repo = Path(repo)
        rec = json.loads((repo / "export.json").read_text())
        self.F = int(rec["shape"])
        self.margin = int(rec["margin"])
        self.mel = MelFrontend.load(repo / "frontend")
        self.model = await AIModel.load(repo / rec["aimodel"], options(compute))
        self.scan = self.model.load_function("scan")
        self.head = self.model.load_function("head")
        self.calls = 0
        return self

    async def embed_samples(self, samples: np.ndarray) -> np.ndarray:
        """float32 [N] at 16 kHz, mono -> embedding float32 [192]."""
        mel, n = self.mel(samples)
        return await self.embed_mel(mel, n)

    async def embed_mel(self, mel: np.ndarray, n: int) -> np.ndarray:
        """mel [80, >= n] with the utterance's n valid frames first -> embedding float32 [192]."""
        if n < 1:
            raise ValueError("the clip is shorter than one 10 ms frame")
        wins = windows(n, self.F, self.margin)
        n_inv = 1.0 / n
        zero = np.zeros(C_OUT, np.float32)
        one = np.ones(C_OUT, np.float32)

        sums = [np.zeros(c, np.float64) for c in SE_CHANNELS]
        for _ in range(SWEEPS):
            new = [np.zeros(c, np.float64) for c in SE_CHANNELS]
            for w in wins:
                o = await self._scan(mel, n, w, n_inv, sums, zero, one)
                for j in range(5):
                    new[j] += o[f"se{j}"]
            sums = new

        sx = np.zeros(C_OUT)
        sxx = np.zeros(C_OUT)
        for w in wins:
            o = await self._scan(mel, n, w, n_inv, sums, zero, one)
            sx += o["x_sum"]
            sxx += o["x_sqsum"]
        pmean = sx / n
        pstd = np.sqrt(np.maximum(sxx / n - pmean * pmean, STD_EPS))

        mx = None
        den, ax, axx = np.zeros(C_OUT), np.zeros(C_OUT), np.zeros(C_OUT)
        for w in wins:
            o = await self._scan(mel, n, w, n_inv, sums, pmean.astype(np.float32), pstd.astype(np.float32))
            lmax = o["lmax"]
            if mx is None:
                mx, scale, f = lmax, np.zeros(C_OUT), np.ones(C_OUT)
            else:
                m_new = np.maximum(mx, lmax)
                scale, f, mx = np.exp(mx - m_new), np.exp(lmax - m_new), m_new
            den = den * scale + o["den"] * f
            ax = ax * scale + o["ax"] * f
            axx = axx * scale + o["axx"] * f
        mu = ax / den
        sg = np.sqrt(np.maximum(axx / den - mu * mu, STD_EPS))

        out = await self.head({"mu": _nd(mu[None]), "sg": _nd(sg[None])})
        self.calls += 1
        return out["emb"].numpy().astype(np.float32)[0]

    async def _scan(self, mel, n, w, n_inv, sums, pmean, pstd):
        s, lo, hi = w
        F = self.F
        take = max(0, min(F, n - s))
        mw = np.zeros((1, N_MELS, F), np.float32)
        mw[0, :, :take] = mel[:, s:s + take]
        vw = np.zeros((1, 1, F), np.float32)
        vw[0, 0, :take] = 1
        kw = np.zeros((1, 1, F), np.float32)
        kw[0, 0, lo - s:hi - s] = 1
        feeds = {"mel": _nd(mw), "valid": _nd(vw), "keep": _nd(kw), "n_inv": _nd(np.array([[n_inv]])),
                 "mean": _nd(pmean[None]), "std": _nd(pstd[None])}
        for j in range(5):
            feeds[f"s{j}"] = _nd(sums[j][None])
        out = await self.scan(feeds)
        self.calls += 1
        res = {k: v.numpy().astype(np.float64)[0] for k, v in out.items()}
        # The Python runtime returns NDArray storage to its pool only on collection; collect regularly or a long
        # utterance (hundreds of calls) can exhaust it. Swift hosts are not affected.
        del out, feeds
        if self.calls % GC_EVERY == 0:
            gc.collect()
        return res


def _nd(a):
    from coreai.runtime import NDArray
    return NDArray(np.ascontiguousarray(a, dtype=np.float32))


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def score(a, b) -> float:
    """NeMo's verification score, (cos + 1) / 2; the same speaker at >= 0.7."""
    return (cosine(a, b) + 1.0) / 2.0


def read_wav(path: str | Path) -> np.ndarray:
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{path}: {sr} Hz; resample to 16 kHz first (the model was trained on 16 kHz)")
    return x


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Compare two recordings with TitaNet-Large on Core AI.")
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--repo", required=True, help="a local copy of smdesai/TitaNet-Large-CoreAI: "
                    "hf download smdesai/TitaNet-Large-CoreAI --local-dir DIR")
    ap.add_argument("--compute", default="gpu", choices=["gpu", "ane", "cpu", "default"])
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    tn = await TitaNet.load(args.repo, args.compute)
    ea = await tn.embed_samples(read_wav(args.a))
    eb = await tn.embed_samples(read_wav(args.b))
    sc = score(ea, eb)
    print(f"score {sc:.4f} (cos {2 * sc - 1:.4f}): {'same speaker' if sc >= args.threshold else 'different speakers'}")


if __name__ == "__main__":
    asyncio.run(_main())
