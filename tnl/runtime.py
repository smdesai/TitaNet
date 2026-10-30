"""Drive an exported bundle on the Core AI runtime, with the host logic a shipping host would implement.

  emb = await Embedder.load(bundle_dir, options)          # titanet.aimodel + export.json
  e = await emb.embed(mel, n_valid)                       # [192] float32

Two paths, chosen by the utterance's length and by what the bundle exports:

  once   n <= F and the bundle has `once`: one call.  The latency fast path.
  scan   otherwise: the window is swept over the utterance and the model's three global reductions are rebuilt
         on the host.  Seven passes over the windows -- five to drive the SqueezeExcite gate fixed point (block
         j's mean depends only on gates < j, so sweep k makes gate k-1 exact and five sweeps make all five
         exact), one for the pooling mean/std, one for the online attention softmax.

Everything the host holds is a 3072-wide vector: no encoder output is ever kept for the whole utterance, so host
memory does not grow with duration.  Accumulation is float64 on the host even when the graph is fp16, because
summing thousands of frames is exactly where fp16 would lose the reduction.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np

from . import model as M
from .stream import windows

SWEEPS = 5                 # the triangular fixed point is exact after one sweep per SqueezeExcite
GC_EVERY = 32              # see _scan: the runtime's ioSurface NDArray pool is not reclaimed by refcount alone


def options(name: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions
    return {"default": SpecializationOptions.default(), "cpu": SpecializationOptions.cpu_only(),
            "gpu": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()),
            "ane": SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())}[name]


def bundle_path(d: Path, stem: str = "titanet") -> Path:
    compiled = sorted(d.glob(f"{stem}.*.aimodelc"))
    return compiled[0] if compiled else d / f"{stem}.aimodel"


class Embedder:
    @classmethod
    async def load(cls, d: Path, opts, path: Path | None = None):
        from coreai.runtime import AIModel
        self = cls()
        d = Path(d)
        self.rec = json.loads((d / "export.json").read_text())
        self.dtype = np.float16 if self.rec["compute_dtype"] == "float16" else np.float32
        self.F = int(self.rec["shape"])
        self.margin = int(self.rec.get("margin", M.MARGIN))
        self.entrypoints = list(self.rec["entrypoints"])
        self.model = await AIModel.load(path or bundle_path(d), opts)
        self.fns = {}
        self.calls = 0
        self.sweeps = SWEEPS
        return self

    def _fn(self, name: str):
        if name not in self.fns:
            self.fns[name] = self.model.load_function(name)
        return self.fns[name]

    def _nd(self, a):
        from coreai.runtime import NDArray
        return NDArray(np.ascontiguousarray(a, dtype=self.dtype))

    # ---- window tensors ---------------------------------------------------------------------
    def _window(self, mel: np.ndarray, n: int, s: int, keep: tuple[int, int]):
        F = self.F
        mw = np.zeros((1, M.N_MELS, F), np.float32)
        take = max(0, min(F, n - s))
        if take:
            mw[0, :, :take] = mel[:, s:s + take]
        vw = np.zeros((1, 1, F), np.float32)
        vw[0, 0, :take] = 1.0
        kw = np.zeros((1, 1, F), np.float32)
        kw[0, 0, keep[0] - s:keep[1] - s] = 1.0
        return mw, vw, kw

    async def _scan(self, mel, n, s, keep, n_inv, sums, pmean, pstd):
        mw, vw, kw = self._window(mel, n, s, keep)
        feeds = {"mel": self._nd(mw), "valid": self._nd(vw), "keep": self._nd(kw),
                 "n_inv": self._nd(np.array([[n_inv]], np.float32)),
                 "mean": self._nd(pmean[None]), "std": self._nd(pstd[None])}
        for j in range(5):
            feeds[f"s{j}"] = self._nd(sums[j][None])
        out = await self._fn("scan")(feeds)
        self.calls += 1
        res = {k: v.numpy().astype(np.float64)[0] for k, v in out.items()}
        # Each call builds 11 input and 11 output NDArrays. `NDArray.numpy()` copies (there is no writable view,
        # so buffers cannot be pre-allocated and reused), and the runtime's ioSurface pool is not returned by
        # Python refcounting alone: without this, a long utterance dies partway with
        #   NDArray+Pool.swift: Fatal error: Failed to allocate storage for NDArray with byteCount: 12288,
        #   sk: ioSurface, st: float32
        # observed at ~113 calls. The Swift host on the phone ran 274 calls with no such limit, so this is a
        # Python-host artefact, not a property of the graph.
        del out, feeds
        if self.calls % GC_EVERY == 0:
            gc.collect()
        return res

    # ---- the two paths ----------------------------------------------------------------------
    async def embed(self, mel: np.ndarray, n: int, force: str | None = None) -> np.ndarray:
        """mel [80, >= n] with the utterance's n valid frames first."""
        if n < 1:
            raise ValueError("no valid mel frame: the embedding is undefined")
        want = force or ("once" if (n <= self.F and "once" in self.entrypoints) else "scan")
        if want == "once":
            if "once" not in self.entrypoints:
                raise ValueError("this bundle has no `once` entrypoint")
            if n > self.F:
                raise ValueError(f"{n} frames exceed the {self.F}-frame window")
            mw, vw, _ = self._window(mel, n, 0, (0, n))
            out = await self._fn("once")({"mel": self._nd(mw), "valid": self._nd(vw)})
            self.calls += 1
            return out["emb"].numpy().astype(np.float32)[0]
        return await self._scan_embed(mel, n)

    async def _scan_embed(self, mel: np.ndarray, n: int) -> np.ndarray:
        wins = windows(n, self.F, self.margin)
        n_inv = 1.0 / float(n)
        zero_mean = np.zeros(M.C_OUT, np.float32)
        unit_std = np.ones(M.C_OUT, np.float32)

        # 1. the SqueezeExcite gate fixed point: `s_j` are summed means, divided inside the graph
        sums = [np.zeros(M.SE_CHANNELS[j], np.float64) for j in range(5)]
        for _ in range(self.sweeps):
            new = [np.zeros(M.SE_CHANNELS[j], np.float64) for j in range(5)]
            for s, lo, hi in wins:
                o = await self._scan(mel, n, s, (lo, hi), n_inv, sums, zero_mean, unit_std)
                for j in range(5):
                    new[j] += o[f"se{j}"]
            sums = new

        # 2. the pooling mean / std of the encoder output
        sx = np.zeros(M.C_OUT, np.float64)
        sxx = np.zeros(M.C_OUT, np.float64)
        for s, lo, hi in wins:
            o = await self._scan(mel, n, s, (lo, hi), n_inv, sums, zero_mean, unit_std)
            sx += o["x_sum"]
            sxx += o["x_sqsum"]
        pmean = sx / n
        pstd = np.sqrt(np.maximum(sxx / n - pmean * pmean, M.STD_EPS))

        # 3. the attention softmax over time, combined across windows by running maximum
        mx = None
        den = np.zeros(M.C_OUT, np.float64)
        ax = np.zeros(M.C_OUT, np.float64)
        axx = np.zeros(M.C_OUT, np.float64)
        for s, lo, hi in wins:
            o = await self._scan(mel, n, s, (lo, hi), n_inv, sums,
                                 pmean.astype(np.float32), pstd.astype(np.float32))
            lmax = o["lmax"]
            if mx is None:
                mx, scale, f = lmax, np.zeros(M.C_OUT), np.ones(M.C_OUT)
            else:
                m_new = np.maximum(mx, lmax)
                scale = np.exp(mx - m_new)
                f = np.exp(lmax - m_new)
                mx = m_new
            den = den * scale + o["den"] * f
            ax = ax * scale + o["ax"] * f
            axx = axx * scale + o["axx"] * f
        mu = ax / den
        sg = np.sqrt(np.maximum(axx / den - mu * mu, M.STD_EPS))

        out = await self._fn("head")({"mu": self._nd(mu[None]), "sg": self._nd(sg[None])})
        self.calls += 1
        return out["emb"].numpy().astype(np.float32)[0]


def cosine(a, b) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def score(a, b) -> float:
    """NeMo's verification score: (cos + 1) / 2."""
    return (cosine(a, b) + 1.0) / 2.0
