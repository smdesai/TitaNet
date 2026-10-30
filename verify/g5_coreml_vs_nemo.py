"""Gate 5: is the Core ML export the model? The windowed host of tnl/runtime.py, calling Core ML instead of Core AI,
over NeMo's own mel for the length-stratified fixtures, against NeMo's own embedding.

  _work/venvs/coreml/bin/python verify/g5_coreml_vs_nemo.py --bundle _work/coreml/c_scan_f512_fp32 \
      --fixtures _work/fixtures_long --compute cpu_and_gpu

compute: cpu_only | cpu_and_gpu | cpu_and_ne | all. Prints one line per utterance and a JSON summary, and exits
non-zero unless every embedding is finite and the worst cosine clears the bar for the export's precision (BARS, or
--bar).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import model as M  # noqa: E402
from tnl import paths as P  # noqa: E402
from tnl.stream import windows  # noqa: E402

SWEEPS = 5

# Pass bars on the worst cosine to NeMo, per precision, each a little under what the 40 fixtures measure:
# fp32 0.9999999992 (CPU) / 0.99999998 (GPU); fp16c 0.99682 (ANE) / 0.99966 (GPU); fp16 0.98529 (ANE).
BARS = {"fp32": 0.9999, "fp16c": 0.996, "fp16": 0.98}


class CoreMLHost:
    def __init__(self, bundle: Path, compute: str):
        import coremltools as ct
        rec = json.loads((bundle / "export.json").read_text())
        cu = {"cpu_only": ct.ComputeUnit.CPU_ONLY, "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
              "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE, "all": ct.ComputeUnit.ALL}[compute]
        t0 = time.monotonic()
        self.scan = ct.models.MLModel(str(bundle / rec["models"]["scan"]["file"]), compute_units=cu)
        self.head = ct.models.MLModel(str(bundle / rec["models"]["head"]["file"]), compute_units=cu)
        self.load_s = time.monotonic() - t0
        self.F, self.margin = int(rec["shape"]), int(rec["margin"])
        self.moment_scale = float(rec.get("moment_scale", 1.0))
        self.calls = 0

    def _scan(self, mel, n, w, n_inv, sums, pmean, pstd):
        s, lo, hi = w
        F = self.F
        take = max(0, min(F, n - s))
        mw = np.zeros((1, M.N_MELS, F), np.float32)
        mw[0, :, :take] = mel[:, s:s + take]
        vw = np.zeros((1, 1, F), np.float32)
        vw[0, 0, :take] = 1
        kw = np.zeros((1, 1, F), np.float32)
        kw[0, 0, lo - s:hi - s] = 1
        feeds = {"mel": mw, "valid": vw, "keep": kw, "n_inv": np.array([[n_inv]], np.float32),
                 "mean": pmean[None].astype(np.float32), "std": pstd[None].astype(np.float32)}
        for j in range(5):
            feeds[f"s{j}"] = sums[j][None].astype(np.float32)
        self.calls += 1
        return {k: v.astype(np.float64)[0] for k, v in self.scan.predict(feeds).items()}

    def embed(self, mel: np.ndarray, n: int) -> np.ndarray:
        wins = windows(n, self.F, self.margin)
        n_inv = 1.0 / n
        zero, one = np.zeros(M.C_OUT), np.ones(M.C_OUT)
        sums = [np.zeros(c) for c in M.SE_CHANNELS]
        for _ in range(SWEEPS):
            new = [np.zeros(c) for c in M.SE_CHANNELS]
            for w in wins:
                o = self._scan(mel, n, w, n_inv, sums, zero, one)
                for j in range(5):
                    new[j] += o[f"se{j}"]
            sums = new
        sx, sxx = np.zeros(M.C_OUT), np.zeros(M.C_OUT)
        for w in wins:
            o = self._scan(mel, n, w, n_inv, sums, zero, one)
            sx += o["x_sum"]
            sxx += o["x_sqsum"]
        pmean = sx / n
        pstd = np.sqrt(np.maximum(sxx / n - pmean * pmean, M.STD_EPS))
        # attention moments: plain (ax, axx) or, from an fp16c export, centred on the pooling mean and scaled by S
        # (ax_c, axx_c). The mean goes in already rounded to fp16 so the graph subtracts exactly what is added back.
        pm16 = pmean.astype(np.float16).astype(np.float64)
        S = self.moment_scale
        mx = None
        den, ax, axx = np.zeros(M.C_OUT), np.zeros(M.C_OUT), np.zeros(M.C_OUT)
        centred = False
        for w in wins:
            o = self._scan(mel, n, w, n_inv, sums, pm16, pstd)
            centred = "ax_c" in o
            a1, a2 = (o["ax_c"], o["axx_c"]) if centred else (o["ax"], o["axx"])
            lmax = o["lmax"]
            if mx is None:
                mx, scale, f = lmax, np.zeros(M.C_OUT), np.ones(M.C_OUT)
            else:
                m_new = np.maximum(mx, lmax)
                scale, f, mx = np.exp(mx - m_new), np.exp(lmax - m_new), m_new
            den = den * scale + o["den"] * f
            ax = ax * scale + a1 * f
            axx = axx * scale + a2 * f
        if centred:
            q = ax / (S * den)
            sg = np.sqrt(np.maximum(axx / (S * S * den) - q * q, M.STD_EPS))
            mu = pm16 + q
        else:
            mu = ax / den
            sg = np.sqrt(np.maximum(axx / den - mu * mu, M.STD_EPS))
        self.calls += 1
        out = self.head.predict({"mu": mu[None].astype(np.float32), "sg": sg[None].astype(np.float32)})
        return out["emb"].astype(np.float32)[0]


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, default=P.FIXTURES_LONG)
    ap.add_argument("--compute", default="cpu_and_gpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bar", type=float, default=None, help="pass bar on the worst cosine (default: BARS)")
    ap.add_argument("--save", type=Path, help="write the embeddings (and NeMo's) to this .npz")
    args = ap.parse_args()

    host = CoreMLHost(args.bundle, args.compute)
    rows = [json.loads(l) for l in (args.fixtures / "index.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    cos, lat, embs, refs = [], [], [], []
    for i, r in enumerate(rows):
        z = np.load(args.fixtures / r["file"])
        n = int(z["mel_len"])
        host.calls = 0
        t = time.monotonic()
        e = host.embed(z["mel"].astype(np.float32), n)
        dt = time.monotonic() - t
        c = cosine(e, z["emb"])
        embs.append(e)
        refs.append(z["emb"])
        cos.append(c)
        lat.append(dt)
        print(f"[{i + 1}/{len(rows)}] {r['id']:34s} {r['seconds']:5.1f}s n={n:5d} calls={host.calls:4d} "
              f"cos={c:.8f} {dt:.2f}s", flush=True)
    if args.save:
        np.savez(args.save, emb=np.stack(embs), nemo=np.stack(refs), ids=np.array([r["id"] for r in rows]))
    s = sorted(cos)
    print(json.dumps({"bundle": args.bundle.name, "compute": args.compute, "n": len(cos), "cos_min": s[0],
                      "cos_mean": float(np.mean(cos)), "cos_p10": s[len(s) // 10], "load_s": round(host.load_s, 2),
                      "latency_mean_s": round(float(np.mean(lat)), 3), "latency_max_s": round(max(lat), 3),
                      "all_finite": bool(np.all(np.isfinite(cos)))}, indent=1), flush=True)
    precision = json.loads((args.bundle / "export.json").read_text())["precision"]
    bar = args.bar if args.bar is not None else BARS[precision]
    ok = bool(np.all(np.isfinite(cos))) and s[0] > bar
    print(f"{'PASS' if ok else 'FAIL'}: worst cosine {s[0]:.10f} vs bar {bar} ({precision})", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
