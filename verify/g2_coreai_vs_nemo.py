"""Gate 2: the exported bundle, on the Core AI runtime, against NeMo's own embeddings on real audio.

  python verify/g2_coreai_vs_nemo.py --bundle _work/bundles/w_scan_f512 --fixtures _work/fixtures_long \
         --compute gpu [--path-both] [--out FILE]

The fixtures carry NeMo's preprocessor output and NeMo's embedding for the same utterance, so this measures the
conversion only -- graph authoring, precision arm, windowing and host reconstruction -- and not a mel frontend.
Reports cosine to NeMo per utterance, plus the number of graph calls the host made, and (with --path-both, on a
bundle that has `once`) the `once` vs `scan` agreement where both paths are legal.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import runtime as RT  # noqa: E402


async def run(args) -> int:
    rec = json.loads((args.bundle / "export.json").read_text())
    emb = await RT.Embedder.load(args.bundle, RT.options(args.compute))
    if args.sweeps:
        emb.sweeps = args.sweeps
    rows = [json.loads(l) for l in (args.fixtures / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["seconds"])
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]

    recs = []
    for k, r in enumerate(rows):
        z = np.load(args.fixtures / r["file"])
        n = int(z["mel_len"])
        mel = z["mel"].astype(np.float32)
        before = emb.calls
        t0 = time.monotonic()
        e = await emb.embed(mel, n)
        dt = time.monotonic() - t0
        row = {"id": r["id"], "seconds": r["seconds"], "n": n, "cos": RT.cosine(e, z["emb"]),
               "calls": emb.calls - before, "latency_s": round(dt, 3),
               "path": "once" if (n <= emb.F and "once" in emb.entrypoints) else "scan",
               "finite": bool(np.isfinite(e).all())}
        if args.path_both and "once" in emb.entrypoints and n <= emb.F:
            e2 = await emb.embed(mel, n, force="scan")
            row["once_vs_scan_cos"] = RT.cosine(e, e2)
            row["scan_cos"] = RT.cosine(e2, z["emb"])
        recs.append(row)
        print(f"[{k + 1}/{len(rows)}] {r['id']} {r['seconds']:5.1f}s n={n:5d} {row['path']:5s} "
              f"calls={row['calls']:3d} cos={row['cos']:.8f} {row['latency_s']:.2f}s", flush=True)

    c = np.array([r["cos"] for r in recs], float)
    by_path = {}
    for p in ("once", "scan"):
        v = np.array([r["cos"] for r in recs if r["path"] == p], float)
        if len(v):
            by_path[p] = {"n": int(len(v)), "min": float(v.min()), "mean": float(v.mean())}
    summary = {"bundle": str(args.bundle), "arm": rec["arm"], "shape": rec["shape"],
               "entrypoints": rec["entrypoints"], "compute": args.compute, "sweeps": emb.sweeps,
               "n": len(recs),
               "cos_min": float(c.min()), "cos_mean": float(c.mean()), "cos_p10": float(np.percentile(c, 10)),
               "by_path": by_path,
               "all_finite": all(r["finite"] for r in recs),
               "calls_max": max(r["calls"] for r in recs),
               "latency_mean_s": float(np.mean([r["latency_s"] for r in recs])),
               "latency_max_s": float(np.max([r["latency_s"] for r in recs]))}
    if any("once_vs_scan_cos" in r for r in recs):
        v = [r["once_vs_scan_cos"] for r in recs if "once_vs_scan_cos" in r]
        summary["once_vs_scan_cos_min"] = float(min(v))
    print("\n" + json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": recs}, indent=1) + "\n")
    return 0 if (summary["all_finite"] and summary["cos_min"] > args.bar) else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--compute", default="gpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N (length-sorted) utterances; lets a long suite run in shards, one "
                         "process each, around the runtime's cumulative NDArray pool limit")
    ap.add_argument("--bar", type=float, default=0.9999, help="the current implementation measures 0.99998 vs NeMo")
    ap.add_argument("--path-both", action="store_true")
    ap.add_argument("--sweeps", type=int, default=0, help="override the gate fixed-point sweeps (default 5)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
