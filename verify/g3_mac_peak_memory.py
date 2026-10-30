"""Gate 3: peak process memory of one bundle on the Mac Core AI runtime.

  python verify/g3_mac_peak_memory.py --bundle DIR --compute gpu --frames 408,828,2052,6472

Reports `ru_maxrss` (peak resident set, the high-water mark the kernel recorded for this process) around a load
and a full run at each utterance length.  Run one bundle per process -- a peak is only meaningful if nothing
else was loaded first.

Treat this as a WEAK cross-check, not the budget.  Measured here, a bare CPython + torch + coreai process is
already ~190 MB resident before any bundle is loaded, so the process peak is dominated by the host interpreter
and the Mac GPU runtime's own allocator, and it did not rank bundles the way the phone does (f256 measured a
larger delta than f512).  The figure that decides the <120 MB budget is the iPhone's `phys_footprint`, measured
by apps/LeanGate; `peak_delta_mb` below is reported only because a delta is less misleading than an absolute.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import resource
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import runtime as RT  # noqa: E402

MB = 1024 * 1024


def peak_mb() -> float:
    """Peak resident set in MB.  On Darwin getrusage reports ru_maxrss in BYTES (on Linux, in kilobytes)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / MB if sys.platform == "darwin" else v / 1024


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / MB
    except Exception:
        return -1.0


async def run(args) -> int:
    stages = {"launch": {"peak": peak_mb(), "rss": rss_mb()}}
    emb = await RT.Embedder.load(args.bundle, RT.options(args.compute))
    stages["loaded"] = {"peak": peak_mb(), "rss": rss_mb()}
    rng = np.random.default_rng(0)
    per_n = {}
    for n in args.frames:
        mel = (rng.standard_normal((80, n)) * 0.5).astype(np.float32)
        e = await emb.embed(mel, n)
        assert np.isfinite(e).all()
        per_n[n] = {"peak": peak_mb(), "rss": rss_mb(), "calls": emb.calls}
        print(f"  n={n:5d} peak={per_n[n]['peak']:8.1f} MB rss={per_n[n]['rss']:8.1f} MB", flush=True)
    rec = json.loads((args.bundle / "export.json").read_text())
    out = {"bundle": str(args.bundle), "arm": rec["arm"], "shape": rec["shape"],
           "entrypoints": rec["entrypoints"], "compute": args.compute,
           "stages": stages, "per_n": per_n, "peak_mb": peak_mb(),
           "peak_delta_mb": peak_mb() - stages["launch"]["peak"],
           "note": "weak cross-check; host interpreter baseline dominates. See apps/LeanGate for phys_footprint."}
    print(json.dumps(out, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--compute", default="gpu")
    ap.add_argument("--frames", default="408,828,2052,6472")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.frames = [int(x) for x in args.frames.split(",")]
    sys.exit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
