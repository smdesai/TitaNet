"""Gate 6: a published Hugging Face repository, end to end from WAV, against NeMo's own embeddings.

  _work/venvs/coreai/bin/python verify/g6_hosts_vs_nemo.py --runtime coreai --repo DIR [--compute gpu]
  _work/venvs/coreml/bin/python verify/g6_hosts_vs_nemo.py --runtime coreml --repo DIR --variant fp32 [--compute gpu]

DIR is a local copy of smdesai/TitaNet-Large-CoreAI or -CoreML (hf download ... --local-dir DIR, or _work/hf/<name>
as export/package_hf.py stages it). Each utterance of the length-stratified fixtures (oracle/make_long_fixtures.py)
is read from its WAV and run through the standalone host in hosts/python (its own numpy mel front end, the windowed
host, the repository's model files), then compared with NeMo's `get_embedding` from the fixture. Unlike g2 / g5,
which feed NeMo's mel, this also covers the front end; a frame-count mismatch with NeMo is reported separately.
Exits non-zero unless the worst cosine clears the bar (Core AI and fp32 0.9999, fp16c 0.996; --bar overrides)
and no frame count differs.
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
sys.path.insert(0, str(ROOT / "hosts" / "python"))
from tnl import paths as P  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=("coreai", "coreml"), required=True)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--variant", default="fp32", choices=("fp32", "fp16c"), help="coreml only")
    ap.add_argument("--compute", default="gpu")
    ap.add_argument("--fixtures", type=Path, default=P.FIXTURES_LONG)
    ap.add_argument("--suite", type=Path, default=P.EVAL / "vox1_test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bar", type=float, default=None, help="pass bar on the worst cosine")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.runtime == "coreai":
        import titanet_coreai as T
        tn = await T.TitaNet.load(args.repo, args.compute)
        async def embed(x):
            return await tn.embed_samples(x)
    else:
        import titanet_coreml as T
        tn = T.TitaNet.load(args.repo, args.variant, args.compute)
        async def embed(x):
            return tn.embed_samples(x)

    audio = {r["id"]: r["audio"] for r in map(json.loads, (args.suite / "manifest.jsonl").read_text().splitlines()) if r}
    rows = [json.loads(l) for l in (args.fixtures / "index.jsonl").read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["seconds"])
    if args.limit:
        rows = rows[: args.limit]
    recs = []
    for i, r in enumerate(rows):
        ref = np.load(args.fixtures / r["file"])
        x = T.read_wav(args.suite / audio[r["id"]])
        t = time.monotonic()
        e = await embed(x)
        dt = time.monotonic() - t
        c = T.cosine(e, ref["emb"])
        frames_match = len(x) // 160 == int(ref["mel_len"])
        recs.append({"id": r["id"], "seconds": r["seconds"], "cos": c, "frames_match": frames_match, "latency_s": dt})
        print(f"[{i + 1}/{len(rows)}] {r['id']:34s} {r['seconds']:5.1f}s cos={c:.10f} {dt:.2f}s"
              f"{'' if frames_match else '  FRAME COUNT DIFFERS'}", flush=True)
    cos = sorted(x["cos"] for x in recs)
    summary = {"runtime": args.runtime, "variant": args.variant if args.runtime == "coreml" else None,
               "compute": args.compute, "repo": args.repo.name, "n": len(recs), "cos_min": cos[0],
               "cos_mean": float(np.mean(cos)), "frame_mismatches": sum(not x["frames_match"] for x in recs),
               "latency_mean_s": float(np.mean([x["latency_s"] for x in recs]))}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**summary, "rows": recs}, indent=1) + "\n")
    bar = args.bar if args.bar is not None else (0.996 if args.runtime == "coreml" and args.variant == "fp16c" else 0.9999)
    ok = cos[0] > bar and summary["frame_mismatches"] == 0
    print(f"{'PASS' if ok else 'FAIL'}: worst cosine {cos[0]:.10f} vs bar {bar}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
