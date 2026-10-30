"""Freeze the tensor fixtures verify/g1 reads (the coreai venv has no NeMo): _work/fixtures/<id>.npz + index.jsonl.

  _work/venvs/oracle/bin/python oracle/make_fixtures.py [--per-suite 40] [--threads 4]

Per suite, --per-suite utterances evenly spaced in duration (shortest to longest). Per utterance, from NeMo fp32
(oracle/nemo_ref.embed): samples (float32), mel [80, frames] (NeMo's preprocessor output with its pad_to-16 frames,
the reference for the host mel), mel_len, emb [192], pooled [6144] (the attentive statistics), and the per-block
max |x| over the valid frames.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from oracle import nemo_ref as R  # noqa: E402
from tnl import paths as P  # noqa: E402

SUITES = ["vox1_test", "ls_clean", "ls_other"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-suite", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = P.FIXTURES
    out.mkdir(parents=True, exist_ok=True)
    model = R.load_model()
    index = []
    for suite in SUITES:
        rows = sorted((json.loads(line) for line in (P.EVAL / suite / "manifest.jsonl").open()),
                      key=lambda r: r["seconds"])
        for i in np.linspace(0, len(rows) - 1, args.per_suite).round().astype(int):
            r = rows[i]
            path = out / f"{suite}__{r['id'].replace('/', '+')}.npz"
            x = R.read_audio(P.EVAL / suite / r["audio"])
            o = R.embed(model, x, layers=True)
            block_max = [float(b.abs().max()) for b in o["blocks"]]
            np.savez(path, samples=x, mel=o["mel"].numpy(), mel_len=o["mel_len"], emb=o["emb"].numpy(),
                     pooled=o["pooled"].numpy(), block_max=np.array(block_max))
            rec = {"file": path.name, "suite": suite, "id": r["id"], "speaker": r["speaker"], "seconds": r["seconds"],
                   "mel_len": o["mel_len"], "block_max": max(block_max), "pooled_max": float(o["pooled"].abs().max())}
            index.append(rec)
            print(json.dumps(rec), flush=True)
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))


if __name__ == "__main__":
    main()
