"""Gate 1: my torch graph against NeMo's own tensors, on the fixtures' real audio.

  python verify/g1_torch_vs_nemo.py [--limit N] [--no-fold]

Fixtures: oracle/make_fixtures.py (oracle venv). Each fixture carries NeMo's mel, its pooled [6144] vector and its emb [192], captured from the publisher's
modules. This gate feeds the same mel and compares, so a mismatch is mine and not the preprocessor's.
Also checks that `fold_pool_concat` is arithmetically free (fold vs no-fold on the same input).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import model as M  # noqa: E402
from tnl import paths as P  # noqa: E402

SNAP = P.SNAPSHOT
FIX = P.FIXTURES


def cos(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sd = M.load_state_dict(SNAP)
    fold = M.TitaNet.from_snapshot(SNAP, sd, fold_pool_concat=True).double()
    plain = M.TitaNet.from_snapshot(SNAP, sd, fold_pool_concat=False).double()

    rows = [json.loads(l) for l in (FIX / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    worst = {"emb_cos": 1.0, "pooled_cos": 1.0, "emb_rel": 0.0, "fold_delta": 0.0}
    recs = []
    for r in rows:
        z = np.load(FIX / r["file"])
        mel = torch.tensor(z["mel"], dtype=torch.float64)[None]
        n_valid = int(z["mel_len"])
        valid = torch.zeros(1, 1, mel.shape[2], dtype=torch.float64)
        valid[0, 0, :n_valid] = 1.0
        with torch.no_grad():
            n = valid.sum(dim=2)
            x = fold.enc(mel, valid, n)
            pooled = fold.pool(x, valid, n)
            emb = fold.head(pooled)
            emb_plain = plain(mel, valid)
        e = emb[0].numpy()
        ec, pc = cos(e, z["emb"]), cos(pooled[0].numpy(), z["pooled"])
        rel = float(np.abs(e - z["emb"]).max() / (np.abs(z["emb"]).max() + 1e-30))
        fd = float(np.abs(emb_plain[0].numpy() - e).max())
        worst["emb_cos"] = min(worst["emb_cos"], ec)
        worst["pooled_cos"] = min(worst["pooled_cos"], pc)
        worst["emb_rel"] = max(worst["emb_rel"], rel)
        worst["fold_delta"] = max(worst["fold_delta"], fd)
        recs.append({"id": r["id"], "seconds": r["seconds"], "emb_cos": ec, "pooled_cos": pc, "emb_rel": rel,
                     "fold_delta": fd})

    summary = {"n": len(recs), "worst": worst,
               "pass": worst["emb_cos"] > 1 - 1e-9 and worst["pooled_cos"] > 1 - 1e-9 and worst["fold_delta"] < 1e-9}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
