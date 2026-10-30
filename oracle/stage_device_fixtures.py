"""Stage NeMo's mel and NeMo's embedding as flat binaries the iOS gate can read.

  python oracle/stage_device_fixtures.py --fixtures _work/fixtures_long --out _work/device_stage/LeanAssets

Writes, under <out>/audio/:
  manifest.jsonl   one object per utterance:
                   {"id", "mel": "mel/<name>.f32", "mel_len", "frames", "emb": "emb/<name>.f32", "seconds"}
  mel/<name>.f32   float32 little-endian, [80, frames] row-major (80 contiguous rows of `frames` samples)
  emb/<name>.f32   float32 little-endian, [192]

Why mel and not audio: the point of the on-device gate is whether the Swift host and the phone's Core AI runtime
reproduce the reference embedding, so both sides must be fed byte-identical features.  Shipping NeMo's own
preprocessor output keeps a mel front end out of the comparison; verifying a front end is a separate gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (args.fixtures / "index.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows.sort(key=lambda r: r["seconds"])
    if args.limit:
        rows = rows[: args.limit]

    audio = args.out / "audio"
    (audio / "mel").mkdir(parents=True, exist_ok=True)
    (audio / "emb").mkdir(parents=True, exist_ok=True)

    manifest, total = [], 0
    for r in rows:
        z = np.load(args.fixtures / r["file"])
        mel = np.ascontiguousarray(z["mel"], dtype="<f4")
        emb = np.ascontiguousarray(z["emb"], dtype="<f4")
        assert mel.shape[0] == 80 and emb.shape == (192,), (mel.shape, emb.shape)
        name = r["file"].replace(".npz", "")
        mel.tofile(audio / "mel" / f"{name}.f32")
        emb.tofile(audio / "emb" / f"{name}.f32")
        total += mel.nbytes + emb.nbytes
        manifest.append({"id": r["id"], "mel": f"mel/{name}.f32", "mel_len": int(z["mel_len"]),
                         "frames": int(mel.shape[1]), "emb": f"emb/{name}.f32",
                         "seconds": r["seconds"]})
    (audio / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")

    # read back one file exactly as the app will, so a layout mistake fails here and not on the phone
    m0 = manifest[0]
    back = np.fromfile(audio / m0["mel"], dtype="<f4").reshape(80, m0["frames"])
    z0 = np.load(args.fixtures / (m0["mel"].split("/")[1].replace(".f32", ".npz")))
    assert np.array_equal(back, z0["mel"].astype("<f4")), "mel round-trip mismatch"
    print(f"{len(manifest)} utterances, {min(m['seconds'] for m in manifest):.2f}-"
          f"{max(m['seconds'] for m in manifest):.2f}s, {total / 1e6:.1f} MB -> {audio}")


if __name__ == "__main__":
    main()
