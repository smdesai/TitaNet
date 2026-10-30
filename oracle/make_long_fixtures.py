"""NeMo's own mel and embedding for a length-stratified sample of VoxCeleb1-test (oracle venv only).

  _work/venvs/oracle/bin/python oracle/make_long_fixtures.py [--out _work/fixtures_long] [--per-bin 8]

The shipped fixtures are all ~4 s, which cannot exercise a windowed port at all.  This draws a sample across the
corpus's real length range (3.9 - 69 s) and stores, per utterance, the preprocessor's mel, its valid length and
`get_embedding`'s embedding, so every later comparison is against the publisher's own numbers on real audio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import paths as P  # noqa: E402

NEMO_FILE = P.NEMO_FILE
SUITE = P.EVAL / "vox1_test"
BINS = [(3.5, 5.2), (5.2, 10.3), (10.3, 20.5), (20.5, 41.0), (41.0, 100.0)]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=P.FIXTURES_LONG)
    ap.add_argument("--per-bin", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nl
    nl.setLevel(nl.ERROR)

    rows = [json.loads(l) for l in (SUITE / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = np.random.default_rng(args.seed)
    pick = []
    for lo, hi in BINS:
        cand = [r for r in rows if lo <= r["seconds"] < hi]
        if not cand:
            continue
        idx = rng.choice(len(cand), size=min(args.per_bin, len(cand)), replace=False)
        pick += [cand[i] for i in sorted(idx)]
    print(f"{len(pick)} utterances, {min(r['seconds'] for r in pick):.2f}-{max(r['seconds'] for r in pick):.2f}s",
          flush=True)

    m = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(str(NEMO_FILE), map_location="cpu")
    m.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for i, r in enumerate(pick):
        wav = SUITE / r["audio"] if not str(r["audio"]).startswith("/") else Path(r["audio"])
        if not wav.exists():
            wav = SUITE / "wav" / r["audio"]
        x, sr = sf.read(wav)
        assert sr == 16000 and x.ndim == 1, (wav, sr, np.shape(x))
        sig = torch.from_numpy(x.astype(np.float32))[None]
        mel, mel_len = m.preprocessor(input_signal=sig, length=torch.tensor([sig.shape[1]]))
        enc, enc_len = m.encoder(audio_signal=mel, length=mel_len)
        _, emb = m.decoder(encoder_output=enc, length=enc_len)
        ref = m.get_embedding(str(wav)).cpu().numpy().ravel()
        e = emb[0].cpu().numpy()
        assert np.array_equal(e, ref), f"forward != get_embedding for {r['id']}"
        name = r["id"].replace("/", "+") + ".npz"
        np.savez(args.out / name, mel=mel[0].cpu().numpy().astype(np.float32), mel_len=int(mel_len),
                 emb=e.astype(np.float32), samples_n=sig.shape[1])
        index.append({"file": name, "id": r["id"], "speaker": r["speaker"], "seconds": r["seconds"],
                      "mel_len": int(mel_len), "mel_frames": int(mel.shape[2])})
        print(f"  [{i + 1}/{len(pick)}] {r['id']} {r['seconds']:.2f}s mel_len={int(mel_len)}", flush=True)
    (args.out / "index.jsonl").write_text("\n".join(json.dumps(r) for r in index) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
