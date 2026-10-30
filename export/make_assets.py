"""Write the host's front-end constants from the checkpoint into an assets directory (default _work/assets).

  _work/venvs/coreai/bin/python export/make_assets.py [--out DIR]

  config.json      sample_rate, hop, n_fft, win, n_mels, emb, threshold (NeMo's verify_speakers default)
  mel_fb.f32       preprocessor.featurizer.fb [80, 257] float32 little-endian, bit for bit
  mel_window.f32   preprocessor.featurizer.window [400] (Hann, periodic=False), bit for bit
Every host reads these: the Swift MelFrontend (apps/TitaNetStudio/Resources, hosts/swift) and the Python hosts
(frontend/ in each Hugging Face repository).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import model as M  # noqa: E402
from tnl import paths as P  # noqa: E402

N_FFT, WIN = 512, 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=P.ASSETS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sd = M.load_state_dict(P.SNAPSHOT)
    fb = sd["preprocessor.featurizer.fb"][0].numpy().astype("<f4")
    win = sd["preprocessor.featurizer.window"].numpy().astype("<f4")
    assert fb.shape == (M.N_MELS, N_FFT // 2 + 1) and win.shape == (WIN,), (fb.shape, win.shape)
    fb.tofile(args.out / "mel_fb.f32")
    win.tofile(args.out / "mel_window.f32")
    cfg = {"sample_rate": 16000, "hop": M.HOP, "n_fft": N_FFT, "win": WIN, "n_mels": M.N_MELS, "emb": M.EMB,
           "threshold": 0.7}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    print(json.dumps(cfg), "->", args.out)


if __name__ == "__main__":
    main()
