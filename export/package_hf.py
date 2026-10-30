"""Stage the two Hugging Face repositories from the exports, exactly as published.

  _work/venvs/coreai/bin/python export/package_hf.py [--out _work/hf]

  <out>/TitaNet-Large-CoreAI/   titanet.aimodel/, export.json            from _work/bundles/w_scan_f512
                                frontend/{config.json, mel_fb.f32, mel_window.f32}   from _work/assets
                                README.md                                from hf/TitaNet-Large-CoreAI
  <out>/TitaNet-Large-CoreML/   TitaNet{Scan,Head}_{fp32,fp16c}.mlpackage, export_{fp32,fp16c}.json
                                                                         from _work/coreml/c_scan_f512_{fp32,fp16c}
                                frontend/, README.md                     as above, README from hf/TitaNet-Large-CoreML
The Core ML model files carry the variant in their names (Xcode compiles every .mlpackage to the bundle root, so two
`TitaNetScan`s would collide) and each export json is rewritten to match. Upload with `hf upload` (CONVERSION.md).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tnl import paths as P  # noqa: E402

FRONTEND = ("config.json", "mel_fb.f32", "mel_window.f32")


def frontend(dst: Path, assets: Path) -> None:
    (dst / "frontend").mkdir(parents=True)
    for f in FRONTEND:
        shutil.copy2(assets / f, dst / "frontend" / f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=P.WORK / "hf")
    ap.add_argument("--coreai", type=Path, default=P.WORK / "bundles" / "w_scan_f512")
    ap.add_argument("--coreml", type=Path, default=P.WORK / "coreml", help="holds c_scan_f512_{fp32,fp16c}")
    ap.add_argument("--assets", type=Path, default=P.ASSETS)
    args = ap.parse_args()

    ai, ml = args.out / "TitaNet-Large-CoreAI", args.out / "TitaNet-Large-CoreML"
    for d in (ai, ml):
        if d.exists():
            sys.exit(f"{d} exists: staging never overwrites a repository")

    # Core AI: the bundle and its record as exported
    ai.mkdir(parents=True)
    shutil.copytree(args.coreai / "titanet.aimodel", ai / "titanet.aimodel")
    shutil.copy2(args.coreai / "export.json", ai / "export.json")
    frontend(ai, args.assets)
    shutil.copy2(ROOT / "hf" / ai.name / "README.md", ai / "README.md")

    # Core ML: both variants, renamed per variant
    ml.mkdir(parents=True)
    for v in ("fp32", "fp16c"):
        src = args.coreml / f"c_scan_f512_{v}"
        rec = json.loads((src / "export.json").read_text())
        for name, m in rec["models"].items():
            renamed = f"TitaNet{name.capitalize()}_{v}.mlpackage"
            shutil.copytree(src / m["file"], ml / renamed)
            m["file"] = renamed
        for k in ("export_seconds", "bytes"):      # this machine's, not the model's
            rec.pop(k, None)
        (ml / f"export_{v}.json").write_text(json.dumps(rec, indent=1) + "\n")
    frontend(ml, args.assets)
    shutil.copy2(ROOT / "hf" / ml.name / "README.md", ml / "README.md")

    for d in (ai, ml):
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        print(f"{d}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
