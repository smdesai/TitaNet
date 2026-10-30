"""Where everything regenerable lives: under _work/ in the repository, or $TITANET_WORK.

  venvs/        the three Python environments (setup/setup_envs.sh)
  snapshot/     nvidia/speakerverification_en_titanet_large at a pinned revision (setup/fetch_snapshot.py), with the
                .nemo unpacked into snapshot/nemo/ and PROVENANCE.json
  eval/         the evaluation audio (setup/fetch_eval_audio.py)
  fixtures/     NeMo's tensors for 120 short utterances (oracle/make_fixtures.py), read by verify/g1
  fixtures_long/  NeMo's mel and embedding for 40 utterances of 4-65 s (oracle/make_long_fixtures.py), read by
                verify/g2, g5, g6
  assets/       the host's front-end constants (export/make_assets.py)
  bundles/      Core AI exports (export/export_coreai.py)
  coreml/       Core ML exports (export/export_coreml.py)
  hf/           the Hugging Face repositories as staged (export/package_hf.py)
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("TITANET_WORK", ROOT / "_work")).resolve()
SNAPSHOT = WORK / "snapshot"
NEMO_FILE = SNAPSHOT / "speakerverification_en_titanet_large.nemo"
EVAL = WORK / "eval"
FIXTURES = WORK / "fixtures"
FIXTURES_LONG = WORK / "fixtures_long"
ASSETS = WORK / "assets"
