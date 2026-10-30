#!/bin/bash
# Build the three Python environments under _work/venvs (run from anywhere: setup/setup_envs.sh). Needs uv and git.
#   oracle   NeMo Speech (GitHub NVIDIA-NeMo/Speech, pinned commit, version 3.1.0) with its ASR extra (the speaker
#            models live in nemo.collections.asr), torch CPU. The reference: makes the fixtures; nothing else uses it.
#   coreai   Core AI export, the Core AI gates, asset and fixture tools. No NeMo: tnl/model.py reads the checkpoint.
#   coreml   Core ML export and gates: coremltools 9.0, which is tested against torch up to 2.7.0.
# Each venv installs against setup/<venv>-constraints.txt (the frozen versions every gate in this repository ran
# with), so transitive dependencies are pinned as well as the direct ones. NeMo is cloned at a pinned commit into
# _work/ref. TITANET_WORK=<dir> builds somewhere other than _work.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # the repository root, resolved before the cd below
W="${TITANET_WORK:-$HERE/_work}"
R="$HERE/setup"
mkdir -p "$W/venvs" "$W/ref"
cd "$W"

NEMO_COMMIT=43dd97ddf50dc3b3104cc22471f6e30407d1b18c   # NVIDIA-NeMo/Speech main, version 3.1.0, 2026-10-27

clone() {   # clone <url> <dir> <commit>: cloned beside and moved into place, so an interrupted clone is not reused
  if [ ! -d "$2" ]; then
    rm -rf "$2.partial"
    git clone -q "$1" "$2.partial"
    mv "$2.partial" "$2"
  fi
  git -C "$2" checkout -q "$3"
}

clone https://github.com/NVIDIA-NeMo/Speech.git ref/nemo "$NEMO_COMMIT"

[ -x venvs/oracle/bin/python ] || uv venv -q -p 3.12 venvs/oracle   # re-runs keep an existing venv
uv pip install -q --python venvs/oracle/bin/python "torch==2.9.0" "torchaudio==2.9.0" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install -q --python venvs/oracle/bin/python -c "$R/oracle-constraints.txt" "$W/ref/nemo[asr]" "soundfile" "jiwer"
echo "oracle ready"

[ -x venvs/coreai/bin/python ] || uv venv -q -p 3.12 venvs/coreai   # re-runs keep an existing venv
uv pip install -q --python venvs/coreai/bin/python "torch==2.9.0" --index-url https://download.pytorch.org/whl/cpu
PINS=("torch==2.9.0" "coreai-torch==0.4.3" "coreai-core==1.0.0b3" "numpy==2.4.6" "huggingface-hub==2.1.1"
      "safetensors==0.8.0" "pytest==9.1.1" "pyyaml==6.0.3" "sentencepiece==0.2.2" "soundfile==0.14.0" "jiwer==4.0.0"
      "pyarrow==25.0.1" "whisper_normalizer==0.1.15")
uv pip install -q --python venvs/coreai/bin/python -c "$R/coreai-constraints.txt" "${PINS[@]}"
echo "coreai ready"

[ -x venvs/coreml/bin/python ] || uv venv -q -p 3.12 venvs/coreml   # re-runs keep an existing venv
uv pip install -q --python venvs/coreml/bin/python -c "$R/coreml-constraints.txt" \
  "torch==2.7.0" "coremltools==9.0" "numpy" "soundfile" "pyyaml"
echo "coreml ready"

venvs/oracle/bin/python -c "import importlib.metadata as m; print('oracle', {p: m.version(p) for p in ['torch','nemo-toolkit','lightning']})"
venvs/coreai/bin/python -c "import importlib.metadata as m; print('coreai', {p: m.version(p) for p in ['torch','coreai-torch','coreai-core','numpy']})"
venvs/coreml/bin/python -c "import importlib.metadata as m; print('coreml', {p: m.version(p) for p in ['torch','coremltools','numpy']})"
