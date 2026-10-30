#!/bin/bash
# Stage what the on-device gates read, from the exports and the long fixtures:
#   _work/device_stage/LeanAssets/   audio/ (NeMo's mel + embedding per utterance) and bundles/w_scan_f512 (Core AI)
#   _work/device_stage/MLAssets/     audio/ and bundles/c_scan_f512_{fp32,fp16c} (Core ML, compiled on this Mac)
# Then push with apps/LeanGate/device.sh push <UDID> and apps/MLGate/device.sh push <UDID> (CONVERSION.md, step 8).
set -euo pipefail
cd "$(dirname "$0")/../.."
W="${TITANET_WORK:-_work}"
PY="$W/venvs/coreai/bin/python"
S="$W/device_stage"

for stage in LeanAssets MLAssets; do
  "$PY" oracle/stage_device_fixtures.py --fixtures "$W/fixtures_long" --out "$S/$stage"
done

# Core AI: the bundle as exported
if [ ! -d "$S/LeanAssets/bundles/w_scan_f512" ]; then
  mkdir -p "$S/LeanAssets/bundles"
  cp -R "$W/bundles/w_scan_f512" "$S/LeanAssets/bundles/"
fi

# Core ML: each variant's two packages compiled to .mlmodelc, and its export.json pointing at them
for v in fp32 fp16c; do
  D="$S/MLAssets/bundles/c_scan_f512_$v"
  [ -d "$D" ] && continue
  mkdir -p "$D"
  for m in TitaNetScan TitaNetHead; do
    xcrun coremlcompiler compile "$W/coreml/c_scan_f512_$v/$m.mlpackage" "$D" > /dev/null
  done
  "$PY" - "$W/coreml/c_scan_f512_$v/export.json" "$D/export.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
for m in r["models"].values():
    m["file"] = m["file"].replace(".mlpackage", ".mlmodelc")
json.dump(r, open(sys.argv[2], "w"), indent=1)
EOF
done
du -sh "$S"/*
