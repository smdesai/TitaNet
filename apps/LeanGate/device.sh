#!/bin/bash
# Drive LeanGate on a phone. Run from anywhere; paths are relative to the titanet-lean root.
#   apps/LeanGate/device.sh install <UDID>            build (Release) and install the app
#   apps/LeanGate/device.sh push <UDID> [stage]       copy the stage (default _work/device_stage/LeanAssets) to
#                                                        Library/Application Support/LeanAssets, then check sizes
#   apps/LeanGate/device.sh run <UDID> <gate.json>    push gate.json, launch, wait for "finished" / "failed",
#                                                        pull Documents/titanet_gate into _work/device/<UDID>/
# Same subcommands and status.json polling as the earlier TitaNet Core AI port's TitaNetGate.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=/opt/homebrew/bin:$PATH          # xcodegen is not on a non-interactive PATH
CMD="${1:?install | push | run}"
U="${2:?UDID}"
# Signing comes from the environment (XcodeGen expands it into the project):
: "${BUNDLE_ID_PREFIX:?export BUNDLE_ID_PREFIX, e.g. com.yourname}"
BID="${LEANGATE_BID:-$BUNDLE_ID_PREFIX.leangate}"
APP=_work/leangate_dd/Build/Products/Release-iphoneos/LeanGate.app
DEST="Library/Application Support/LeanAssets"
OUT="_work/device/$U"
mkdir -p "$OUT"

case "$CMD" in
install)
  : "${DEVELOPMENT_TEAM:?export DEVELOPMENT_TEAM=<your Apple developer team ID>}"
  (cd apps/LeanGate && xcodegen generate --quiet)
  xcodebuild -project apps/LeanGate/LeanGate.xcodeproj -scheme LeanGate -configuration Release \
      -destination 'generic/platform=iOS' -derivedDataPath _work/leangate_dd \
      -clonedSourcePackagesDirPath _work/leangate_spm -allowProvisioningUpdates build -quiet
  xcrun devicectl device install app --device "$U" "$APP"
  ;;
push)
  S="${3:-_work/device_stage/LeanAssets}"
  xcrun devicectl device copy to --device "$U" --domain-type appDataContainer --domain-identifier "$BID" \
      --source "$S" --destination "$DEST"
  # `copy to` can exit 0 with a file cut short: compare sizes
  xcrun devicectl device info files --device "$U" --domain-type appDataContainer --domain-identifier "$BID" \
      --subdirectory "$DEST" --json-output "$OUT/files.json" > /dev/null
  python3 - "$S" "$OUT/files.json" <<'EOF'
import json, os, sys
stage, listing = sys.argv[1], sys.argv[2]
d = json.load(open(listing))
dev = {f["relativePath"]: f["metadata"]["size"] for f in d["result"]["files"] if not f["resources"]["isDirectory"]}
local = {os.path.relpath(os.path.join(r, f), stage): os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(stage) for f in fs}
bad = [k for k in local if dev.get(k) != local[k]]
print(f"{len(local)} files, {sum(local.values()) / 1e6:.1f} MB on the phone" + (f"; MISMATCHED {bad[:5]}" if bad else ", sizes match"))
sys.exit(1 if bad else 0)
EOF
  ;;
run)
  G="${3:?gate.json}"
  T=$(mktemp -d)
  trap 'rm -rf "$T"' EXIT
  cp "$G" "$T/gate.json"
  xcrun devicectl device copy to --device "$U" --domain-type appDataContainer --domain-identifier "$BID" \
      --source "$T/gate.json" --destination "$DEST/gate.json" > /dev/null
  xcrun devicectl device process launch --device "$U" --terminate-existing "$BID" > /dev/null
  echo "launched with $(cat "$G")"
  phase=""; last=$(date +%s); STALL="${STALL_SECONDS:-900}"
  rm -f "$OUT/status.json"
  while true; do
    sleep 20
    now=""
    if xcrun devicectl device copy from --device "$U" --domain-type appDataContainer --domain-identifier "$BID" \
        --source Documents/titanet_gate/status.json --destination "$OUT/status.json" > /dev/null 2>&1; then
      # a partial copy reads as no phase, not as a failure of this script
      now=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("phase", ""))' "$OUT/status.json" 2>/dev/null || true)
    fi
    if [ -n "$now" ] && [ "$now" != "$phase" ]; then echo "$(date +%H:%M:%S) $now"; phase="$now"; last=$(date +%s); fi
    case "$now" in finished|failed) break ;; esac
    # no new phase for STALL seconds (or no status at all): is the app still alive?
    if [ $(( $(date +%s) - last )) -gt "$STALL" ]; then
      if ! xcrun devicectl device info processes --device "$U" 2>/dev/null | grep -q LeanGate; then
        echo "$(date +%H:%M:%S) STALLED: the app is not running (last phase '${phase:-none}')"; break
      fi
      last=$(date +%s)
    fi
  done
  xcrun devicectl device copy from --device "$U" --domain-type appDataContainer --domain-identifier "$BID" \
      --source Documents/titanet_gate --destination "$OUT" > /dev/null
  cat "$OUT/status.json"; echo
  ;;
esac
