#!/bin/bash
# Build TitaNet Studio (Release) and optionally install it on a phone. The models are not bundled: the app downloads
# each runtime's from Hugging Face on first use.
#   apps/TitaNetStudio/build.sh                  build only
#   apps/TitaNetStudio/build.sh install <UDID>   build, install, launch
set -euo pipefail
# Signing comes from the environment (XcodeGen expands it into the project):
: "${BUNDLE_ID_PREFIX:?export BUNDLE_ID_PREFIX, e.g. com.yourname}"
: "${DEVELOPMENT_TEAM:?export DEVELOPMENT_TEAM=<your Apple developer team ID>}"
cd "$(dirname "$0")/../.."
export PATH=/opt/homebrew/bin:$PATH          # xcodegen is not on a non-interactive PATH
APP=_work/studio_dd/Build/Products/Release-iphoneos/TitaNetStudio.app

(cd apps/TitaNetStudio && xcodegen generate --quiet)
xcodebuild -project apps/TitaNetStudio/TitaNetStudio.xcodeproj -scheme TitaNetStudio -configuration Release \
    -destination 'generic/platform=iOS' -derivedDataPath _work/studio_dd -allowProvisioningUpdates build -quiet
for f in mel_fb.f32 mel_window.f32; do
  test -e "$APP/$f" || { echo "$f did not land in the app bundle"; exit 1; }
done
if find "$APP" \( -name "*.aimodel" -o -name "*.mlmodelc" -o -name "*.mlpackage" \) | grep -q .; then
  echo "a model is in the app bundle; models are downloaded, not shipped"; exit 1
fi
echo "built $APP ($(du -sh "$APP" | cut -f1))"

if [ "${1:-}" = install ]; then
  U="${2:?UDID}"
  xcrun devicectl device install app --device "$U" "$APP"
  xcrun devicectl device process launch --device "$U" --terminate-existing "$BUNDLE_ID_PREFIX.titanetstudio"
fi
