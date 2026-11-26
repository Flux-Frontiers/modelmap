#!/usr/bin/env bash
# Build modelmap-<version>.dmg: Developer ID signed, notarized and stapled.
#
# The Release build from `make build` is signed ad hoc. This script copies
# it and signs it again from the inside out with the Developer ID
# Application identity in the keychain: every Mach-O file of the bundled
# Python, then the app, with the hardened runtime and a secure timestamp.
# The app is notarized and stapled, then the disk image holding it is
# signed, notarized and stapled in a pass of its own: Gatekeeper rejects an
# image whose app alone was stapled (knowledge_press, app/RUNBOOK.md).
#
# Usage: scripts/make_dmg.sh   (from macos/; `make dmg` runs it after `make build`)
#
# NOTARY_PROFILE  notarytool keychain profile (default modelmap-notary, an
#                 App Store Connect API key stored with `xcrun notarytool
#                 store-credentials`)
# NOTARIZE=0      sign and package only, for a local check

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' ../pyproject.toml)"
PROFILE="${NOTARY_PROFILE:-modelmap-notary}"
NOTARIZE="${NOTARIZE:-1}"
BUILT="build/Build/Products/Release/modelmap.app"
OUT="build/dmg"
APP="$OUT/modelmap.app"
DMG="$OUT/modelmap-$VERSION.dmg"

IDENTITY="$(security find-identity -v -p codesigning \
  | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)"
if [ -z "$IDENTITY" ]; then
  echo "error: no Developer ID Application identity in the keychain" >&2
  exit 1
fi
[ -d "$BUILT" ] || { echo "error: $BUILT is missing; run make build" >&2; exit 1; }

sign() {
  # The secure timestamp is a call to Apple's server, which fails now and
  # then with "Input/output error"; try three times.
  local try
  for try in 1 2 3; do
    codesign --force --timestamp --options runtime --sign "$IDENTITY" "$@" 2>"$OUT/sign.err" \
      && return 0
    echo "codesign attempt $try failed: $(tail -1 "$OUT/sign.err")" >&2
    sleep 2
  done
  return 1
}

notarize() {
  # notarytool can exit 0 for a submission it rejects: read the status.
  local file="$1" result id
  result="$(xcrun notarytool submit "$file" --keychain-profile "$PROFILE" --wait)"
  echo "$result" | rg -q 'status: Accepted' && return 0
  id="$(echo "$result" | sed -n 's/^ *id: //p' | head -1)"
  echo "$result" >&2
  [ -n "$id" ] && xcrun notarytool log "$id" --keychain-profile "$PROFILE" >&2
  echo "error: notarization of $file was not accepted" >&2
  exit 1
}

rm -rf "$OUT"
mkdir -p "$OUT"
ditto "$BUILT" "$APP"

# Inside out: every Mach-O of the bundled Python first, then the app.
count=0
while IFS= read -r -d '' file; do
  if file -b "$file" | rg -q '^Mach-O'; then
    sign "$file"
    count=$((count + 1))
  fi
done < <(find "$APP/Contents/Resources/Python" -type f -print0)
sign --entitlements ModelMap/ModelMap.entitlements "$APP"
echo "Signed $count Python binaries and the app as: $IDENTITY"

# The signed interpreter must still run modelmap under the hardened runtime.
# -B: writing bytecode into the bundle would break the seal just made.
"$APP/Contents/Resources/Python/bin/python3" -I -B -c 'import modelmap.serve' \
  || { echo "error: the signed Python cannot import modelmap" >&2; exit 1; }
# Verify after the check, so anything it changed is caught here, not by Apple.
codesign --verify --strict --deep "$APP"

if [ "$NOTARIZE" = 1 ]; then
  ditto -c -k --keepParent "$APP" "$OUT/modelmap.zip"
  notarize "$OUT/modelmap.zip"
  xcrun stapler staple -q "$APP"
  rm "$OUT/modelmap.zip"
fi

STAGE="$OUT/stage"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/modelmap.app"
ln -s /Applications "$STAGE/Applications"
hdiutil create -quiet -volname "modelmap $VERSION" -srcfolder "$STAGE" -format UDZO -ov "$DMG"
rm -rf "$STAGE"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"

if [ "$NOTARIZE" = 1 ]; then
  notarize "$DMG"
  xcrun stapler staple -q "$DMG"
  spctl --assess --type open --context context:primary-signature --verbose "$DMG"
  spctl --assess --type execute --verbose "$APP"
fi
echo "Built $DMG ($(du -h "$DMG" | cut -f1))"
