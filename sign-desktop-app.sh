#!/bin/bash
# Ad-hoc code-sign "Lysa Desktop.app".
#
# Why: macOS keys Full Disk Access (and other TCC permissions) to an app's
# code signature. An UNSIGNED app has no stable identity, so an FDA grant you
# toggle on doesn't actually take effect on next launch. Ad-hoc signing gives
# the bundle a stable CDHash, which makes the grant stick.
#
# This only signs the .app bundle (launcher script + Info.plist + icon). Your
# actual app code (desktop.py, lysa/, static/) lives OUTSIDE the bundle, so
# editing code does NOT invalidate this signature — you rarely need to re-run
# this (only if you change the launcher script or Info.plist).
#
# Run after cloning, or after editing the launcher:
#     ./sign-desktop-app.sh
# Then (re)grant Full Disk Access to "Lysa Desktop" in System Settings.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
APP="Lysa Desktop.app"

codesign --force --deep -s - "$APP"
echo "Signed:"
codesign -dvv "$APP" 2>&1 | grep -E "Identifier|Signature|CDHash" | sed 's/^/  /'

# Refresh Launch Services so Finder picks up the new identity.
LSREG="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[ -x "$LSREG" ] && "$LSREG" -f "$APP" && echo "  Re-registered with Launch Services"

cat <<'NOTE'

Next steps (one time):
  1. System Settings -> Privacy & Security -> Full Disk Access
  2. If "Lysa Desktop" is already listed, select it and click - to REMOVE the
     stale (unsigned) entry first.
  3. Click +, add "Lysa Desktop.app", and turn its switch ON.
  4. Double-click Lysa Desktop.
NOTE
