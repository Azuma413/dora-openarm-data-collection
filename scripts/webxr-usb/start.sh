#!/usr/bin/env bash
# Quest(USB) -> adb reverse -> local TURN(TCP) -> dora-openarm-webxr
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
adb wait-for-device
adb reverse tcp:8443 tcp:8443
adb reverse tcp:3479 tcp:3479
adb reverse --list
echo "Open https://localhost:8443/ in the Quest browser."
exec turnserver -c "$here/turnserver.conf" --pidfile "${XDG_RUNTIME_DIR:-/tmp}/webxr-turn.pid"
