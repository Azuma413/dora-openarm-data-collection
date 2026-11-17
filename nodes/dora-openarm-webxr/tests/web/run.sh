#!/usr/bin/env bash
# openarm-quest-teleop addition (Apache-2.0, like the rest of this directory).
#
# Browser test of the patched WebXR page: installs Playwright and IWER here on
# first use, downloads Playwright's Chromium, then runs webxr_page.test.mjs.
# Needs Node.js >= 20, openssl, and the repository venv (uv sync) for dora and
# the webxr node. Uses port 18443 (WEB_PORT to change) and runs `dora run`.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"
if [ ! -d node_modules/playwright ] || [ ! -d node_modules/iwer ]; then
	npm ci --no-audit --no-fund
fi
npx playwright install chromium
exec node webxr_page.test.mjs "$@"
