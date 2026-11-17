# Vendored: enactic/dora-openarm-webxr

- Upstream: https://github.com/enactic/dora-openarm-webxr
- Commit: `52cb5bb3342377b3ab7e5f0d8d791738638dba01` (2026-09-15, version 0.6.0)
- License: Apache-2.0 (`LICENSE` in this directory)

Copied with `git archive` and trimmed (`.github/` and `.pre-commit-config.yaml` removed).
It is vendored rather than installed from git because the node serves the WebXR page
from its own package directory, so page fixes need a local copy.

Local modifications are listed in `PATCHES.md`. The combined diff against upstream
can be regenerated with:

```bash
git -C /path/to/dora-openarm-webxr diff 52cb5bb -- . > /dev/null  # upstream checkout
diff -ru <upstream-checkout> third_party/dora-openarm-webxr
```
