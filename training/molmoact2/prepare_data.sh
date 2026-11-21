#!/usr/bin/env bash
# Build everything the training run reads from the curated dataset:
#   1. the training view (head_left + wrist_right, target_bbox column),
#   2. the ZED depth-token companion dataset,
#   3. the bbox-action binding pairs.
# Usage: ./prepare_data.sh [SOURCE_DATASET]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
SOURCE="${1:-$HOME/openarm-datasets/webxr-pedestal-zed-lerobot-curated}"
REPO_ID="${OPENARM_REPO_ID:-$(basename "$SOURCE")-molmoact2}"
DATA_ROOT="${LEROBOT_DATA_ROOT:-$HOME/openarm-datasets/molmoact2}"
DEPTH_ROOT="${LEROBOT_DEPTH_DATA_ROOT:-$HOME/openarm-datasets/molmoact2-depth}"
export PYTHONPATH="$HERE:$HERE/../../lib/molmoact2/experiments:$HERE/../../lib/molmoact2/experiments/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$HERE/prepare_dataset.py" "$SOURCE" "$DATA_ROOT/$REPO_ID" --overwrite
"$PY" "$HERE/generate_zed_depth_annotation.py" "$SOURCE" "$DATA_ROOT/$REPO_ID" "$DEPTH_ROOT/$REPO_ID" \
  --compare-depth-anything "${COMPARE_FRAMES:-12}" --compare-dir "$DEPTH_ROOT/$REPO_ID-compare"
"$PY" "$HERE/generate_zed_depth_annotation.py" "$SOURCE" "$DATA_ROOT/$REPO_ID" "$DEPTH_ROOT/$REPO_ID" --overwrite
"$PY" "$HERE/mine_binding_pairs.py" "$DATA_ROOT/$REPO_ID"
