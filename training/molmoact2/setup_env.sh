#!/usr/bin/env bash
# Build the MolmoAct2 training environment (mirrors lib/molmoact2/experiments/Dockerfile,
# minus vLLM/flash-attn extras that the LeRobot fine-tuning path does not need).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$HERE/../../lib/molmoact2/experiments"
CUDA="${CUDA_VERSION_PATH:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
cd "$HERE"
uv venv -p 3.12 .venv
export VIRTUAL_ENV="$HERE/.venv"
uv pip install --index-url "https://download.pytorch.org/whl/$CUDA" \
  "torch==$TORCH_VERSION" torchvision torchaudio
uv pip install "torchcodec==0.10.0"
uv pip install -e "$EXP[train]"
uv pip install -e "$EXP/lerobot" --index-strategy unsafe-best-match \
  --extra-index-url "https://download.pytorch.org/whl/$CUDA"
# LeRobot may re-resolve torch; pin it back.
uv pip install --reinstall --index-url "https://download.pytorch.org/whl/$CUDA" \
  "torch==$TORCH_VERSION" torchvision torchaudio
uv pip install --reinstall --no-deps "torchcodec==0.10.0"
uv pip install opencv-python-headless pandas pyarrow
