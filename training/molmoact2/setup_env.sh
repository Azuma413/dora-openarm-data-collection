#!/usr/bin/env bash
# Build the MolmoAct2 training environment (mirrors lib/molmoact2/experiments/Dockerfile,
# minus vLLM/flash-attn extras that the LeRobot fine-tuning path does not need).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$HERE/../../lib/molmoact2/experiments"
CUDA="${CUDA_VERSION_PATH:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
TORCH_INDEX="https://download.pytorch.org/whl/$CUDA"
cd "$HERE"
uv venv -p 3.12 .venv
export VIRTUAL_ENV="$HERE/.venv"
uv pip install --index-url "$TORCH_INDEX" "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" torchaudio
uv pip install "torchcodec==0.10.0"
uv pip install -e "$EXP[train]"
# The vendored LeRobot 0.5.0 depends on pynput -> evdev, which needs kernel headers to
# build and is only used for keyboard teleop. Install it without dependencies and add
# the rest of its runtime requirements explicitly.
uv pip install --no-deps -e "$EXP/lerobot"
uv pip install --index-strategy unsafe-best-match --extra-index-url "$TORCH_INDEX" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchcodec==0.10.0" \
  "datasets>=4.0.0,<5.0.0" "diffusers>=0.27.2,<0.36.0" "huggingface-hub>=1.0.0,<2.0.0" \
  "accelerate>=1.10.0,<2.0.0" "numpy>=2.0.0,<2.3.0" "setuptools>=71.0.0,<81.0.0" \
  "packaging>=24.2,<26.0" "einops>=0.8.0,<0.9.0" "opencv-python-headless>=4.9.0,<4.13.0" \
  "av>=15.0.0,<16.0.0" "jsonlines>=4.0.0,<5.0.0" "pyserial>=3.5,<4.0" "draccus==0.10.0" \
  "gymnasium>=1.1.1,<2.0.0" "deepdiff>=7.0.1,<9.0.0" "imageio[ffmpeg]>=2.34.0,<3.0.0" \
  "termcolor>=2.4.0,<4.0.0" pandas pyarrow scipy
