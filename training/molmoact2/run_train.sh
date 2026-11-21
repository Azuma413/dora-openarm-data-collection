#!/usr/bin/env bash
# MolmoAct2-Think depth fine-tuning on the OpenArm compress-patch dataset, with the
# target bbox drawn on the front image and the bbox-action binding loss.
#
#   PROFILE=smoke ./run_train.sh     # a few steps on one 24 GB GPU (LoRA, batch 1)
#   PROFILE=h200  ./run_train.sh     # the real run (NPROC GPUs, H200-class)
#   MODE=fft PROFILE=h200 ./run_train.sh   # full fine-tuning instead of LoRA
#
# Any extra arguments are passed to the trainer, e.g. --max_duration=5000.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXP="$HERE/../../lib/molmoact2/experiments"
PROFILE="${PROFILE:-h200}"
MODE="${MODE:-lora}"
NPROC="${NPROC:-1}"
export OPENARM_REPO_ID="${OPENARM_REPO_ID:-webxr-pedestal-zed-lerobot-curated-molmoact2}"
export LEROBOT_DATA_ROOT="${LEROBOT_DATA_ROOT:-$HOME/openarm-datasets/molmoact2}"
export LEROBOT_DEPTH_DATA_ROOT="${LEROBOT_DEPTH_DATA_ROOT:-$HOME/openarm-datasets/molmoact2-depth}"
# olmo.data.get_dataset reads MOLMO_DATA_DIR unconditionally; the LeRobot path does not use it.
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-$HOME/.cache/molmo_data}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export OPENARM_MOLMOACT2_PATCH=1
export MOLMOACT2_BINDING_WEIGHT="${MOLMOACT2_BINDING_WEIGHT:-0.5}"
export PYTHONPATH="$HERE/site:$HERE:$EXP:$EXP/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
EXP_NAME="${EXP_NAME:-molmoact2-think-openarm-bbox-$MODE-$PROFILE}"
SAVE="${SAVE_FOLDER:-$HERE/checkpoints/$EXP_NAME}"

case "$PROFILE" in
  smoke)
    BATCH=(--device_batch_size=1 --global_batch_size=1 --max_duration="${MAX_STEPS:-10}" --log_interval=1
           --num_workers=2 --save_interval=100000) ;;
  h200)
    BATCH=(--device_batch_size="${DEVICE_BATCH:-8}" --global_batch_size="${GLOBAL_BATCH:-64}"
           --max_duration="${MAX_STEPS:-5000}" --log_interval=20 --num_workers=16 --pin_memory=true
           --save_interval="${SAVE_INTERVAL:-2500}" --save_num_checkpoints_to_keep=4 --data.timeout=900) ;;
  *) echo "unknown PROFILE=$PROFILE" >&2; exit 1 ;;
esac

case "$MODE" in
  lora)
    TUNE=(--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=true --lora_rank=64
          --lora_dropout=0.0 --llm_learning_rate=5e-5 --vit_learning_rate=5e-5 --connector_learning_rate=5e-5
          --action_expert_learning_rate=5e-5) ;;
  fft)
    TUNE=(--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=false
          --llm_learning_rate=1e-5 --vit_learning_rate=5e-6 --connector_learning_rate=5e-6
          --action_expert_learning_rate=5e-5) ;;
  *) echo "unknown MODE=$MODE" >&2; exit 1 ;;
esac

WANDB=(--wandb=null)
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB=(--wandb.name="$EXP_NAME" --wandb.project="$WANDB_PROJECT" --wandb.entity="${WANDB_ENTITY:-}")
fi

cd "$HERE"
exec "$HERE/.venv/bin/torchrun" --standalone --nproc-per-node="$NPROC" "$HERE/train.py" \
  allenai/MolmoAct2-Think openarm_bbox \
  "${WANDB[@]}" "${BATCH[@]}" "${TUNE[@]}" \
  --save_folder="$SAVE" \
  --activation_checkpointing=false \
  --packing=false --dynamic_seq_len=true --crop_mode=resize --frame_loading_backend=av \
  --num_flow_timesteps=8 --mask_action_dim_padding=true --random_camera_order=none \
  --use_annotated_task=false --sample_annotated_task=false \
  --enable_depth_reasoning=true --num_depth_tokens=128 --num_depth_tokens_per_image=100 \
  --depth_code_input_noise_rate=0.1 \
  --style_robot_action=1.0 --style_robot_depth=0.0 --style_robot_depth_action=1.0 \
  "$@"
