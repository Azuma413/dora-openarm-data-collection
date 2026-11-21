#!/usr/bin/env python
"""MolmoAct2-Think fine-tuning entry point for the OpenArm compress-patch dataset.

Registers the ``openarm_bbox`` LeRobot mixture, applies ``molmoact2_patches`` (bbox
overlay, depth supervision fix, bbox-action binding loss) and then runs the release
trainer's ``launch_scripts/train_lerobot.py`` main with the remaining arguments.
Launch it through torchrun (see ``run_train.sh``)::

    torchrun --standalone --nproc-per-node=1 train.py allenai/MolmoAct2-Think openarm_bbox [flags...]

Environment:
  OPENARM_REPO_ID   dataset directory name under $LEROBOT_DATA_ROOT
                    (default webxr-pedestal-zed-lerobot-curated-molmoact2)
  OPENARM_CAMERAS   comma-separated camera keys (default head_left,wrist_right)
  plus the MOLMOACT2_* variables documented in molmoact2_patches.py.
"""

from __future__ import annotations

import os
import sys

import molmoact2_patches

from launch_scripts import data_mixtures  # noqa: E402  (path set up by molmoact2_patches)

DEFAULT_REPO_ID = "webxr-pedestal-zed-lerobot-curated-molmoact2"
DEFAULT_CAMERAS = "observation.images.head_left,observation.images.wrist_right"


def build_openarm_bbox():
    return data_mixtures.build_single_lerobot_mixture(
        name="openarm_bbox",
        tag="openarm_bimanual_bbox",
        repo_ids=[os.environ.get("OPENARM_REPO_ID", DEFAULT_REPO_ID)],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=os.environ.get("OPENARM_CAMERAS", DEFAULT_CAMERAS).split(","),
        normalize_gripper=False,
        setup_type="bimanual openarm robotic arms",
        # The recorded action is the absolute joint target of both 7-DoF arms plus grippers,
        # the same convention as the YAM/SO100/DROID pretraining data.
        control_mode="absolute joint pose",
        action_horizon=30,
        n_action_steps=30,
    )


def main() -> None:
    molmoact2_patches.apply()
    data_mixtures.MOLMOACT2_LEROBOT_MIXTURES["openarm_bbox"] = build_openarm_bbox
    from launch_scripts import train_lerobot

    sys.argv = [train_lerobot.__file__, *sys.argv[1:]]
    train_lerobot.main()


if __name__ == "__main__":
    main()
