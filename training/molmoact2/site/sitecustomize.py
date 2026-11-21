"""Apply the OpenArm MolmoAct2 patches in every interpreter of a training run.

The release trainer starts DataLoader workers with the ``spawn`` method, so class
patches made in the launcher process do not reach the workers (they re-import the
modules and unpickle datasets/collators against the unpatched classes). ``run_train.sh``
puts this directory on PYTHONPATH and sets OPENARM_MOLMOACT2_PATCH=1, which makes every
Python process of the run (trainer ranks and their data workers) apply the patches at
startup.
"""

import os

if os.environ.get("OPENARM_MOLMOACT2_PATCH") == "1":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import molmoact2_patches

    molmoact2_patches.apply()
