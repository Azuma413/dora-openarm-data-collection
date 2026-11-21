"""Helpers shared by the MolmoAct2 data-preparation scripts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent.parent / "lib" / "molmoact2" / "experiments"


def load_upstream_depth_annotation():
    """Import ``experiments/scripts/generate_depth_annotation.py`` (not a package) as a module."""
    name = "molmoact2_generate_depth_annotation"
    if name in sys.modules:
        return sys.modules[name]
    path = EXPERIMENTS / "scripts" / "generate_depth_annotation.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
