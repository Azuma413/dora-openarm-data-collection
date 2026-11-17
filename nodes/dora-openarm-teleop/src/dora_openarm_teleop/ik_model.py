"""A copy of the IK's MuJoCo model with the arm joint ranges narrowed to the gate's.

dora-openarm-ik integrates its own joint state inside the MJCF joint ranges (openarm_control
ArmJointLimit reads ``model.jnt_range``; without --limit-velocity, ArmConfigurationLimit does
the same through mink). Those ranges are much wider than the gate's (right J1 up to 3.49 rad
against 2.0 in joint_limits.yaml), so the IK happily plans configurations the
gate can only hold (reason "ik_limit", GATE-1). Given the gate's ranges the IK brakes into
the limit instead and its joints stay mutually consistent.

The IK node loads openarm_control's default model, the cell scene (cell/cell.xml), not the
arms-only openarm_bimanual.xml: that file has no "home" keyframe, so the IK would start from
qpos 0 (J4 = 0, straight arms) and its null-space posture task would pull towards it. The
cell scene attaches openarm_bimanual.xml through ``<model file=...>``, and the joint ranges
live in that file, so two files are written:

  <out_path>                          cell.xml, compiler dirs and the model file absolute
  <out_path stem>.openarm_bimanual.xml   openarm_bimanual.xml with narrowed arm ranges

Both only reference absolute paths, so out_path may live anywhere (runs/<name>/ik_model.xml).
Nothing else changes: bodies, sites, keyframes and the finger ranges are copied verbatim.

The joint velocities get the same treatment through ``write_ik_config`` (P3): openarm_control's
built-in --limit-velocity caps (2.0 2.0 3.14 3.14 6.3 6.3 6.3 rad/s) are faster than the
gate's (sim 0.8x, real 0.5x of 2.0 2.0 3.3 3.3 6.3 6.3 6.3), so on a fast reach the IK ran
ahead, the gate clipped every tick and the command trailed the IK (15 mm in the e2e run).
"""

from __future__ import annotations

import io
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from dora_openarm_teleop import model_info
from dora_openarm_teleop.gate import (
    ARM_DOF,
    IK_VELOCITY_FRACTION,
    PROFILES,
    SIDES,
    ik_velocity_caps,
    load_limits,
)

# The only key openarm_control 973c2d7 reads from --config (ik_params._load_velocity_caps):
# exactly 7 finite, positive rad/s values, J1..J7, applied to both arms, and only when the
# IK also runs with --limit-velocity (otherwise the file is ignored without a word).
IK_CONFIG_KEY = "arm_velocity_limits"

# The scene attaches the arms under this <model> asset name (cell/cell.xml).
_ARMS_MODEL_ASSET = "bimanual"
_DIR_ATTRIBUTES = ("assetdir", "meshdir", "texturedir")


def ik_base_xml() -> Path:
    """The model dora-openarm-ik loads without --xml (openarm_control's default)."""
    return model_info.scene_xml("cell")


def arm_joint_name(side: str, index: int) -> str:
    """MJCF name of arm joint ``index`` (0-based, J1..J7)."""
    return f"openarm_{side}_joint{index + 1}"


def narrowed_ranges(
    limits_yaml: str | Path, margin: float, arms_xml: str | Path
) -> dict[str, tuple[float, float]]:
    """Joint name -> (lower, upper): the gate limits intersected with the MJCF, minus margin.

    With the gate's own limit_margin this is exactly the range the gate clips commands
    to, so an IK solution that respects it is never clipped.
    """
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be a finite, non-negative number")
    limits = load_limits(limits_yaml)
    joints = _joints_by_name(ET.parse(arms_xml).getroot())
    ranges = {}
    for side in SIDES:
        for j in range(ARM_DOF):
            name = arm_joint_name(side, j)
            mjcf_lower, mjcf_upper = _joint_range(joints, name)
            lower = max(float(limits[side].lower[j]), mjcf_lower) + margin
            upper = min(float(limits[side].upper[j]), mjcf_upper) - margin
            if not lower < upper:
                raise ValueError(f"{name}: margin {margin} leaves an empty range")
            ranges[name] = (lower, upper)
    return ranges


def write_ik_model(limits_yaml: Path, margin: float, out_path: Path) -> Path:
    """Write the IK model with the gate's joint ranges; returns ``out_path`` (absolute).

    Pass the result to dora-openarm-ik as ``--xml <path>``.
    """
    return _write(ik_model_files(limits_yaml, margin, out_path), out_path)


def ik_model_files(limits_yaml: Path, margin: float, out_path: Path) -> dict[Path, bytes]:
    """{absolute path: content} of the files write_ik_model writes, without writing them.

    The content depends on ``out_path`` (the scene names its arms file by absolute path).
    """
    out_path = Path(out_path).resolve()
    base = ik_base_xml()
    scene = ET.parse(base)
    asset = _arms_model_asset(scene.getroot(), base)
    arms_source = (base.parent / asset.attrib["file"]).resolve()
    ranges = narrowed_ranges(limits_yaml, margin, arms_source)

    arms = ET.parse(arms_source)
    _absolute_dirs(arms.getroot(), arms_source.parent)
    joints = _joints_by_name(arms.getroot())
    for name, (lower, upper) in ranges.items():
        joints[name].set("range", f"{lower!r} {upper!r}")
    arms_out = out_path.with_name(f"{out_path.stem}.openarm_bimanual.xml")

    _absolute_dirs(scene.getroot(), base.parent)
    asset.set("file", str(arms_out))
    return {arms_out: _xml_bytes(arms), out_path: _xml_bytes(scene)}


def write_ik_config(
    limits_yaml: str | Path,
    profile: str,
    velocity_scale: float | None,
    out_path: str | Path,
) -> Path:
    """Write the IK's --config with J1-J7 velocity caps at 0.9 x the gate's; returns out_path.

    ``velocity_scale`` is the gate's (``--velocity-scale``); None takes the profile's
    (sim 0.8, real 0.5). Pass the result to dora-openarm-ik as ``--config <path>``, together
    with ``--limit-velocity``. openarm_control bounds each joint's step per solve to
    cap / --tick-hz, so the IK output moves at most at these speeds when it solves once per
    clutch tick. The gripper is passed through by the IK and stays limited by the gate only.
    """
    return _write(ik_config_files(limits_yaml, profile, velocity_scale, out_path), out_path)


def ik_config_files(
    limits_yaml: str | Path,
    profile: str,
    velocity_scale: float | None,
    out_path: str | Path,
) -> dict[Path, bytes]:
    """{absolute path: content} of the file write_ik_config writes, without writing it."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r} (known: {', '.join(PROFILES)})")
    scale = PROFILES[profile].velocity_scale if velocity_scale is None else float(velocity_scale)
    caps = ik_velocity_caps(load_limits(limits_yaml), scale)
    out_path = Path(out_path).resolve()
    values = ", ".join(repr(float(v)) for v in caps)
    text = (
        "# Written by dora_openarm_teleop.ik_model.write_ik_config from the gate's limits and velocity scale.\n"
        f"# dora-openarm-ik --limit-velocity --config: J1..J7 rad/s for both arms,\n"
        f"# {IK_VELOCITY_FRACTION} x the gate's caps (profile {profile}, velocity_scale {scale!r}"
        f" x joint_velocity_limits of {Path(limits_yaml).resolve()}).\n"
        f"{IK_CONFIG_KEY}: [{values}]\n"
    )
    return {out_path: text.encode("utf-8")}


def _write(files: dict[Path, bytes], out_path: str | Path) -> Path:
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return Path(out_path).resolve()


def _xml_bytes(tree: ET.ElementTree) -> bytes:
    buffer = io.BytesIO()
    tree.write(buffer, encoding="utf-8", xml_declaration=True)
    return buffer.getvalue()


def _joints_by_name(root: ET.Element) -> dict[str, ET.Element]:
    return {j.attrib["name"]: j for j in root.iter("joint") if "name" in j.attrib}


def _joint_range(joints: dict[str, ET.Element], name: str) -> tuple[float, float]:
    try:
        text = joints[name].attrib["range"]
    except KeyError:
        # A range inherited from a default class would need resolving the class chain;
        # refuse instead of silently leaving the IK on the wide range.
        raise ValueError(f"{name}: no explicit range attribute in the arms MJCF") from None
    lower, upper = (float(v) for v in text.split())
    return lower, upper


def _arms_model_asset(root: ET.Element, source: Path) -> ET.Element:
    for element in root.iter("model"):
        if element.attrib.get("name") == _ARMS_MODEL_ASSET and "file" in element.attrib:
            return element
    raise ValueError(f"{source}: no <model name={_ARMS_MODEL_ASSET!r} file=...> asset")


def _absolute_dirs(root: ET.Element, source_dir: Path) -> None:
    """Make the compiler's asset directories absolute, so files resolve from anywhere.

    Without a directory attribute MuJoCo resolves files next to the model file, so
    meshdir and texturedir are pinned to the source directory in that case.
    """
    compilers = list(root.iter("compiler"))
    if not compilers:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
        compilers = [compiler]
    has_dir = any(attr in c.attrib for c in compilers for attr in _DIR_ATTRIBUTES)
    for compiler in compilers:
        for attr in _DIR_ATTRIBUTES:
            if attr in compiler.attrib:
                compiler.set(attr, str((source_dir / compiler.attrib[attr]).resolve()))
    if not has_dir:
        compilers[0].set("meshdir", str(source_dir.resolve()))
        compilers[0].set("texturedir", str(source_dir.resolve()))
