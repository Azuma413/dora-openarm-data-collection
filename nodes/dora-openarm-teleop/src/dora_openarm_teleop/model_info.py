"""Facts about the OpenArm cell scenes, computed with MuJoCo (docs/teleop.md).

Ported from openarm-quest-teleop's oqt.model_info, keeping only the scenes
openarm_mujoco ships and the stock pinch gripper. Every value below is computed
from the installed MJCF at keyframe "home", so a model update shows up as a
failing unit test instead of a silently wrong clamp.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from openarm_mujoco.v2 import asset_path

from dora_openarm_teleop import frames

# Scene name -> MJCF path relative to openarm_mujoco's v2 assets. All cell
# scenes share the "home" keyframe and the arm_origin site.
SCENES: dict[str, str] = {
    "cell": "cell/cell.xml",
    "demo": "cell/demo.xml",
    "move_puck": "cell/move_puck_cell_scene.xml",
    "peg_socket": "cell/peg_socket_cell_scene.xml",
    "tool": "cell/tool_cell_scene.xml",
    "articulated": "cell/articulated_cell_scene.xml",
    "bottle": "cell/bottle_cell_scene.xml",
    "valve": "cell/valve_cell_scene.xml",
}

SIDES: tuple[str, ...] = ("right", "left")


def scene_xml(name: str) -> Path:
    """Absolute path of a named cell scene inside the installed openarm_mujoco assets."""
    try:
        relative = SCENES[name]
    except KeyError:
        known = ", ".join(sorted(SCENES))
        raise ValueError(f"unknown scene {name!r} (known: {known})") from None
    return Path(asset_path(relative))


def bimanual_xml() -> Path:
    """Absolute path of the arms-only bimanual model."""
    return Path(asset_path("openarm_bimanual.xml"))


# ---------------------------------------------------------------------------
# Facts derived from the MJCF (docs/teleop.md).
#
# Nothing below is a copied constant: every value is computed with MuJoCo from
# the installed openarm_mujoco files at keyframe "home", so a model update
# shows up as a failing unit test instead of a silently wrong clamp. All poses
# are in the arm_origin site frame (x forward, y left, z up), the frame of
# POSE8. Results are cached per scene; public functions return copies.

ORIGIN_SITE = "arm_origin"
HOME_KEYFRAME = "home"
TABLE_GEOM = "cell_table_col"


@functools.cache
def _compiled(path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(path)


def load_model(scene: str = "cell") -> mujoco.MjModel:
    """Compiled model of a named scene (cached; treat as read-only)."""
    return _compiled(str(scene_xml(scene)))


def _require_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"{kind.name} {name!r} not found in the model")
    return index


def _home_data(model: mujoco.MjModel) -> mujoco.MjData:
    """Keyframe "home" with both grippers closed (finger joints 0).

    The facts here are defined at closed fingers: the clutch starts every arm with
    gripper 0 and poses the finger hulls from there. openarm_mujoco 2.3.0's "home"
    opens the grippers (+-0.785 rad) where earlier versions closed them, so the
    keyframe's finger values are overridden rather than trusted.
    """
    data = mujoco.MjData(model)
    key = _require_id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEYFRAME)
    mujoco.mj_resetDataKeyframe(model, data, key)
    for side in SIDES:
        for joint_name in ("finger_joint1", "finger_joint2"):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_{joint_name}"
            )
            if joint >= 0:
                data.qpos[model.jnt_qposadr[joint]] = 0.0
    mujoco.mj_kinematics(model, data)
    return data


def _origin_frame(data: mujoco.MjData, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """(position, rotation matrix) of the arm_origin site in world."""
    return data.site_xpos[origin].copy(), data.site_xmat[origin].reshape(3, 3).copy()


@dataclass(frozen=True)
class FingerHull:
    """One finger's collision hull, posable for any gripper angle.

    The finger is a single hinge on the end-effector body, so its hull in the
    ee_control_point site frame is ``anchor + rotation · Rot(axis, ratio·g) · points``
    for gripper (finger_joint1) angle ``g``. Evaluating that per tick is a few
    hundred vector rotations, far cheaper than a MuJoCo forward pass.
    """

    anchor: np.ndarray  # (3,) joint anchor, site frame
    rotation: np.ndarray  # (3, 3) finger body orientation at q = 0, site frame
    axis: np.ndarray  # (3,) unit joint axis, finger body frame
    points: np.ndarray  # (N, 3) hull vertices relative to the anchor, body frame
    ratio: float  # this finger's joint angle per gripper angle (mimic equality)

    def points_in_site(self, gripper: float) -> np.ndarray:
        # Rodrigues directly: this runs every engaged tick, per finger.
        angle = self.ratio * float(gripper)
        x, y, z = self.axis
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        turn = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
        return self.anchor + self.points @ (self.rotation @ turn).T


def _hull_vertices(model: mujoco.MjModel, mesh: int) -> np.ndarray:
    """Convex-hull vertices of a mesh in its geom frame.

    MuJoCo collides meshes through this hull, so it is exactly the surface
    that can touch the table. Layout of mesh_graph: numvert, numface,
    vert_edgeadr[numvert], vert_globalid[numvert], ...
    """
    start = model.mesh_vertadr[mesh]
    vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    graph = model.mesh_graphadr[mesh]
    if graph < 0:
        return vertices.copy()
    count = model.mesh_graph[graph]
    ids = model.mesh_graph[graph + 2 + count : graph + 2 + 2 * count]
    return vertices[ids].copy()


_BOX_SIGNS = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T


def _geom_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Vertices of one collision geom in its own frame.

    Only the two shapes the two jaws are made of. A mesh is taken as the convex
    hull MuJoCo actually collides; a box as its eight corners, which is the same
    thing for a box. Anything else raises rather than being silently left out of
    the hull -- the hull is a safety clamp, and a missing finger is a clamp that
    lets the real one through the table.
    """
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        return _hull_vertices(model, model.geom_dataid[geom])
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return _BOX_SIGNS * model.geom_size[geom]
    raise ValueError(
        f"finger collision geom {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)!r} "
        f"is a {mujoco.mjtGeom(int(kind)).name}; the fingertip hull handles meshes and boxes"
    )


def _subtree(model: mujoco.MjModel, root: int) -> np.ndarray:
    """``root`` and every body below it (MuJoCo numbers bodies parents first)."""
    inside = np.zeros(model.nbody, dtype=bool)
    inside[root] = True
    for body in range(root + 1, model.nbody):
        inside[body] = inside[model.body_parentid[body]]
    return np.flatnonzero(inside)


def _mimic_ratio(model: mujoco.MjModel, leader: int, follower: int) -> float:
    """Follower angle per leader angle from a linear joint equality (polycoef "0 k")."""
    for eq in range(model.neq):
        if model.eq_type[eq] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        pair = (int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]))
        coef = model.eq_data[eq, :5]
        if pair not in ((leader, follower), (follower, leader)):
            continue
        if coef[0] != 0.0 or np.any(coef[2:] != 0.0) or coef[1] == 0.0:
            raise ValueError("finger mimic equality is not a pure linear ratio")
        # MuJoCo: joint1 = polycoef(joint2), i.e. obj1 = k * obj2.
        return float(1.0 / coef[1]) if pair == (leader, follower) else float(coef[1])
    raise ValueError("no mimic equality couples the two finger joints")


def _finger_hulls(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> tuple[FingerHull, ...]:
    site = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
    site_pos = data.site_xpos[site]
    site_rot = data.site_xmat[site].reshape(3, 3)
    leader = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint1")
    hulls = []
    for index, (finger, joint_name) in enumerate(
        (("inner", "finger_joint1"), ("outer", "finger_joint2"))
    ):
        body = _require_id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_ee_{finger}_finger")
        joint = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_{joint_name}")
        if data.qpos[model.jnt_qposadr[joint]] != 0.0:
            raise ValueError(f"keyframe {HOME_KEYFRAME!r} must have closed fingers")
        body_pos = data.xpos[body]
        body_rot = data.xmat[body].reshape(3, 3)
        # The whole subtree, not just the finger body's own geoms, so a jaw
        # built from several bodies (a fitted soft finger, say) is covered too.
        inside = np.zeros(model.nbody, dtype=bool)
        inside[_subtree(model, body)] = True
        local = []
        for geom in range(model.ngeom):
            collides = model.geom_contype[geom] or model.geom_conaffinity[geom]
            if not inside[model.geom_bodyid[geom]] or not collides:
                continue
            vertices = _geom_points(model, geom)
            world = data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T
            local.append((world - body_pos) @ body_rot)
        if not local:
            raise ValueError(f"no collision geom under {side} {finger} finger")
        points = np.concatenate(local) - model.jnt_pos[joint]
        hulls.append(
            FingerHull(
                anchor=(data.xanchor[joint] - site_pos) @ site_rot,
                rotation=site_rot.T @ body_rot,
                axis=model.jnt_axis[joint] / np.linalg.norm(model.jnt_axis[joint]),
                points=points,
                ratio=1.0 if index == 0 else _mimic_ratio(model, leader, joint),
            )
        )
    return tuple(hulls)


def _line_meeting_point(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray:
    """Midpoint of the common perpendicular of two lines (their intersection if they meet)."""
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w, d2 @ w
    denom = a * c - b * b
    if denom < 1e-12:
        return p2.copy()
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    return 0.5 * ((p1 + t * d1) + (p2 + s * d2))


@dataclass(frozen=True)
class _SceneFacts:
    home: dict[str, np.ndarray]
    shoulder: dict[str, np.ndarray]
    gripper_range: dict[str, tuple[float, float]]
    table_z: float | None
    hulls: dict[str, tuple[FingerHull, ...]]
    arm_qpos_adr: dict[str, tuple[int, ...]]  # joint1..7, finger1, finger2


def _facts(scene: str) -> _SceneFacts:
    return _facts_of(str(scene_xml(scene)))


@functools.cache
def _facts_of(path: str) -> _SceneFacts:
    model = _compiled(path)
    data = _home_data(model)
    origin = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, ORIGIN_SITE)
    origin_pos, origin_rot = _origin_frame(data, origin)

    def in_origin(world: np.ndarray) -> np.ndarray:
        return (world - origin_pos) @ origin_rot

    home, shoulder, gripper_range, hulls, adr = {}, {}, {}, {}, {}
    for side in SIDES:
        site = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
        rotation = origin_rot.T @ data.site_xmat[site].reshape(3, 3)
        home[side] = np.concatenate(
            [in_origin(data.site_xpos[site]), frames.matrix_to_quat(rotation)]
        )

        j1, j2 = (
            _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}")
            for i in (1, 2)
        )
        meet = _line_meeting_point(
            data.xanchor[j1], data.xaxis[j1], data.xanchor[j2], data.xaxis[j2]
        )
        shoulder[side] = in_origin(meet)

        finger = _require_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint1")
        actuators = [
            a
            for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT
            and model.actuator_trnid[a, 0] == finger
        ]
        if len(actuators) != 1:
            raise ValueError(f"expected one actuator on {side} finger_joint1, got {len(actuators)}")
        low, high = model.actuator_ctrlrange[actuators[0]]
        gripper_range[side] = (float(low), float(high))

        hulls[side] = _finger_hulls(model, data, side)
        names = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
        names += [f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"]
        adr[side] = tuple(
            int(model.jnt_qposadr[_require_id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names
        )

    table_z = None
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM)
    if table >= 0 and model.geom_type[table] == mujoco.mjtGeom.mjGEOM_BOX:
        half = model.geom_size[table]
        corners = (
            data.geom_xpos[table] + (_BOX_SIGNS * half) @ data.geom_xmat[table].reshape(3, 3).T
        )
        table_z = float(np.max(in_origin(corners)[:, 2]))

    return _SceneFacts(home, shoulder, gripper_range, table_z, hulls, adr)


def home_ee_poses(scene: str = "cell") -> dict[str, np.ndarray]:
    """End-effector (ee_control_point) pose7 per side at keyframe "home"."""
    return {side: pose.copy() for side, pose in _facts(scene).home.items()}


def table_top_z_arm_origin(scene: str = "cell") -> float:
    """Height of the cell table's top face in the arm_origin frame (lifter at home)."""
    table_z = _facts(scene).table_z
    if table_z is None:
        raise ValueError(f"scene {scene!r} has no box geom {TABLE_GEOM!r}")
    return table_z


def shoulder_positions(scene: str = "cell") -> dict[str, np.ndarray]:
    """Centre of each spherical shoulder in the arm_origin frame.

    Taken as the meeting point of the joint1 and joint2 axes (joint3's axis
    passes through it too). joint1's own anchor lies 6 cm inboard on the same
    axis, which would make a reach sphere centred there too long sideways and
    too short across the body.
    """
    return {side: p.copy() for side, p in _facts(scene).shoulder.items()}


def gripper_ranges(scene: str = "cell") -> dict[str, tuple[float, float]]:
    """ctrlrange of the actuator driving each finger_joint1 (0 is closed)."""
    return dict(_facts(scene).gripper_range)


def finger_hulls(side: str, scene: str = "cell") -> tuple[FingerHull, ...]:
    """Posable collision hulls of the inner and outer finger of one side."""
    return _facts(scene).hulls[side]


def fingertip_points(side: str, gripper: float = 0.0, scene: str = "cell") -> np.ndarray:
    """All finger collision-hull vertices, (N, 3) in the ee_control_point site frame."""
    return np.concatenate([hull.points_in_site(gripper) for hull in finger_hulls(side, scene)])


def fingertip_offset(side: str, scene: str = "cell") -> np.ndarray:
    """The fingertip point used by the single-point table clamp (docs/teleop.md, Clutch).

    Method: pose both fingers' collision hulls at gripper 0 (closed), express
    their vertices in the ee_control_point site frame and pick the vertex with
    the lowest z. The fingers extend along the site's -z, so this is the point
    that touches first when the gripper points straight down.
    """
    points = fingertip_points(side, 0.0, scene)
    return points[np.argmin(points[:, 2])].copy()


def ee_poses_from_qpos(qpos16: np.ndarray, scene: str = "cell") -> dict[str, np.ndarray]:
    """Forward kinematics of a QPOS16 (right 7 joints + gripper, then left).

    Everything else (lifter, objects) stays at keyframe "home". Returns the
    ee_control_point pose7 per side in the arm_origin frame.
    """
    values = np.asarray(qpos16, dtype=np.float64).reshape(-1)
    if values.shape != (16,) or not np.all(np.isfinite(values)):
        raise ValueError("qpos16 must be 16 finite values")
    facts = _facts(scene)
    model = load_model(scene)
    data = _home_data(model)
    for offset, side in ((0, "right"), (8, "left")):
        adr = facts.arm_qpos_adr[side]
        data.qpos[list(adr[:7])] = values[offset : offset + 7]
        ratio = facts.hulls[side][1].ratio
        data.qpos[adr[7]] = values[offset + 7]
        data.qpos[adr[8]] = ratio * values[offset + 7]
    mujoco.mj_kinematics(model, data)
    origin = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, ORIGIN_SITE)
    origin_pos, origin_rot = _origin_frame(data, origin)
    poses = {}
    for side in SIDES:
        site = _require_id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee_control_point")
        rotation = origin_rot.T @ data.site_xmat[site].reshape(3, 3)
        poses[side] = np.concatenate(
            [(data.site_xpos[site] - origin_pos) @ origin_rot, frames.matrix_to_quat(rotation)]
        )
    return poses
