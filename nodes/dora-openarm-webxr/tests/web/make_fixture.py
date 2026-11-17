# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).

"""Write the scene asset and the ground truth the 3D page test runs against.

The page drives a glTF tree with MuJoCo joint values (``static/mjkin.js``), so
what has to be tested is that its kinematics agree with MuJoCo's own. This
writes a small MJCF with one of every joint type the exporter emits, exports it
in exactly the layout ``scripts/mjcf_to_glb.py`` writes -- one node per body,
carrying its rest transform and, in ``extras.joints``, the joints that move it
-- and records MuJoCo's own ``xpos``/``xquat`` for a handful of poses.

    uv run python third_party/dora-openarm-webxr/tests/web/make_fixture.py

The outputs (``fixture/arm.glb``, ``fixture/trajectory.json``) are committed,
so the test itself needs neither MuJoCo nor this script. Re-run it only when
the exporter's layout changes; the page reads that layout and nothing else.

The glTF writer here is derived from the openarm-quest-teleop scene exporter
prototype, cut down to the geometry this fixture needs (boxes only).
"""

from __future__ import annotations

import json
import math
import pathlib
import struct

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent
OUT = HERE / "fixture"

# One of every joint type mjkin.js implements, and a body with two joints on
# it, because that is where a wrong multiplication order shows up. The link
# lengths are round numbers so a failure is readable in metres.
MJCF = """
<mujoco model="fixture_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="table" pos="0.4 0 0.5">
      <geom name="table_top" type="box" size="0.5 0.35 0.02" rgba="0.55 0.5 0.45 1"/>
    </body>
    <body name="arm_base" pos="0 0 0.9">
      <geom name="base_box" type="box" size="0.06 0.06 0.06" rgba="0.3 0.32 0.36 1"/>
      <body name="link1" pos="0 0 0.06">
        <joint name="joint1" type="hinge" axis="0 0 1" pos="0 0 0"/>
        <geom name="link1_box" type="box" size="0.04 0.04 0.12" pos="0 0 0.12"
              rgba="0.7 0.7 0.72 1"/>
        <body name="link2" pos="0 0 0.24" quat="0.9238795 0 0.3826834 0">
          <joint name="joint2" type="hinge" axis="0 1 0" pos="0.01 0 0.02"/>
          <geom name="link2_box" type="box" size="0.03 0.03 0.1" pos="0 0 0.1"
                rgba="0.8 0.78 0.7 1"/>
          <body name="wrist" pos="0 0 0.2">
            <joint name="joint3" type="hinge" axis="1 0 0" pos="0 0 0"/>
            <joint name="slide1" type="slide" axis="0 0 1" range="-0.05 0.05"/>
            <geom name="wrist_box" type="box" size="0.025 0.05 0.02"
                  rgba="0.2 0.6 0.8 1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.4 0.1 0.6">
      <freejoint name="cube_free"/>
      <geom name="cube_box" type="box" size="0.03 0.03 0.03" rgba="0.9 0.3 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# The bodies the ground truth covers: everything that moves, plus one that
# does not, so a page that moved a static body would be caught too.
TRACKED = ("table", "arm_base", "link1", "link2", "wrist", "cube")


def box_mesh(half):
    """Positions, normals and triangles of an axis-aligned box."""
    hx, hy, hz = (float(v) for v in half)
    positions: list[list[float]] = []
    normals: list[list[float]] = []
    indices: list[int] = []
    axes = [
        ((1, 0, 0), hx),
        ((-1, 0, 0), hx),
        ((0, 1, 0), hy),
        ((0, -1, 0), hy),
        ((0, 0, 1), hz),
        ((0, 0, -1), hz),
    ]
    for normal, _ in axes:
        # Two in-plane axes for this face.
        axis = np.array(normal, dtype=np.float64)
        helper = (
            np.array([0.0, 0.0, 1.0])
            if abs(axis[2]) < 0.9
            else np.array([1.0, 0.0, 0.0])
        )
        u = np.cross(axis, helper)
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)
        size = np.array([hx, hy, hz], dtype=np.float64)
        centre = axis * size
        start = len(positions)
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            corner = centre + u * (su * abs(u) @ size) + v * (sv * abs(v) @ size)
            positions.append([float(c) for c in corner])
            normals.append([float(c) for c in axis])
        indices += [start, start + 1, start + 2, start, start + 2, start + 3]
    return (
        np.array(positions, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(indices, dtype=np.uint32).reshape(-1, 3),
    )


class GLB:
    """Just enough glTF 2.0 to write the exporter's node layout."""

    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.meshes: list[dict] = []
        self.materials: list[dict] = []
        self.accessors: list[dict] = []
        self.views: list[dict] = []
        self.buffer = bytearray()

    def _align(self, boundary: int = 4) -> None:
        while len(self.buffer) % boundary:
            self.buffer.append(0)

    def add_view(self, data: bytes, target: int | None = None) -> int:
        self._align()
        offset = len(self.buffer)
        self.buffer += data
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(self, array, kind: str, component: int, target: int) -> int:
        view = self.add_view(array.tobytes(), target)
        flat = (
            array.reshape(-1, array.shape[-1])
            if array.ndim > 1
            else array.reshape(-1, 1)
        )
        self.accessors.append(
            {
                "bufferView": view,
                "componentType": component,
                "count": int(array.shape[0]) if array.ndim > 1 else int(array.size),
                "type": kind,
                "min": [float(v) for v in flat.min(axis=0)],
                "max": [float(v) for v in flat.max(axis=0)],
            }
        )
        return len(self.accessors) - 1

    def material(self, rgba, name: str) -> int:
        self.materials.append(
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [float(v) for v in rgba],
                    "metallicFactor": 0.1,
                    "roughnessFactor": 0.8,
                },
            }
        )
        return len(self.materials) - 1

    def mesh(self, positions, normals, indices, material: int, name: str) -> int:
        self.meshes.append(
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": self.add_accessor(
                                positions, "VEC3", 5126, 34962
                            ),
                            "NORMAL": self.add_accessor(normals, "VEC3", 5126, 34962),
                        },
                        "indices": self.add_accessor(
                            indices.reshape(-1).astype(np.uint32), "SCALAR", 5125, 34963
                        ),
                        "material": material,
                    }
                ],
            }
        )
        return len(self.meshes) - 1

    def write(self, path: pathlib.Path, extras: dict) -> None:
        gltf = {
            "asset": {"version": "2.0", "generator": "oqt webxr test fixture"},
            "scene": 0,
            "scenes": [{"nodes": [0], "extras": extras}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "materials": self.materials,
            "accessors": self.accessors,
            "bufferViews": self.views,
            "buffers": [{"byteLength": len(self.buffer)}],
        }
        body = json.dumps(gltf, separators=(",", ":"), sort_keys=True).encode("utf-8")
        body += b" " * ((4 - len(body) % 4) % 4)
        binary = bytes(self.buffer)
        binary += b"\0" * ((4 - len(binary) % 4) % 4)
        total = 12 + 8 + len(body) + 8 + len(binary)
        with open(path, "wb") as out:
            out.write(struct.pack("<III", 0x46546C67, 2, total))
            out.write(struct.pack("<II", len(body), 0x4E4F534A))
            out.write(body)
            out.write(struct.pack("<II", len(binary), 0x004E4942))
            out.write(binary)


def quat_wxyz_to_xyzw(q):
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def export(model: mujoco.MjModel, path: pathlib.Path) -> None:
    glb = GLB()
    glb.nodes.append({"name": "world", "children": []})
    for body in range(1, model.nbody):
        glb.nodes.append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body),
                "translation": [float(v) for v in model.body_pos[body]],
                "rotation": quat_wxyz_to_xyzw(model.body_quat[body]),
                "children": [],
            }
        )
    for body in range(1, model.nbody):
        glb.nodes[int(model.body_parentid[body])]["children"].append(body)

    joints: dict[str, list] = {}
    for jid in range(model.njnt):
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])
        )
        joints.setdefault(body, []).append(
            {
                "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                "type": ["free", "ball", "slide", "hinge"][int(model.jnt_type[jid])],
                "axis": [float(v) for v in model.jnt_axis[jid]],
                "pos": [float(v) for v in model.jnt_pos[jid]],
                "qposadr": int(model.jnt_qposadr[jid]),
                # A hinge or slide can be exported at a non-zero reference
                # angle; qpos is measured from it.
                "ref": float(model.qpos0[int(model.jnt_qposadr[jid])])
                if int(model.jnt_type[jid]) in (2, 3)
                else 0.0,
            }
        )
    for body in range(1, model.nbody):
        name = glb.nodes[body]["name"]
        if name in joints:
            glb.nodes[body]["extras"] = {"joints": joints[name]}

    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        material = glb.material(model.geom_rgba[geom], name)
        positions, normals, indices = box_mesh(model.geom_size[geom])
        mesh = glb.mesh(positions, normals, indices, material, name)
        glb.nodes.append(
            {
                "name": name,
                "translation": [float(v) for v in model.geom_pos[geom]],
                "rotation": quat_wxyz_to_xyzw(model.geom_quat[geom]),
                "mesh": mesh,
            }
        )
        glb.nodes[int(model.geom_bodyid[geom])]["children"].append(len(glb.nodes) - 1)

    glb.write(
        path,
        {
            "nq": int(model.nq),
            "qpos0": [float(v) for v in model.qpos0],
            "bodies": [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                for b in range(model.nbody)
            ],
        },
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    model = mujoco.MjModel.from_xml_string(MJCF)
    data = mujoco.MjData(model)
    export(model, OUT / "arm.glb")

    frames = []
    for step in range(12):
        phase = step / 11.0
        qpos = np.array(model.qpos0, dtype=np.float64)
        qpos[0] = 1.2 * math.sin(2 * math.pi * phase)
        qpos[1] = 0.9 * math.sin(2 * math.pi * phase + 1.0)
        qpos[2] = -0.7 * math.cos(2 * math.pi * phase)
        qpos[3] = 0.04 * math.sin(4 * math.pi * phase)
        # The free cube: a circle over the table, tipping as it goes.
        angle = 2 * math.pi * phase
        qpos[4:7] = [0.4 + 0.1 * math.cos(angle), 0.1 * math.sin(angle), 0.62]
        half = 0.5 * angle
        qpos[7:11] = [math.cos(half), 0.0, math.sin(half), 0.0]
        data.qpos[:] = qpos
        mujoco.mj_kinematics(model, data)
        frames.append(
            {
                "qpos": [round(float(v), 9) for v in qpos],
                "bodies": {
                    name: [round(float(v), 9) for v in data.xpos[_body(model, name)]]
                    + [round(float(v), 9) for v in data.xquat[_body(model, name)]]
                    for name in TRACKED
                },
            }
        )

    (OUT / "trajectory.json").write_text(
        json.dumps(
            {
                "model": "fixture_arm",
                "nq": int(model.nq),
                "qpos0": [float(v) for v in model.qpos0],
                "tracked": list(TRACKED),
                "free_object": "cube",
                "frames": frames,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT / 'arm.glb'} ({(OUT / 'arm.glb').stat().st_size} bytes)")
    print(f"wrote {OUT / 'trajectory.json'} with {len(frames)} frames")


def _body(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


if __name__ == "__main__":
    main()
