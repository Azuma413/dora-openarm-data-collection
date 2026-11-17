# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).

"""Ground truth for the 3D page test, for a scene.glb a run actually exported.

``make_fixture.py`` writes the committed fixture and the trajectory the page
test runs against. That fixture is a toy arm, so its trajectory names its own
bodies (``wrist``, ``cube``) and carries its own ``nq``; point the test at a
real ``runs/<name>/scene.glb`` instead and there is no ground truth to compare
the drawn robot against.

This writes one, in the same shape, for any asset ``scripts/mjcf_to_glb.py``
produced::

    uv run python third_party/dora-openarm-webxr/tests/web/scene_truth.py \\
        runs/<name>/scene.glb -o trajectory.json

The poses come from MuJoCo itself: the export's manifest (``<glb>.json``) names
the MJCF it was made from, so the model is recompiled and ``mj_kinematics``
answers for each frame. Without the manifest -- somebody handed over a bare
GLB -- it falls back to ``mjcf_to_glb.body_world_poses``, the exporter's
reference implementation, which ``tests/unit/test_mjcf_to_glb.py`` pins to
``mj_kinematics`` to under 0.01 mm on these same scenes.

The poses are pure forward kinematics, so the configurations need only be
valid, not physical: they are the export's own rest pose with the hinges and
slides swung about it and any free body carried round a small circle.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
# .../third_party/dora-openarm-webxr/tests/web -> the repository root.
REPO_ROOT = HERE.parents[3]
EXPORTER = REPO_ROOT / "scripts" / "mjcf_to_glb.py"
# How far a hinge or a slide is swung about the rest pose. Small enough that
# the scene stays recognisable in a capture, large enough that a body moving
# the wrong way is unmistakable.
HINGE_SWING_RAD = 0.35
SLIDE_SWING_M = 0.02
FREE_RADIUS_M = 0.05


def load_exporter():
    """scripts/mjcf_to_glb.py, imported by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("mjcf_to_glb", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass resolves through sys.modules
    spec.loader.exec_module(module)
    return module


def read_glb(path: pathlib.Path) -> dict:
    """The JSON chunk of a binary glTF."""
    payload = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<III", payload, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path} is not a binary glTF")
    chunk_length, chunk_type = struct.unpack_from("<II", payload, 12)
    if chunk_type != 0x4E4F534A:
        raise ValueError(f"{path}: the first chunk is not JSON")
    return json.loads(payload[20 : 20 + chunk_length])


def moving_bodies(document: dict, limit: int = 6) -> list[str]:
    """Bodies whose pose a joint value can change, deepest first.

    A body with no joint anywhere above it never moves, and comparing the page
    against MuJoCo there would pass however wrong the kinematics are.
    """
    nodes = document["nodes"]
    parent = {c: i for i, node in enumerate(nodes) for c in node.get("children", [])}
    depth: dict[int, int] = {}
    driven: list[tuple[int, int]] = []
    for index, node in enumerate(nodes):
        chain = [index]
        while chain[-1] in parent:
            chain.append(parent[chain[-1]])
        depth[index] = len(chain) - 1
        if any(nodes[i].get("extras", {}).get("joints") for i in chain):
            driven.append((depth[index], index))
    driven.sort(reverse=True)
    names = [nodes[index].get("name", f"node{index}") for _, index in driven]
    # Deepest first (the wrist and the fingers, where an error in any joint
    # above accumulates), then spread over the rest so a mistake in one link
    # is not hidden by looking only at the tip.
    if len(names) <= limit:
        return names
    step = max(1, len(names) // limit)
    return names[:2] + names[2 : limit * step : step][: limit - 2]


def joint_table(document: dict) -> list[dict]:
    """Every joint in the asset, in MuJoCo's own order."""
    joints = [
        joint
        for node in document["nodes"]
        for joint in node.get("extras", {}).get("joints", [])
    ]
    return sorted(joints, key=lambda joint: int(joint["qposadr"]))


def frames(document: dict, count: int) -> list[np.ndarray]:
    """`count` valid configurations, swung about the export's own rest pose."""
    extras = document["scenes"][document.get("scene", 0)].get("extras", {})
    nq = int(extras["nq"])
    rest = np.asarray(extras.get("qpos0") or [0.0] * nq, dtype=np.float64)
    joints = joint_table(document)
    out = []
    for step in range(count):
        phase = 2 * math.pi * step / count
        qpos = rest.copy()
        for order, joint in enumerate(joints):
            adr = int(joint["qposadr"])
            wave = math.sin(phase + 0.7 * order)
            if joint["type"] == "hinge":
                qpos[adr] = rest[adr] + HINGE_SWING_RAD * wave
            elif joint["type"] == "slide":
                qpos[adr] = rest[adr] + SLIDE_SWING_M * wave
            elif joint["type"] == "ball":
                half = 0.5 * HINGE_SWING_RAD * wave
                qpos[adr : adr + 4] = [math.cos(half), math.sin(half), 0.0, 0.0]
            elif joint["type"] == "free":
                qpos[adr] = rest[adr] + FREE_RADIUS_M * math.cos(phase + order)
                qpos[adr + 1] = rest[adr + 1] + FREE_RADIUS_M * math.sin(phase + order)
                half = 0.5 * phase
                qpos[adr + 3 : adr + 7] = [math.cos(half), 0.0, 0.0, math.sin(half)]
        out.append(qpos)
    return out


def mujoco_poses(source: pathlib.Path, gripper: str | None, tracked: list[str], poses):
    """MuJoCo's own xpos/xquat for each configuration, or None if it cannot load."""
    try:
        import mujoco

        exporter = load_exporter()
        model, _ = exporter.load_scene_model(xml=source, gripper=gripper)
        data = mujoco.MjData(model)
        ids = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in tracked
        }
        if any(index < 0 for index in ids.values()) or model.nq != len(poses[0]):
            return None
        out = []
        for qpos in poses:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            out.append(
                {
                    name: [float(v) for v in data.xpos[index]]
                    + [float(v) for v in data.xquat[index]]
                    for name, index in ids.items()
                }
            )
        return out
    except Exception as error:  # noqa: BLE001 - the fallback below is the point
        print(
            f"scene_truth: {source}: {error}; using the exporter's kinematics",
            flush=True,
        )
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("glb", type=pathlib.Path)
    parser.add_argument("-o", "--out", type=pathlib.Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument(
        "--bodies", default="", help="comma separated; default: the moving ones"
    )
    args = parser.parse_args(argv)

    document = read_glb(args.glb)
    extras = document["scenes"][document.get("scene", 0)].get("extras", {})
    tracked = [n.strip() for n in args.bodies.split(",") if n.strip()] or moving_bodies(
        document
    )
    poses = frames(document, args.frames)

    manifest_path = args.glb.with_suffix(args.glb.suffix + ".json")
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bodies = None
    if manifest.get("source"):
        bodies = mujoco_poses(
            pathlib.Path(manifest["source"]), manifest.get("gripper"), tracked, poses
        )
    truth = "mujoco"
    if bodies is None:
        truth = "body_world_poses"
        exporter = load_exporter()
        bodies = []
        for qpos in poses:
            computed = exporter.body_world_poses(document, qpos)
            bodies.append(
                {
                    name: [float(v) for v in computed[name][0]]
                    + [float(v) for v in computed[name][1]]
                    for name in tracked
                }
            )

    free = next(
        (
            node.get("name")
            for node in document["nodes"]
            for joint in node.get("extras", {}).get("joints", [])
            if joint["type"] == "free" and node.get("name") in tracked
        ),
        None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": extras.get("scene") or args.glb.stem,
                "source": str(args.glb),
                "truth": truth,
                "nq": int(extras["nq"]),
                "qpos0": [float(v) for v in extras.get("qpos0", [])],
                "tracked": tracked,
                "free_object": free,
                "frames": [
                    {
                        "qpos": [round(float(v), 9) for v in qpos],
                        "bodies": bodies[index],
                    }
                    for index, qpos in enumerate(poses)
                ],
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {args.out}: {len(poses)} frames, nq {extras['nq']}, "
        f"{len(tracked)} bodies ({', '.join(tracked)}) from {truth}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
