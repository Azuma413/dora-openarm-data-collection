"""End to end: scripted hands -> clutch -> IK -> gate -> MuJoCo, under a real dora.

The webxr node is the only part replaced (by fake_hands.py); the clutch, the IK
launcher, the gate, the HUD and dora-openarm-mujoco run as they do in
modules/leader-webxr.yaml and modules/robot-mujoco.yaml, with the same args.
Two runs of about 8 s each. Run with `pytest -m e2e -s` to see the numbers.

What is asserted is the pipeline's own tracking: the gate's command, through the
IK model's forward kinematics, against the clutch target. The simulated arm is
checked more loosely, from where it settled before the move: dora-openarm-mujoco's
position control has no gravity feed-forward, like the real driver (tau = 0), so
the plant sits 10-17 mm below its command whatever the pipeline does.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dora_openarm_teleop import frames, model_info

pytestmark = pytest.mark.e2e

HERE = Path(__file__).resolve().parent
RUN_S = 6

FLOW = """
nodes:
  - id: hands
    path: {here}/fake_hands.py
    env:
      MOVE_X: "{move_x}"
      MOVE_S: "{move_s}"
      ROT_DEG: "{rot_deg}"
    inputs:
      tick: dora/timer/millis/4
    outputs: [pose_right, pose_left, grip_right, grip_left]

  - id: clutch
    path: dora-openarm-clutch
    args: --scene demo --tick-hz 250
    env:
      ROBOT: mujoco
    inputs:
      tick: dora/timer/millis/4
      pose_right: hands/pose_right
      pose_left: hands/pose_left
      grip_right: hands/grip_right
      grip_left: hands/grip_left
    outputs: [target_right, target_left, clutch_state]

  - id: ik
    path: dora-openarm-teleop-ik
    args: --tick-hz 250 --cache-dir {cache} {ik_extra}
    env:
      ROBOT: mujoco
    inputs:
      target_right: {{source: clutch/target_right, queue_size: 10}}
      target_left: {{source: clutch/target_left, queue_size: 10}}
    outputs: [position_right, position_left, status]

  - id: gate
    path: dora-openarm-gate
    env:
      ROBOT: mujoco
    inputs:
      tick: dora/timer/millis/4
      position_right: ik/position_right
      position_left: ik/position_left
      state_right: mujoco/state_right
      state_left: mujoco/state_left
    outputs: [cmd_right, cmd_left, gate_state]

  - id: hud
    path: dora-openarm-teleop-hud
    inputs:
      tick: dora/timer/millis/100
      clutch_state: clutch/clutch_state
      gate_state: gate/gate_state
    outputs: [hud]

  - id: mujoco
    path: dora-openarm-mujoco
    args: --scene demo --keyframe home --enable-collision --ctrl --no-render
    inputs:
      move_position_right: gate/cmd_right
      move_position_left: gate/cmd_left
      request_state: dora/timer/millis/4
    outputs: [status, state_right, state_left]

  - id: probe
    path: {here}/probe.py
    env:
      OUT: "{out}"
    inputs:
      clutch_state: clutch/clutch_state
      gate_state: gate/gate_state
      hud: hud/hud
      target_right: clutch/target_right
      cmd_right: gate/cmd_right
      cmd_left: gate/cmd_left
      state_right: mujoco/state_right
      state_left: mujoco/state_left
"""


# Right hand: +0.10 m along x and 45 deg about z in 0.5 s (0.2 m/s, 90 deg/s).
MOTION = {"move_x": 0.10, "move_s": 0.5, "rot_deg": 45.0}
MOVE_START_S, MOVE_END_S = 1.5, 2.0
# The per-solve frame-error limits of guchio123/dora-openarm-data-collection 6676c7c.
FORK_LIMITS = (
    "-- --frame-position-error-limit 0.001 --frame-orientation-error-limit 0.004 "
    "--nullspace-cost 5.0 --nullspace-return-rate 2.0"
)


def _run(tmp_path: Path, ik_extra: str = "") -> dict:
    if shutil.which("dora") is None:
        pytest.skip("dora CLI not on PATH")
    out = tmp_path / "probe.json"
    flow = tmp_path / "flow.yaml"
    flow.write_text(
        FLOW.format(here=HERE, out=out, cache=tmp_path / "cache", ik_extra=ik_extra, **MOTION)
    )
    env = {**os.environ, "MUJOCO_GL": "egl"}
    result = subprocess.run(
        ["dora", "run", str(flow), "--stop-after", f"{RUN_S}s"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert out.is_file(), result.stdout[-3000:] + result.stderr[-3000:]
    return json.loads(out.read_text())


def _states(log: dict, name: str) -> list:
    return [json.loads(value).get("right") for _, value in log.get(name, [])]


def _ee(right: list[float], left: list[float]) -> dict[str, np.ndarray]:
    return model_info.ee_poses_from_qpos(np.array([*right[:8], *left[:8]]), "demo")


def _at(log: dict, name: str, when: float) -> list[float]:
    """The value of `name` closest to `when` seconds after the first target."""
    t0 = log["target_right"][0][0]
    rows = log[name]
    return min(rows, key=lambda row: abs(row[0] - t0 - when))[1]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(frames.quat_angle(frames.quat_delta(a, b))))


def tracking(log: dict) -> dict[str, float]:
    """Command-vs-target error while moving and after, and the settling time."""
    t0 = log["target_right"][0][0]
    targets = {round(t - t0, 3): np.array(v) for t, v in log["target_right"]}
    worst_mm = worst_deg = 0.0
    settled = None
    final = log["target_right"][-1][1]
    for (t, right), (_, left) in zip(log["cmd_right"], log["cmd_left"], strict=False):
        rel = t - t0
        if rel < MOVE_START_S:
            continue
        key = min(targets, key=lambda k: abs(k - rel))
        pose = _ee(right, left)["right"]
        target = targets[key]
        worst_mm = max(worst_mm, float(np.linalg.norm(pose[:3] - target[:3])) * 1000)
        worst_deg = max(worst_deg, _angle_deg(pose[3:7], target[3:7]))
        close = (
            np.linalg.norm(pose[:3] - np.array(final[:3])) < 0.002
            and _angle_deg(pose[3:7], np.array(final[3:7])) < 1.0
        )
        if rel >= MOVE_END_S and close and settled is None:
            settled = rel - MOVE_END_S
        elif not close:
            settled = None
    return {
        "worst_mm": worst_mm,
        "worst_deg": worst_deg,
        "settle_s": float("inf") if settled is None else settled,
    }


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    return _run(tmp_path_factory.mktemp("e2e"))


def test_the_right_arm_engages_and_the_gate_stays_engaged(run: dict) -> None:
    clutch = _states(run, "clutch_state")
    assert "engaged" in clutch, clutch
    gate = [s.get("state") if isinstance(s, dict) else s for s in _states(run, "gate_state")]
    assert gate[-1] == "engaged", gate


def test_the_command_tracks_the_hand_closely(run: dict) -> None:
    numbers = tracking(run)
    print(f"\nthis pipeline: {numbers}")
    assert numbers["worst_mm"] < 5.0, numbers
    assert numbers["worst_deg"] < 3.0, numbers
    assert numbers["settle_s"] < 0.1, numbers


def test_the_simulated_arm_follows_from_where_it_settled(run: dict) -> None:
    before = _ee(_at(run, "state_right", 1.4), _at(run, "state_left", 1.4))
    after = _ee(_at(run, "state_right", 5.0), _at(run, "state_left", 5.0))
    moved = after["right"][:3] - before["right"][:3]
    assert moved[0] == pytest.approx(0.10, abs=0.015), moved
    assert np.linalg.norm(moved[1:]) < 0.015, moved
    assert _angle_deg(before["right"][3:7], after["right"][3:7]) == pytest.approx(45.0, abs=5.0)
    # The other arm was never gripped and stays put.
    assert np.linalg.norm(after["left"][:3] - before["left"][:3]) < 0.005


def test_the_command_never_outruns_the_gate(run: dict) -> None:
    times = np.array([t for t, _ in run["cmd_right"]])
    cmd = np.array([v for _, v in run["cmd_right"]])
    steps = np.abs(np.diff(cmd[:, :7], axis=0))
    dt = np.clip(np.diff(times), 1e-3, None)[:, None]
    caps = 0.8 * np.array([2.0, 2.0, 3.3, 3.3, 6.3, 6.3, 6.3])
    # The probe timestamps on receipt: allow one tick of delivery jitter.
    assert np.all(steps <= caps * (dt + 0.004) + 1e-4)


def test_the_hud_reports_the_follow(run: dict) -> None:
    lines = [json.loads(value)["lines"] for _, value in run.get("hud", [])]
    assert lines, "no HUD message"
    assert any("R:FOLLOW" in line for message in lines for line in message), lines[-1]


def test_the_forks_ik_error_limits_lag_behind(run: dict, tmp_path: Path) -> None:
    """The same motion with the per-solve error limits the fork added to the IK.

    0.004 rad per 4 ms solve is 1 rad/s (57 deg/s) at most, so a 90 deg/s turn falls
    behind and settles late; this pipeline (no error limits) does not.
    """
    ours = tracking(run)
    fork = tracking(_run(tmp_path, FORK_LIMITS))
    print(f"\nthis pipeline: {ours}\nfork limits:   {fork}")
    assert fork["worst_deg"] > 3 * ours["worst_deg"] + 1.0, (ours, fork)
    assert fork["settle_s"] > ours["settle_s"] + 0.1, (ours, fork)
