"""PolicyNode without dora or torch: events in, outputs recorded."""

import json
import time

import cv2
import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_policy import web
from dora_openarm_policy.main import PolicyNode, build_parser, qpos_array, resolve_defaults
from dora_openarm_policy.model import decode_image


class FakePolicy:
    horizon = 30

    def __init__(self):
        self.calls = []

    def predict(self, head, wrist, state16, bbox, seed=None):
        self.calls.append((head, wrist, state16.copy(), bbox.copy()))
        return np.tile(state16, (30, 1)) + np.arange(30)[:, None] * 0.001

    def model_view(self, head, wrist, bbox):
        return np.zeros((10, 10, 3), np.uint8)


def jpeg(rgb):
    ok, data = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return pa.array(np.frombuffer(data.tobytes(), np.uint8))


def state(q):
    return pa.array(
        [
            {
                "qpos": np.asarray(q, np.float32),
                "qvel": np.zeros(8, np.float32),
                "qtorque": np.zeros(8, np.float32),
            }
        ]
    )


def ev(input_id, value, **metadata):
    return {"type": "INPUT", "id": input_id, "value": value, "metadata": metadata}


@pytest.fixture
def node():
    sent = []
    args = build_parser().parse_args([])
    resolve_defaults(args)
    n = PolicyNode(args, web.Shared(), lambda o, v, m: sent.append((o, v)), log=lambda m: None)
    n.sent = sent
    return n


def feed_world(n, q=0.1, clutch="released"):
    for side in ("right", "left"):
        n.on_event(ev(f"state_{side}", state(np.full(8, q)), start_epoch=1))
        n.on_event(ev(f"ik_{side}", qpos_array(np.full(8, q))))
        n.on_event(ev(f"cmd_{side}", qpos_array(np.full(8, q))))
    n.on_event(
        ev(
            "gate_state",
            pa.array(
                [
                    json.dumps(
                        {s: {"state": "engaged", "reason": "", "homing": False} for s in ("right", "left")}
                    )
                ]
            ),
        )
    )
    n.on_event(ev("clutch_state", pa.array([json.dumps({"right": "released", "left": clutch})])))
    image = np.zeros((720, 1280, 3), np.uint8)
    image[..., 0] = 200  # red in RGB
    n.on_event(ev("camera_head_left", jpeg(image), encoding="jpeg", width=1280, height=720))
    n.on_event(
        ev(
            "camera_wrist_right",
            jpeg(np.zeros((600, 960, 3), np.uint8)),
            encoding="jpeg",
            width=960,
            height=600,
        )
    )


def outputs(n, name):
    return [v for o, v in n.sent if o == name]


def test_the_ik_passes_through_until_started(node):
    feed_world(node)
    assert len(outputs(node, "position_right")) == 1
    node.on_event(ev("tick", pa.array([0])))
    assert len(outputs(node, "position_right")) == 1


def test_start_is_refused_without_a_bbox_or_during_the_intro(node):
    node.policy = FakePolicy()
    feed_world(node, clutch="waiting")
    assert "bbox" in node._why_not_start(time.monotonic())
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    assert "intro" in node._why_not_start(time.monotonic())


def test_ui_proceed_and_reset_reach_the_clutch(node):
    feed_world(node, clutch="waiting")
    node.shared.requests.put(("action", "proceed"))
    node.on_event(ev("tick", pa.array([0])))
    node.shared.requests.put(("action", "reset"))
    node.on_event(ev("tick", pa.array([0])))
    assert [v[0].as_py() for v in outputs(node, "clutch_control")] == ["proceed", "cancel"]


def test_running_uses_rgb_the_bbox_in_1280_px_and_stops_from_the_ui(node):
    node.policy = FakePolicy()
    feed_world(node)
    node.shared.requests.put(("bbox", [0.25, 0.5, 0.5, 0.75]))
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "running"
    obs = node._snapshot()
    head_cam, _wrist_cam, bbox, state16 = obs
    head = decode_image(head_cam.data, head_cam.encoding, head_cam.width, head_cam.height)
    assert head[..., 0].mean() > 150 and head[..., 2].mean() < 50  # RGB, not BGR
    np.testing.assert_allclose(node._bbox_px(bbox), [320, 360, 640, 540])
    np.testing.assert_allclose(state16, 0.1, atol=1e-6)
    chunk = node.policy.predict(head, None, state16, node._bbox_px(bbox))
    node.results.put(("chunk", chunk, time.monotonic(), node.ctl.run_id, {"latency_ms": 1}))
    before = len(outputs(node, "position_right"))
    node.on_event(ev("tick", pa.array([0])))
    assert len(outputs(node, "position_right")) == before + 1
    # The IK is no longer forwarded.
    node.on_event(ev("ik_right", qpos_array(np.full(8, 0.9))))
    assert len(outputs(node, "position_right")) == before + 1
    node.shared.requests.put(("action", "stop"))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "stopped"


def test_the_policy_stops_when_the_gate_holds(node):
    node.policy = FakePolicy()
    feed_world(node)
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "running"
    node.on_event(
        ev(
            "gate_state",
            pa.array(
                [
                    json.dumps(
                        {"right": {"state": "holding", "reason": "stale_state"}, "left": {"state": "engaged"}}
                    )
                ]
            ),
        )
    )
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "stopped" and "gate right" in node.ctl.reason


def test_an_episode_end_while_running_goes_back_before_the_clutch_hears_it(node):
    node.policy = FakePolicy()
    feed_world(node)
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    node.on_event(ev("command", pa.array(["fail"])))
    assert node.ctl.mode == "returning" and outputs(node, "clutch_control") == []
    node.ctl._ret_t -= 10.0  # as if the return had run its course
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "passthrough"
    assert [v[0].as_py() for v in outputs(node, "clutch_control")] == ["fail"]


def test_quit_is_forwarded_and_ends_the_node(node):
    assert node.on_event(ev("command", pa.array(["quit"]))) is False
    assert [v[0].as_py() for v in outputs(node, "clutch_control")] == ["quit"]


def test_a_raw_bgra_image_becomes_rgb():
    bgra = np.zeros((2, 2, 4), np.uint8)
    bgra[..., 2] = 255  # red
    rgb = decode_image(bgra.tobytes(), "bgra8", 2, 2)
    assert rgb.shape == (2, 2, 3) and rgb[0, 0, 0] == 255 and rgb[0, 0, 2] == 0


def test_head_left_is_frozen_after_the_intro_and_after_a_run(node):
    node.policy = FakePolicy()
    feed_world(node, clutch="waiting")
    assert node.snapshot is None
    node.on_event(ev("clutch_state", pa.array([json.dumps({"right": "stale", "left": "stale"})])))
    node.on_event(ev("tick", pa.array([0])))
    assert node.snapshot is None  # not before the arms settle
    node._snapshot_due = (0.0, node._snapshot_due[1])
    node.on_event(ev("tick", pa.array([0])))
    assert node.snapshot["label"] == "初期動作後" and node.shared.snapshot_jpeg == node.shared.head_jpeg
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    node.shared.requests.put(("action", "stop"))
    node.on_event(ev("tick", pa.array([0])))
    node._snapshot_due = (0.0, node._snapshot_due[1])
    node.on_event(ev("tick", pa.array([0])))
    assert node.snapshot["label"] == "動作完了後" and node.snapshot["seq"] == 2


def driver_state(q, statuses):
    return pa.array(
        [
            {
                "qpos": np.asarray(q, np.float32),
                "qvel": np.zeros(8, np.float32),
                "qtorque": np.zeros(8, np.float32),
                "motor_status": statuses,
            }
        ]
    )


def test_a_motor_that_restarted_without_torque_stops_the_policy(node):
    node.policy = FakePolicy()
    feed_world(node)
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "running"
    restarted = ["ENABLED"] * 8
    restarted[5] = "DISABLED"
    node.on_event(ev("state_left", driver_state(np.full(8, 0.1), restarted), start_epoch=1))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "stopped" and node.ctl.reason == "left arm J6=DISABLED"
    assert "J6=DISABLED" in node._why_not_start(time.monotonic())
    # A paused arm (all disabled) is not a fault.
    node.on_event(ev("state_left", driver_state(np.full(8, 0.1), ["DISABLED"] * 8), start_epoch=1))
    assert node.motor_faults["left"] == []


def test_a_voice_stops_the_policy_with_its_own_reason(node):
    node.policy = FakePolicy()
    feed_world(node)
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    node.shared.requests.put(("action", "voice_stop"))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.mode == "stopped" and node.ctl.reason == "voice"


def test_dotenv_fills_unset_or_empty_variables_only(tmp_path, monkeypatch):
    from dora_openarm_policy.main import load_dotenv

    env = tmp_path / ".env"
    env.write_text(
        "# comment\nOPENAI_API_KEY=sk-test\nexport POLICY='molmoact2'\nVLM_MODEL=from-file  # note\n"
        "ALREADY=file\nnot a line\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("POLICY", "")  # the module passes it empty
    monkeypatch.setenv("ALREADY", "shell")
    monkeypatch.delenv("VLM_MODEL", raising=False)
    assert sorted(load_dotenv(env)) == ["OPENAI_API_KEY", "POLICY", "VLM_MODEL"]
    import os

    assert os.environ["OPENAI_API_KEY"] == "sk-test" and os.environ["POLICY"] == "molmoact2"
    assert os.environ["VLM_MODEL"] == "from-file" and os.environ["ALREADY"] == "shell"
    assert load_dotenv(tmp_path / "missing") == []


def test_per_policy_defaults_and_seed():
    small = build_parser().parse_args([])
    resolve_defaults(small)
    assert (small.execute_steps, small.blend_s, small.seed) == (10, 0.15, 0)
    molmo = build_parser().parse_args(["--policy", "molmoact2", "--seed", "random"])
    resolve_defaults(molmo)
    assert (molmo.execute_steps, molmo.blend_s, molmo.seed) == (10, 0.4, None)
    given = build_parser().parse_args(["--policy", "molmoact2", "--execute-steps", "8", "--blend-s", "0.2"])
    resolve_defaults(given)
    assert (given.execute_steps, given.blend_s) == (8, 0.2)


def test_speed_from_the_ui_and_per_policy_default(node):
    assert node.ctl.speed == 1.0
    node.shared.requests.put(("speed", 0.3))
    node.on_event(ev("tick", pa.array([0])))
    assert node.ctl.speed == pytest.approx(0.3)
    molmo = build_parser().parse_args(["--policy", "molmoact2"])
    resolve_defaults(molmo)
    assert molmo.speed == 0.5


def test_vad_stats_are_logged_on_a_new_onset_and_every_10_s(node):
    lines = []
    node._print = lines.append
    stats = {
        "level_p50": -70.2,
        "level_max": -61,
        "floor": -72,
        "sensitivity": 12,
        "min_level": -58,
        "onsets": 0,
    }
    node.shared.requests.put(("vad_stats", stats))
    node.on_event(ev("tick", pa.array([0])))
    node.shared.requests.put(("vad_stats", stats))
    node.on_event(ev("tick", pa.array([0])))
    node.shared.requests.put(("vad_stats", {**stats, "onsets": 1}))
    node.on_event(ev("tick", pa.array([0])))
    detection = [line for line in lines if "voice detection" in line]
    assert len(detection) == 2 and "floor -72 dB" in detection[0] and "onsets=1" in detection[1]


def test_the_stall_check_runs_once_a_second_while_running(node):
    lines = []
    node._print = lines.append
    node.policy = FakePolicy()
    feed_world(node)
    node.bbox = [0.4, 0.4, 0.6, 0.6]
    node.shared.requests.put(("action", "start"))
    node.on_event(ev("tick", pa.array([0])))
    node._diag_t -= 2.0
    node.on_event(ev("tick", pa.array([0])))
    assert not [line for line in lines if "stall check" in line]  # no chunk yet
    chunk = np.tile(np.full(16, 0.1), (30, 1))
    node.results.put(("chunk", chunk, time.monotonic(), node.ctl.run_id, {"latency_ms": 1}))
    node.on_event(ev("tick", pa.array([0])))
    node._diag_t -= 2.0
    node.on_event(ev("tick", pa.array([0])))
    checks = [line for line in lines if "stall check" in line]
    assert checks and "R: target" in checks[0] and "L: target" in checks[0]
    assert set(node.diag) == {"right", "left"} and "verdict" in node.diag["right"]
