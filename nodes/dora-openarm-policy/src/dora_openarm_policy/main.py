"""Dora node ``dora-openarm-policy``: the small flow policy between the IK and the gate.

It sits where the gate's ``position_*`` inputs are. Until the operator starts the
policy it forwards the IK's output unchanged, so the clutch runs the left arm's
intro (turn, open, wait, close, turn back) and holds both arms as in data
collection. Then the policy drives both arms with 30-step joint chunks, re-planned
every --execute-steps, conditioned on the bbox the operator drags on head_left in
the browser UI (http://localhost:8010 by default). The gate stays in the loop:
joint ranges, speed caps and jump checks apply to the policy as to the operator.

Inputs:
  tick                      leader tick (250 Hz)
  ik_right/left             QPOS8 from the IK (what the clutch asks for)
  cmd_right/left            QPOS8 the gate sends to the robot (where Stop holds)
  state_right/left          STATE8 measured joints (policy state; start_epoch)
  camera_head_left          JPEG, ZED left (policy input, UI view)
  camera_wrist_right        JPEG (policy input)
  command                   ui/command; "success"/"fail"/"cancel" while the policy
                            has the arms first ease them back, then go to the clutch
  gate_state, clutch_state  JSON, shown in the UI and checked before/while running

Outputs:
  position_right/left       QPOS8 to the gate
  clutch_control            the clutch's control: ui/command, plus "proceed" (the
                            patch is in the gripper) and "cancel" (go back and run
                            the intro again) from the UI

The policy stops by itself (holds) when the gate stops following (any state but
engaged), when a camera or the arm state goes stale, or when inference fails.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa

from dora_openarm_policy import web
from dora_openarm_policy.controller import (
    PASSTHROUGH,
    RUNNING,
    SIDES,
    STOPPED,
    Controller,
    ControllerParams,
)
from dora_openarm_policy.diagnostics import StallMonitor
from dora_openarm_policy.model import BBOX_REF, decode_image

QPOS_TYPE = pa.struct({"qpos": pa.list_(pa.float32())})
EPISODE_END_COMMANDS = frozenset({"success", "fail", "cancel"})
CAMERAS = {"camera_head_left": "head_left", "camera_wrist_right": "wrist_right"}
# Clutch states in which the intro still has the arm.
INTRO_STATES = frozenset({"setup", "waiting"})
# How long after the intro ends or the arms stop head_left is frozen for the UI.
SNAPSHOT_SETTLE_S = 0.5


def qpos_array(q: np.ndarray) -> pa.Array:
    return pa.array([{"qpos": np.asarray(q, np.float32)}], type=QPOS_TYPE)


def values(value: pa.Array, key: str) -> np.ndarray:
    if pa.types.is_struct(value.type):
        value = value.field(key)[0].values
    return np.asarray(value.to_numpy(zero_copy_only=False), dtype=np.float64)


def motor_faults(value: pa.Array) -> list[str]:
    """Motors that are not driving the arm, from dora-openarm's motor_status.

    The same rule as the gate's state_faults: anything but ENABLED/DISABLED (SILENT,
    OVERCURRENT, ...), and DISABLED while other motors are ENABLED (a motor that
    restarted mid-run and holds no torque until the arm is stopped and started).
    """
    try:
        if not pa.types.is_struct(value.type) or "motor_status" not in value.type.names:
            return []
        statuses = list(value.field("motor_status")[0].as_py() or [])
    except Exception:  # noqa: BLE001 - nothing to judge by
        return []
    names = [f"J{i + 1}" if i < 7 else "gripper" for i in range(len(statuses))]
    faults = [f"{n}={s}" for n, s in zip(names, statuses) if s not in ("ENABLED", "DISABLED")]
    if "ENABLED" in statuses:
        faults += [f"{n}=DISABLED" for n, s in zip(names, statuses) if s == "DISABLED"]
    return faults


def start_epoch(metadata: Any) -> int | None:
    value = dict(metadata or {}).get("start_epoch")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class _Camera:
    __slots__ = ("data", "encoding", "height", "t", "width")

    def __init__(self, data: bytes, encoding: str, width: Any, height: Any, t: float) -> None:
        self.data, self.encoding, self.width, self.height, self.t = data, encoding, width, height, t


class PolicyNode:
    def __init__(self, args: argparse.Namespace, shared: web.Shared, send, log=None) -> None:
        self.args = args
        self.shared = shared
        self.send = send
        self._log_lines: collections.deque[str] = collections.deque(maxlen=40)
        self._print = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.ctl = Controller(ControllerParams(return_speed=args.return_speed, blend_s=args.blend_s))
        self.replan_s = args.execute_steps / 30.0
        self.ctl.speed = args.speed
        self.lock = threading.Lock()  # cameras, state, bbox for the inference thread
        self.cams: dict[str, _Camera] = {}
        self.state: dict[str, tuple[np.ndarray, float]] = {}
        self.epochs: dict[str, int] = {}
        self.bbox: list[float] | None = None  # normalized x1, y1, x2, y2
        self.gate_state: dict[str, Any] = {}
        self.clutch_state: dict[str, Any] = {}
        self.motor_faults: dict[str, list[str]] = {}
        self.presets_status: list[dict] = []
        self._vad_onsets = None
        self._vad_logged = 0.0
        # Once a second while running: is a stall the policy or the arm? (diagnostics.py)
        self.stall = StallMonitor()
        self.state_extra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.diag: dict[str, dict] = {}
        self._diag_t = 0.0
        self.policy = None
        self.policy_error: str | None = None
        self.policy_info: dict[str, Any] = {}
        self.results: queue.Queue = queue.Queue()
        self.infer: dict[str, Any] = {"count": 0}
        self._test = threading.Event()
        self.done = threading.Event()
        self._run_started = 0.0
        self._last_status = 0.0
        self._mode_seen = self.ctl.mode
        # (when, label): freeze head_left for the UI then, once the arms have settled.
        self._snapshot_due: tuple[float, str] | None = None
        self.snapshot: dict[str, Any] | None = None

    # -- logging ------------------------------------------------------------

    def log(self, message: str) -> None:
        self._print(f"[dora-openarm-policy] {message}")
        self._log_lines.append(time.strftime("%H:%M:%S ") + message)

    # -- policy loading and inference (threads) -----------------------------

    def load_policy(self) -> None:
        try:
            t0 = time.monotonic()
            if self.args.policy == "molmoact2":
                from dora_openarm_policy.model import MolmoAct2Runner

                checkpoint = self.args.molmoact2_checkpoint
                policy = MolmoAct2Runner(
                    Path(checkpoint), Path(self.args.molmoact2_experiments), steps=self.args.flow_steps
                )
            else:
                from dora_openarm_policy.model import Policy

                checkpoint = self.args.checkpoint
                policy = Policy(Path(checkpoint), Path(self.args.small_flow), steps=self.args.flow_steps)
            warm = policy.warm_up()
            self.policy_info = {
                "name": "MolmoAct2" if self.args.policy == "molmoact2" else "small flow",
                "checkpoint": str(checkpoint),
                "trained_steps": policy.step_trained,
                "horizon": policy.horizon,
                "load_s": round(time.monotonic() - t0, 1),
                "warm_ms": round(warm * 1000, 1),
            }
            self.policy = policy
            self.log(
                f"policy loaded: {self.policy_info['name']} {checkpoint} ({policy.step_trained}), "
                f"{warm * 1000:.0f} ms per inference"
            )
        except Exception as error:  # noqa: BLE001 - shown in the UI, the node keeps passing through
            self.policy_error = f"{type(error).__name__}: {error}"
            self.log(f"policy failed to load: {self.policy_error}")
            traceback.print_exc()

    def _bbox_px(self, bbox: list[float]) -> np.ndarray:
        return np.array(
            [bbox[0] * BBOX_REF[0], bbox[1] * BBOX_REF[1], bbox[2] * BBOX_REF[0], bbox[3] * BBOX_REF[1]],
            np.float32,
        )

    def _snapshot(self):
        with self.lock:
            head = self.cams.get("head_left")
            wrist = self.cams.get("wrist_right")
            bbox = list(self.bbox) if self.bbox else None
            state = [self.state.get(s) for s in SIDES]
        if head is None or wrist is None or bbox is None or any(s is None for s in state):
            return None
        state16 = np.concatenate([state[0][0][:8], state[1][0][:8]])
        return head, wrist, bbox, state16

    def infer_loop(self) -> None:
        last_run: int | None = None
        last_obs = 0.0
        last_preview = 0.0
        while not self.done.is_set():
            if self.policy is None:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            mode, run_id = self.ctl.mode, self.ctl.run_id
            due = mode == RUNNING and (run_id != last_run or now - last_obs >= self.replan_s)
            test = not due and self._test.is_set()
            if not due and not test:
                if now - last_preview > 0.5:
                    last_preview = now
                    self._preview()
                time.sleep(0.002)
                continue
            snap = self._snapshot()
            if snap is None:
                if test:
                    self._test.clear()
                    self.results.put(("test_error", "no camera image, arm state or bbox yet"))
                time.sleep(0.01)
                continue
            head_cam, wrist_cam, bbox, state16 = snap
            t_obs = time.monotonic()
            try:
                head = decode_image(head_cam.data, head_cam.encoding, head_cam.width, head_cam.height)
                wrist = decode_image(wrist_cam.data, wrist_cam.encoding, wrist_cam.width, wrist_cam.height)
                bbox_px = self._bbox_px(bbox)
                chunk = self.policy.predict(head, wrist, state16, bbox_px, seed=self.args.seed)
            except Exception as error:  # noqa: BLE001 - reported, and the policy stops
                traceback.print_exc()
                self._test.clear()
                self.results.put(("error", f"{type(error).__name__}: {error}", run_id))
                last_run, last_obs = run_id, time.monotonic()
                continue
            latency = time.monotonic() - t_obs
            info = self._chunk_info(chunk, state16, latency, t_obs - min(head_cam.t, wrist_cam.t))
            if due:
                self.results.put(("chunk", chunk, t_obs, run_id, info))
                last_run, last_obs = run_id, t_obs
            else:
                self._test.clear()
                self.results.put(("test", info))
            self._store_view(head, wrist, bbox_px)
            last_preview = time.monotonic()

    @staticmethod
    def _chunk_info(chunk: np.ndarray, state16: np.ndarray, latency: float, image_age: float) -> dict:
        joints = np.r_[0:7, 8:15]
        gap = np.abs(chunk[0, joints] - state16[joints])
        travel = np.abs(chunk[-1, joints] - chunk[0, joints])
        return {
            "latency_ms": round(latency * 1000, 1),
            "image_age_ms": round(image_age * 1000, 1),
            # The first target against where the arms are, and how far the chunk goes, rad.
            "gap_rad": round(float(gap.max()), 3),
            "gap_joint": _joint_name(int(joints[int(gap.argmax())])),
            "travel_rad": round(float(travel.max()), 3),
            "travel_joint": _joint_name(int(joints[int(travel.argmax())])),
        }

    def _preview(self) -> None:
        snap = self._snapshot_images()
        if snap is None:
            return
        head_cam, wrist_cam, bbox = snap
        try:
            head = decode_image(head_cam.data, head_cam.encoding, head_cam.width, head_cam.height)
            wrist = decode_image(wrist_cam.data, wrist_cam.encoding, wrist_cam.width, wrist_cam.height)
        except Exception:  # noqa: BLE001 - only a preview
            return
        bbox_px = self._bbox_px(bbox) if bbox else np.full(4, np.nan, np.float32)
        self._store_view(head, wrist, bbox_px)

    def _snapshot_images(self):
        with self.lock:
            head = self.cams.get("head_left")
            wrist = self.cams.get("wrist_right")
            bbox = list(self.bbox) if self.bbox else None
        if head is None or wrist is None:
            return None
        return head, wrist, bbox

    def _store_view(self, head: np.ndarray, wrist: np.ndarray, bbox_px: np.ndarray) -> None:
        if self.policy is None:
            return
        view = self.policy.model_view(head, wrist, bbox_px)
        view = cv2.resize(view, (view.shape[1] * 2, view.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        ok, jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            with self.shared.lock:
                self.shared.view_jpeg = jpeg.tobytes()

    # -- the dora loop ------------------------------------------------------

    def on_event(self, event: dict[str, Any]) -> bool:
        """Handle one dora event. False once the node should stop."""
        kind = event.get("type")
        if kind == "STOP":
            return False
        if kind == "INPUT_CLOSED":
            return event.get("id") != "command"
        if kind != "INPUT":
            return True
        now = time.monotonic()
        input_id = event["id"]
        value = event["value"]
        metadata = event.get("metadata") or {}
        if input_id == "tick":
            self._tick(now)
        elif input_id in ("ik_right", "ik_left"):
            side = input_id.removeprefix("ik_")
            self.ctl.on_ik(side, values(value, "qpos"))
            if self.ctl.mode == PASSTHROUGH:
                self.send(f"position_{side}", value, metadata)
        elif input_id in ("cmd_right", "cmd_left"):
            self.ctl.on_cmd(input_id.removeprefix("cmd_"), values(value, "qpos"))
        elif input_id in ("state_right", "state_left"):
            self._on_state(input_id.removeprefix("state_"), value, metadata, now)
        elif input_id in CAMERAS:
            self._on_camera(CAMERAS[input_id], value, metadata, now)
        elif input_id == "gate_state":
            self.gate_state = _json(value) or self.gate_state
        elif input_id == "clutch_state":
            before = self.clutch_state.get("left")
            self.clutch_state = _json(value) or self.clutch_state
            if before in INTRO_STATES and self.clutch_state.get("left") not in INTRO_STATES:
                self._snapshot_due = (now + SNAPSHOT_SETTLE_S, "初期動作後")
        elif input_id == "command":
            return self._on_command(str(value[0].as_py()).strip(), now)
        return True

    def _on_state(self, side: str, value: pa.Array, metadata: dict, now: float) -> None:
        q = values(value, "qpos")
        if q.shape[0] < 8:
            return
        with self.lock:
            self.state[side] = (q[:8].copy(), now)
        names = value.type.names if pa.types.is_struct(value.type) else []
        if "qvel" in names and "qtorque" in names:
            self.state_extra[side] = (values(value, "qvel")[:8].copy(), values(value, "qtorque")[:8].copy())
        faults = motor_faults(value)
        if faults != self.motor_faults.get(side, []):
            if faults:
                hint = (
                    " (the motor restarted: no torque until the arm is stopped and started)"
                    if any(f.endswith("=DISABLED") for f in faults)
                    else ""
                )
                self.log(f"{side} arm motors: {', '.join(faults)}{hint}")
            else:
                self.log(f"{side} arm motors: all answering again")
            self.motor_faults[side] = faults
        epoch = start_epoch(metadata)
        if epoch is not None:
            previous = self.epochs.get(side)
            self.epochs[side] = epoch
            if previous is not None and epoch != previous and self.ctl.mode != PASSTHROUGH:
                self.ctl.to_passthrough()
                self.log(f"the {side} arm restarted (start_epoch {epoch}): policy off, back to the clutch")

    def _on_camera(self, name: str, value: pa.Array, metadata: dict, now: float) -> None:
        data = value.to_numpy(zero_copy_only=False).tobytes()
        encoding = str(metadata.get("encoding", "jpeg"))
        cam = _Camera(data, encoding, metadata.get("width"), metadata.get("height"), now)
        with self.lock:
            self.cams[name] = cam
        if name == "head_left":
            if encoding.lower() in ("jpeg", "jpg"):
                self.shared.put_head(data)
            else:
                try:
                    rgb = decode_image(data, encoding, cam.width, cam.height)
                    ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if ok:
                        self.shared.put_head(jpeg.tobytes())
                except Exception:  # noqa: BLE001, S110 - only the UI view
                    pass

    def _on_command(self, command: str, now: float) -> bool:
        if command == "quit":
            self._clutch("quit")
            return False
        if command in EPISODE_END_COMMANDS and self.ctl.mode != PASSTHROUGH:
            why = self.ctl.go_back(now, then=command)
            self.log(
                f"ui {command}: easing back to the clutch first" if why is None else f"ui {command}: {why}"
            )
            if why is not None:
                self._clutch(command)
            return True
        self._clutch(command)
        return True

    def _clutch(self, command: str) -> None:
        self.send("clutch_control", pa.array([command]), {})

    def _tick(self, now: float) -> None:
        self._take_requests(now)
        if self._snapshot_due is not None and now >= self._snapshot_due[0]:
            self._take_snapshot(self._snapshot_due[1])
        self._take_results(now)
        if self.ctl.mode == RUNNING:
            why = self._why_stop(now)
            if why:
                self.ctl.stop(now, why)
                self.log(f"stopped: {why}")
        before = self.ctl.mode
        out = self.ctl.step(now)
        if out is not None:
            metadata = {"timestamp": time.time_ns()}
            for side in SIDES:
                self.send(f"position_{side}", qpos_array(out[side]), metadata)
        if self.ctl.mode == RUNNING and out is not None:
            for side in SIDES:
                self.stall.on_target(side, now, out[side])
            if now - self._diag_t >= 1.0:
                self._diag_t = now
                self._sample_stall(now)
        if before != self.ctl.mode or self._mode_seen != self.ctl.mode:
            self._mode_changed()
        if now - self._last_status >= 0.1:
            self._last_status = now
            self._publish_status(now)

    def _sample_stall(self, now: float) -> None:
        if self.ctl.chunk_age_s(now) is None:
            return  # holding for the first chunk, not a stall
        parts = []
        for side in SIDES:
            with self.lock:
                measured = self.state.get(side, (None, 0.0))[0]
            velocity, torque = self.state_extra.get(side, (None, None))
            sample = self.stall.sample(side, now, self.ctl.cmd.get(side), measured, velocity, torque)
            if sample is None:
                continue
            self.diag[side] = sample.as_dict()
            parts.append(f"{side[0].upper()}: {sample.text()}")
        if parts:
            # The terminal/log only, not the UI's log list (once a second would flood it).
            self._print("[dora-openarm-policy] stall check  " + "  |  ".join(parts))

    def _mode_changed(self) -> None:
        self.log(f"mode: {self.ctl.mode}")
        if self.ctl.mode == RUNNING:
            self.stall.reset()
            self.diag = {}
        self._mode_seen = self.ctl.mode
        if self.ctl.mode == STOPPED:
            self._snapshot_due = (time.monotonic() + SNAPSHOT_SETTLE_S, "動作完了後")
        if self.ctl.mode == PASSTHROUGH and self.ctl.after_return:
            then, self.ctl.after_return = self.ctl.after_return, None
            self._clutch(then)
            self.log(f"back with the clutch; sent {then!r} to it")

    def _take_snapshot(self, label: str) -> None:
        self._snapshot_due = None
        with self.shared.lock:
            jpeg = self.shared.head_jpeg
            if jpeg is None:
                return
            self.shared.snapshot_jpeg = jpeg
        seq = (self.snapshot or {}).get("seq", 0) + 1
        self.snapshot = {"seq": seq, "label": label, "time": time.strftime("%H:%M:%S")}
        self.log(f"head_left snapshot for the bbox: {label}")

    def _take_requests(self, now: float) -> None:
        while True:
            try:
                kind, payload = self.shared.requests.get_nowait()
            except queue.Empty:
                return
            if kind == "vad_stats":
                self._on_vad_stats(payload, now)
                continue
            if kind == "speed":
                self.ctl.set_speed(now, payload)
                self.log(f"playback speed {self.ctl.speed * 100:.0f}%")
                continue
            if kind == "bbox":
                with self.lock:
                    self.bbox = payload
                self.log(
                    "bbox cleared"
                    if payload is None
                    else "bbox set: "
                    + ", ".join(f"{v:.0f}" for v in self._bbox_px(payload))
                    + " (1280x720 px)"
                )
                if payload is None and self.ctl.mode == RUNNING:
                    self.ctl.stop(now, "bbox cleared")
                    self.log("stopped: bbox cleared")
            elif payload in ("stop", "voice_stop"):
                if self.ctl.mode == RUNNING or self.ctl.mode == "returning":
                    who = "voice" if payload == "voice_stop" else "UI"
                    self.ctl.stop(now, who)
                    self.log(f"stopped by the {who}")
            elif payload == "start":
                why = self._why_not_start(now)
                if why is None:
                    why = self.ctl.start(now)
                if why is None:
                    self._run_started = now
                    self.log(f"policy started (run {self.ctl.run_id})")
                else:
                    self.log(f"cannot start: {why}")
            elif payload == "proceed":
                if self.ctl.mode == PASSTHROUGH:
                    self._clutch("proceed")
                    self.log("sent 'proceed' to the clutch (the gripper closes, then turns back)")
                else:
                    self.log("'proceed' is only for the intro; the policy has the arms")
            elif payload == "reset":
                why = self.ctl.go_back(now, then="cancel")
                if why is not None:
                    self.log(f"cannot go back: {why}")
                elif self.ctl.mode == PASSTHROUGH and self.ctl.after_return:
                    self._mode_changed()
                else:
                    self.log("easing back to the clutch's pose, then the intro again")
            elif payload == "snapshot":
                self._take_snapshot("手動")
            elif payload == "test":
                if self.policy is None:
                    self.log("test: the policy is not loaded")
                elif self.ctl.mode == RUNNING:
                    self.log("test: the policy is running")
                else:
                    self._test.set()

    def _on_vad_stats(self, stats: dict, now: float) -> None:
        """The browser's speech detection, every 2 s: logged every 10 s, and at once
        when it heard a new onset, so misdetection can be read back from the log."""
        onsets = stats.get("onsets")
        new_onset = onsets is not None and onsets != self._vad_onsets
        self._vad_onsets = onsets
        if not new_onset and now - self._vad_logged < 10.0:
            return
        self._vad_logged = now

        def db(key):
            value = stats.get(key)
            return "-" if not isinstance(value, (int, float)) else f"{value:.0f}"

        self._print(
            "[dora-openarm-policy] voice detection: level p50 "
            f"{db('level_p50')} max {db('level_max')} dB, floor {db('floor')} dB, "
            f"sensitivity {db('sensitivity')} dB, min {db('min_level')} dB, "
            f"speaking={stats.get('speaking')} onsets={onsets} sent={stats.get('sent')} "
            f"silent_frames={stats.get('silent_frames')} tts={stats.get('tts')} rate={stats.get('rate')}"
        )

    def _take_results(self, now: float) -> None:
        while True:
            try:
                item = self.results.get_nowait()
            except queue.Empty:
                return
            if item[0] == "chunk":
                _, chunk, t_obs, run_id, info = item
                if self.ctl.on_chunk(chunk, t_obs, run_id, now):
                    self.infer.update(info)
                    self.infer["count"] += 1
            elif item[0] == "error":
                self.infer["error"] = item[1]
                if self.ctl.mode == RUNNING and item[2] == self.ctl.run_id:
                    self.ctl.stop(now, f"inference failed: {item[1]}")
                    self.log(f"stopped: inference failed: {item[1]}")
            elif item[0] == "test":
                self.infer["test"] = item[1]
                info = item[1]
                self.log(
                    f"test inference: first target {info['gap_rad']} rad from the arm "
                    f"({info['gap_joint']}), chunk moves up to {info['travel_rad']} rad "
                    f"({info['travel_joint']}), {info['latency_ms']} ms"
                )
            elif item[0] == "test_error":
                self.log(f"test: {item[1]}")

    def _why_stop(self, now: float) -> str | None:
        for side in SIDES:
            if self.motor_faults.get(side):
                return f"{side} arm {', '.join(self.motor_faults[side])}"
        for side in SIDES:
            gate = self.gate_state.get(side) or {}
            if gate.get("state") not in (None, "engaged"):
                return f"gate {side}: {gate.get('state')} ({gate.get('reason')})"
        with self.lock:
            cams = dict(self.cams)
            state = dict(self.state)
        for name in CAMERAS.values():
            cam = cams.get(name)
            if cam is None or now - cam.t > self.args.camera_timeout:
                return f"{name} camera stale"
        for side in SIDES:
            if side not in state or now - state[side][1] > self.args.state_timeout:
                return f"{side} arm state stale"
        if self.ctl.chunk_age_s(now) is None and now - self._run_started > 3.0:
            return "no chunk from the policy within 3 s"
        age = self.ctl.chunk_age_s(now)
        if age is not None and age > 1.5:
            return f"the last chunk is {age:.1f} s old"
        return None

    def _why_not_start(self, now: float) -> str | None:
        if self.policy is None:
            return self.policy_error or "the policy is still loading"
        if self.bbox is None:
            return "drag a bbox on head_left first"
        if self.ctl.mode == "returning":
            return "the arms are going back"
        for side in SIDES:
            if self.motor_faults.get(side):
                return f"{side} arm {', '.join(self.motor_faults[side])} (stop and start the arm)"
        for side in SIDES:
            gate = self.gate_state.get(side) or {}
            if gate.get("state") != "engaged":
                return f"gate {side} is {gate.get('state')} ({gate.get('reason')})"
            if self.clutch_state.get(side) in INTRO_STATES:
                return f"the {side} arm's intro is not done ({self.clutch_state.get(side)})"
        with self.lock:
            cams = dict(self.cams)
            state = dict(self.state)
        for name in CAMERAS.values():
            if name not in cams or now - cams[name].t > self.args.camera_timeout:
                return f"no fresh {name} image"
        for side in SIDES:
            if side not in state or now - state[side][1] > self.args.state_timeout:
                return f"no fresh {side} arm state"
        return None

    def _publish_status(self, now: float) -> None:
        with self.lock:
            cams = {n: round(now - c.t, 3) for n, c in self.cams.items()}
            size = None
            if "head_left" in self.cams:
                c = self.cams["head_left"]
                size = [c.width, c.height]
            bbox = list(self.bbox) if self.bbox else None
        status = {
            "mode": self.ctl.mode,
            "reason": self.ctl.reason,
            "run": self.ctl.run_id,
            "bbox": bbox,
            "bbox_px": None if bbox is None else [round(float(v)) for v in self._bbox_px(bbox)],
            "head_size": size,
            "camera_age_s": cams,
            "gate": self.gate_state,
            "motor_faults": self.motor_faults,
            "clutch": self.clutch_state,
            "policy": {"loaded": self.policy is not None, "error": self.policy_error, **self.policy_info},
            "inference": dict(self.infer),
            "execute_steps": self.args.execute_steps,
            "speed": self.ctl.speed,
            "diag": self.diag if self.ctl.mode == RUNNING else {},
            "chunk_left_s": round(self.ctl.chunk_left_s(now), 2),
            "can_start": self._why_not_start(now),
            "snapshot": self.snapshot,
            "log": list(self._log_lines),
        }
        with self.shared.lock:
            status["assistant"] = dict(self.shared.assistant_status)
            status["presets"] = self.presets_status
            self.shared.status = status


_JOINTS = (
    [f"right_j{i}" for i in range(1, 8)]
    + ["right_gripper"]
    + [f"left_j{i}" for i in range(1, 8)]
    + ["left_gripper"]
)


def _joint_name(index: int) -> str:
    return _JOINTS[index]


def _json(value: pa.Array) -> dict | None:
    try:
        obj = json.loads(str(value[0].as_py()))
    except (IndexError, TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def build_parser() -> argparse.ArgumentParser:
    env = os.environ.get
    p = argparse.ArgumentParser(prog="dora-openarm-policy", description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--policy",
        choices=["small_flow", "molmoact2"],
        default=env("POLICY") or "small_flow",
        help="small_flow (training/small_flow) or molmoact2 (training/molmoact2, a merged checkpoint)",
    )
    p.add_argument(
        "--molmoact2-checkpoint",
        default=env("MOLMOACT2_CHECKPOINT")
        or "training/molmoact2/checkpoints/molmoact2-think-openarm-bbox-lora-h200/step2000-merged",
    )
    p.add_argument(
        "--molmoact2-experiments",
        default=env("MOLMOACT2_EXPERIMENTS") or "lib/molmoact2/experiments",
        help="the MolmoAct2 release code (git submodule lib/molmoact2)",
    )
    p.add_argument(
        "--checkpoint", default=env("POLICY_CHECKPOINT", "training/small_flow/runs/flow40k_v3/step20000.pt")
    )
    p.add_argument(
        "--small-flow",
        default=env("SMALL_FLOW_PY", "training/small_flow/small_flow.py"),
        help="the training script whose model and preprocessing are used",
    )
    p.add_argument(
        "--execute-steps",
        type=int,
        default=int(env("EXECUTE_STEPS") or 0) or None,
        help="chunk steps (30 fps) executed before the next inference's chunk takes over "
        "(default 10); keep it plus the inference time under the chunk's 30 steps",
    )
    p.add_argument(
        "--flow-steps", type=int, default=int(env("FLOW_STEPS", "10")), help="Euler steps per sample"
    )
    p.add_argument(
        "--return-speed",
        type=float,
        default=float(env("RETURN_SPEED", "0.3")),
        help="rad/s for the joint that moves most when easing back to the clutch",
    )
    p.add_argument(
        "--blend-s",
        type=float,
        default=float(env("BLEND_S") or 0) or None,
        help="seconds over which a new chunk is cross-faded in (default 0.15, MolmoAct2 0.4); "
        "longer if the gate holds with 'jump' because consecutive chunks disagree",
    )
    p.add_argument(
        "--speed",
        type=float,
        default=float(env("POLICY_SPEED") or 0) or None,
        help="playback speed of the chunks, 0.05..1 (default 1, MolmoAct2 0.5); the UI's slider "
        "changes it while running",
    )
    p.add_argument(
        "--seed",
        default=env("POLICY_SEED") or "0",
        help="the flow's starting noise for every inference, so that consecutive chunks of the "
        "same scene agree; 'random' draws fresh noise each time",
    )
    p.add_argument("--camera-timeout", type=float, default=0.5)
    p.add_argument("--state-timeout", type=float, default=0.25)
    p.add_argument("--host", default=env("POLICY_UI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(env("POLICY_UI_PORT", "8010")))
    # The voice assistant (assistant.py). The key comes from OPENAI_API_KEY, or
    # from the file OPENAI_API_KEY_FILE (default ~/.config/openai/api_key).
    p.add_argument("--presets", default=env("PATCH_PRESETS", "configs/patch_presets.yaml"))
    p.add_argument("--vlm-model", default=env("VLM_MODEL", "gpt-6.2-sol"))
    p.add_argument("--stt-model", default=env("STT_MODEL", "gpt-4o-transcribe"))
    p.add_argument("--tts-model", default=env("TTS_MODEL", "gpt-4o-mini-tts"))
    p.add_argument("--tts-voice", default=env("TTS_VOICE", "alloy"))
    return p


def load_dotenv(path: Path) -> list[str]:
    """KEY=VALUE lines of ``path`` into os.environ where the variable is unset or
    empty (the module passes some as "" so .env can set them). Returns the keys set.

    Comments (#), blank lines, ``export KEY=...`` and quoted values are understood;
    nothing is expanded. A missing file is not an error.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    set_keys = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.replace("_", "").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if not os.environ.get(key):
            os.environ[key] = value
            set_keys.append(key)
    return set_keys


def resolve_defaults(args: argparse.Namespace) -> None:
    """Per-policy defaults. MolmoAct2's consecutive chunks disagree more (its samples
    vary more, and at ~0.5 s per inference a chunk is 16 steps old when it arrives),
    so it cross-fades longer. It re-plans as often as the small flow policy: at 20
    steps the next chunk would arrive after the current one (1 s) has run out."""
    slow = args.policy == "molmoact2"
    if args.execute_steps is None:
        args.execute_steps = 10
    if args.blend_s is None:
        args.blend_s = 0.4 if slow else 0.15
    if args.speed is None:
        args.speed = 0.5 if slow else 1.0
    args.speed = min(max(args.speed, 0.05), 1.0)
    seed = str(args.seed).strip().lower()
    args.seed = None if seed in ("random", "none", "") else int(seed)


def openai_api_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY", "").strip() or None


def make_assistant(args: argparse.Namespace, shared: web.Shared, log):
    """The voice assistant, or None without presets. Without a key it runs disabled
    (the UI says why) so the rest of the node works as before."""
    from dora_openarm_policy.assistant import Assistant, OpenAIBackend
    from dora_openarm_policy.presets import Presets

    try:
        presets = Presets.load(Path(args.presets))
    except (OSError, ValueError, KeyError, TypeError) as error:
        log(f"assistant off: presets {args.presets}: {error}")
        return None, []
    backend, why = None, None
    key = openai_api_key()
    if key is None:
        why = "OPENAI_API_KEY is not set (put it in .env at the repository root)"
    else:
        try:
            backend = OpenAIBackend(key, args.vlm_model, args.stt_model, args.tts_model, args.tts_voice)
        except Exception as error:  # noqa: BLE001 - shown in the UI
            why = f"OpenAI client: {error}"
    log(
        f"assistant: {len(presets.items)} presets from {args.presets}; "
        + (
            f"VLM {args.vlm_model}, STT {args.stt_model}, TTS {args.tts_model}/{args.tts_voice}"
            if backend
            else f"off: {why}"
        )
    )
    return Assistant(shared, presets, backend, log, setup_error=why), presets.as_status()


def main(argv: list[str] | None = None) -> None:
    # Before the parser reads its defaults from the environment. The node runs in
    # the dataflow's directory, the repository root.
    loaded = load_dotenv(Path(os.environ.get("ENV_FILE") or ".env"))
    args = build_parser().parse_args(argv)
    if loaded:
        # Names only: values (the API key) never go to the log.
        print(f"[dora-openarm-policy] .env set {', '.join(sorted(loaded))}", file=sys.stderr, flush=True)
    resolve_defaults(args)
    if not 1 <= args.execute_steps <= 29:
        raise SystemExit("--execute-steps must be 1..29")
    from dora import Node

    shared = web.Shared()
    node = None

    def send(output: str, value: pa.Array, metadata: dict) -> None:
        node.send_output(output, value, metadata)

    policy_node = PolicyNode(args, shared, send)
    # Before the event loop, not beside it: importing torch and loading the
    # checkpoint hold the GIL for seconds, which would stall the IK pass-through
    # (the gate holds with stale_ik) if the loop were already running.
    policy_node.load_policy()
    node = Node()
    server = web.serve(shared, args.host, args.port)
    policy_node.log(
        f"UI on http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    )
    threading.Thread(target=policy_node.infer_loop, name="policy-infer", daemon=True).start()
    assistant, policy_node.presets_status = make_assistant(args, shared, policy_node.log)
    if assistant is not None:
        threading.Thread(target=assistant.run, name="assistant", daemon=True).start()
    try:
        for event in node:
            if not policy_node.on_event(event):
                break
    finally:
        policy_node.done.set()
        if assistant is not None:
            assistant.done.set()
        shared.close()
        server.shutdown()
        policy_node.log("stopped")


if __name__ == "__main__":
    main()
