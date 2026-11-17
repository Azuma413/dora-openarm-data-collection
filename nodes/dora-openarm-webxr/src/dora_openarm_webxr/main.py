# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebXR server node for OpenArm teleoperation.

This dora-rs node talks WebRTC with a VR device such as Meta Quest 3 or
PICO 4. Controller poses arrive on an unreliable data channel and head
camera frames leave on video tracks; :mod:`.webrtc` owns that half.
Each received frame updates the latest controller targets. Dora ticks
drive coordinate conversion, One Euro filtering, rate limiting and hand
pose output. Trigger, joystick, button and head-reference outputs remain
driven by received frames, independently of the pose tick.

The published poses are expressed in the scene's ``arm_origin`` site
frame (chest-level origin between the arms), not in world coordinates.
Downstream IK interprets targets in the same frame. The hand position
is relative to the headset but keeps the world axes, so looking around
does not drag the target with the head.

The headset pose that the hands are made relative to is published as
is on ``pose_reference``, in the WebXR reference space, for consumers
that drive something from head motion such as a neck.

The node serves the front-end itself over HTTPS by default, which
WebXR requires of any page it runs in. Passing ``--offer`` runs it as a
pure WebRTC peer instead, with no HTTP server at all: another service
hosts the page and brokers signaling, and this node needs no
certificate of its own -- WebRTC brings its own.

The Web server and the dora-rs event loop run concurrently in a single
asyncio event loop; the node shuts down when the dora-rs node receives
a ``STOP`` event or ``quit`` on its ``control`` input, or, in WebRTC-only
mode, when the one browser leaves.

openarm-quest-teleop patches (see PATCHES.md): the ``hud`` and
``control`` inputs, the One Euro parameters from the environment, the
pairing token on ``POST /offer`` (``PAIRING_TOKEN``), smoother resets
after input gaps (``SMOOTHER_RESET_GAP``), the latched neck pivot
(``PIVOT_MODE``) and the 3D scene: the ``scene_state`` input, forwarded
to the active headset on the unreliable ``state`` channel **only when the
view actually draws the scene**, and the exported glTF named by
``SCENE_GLB``, served at ``/scene/scene.glb`` and announced -- path, hash
and the model's own body list -- in the configuration message.

Also a line a minute (``_FrameStats``) saying what became of the page's
frames: the gaps between them as percentiles, how many were longer than
the freshness window downstream measures them against (``POSE_GAP_MS``),
and how many frames were dropped for each reason this node can see.

And the headset's own log (``QUEST_LOG``): the page reports what went
wrong in it -- uncaught errors, refused sessions, a lost WebGL context --
over the reliable ``control`` channel when WebRTC is up and over ``POST
/log`` when it is not, and this node appends every record to one JSONL
file, mirroring the errors to its stderr. See :class:`_QuestLog`.
"""

import argparse
import asyncio
import collections
import datetime
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import pathlib
import re
import secrets
import struct
import sys
import time
import zlib

import dora
import numpy as np
import pyarrow as pa
import uvicorn
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from scipy.spatial.transform import Rotation

from . import calibration, video, webrtc
from .smoothing import OneEuroPoseSmoother

args = None
node = None
server = None
webrtc_server = None
_state = None
_running = True
_MAX_LINEAR_SPEED = 1.0
_MAX_ANGULAR_SPEED = 6.0
_POSE_TIMEOUT = 0.0
# One Euro filter parameters, from --one-euro-min-cutoff / --one-euro-beta
# (ONE_EURO_MIN_CUTOFF / ONE_EURO_BETA). The defaults are the values this
# node always used; a higher cutoff trades jitter for less of the 70-80 ms
# steady tracking lag they give.
_ONE_EURO_MIN_CUTOFF = 2.0
_ONE_EURO_BETA = 0.04

# openarm-quest-teleop patch: a hand whose input stopped for longer than
# this many seconds gets its One Euro filter reset rather than suspended,
# from --smoother-reset-gap (SMOOTHER_RESET_GAP). A suspended filter
# keeps the pose from before the gap and slides from it to the hand at up
# to the speed caps once tracking returns, and a clutch that let go during
# the gap would re-anchor on that sliding pose and carry the rest of the
# slide into the arm. The default is the clutch's own freshness window:
# shorter gaps keep the clutch engaged, and a raw step there would pass
# straight into its target, so those still suspend.
_SMOOTHER_RESET_GAP = 0.1

# openarm-quest-teleop patch: what the hand positions are made relative
# to, from --pivot-mode (PIVOT_MODE). "live" (upstream) subtracts the neck
# pivot of every frame, so leaning or stepping moves the target 1:1 with
# the body. "latched" subtracts the pivot latched from the first head pose
# of a session, and again after a gap in head poses longer than
# _SMOOTHER_RESET_GAP (which a recenter always is): a relative clutch only
# reads displacements, and those should be the hand's alone.
_PIVOT_MODES = ("live", "latched")
_PIVOT_MODE = "live"

# openarm-quest-teleop patch: the freshness window downstream measures this
# node's frames against, in seconds, from --pose-gap-ms (POSE_GAP_MS). Nothing
# here behaves differently because of it; it is the threshold the once-a-minute
# pose line counts gaps over, so that the count reads "how many times the clutch
# could have let go of an arm" rather than a number this node chose for itself.
# `oqt run` sets it from the clutch node's own --stale-ms.
_POSE_GAP_S = 0.1

# openarm-quest-teleop patch: the secret POST /offer must carry, from the
# PAIRING_TOKEN environment variable (never a command-line option, which
# any local user could read from the process list). None accepts every
# offer, as upstream did, and says so once.
_PAIRING_TOKEN = None
_warned_unpaired = False


def pairing_token_from_environment(environ) -> str | None:
    """Return the pairing token from PAIRING_TOKEN or PAIRING_TOKEN_FILE.

    Patch 19: a dataflow is a file that gets shared and committed, so the
    token is better named by path than written into the dataflow's `env:`.
    PAIRING_TOKEN wins when both are set. A PAIRING_TOKEN_FILE that is set
    but cannot be read stops the node, rather than silently accepting every
    offer: the operator asked for pairing and must not get none.
    """
    token = environ.get("PAIRING_TOKEN", "").strip()
    if token:
        return token
    path = environ.get("PAIRING_TOKEN_FILE", "").strip()
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as f:
            token = f.read().strip()
    except OSError as error:
        raise SystemExit(
            f"PAIRING_TOKEN_FILE={path!r} cannot be read ({error}). "
            "Run scripts/webxr-setup to create it."
        ) from error
    if not token:
        raise SystemExit(f"PAIRING_TOKEN_FILE={path!r} is empty.")
    return token


# Relative pose to robot workspace mapping.
# We may need to adjust this.
_ROBOT_ROTATION_MATRIX: np.ndarray = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
_ROBOT_ROTATION = Rotation.from_matrix(_ROBOT_ROTATION_MATRIX)

# Relative pose is computed from the viewer position.
# We need to move it to OpenArm position.
#
# Neutral hand position relative to the arm_origin site (chest level).
# Overridden by ``pose: frame_offset`` in the view configuration file.
_FRAME_OFFSET_CELL: np.ndarray = np.array([-0.085, 0, -0.14], dtype=np.float32)

# The client reads the controller from the WebXR target ray space, which is the
# OpenXR aim pose: its -Z points where the controller points, the same
# convention the Unity sender published. So the frame above and the turn below,
# both carried over from that sender, get the pose they were tuned for with no
# fixed conversion in between. The grip pose the client used to send runs its -Z
# along the handle toward the thumb instead, and turning that into a pointing
# pose here meant hardcoding one headset's handle geometry; the aim pose leaves
# every runtime to state its own.
#
# This turn maps the aim frame onto the end effector one, which points the
# gripper along its own -z and opens it across y.
_CONTROLLER_TO_EE: Rotation = Rotation.from_euler("z", 90, degrees=True)

# Eyes to the neck's rotation axis, in the headset's own frame: -Z is forward
# and +Y is up here, so this reads as below and behind the face.
#
# The head does not turn about the headset. The rotation axis is in the neck,
# and the headset rides ahead of it, so a head turn swings the headset along an
# arc. Subtracting the headset position alone therefore still reads that arc as
# the operator translating, and drags the target with it -- 4 to 6 cm at 30
# degrees and 11 cm at 90. Subtracting the pivot instead leaves the reference
# motionless under head rotation while it keeps translating when the operator
# leans or walks, so the workspace follows the body without inheriting the neck.
#
# This is the neck model that 3DOF VR used in reverse: there it synthesised the
# eye translation a headset could not measure, here it removes the one we can.
#
# An estimate, not a measurement, and anatomy varies: override it per operator
# with ``pose: neck_pivot_offset`` in the view configuration file, or measure it
# in a session started with --calibration, by holding the Y button down while
# turning the head. Setting it to [0, 0, 0] restores the plain headset
# subtraction, which is also how the two can be compared.
_NECK_PIVOT_OFFSET: np.ndarray = np.array([0.0, -0.075, 0.080], dtype=np.float32)

# Whether the Y button measures the neck pivot at all, from --calibration.
# Off unless asked for: the button is published as an output in its own right
# and a dataflow may have wired it to something the operator holds down, so
# outside a calibration session it stays an ordinary button and the hands never
# stop. Measuring is a thing the operator sets out to do, not something a press
# can start by accident.
_CALIBRATION_ENABLED: bool = False

# The controller buttons published as outputs, and the names --quit-button
# accepts.
_BUTTONS = ("a", "b", "x", "y")

# The buttons that shut the node down when pressed, from --quit-button.
# Naming any button also publishes ``"quit"`` on the ``command`` output
# when the node exits -- however it exits, not just by the button -- which
# is what a dora-openarm-quitter node listens for, so the rest of the
# dataflow ends with this node even when an error ends it.
# Empty unless asked for, for the same reason --calibration is off by
# default: the buttons are outputs in their own right, and quitting is a
# thing the operator sets out to do, not something a press wired to
# another purpose can do by accident.
_QUIT_BUTTONS: tuple = ()

# Where an accepted run is written and where a saved one is read back from, at
# startup, from --neck-pivot-file. A fit lives as long as the session
# otherwise, and the whole point of measuring an operator is keeping the number
# they measured.
_NECK_PIVOT_FILE = None

# The ICE servers every peer is built with, from --ice-servers. None means
# webrtc's default public STUN server; the signaling service passes its own
# list -- short-lived TURN credentials included -- when a direct path cannot
# be counted on.
_ICE_SERVERS = None

# openarm-quest-teleop patch: the exported glTF of the scene the sim is
# running, from --scene-glb (SCENE_GLB), and what the configuration message
# says about it. The file is served at _SCENE_URL and nothing else in its
# directory is: a run directory also holds the dataflow, and that carries the
# pairing token. None (unset, missing or unreadable) means the page is told
# nothing about a scene and stays on the camera panel it has today.
_SCENE_GLB = None
_SCENE = None
_SCENE_URL = "/scene/scene.glb"

# The same export, gzipped once at startup, as {"hash": ..., "body": ...}, or
# None when the file did not shrink. A glTF binary is mostly float32 vertex and
# index data, which deflate halves: 4,716,792 -> 2,022,637 B for the demo cell
# and 3,930,400 -> 1,639,480 B for the compress one, both measured through a
# real browser (tests/web/webxr_page.test.mjs reads the page's own Resource
# Timing). Over the loopback the whole fetch is milliseconds and none of this
# matters; over the headset's Wi-Fi it is most of the first-load cost, and the
# operator waits through it before the robot appears.
#
# Compressed once rather than per request or by a middleware: the file cannot
# change under a running node (it is read at startup like the view
# configuration), and this node shares its host with the simulator, so the
# per-request CPU would come out of the loop that answers the operator's hands.
# Kept in memory rather than written next to the export, which lives in a run
# directory this node only ever reads. Keyed by the hash, so a scene swapped in
# under the endpoint can never be served the compression of the previous one.
_SCENE_GZIP = None
# Level 6 is zlib's default and where this file stops paying. The 4.72 MB demo
# cell, median of 9 runs on this host: level 1 is 2,104,492 B in 40.9 ms, level
# 6 is 2,022,637 B in 66.5 ms, level 9 is 2,022,092 B in 88.1 ms. So 9 buys 545
# bytes (0.03 %) for a third more CPU, and 1 gives back 4 % of the size to save
# 26 ms of a cost paid once at startup, which no headset ever waits for.
_SCENE_GZIP_LEVEL = 6

# A glTF binary starts with a 12-byte header and a JSON chunk; the exporter
# puts the model's body list and nq in that chunk's scene extras, and they are
# what the page checks an incoming scene state against. Only the header and the
# chunk length are read from the file here, never the whole 11 MB of meshes.
_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_JSON_CHUNK = 0x4E4F534A  # "JSON"
# The largest JSON chunk worth reading. The compress cell's is about 200 kB
# (222 bodies with their joints); a file whose chunk is far bigger than that is
# not one of ours.
_GLB_JSON_LIMIT = 8 << 20

# How many scene states may be forwarded between two lines about the stream, so
# it says how it is doing once a minute at 72 Hz rather than per frame.
_STATE_REPORT_EVERY = 4096
_state_forwarded = 0
_state_reported = 0
_state_reported_at = 0.0
_state_bytes_reported = 0

# Whether the view this node was started with draws the 3D scene at all
# (openarm-quest-teleop patch, set in main() from video.draws_scene). False
# leaves the ``state`` data channel unopened and drops every scene_state input
# where it arrives: the page would register no handler for it, and on the
# half-duplex link a headset sits on those bytes are taken from the hand poses
# coming the other way. Counted, so the log says how much was refused.
_FORWARD_SCENE_STATE = True
_scene_state_unwanted = 0
_warned_scene_state_unwanted = False

# The longest scene state this node will forward. The largest scene measured is
# 1.4 kB of JSON (the compress cell, 191 joint values); anything past this is a
# sender that is not the sim node, and forwarding it would only stall the
# channel.
_SCENE_STATE_MAX_BYTES = 1 << 20
_warned_scene_state = False

# openarm-quest-teleop patch: how often this node says what happened to the
# page's frames, in seconds. One line a minute, on the pose tick so that it is
# printed whether or not any frame arrived -- a headset that stopped sending is
# exactly the case worth seeing.
_POSE_REPORT_S = 60.0


def _percentile(ordered, fraction):
    """Return the value at ``fraction`` of a sorted, non-empty list."""
    index = round(fraction * (len(ordered) - 1))
    return ordered[min(max(index, 0), len(ordered) - 1)]


class _FrameStats:
    """What became of the page's frames in one reporting window.

    openarm-quest-teleop patch, and the one line that separates every way
    teleoperation can stop. A hand stops following when its pose or its grip
    goes stale downstream, and from the node's own log that used to be
    indistinguishable from a busy sim, a slow page or a lossy radio.

    What is measured, and what each reading means:

    * the gaps between ACCEPTED frames, as percentiles plus the maximum. A
      healthy median with a fat maximum is bursty loss on the link (a Wi-Fi
      retry exhaustion takes out seven consecutive frames at 72 Hz, which is
      the whole freshness window, while leaving the median at 13.9 ms). A
      median that has risen to 30-50 ms is the page's own animation frame
      falling behind, and no percentile is a substitute for the maximum:
      every candidate here trips the clutch on a single interval, never on a
      rate.
    * how many of those gaps were longer than the downstream freshness window
      (``--pose-gap-ms``, which ``oqt run`` sets from the clutch's own
      --stale-ms), which is the count of times an arm could have let go.
    * frames refused because their ``sequence`` was not newer: the unordered
      channel delivering one late. That path is silent otherwise, and it
      throws away the frame's grip along with its pose.
    * frames carrying no head pose, without which no hand can be published.
    * frames in which a hand was left out because the page saw its controller
      as untracked (``ar.js``: an emulated pose has its pose, trigger and grip
      withheld). That is one hand going stale on its own -- a controller out
      of the cameras' sight -- and it is a different fault from a link gap,
      which takes both hands at once because one frame carries both.
    * how many times the cached frame aged past ``--pose-timeout``, i.e. how
      often the node itself let go of the hands.

    A silence because the page deliberately sent nothing -- the system menu
    open, the headset off (``session.visibilityState``) -- is NOT separable
    here: it reaches this node as an ordinary gap. Only the page can tell "I
    never sent it" from "I sent it and it was lost".
    """

    # A window is a minute; a Quest sends 72 frames a second, so 4,320 floats.
    # The cap is only there so that a sender gone mad cannot grow the list
    # without bound -- the count and the maximum stay right either way.
    _MAX_GAPS = 1 << 16

    def __init__(self, now=None):
        """Start an empty window at ``now`` (default: now) with no frame seen.

        The default matters: a window that started at 0.0 on a perf_counter
        clock is already a million seconds old, so the very first frame would
        report a window of the machine's uptime and a rate of 0.0 Hz.
        """
        self.ever = False
        self.last_at = None
        self.start(time.perf_counter() if now is None else now)

    def start(self, now):
        """Begin a fresh window; the frame clock carries on across it."""
        self.started_at = now
        self.gaps_ms = []
        self.accepted = 0
        self.out_of_sequence = 0
        self.no_reference = 0
        self.untracked = {"right": 0, "left": 0}
        self.timeouts = 0

    def out_of_order(self):
        """Note a frame refused because its sequence was not newer."""
        self.out_of_sequence += 1

    def broke(self):
        """Forget the frame clock (a session started or ended).

        Without this the first frame of a session would be timed against the
        last frame of the previous one, and every reconnect would report a gap
        the size of the pause between headsets.
        """
        self.last_at = None

    def accept(self, response, now):
        """Note an accepted frame: its gap, and which hands it left out."""
        if self.last_at is not None and len(self.gaps_ms) < self._MAX_GAPS:
            self.gaps_ms.append((now - self.last_at) * 1e3)
        self.last_at = now
        self.accepted += 1
        self.ever = True
        if not response.get("pose_reference"):
            self.no_reference += 1
            return
        for side in ("right", "left"):
            if (
                not response.get(f"pose_{side}")
                or response.get(f"trigger_{side}") is None
            ):
                self.untracked[side] += 1

    def timed_out(self):
        """Note the cached frame being dropped for age (--pose-timeout)."""
        self.timeouts += 1

    def due(self, now):
        """Whether a line is owed: a full window, and a headset worth reporting."""
        return now - self.started_at >= _POSE_REPORT_S and (self.ever or self.accepted)

    def line(self, now, gap_s):
        """Return the one line, counting gaps longer than ``gap_s`` as lost arms."""
        elapsed = max(now - self.started_at, 1e-9)
        head = f"poses: {self.accepted} in {elapsed:.1f} s"
        if self.accepted:
            head += f" ({self.accepted / elapsed:.1f} Hz)"
        parts = [head]
        if self.gaps_ms:
            ordered = sorted(self.gaps_ms)
            over = sum(1 for gap in ordered if gap > gap_s * 1e3)
            parts.append(
                f"gap p50 {_percentile(ordered, 0.5):.1f} "
                f"p90 {_percentile(ordered, 0.9):.1f} "
                f"p99 {_percentile(ordered, 0.99):.1f} "
                f"max {ordered[-1]:.0f} ms, "
                f"{over} over {gap_s * 1e3:g} ms"
            )
        dropped = []
        if self.out_of_sequence:
            dropped.append(f"{self.out_of_sequence} out of sequence")
        if self.no_reference:
            dropped.append(f"{self.no_reference} with no head pose")
        for side in ("right", "left"):
            if self.untracked[side]:
                dropped.append(f"{self.untracked[side]} {side} untracked")
        if self.timeouts:
            dropped.append(f"{self.timeouts} timed out")
        parts.append("dropped " + (", ".join(dropped) if dropped else "none"))
        return "; ".join(parts)


_frames = _FrameStats()


# How many headset poses a run may hold. A button left held down stops growing
# here instead of the process, and at the headset's display rate this is some
# twenty seconds, well past any deliberate shake.
_CALIBRATION_CAPACITY = 2000

# Fewer poses than this and the run is too thin to fit from. At the headset's
# display rate this is a second or so of holding, so it catches the operator
# who lets the button go before turning their head at all.
_CALIBRATION_MIN_SAMPLES = 100

# A run has to turn the head far enough about every axis for the fit to see
# all three offset components. 0.02 is about a 20 degree sweep, which a
# deliberate shake passes twice over.
_CALIBRATION_MIN_OBSERVABILITY = 0.02

# The headset's own axes, in the order the offset components come in.
_OFFSET_AXIS_NAMES = ("lateral", "vertical", "fore-aft")

# The head motion that pins each of those components. A rotation cannot see
# the offset along the axis it turns about, so what is missing is always a
# turn about a different one: only yawing hides the vertical offset, and
# only nodding hides the lateral one.
_PINNING_MOTION = ("side to side", "up and down", "side to side")

# How far the fitted pivot may still wander over the run, in meters. Body
# motion lands here, and so does the model error: a neck yaws and nods about
# joints a few centimeters apart rather than the one point fitted here, so
# some residual is the anatomy rather than the operator. 20 mm leaves the
# offset good to about a centimeter, against the 11 cm error it removes.
_CALIBRATION_MAX_RESIDUAL = 0.020

# Where a neck can be, relative to the eyes, in meters: on the midline, and
# below and behind them. A run that satisfies everything above can still land
# somewhere a body does not go, and this is the last thing between that and
# the arm following it.
_CALIBRATION_MAX_LATERAL = 0.05
_CALIBRATION_MAX_VERTICAL = 0.20
_CALIBRATION_MAX_FORE_AFT = 0.20


class _PivotCalibration:
    """Collects headset poses while the operator holds the Y button down.

    The button state arrives once per frame rather than as a press and a
    release event, so the edges are found here. Reading it that way is
    also the failsafe: a controller that falls asleep mid-run stops
    reporting the button at all, the caller reads that as not pressed,
    and the run ends instead of leaving the hands stopped for good.

    A node started without ``--calibration`` holds a disabled one, which
    keeps no poses and never stops the hands: the Y button reaches its
    output and nothing else happens.
    """

    def __init__(self, enabled=False, capacity=_CALIBRATION_CAPACITY):
        self._enabled = enabled
        self._running = False
        self._samples = collections.deque(maxlen=capacity)

    @property
    def collecting(self):
        """Whether a run is under way, so poses belong to it."""
        return self._running

    def update(self, pressed):
        """Take the button state for a frame, returning a finished run.

        The press is the run: it starts the frame the button goes down
        and is fitted the frame it comes up. Nothing waits for a hold,
        because only a session started with ``--calibration`` reaches
        here, and in one the operator pressing Y wants exactly this. The
        headset says so on its own display, so the hands stopping is no
        longer the only signal the operator gets.

        Args:
          pressed: whether the Y button is down this frame.

        Returns:
          The run's poses as ``(N, 7)`` rows of ``[x, y, z, qx, qy, qz,
          qw]``, which is the quaternion order ``Rotation.from_quat``
          reads, or None if this frame ended no run worth fitting.

        """
        if not self._enabled:
            return None

        if pressed:
            if not self._running:
                self._running = True
                self._samples.clear()
            return None

        running, self._running = self._running, False
        if not running or not self._samples:
            return None
        return np.array(self._samples, dtype=np.float64)

    def add(self, reference):
        """Keep a headset pose if a run is under way, otherwise drop it."""
        if not self.collecting:
            return
        self._samples.append(
            [
                reference["x"],
                reference["y"],
                reference["z"],
                reference["qx"],
                reference["qy"],
                reference["qz"],
                reference["qw"],
            ]
        )


def _apply_pivot_calibration(samples):
    """Fit the neck pivot from a run, say what came of it, and use it.

    The fitted offset only replaces the one in use if it passes every
    check, so a run that went wrong costs the operator the shake and
    nothing else.

    Returns what to tell the headset about the run, since the operator is
    wearing one and cannot read the node's output from inside it.
    """
    offset, diagnostics = calibration.fit_pivot_offset(
        samples[:, :3], Rotation.from_quat(samples[:, 3:])
    )
    reason = _check_pivot_offset(offset, diagnostics)
    if reason is not None:
        print(f"neck pivot calibration rejected: {reason}", flush=True)
        return {"type": "calibration-result", "accepted": False, "reason": reason}

    global _NECK_PIVOT_OFFSET
    _NECK_PIVOT_OFFSET = offset.astype(np.float32)
    saved_to = _save_neck_pivot_offset(offset)

    formatted = ", ".join(f"{component:.3f}" for component in offset)
    kept = (
        f"  Written to {saved_to}, which this node reads at startup."
        if saved_to is not None
        else "  Kept for this session only; --neck-pivot-file writes it down."
    )
    print(
        f"neck pivot calibration applied from {diagnostics['samples']} poses: "
        f"the pivot held to {diagnostics['residual_rms'] * 1000:.1f} mm while "
        f"the headset moved {diagnostics['headset_rms'] * 1000:.1f} mm.\n"
        f"  neck_pivot_offset: [{formatted}]\n" + kept,
        flush=True,
    )
    return {
        "type": "calibration-result",
        "accepted": True,
        "offset": [float(component) for component in offset],
        "samples": int(diagnostics["samples"]),
        "residual_mm": float(diagnostics["residual_rms"]) * 1000.0,
        "headset_mm": float(diagnostics["headset_rms"]) * 1000.0,
        "saved_to": saved_to,
    }


def _save_neck_pivot_offset(offset):
    """Write an accepted offset where the next run reads it.

    The document nests the offset the way the view configuration file
    nests it, so the measurement can be pasted there as it stands.

    Returns the path written, or None if there was nowhere to write it or
    the write failed. A measurement that cannot be saved is still worth
    having for the session, so a failure is reported and nothing else.
    """
    if _NECK_PIVOT_FILE is None:
        return None
    document = {
        "pose": {
            "neck_pivot_offset": [round(float(component), 4) for component in offset]
        }
    }
    try:
        with open(_NECK_PIVOT_FILE, "w", encoding="utf-8") as output:
            output.write("# Measured by dora-openarm-webxr --calibration.\n")
            yaml.safe_dump(document, output)
    except OSError as error:
        print(f"cannot write {_NECK_PIVOT_FILE}: {error}", flush=True)
        return None
    return str(_NECK_PIVOT_FILE)


def _read_neck_pivot_offset(path):
    """Read a measured offset back, or None if there is not one to read."""
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as input:
            document = yaml.safe_load(input)
    except FileNotFoundError:
        # Nothing measured yet, which is every first run.
        return None
    except (OSError, yaml.YAMLError) as error:
        print(f"cannot read {path}: {error}", flush=True)
        return None
    pose = document.get("pose") if isinstance(document, dict) else None
    offset = pose.get("neck_pivot_offset") if isinstance(pose, dict) else None
    if offset is None:
        return None
    return np.array(offset, dtype=np.float32).reshape(3)


def _configure_neck_pivot(pose_configuration, path):
    """Settle which neck pivot offset the session starts with.

    The built-in estimate is the floor, the view configuration file
    overrides it, and a measured one overrides that: of the three it is
    the only one that came from the operator wearing the headset. Said
    out loud rather than applied quietly, because a file left behind by
    someone else's head would otherwise be invisible.
    """
    global _NECK_PIVOT_OFFSET
    configured = pose_configuration.get("neck_pivot_offset")
    if configured is not None:
        _NECK_PIVOT_OFFSET = np.array(configured, dtype=np.float32).reshape(3)

    measured = _read_neck_pivot_offset(path)
    if measured is not None:
        _NECK_PIVOT_OFFSET = measured
        formatted = ", ".join(f"{component:.3f}" for component in measured)
        print(f"neck pivot offset read from {path}: [{formatted}]", flush=True)


def _check_pivot_offset(offset, diagnostics):
    """Why a fitted neck pivot offset is not worth applying, or None.

    The fit itself reports what the run could and could not see; the
    thresholds live here, next to the operator who has to be told what to
    do differently.
    """
    samples = diagnostics["samples"]
    if samples < _CALIBRATION_MIN_SAMPLES:
        return (
            f"only {samples} headset poses came in; "
            "hold Y down for the whole head turn, not just a tap"
        )

    observability = diagnostics["observability"]
    if observability[0] < _CALIBRATION_MIN_OBSERVABILITY:
        axis = int(np.argmax(np.abs(diagnostics["observability_axes"][0])))
        return (
            f"the head did not turn enough to see the {_OFFSET_AXIS_NAMES[axis]} "
            f"offset; turn it {_PINNING_MOTION[axis]} as well"
        )

    residual = diagnostics["residual_rms"]
    if residual > _CALIBRATION_MAX_RESIDUAL:
        return (
            f"the pivot still moved {residual * 1000:.0f} mm over the run; "
            "hold the body still and turn only the head"
        )

    lateral, vertical, fore_aft = offset
    if (
        abs(lateral) > _CALIBRATION_MAX_LATERAL
        or not -_CALIBRATION_MAX_VERTICAL <= vertical <= 0.0
        or not 0.0 <= fore_aft <= _CALIBRATION_MAX_FORE_AFT
    ):
        return (
            f"the fitted offset [{lateral:.3f}, {vertical:.3f}, {fore_aft:.3f}] "
            "is not where a neck is: it belongs on the midline, below the eyes "
            "and behind them"
        )
    return None


app = FastAPI()


def _map_trigger_to_gripper(trigger: float, side: str) -> float:
    """Map a trigger value (0.0–1.0) to a calibrated gripper angle in radians."""
    trigger = float(np.clip(trigger, 0.0, 1.0))
    if side == "right":
        open_deg, closed_deg = -45.0, 10.0
    elif side == "left":
        open_deg, closed_deg = 45.0, -10.0
    else:
        raise ValueError(f"Unsupported gripper side: {side!r}")
    return float(np.deg2rad(open_deg + trigger * (closed_deg - open_deg)))


def _neck_pivot(reference):
    """Return the neck pivot of a viewer pose, in the reference space.

    The offset is turned into world axes, so "behind the face" follows
    where the head faces.
    """
    reference_rotation = Rotation.from_quat(
        [reference["qx"], reference["qy"], reference["qz"], reference["qw"]]
    )
    return np.array(
        [reference["x"], reference["y"], reference["z"]], dtype=np.float32
    ) + reference_rotation.apply(_NECK_PIVOT_OFFSET)


def _adjust_pose(pose, reference, smoother, smoother_time, pivot=None):
    """Convert WebXR style pose to our style.

    ``pose`` and ``reference`` (the viewer pose) are in the same
    world-fixed reference space. Only the position is made relative to
    the viewer, by subtracting it in the world axes. The viewer
    rotation is never applied: turning the head must not move the
    target. The controller orientation is passed through as its world
    orientation for the same reason.

    What is subtracted is the neck pivot rather than the headset itself,
    since the headset orbits that pivot as the head turns and would
    otherwise carry the arc into the target. The viewer rotation is used
    to place the pivot, which is not the same as applying it to the
    hand: it only says which way the operator is facing, so the point
    behind their face can be found.

    ``pivot`` (openarm-quest-teleop patch) is a pivot position latched
    earlier in the session, subtracted instead of this reference's own
    pivot; None keeps the live pivot.

    WebXR style:
      * right-handed
      * {
          x: X, (meter)
          y: Y, (meter)
          z: Z, (meter)
          qx: QX, (quaternion)
          qy: QY, (quaternion)
          qz: QZ, (quaternion)
          qw: QW, (quaternion)
        }

    Our style:
      * right-handed
      * [x, y, z, qw, qx, qy, qz]
    """
    if pivot is None:
        pivot = _neck_pivot(reference)
    position = (
        np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float32) - pivot
    ).astype(np.float32)
    rotation = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])

    position = _ROBOT_ROTATION.apply(position) + _FRAME_OFFSET_CELL
    rotation = _ROBOT_ROTATION * rotation * _CONTROLLER_TO_EE
    quaternion = rotation.as_quat()

    adjusted_pose = np.array(
        [
            position[0],  # x
            position[1],  # y
            position[2],  # z
            quaternion[3],  # qw
            quaternion[0],  # qx
            quaternion[1],  # qy
            quaternion[2],  # qz
        ],
        dtype=np.float32,
    )
    return pa.array(smoother.smooth(smoother_time, adjusted_pose))


_POSE_STRUCT_TYPE = pa.struct({"pose": pa.list_(pa.float32())})


def _build_pose_output(pose: np.ndarray) -> pa.Array:
    """Wrap a pose array as a length-1 StructArray: [{"pose": [...]}]."""
    return pa.array([{"pose": pose}], type=_POSE_STRUCT_TYPE)


def _build_head_pose_output(pose: dict) -> pa.Array:
    """Wrap the headset pose, in our style: [x, y, z, qw, qx, qy, qz].

    Unrotated, unlike the hands. `_adjust_pose` applies `_ROBOT_ROTATION` and a
    z+90 fix to put a controller where the arm expects its end effector; both
    are arm conventions and neither means anything on a head. Consumers map the
    WebXR frame (x right, y up, -z forward) into their own body frame, so the
    rig's wiring stays in the consumer's config rather than baked in here.

    Not made relative either: the hand positions are relative to this pose, so
    subtracting it from itself would leave nothing to read the head from.

    Not smoothed either: the One Euro smoothers are per-hand and stateful, and
    a neck has its own rate limiting downstream.
    """
    head_pose = np.array(
        [
            pose["x"],
            pose["y"],
            pose["z"],
            pose["qw"],
            pose["qx"],
            pose["qy"],
            pose["qz"],
        ],
        dtype=np.float32,
    )
    return _build_pose_output(head_pose)


class _ConnectionState:
    """Per-session state carried across the frame messages.

    The smoothers and the pivot calibration are stateful, and a new
    session must not inherit the last one's history, so ``session-start``
    replaces this wholesale rather than clearing it piece by piece.
    """

    def __init__(self):
        """Start with fresh smoothers and no frame seen yet."""
        self.smoothers = {
            side: OneEuroPoseSmoother(
                min_cutoff=_ONE_EURO_MIN_CUTOFF,
                beta=_ONE_EURO_BETA,
                d_cutoff=1.5,
                max_linear_speed=_MAX_LINEAR_SPEED,
                max_angular_speed=_MAX_ANGULAR_SPEED,
            )
            for side in ("right", "left")
        }
        self.pivot_calibration = _PivotCalibration(enabled=_CALIBRATION_ENABLED)
        self.last_sequence = -1
        self.latest_frame = None
        self.received_at = 0.0
        # openarm-quest-teleop patch, all on the perf_counter clock of
        # received_at. When a frame last carried each hand's pose and
        # trigger together with a head pose, which is what a published
        # hand needs; and which of those receive times each hand was last
        # published from, so a gap in the input shows up even when every
        # tick in between kept publishing a stale frame.
        self.hand_received_at = {"right": None, "left": None}
        self.hand_published_from = {"right": None, "left": None}
        # When a frame last carried a head pose, and the pivot latched from
        # one (PIVOT_MODE=latched only).
        self.reference_received_at = None
        self.latched_pivot = None


def _process_frame(response, state):
    """Cache raw targets and publish receive-driven outputs for one frame.

    Frames ride an unordered channel that never retransmits, so one can
    arrive after a newer one has already been published. Publishing it
    would drag the arms back to a pose the operator has left, so a frame
    whose ``sequence`` is not newer than the newest processed one is
    dropped. A frame without ``sequence`` is always processed.
    """
    sequence = response.get("sequence")
    if sequence is not None:
        if sequence <= state.last_sequence:
            # openarm-quest-teleop patch: counted, because this drops the
            # whole frame -- grip and buttons with it -- and used to be
            # invisible. The once-a-minute line says how often it happens.
            _frames.out_of_order()
            return
        state.last_sequence = sequence
    metadata = {"timestamp": time.time_ns()}
    received_at = time.perf_counter()
    _frames.accept(response, received_at)
    # openarm-quest-teleop patch: a frame that ends a silence longer than the
    # pose timeout resets the filters, as the tick that cleared the frame
    # would have, even if no tick ran between the timeout and this frame.
    # Downstream freshness checks (the clutch's) time gaps sample to sample,
    # so the reset must not depend on where a tick happened to fall.
    if _POSE_TIMEOUT > 0.0 and received_at - state.received_at > _POSE_TIMEOUT:
        for smoother in state.smoothers.values():
            smoother.reset()
    state.latest_frame = response
    state.received_at = received_at
    # An absent button is a released one, so a controller that falls
    # asleep mid-run cannot leave the hands stopped. The client only
    # sends the buttons on the profiles it knows, and on those it sends
    # them every frame.
    samples = state.pivot_calibration.update(bool(response.get("button_y")))
    if samples is not None:
        # Back to the headset as well as the node's output: the operator
        # cannot see the output while wearing one.
        webrtc_server.send_control(_apply_pivot_calibration(samples))
    node.send_output(
        "vr_receive_times",
        pa.array([metadata["timestamp"]], type=pa.int64()),
        metadata,
    )
    reference = response.get("pose_reference")
    if reference:
        node.send_output(
            "pose_reference",
            _build_head_pose_output(reference),
            metadata,
        )
        state.pivot_calibration.add(reference)
        _track_input(response, reference, state)
    for button in _BUTTONS:
        name = f"button_{button}"
        if name in response:
            pressed = bool(response[name])
            node.send_output(
                name,
                pa.array([pressed], type=pa.bool_()),
                metadata,
            )
            if pressed and button in _QUIT_BUTTONS:
                _stop()
    for side in ["right", "left"]:
        trigger = f"trigger_{side}"
        if trigger in response:
            node.send_output(
                trigger,
                pa.array([response[trigger]], type=pa.float32()),
                metadata,
            )
        grip = f"grip_{side}"
        if grip in response:
            node.send_output(
                grip,
                pa.array([response[grip]], type=pa.float32()),
                metadata,
            )
        joystick = f"joystick_{side}"
        if joystick in response:
            axes = response[joystick]
            # The xr-standard mapping reserves the first axis pair for
            # the touchpad and the second for the thumbstick, so the
            # stick is axes[2:4] wherever the controller has one; a
            # device with only a touchpad reports that pair alone and it
            # takes its place. The y sign is flipped to keep the
            # convention the Unity sender published, which is the one the
            # downstream nodes were written against.
            x, y = (axes[2], axes[3]) if len(axes) >= 4 else axes[:2]
            y = -y
            node.send_output(
                f"joystick_x_{side}",
                pa.array([x], type=pa.float32()),
                metadata,
            )
            node.send_output(
                f"joystick_y_{side}",
                pa.array([y], type=pa.float32()),
                metadata,
            )


def _track_input(response, reference, state):
    """Note which hands a frame with a head pose carries, and latch the pivot.

    openarm-quest-teleop patch. Hands are only published with a head pose,
    so a gap in head poses longer than the reset gap is a gap at least
    that long in every hand's poses, or in its grips when the frames
    stopped altogether: the clutch has let go of both arms by the time the
    pivot moves, so re-latching cannot drag an engaged arm. A recenter is
    one such gap, since the page pauses its frames for 300 ms after the
    local space resets.
    """
    now = state.received_at
    for side in ("right", "left"):
        if response.get(f"pose_{side}") and response.get(f"trigger_{side}") is not None:
            state.hand_received_at[side] = now
    if _PIVOT_MODE == "latched" and (
        state.latched_pivot is None
        or state.reference_received_at is None
        or now - state.reference_received_at > _SMOOTHER_RESET_GAP
    ):
        state.latched_pivot = _neck_pivot(reference)
    state.reference_received_at = now


def _input_gap(state, side, now):
    """Whether a hand's input stopped for longer than the reset gap."""
    received_at = state.hand_received_at[side]
    return received_at is None or now - received_at > _SMOOTHER_RESET_GAP


def _report_frames(now):
    """Print the once-a-minute pose line, if one is due, and start a window."""
    if _frames.due(now):
        print(_frames.line(now, _POSE_GAP_S), flush=True)
        _frames.start(now)


def _publish_poses(state):
    """Filter the latest raw targets and publish hand poses once per tick."""
    now = time.perf_counter()
    # On the tick rather than on a frame, so the line still arrives -- and
    # says 0 frames -- when the headset is what stopped.
    _report_frames(now)
    timed_out = _POSE_TIMEOUT > 0.0 and now - state.received_at > _POSE_TIMEOUT
    if timed_out:
        if state.latest_frame is not None:
            # The transition only: every later tick of the same silence is
            # the same dropout, and counting them would report the tick rate.
            _frames.timed_out()
        state.latest_frame = None
    response = state.latest_frame or {}
    reference = response.get("pose_reference")
    metadata = {"timestamp": time.time_ns()}
    for side, smoother in state.smoothers.items():
        pose = response.get(f"pose_{side}")
        trigger = response.get(f"trigger_{side}")
        if (
            not pose
            or not reference
            or trigger is None
            or state.pivot_calibration.collecting
        ):
            # openarm-quest-teleop patch: only a short dropout keeps the
            # filtered pose to resume from. After a long one -- or once
            # the pose timeout has let go of the frame -- the first pose
            # back must be the hand itself, not a slide toward it.
            if timed_out or _input_gap(state, side, now):
                smoother.reset()
            else:
                smoother.suspend(now)
            continue
        # openarm-quest-teleop patch: the same, for a gap that no tick saw
        # as missing -- frames that stopped while ticks kept publishing
        # the last one, or ticks that stopped themselves.
        received_at = state.hand_received_at[side]
        published_from = state.hand_published_from[side]
        if (
            received_at is not None
            and published_from is not None
            and received_at - published_from > _SMOOTHER_RESET_GAP
        ):
            smoother.reset()
        state.hand_published_from[side] = received_at
        adjusted_pose = _adjust_pose(
            pose, reference, smoother, now, pivot=state.latched_pivot
        )
        gripper_angle = _map_trigger_to_gripper(trigger, side)
        node.send_output(
            f"pose_{side}",
            _build_pose_output(np.concatenate([adjusted_pose, [gripper_angle]])),
            metadata,
        )


def _on_frame(payload):
    """Hand one frame message from the transport to the pose pipeline."""
    _process_frame(payload, _state)


def _on_session_start():
    """Begin a session: fresh smoothers, fresh frame numbering."""
    global _state
    _state = _ConnectionState()
    # The frame counters carry on across sessions -- a minute is a minute --
    # but the gap clock does not: the first frame of this session must not be
    # timed against the last frame of the previous headset's.
    _frames.broke()
    node.send_output("status", pa.array(["ready"]), {"timestamp": time.time_ns()})


def _on_session_end():
    """Invalidate the active session's cached targets.

    The filters are reset rather than suspended (openarm-quest-teleop
    patch): whatever comes next starts from the hand, not from where this
    session left off.
    """
    _state.latest_frame = None
    _frames.broke()
    for smoother in _state.smoothers.values():
        smoother.reset()


# The HUD is a few centimetres of text in front of the operator's eyes, so
# a message is cut to what the panel can show rather than drawn off its
# edge. Longer text is a bug in whoever sent it, not something to render.
_HUD_MAX_LINES = 5
_HUD_MAX_LINE_LENGTH = 80

# Which controllers a HUD message may buzz.
_HAPTIC_TARGETS = ("right", "left", "both")

# The ``control`` input command that ends the node, the same word the
# rest of the dataflow's nodes stop on.
_QUIT = "quit"


def _first_string(value):
    """Return the first element of a utf8 input as a str, or None.

    Accepts anything with ``to_pylist`` (the Arrow array dora delivers)
    or a plain str, so a sender that wraps its string differently is not
    silently ignored.
    """
    if isinstance(value, str):
        return value
    try:
        items = value.to_pylist()
    except AttributeError:
        return None
    if not items:
        return None
    item = items[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8", errors="replace")
    return item if isinstance(item, str) else None


def _parse_hud(value):
    """Turn a ``hud`` input into the control message the page draws.

    The input is a JSON object ``{"lines": [...], "recording": bool,
    "haptic": "right" | "left" | "both" | null}``. Every message is the
    whole HUD: a missing ``lines`` clears the text rather than keeping
    the last one, so the page never shows a status nobody is still
    asserting.

    Returns the message, or None (and says why) when the input is not a
    JSON object: a malformed HUD must not take the teleoperation down.
    """
    text = _first_string(value)
    if text is None:
        print("hud: expected a utf8 string input", flush=True)
        return None
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        print(f"hud: not JSON: {error}", flush=True)
        return None
    if not isinstance(document, dict):
        print(f"hud: not a JSON object: {document!r}", flush=True)
        return None

    lines = document.get("lines") or []
    if isinstance(lines, str):
        lines = lines.splitlines()
    if not isinstance(lines, list):
        print(f"hud: lines is not a list: {lines!r}", flush=True)
        lines = []
    lines = [str(line)[:_HUD_MAX_LINE_LENGTH] for line in lines[:_HUD_MAX_LINES]]

    haptic = document.get("haptic")
    if haptic not in _HAPTIC_TARGETS:
        if haptic is not None:
            print(f"hud: ignoring haptic {haptic!r}", flush=True)
        haptic = None

    return {
        "type": "hud",
        "lines": lines,
        "recording": bool(document.get("recording", False)),
        "haptic": haptic,
    }


def _on_hud(value):
    """Forward a ``hud`` input to the headset."""
    message = _parse_hud(value)
    if message is not None:
        webrtc_server.send_hud(message)


def _control_command(value):
    """Return the command a ``control`` input carries, stripped, or None."""
    text = _first_string(value)
    return text.strip() if text is not None else None


def _is_uint8_array(value):
    """Whether a dora input value is the uint8 array binary payloads arrive as."""
    arrow_type = getattr(value, "type", None)
    return arrow_type is not None and pa.types.is_uint8(arrow_type)


def _parse_scene_state(value):
    """Turn a ``scene_state`` input into the payload the page's channel takes.

    openarm-quest-teleop patch. The sim node sends either a utf8 JSON
    string or, with ``--scene-state-format binary``, the message's bytes
    as a uint8 array; both go to the browser as they are. This node never
    parses the scene itself -- it only has to know how big it is and that
    it is one of those two shapes -- so the codec can gain a field, or
    change encoding, without a change here.

    Returns the ``str`` or ``bytes`` to send, or None (saying why, once)
    for anything else: a malformed scene must not take teleoperation
    down, and one per frame would drown the log.
    """
    global _warned_scene_state
    if isinstance(value, (str, bytes)):
        payload = value
    elif _is_uint8_array(value):
        # The binary encoding, as the camera JPEGs arrive: raw bytes.
        payload = value.to_numpy(zero_copy_only=False).tobytes()
    else:
        payload = _first_string(value)
    if not isinstance(payload, (str, bytes)) or not payload:
        if not _warned_scene_state:
            _warned_scene_state = True
            print(
                "scene_state: expected a utf8 JSON string or a uint8 array, "
                f"got {type(value).__name__}",
                flush=True,
            )
        return None
    size = len(payload.encode("utf-8")) if isinstance(payload, str) else len(payload)
    if size > _SCENE_STATE_MAX_BYTES:
        if not _warned_scene_state:
            _warned_scene_state = True
            print(f"scene_state: dropping a {size} byte message", flush=True)
        return None
    return payload


def _on_scene_state(value):
    """Forward one scene state to the active headset, or drop it.

    openarm-quest-teleop patch: dropped where it arrives, without so much as
    being parsed, when this view draws no 3D scene. The page registers no
    handler for the ``state`` channel outside a 3D view, and the node opens
    none, so forwarding would put a message a frame on the headset's radio for
    nothing -- and on a half-duplex channel that is airtime taken from the hand
    poses coming back. Said once, with the rate, so a flow that wires the input
    to a 2D view is found out rather than silently paid for.
    """
    global _state_forwarded, _state_reported, _state_reported_at
    global _state_bytes_reported, _scene_state_unwanted, _warned_scene_state_unwanted
    if not _FORWARD_SCENE_STATE:
        _scene_state_unwanted += 1
        if not _warned_scene_state_unwanted:
            _warned_scene_state_unwanted = True
            print(
                "scene_state: this view draws no 3D scene "
                f"({video.view_configuration().get('view')!r}), so the stream is dropped "
                "here and no state channel is offered. Nothing was sent to the headset; "
                "run the sim with --scene-state-hz 0, or the headset on --view scene",
                flush=True,
            )
        return
    payload = _parse_scene_state(value)
    if payload is None:
        return
    if _state_reported_at == 0.0:
        # The first window starts with the first message, not at node startup:
        # a dataflow can sit for a while before the sim begins publishing.
        _state_reported_at = time.perf_counter()
    webrtc_server.send_state(payload)
    _state_forwarded += 1
    if _state_forwarded % _STATE_REPORT_EVERY == 0:
        now = time.perf_counter()
        elapsed = max(now - _state_reported_at, 1e-9) if _state_reported_at else 0.0
        dropped = webrtc_server.state_dropped - _state_reported
        sent_bytes = webrtc_server.state_bytes - _state_bytes_reported
        sent = _STATE_REPORT_EVERY - dropped
        line = f"scene_state: {_STATE_REPORT_EVERY} forwarded"
        if elapsed > 0.0:
            line += f" in {elapsed:.1f} s ({_STATE_REPORT_EVERY / elapsed:.1f} Hz)"
        if sent > 0:
            line += f", {sent_bytes / sent:.0f} B each"
        if dropped:
            # Said as what it measures. The old wording ("did not fit the
            # headset link") read as a capacity verdict; the test is
            # bufferedAmount against a byte limit, which the node's own event
            # loop trips as readily as a slow radio, so the peak backlog is
            # named next to it and the reader can tell the two apart.
            line += (
                f"; {dropped} dropped with more than "
                f"{webrtc_server.state_buffer_limit} B already queued "
                f"(peak {webrtc_server.state_buffered_peak} B)"
            )
        else:
            line += "; none dropped"
        print(line, flush=True)
        _state_reported = webrtc_server.state_dropped
        _state_bytes_reported = webrtc_server.state_bytes
        _state_reported_at = now
        webrtc_server.state_buffered_peak = 0


def _read_glb_extras(path):
    """Return the scene extras of a glTF binary, or {} if there are none.

    Only the JSON chunk is read, so the mesh data of an 11 MB export is
    never loaded to answer "which bodies does it hold".
    """
    try:
        with open(path, "rb") as glb:
            header = glb.read(20)
            if len(header) < 20:
                raise ValueError("shorter than a glTF header")
            magic, _version, _length, chunk_length, chunk_type = struct.unpack(
                "<IIIII", header
            )
            if magic != _GLB_MAGIC or chunk_type != _GLB_JSON_CHUNK:
                raise ValueError("not a binary glTF with a JSON chunk")
            if chunk_length > _GLB_JSON_LIMIT:
                raise ValueError(f"JSON chunk of {chunk_length} bytes")
            document = json.loads(glb.read(chunk_length).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as error:
        print(f"cannot read the scene extras of {path}: {error}", flush=True)
        return {}
    scenes = document.get("scenes") if isinstance(document, dict) else None
    scene = scenes[0] if isinstance(scenes, list) and scenes else {}
    extras = scene.get("extras") if isinstance(scene, dict) else None
    return extras if isinstance(extras, dict) else {}


def _compress_scene(payload, digest):
    """Gzip one export, or return None if compressing it is not worth it.

    openarm-quest-teleop patch. ``mtime=0`` keeps the compressed bytes a
    function of the file alone, the way the export itself is: two nodes
    serving the same scene then serve the same bytes, and the ETag below
    can be derived from the file's hash rather than from the clock.

    A file that does not shrink (an already-compressed asset, or one small
    enough that the gzip framing costs more than it saves) gets None, and
    the endpoint serves it as it always did.
    """
    try:
        body = gzip.compress(payload, _SCENE_GZIP_LEVEL, mtime=0)
    except (MemoryError, OSError, zlib.error) as error:
        # Never fatal: an uncompressed scene is slow, no scene is a black
        # headset (X5).
        print(f"3D scene: cannot gzip it ({error}); serving it as it is", flush=True)
        return None
    if len(body) >= len(payload):
        return None
    return {"hash": digest, "body": body}


def _configure_scene(path):
    """Settle what the configuration message says about the 3D scene.

    openarm-quest-teleop patch. The hash is what lets a page refuse a
    glTF it has cached from an earlier run: the exporter is deterministic,
    so the same scene hashes the same and a different one does not. ``nq``
    and the body list come from the export itself, which is the only thing
    that can say which joint value the page will apply where.

    Also gzips the file into ``_SCENE_GZIP``, once, here: this is the only
    place the whole export is already in memory, and the endpoint below
    then answers from it without touching the disk or the CPU again.

    Returns the ``scene`` block, or None when there is no readable file,
    in which case the page is told nothing and keeps working as it does
    without a 3D scene at all.
    """
    global _SCENE_GZIP
    _SCENE_GZIP = None
    if path is None:
        return None
    path = pathlib.Path(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        print(f"no 3D scene: cannot read {path}: {error}", flush=True)
        return None
    extras = _read_glb_extras(path)
    scene = {
        "url": _SCENE_URL,
        "hash": hashlib.blake2b(payload, digest_size=16).hexdigest(),
        "bytes": len(payload),
    }
    if isinstance(extras.get("nq"), int):
        scene["nq"] = extras["nq"]
    bodies = extras.get("bodies")
    if isinstance(bodies, list) and all(isinstance(name, str) for name in bodies):
        scene["bodies"] = bodies
    _SCENE_GZIP = _compress_scene(payload, scene["hash"])
    served = (
        f"{len(_SCENE_GZIP['body'])} gzipped"
        if _SCENE_GZIP is not None
        else "uncompressed (it does not shrink)"
    )
    print(
        f"3D scene: {path} ({scene['bytes']} bytes, blake2b {scene['hash']}, "
        f"nq {scene.get('nq', 'unknown')}, {len(scene.get('bodies', []))} bodies) "
        f"served at {_SCENE_URL} as {served}",
        flush=True,
    )
    return scene


# ---------------------------------------------------------------------------
# openarm-quest-teleop patch: the headset's own log (T3).
#
# Debugging the page through the headset's developer mode means a cable, a
# paired laptop and chrome://inspect, at the moment when the operator is
# already wearing a headset that stopped working. So the page keeps its own
# bounded buffer of what went wrong and sends it here, and this node appends
# it to one file next to the run's dora log.
#
# Two ways in, because the case worth debugging is usually the case where
# WebRTC is broken:
#
# * the reliable ``control`` data channel, when there is a peer (no token: the
#   peer only exists because an offer carried one), and
# * ``POST /log``, which needs no WebRTC at all and carries the pairing token
#   in its body -- ``navigator.sendBeacon`` cannot set a header, and a beacon
#   on pagehide is the only thing a crashing page can still do.
#
# Everything below is written so that a page that has gone mad cannot take
# the node down with it: a bounded read rather than an unbounded one, a rate
# limit per peer and per record, a cap on the file with one rotation, and a
# sanitiser that truncates and flattens whatever it is handed. Nothing here
# ever raises into the event loop that answers the operator's hands.

# What this node is: the package version and a hash of the very files it
# serves the page from. Said once at startup and pushed to every page in the
# configuration message, because the page cannot know which build of itself
# the headset's HTTP cache handed it -- and a page running last week's
# bundle against this week's node is a whole class of reports that otherwise
# looks like a node that has gone mad. The static hash and not the version
# alone: the version does not move while the page is being worked on, which
# is exactly when a cached bundle is most likely.
_BUILD = None


def _build_identity(directory=None):
    """Return this node's build: its version plus a hash of the served files."""
    try:
        version = importlib.metadata.version("dora-openarm-webxr")
    except importlib.metadata.PackageNotFoundError:
        version = "0"
    directory = pathlib.Path(directory or pathlib.Path(__file__).parent / "static")
    digest = hashlib.blake2b(digest_size=8)
    try:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
                digest.update(path.read_bytes())
    except OSError as error:
        # Only a label on a log line; a node that cannot read its own static
        # directory has worse problems and will meet them at the first GET.
        print(f"WARNING: cannot hash the served page: {error}", flush=True)
        return version
    return f"{version}+{digest.hexdigest()}"


# The most a POST /log body may be. A page buffer worth sending is a few tens
# of kilobytes; past this the request is refused with 413 rather than read.
QUEST_LOG_MAX_BODY = 64 << 10
# ... and the most of an oversized body that is read and thrown away before
# the connection is dropped, so that an ordinary too-big POST still gets its
# 413 back instead of a reset socket, while a sender that streams for ever
# does not hold this task for ever either.
QUEST_LOG_DRAIN_LIMIT = 16 << 20
# Records per envelope; the rest are dropped and counted.
QUEST_LOG_MAX_RECORDS = 64
# Per string value, per key, and per list/object.
QUEST_LOG_MAX_STRING = 2048
QUEST_LOG_MAX_KEY = 64
QUEST_LOG_MAX_ITEMS = 64
# How deep a record may nest before the rest becomes one string.
QUEST_LOG_MAX_DEPTH = 6
# The page's session id, as kept in every line.
QUEST_LOG_MAX_SESSION = 64
# The file's cap, and its one rotation: quest.log -> quest.log.1, so a run
# never costs more than twice this on disk.
QUEST_LOG_MAX_BYTES = 4 << 20
QUEST_LOG_ROTATED_SUFFIX = ".1"
# Records a second, averaged, with a burst: a page in a console.error loop at
# the frame rate is 72 a second per headset, and this still leaves room for
# the buffer of a session that reconnects.
QUEST_LOG_RECORD_RATE = 200.0
QUEST_LOG_RECORD_BURST = 400.0
# POSTs a second, per peer address, with a burst for a page flushing its
# buffer in several pieces. Beyond it, 429.
QUEST_LOG_REQUEST_RATE = 5.0
QUEST_LOG_REQUEST_BURST = 20.0
# Peer addresses the request limiter remembers; the least recently seen is
# forgotten first, so a spoofed flood cannot grow it without bound.
QUEST_LOG_MAX_PEERS = 64
# How many error records are mirrored to stderr, where they land in the run's
# dora log. A trickle, because that log is where the operator reads about the
# robot; the file next to it has every one.
QUEST_LOG_MIRROR_RATE = 0.5
QUEST_LOG_MIRROR_BURST = 10.0
# After a write fails (a full disk, a run directory pulled out from under us)
# nothing is written for this long, then it is tried again. Teleoperation must
# not pay for a broken log file on every record.
QUEST_LOG_RETRY_S = 5.0
# The levels a record may carry; anything else is recorded as "info".
QUEST_LOG_LEVELS = ("debug", "info", "warn", "error")
QUEST_LOG_DEFAULT_LEVEL = "info"
# What a redacted value becomes.
QUEST_LOG_REDACTED = "<redacted>"
# A pairing token shorter than this is not searched for in the log text: the
# real one is 32 characters of token_urlsafe, and blanking every occurrence of
# a two-letter string would shred the very messages this file exists for.
QUEST_LOG_MIN_SECRET = 8

# Anything with a scheme and an authority. Only the query and the fragment are
# taken off it: the path is what makes a stack trace readable ("ar.js:66:13"),
# and the fragment is where the page's pairing token lives (#k=...).
#
# The lookbehind and the possessive `*+` are what keep this linear, and they
# are not decoration. A scheme cannot follow a letter or a digit, so on a long
# run of them -- which is exactly what a page in trouble sends: a base64 blob,
# a 60 kB string of one character -- only the first character of the run is a
# candidate start, and the run is scanned once instead of once per position.
# Without them this expression is quadratic and a single 64 kB word costs
# MINUTES of the event loop that answers the operator's hands.
_URL = re.compile(r"(?<![a-zA-Z0-9+.\-])[a-zA-Z][a-zA-Z0-9+.\-]*+://[^\s\"'<>\\]*+")
# ... and the same key=value in a fragment or query that was not part of a
# whole URL ("#k=abc" on its own), for the page that logs only the fragment.
_SECRET_PARAMETER = re.compile(
    r"(?i)([?#&])(k|token|key|secret|pairing[_-]?token)=[^&\s\"']*"
)
# ... and the same thing quoted in a JSON body the page logged: the offer it
# posts carries {"sdp": ..., "token": ...}. The same three names the page's own
# redaction uses, and not "key", which is also what a keyboard event calls the
# letter that was pressed. A token this node knows is already gone by the time
# this runs; this is for one it does not (another node's).
_SECRET_JSON = re.compile(r'(?i)("(?:k|token|pairing[_-]?token)"\s*:\s*")[^"\\]*"')

# Where the headset's log is written, from --quest-log (QUEST_LOG), and the
# writer itself. None means this node has no sink: POST /log answers 503 and
# the page is expected to keep its own buffer.
_quest_log = None


class _TokenBucket:
    """Rate limit: ``burst`` at once, then ``rate`` a second.

    A bucket rather than a counter per window, so a page that flushes a
    buffer at reconnect is not cut in half by a window boundary.
    """

    def __init__(self, rate, burst, clock=time.monotonic):
        """Start full, at ``rate`` tokens a second up to ``burst``."""
        self._rate = float(rate)
        self._burst = float(burst)
        self._tokens = float(burst)
        self._clock = clock
        self._at = clock()

    def take(self):
        """Spend one token; False when there is none (i.e. refuse this one)."""
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + (now - self._at) * self._rate)
        self._at = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


class _PeerLimiter:
    """One :class:`_TokenBucket` per peer address, with a bounded memory.

    Per peer rather than global: two headsets are a supported configuration
    (a handover, a second operator watching), and one of them flooding must
    not make the other's crash report disappear.
    """

    def __init__(
        self,
        rate=QUEST_LOG_REQUEST_RATE,
        burst=QUEST_LOG_REQUEST_BURST,
        max_peers=QUEST_LOG_MAX_PEERS,
        clock=time.monotonic,
    ):
        """Remember at most ``max_peers`` peers, the oldest forgotten first."""
        self._rate = rate
        self._burst = burst
        self._max_peers = max_peers
        self._clock = clock
        self._buckets = collections.OrderedDict()

    def take(self, peer):
        """Whether ``peer`` may make one more request now."""
        bucket = self._buckets.pop(peer, None)
        if bucket is None:
            bucket = _TokenBucket(self._rate, self._burst, self._clock)
        self._buckets[peer] = bucket
        while len(self._buckets) > self._max_peers:
            self._buckets.popitem(last=False)
        return bucket.take()


# The limiter for POST /log. Module level, and not part of _QuestLog, because
# it has to refuse a flood whether or not this node has a file to write to.
_log_requests = _PeerLimiter()


def _redact(text):
    """Return ``text`` with the pairing token and every URL secret removed.

    Three things go, in this order: the query and fragment of every URL (the
    page's own URL carries the token in its fragment), the configured pairing
    token wherever else it appears, and a bare ``#k=...`` that was not part of
    a URL. What is left -- the origin, the path, the line and column of a
    stack frame -- is what makes the record worth keeping.

    The URLs go first because what replaces a secret contains ``<`` and
    ``>``, which is exactly where a URL ends: redacting inside one first
    would cut the URL in half and leave the rest of it behind.

    Unconditional, because the page cannot be trusted to remember: a stack
    trace of a handler defined in the page's own URL carries the fragment
    with it, and nobody writing telemetry on the other side should have to
    think about that.
    """
    if not isinstance(text, str):
        return text
    text = _URL.sub(_redact_url, text)
    token = _PAIRING_TOKEN
    if token and len(token) >= QUEST_LOG_MIN_SECRET:
        text = text.replace(token, QUEST_LOG_REDACTED)
    text = _SECRET_PARAMETER.sub(rf"\1\2={QUEST_LOG_REDACTED}", text)
    return _SECRET_JSON.sub(rf'\1{QUEST_LOG_REDACTED}"', text)


def _redact_url(match):
    """Replace one URL's query and fragment, keeping scheme, host and path."""
    url = match.group(0)
    head = url
    fragment = query = None
    head, marker, rest = head.partition("#")
    if marker:
        fragment = rest
    head, marker, rest = head.partition("?")
    if marker:
        query = rest
    if query is not None:
        head += f"?{QUEST_LOG_REDACTED}"
    if fragment is not None:
        head += f"#{QUEST_LOG_REDACTED}"
    return head


def _clean_text(value, limit=QUEST_LOG_MAX_STRING):
    """One string as it is written: redacted, then cut to ``limit``."""
    text = _redact(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _clean(value, depth=0):
    """Return ``value`` reduced to something small and safe to write.

    Strings are redacted and truncated, containers are cut to
    ``QUEST_LOG_MAX_ITEMS`` and ``QUEST_LOG_MAX_DEPTH``, and anything else --
    including a number long enough to be a payload of its own -- becomes a
    truncated string. The page is not trusted to have been reasonable.
    """
    if isinstance(value, str):
        return _clean_text(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        # A JSON integer has no size limit; a megabyte of digits would be
        # written out in full.
        return value if abs(value) < (1 << 63) else _clean_text(str(value), 64)
    if isinstance(value, float):
        # NaN and Infinity are not JSON, and json.dumps writes them anyway.
        return value if math.isfinite(value) else _clean_text(repr(value), 32)
    if depth >= QUEST_LOG_MAX_DEPTH:
        return _clean_text(repr(value))
    if isinstance(value, dict):
        return {
            _clean_text(str(key), QUEST_LOG_MAX_KEY): _clean(item, depth + 1)
            for key, item in list(value.items())[:QUEST_LOG_MAX_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [_clean(item, depth + 1) for item in list(value)[:QUEST_LOG_MAX_ITEMS]]
    return _clean_text(repr(value))


def _clean_level(value):
    """One of QUEST_LOG_LEVELS, whatever the page said."""
    if isinstance(value, str) and value.strip().lower() in QUEST_LOG_LEVELS:
        return value.strip().lower()
    return QUEST_LOG_DEFAULT_LEVEL


class _QuestLog:
    """The file the headset's own records are appended to.

    One JSON object per line (JSONL), so `tail -f` works, a truncated last
    line costs one record, and `oqt` can summarise the file without parsing a
    format of its own. Each line is the page's record with this node's own
    fields put in front of it: when it arrived, from which peer and over
    which transport, which page session it belongs to, and a sequence number
    of this node's own -- a gap in that number is this node admitting it
    dropped something, which no counter kept elsewhere could tell the
    operator reading the file afterwards.

    Everything the page sends is passed through except what could carry the
    pairing token, which is taken out unconditionally (see :func:`_redact`).
    """

    def __init__(
        self,
        path,
        *,
        max_bytes=QUEST_LOG_MAX_BYTES,
        max_records=QUEST_LOG_MAX_RECORDS,
        record_rate=QUEST_LOG_RECORD_RATE,
        record_burst=QUEST_LOG_RECORD_BURST,
        mirror_rate=QUEST_LOG_MIRROR_RATE,
        mirror_burst=QUEST_LOG_MIRROR_BURST,
        retry_s=QUEST_LOG_RETRY_S,
        clock=time.monotonic,
        now=None,
        stderr=None,
    ):
        """Append to ``path``, which is opened on the first record.

        ``clock`` drives the rate limits and ``now`` stamps the records
        (default: the local wall clock); the tests pass both.
        """
        self.path = pathlib.Path(path)
        self._max_bytes = max_bytes
        self._max_records = max_records
        self._records = _TokenBucket(record_rate, record_burst, clock)
        self._mirror = _TokenBucket(mirror_rate, mirror_burst, clock)
        self._retry_s = retry_s
        self._clock = clock
        self._now = now or (
            lambda: datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="milliseconds")
        )
        # None means "whatever sys.stderr is when the line is written", not
        # whatever it was when the node started: dora replaces it, and so
        # does pytest.
        self._stderr = stderr
        self._file = None
        self._size = 0
        self._blocked_until = None
        self._sequence = 0
        self._dropped_since_write = 0
        self._mirror_quiet = False
        # Public counters, for the line the node prints when it stops.
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self.warnings = 0
        self.rotations = 0
        self.write_failures = 0
        # What the PAGE said it dropped before it reached us: a different
        # fault from this node's own drops, and kept apart from them.
        self.page_dropped = 0

    # -- taking records in ---------------------------------------------------

    def accept(self, envelope, peer, via):
        """Write one envelope's records; how many were written, or None.

        None means the envelope was not one at all, which over HTTP is a 400.
        Anything else -- a record that is not an object, a batch longer than
        the cap, a flood -- is counted and dropped, because half a report is
        still worth having and the page cannot fix it from where it is.
        """
        if not isinstance(envelope, dict):
            return None
        records = envelope.get("records")
        if not isinstance(records, list):
            return None
        session = (
            _clean_text(str(envelope.get("session", "")), QUEST_LOG_MAX_SESSION)
            or "unknown"
        )
        meta = self._envelope_fields(envelope)
        written = self._page_dropped(envelope, peer, via, session, meta)
        for record in records[: self._max_records]:
            if not isinstance(record, dict) or not self._records.take():
                self.dropped += 1
                self._dropped_since_write += 1
                continue
            written += self._emit(record, peer, via, session, meta)
        extra = len(records) - self._max_records
        if extra > 0:
            self.dropped += extra
            self._dropped_since_write += extra
        return written

    @staticmethod
    def _envelope_fields(envelope):
        """Return the envelope's own fields, to repeat on every line it carries.

        Repeated rather than written once per envelope, because every line of
        this file is read on its own: an operator greps one error out and
        sends it, and it has to say which page build, which node build and
        which batch it belongs to without the lines around it.

        ``page`` is the page's own envelope counter, so a GAP in it is an
        envelope that never arrived -- which is the one thing the node's own
        ``seq`` cannot show.
        """
        fields = {}
        for key, name in (
            ("build", "build"),  # which build of the page this is
            ("node", "node"),  # which build of the node the page was told
            ("page", "page"),  # the page's envelope counter
            ("reason", "flush"),  # why this batch was flushed (beacon, timer)
            ("v", "v"),  # the record schema the page is writing
        ):
            value = envelope.get(key)
            if isinstance(value, str) and value:
                fields[name] = _clean_text(value, QUEST_LOG_MAX_KEY)
            elif isinstance(value, int) and not isinstance(value, bool):
                fields[name] = value
        return fields

    def _page_dropped(self, envelope, peer, via, session, meta):
        """Note what the PAGE's own buffer threw away before this batch.

        The page's ring is bounded too, and a page that filled it is exactly
        a page in trouble. Without this line the gap would look like the node
        losing records, which is a different fault with a different fix.
        """
        dropped = envelope.get("dropped")
        if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped <= 0:
            return 0
        self.page_dropped += dropped
        return self._line(
            {
                "level": "warn",
                "event": "page-dropped",
                "message": f"the page's own buffer dropped {dropped} records",
                "count": dropped,
            },
            peer,
            via,
            session,
            meta,
        )

    def _emit(self, record, peer, via, session, meta):
        """Write one record, preceded by a note of anything dropped before it."""
        if self._dropped_since_write:
            dropped, self._dropped_since_write = self._dropped_since_write, 0
            self._line(
                {
                    "level": "warn",
                    "event": "quest-log-dropped",
                    "message": f"{dropped} records were dropped before this one",
                    "count": dropped,
                },
                peer,
                via,
                session,
                meta,
            )
        return self._line(record, peer, via, session, meta)

    def _line(self, record, peer, via, session, meta):
        """Shape one record into a line and write it; 1 if it was written."""
        level = _clean_level(record.get("level"))
        event = _clean_text(str(record.get("event", "")), QUEST_LOG_MAX_KEY) or "log"
        self._sequence += 1
        line = {
            "received": self._now(),
            "level": level,
            "event": event,
            "session": session,
            "peer": peer,
            "via": via,
            "seq": self._sequence,
            **meta,
        }
        for key, value in list(record.items())[:QUEST_LOG_MAX_ITEMS]:
            name = _clean_text(str(key), QUEST_LOG_MAX_KEY)
            # The node's own fields are the ones a reader trusts, so a page
            # cannot overwrite them; the token is never written at all.
            if name in line or name == "token":
                continue
            line[name] = _clean(value)
        if level == "error":
            self.errors += 1
            self._mirror_error(line)
        elif level == "warn":
            self.warnings += 1
        if not self._write(json.dumps(line, ensure_ascii=False, default=str)):
            return 0
        self.written += 1
        return 1

    def _mirror_error(self, line):
        """Put one error where the operator already looks: this node's stderr.

        dora tags a node's stderr and relays it, so the line lands in
        ``runs/<name>/dora_run.log`` beside everything else about the run.
        Rate limited hard: the file has every record, and this is only there
        so that an operator reading the robot's log learns that the headset
        has something to say at all.
        """
        if not self._mirror.take():
            if not self._mirror_quiet:
                self._mirror_quiet = True
                self._say(f"further headset errors are only in {self.path}")
            return
        self._mirror_quiet = False
        # "msg" is what static/telemetry.js calls it; "message" is what a
        # plain caller (or a test, or another page) would write. Whichever
        # the record used, the operator gets the sentence.
        message = line.get("msg") or line.get("message")
        detail = f": {message}" if isinstance(message, str) and message else ""
        repeats = line.get("n")
        if isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 1:
            # The page coalesces a repeat rather than sending it 72 times a
            # second, so the count is the whole story of how bad it is.
            detail += f" (x{repeats})"
        self._say(f"headset error [{line['session']}] {line['event']}{detail}")

    def _say(self, text):
        try:
            print(
                f"quest.log: {text}",
                file=self._stderr if self._stderr is not None else sys.stderr,
                flush=True,
            )
        except (OSError, ValueError):
            pass

    # -- the file ------------------------------------------------------------

    def _write(self, text):
        """Append one line, rotating first if it would not fit. True if written."""
        if self._blocked_until is not None:
            if self._clock() < self._blocked_until:
                self.dropped += 1
                self._dropped_since_write += 1
                return False
            self._blocked_until = None
        data = text.encode("utf-8", "replace") + b"\n"
        try:
            if self._file is None:
                self._open()
            if self._size and self._size + len(data) > self._max_bytes:
                self._rotate()
            self._file.write(data)
            self._file.flush()
        except OSError as error:
            # A full disk or a deleted run directory must not stop the node,
            # and must not cost a syscall per record either.
            self.write_failures += 1
            self.dropped += 1
            self._dropped_since_write += 1
            self._blocked_until = self._clock() + self._retry_s
            self._close()
            self._say(f"cannot write {self.path}: {error}")
            return False
        self._size += len(data)
        return True

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "ab")
        self._size = self.path.stat().st_size

    def _rotate(self):
        """Move the file aside once, so a run costs at most twice the cap."""
        self._close()
        rotated = self.path.with_name(self.path.name + QUEST_LOG_ROTATED_SUFFIX)
        os.replace(self.path, rotated)
        self.rotations += 1
        self._say(f"over {self._max_bytes} bytes; the older records are in {rotated}")
        self._open()

    def _close(self):
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
        self._size = 0

    def summary(self):
        """One line about what the headset reported, for the node's own log."""
        parts = [f"{self.written} records"]
        if self.errors:
            parts.append(f"{self.errors} errors")
        if self.warnings:
            parts.append(f"{self.warnings} warnings")
        if self.dropped:
            parts.append(f"{self.dropped} dropped")
        if self.page_dropped:
            parts.append(f"{self.page_dropped} dropped by the page")
        if self.rotations:
            parts.append(f"{self.rotations} rotations")
        return f"quest.log: {', '.join(parts)} ({self.path})"

    def close(self):
        """Say what was written and close the file."""
        print(self.summary(), flush=True)
        self._close()


def _open_quest_log(path):
    """Return the sink for ``--quest-log``, or None when there is none.

    Never fatal: a node that cannot write the headset's log still
    teleoperates, and saying so once at startup is the whole remedy.
    """
    if path is None:
        print(
            "quest log: off (--quest-log / QUEST_LOG unset); the page's own error "
            "reports are refused with 503 and it keeps them buffered",
            flush=True,
        )
        return None
    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"WARNING: cannot use {path} as the headset's log: {error}", flush=True)
        return None
    print(
        f"quest log: {path} (the headset's own errors, over the control channel "
        "and POST /log)",
        flush=True,
    )
    return _QuestLog(path)


def _on_quest_log(payload, peer):
    """Take one log envelope off a peer's ``control`` channel.

    No pairing token is asked for here: the peer exists only because an offer
    carried one. Nothing raises, because this runs inside aiortc's own
    callback.
    """
    if _quest_log is None:
        return
    try:
        _quest_log.accept(payload, peer=peer, via="control")
    except Exception as error:  # noqa: BLE001 - a bad record must not end the session
        print(f"WARNING: cannot record a headset log message: {error!r}", flush=True)


async def _read_bounded(request, limit, drain_limit=QUEST_LOG_DRAIN_LIMIT):
    """Return the request body, or None if it is longer than ``limit`` bytes.

    Reads the stream rather than ``request.body()``, which would buffer
    whatever was sent. An oversized body is still read -- and discarded --
    up to ``drain_limit``, so an honest client that sent too much gets the
    413 it deserves instead of a connection reset in the middle of writing;
    past that the connection is dropped instead.
    """
    declared = request.headers.get("content-length")
    too_large = False
    if declared is not None:
        try:
            too_large = int(declared) > limit
        except ValueError:
            too_large = False
    chunks = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if too_large or total > limit:
            too_large = True
            chunks.clear()
            if total > drain_limit:
                break
            continue
        chunks.append(chunk)
    return None if too_large else b"".join(chunks)


@app.post("/log")
@app.post("/telemetry")
async def _log_endpoint(request: Request):
    """Take the headset's own log records over plain HTTPS.

    openarm-quest-teleop patch (T3). The body is the envelope the page also
    sends on the ``control`` channel -- ``session``, ``records`` and the
    fields :meth:`_QuestLog._envelope_fields` keeps -- plus the pairing
    token, because this path is reachable by anything that reaches the port
    and because ``navigator.sendBeacon``, the only thing a page being torn
    down can still use, cannot set a header.

    This is the path that matters: when WebRTC is what broke, the control
    channel is exactly what the page cannot use to say so.

    Two paths, one endpoint: ``/log`` is the name every current caller uses
    (static/telemetry.js and both browser tests), and ``/telemetry`` is kept
    as an alias because a headset can be holding an older cached page that
    posts there -- which is precisely the population this feature exists to
    hear from. A page that reached neither would get a 404, back off and keep
    buffering -- silently, which is the one failure this feature cannot
    afford.

    The answer carries no body (a beacon could not read one anyway): 204 when
    the records were taken, 413 for a body past ``QUEST_LOG_MAX_BODY``, 429
    when this peer is posting too often, 403 for a wrong or missing token,
    503 when this node has no file to write to (the page should keep its
    buffer) and 400 for a body that is not an envelope.
    """
    client = request.client.host if request.client else "unknown"
    # Cheapest check first: a peer that is over its budget is refused before
    # its body is read at all, so a flood costs this node nothing but the
    # socket it arrived on.
    if not _log_requests.take(client):
        return _error(429, "too many log posts")
    try:
        body = await _read_bounded(request, QUEST_LOG_MAX_BODY)
    except Exception as error:  # noqa: BLE001 - a headset that went away mid-post
        # Starlette raises ClientDisconnect here, and a page being torn down
        # is precisely when this endpoint is used. Rate limited by the budget
        # spent above.
        print(f"a log post from {client} was cut off: {error!r}", flush=True)
        return _error(400, "the log post was cut off")
    if body is None:
        return _error(413, f"a log post must be at most {QUEST_LOG_MAX_BODY} bytes")
    try:
        envelope = json.loads(body)
    except (ValueError, RecursionError):
        envelope = None
    if not _pairing_token_accepted(envelope):
        print(
            f"refused a log post from {client}: wrong or missing pairing token",
            flush=True,
        )
        return _error(403, "wrong or missing pairing token")
    if _quest_log is None:
        return _error(503, "this node was not given a log file (--quest-log)")
    written = _quest_log.accept(envelope, peer=client, via="http")
    if written is None:
        return _error(400, 'expected a JSON object with a "records" list')
    return Response(status_code=204)


def _pairing_token_accepted(offer):
    """Whether an offer body carries the pairing token, if one is required.

    openarm-quest-teleop patch. Anything that reaches the port could
    otherwise post an offer and start a session: the node listens on
    every interface, and whichever session is active drives the robot.
    """
    global _warned_unpaired
    if _PAIRING_TOKEN is None:
        if not _warned_unpaired:
            _warned_unpaired = True
            print(
                "WARNING: PAIRING_TOKEN is not set, so any client that reaches "
                "this port can start a teleoperation session",
                flush=True,
            )
        return True
    token = offer.get("token") if isinstance(offer, dict) else None
    if not isinstance(token, str):
        return False
    # Compared as bytes: compare_digest refuses non-ASCII str, and a
    # malformed token must be a 403, not a crash.
    return secrets.compare_digest(token.encode("utf-8"), _PAIRING_TOKEN.encode("utf-8"))


def _error(status_code, message):
    return JSONResponse({"error": message}, status_code=status_code)


@app.post("/offer")
async def _offer_endpoint(request: Request):
    """Answer a browser's SDP offer.

    Only reachable when this node serves the page itself. In WebRTC-only
    mode the offer arrives at startup instead and no HTTP server runs.

    openarm-quest-teleop patch: the body is ``{"sdp": ..., "token": ...}``.
    A missing or wrong pairing token is refused with 403 before any peer
    exists, a body that is not an offer with 400, and an offer while too
    many peers are still negotiating with 503.
    """
    client = request.client.host if request.client else "unknown"
    try:
        offer = json.loads(await request.body())
    except (ValueError, RecursionError):
        # Not JSON (JSONDecodeError and UnicodeDecodeError are both
        # ValueErrors): no token can be read from it either.
        offer = None
    if not _pairing_token_accepted(offer):
        print(
            f"refused an offer from {client}: wrong or missing pairing token",
            flush=True,
        )
        return _error(403, "wrong or missing pairing token")
    if not isinstance(offer, dict) or not isinstance(offer.get("sdp"), str):
        return _error(400, 'expected a JSON object with an "sdp" string')
    try:
        answer_sdp = await webrtc_server.answer(offer["sdp"])
    except webrtc.PendingPeerLimitError as error:
        print(f"refused an offer from {client}: {error}", flush=True)
        return _error(503, str(error))
    except Exception as error:  # noqa: BLE001
        # aiortc reports a malformed SDP with whatever its parser tripped
        # on (ValueError, KeyError, AssertionError, ...), and answer() has
        # already closed the peer: the offer was bad, not the node.
        print(f"cannot answer an offer from {client}: {error!r}", flush=True)
        return _error(400, f"cannot answer the offer: {error!r}")
    return {"sdp": answer_sdp, "type": "answer"}


def _accepts_gzip(request):
    """Whether the client asked for gzip, honouring ``gzip;q=0``.

    A q of 0 means "not this one" (RFC 9110 12.5.3), and a client that
    says so means it -- a proxy unpacking the stream, or a fetch that
    wants the bytes the ETag names. An explicit ``gzip`` beats a ``*``,
    which is what the wildcard is for.
    """
    quality = wildcard = None
    for part in request.headers.get("accept-encoding", "").split(","):
        name, _, parameters = part.strip().partition(";")
        name = name.lower()
        if name not in ("gzip", "*"):
            continue
        weight = 1.0
        parameters = parameters.strip().lower()
        if parameters.startswith("q="):
            try:
                weight = float(parameters[2:])
            except ValueError:
                weight = 1.0
        if name == "gzip":
            quality = weight
        else:
            wildcard = weight
    if quality is None:
        quality = wildcard
    return quality is not None and quality > 0.0


def _matches_etag(header, etag):
    """Whether an ``If-None-Match`` header covers ``etag``."""
    if header is None:
        return False
    if header.strip() == "*":
        return True
    # Weak comparison, which is what If-None-Match uses: W/"x" matches "x".
    wanted = etag.strip('"')
    for candidate in header.split(","):
        candidate = candidate.strip()
        if candidate.startswith(("W/", "w/")):
            candidate = candidate[2:]
        if candidate.strip('"') == wanted:
            return True
    return False


@app.get(_SCENE_URL)
async def _scene_endpoint(request: Request):
    """Serve the one exported glTF the page draws the 3D scene from.

    openarm-quest-teleop patch. One file by name, not a mounted
    directory: the export sits next to the run's dataflow, and that
    carries the pairing token. Declared before the static mount below,
    which matches every path.

    Served gzipped to any client that asks for it (every browser does),
    from the copy compressed at startup: 4.72 MB becomes 2.02 MB, which
    over the headset's Wi-Fi is the difference the operator waits through
    before the robot appears. The two representations carry different
    ETags because they are different bytes, and ``Vary`` tells any cache
    between here and the headset that the two are not interchangeable.
    """
    if _SCENE is None or _SCENE_GLB is None:
        return _error(404, "this node has no 3D scene")
    gzipped = (
        _SCENE_GZIP
        if _SCENE_GZIP is not None
        and _SCENE_GZIP["hash"] == _SCENE["hash"]
        and _accepts_gzip(request)
        else None
    )
    # The page checks the hash from the configuration message before it trusts
    # a cached copy, so a long cache is safe and saves a headset re-downloading
    # megabytes on every reconnect. Past that hour the browser revalidates, and
    # the ETag turns that into a 304 instead of the megabytes again.
    etag = f'"{_SCENE["hash"]}-gzip"' if gzipped else f'"{_SCENE["hash"]}"'
    headers = {
        "Cache-Control": "public, max-age=3600",
        "ETag": etag,
        "Vary": "Accept-Encoding",
    }
    if _matches_etag(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    if gzipped is None:
        return FileResponse(_SCENE_GLB, media_type="model/gltf-binary", headers=headers)
    return Response(
        gzipped["body"],
        media_type="model/gltf-binary",
        headers={**headers, "Content-Encoding": "gzip"},
    )


base_dir = os.path.dirname(__file__)
app.mount("/", StaticFiles(directory=f"{base_dir}/static", html=True), name="static")


def _serving():
    """Whether the node should keep running, in either signaling mode.

    In WebRTC-only mode the one browser leaving ends the session: with
    no HTTP server, no other browser can ever take its place.
    """
    if not _running:
        return False
    return webrtc_server is None or webrtc_server.running


def _stop():
    """Stop both halves, whichever noticed first."""
    global _running
    _running = False
    if server is not None:
        server.should_exit = True


def _send_quit_command():
    """Publish ``"quit"`` on the ``command`` output, if --quit-button is on.

    Sent when the dora loop ends rather than when the button goes
    down, so that every ending says it -- the button, the one browser
    leaving, an error -- and a dora-openarm-quitter node wired to this
    output ends the rest of the dataflow whichever of those it was. A
    node that died of an error and said nothing would leave the
    downstream nodes running with nobody driving them.

    And sent right then, before the WebRTC peers and the Web server
    are torn down, because dora only carries outputs from a node it
    still considers running: waiting until the process is on its way
    out is waiting until nobody can hear it.

    Gated on --quit-button for the usual reason things are off here by
    default: ending the whole dataflow is a thing the operator asks
    for, and a dataflow that never asked keeps its own idea of when to
    stop.
    """
    if not _QUIT_BUTTONS:
        return
    try:
        node.send_output(
            "command",
            pa.array(["quit"]),
            {"timestamp": time.time_ns()},
        )
    except Exception as error:  # noqa: BLE001
        # A dataflow already torn down can refuse the send, and the
        # quitter is gone with it: nothing is left to tell, so this must
        # not replace whatever error is already on its way out.
        print(f"cannot send quit command: {error}", flush=True)


async def _main_uvicorn():
    await server.serve()
    _stop()


async def _main_dora():
    try:
        while _serving():
            if node.is_empty():
                # Keep WebRTC responsive; the timeout lets shutdown flags be checked.
                event = await asyncio.to_thread(node.next, 0.1)
            else:
                event = node.next()
            # None is the event stream closing under us, which is how a
            # dataflow being torn down can look from here: treat it like
            # STOP rather than crashing out of the loop with _stop() unrun.
            if event is None or event["type"] == "STOP":
                break
            if event["type"] == "INPUT" and event["id"] == "tick":
                _publish_poses(_state)
            elif event["type"] == "INPUT" and event["id"] == "hud":
                _on_hud(event["value"])
            elif event["type"] == "INPUT" and event["id"] == "scene_state":
                _on_scene_state(event["value"])
            elif event["type"] == "INPUT" and event["id"] == "control":
                # Ending here takes the same way out as STOP: the finally
                # below, then the peers closed and uvicorn stopped by the
                # caller. Every other command (reset) is not this node's.
                if _control_command(event["value"]) == _QUIT:
                    print("quit received on the control input", flush=True)
                    break
            else:
                video.handle_event(event)
            # Service WebRTC even when the Dora input queue stays busy.
            await asyncio.sleep(0)
    finally:
        _send_quit_command()
        _stop()


async def _main_hosted():
    """Serve the page over HTTPS and answer offers posted to it."""
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        ssl_keyfile=args.tls_key_file,
        ssl_certfile=args.tls_certificate_file,
        log_level="info",
        # The headset's browser keeps idle keep-alive sockets open, and
        # uvicorn's graceful shutdown would wait on them forever ("Waiting
        # for connections to close"). Give it a moment, then go.
        timeout_graceful_shutdown=3,
    )
    global server
    server = uvicorn.Server(config)

    task_uvicorn = asyncio.create_task(_main_uvicorn())
    task_dora = asyncio.create_task(_main_dora())

    await task_dora
    # The peers are closed before uvicorn is waited on: its graceful
    # shutdown can sit on a browser's idle keep-alive sockets, and the
    # headset must hear the close -- and end its session -- without
    # waiting behind that.
    await webrtc_server.close()
    await task_uvicorn


async def _main_webrtc_only():
    """Run the single peer whose offer was handed in at startup.

    This is a one-shot connection: the offer is fixed at startup, so the
    node runs that one peer for its whole life and reconnecting means
    restarting the node.
    """
    task_dora = asyncio.create_task(_main_dora())
    try:
        await webrtc_server.negotiate_oneshot(
            args.offer,
            args.answer_host,
            args.answer_port,
            args.connect_timeout,
        )
    except (OSError, RuntimeError) as error:
        # Nobody applied the answer, or the media path never came up.
        # Exiting lets a supervisor restart us for a fresh offer.
        print(f"WebRTC-only mode failed: {error}", flush=True)
        _stop()
    await task_dora
    await webrtc_server.close()


async def _main_async():
    global webrtc_server, _state
    _state = _ConnectionState()
    # The first pose window starts when the node starts serving, not when the
    # module was imported: whatever is between the two is not a measurement.
    _frames.start(time.perf_counter())
    webrtc_server = webrtc.WebRTCServer(
        on_frame=_on_frame,
        on_session_start=_on_session_start,
        on_session_end=_on_session_end,
        # openarm-quest-teleop patch: the headset's own log, when it can
        # still reach us the reliable way, and which build of the page this
        # node is serving, so the page can tell a cached one from this one.
        on_log=_on_quest_log,
        build=_BUILD,
        calibration_enabled=_CALIBRATION_ENABLED,
        ice_servers=_ICE_SERVERS,
        scene=_SCENE,
        # openarm-quest-teleop patch: no ``state`` channel where nothing
        # would read one. See video.draws_scene.
        state_channel=_FORWARD_SCENE_STATE,
    )
    try:
        if args.offer:
            await _main_webrtc_only()
        else:
            await _main_hosted()
    finally:
        # Says what the headset reported, whichever way the node ends, so the
        # number is in the run's dora log next to the pose line.
        if _quest_log is not None:
            _quest_log.close()


def _answer_port_default():
    """Return the answer port from ANSWER_PORT, or None if it is unset."""
    raw = os.getenv("ANSWER_PORT")
    if raw is None:
        return None
    return int(raw)


def _quit_buttons_from_environment():
    """Return the quit buttons from QUIT_BUTTON, or [] if it is unset.

    Comma-separated, so the environment can name several buttons the way
    repeating --quit-button does.
    """
    raw = os.getenv("QUIT_BUTTON")
    if raw is None:
        return []
    return [button.strip() for button in raw.split(",") if button.strip()]


def _environment_flag(name):
    """Read an on/off option's environment variable.

    Anything but a plain negative turns it on, so that a dataflow YAML
    can carry ``CALIBRATION: false`` and have it mean off.
    """
    return os.getenv(name, "").strip().lower() not in ("", "0", "false", "no")


def build_parser():
    """Return the node's argument parser (patch 20).

    Separate from main() so a dataflow checker can read this node's
    arguments exactly as the node will, defaults from the environment
    included, without starting it.
    """
    parser = argparse.ArgumentParser(description="WebXR server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8443")),
        help="Server port (default: 8443)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("HOST", "0.0.0.0"),
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--tls-certificate-file",
        type=pathlib.Path,
        default=os.getenv("TLS_CERTIFICATE_FILE"),
        help="TLS certificate file. Required unless --offer is given",
    )
    parser.add_argument(
        "--tls-key-file",
        type=pathlib.Path,
        default=os.getenv("TLS_KEY_FILE"),
        help=(
            "TLS key file for the certificate file. Required unless --offer is given"
        ),
    )
    parser.add_argument(
        "--offer",
        type=str,
        default=os.getenv("OFFER"),
        help=(
            "Browser SDP offer, which runs the node as a pure WebRTC peer "
            "with no HTTP server: another service hosts the page and brokers "
            "signaling, so no TLS certificate is needed. --host, --port and "
            "the TLS options are ignored"
        ),
    )
    parser.add_argument(
        "--answer-host",
        type=str,
        default=os.getenv("ANSWER_HOST", "127.0.0.1"),
        help="Host to write the SDP answer to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--answer-port",
        type=int,
        default=_answer_port_default(),
        help="Port to write the SDP answer to",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=float(os.getenv("CONNECT_TIMEOUT", "60")),
        help="Seconds to wait for the browser to connect (default: 60)",
    )
    parser.add_argument(
        "--ice-servers",
        type=str,
        default=os.getenv("ICE_SERVERS"),
        help=(
            "ICE servers to build every peer with, as JSON in the form a "
            "browser RTCPeerConnection takes: "
            '[{"urls": [...], "username": ..., "credential": ...}]. '
            "This is how a signaling service passes the short-lived TURN "
            "credentials it mints, for networks where a direct path cannot "
            "be counted on. On one LAN pass '[]': host candidates connect, "
            "and an unreachable STUN server delays every answer by 5 s "
            "(default: a public STUN server only)"
        ),
    )
    parser.add_argument(
        "--quit-button",
        action="append",
        choices=list(_BUTTONS),
        help=(
            "Controller button that shuts the node down when pressed. May be "
            "repeated for several buttons; the QUIT_BUTTON environment "
            "variable takes a comma-separated list (default: none)"
        ),
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        default=_environment_flag("CALIBRATION"),
        help="Measure the neck pivot with the Y button (default: off)",
    )
    parser.add_argument(
        "--neck-pivot-file",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("NECK_PIVOT_FILE", "neck_pivot.yaml")),
        help=(
            "YAML file a measured neck pivot offset is written to and read "
            "back from (default: neck_pivot.yaml)"
        ),
    )
    parser.add_argument(
        "--max-linear-speed",
        type=float,
        default=float(os.getenv("MAX_LINEAR_SPEED", "1.0")),
        help="Maximum hand pose translation speed in m/s; 0 disables it (default: 1.0).",
    )
    parser.add_argument(
        "--max-angular-speed",
        type=float,
        default=float(os.getenv("MAX_ANGULAR_SPEED", "6.0")),
        help="Maximum hand pose rotation speed in rad/s; 0 disables it (default: 6.0).",
    )
    parser.add_argument(
        "--pose-timeout",
        type=float,
        default=float(os.getenv("POSE_TIMEOUT", "0")),
        help="Pause hand pose output after this many seconds without a new frame; 0 disables it (default).",
    )
    # String defaults, so argparse applies type=float to the environment
    # value too and a malformed one is a usage error rather than a
    # traceback.
    parser.add_argument(
        "--one-euro-min-cutoff",
        type=float,
        default=os.getenv("ONE_EURO_MIN_CUTOFF", "2.0"),
        help=(
            "One Euro filter minimum cutoff in Hz: higher follows the hand with "
            "less lag and more jitter; must be positive (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--one-euro-beta",
        type=float,
        default=os.getenv("ONE_EURO_BETA", "0.04"),
        help=(
            "One Euro filter speed coefficient: higher cuts the lag of fast "
            "motion more (default: 0.04)."
        ),
    )
    parser.add_argument(
        "--smoother-reset-gap",
        type=float,
        default=os.getenv("SMOOTHER_RESET_GAP", "0.1"),
        help=(
            "Reset a hand's filter, instead of resuming from its last pose, "
            "after its input stopped for longer than this many seconds; must "
            "be positive (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--pose-gap-ms",
        type=float,
        default=float(os.getenv("POSE_GAP_MS", "100")),
        help=(
            "The freshness window downstream measures these frames against, "
            "in milliseconds. Changes no behaviour: it is the threshold the "
            "once-a-minute pose line counts gaps over, so that the count reads "
            "as the number of times a consumer could have gone stale rather "
            "than a number this node picked. `oqt run` sets it from the "
            "clutch's --stale-ms (default: 100)."
        ),
    )
    parser.add_argument(
        "--scene-glb",
        type=pathlib.Path,
        default=os.getenv("SCENE_GLB") or None,
        help=(
            "Binary glTF of the scene the sim is running, exported from the "
            "model it loaded. Served at /scene/scene.glb and announced in the "
            "configuration message with its hash, so the page can draw the "
            "robot in 3D over passthrough and refuse a stale cached copy. "
            "Without it (or if the file cannot be read) the page is told "
            "nothing and keeps to the camera panel (default: none)."
        ),
    )
    parser.add_argument(
        "--quest-log",
        type=pathlib.Path,
        default=os.getenv("QUEST_LOG") or None,
        help=(
            "File the headset's own log records are appended to, one JSON "
            "object per line. The page sends what went wrong in it -- uncaught "
            "errors, a lost WebGL context, a session that ended -- on the "
            "reliable data channel, or by POST /log when WebRTC is what broke. "
            "Error records are also mirrored to this node's stderr. Without it "
            "the node has no sink: POST /log answers 503 and the page keeps its "
            "own buffer (default: none). `oqt run` points it at the run "
            "directory."
        ),
    )
    parser.add_argument(
        "--pivot-mode",
        choices=list(_PIVOT_MODES),
        default=os.getenv("PIVOT_MODE", "live"),
        help=(
            "What hand positions are relative to: 'live' follows the neck "
            "pivot of every frame, 'latched' holds the pivot from the start "
            "of the session and after gaps, so leaning does not move the "
            "hands (default: live)."
        ),
    )
    video.add_arguments(parser)
    return parser


def main():
    """Run WebXR server."""
    parser = build_parser()
    global args
    args = parser.parse_args()
    for name in (
        "max_linear_speed",
        "max_angular_speed",
        "pose_timeout",
        "one_euro_beta",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    # Zero would never let the filtered pose move while the hand is still.
    if not np.isfinite(args.one_euro_min_cutoff) or args.one_euro_min_cutoff <= 0.0:
        parser.error("--one-euro-min-cutoff must be finite and positive")
    # Zero would reset the filters between every two frames.
    if not np.isfinite(args.smoother_reset_gap) or args.smoother_reset_gap <= 0.0:
        parser.error("--smoother-reset-gap must be finite and positive")
    # argparse never checks a default against choices.
    if args.pivot_mode not in _PIVOT_MODES:
        parser.error(
            f"PIVOT_MODE must be one of {', '.join(_PIVOT_MODES)}, "
            f"not {args.pivot_mode!r}"
        )
    # Only a label on a log line, so a nonsensical value must not stop a run
    # that would otherwise teleoperate fine; it falls back to the default.
    if not np.isfinite(args.pose_gap_ms) or args.pose_gap_ms <= 0.0:
        print(
            f"--pose-gap-ms must be finite and positive, not {args.pose_gap_ms!r}; "
            "reporting gaps over 100 ms",
            flush=True,
        )
        args.pose_gap_ms = 100.0
    global _MAX_LINEAR_SPEED, _MAX_ANGULAR_SPEED, _POSE_TIMEOUT, _POSE_GAP_S
    global _ONE_EURO_MIN_CUTOFF, _ONE_EURO_BETA
    global _SMOOTHER_RESET_GAP, _PIVOT_MODE, _PAIRING_TOKEN
    _POSE_GAP_S = args.pose_gap_ms / 1000.0
    _MAX_LINEAR_SPEED = args.max_linear_speed
    _MAX_ANGULAR_SPEED = args.max_angular_speed
    _POSE_TIMEOUT = args.pose_timeout
    _ONE_EURO_MIN_CUTOFF = args.one_euro_min_cutoff
    _ONE_EURO_BETA = args.one_euro_beta
    _SMOOTHER_RESET_GAP = args.smoother_reset_gap
    _PIVOT_MODE = args.pivot_mode
    # Surrounding whitespace is dropped, so a token read from a file with
    # its trailing newline still matches the one in the page's URL; an
    # empty value means unset, like ICE_SERVERS.
    _PAIRING_TOKEN = pairing_token_from_environment(os.environ)

    # argparse cannot express "required unless another option is given",
    # and WebXR only runs on an HTTPS page, so the certificate is required
    # of every mode that serves the page itself.
    if args.offer:
        if args.answer_port is None:
            parser.error("--answer-port is required with --offer")
    elif args.tls_certificate_file is None or args.tls_key_file is None:
        parser.error(
            "--tls-certificate-file and --tls-key-file are required unless "
            "--offer is given"
        )

    # Resolved here, not as the argparse default: with action="append",
    # command-line buttons would be added to an environment default
    # instead of replacing it. The values also need checking by hand,
    # because argparse never checks a default against choices.
    quit_buttons = args.quit_button
    if quit_buttons is None:
        quit_buttons = _quit_buttons_from_environment()
    for button in quit_buttons:
        if button not in _BUTTONS:
            parser.error(
                f"QUIT_BUTTON must name buttons among {', '.join(_BUTTONS)}, "
                f"not {button!r}"
            )
    if args.calibration and "y" in quit_buttons:
        parser.error(
            "--quit-button y conflicts with --calibration, which uses the Y "
            "button to measure the neck pivot"
        )

    # Parsed at startup so a malformed list is a startup error, not a
    # peer that quietly cannot connect. An empty value means unset, so a
    # dataflow YAML can carry ``ICE_SERVERS: ""`` and have it mean the
    # default.
    ice_servers = None
    if args.ice_servers:
        try:
            ice_servers = webrtc.parse_ice_servers(args.ice_servers)
        except (TypeError, ValueError) as error:
            parser.error(f"--ice-servers: {error}")

    video.configure(args)

    global _CALIBRATION_ENABLED, _ICE_SERVERS, _NECK_PIVOT_FILE, _QUIT_BUTTONS
    _CALIBRATION_ENABLED = args.calibration
    _ICE_SERVERS = ice_servers
    _NECK_PIVOT_FILE = args.neck_pivot_file
    _QUIT_BUTTONS = tuple(quit_buttons)

    # openarm-quest-teleop patch: the headset's own log. Opened before the
    # first peer can connect, and lazily: a node that never hears from a
    # headset leaves no empty file behind.
    global _quest_log, _BUILD
    _quest_log = _open_quest_log(args.quest_log)
    # Said before any page connects, so the run's log records which build of
    # the page was served -- the other half of what a page reports about
    # itself when a headset has been serving an old bundle from its cache.
    _BUILD = _build_identity()
    print(f"page build: {_BUILD}", flush=True)

    # Read once at startup, like the view configuration: a scene is written
    # before the dataflow starts and does not change under a running node.
    global _SCENE_GLB, _SCENE
    _SCENE = _configure_scene(args.scene_glb)
    _SCENE_GLB = args.scene_glb if _SCENE is not None else None

    # openarm-quest-teleop patch: whether this run will carry the 3D scene to
    # the headset at all. Both halves have to be true -- a view whose page
    # draws the scene, and an asset for it to draw -- because either one
    # missing leaves the page with no handler for the state channel, and the
    # stream would then be bytes on the radio that nothing reads. Said at
    # startup so the choice is in the log before a headset connects.
    global _FORWARD_SCENE_STATE
    _FORWARD_SCENE_STATE = video.draws_scene() and _SCENE is not None
    view_name = video.view_configuration().get("view")
    if _FORWARD_SCENE_STATE:
        print(
            f"scene_state: forwarding to the headset (view {view_name!r})", flush=True
        )
    elif video.draws_scene():
        print(
            f"scene_state: view {view_name!r} draws the 3D scene but there is no "
            "glTF to draw it from, so no state channel is offered and the stream "
            "is dropped here",
            flush=True,
        )
    else:
        print(
            f"scene_state: off (view {view_name!r} draws no 3D scene); no state "
            "channel is offered and nothing is forwarded to the headset",
            flush=True,
        )

    # Read once at startup; restart the dataflow to apply a change.
    pose_configuration = video.view_configuration().get("pose") or {}

    frame_offset = pose_configuration.get("frame_offset")
    if frame_offset is not None:
        global _FRAME_OFFSET_CELL
        _FRAME_OFFSET_CELL = np.array(frame_offset, dtype=np.float32).reshape(3)

    _configure_neck_pivot(pose_configuration, args.neck_pivot_file)

    global node
    node = dora.Node()

    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
