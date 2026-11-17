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

"""Head camera video downlink for the WebXR front-end.

Takes the JPEG images of the robot's head camera and forwards them to
the VR device, so the operator can see the robot's workspace while
teleoperating. The image is drawn on a panel fixed in the room.

Frames leave on their own WebRTC video track, one per eye, so they never
delay the pose messages that feed IK. This module only keeps the newest
frame per eye; :mod:`.webrtc` owns the tracks that encode them. How the
panel is placed is tuned in ``example/view_camera.yaml``.
"""

import argparse
import asyncio
import os
import pathlib

import yaml

# dora-rs input IDs mapped to the eye that the frame is rendered on.
# The default mono view uses only the right eye; the stereo view uses
# both.
CAMERA_INPUTS = {
    "camera_head_left": "left",
    "camera_head_right": "right",
    "camera_ceiling": "main",
    "camera_wrist_left": "arm_left",
    "camera_wrist_right": "arm_right",
}

# Used when no --view-configuration-file is given.
DEFAULT_VIEW_CONFIGURATION: dict = {
    "view": "mono",
    "session": {"mode": "immersive-ar"},
    "panel": {"lock": "room", "distance": 1.3, "width": 1.5},
}

TRACKS = ("left", "right", "main", "arm_left", "arm_right")

_frames: dict = {track: None for track in TRACKS}
# Incremented on every frame so that a track can tell a new frame from a
# repeated one and always encode the most recent one.
_sequences: dict = {track: 0 for track in TRACKS}
# One event per eye, because each eye has its own track waiting on it.
# A shared event would need every waiter to agree on when to clear it.
_events: dict = {track: asyncio.Event() for track in TRACKS}

_view_configuration: dict = DEFAULT_VIEW_CONFIGURATION


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the head camera options to the node's argument parser."""
    parser.add_argument(
        "--view-configuration-file",
        type=pathlib.Path,
        default=os.getenv("VIEW_CONFIGURATION_FILE"),
        help="YAML file with the head camera panel parameters",
    )


def configure(args: argparse.Namespace) -> None:
    """Read the view configuration at startup if a file was given.

    Read once; restart the dataflow to apply a change.
    """
    path = getattr(args, "view_configuration_file", None)
    if path is None:
        return
    global _view_configuration
    try:
        with open(path, encoding="utf-8") as input:
            _view_configuration = yaml.safe_load(input)
    except (OSError, yaml.YAMLError) as error:
        # Keep the default so a broken file cannot stop the node.
        print(f"cannot read {path}: {error}", flush=True)


def view_configuration() -> dict:
    """Return the view configuration read at startup."""
    return _view_configuration


# openarm-quest-teleop patch: the views whose page draws the simulated scene as
# real objects around the operator, and so registers a handler for the ``state``
# data channel. Kept in step with SCENE_VIEWS in ``static/ar.js``, which is where
# the page decides the same thing; a view that is not here never reads a scene
# state message, and the node must not spend the headset's link sending one.
SCENE_VIEWS = ("scene", "scene+panel")


def draws_scene() -> bool:
    """Whether this view draws the 3D scene (openarm-quest-teleop patch).

    The page only calls ``connection.onState`` from a 3D view (``ar.js``
    ``SCENE_VIEWS`` -> ``openSceneView``); anywhere else the ``state`` channel's
    messages are received, decoded and thrown away. The bytes are still on the
    air, though, and on the half-duplex 2.4 GHz link a headset usually sits on
    they take their contention cycles ahead of the hand poses coming back --
    which is what turns a 72 Hz stream nobody draws into an arm that stops
    following. So this is what gates both the channel and the forwarding.
    """
    return _view_configuration.get("view") in SCENE_VIEWS


def eyes() -> list:
    """Return the camera tracks this view draws, in negotiation order.

    Every view includes the main overview and both arm cameras. ``stereo``
    adds both head cameras, while ``mono`` and ``scene+panel`` add the right
    head camera. The order is fixed because the browser tells tracks apart
    by negotiation order.
    """
    view = _view_configuration.get("view")
    auxiliary = ["main", "arm_left", "arm_right"]
    if view in {"none", "scene"}:
        return auxiliary
    if view == "stereo":
        return ["left", "right", *auxiliary]
    return ["right", *auxiliary]


def handle_event(event) -> bool:
    """Store a camera frame. Return whether the event was ours."""
    if event["type"] != "INPUT" or event["id"] not in CAMERA_INPUTS:
        return False
    eye = CAMERA_INPUTS[event["id"]]
    # The camera node sends JPEG data as a uint8 array.
    _frames[eye] = event["value"].to_numpy(zero_copy_only=False).tobytes()
    _sequences[eye] += 1
    _events[eye].set()
    return True


async def wait_next(eye: str, seen_sequence: int) -> tuple:
    """Wait for a frame of ``eye`` newer than ``seen_sequence``, return it.

    Returns the JPEG payload and its sequence number. A slow encoder
    skips ahead to the newest frame rather than building a queue, which
    is the right trade for teleoperation video.
    """
    while _sequences[eye] == seen_sequence:
        _events[eye].clear()
        await _events[eye].wait()
    return _frames[eye], _sequences[eye]


def reset() -> None:
    """Forget every stored frame. For tests, which reuse the module."""
    for eye in _frames:
        _frames[eye] = None
        _sequences[eye] = 0
        _events[eye] = asyncio.Event()
