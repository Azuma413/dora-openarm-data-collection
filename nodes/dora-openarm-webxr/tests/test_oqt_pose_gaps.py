# openarm-quest-teleop addition to the vendored dora-openarm-webxr tests
# (Apache-2.0, like the rest of this directory).
#
# Tests for PATCHES.md 12 and 13: the One Euro filters are reset, not
# suspended, after an input gap longer than SMOOTHER_RESET_GAP, and
# PIVOT_MODE=latched makes hand positions relative to a neck pivot latched
# at the start of a session and after gaps, so head motion while gripping
# does not move the published hands.

import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dora_openarm_webxr import main
from dora_openarm_webxr.smoothing import OneEuroPoseSmoother

# The Quest's frame rate and the dataflow's pose tick.
FRAME_DT = 1.0 / 72.0
TICK_DT = 0.005

HEAD = (0.0, 1.6, 0.0)
HAND = (0.2, 1.1, -0.4)


def _webxr_pose(position, yaw_deg=0.0):
    q = Rotation.from_euler("y", yaw_deg, degrees=True).as_quat()
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "z": float(position[2]),
        "qx": float(q[0]),
        "qy": float(q[1]),
        "qz": float(q[2]),
        "qw": float(q[3]),
    }


def _frame(head=HEAD, hand=HAND, head_yaw=0.0, drop=()):
    """One page frame with the right controller, as ar.js sends it."""
    frame = {
        "type": "frame",
        "pose_reference": _webxr_pose(head, head_yaw),
        "pose_right": _webxr_pose(hand),
        "trigger_right": 0.0,
        "grip_right": 1.0,
    }
    for key in drop:
        frame.pop(key, None)
    return frame


def _raw_position(head=HEAD, hand=HAND, head_yaw=0.0):
    """The published position of an unfiltered hand, with the live pivot."""
    pose = main._adjust_pose(
        _webxr_pose(hand), _webxr_pose(head, head_yaw), OneEuroPoseSmoother(), 0.0
    )
    return np.asarray(pose.to_numpy(zero_copy_only=False))[:3]


class _RecordingNode:
    def __init__(self, clock):
        self._clock = clock
        self.outputs = []

    def send_output(self, output_id, value, metadata=None):
        self.outputs.append((self._clock[0], output_id, value))


class _Transport:
    def send_control(self, payload):
        pass


class _Rig:
    """Frames at 72 Hz and pose ticks at 200 Hz on a fake perf_counter."""

    def __init__(self, monkeypatch):
        self.clock = [100.0]
        self.node = _RecordingNode(self.clock)
        monkeypatch.setattr(main, "node", self.node)
        monkeypatch.setattr(main, "webrtc_server", _Transport())
        monkeypatch.setattr(main.time, "perf_counter", lambda: self.clock[0])
        self.state = main._ConnectionState()
        self.sequence = 0
        self._frame_index = None

    def run(self, seconds, frame):
        """Tick for ``seconds``; ``frame(t)`` gives each frame, or None for none.

        Returns the time of the first frame sent, or None.
        """
        first = None
        for _ in range(round(seconds / TICK_DT)):
            self.clock[0] += TICK_DT
            index = int(self.clock[0] / FRAME_DT)
            if index != self._frame_index:
                self._frame_index = index
                payload = frame(self.clock[0])
                if payload is not None:
                    self.sequence += 1
                    main._process_frame(
                        {**payload, "sequence": self.sequence}, self.state
                    )
                    first = self.clock[0] if first is None else first
            main._publish_poses(self.state)
        return first

    def positions(self, since=None):
        """Published right-hand positions, from the tick at ``since`` on."""
        return [
            np.asarray(value.to_pylist()[0]["pose"][:3])
            for t, output_id, value in self.node.outputs
            if output_id == "pose_right" and (since is None or t >= since)
        ]


@pytest.fixture
def rig(monkeypatch):
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.0)
    # raising=False, here and below, so that the scenarios also run (and
    # fail on behaviour) against a node without these patches.
    monkeypatch.setattr(main, "_SMOOTHER_RESET_GAP", 0.1, raising=False)
    monkeypatch.setattr(main, "_PIVOT_MODE", "live", raising=False)
    return _Rig(monkeypatch)


# Smoother reset after gaps (PATCHES.md 12)

MOVED = (HAND[0] + 0.3, HAND[1], HAND[2])  # 30 cm to the right, unseen


def _gap_frames(kind):
    return {
        # ar.js leaves out an untracked controller's pose, trigger and grip.
        "pose_missing": lambda t: _frame(
            hand=MOVED, drop=("pose_right", "trigger_right", "grip_right")
        ),
        # ... and an untracked head's pose_reference.
        "reference_missing": lambda t: _frame(hand=MOVED, drop=("pose_reference",)),
        "trigger_missing": lambda t: _frame(hand=MOVED, drop=("trigger_right",)),
        # A Wi-Fi stall or a recenter pause: no frames, ticks go on
        # publishing the last one.
        "frames_stopped": lambda t: None,
    }[kind]


@pytest.mark.parametrize(
    "kind", ["pose_missing", "reference_missing", "trigger_missing", "frames_stopped"]
)
def test_long_gap_resets_so_the_first_pose_back_is_the_hand(rig, kind):
    # CLUTCH-1: the hand comes back 30 cm from where it was lost. A clutch
    # that let go during the gap re-anchors on the first pose it sees, so
    # that pose must be the hand, not the start of a 1 m/s slide toward it.
    rig.run(1.0, lambda t: _frame())
    rig.run(0.15, _gap_frames(kind))
    returned_at = rig.run(0.05, lambda t: _frame(hand=MOVED))
    first_back = rig.positions(since=returned_at)[0]
    np.testing.assert_allclose(first_back, _raw_position(hand=MOVED), atol=1e-5)


def test_short_gap_still_suspends(rig):
    # Below the gap the clutch stays engaged, so a raw step would go
    # straight into its target: the filter resumes from its last pose.
    moved = (HAND[0] + 0.03, HAND[1], HAND[2])
    rig.run(1.0, lambda t: _frame())
    before = rig.positions()[-1]
    rig.run(
        0.05,
        lambda t: _frame(
            hand=moved, drop=("pose_right", "trigger_right", "grip_right")
        ),
    )
    returned_at = rig.run(0.05, lambda t: _frame(hand=moved))
    first_back = rig.positions(since=returned_at)[0]
    step = np.linalg.norm(first_back - before)
    assert 0.0 < step <= main._MAX_LINEAR_SPEED * TICK_DT + 1e-6


def test_pose_timeout_resets_the_smoother(rig, monkeypatch):
    # A pose timeout shorter than the reset gap still lets go of the
    # filter along with the frame.
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.05)
    rig.run(1.0, lambda t: _frame())
    rig.run(0.07, lambda t: None)
    assert rig.state.latest_frame is None
    assert rig.state.smoothers["right"].p_prev is None
    returned_at = rig.run(0.05, lambda t: _frame(hand=MOVED))
    first_back = rig.positions(since=returned_at)[0]
    np.testing.assert_allclose(first_back, _raw_position(hand=MOVED), atol=1e-5)


def test_stall_between_pose_timeout_and_reset_gap_resets(rig, monkeypatch):
    # CLUTCH-1 residual (P1): with the flow's POSE_TIMEOUT 0.1 <= the
    # clutch's fresh_s 0.1 < SMOOTHER_RESET_GAP 0.12, a 105 ms frame stall is
    # shorter than the reset gap, but the timeout has let go of the frame
    # (so the clutch saw the pose stop and will settle), and the filter must
    # go with it: the first pose back is the hand, not a slide toward it.
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.1)
    monkeypatch.setattr(main, "_SMOOTHER_RESET_GAP", 0.12, raising=False)
    rig.run(1.0, lambda t: _frame())
    before = rig.positions()[-1]
    last_frame_at = rig.state.received_at
    stall_ends = last_frame_at + 0.105
    while rig.clock[0] + TICK_DT < stall_ends:
        rig.clock[0] += TICK_DT
        main._publish_poses(rig.state)
    assert rig.state.latest_frame is None  # the timeout let go of the frame
    rig.clock[0] = stall_ends
    rig.sequence += 1
    main._process_frame({**_frame(hand=MOVED), "sequence": rig.sequence}, rig.state)
    main._publish_poses(rig.state)
    first_back = rig.positions(since=stall_ends)[0]
    np.testing.assert_allclose(first_back, _raw_position(hand=MOVED), atol=1e-5)
    assert np.linalg.norm(first_back - before) > 0.29


def test_stall_past_pose_timeout_resets_even_if_no_tick_saw_it(rig, monkeypatch):
    # The clutch times a grip gap from sample to sample, so a 101 ms stall
    # disarms it even though no 5 ms pose tick fell between the timeout and
    # the frame that ended the stall. The filter must be reset all the same,
    # or the pose it re-anchors on after the stall is a slide.
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.1)
    monkeypatch.setattr(main, "_SMOOTHER_RESET_GAP", 0.12, raising=False)
    rig.run(1.0, lambda t: _frame())
    last_frame_at = rig.state.received_at
    rig.clock[0] = last_frame_at + 0.0975  # the last tick before the timeout
    main._publish_poses(rig.state)
    assert rig.state.latest_frame is not None
    rig.clock[0] = last_frame_at + 0.101  # the frame, between two ticks
    rig.sequence += 1
    main._process_frame({**_frame(hand=MOVED), "sequence": rig.sequence}, rig.state)
    rig.clock[0] = last_frame_at + 0.1025
    main._publish_poses(rig.state)
    first_back = rig.positions(since=rig.clock[0])[0]
    np.testing.assert_allclose(first_back, _raw_position(hand=MOVED), atol=1e-5)


def test_frames_within_pose_timeout_do_not_reset(rig, monkeypatch):
    # Frames 90 ms apart are under the timeout: the filter keeps going.
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.1)
    monkeypatch.setattr(main, "_SMOOTHER_RESET_GAP", 0.12, raising=False)
    rig.run(1.0, lambda t: _frame())
    before = rig.positions()[-1]
    last_frame_at = rig.state.received_at
    rig.clock[0] = last_frame_at + 0.09
    rig.sequence += 1
    main._process_frame({**_frame(hand=MOVED), "sequence": rig.sequence}, rig.state)
    main._publish_poses(rig.state)
    first_back = rig.positions(since=rig.clock[0])[0]
    # Resumed toward the hand at the speed cap, not stepped onto it.
    assert np.linalg.norm(first_back - before) < 0.1
    assert np.linalg.norm(first_back - _raw_position(hand=MOVED)) > 0.2


def test_session_end_resets_the_smoothers(rig, monkeypatch):
    rig.run(0.5, lambda t: _frame())
    monkeypatch.setattr(main, "_state", rig.state)
    main._on_session_end()
    for smoother in rig.state.smoothers.values():
        assert smoother.p_prev is None
        assert smoother.t_prev is None


def test_reset_gap_is_configurable(rig, monkeypatch):
    # With SMOOTHER_RESET_GAP=0.3 a 150 ms gap is a short one.
    monkeypatch.setattr(main, "_SMOOTHER_RESET_GAP", 0.3, raising=False)
    rig.run(1.0, lambda t: _frame())
    before = rig.positions()[-1]
    rig.run(0.15, _gap_frames("pose_missing"))
    returned_at = rig.run(0.05, lambda t: _frame(hand=MOVED))
    first_back = rig.positions(since=returned_at)[0]
    assert (
        np.linalg.norm(first_back - before) <= main._MAX_LINEAR_SPEED * TICK_DT + 1e-6
    )


# Latched neck pivot (PATCHES.md 13)


def _lean_and_turn(rig):
    """Hold still, then lean 15 cm forward and turn the head 60 degrees."""
    rig.run(0.5, lambda t: _frame())
    start = rig.positions()[-1]
    t0 = rig.clock[0]
    rig.run(1.0, lambda t: _frame(head=(0.0, 1.6, -0.15 * min(1.0, (t - t0) / 0.8))))
    t1 = rig.clock[0]
    rig.run(
        0.6,
        lambda t: _frame(
            head=(0.0, 1.6, -0.15), head_yaw=60.0 * min(1.0, (t - t1) / 0.5)
        ),
    )
    return start, rig.positions(since=t0)


def test_latched_pivot_ignores_head_motion(rig, monkeypatch):
    # FRAME-1: with the controller still in the room, leaning in to look
    # at the arm must not move the target.
    monkeypatch.setattr(main, "_PIVOT_MODE", "latched", raising=False)
    start, positions = _lean_and_turn(rig)
    drift = max(np.linalg.norm(position - start) for position in positions)
    assert drift < 1e-5


def test_live_pivot_still_follows_the_body(rig):
    # The upstream default is unchanged: the lean moves the hand 15 cm.
    start, positions = _lean_and_turn(rig)
    moved = np.linalg.norm(positions[-1] - start)
    assert moved > 0.14


def test_latched_pivot_relatches_after_a_gap(rig, monkeypatch):
    monkeypatch.setattr(main, "_PIVOT_MODE", "latched", raising=False)
    rig.run(0.5, lambda t: _frame())
    start = rig.positions()[-1]
    leaned = (0.0, 1.6, -0.15)

    # The operator leans during a gap the clutch lets go on: the pivot is
    # latched again from the first head pose after it.
    rig.run(0.15, lambda t: None)
    returned_at = rig.run(0.2, lambda t: _frame(head=leaned))
    after_gap = rig.positions(since=returned_at)
    np.testing.assert_allclose(after_gap[0], _raw_position(head=leaned), atol=1e-5)
    assert np.linalg.norm(after_gap[0] - start) > 0.14

    # A gap shorter than SMOOTHER_RESET_GAP keeps the clutch engaged, so
    # it keeps the pivot too: leaning back changes nothing.
    rig.run(0.05, lambda t: None)
    returned_at = rig.run(0.2, lambda t: _frame(head=HEAD))
    for position in rig.positions(since=returned_at):
        np.testing.assert_allclose(position, after_gap[-1], atol=1e-5)


def test_latched_pivot_relatches_after_a_recenter(rig, monkeypatch):
    # A recenter moves the local space under both poses; the page pauses
    # its frames for 300 ms, and the pivot is latched again in the new
    # space instead of mixing it with the old one.
    monkeypatch.setattr(main, "_PIVOT_MODE", "latched", raising=False)
    rig.run(0.5, lambda t: _frame())
    rig.run(0.3, lambda t: None)
    head, hand = (0.4, 1.5, 0.3), (0.5, 1.0, -0.1)
    returned_at = rig.run(0.1, lambda t: _frame(head=head, hand=hand, head_yaw=90.0))
    np.testing.assert_allclose(
        rig.positions(since=returned_at)[0],
        _raw_position(head=head, hand=hand, head_yaw=90.0),
        atol=1e-5,
    )


def test_new_session_latches_again(rig, monkeypatch):
    monkeypatch.setattr(main, "_PIVOT_MODE", "latched", raising=False)
    monkeypatch.setattr(main, "_state", rig.state)
    rig.run(0.3, lambda t: _frame())
    assert rig.state.latched_pivot is not None
    main._on_session_start()
    assert main._state.latched_pivot is None


# The once-a-minute pose line (the LOST finding)
#
# The real 7-minute session of 2026-09-16 recorded 106 entries into the
# clutch's stale state and not one line saying why. Every candidate cause --
# bursty Wi-Fi loss, the page's own frame rate, a controller out of the
# cameras' sight, the unordered channel delivering a frame late, the node's own
# pose timeout -- reaches the log as the same silence. These tests pin the line
# that separates them.


@pytest.fixture
def frames(rig, monkeypatch):
    """A fresh window on the rig's fake clock, counted against a 100 ms window."""
    stats = main._FrameStats(rig.clock[0])
    monkeypatch.setattr(main, "_frames", stats, raising=False)
    monkeypatch.setattr(main, "_POSE_GAP_S", 0.1, raising=False)
    return stats


def _both(drop=(), **kwargs):
    """One frame carrying BOTH controllers, as the page sends with both tracked.

    ``_frame`` above carries only the right one, which the node would rightly
    report as a left hand the page left out of every frame.
    """
    frame = _frame(**kwargs)
    frame["pose_left"] = frame["pose_right"]
    frame["trigger_left"] = 0.0
    frame["grip_left"] = 1.0
    for key in drop:
        frame.pop(key, None)
    return frame


def _pose_lines(capsys):
    return [t for t in capsys.readouterr().out.splitlines() if t.startswith("poses:")]


def _pose_line(capsys):
    lines = _pose_lines(capsys)
    assert len(lines) == 1, lines
    return lines[0]


def _gaps(line):
    """(p50, p90, p99, max, over, window) out of the line's gap clause."""
    match = re.search(
        r"gap p50 ([\d.]+) p90 ([\d.]+) p99 ([\d.]+) max (\d+) ms, (\d+) over ([\d.]+) ms",
        line,
    )
    assert match, line
    return [float(value) for value in match.groups()]


def test_the_minute_line_separates_a_healthy_median_from_the_tail(rig, frames, capsys):
    """The discriminator the investigation had to reconstruct from a state-change log.

    A burst of lost frames on the radio opens one hole longer than the clutch's
    freshness window while leaving the median at a perfectly healthy 13.9 ms.
    Any mean-rate metric -- the existing 5 s TickMonitor included -- shows
    nothing at all; the maximum and the over-the-window count are the reading.
    """

    def frame(t):
        # One burst of nine frames, the shape a Wi-Fi retry exhaustion has.
        return None if 130.0 <= t < 130.125 else _both()

    rig.run(61.0, frame)
    line = _pose_line(capsys)
    p50, p90, p99, worst, over, window = _gaps(line)
    assert window == 100.0  # the clutch's own --stale-ms, not a number of ours
    assert 13.0 <= p50 <= 15.0 and 13.0 <= p90 <= 15.0
    assert worst >= 125.0  # the burst, which is what actually let the arm go
    assert over == 1
    assert p99 < 100.0  # even the 99th percentile says nothing is wrong
    # The frames themselves are healthy: about 72 a second for a minute.
    count = int(re.search(r"poses: (\d+) in ", line).group(1))
    assert 4300 <= count <= 4400
    assert re.search(r"\(7[0-9.]+ Hz\)", line)
    assert "dropped none" in line


def test_a_frame_delivered_late_is_counted_rather_than_silently_dropped(
    rig, frames, capsys
):
    # main._process_frame returns early on `sequence <= last_sequence`, which
    # throws away the frame's grip along with its pose. It used to say nothing.
    rig.run(1.0, lambda t: _both())
    for _ in range(3):
        main._process_frame({**_both(), "sequence": 1}, rig.state)
    rig.run(60.0, lambda t: _both())
    assert "3 out of sequence" in _pose_line(capsys)


def test_a_hand_the_page_saw_as_untracked_is_named_per_side(rig, frames, capsys):
    """78 % of the real session's stale entries took BOTH hands on the same line.

    One frame carries both, so whole-frame loss explains those; the 22 %
    minority that took one hand is this -- ar.js withholds an untracked
    controller's pose, trigger and grip. Reported per side, because a link gap
    and a controller out of the cameras' view are different faults.
    """

    def frame(t):
        return _both(drop=("pose_right", "trigger_right")) if t < 130.0 else _both()

    rig.run(61.0, frame)
    line = _pose_line(capsys)
    dropped = int(re.search(r"(\d+) right untracked", line).group(1))
    assert 2000 <= dropped <= 2200  # ~30 s at 72 Hz
    assert "left untracked" not in line


def test_a_frame_with_no_head_pose_is_counted(rig, frames, capsys):
    # Hands are only published with a head pose, so such a frame publishes
    # nothing at all -- and that is not the same fault as a missing hand.
    rig.run(61.0, lambda t: _both(drop=("pose_reference",)))
    line = _pose_line(capsys)
    count = int(re.search(r"(\d+) with no head pose", line).group(1))
    assert 4300 <= count <= 4400
    # A frame with no head pose says nothing about the hands either way.
    assert "untracked" not in line


def test_the_nodes_own_pose_timeout_is_counted_once_per_dropout(
    rig, frames, monkeypatch, capsys
):
    # Every tick of one silence is the same dropout; counting them would
    # report the tick rate instead of the number of times the hands were let
    # go of, which is the number that matters.
    monkeypatch.setattr(main, "_POSE_TIMEOUT", 0.09)
    rig.run(1.0, lambda t: _both())
    rig.run(1.0, lambda t: None)  # 200 ticks, one dropout
    rig.run(1.0, lambda t: _both())
    rig.run(1.0, lambda t: None)
    rig.run(57.0, lambda t: _both())
    assert "2 timed out" in _pose_line(capsys)


def test_the_line_still_arrives_when_the_headset_is_what_stopped(rig, frames, capsys):
    """The tail of the real session: 22 minutes of silence and no line about it.

    Printed on the pose tick, not on a frame, so a page that stopped sending
    says so once a minute rather than leaving the log to the scene stream.
    """
    rig.run(61.0, lambda t: _both())
    capsys.readouterr()
    # Two windows: the first still holds the frames that arrived between the
    # line above and the last one sent, the second is the silence itself.
    rig.run(122.0, lambda t: None)
    line = _pose_lines(capsys)[-1]
    assert line.startswith("poses: 0 in ")
    assert "Hz" not in line and "gap p50" not in line


def test_the_gap_clock_does_not_span_a_session(rig, frames, monkeypatch, capsys):
    # A headset reconnecting after a pause must not be reported as one huge
    # gap: that would be a fault nobody had.
    monkeypatch.setattr(main, "_state", rig.state)
    rig.run(1.0, lambda t: _both())
    rig.run(20.0, lambda t: None)
    main._on_session_start()
    rig.state = main._state
    rig.run(40.0, lambda t: _both())
    _, _, _, worst, over, _ = _gaps(_pose_line(capsys))
    assert worst < 100.0 and over == 0


def test_the_first_window_starts_when_the_node_does(monkeypatch):
    """The first real run reported `poses: 1 in 3026528.4 s; (0.0 Hz)`.

    ``_FrameStats`` had started its window at 0.0 on a perf_counter clock, so
    the first frame was already a machine-uptime old and the line was due at
    once -- with a rate of 0.0 Hz, which is exactly the reading the line exists
    to make trustworthy.
    """
    monkeypatch.setattr(main.time, "perf_counter", lambda: 3_000_000.0)
    stats = main._FrameStats()
    assert stats.started_at == 3_000_000.0
    assert not stats.due(3_000_000.0)
    stats.accept(_both(), 3_000_000.0)
    assert not stats.due(3_000_030.0)
    assert stats.due(3_000_061.0)
    # And the line reads as one clause, not "in 60.0 s; (72.0 Hz)".
    for index in range(1, 4320):
        stats.accept(_both(), 3_000_000.0 + index / 72.0)
    line = stats.line(3_000_060.0, 0.1)
    assert line.startswith("poses: 4320 in 60.0 s (72.0 Hz); gap p50 "), line
