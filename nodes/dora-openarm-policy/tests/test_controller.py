import numpy as np
import pytest

from dora_openarm_policy.controller import (
    PASSTHROUGH,
    RETURNING,
    RUNNING,
    STOPPED,
    Controller,
    ControllerParams,
)


def ready(ctl, cmd=0.0, ik=0.0):
    for side in ("right", "left"):
        ctl.on_cmd(side, np.full(8, cmd))
        ctl.on_ik(side, np.full(8, ik))


def chunk(start, step=0.01, n=30):
    return np.array([np.full(16, start + k * step) for k in range(n)])


def test_passthrough_sends_nothing_itself():
    ctl = Controller()
    ready(ctl)
    assert ctl.mode == PASSTHROUGH and ctl.step(0.0) is None


def test_running_holds_until_the_first_chunk_then_blends_into_it():
    ctl = Controller(ControllerParams(start_blend_s=0.3))
    ready(ctl, cmd=0.5)
    assert ctl.start(10.0) is None and ctl.mode == RUNNING
    np.testing.assert_allclose(ctl.step(10.01)["right"], 0.5)
    assert ctl.on_chunk(chunk(1.0), t_obs=10.0, run_id=ctl.run_id, now=10.05)
    # Right at the hand-over the target is still where it held.
    np.testing.assert_allclose(ctl.step(10.05)["left"], 0.5, atol=1e-9)
    # Halfway through the blend, between the two.
    mid = ctl.step(10.2)["right"][0]
    assert 0.5 < mid < 1.06
    # After the blend: the chunk at now, interpolated (step k at t_obs + k/30).
    out = ctl.step(10.5)["right"][0]
    assert out == pytest.approx(1.0 + 0.5 * 30 * 0.01)


def test_a_new_chunk_is_crossfaded_and_a_stale_one_ignored():
    ctl = Controller(ControllerParams(blend_s=0.1, start_blend_s=0.0))
    ready(ctl)
    ctl.start(0.0)
    ctl.on_chunk(chunk(0.0), 0.0, ctl.run_id, 0.0)
    before = ctl.step(0.3)["right"][0]
    assert ctl.on_chunk(chunk(1.0), 0.3, ctl.run_id, 0.3)
    assert ctl.step(0.3)["right"][0] == pytest.approx(before)
    assert ctl.step(0.45)["right"][0] == pytest.approx(1.0 + 0.15 * 30 * 0.01)
    assert not ctl.on_chunk(chunk(5.0), 0.4, ctl.run_id - 1, 0.5)


def test_the_chunk_end_is_held():
    ctl = Controller(ControllerParams(start_blend_s=0.0))
    ready(ctl)
    ctl.start(0.0)
    ctl.on_chunk(chunk(0.0), 0.0, ctl.run_id, 0.0)
    assert ctl.step(5.0)["right"][0] == pytest.approx(0.29)
    assert ctl.chunk_left_s(5.0) == 0.0


def test_stop_holds_the_gate_command_and_drops_late_chunks():
    ctl = Controller(ControllerParams(start_blend_s=0.0))
    ready(ctl)
    ctl.start(0.0)
    run = ctl.run_id
    ctl.on_chunk(chunk(0.0, step=0.1), 0.0, run, 0.0)
    ctl.step(0.5)
    for side in ("right", "left"):
        ctl.on_cmd(side, np.full(8, 0.7))  # the gate is still catching up
    ctl.stop(0.5, "UI")
    assert ctl.mode == STOPPED and ctl.reason == "UI"
    np.testing.assert_allclose(ctl.step(1.0)["left"], 0.7)
    assert not ctl.on_chunk(chunk(3.0), 0.6, run, 0.6)
    np.testing.assert_allclose(ctl.step(2.0)["right"], 0.7)


def test_go_back_eases_to_the_ik_then_passes_through():
    ctl = Controller(ControllerParams(return_speed=0.5, min_return_s=0.5))
    ready(ctl, cmd=1.0, ik=0.0)
    ctl.start(0.0)
    ctl.stop(0.1)
    assert ctl.go_back(1.0, then="cancel") is None and ctl.mode == RETURNING
    # 1 rad at 0.5 rad/s: 2 s, eased.
    assert 0.0 < ctl.step(2.0)["right"][0] < 1.0
    out = ctl.step(3.0)
    np.testing.assert_allclose(out["left"], 0.0, atol=1e-9)
    assert ctl.mode == PASSTHROUGH and ctl.after_return == "cancel"
    assert ctl.step(3.1) is None


def test_go_back_in_passthrough_is_immediate():
    ctl = Controller()
    ready(ctl)
    assert ctl.go_back(0.0, then="cancel") is None
    assert ctl.mode == PASSTHROUGH and ctl.after_return == "cancel"


def test_start_needs_a_gate_command():
    ctl = Controller()
    assert ctl.start(0.0) is not None and ctl.mode == PASSTHROUGH


def test_the_grippers_stay_where_the_clutch_had_them():
    ctl = Controller(ControllerParams(start_blend_s=0.0))
    ready(ctl, cmd=0.0)
    for side, g in (("right", -0.47), ("left", 0.01)):
        q = np.zeros(8)
        q[7] = g
        ctl.on_cmd(side, q)
    ctl.start(0.0)
    ctl.on_chunk(chunk(0.5), 0.0, ctl.run_id, 0.0)
    out = ctl.step(0.5)
    assert out["right"][7] == pytest.approx(-0.47) and out["left"][7] == pytest.approx(0.01)
    assert out["right"][0] == pytest.approx(0.5 + 15 * 0.01)


def test_half_speed_runs_the_same_path_over_twice_the_time():
    ctl = Controller(ControllerParams(start_blend_s=0.0))
    ready(ctl)
    ctl.start(0.0)
    ctl.speed = 0.5
    ctl.on_chunk(chunk(0.0), 0.0, ctl.run_id, 0.0)
    # Step 15 is reached at 1 s instead of 0.5 s.
    assert ctl.step(1.0)["right"][0] == pytest.approx(0.15)
    # Changing speed mid-chunk keeps the target where it is, then advances at the new rate.
    ctl.set_speed(1.0, 1.0)
    assert ctl.step(1.0)["right"][0] == pytest.approx(0.15)
    assert ctl.step(1.1)["right"][0] == pytest.approx(0.18)
    # The stale-chunk check counts from arrival, not from the stretched clock.
    ctl.set_speed(1.1, 0.1)
    assert ctl.chunk_age_s(1.2) == pytest.approx(1.2)
