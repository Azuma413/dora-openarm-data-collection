import numpy as np

from dora_openarm_policy.diagnostics import StallMonitor


def feed(monitor, side, start, end, speed, t0=0.0, seconds=1.2, hz=250):
    for k in range(int(seconds * hz) + 1):
        t = t0 + k / hz
        q = np.full(8, start + speed * (t - t0))
        monitor.on_target(side, t, q)
    return t0 + seconds


def test_a_still_target_is_the_policy_holding():
    m = StallMonitor()
    now = feed(m, "right", 0.5, 0.5, 0.0)
    s = m.sample("right", now, np.full(8, 0.5), np.full(8, 0.5), np.zeros(8), np.zeros(8))
    assert s.target_speed < 0.01 and s.verdict == "policy holding"


def test_a_moving_target_the_arm_does_not_follow_looks_like_contact():
    m = StallMonitor()
    now = feed(m, "left", 0.0, None, 0.2)  # 0.2 rad/s
    torque = np.zeros(8)
    torque[3] = 6.5
    command = np.full(8, 0.2)
    measured = np.full(8, 0.2)
    measured[1] = 0.1  # J2 held back by 0.1 rad
    s = m.sample("left", now, command, measured, np.zeros(8), torque)
    assert abs(s.target_speed - 0.2) < 0.01
    assert s.tracking_joint == 1 and abs(s.tracking - 0.1) < 1e-9
    assert s.torque_joint == 3 and s.torque == 6.5
    assert s.verdict == "blocked by contact?"
    assert "J2" in s.text() and "J4" in s.text()


def test_a_moving_arm_is_fine():
    m = StallMonitor()
    now = feed(m, "right", 0.0, None, 0.3)
    s = m.sample("right", now, np.full(8, 0.35), np.full(8, 0.34), np.full(8, 0.28), np.zeros(8))
    assert s.verdict == ""
    assert m.sample("left", now, None, None, None, None) is None
