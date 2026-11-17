"""openarm-teleop-check against the repository's own dataflow, and broken copies of it.

Each negative case breaks exactly one coupling in a copy of modules/leader-webxr.yaml
and expects exactly that coupling's error, so a check that fires for everything (or
for nothing) fails here.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from dora_openarm_teleop.check import check

REPO = Path(__file__).resolve().parents[3]
ENV = {"LEADER": "webxr", "ROBOT": "mujoco"}


@pytest.fixture
def flow(tmp_path: Path) -> Path:
    """A copy of dataflow.yaml and modules/ that a test may edit."""
    shutil.copy(REPO / "dataflow.yaml", tmp_path / "dataflow.yaml")
    shutil.copytree(REPO / "modules", tmp_path / "modules")
    return tmp_path / "dataflow.yaml"


def edit(flow: Path, old: str, new: str, module: str = "modules/leader-webxr.yaml") -> None:
    path = flow.parent / module
    text = path.read_text()
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new))


@pytest.mark.parametrize(
    "environ",
    [ENV, {"LEADER": "webxr", "ROBOT": "pedestal"}, {"LEADER": "ker", "ROBOT": "cell"}],
)
def test_the_shipped_dataflow_passes(environ: dict[str, str]) -> None:
    assert check(REPO / "dataflow.yaml", environ) == []


def test_a_leader_without_teleop_nodes_has_nothing_to_check(flow: Path) -> None:
    assert check(flow, {"LEADER": "dummy", "ROBOT": "dummy"}) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        # webxr times out after the clutch has already let go.
        ('POSE_TIMEOUT: "0.09"', 'POSE_TIMEOUT: "0.2"', "POSE_TIMEOUT 0.2 s is longer"),
        ('POSE_TIMEOUT: "0.09"', 'POSE_TIMEOUT: "0"', "POSE_TIMEOUT 0 s is below"),
        ('SMOOTHER_RESET_GAP: "0.12"', 'SMOOTHER_RESET_GAP: "0.1"', "SMOOTHER_RESET_GAP 0.1 s"),
        ('PIVOT_MODE: "latched"', 'PIVOT_MODE: "live"', "PIVOT_MODE is 'live'"),
        (
            "args: --scene demo --tick-hz 250",
            "args: --scene demo --tick-hz 250 --resume-gap-ms 100",
            "--resume-gap-ms 100 must be longer",
        ),
        ("args: --tick-hz 250\n", "args: --tick-hz 200\n", "tick rates differ"),
        (
            "        source: clutch/target_left\n        queue_size: 10",
            "        source: clutch/target_left\n        queue_size: 1",
            "ik target_left needs queue_size",
        ),
        (
            '    path: ../dora-openarm-gate\n    env:\n      ROBOT: "${ROBOT:-dummy}"\n',
            (
                "    path: ../dora-openarm-gate\n    args: --profile real\n    env:\n"
                '      ROBOT: "${ROBOT:-dummy}"\n'
            ),
            ("disagree on the profile", "disagree on the velocity scale"),
        ),
        (
            "    args: --tick-hz 250\n",
            "    args: --tick-hz 250 --limit-margin 0.1\n",
            "disagree on the limit margin",
        ),
        (
            "      state_left: _mod/state_left\n      control: _mod/command\n    outputs:\n      - cmd",
            "      control: _mod/command\n    outputs:\n      - cmd",
            "no state_left",
        ),
    ],
)
def test_each_broken_coupling_is_reported_alone(
    flow: Path, old: str, new: str, expected: str | tuple[str, ...]
) -> None:
    assert check(flow, ENV) == []
    edit(flow, old, new)
    errors = check(flow, ENV)
    # A different profile is also a different speed scale: both are reported.
    wanted = (expected,) if isinstance(expected, str) else expected
    assert len(errors) == len(wanted), errors
    for text, error in zip(wanted, errors, strict=True):
        assert text in error, errors


def test_the_leader_tick_must_match_the_ik(flow: Path) -> None:
    edit(flow, "tick: dora/timer/millis/4", "tick: dora/timer/millis/5", module="dataflow.yaml")
    errors = check(flow, ENV)
    assert len(errors) == 1 and "tick rates differ" in errors[0], errors
    assert "the module's tick 200" in errors[0]
