"""dora-openarm-teleop-ik: what it hands dora-openarm-ik, and where it runs."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pytest
import yaml

from dora_openarm_teleop import ik_node
from dora_openarm_teleop.gate import DEFAULT_LIMITS_PATH, PROFILES, load_limits
from dora_openarm_teleop.ik_model import IK_CONFIG_KEY


def _args(tmp_path: Path, *extra: str, profile: str = "sim") -> argparse.Namespace:
    args = ik_node.build_parser().parse_args(
        ["--tick-hz", "250", "--cache-dir", str(tmp_path), "--profile", profile, *extra]
    )
    return args


def test_no_frame_error_limits_are_passed():
    """0.001 m / 0.004 rad per 4 ms solve caps the hand at 0.25 m/s and the wrist at
    1 rad/s: the lag this launcher exists to remove. The gate bounds joint speed."""
    assert "error-limit" not in ik_node.IK_ARGS
    assert "--limit-velocity" in ik_node.IK_ARGS


@pytest.mark.parametrize("profile", ["sim", "real"])
def test_config_caps_are_nine_tenths_of_the_gate(tmp_path: Path, profile: str):
    model, config, files = ik_node.generated_files(_args(tmp_path, profile=profile))
    caps = yaml.safe_load(files[config].decode())[IK_CONFIG_KEY]
    driver = load_limits(DEFAULT_LIMITS_PATH)["right"].velocity
    scale = PROFILES[profile].velocity_scale
    assert caps == pytest.approx([0.9 * scale * float(v) for v in driver[:7]])
    assert model in files and model.parent == config.parent


def test_settings_that_differ_get_their_own_files(tmp_path: Path):
    sim = ik_node.generated_files(_args(tmp_path, profile="sim"))
    real = ik_node.generated_files(_args(tmp_path, profile="real"))
    margin = ik_node.generated_files(_args(tmp_path, "--limit-margin", "0.1", profile="sim"))
    assert len({sim[1], real[1], margin[1]}) == 3


def test_a_margin_the_gate_rejects_stops_the_ik_too(tmp_path: Path):
    with pytest.raises(ValueError):
        ik_node.generated_files(_args(tmp_path, "--limit-margin", "-1"))


def test_forwarded_argv_puts_extra_args_last(tmp_path: Path):
    args = _args(tmp_path)
    own, extra = ik_node.split_argv(["--tick-hz", "250", "--", "--max-iters", "8"])
    assert own == ["--tick-hz", "250"] and extra == ["--max-iters", "8"]
    argv = ik_node.ik_argv(args, Path("/m.xml"), Path("/c.yaml"), extra)
    assert argv[-2:] == ["--max-iters", "8"]
    assert argv[argv.index("--tick-hz") + 1] == "250.0"
    assert argv[argv.index("--xml") + 1] == "/m.xml"
    assert shlex.split(ik_node.IK_ARGS) == argv[: len(shlex.split(ik_node.IK_ARGS))]


def test_the_generated_files_parse_as_dora_openarm_ik_arguments(tmp_path: Path):
    control = pytest.importorskip("openarm_control")
    args = _args(tmp_path)
    model, config, files = ik_node.generated_files(args)
    ik_node._write(files)
    parser = argparse.ArgumentParser()
    control.register_common_args(parser)
    control.register_ik_args(parser)
    parsed = parser.parse_args(ik_node.ik_argv(args, model, config, []))
    params = control.ik_params_from_args(parsed)
    assert params.dt == pytest.approx(1 / 250)
    assert params.frame_position_error_limit == pytest.approx(0.02)  # upstream default
    assert params.velocity_limits is not None


def _sysfs(tmp_path: Path, capacities: dict[int, int]) -> Path:
    for cpu, capacity in capacities.items():
        (tmp_path / f"cpu{cpu}").mkdir()
        (tmp_path / f"cpu{cpu}" / "cpu_capacity").write_text(f"{capacity}\n")
    return tmp_path


def test_fast_cpus_on_big_little(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ik_node.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3})
    sysfs = _sysfs(tmp_path, {0: 400, 1: 400, 2: 1024, 3: 1000})
    assert ik_node.fast_cpus(sysfs) == {2, 3}


def test_no_pinning_on_a_homogeneous_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ik_node.os, "sched_getaffinity", lambda pid: {0, 1})
    assert ik_node.fast_cpus(_sysfs(tmp_path, {0: 1024, 1: 1024})) is None
    assert ik_node.fast_cpus(tmp_path / "missing") is None


def test_cpu_list_parsing():
    assert ik_node.parse_cpu_list("5-9,15") == {5, 6, 7, 8, 9, 15}
    with pytest.raises(ValueError):
        ik_node.parse_cpu_list(" , ")
