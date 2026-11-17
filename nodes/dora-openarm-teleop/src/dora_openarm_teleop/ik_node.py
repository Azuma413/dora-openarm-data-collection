"""dora-openarm-teleop-ik: dora-openarm-ik, configured to agree with the gate.

The upstream IK node (dora-openarm-kinematics, mink + daqp) is run unchanged, in
this process, after three things it cannot do for itself (docs/teleop.md, "IK"):

1. **Joint ranges.** It solves inside the MJCF's ranges, which are much wider than
   what the gate lets through (right J1 up to 3.49 rad against 2.0). The gate then
   holds the arm with reason ``ik_limit``. A copy of the IK's model with the
   gate's ranges (minus its limit margin) is written and passed as ``--xml``.
2. **Joint speeds.** ``--limit-velocity`` alone uses openarm_control's built-in
   caps (2.0 2.0 3.14 3.14 6.3 6.3 6.3 rad/s), faster than the gate's (sim 0.8x,
   real 0.5x of the driver's). The IK then runs ahead and the gate clips every
   tick of a fast reach. A ``--config`` with 0.9x the gate's caps is written.
3. **Fast cores.** On a big.LITTLE CPU a solve takes ~3x longer on a little core
   (1.1 ms vs 3.7 ms measured on a GB10), close to a 4 ms tick. The process pins
   itself to the big cores when the kernel reports differing ``cpu_capacity``.

Everything else is the IK's own: the base arguments (:data:`IK_ARGS`) keep the
upstream module's tuning and do NOT bound the per-solve frame error
(``--frame-*-error-limit``): 0.001 m / 0.004 rad per 4 ms solve caps the hand at
0.25 m/s and the wrist at ~57 deg/s, which is the lag operators feel. The gate
already bounds joint speed.

Arguments before ``--`` are this launcher's; anything after ``--`` is passed to
dora-openarm-ik after the generated ones (so it can override them).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import sys
from pathlib import Path

from dora_openarm_teleop import gate_node
from dora_openarm_teleop.gate import DEFAULT_LIMITS_PATH, PROFILES, resolve_profile
from dora_openarm_teleop.ik_model import ik_config_files, ik_model_files

# What the upstream leader-webxr module ran, minus the error limits (see above).
IK_ARGS = (
    "--mode bimanual --max-iters 5 --damping 0.1 --posture-cost 0 --lm-damping 0.01 "
    "--limit-velocity"
)
CPU_SYSFS = Path("/sys/devices/system/cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dora-openarm-teleop-ik",
        description="Run dora-openarm-ik with the gate's joint ranges and speeds.",
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        required=True,
        help="the rate targets arrive at (the clutch's tick); the IK's velocity caps are "
        "per solve, cap / tick-hz",
    )
    parser.add_argument("--profile", choices=["auto", *sorted(PROFILES)], default="auto")
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS_PATH)
    parser.add_argument("--limit-margin", type=float, help="the gate's (default 0.05)")
    parser.add_argument("--velocity-scale", type=float, help="the gate's (sim 0.8, real 0.5)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "dora-openarm-teleop",
        help="where the generated model and config are written",
    )
    parser.add_argument(
        "--cpus",
        default="auto",
        help="CPU list to pin to (e.g. 5-9,15-19), 'auto' for the big cores of a "
        "heterogeneous CPU, or 'none'",
    )
    return parser


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1 :]
    return argv, []


def fast_cpus(sysfs: Path = CPU_SYSFS) -> set[int] | None:
    """The big cores of a heterogeneous CPU, or None (homogeneous, or not exposed)."""
    capacities: dict[int, int] = {}
    for entry in sysfs.glob("cpu[0-9]*"):
        try:
            capacities[int(entry.name[3:])] = int((entry / "cpu_capacity").read_text())
        except (OSError, ValueError):
            continue
    try:
        allowed = os.sched_getaffinity(0)
        capacities = {cpu: cap for cpu, cap in capacities.items() if cpu in allowed}
    except (AttributeError, OSError):
        pass
    if not capacities:
        return None
    top = max(capacities.values())
    fast = {cpu for cpu, cap in capacities.items() if cap >= 0.9 * top}
    return None if len(fast) == len(capacities) else fast


def parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition("-")
        cpus.update(range(int(low), int(high or low) + 1))
    if not cpus:
        raise ValueError(f"empty CPU list {text!r}")
    return cpus


def generated_files(args: argparse.Namespace) -> tuple[Path, Path, dict[Path, bytes]]:
    """(model path, config path, {path: bytes}) for these gate settings.

    The gate's own parser resolves the margin and the speed scale, so a value the
    gate would reject stops the IK too, and the defaults can never drift apart.
    """
    gate_argv = ["--profile", args.profile, "--limits", str(args.limits)]
    if args.limit_margin is not None:
        gate_argv += ["--limit-margin", repr(args.limit_margin)]
    if args.velocity_scale is not None:
        gate_argv += ["--velocity-scale", repr(args.velocity_scale)]
    params = gate_node.params_from_args(gate_node.parse_args(gate_argv, environ={}))
    key = hashlib.sha256(
        f"{Path(args.limits).resolve()}|{params.limit_margin!r}|{args.profile}|"
        f"{params.velocity_scale!r}".encode()
    ).hexdigest()[:16]
    directory = Path(args.cache_dir).resolve() / key
    model = directory / "ik_model.xml"
    config = directory / "ik_config.yaml"
    files = ik_model_files(Path(args.limits), params.limit_margin, model)
    files.update(ik_config_files(args.limits, args.profile, params.velocity_scale, config))
    return model, config, files


def ik_argv(args: argparse.Namespace, model: Path, config: Path, extra: list[str]) -> list[str]:
    return [
        *shlex.split(IK_ARGS),
        "--xml",
        str(model),
        "--config",
        str(config),
        "--tick-hz",
        repr(args.tick_hz),
        *extra,
    ]


def _write(files: dict[Path, bytes]) -> None:
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_bytes() == data:
            continue
        temporary = path.with_name(f".{path.name}.{os.getpid()}")
        temporary.write_bytes(data)
        os.replace(temporary, path)


def _pin(cpus: str) -> str:
    if cpus == "none":
        return "not pinned"
    chosen = fast_cpus() if cpus == "auto" else parse_cpu_list(cpus)
    if chosen is None:
        return "not pinned (homogeneous CPU)"
    os.sched_setaffinity(0, chosen)
    return f"pinned to CPUs {sorted(chosen)}"


def main(argv: list[str] | None = None) -> None:
    own, extra = split_argv(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(own)
    args.profile = resolve_profile(args.profile, os.environ)
    model, config, files = generated_files(args)
    _write(files)
    pinned = _pin(args.cpus)
    forwarded = ik_argv(args, model, config, extra)
    print(
        f"[dora-openarm-teleop-ik] profile={args.profile} {pinned}; "
        f"dora-openarm-ik {shlex.join(forwarded)}",
        file=sys.stderr,
        flush=True,
    )
    from dora_openarm_kinematics import ik

    sys.argv = ["dora-openarm-ik", *forwarded]
    ik.main()


if __name__ == "__main__":
    main()
