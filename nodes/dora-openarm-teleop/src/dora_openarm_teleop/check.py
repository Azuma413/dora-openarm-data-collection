"""openarm-teleop-check: refuse a dataflow whose teleop nodes disagree (docs/teleop.md).

The couplings below are not visible to dora. Each one, broken, still starts a
dataflow that moves the arm; it only behaves worse in a way that is hard to see:

* webxr / clutch timing (:mod:`dora_openarm_teleop.timing`): broken, an arm can
  slide 7 cm when re-gripped after a stall, or no input gap is ever bridged.
* the clutch, the IK and the tick all run at one rate: the IK's velocity caps are
  per solve (``cap / --tick-hz``), so a mismatch scales the arm's speed.
* the IK's targets are queued (``queue_size`` >= 2): with 1, a late IK drops one
  side's target and solves the current right hand with the previous left one.
* the IK launcher and the gate resolve the same limits, margin, profile and
  speed scale, or the IK plans what the gate then clips or holds.

Usage (also run by scripts/dora-envsubst before every `dora run`)::

    LEADER=webxr ROBOT=mujoco openarm-teleop-check dataflow.yaml

A dataflow without the teleop nodes passes with nothing to check.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shlex
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from dora_openarm_teleop import clutch_node, gate_node, ik_node, timing

TIMER = re.compile(r"^dora/timer/(millis|secs)/(\d+(?:\.\d+)?)$")
VARIABLE = re.compile(r"\$(?:\{(\w+)(?::-([^}]*))?\}|(\w+))")


def expand(text: str, environ: Mapping[str, str]) -> str:
    """`$VAR`, `${VAR}` and `${VAR:-default}`, as dora and scripts/dora-envsubst do."""

    def one(match: re.Match) -> str:
        name = match.group(1) or match.group(3)
        value = environ.get(name, "")
        if value:
            return value
        return match.group(2) or ""

    return VARIABLE.sub(one, text)


@contextlib.contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Run with exactly ``values`` as os.environ (parsers read defaults from it)."""
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(values)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _args(node: Mapping[str, Any], environ: Mapping[str, str]) -> list[str]:
    return shlex.split(expand(str(node.get("args", "") or ""), environ))


def _env(node: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, str]:
    merged = dict(environ)
    for key, value in (node.get("env") or {}).items():
        merged[str(key)] = expand(str(value), environ)
    return merged


def _source(spec: Any) -> str:
    return str(spec.get("source", "")) if isinstance(spec, Mapping) else str(spec)


def _queue_size(spec: Any) -> int | None:
    return spec.get("queue_size") if isinstance(spec, Mapping) else None


def _executable(node: Mapping[str, Any]) -> str:
    return Path(str(node.get("path", ""))).name


def load_nodes(dataflow: Path, environ: Mapping[str, str]) -> tuple[list[dict], dict[str, list]]:
    """(top-level nodes, {module node id: its nodes}) with `module:` paths expanded."""
    top = yaml.safe_load(dataflow.read_text(encoding="utf-8")) or {}
    modules: dict[str, list] = {}
    for node in top.get("nodes", []):
        if "module" in node:
            path = dataflow.parent / expand(str(node["module"]), environ)
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            modules[str(node["id"])] = list(data.get("nodes", []))
    return list(top.get("nodes", [])), modules


def tick_hz(nodes: list[dict], module_id: str, environ: Mapping[str, str]) -> float | None:
    """The rate of the `tick` a module receives, following one relay node."""
    by_id = {str(n.get("id")): n for n in nodes}
    module = by_id.get(module_id)
    if module is None:
        return None
    source = _source((module.get("inputs") or {}).get("tick", ""))
    for _ in range(4):
        match = TIMER.match(source)
        if match:
            value = float(match.group(2))
            return 1000.0 / value if match.group(1) == "millis" else 1.0 / value
        relay = by_id.get(source.split("/")[0])
        if relay is None:
            return None
        source = _source((relay.get("inputs") or {}).get("tick", ""))
    return None


def check_module(
    nodes: list[dict], environ: Mapping[str, str], leader_hz: float | None
) -> list[str]:
    """Errors of one module's teleop nodes; [] when it has none."""
    found: dict[str, dict] = {}
    for node in nodes:
        name = _executable(node)
        if name in ("dora-openarm-webxr", "dora-openarm-clutch", "dora-openarm-gate"):
            found[name.removeprefix("dora-openarm-")] = node
        elif name == "dora-openarm-teleop-ik":
            found["ik"] = node
    if "clutch" not in found:
        return []
    errors: list[str] = []
    clutch = found["clutch"]
    clutch_env = _env(clutch, environ)
    try:
        with _environment(clutch_env):
            clutch_args = clutch_node.build_parser().parse_args(_args(clutch, environ))
    except SystemExit:
        return [f"clutch: args do not parse: {clutch.get('args')!r}"]
    fresh_s = clutch_args.stale_ms / 1000.0
    resume_s = clutch_args.resume_gap_ms / 1000.0
    stale = f"clutch --stale-ms {clutch_args.stale_ms:g}"
    if resume_s > 0.0 and resume_s <= fresh_s + timing.EPS_S:
        errors.append(
            f"clutch --resume-gap-ms {clutch_args.resume_gap_ms:g} must be longer than "
            f"{stale}: a gap is only bridged once the input has been late for --stale-ms, "
            "so this window bridges nothing (0 turns bridging off explicitly)"
        )

    webxr = found.get("webxr")
    if webxr is not None:
        from dora_openarm_webxr.main import build_parser as webxr_parser

        try:
            with _environment(_env(webxr, environ)):
                webxr_args = webxr_parser().parse_args(
                    [*_args(webxr, environ), "--tls-certificate-file", "-", "--tls-key-file", "-"]
                )
        except SystemExit:
            return [*errors, f"webxr: args do not parse: {webxr.get('args')!r}"]
        timeout = float(webxr_args.pose_timeout)
        gap = float(webxr_args.smoother_reset_gap)
        if timeout < timing.MIN_POSE_TIMEOUT_S - timing.EPS_S:
            errors.append(
                f"webxr POSE_TIMEOUT {timeout:g} s is below {timing.MIN_POSE_TIMEOUT_S} s "
                "(0 = never): late frames would reset the smoothers and the pose would jump"
            )
        elif timeout > fresh_s + timing.EPS_S:
            errors.append(
                f"webxr POSE_TIMEOUT {timeout:g} s is longer than {stale}: a stall between the "
                "two disarms the clutch without a pose gap, and the arm can slide when "
                "re-gripped (POSE_TIMEOUT <= fresh_s < SMOOTHER_RESET_GAP)"
            )
        if gap <= fresh_s + timing.EPS_S:
            errors.append(
                f"webxr SMOOTHER_RESET_GAP {gap:g} s must be longer than {stale}: the neck "
                "pivot would re-latch under an engaged arm"
            )
        if str(webxr_args.pivot_mode) != timing.PIVOT_MODE:
            errors.append(
                f"webxr PIVOT_MODE is {webxr_args.pivot_mode!r}: with 'live', leaning in "
                "while gripping moves the arm; use 'latched'"
            )

    ik = found.get("ik")
    if ik is None:
        errors.append("the clutch feeds no dora-openarm-teleop-ik node")
        return errors
    own, _ = ik_node.split_argv(_args(ik, environ))
    try:
        ik_args = ik_node.build_parser().parse_args(own)
    except SystemExit:
        return [*errors, f"ik: args do not parse: {ik.get('args')!r}"]
    rates = {"clutch --tick-hz": float(clutch_args.tick_hz), "ik --tick-hz": ik_args.tick_hz}
    if leader_hz is not None:
        rates["the module's tick"] = leader_hz
    if max(rates.values()) - min(rates.values()) > 1e-6 * max(rates.values()):
        listed = ", ".join(f"{k} {v:g}" for k, v in rates.items())
        errors.append(
            f"tick rates differ ({listed}): the IK's speed caps are per solve, so the arm "
            "would move at the wrong speed"
        )
    for side in ("right", "left"):
        spec = (ik.get("inputs") or {}).get(f"target_{side}")
        if spec is None:
            errors.append(f"ik has no target_{side} input")
        elif (_queue_size(spec) or 0) < 2:
            errors.append(
                f"ik target_{side} needs queue_size >= 2 (10 recommended): with 1, a late "
                "IK drops one side's target and pairs this tick's right with last tick's left"
            )

    gate = found.get("gate")
    if gate is None:
        errors.append("the IK feeds no dora-openarm-gate node")
        return errors
    try:
        gate_args = gate_node.parse_args(_args(gate, environ), environ=_env(gate, environ))
        gate_params = gate_node.params_from_args(gate_args)
    except (SystemExit, ValueError) as error:
        return [*errors, f"gate: args do not parse: {error}"]
    ik_args.profile = gate_node.resolve_profile(ik_args.profile, _env(ik, environ))
    ik_gate = gate_node.params_from_args(
        gate_node.parse_args(
            [
                "--profile",
                ik_args.profile,
                "--limits",
                str(ik_args.limits),
                *(
                    ["--limit-margin", repr(ik_args.limit_margin)]
                    if ik_args.limit_margin is not None
                    else []
                ),
                *(
                    ["--velocity-scale", repr(ik_args.velocity_scale)]
                    if ik_args.velocity_scale is not None
                    else []
                ),
            ],
            environ={},
        )
    )
    pairs = {
        "profile": (gate_args.profile, ik_args.profile),
        "limits": (Path(gate_args.limits).resolve(), Path(ik_args.limits).resolve()),
        "limit margin": (gate_params.limit_margin, ik_gate.limit_margin),
        "velocity scale": (gate_params.velocity_scale, ik_gate.velocity_scale),
    }
    for what, (at_gate, at_ik) in pairs.items():
        if at_gate != at_ik:
            errors.append(
                f"gate and IK disagree on the {what} ({at_gate} vs {at_ik}): the IK would "
                "plan motion the gate clips or holds"
            )
    for side in ("right", "left"):
        if f"state_{side}" not in (gate.get("inputs") or {}):
            errors.append(f"gate has no state_{side} input: it never engages without one")
    return errors


def check(dataflow: Path, environ: Mapping[str, str]) -> list[str]:
    top, modules = load_nodes(dataflow, environ)
    errors: list[str] = []
    for module_id, nodes in modules.items():
        hz = tick_hz(top, module_id, environ)
        errors.extend(f"{module_id}: {e}" for e in check_module(nodes, environ, hz))
    errors.extend(check_module(top, environ, None))
    return errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openarm-teleop-check", description=__doc__.split("\n")[0]
    )
    parser.add_argument("dataflow", type=Path)
    args = parser.parse_args(argv)
    errors = check(args.dataflow, os.environ)
    for error in errors:
        print(f"openarm-teleop-check: {error}", file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print(f"openarm-teleop-check: {args.dataflow} OK", file=sys.stderr)


if __name__ == "__main__":
    main()
