"""Config rejection invariants (ADR-0007).

Two properties of ``build_config_from_data`` that no per-section test owns, both
driven by the live dataclass fields with the probe values kept below as code, so a
new config knob needs no recorded row:

* **Rescue class.** The layered load discards the global layer on a rescuable
  ``ConfigError`` and re-raises a ``ConstructionError`` (``global_config``). A host-wide
  section's own constructor (``ci_fleet``'s ``__post_init__``) raises ``ValueError`` /
  ``TypeError``, which the loader wraps as ``ConstructionError``. If a use-site rule
  (``NotNull``, ``Typed``, ...) on ``runner_allocation`` / ``runner_scaling`` started
  rejecting such a value first, it would become a rescuable ``FieldError`` and a bad
  value would silently discard the global layer instead of failing loudly.
* **No crash.** Every live config path x probe value either loads or raises a
  ``ConfigError``; nothing else escapes.

Message text is deliberately not compared (tests/test_config_validation.py and the
per-section tests own the grammar).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import types
import typing
from collections.abc import Mapping

import pytest

from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.config_validation import ConfigError, ConstructionError

PROBES: list[tuple[str, object]] = [
    ("none", None), ("true", True), ("false", False), ("0", 0), ("1", 1), ("-1", -1),
    ("2", 2), ("1.5", 1.5), ("0.5", 0.5), ("-0.5", -0.5), ("inf", math.inf), ("nan", math.nan),
    ("empty", ""), ("blank", "  "), ("x", "x"), ("ab", "a b"), ("../x", "../x"),
    ("/abs", "/abs"), ("oldest", "oldest"), ("warn", "warn"), ("http", "http"),
    ("off", "off"), ("front", "front_of_train"), ("codex", "codex"),
    ("claude", "claude-code"), ("manual", "manual"), ("--x", "--x"),
    ("ph_ok", "{prompt_path}"), ("ph_bad", "{bad}"), ("ph_empty", "{}"), ("re_bad", "["),
    ("l_empty", []), ("l_str", ["a"]), ("l_flag", ["--squash"]),
    ("l_dflag", ["--delete-branch"]), ("l_int", [1]), ("l_none", [None]),
    ("l_ph_ok", ["x", "{prompt_path}"]), ("l_ph_bad", ["x", "{bad}"]),
    ("d_empty", {}), ("d_ss", {"a": "b"}), ("d_si", {"a": 1}),
]  # fmt: skip

_PROVIDER_BASE = {
    "base_url": "http://x",
    "api_key_env": "K",
    "model": "m",
    "input_usd_per_mtok": 1,
    "output_usd_per_mtok": 2,
}
_RULE_BASE = {"pattern": "a", "kind": "k"}
# Element seeds for ``tuple[Dataclass, ...]`` fields, by element class (default ``_RULE_BASE``).
_ENTRY_SEEDS = {
    "RoleEntry": {"harness": "devin-shell"},
    "DeferredDemand": {"labels": ["a"], "after_seconds": 60},
}
# A role chain is only valid beside a primary that records sessions (``manual`` -- the worker
# default -- is rejected in any chain with fallbacks), so probes under ``worker.fallbacks`` /
# ``reviewer.fallbacks`` are placed on top of one, else every cell would record ``R``.
_CONTEXT = {
    ("worker", "fallbacks"): {"worker": {"harness": "claude-code", "model": "m"}},
    ("reviewer", "fallbacks"): {"reviewer": {"harness": "claude-code", "model": "m"}},
}


def _unwrap(tp: typing.Any) -> typing.Any:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(members) == 1:
            return members[0]
    return tp


def _is_dc(tp: typing.Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def walk_targets() -> list[dict]:
    """Every (key path, leaf seed, container?) reachable from OrchestratorConfig."""
    out: list[dict] = []

    def visit(cls: type, path: list, seed: dict) -> None:
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if not f.init or f.metadata.get("provenance"):
                continue
            tp = _unwrap(hints[f.name])
            here = [*path, f.name]
            origin, args = typing.get_origin(tp), typing.get_args(tp)
            sub: tuple[list, type, dict] | None = None
            if _is_dc(tp):
                sub = (here, tp, {})
            elif origin is tuple and args and _is_dc(args[0]):
                sub = ([*here, 0], args[0], dict(_ENTRY_SEEDS.get(args[0].__name__, _RULE_BASE)))
            elif origin in (dict, Mapping) and len(args) == 2 and _is_dc(args[1]):
                sub = ([*here, "p"], args[1], dict(_PROVIDER_BASE))
            out.append(
                {"id": ".".join(map(str, here)), "path": here, "seed": seed, "sub": bool(sub)}
            )
            if sub is not None:
                if sub[0] is not here:
                    out.append(
                        {"id": ".".join(map(str, sub[0])), "path": sub[0], "seed": {}, "sub": True}
                    )
                visit(sub[1], sub[0], sub[2])

    for f in dataclasses.fields(OrchestratorConfig):
        if f.metadata.get("provenance"):
            continue
        tp = typing.get_type_hints(OrchestratorConfig)[f.name]
        out.append({"id": f.name, "path": [f.name], "seed": {}, "sub": True})
        if _is_dc(tp):
            visit(tp, [f.name], {})
    # Retired ``supervisor.<key>`` spellings of the relocated fleet_supervisor
    # keys: since #1979 they are unknown keys, and keeping the probes pins the
    # rejection for every future fleet_supervisor knob as well.
    from charlie_work.fleet_supervisor_config import FleetSupervisorConfig

    for f in dataclasses.fields(FleetSupervisorConfig):
        out.append(
            {
                "id": f"supervisor.{f.name}",
                "path": ["supervisor", f.name],
                "seed": {},
                "sub": False,
            }
        )
    return out


def place(path: list, value: object, seed: dict) -> dict:
    """Raw config with ``value`` at ``path``; the leaf's parent mapping starts from ``seed``."""

    def make(i: int) -> typing.Any:
        key, last = path[i], i == len(path) - 1
        child = value if last else make(i + 1)
        if isinstance(key, int):
            return [child]
        node = copy.deepcopy(seed) if last else {}
        node[key] = child
        return node

    raw = make(0)
    for prefix, base in _CONTEXT.items():
        if tuple(path[: len(prefix)]) == prefix:
            ctx = copy.deepcopy(base)
            ctx[prefix[0]].update(raw[prefix[0]])
            raw = {**raw, **ctx}
    return raw


# ------------------------------------------------------------------------- tests
def _build_outcome(raw: dict) -> BaseException | None:
    """The ``ConfigError`` ``build_config_from_data`` raises, else ``None`` (it loaded).

    Anything that is not a ``ConfigError`` propagates, so it fails the calling test.
    """
    try:
        build_config_from_data(copy.deepcopy(raw))
    except ConfigError as err:
        return err
    return None


def _constructor_rejects(cls: type, name: str, value: object) -> bool:
    try:
        cls(**{name: value})
    except (ValueError, TypeError):
        return True
    return False


_HOST_WIDE_SECTIONS = ("runner_allocation", "runner_scaling")


def _is_documented_rescuable_exception(section: str, name: str, value: object) -> bool:
    # ``runner_allocation.deferred_demand`` is the deliberate exception: its use-site
    # ``Entries`` rule makes a bad entry a rescuable ``FieldError`` (config.py, pinned by
    # tests/test_runner_allocation_deferred_demand.py). Only ``null`` stays loud.
    return section == "runner_allocation" and name == "deferred_demand" and value is not None


def test_host_wide_constructor_rejections_stay_non_rescuable():
    """A value the section's own constructor rejects must surface as a ``ConstructionError``.

    The constructor (``ci_fleet.config`` ``__post_init__``) is the oracle, so the
    expectation follows the live dataclass fields, not a recorded row. A use-site rule
    that began rejecting one of these values first would raise a rescuable ``FieldError``
    and make the #665 layered load discard the global layer instead of failing loudly.
    """
    hints = typing.get_type_hints(OrchestratorConfig)
    asserted = dict.fromkeys(_HOST_WIDE_SECTIONS, 0)
    wrong = []
    for section in _HOST_WIDE_SECTIONS:
        cls = _unwrap(hints[section])
        for f in dataclasses.fields(cls):
            for probe, value in PROBES:
                if _is_documented_rescuable_exception(section, f.name, value):
                    continue
                if not _constructor_rejects(cls, f.name, value):
                    continue
                err = _build_outcome({section: {f.name: value}})
                if err is None:
                    continue  # the loader normalized the value (e.g. ``null`` -> default)
                asserted[section] += 1
                if not isinstance(err, ConstructionError):
                    wrong.append(f"{section}.{f.name} <- {probe}: {type(err).__name__}")
    assert all(n > 0 for n in asserted.values()), f"vacuous: {asserted}"
    assert not wrong, "rescuable rejection of a constructor-rejected value:\n" + "\n".join(wrong)


@pytest.mark.parametrize(
    "raw",
    [
        {"runner_allocation": {"max_running_per_repo": {"a/b": 1}}},
        {"runner_allocation": {"max_running_per_repo": [["a/b", 1], ["a/b", 2]]}},
        {"runner_allocation": {"max_running_per_repo": [["a/b", 1]], "min_running_per_repo": 2}},
    ],
    ids=["mapping", "duplicate_repo", "min_above_ceiling"],
)
def test_runner_allocation_cross_key_rejections_stay_non_rescuable(raw):
    assert isinstance(_build_outcome(raw), ConstructionError)


def test_every_live_config_path_and_probe_loads_or_raises_config_error():
    """Nothing but a ``ConfigError`` escapes ``build_config_from_data``, for any probe
    value at any live config path (and for an unknown key under any container)."""
    targets = walk_targets()
    assert len(targets) > 300
    cells = 0
    for t in targets:
        for _, value in PROBES:
            _build_outcome(place(t["path"], value, t["seed"]))
            cells += 1
        if t["sub"]:
            _build_outcome(place([*t["path"], "zzz_unknown"], 1, t["seed"]))
    assert cells == len(targets) * len(PROBES)
