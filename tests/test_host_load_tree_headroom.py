"""Tree-headroom governor tests for host-load backpressure (issue #1903).

Sibling of ``tests/test_host_load.py`` -- that module's attachment point is
saturated, so the #1903 wiring tests live here (the same split pattern as
``test_ci_headroom.py`` / ``test_ci_capacity_backpressure.py``). The older
module covers the measurement layer (``pytest_tree_load`` /
``measure_host_load`` / the process listers) and the #1843 process-count
brake; this one covers the #1903 recalibration:

* the ``pytest_tree_count`` governor clamps by *suite headroom*
  (``max(0, cap - live_trees)``) -- partial grants near the cap instead of
  a straight drop to 0;
* the process-count term survives as the fan-out brake (strict ``>`` trip)
  and dominates when a single suite is run at pathological ``-n`` width;
* the recalibrated defaults (trees = ``cpu_count // 2``, processes =
  ``cpu_count * 3``) do not throttle a stock config at the incident shape:
  3 xdist trees (~20 processes) on a 16-core host;
* config validation for the new ``dispatch.host_load_max_pytest_trees``
  knob.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from charlie_work import host_load, quiesce
from charlie_work.config import (
    ConfigError,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    build_config_from_data,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401

# Self-contained per the zero-cross-test-import guard
# (tests/test_zero_cross_test_import_guard.py): the tiny process-snapshot
# helpers below duplicate test_host_load.py's _proc/_lister/_patch_procs
# rather than importing them.


def _proc(pid: int, ppid: int, command_line: str, name: str = "") -> quiesce.ProcessInfo:
    return quiesce.ProcessInfo(pid=pid, ppid=ppid, name=name, command_line=command_line)


def _lister(
    *procs: quiesce.ProcessInfo, error: str | None = None
) -> tuple[tuple[quiesce.ProcessInfo, ...], str | None]:
    return tuple(procs), error


def _patch_procs(monkeypatch: pytest.MonkeyPatch, *procs: quiesce.ProcessInfo) -> None:
    monkeypatch.setattr(host_load, "list_host_processes", lambda: _lister(*procs))


def _xdist_snapshot(*widths: int) -> list[quiesce.ProcessInfo]:
    """Fabricate one pytest tree per width: a root plus ``width`` xdist
    workers (the fleet's ~6-9-process suite shape from the #1903 report)."""
    procs: list[quiesce.ProcessInfo] = []
    for i, width in enumerate(widths):
        root = 100 + i * 100
        procs.append(_proc(root, 1, "uv run --extra dev pytest -n 6 -q --tb=short"))
        procs += [_proc(root + 1 + j, root, "python -u -c xdist") for j in range(width)]
    return procs


def _build_app(
    tmp_path: Path, *, trees_max: int, processes_max: int = 0, **kwargs: Any
) -> OrchestratorApp:
    """Build an app with explicit host-load knobs (``processes_max`` 0
    isolates the tree term unless a test opts into the brake)."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(
            host_load_max_pytest_processes=processes_max,
            host_load_max_pytest_trees=trees_max,
            default_limit=5,
            **kwargs,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


# ---------------------------------------------------------------------------
# Acceptance: the #1903 incident shape must not clamp under a stock config
# ---------------------------------------------------------------------------


def test_default_config_does_not_clamp_three_xdist_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1903 acceptance: 3 xdist trees (~20 processes) on a 16-core
    host must not clamp dispatch under the default config -- the shape that
    held dispatch at 0 for hours while the fleet sat mostly idle."""
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    _patch_procs(monkeypatch, *_xdist_snapshot(6, 6, 5))  # 3 trees, 20 procs

    config = OrchestratorConfig(dispatch=DispatchConfig(), devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app._apply_concurrency_governor(5)

    # Defaults on 16 cores: process brake 48 (20 < 48), tree cap 8
    # (headroom 8 - 3 = 5 >= requested 5) -- neither term binds.
    assert result.host_load_max_pytest_processes == 48
    assert result.host_load_max_pytest_trees == 8
    assert result.host_load_pytest_trees == 3
    assert result.host_load_pytest_processes == 20
    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.clamped_by is None


def test_default_config_does_not_clamp_two_suites_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The original miscalibration verbatim: 2 ordinary suites (the count
    that tripped the cpu_count process threshold on the 16-core host) must
    leave dispatch untouched."""
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    _patch_procs(monkeypatch, *_xdist_snapshot(8, 8))  # 2 trees, 18 procs

    config = OrchestratorConfig(dispatch=DispatchConfig(), devin=DevinConfig())
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.clamped_by is None


# ---------------------------------------------------------------------------
# Tree-term headroom semantics: partial grants, not a hard 0
# ---------------------------------------------------------------------------


def test_tree_cap_clamps_by_headroom_not_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5 live trees against a cap of 6 leaves room for exactly one more
    suite -- the pass may launch 1 worker, not 0 (the 'clamp by headroom'
    half of the fix)."""
    _patch_procs(monkeypatch, *_xdist_snapshot(7, 7, 7, 7, 7))  # 5 trees
    app = _build_app(tmp_path, trees_max=6)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_pytest_trees == 5
    assert result.dispatch_limit == 1
    assert result.clamped is True
    assert result.clamped_by == "host_load"


def test_tree_cap_at_capacity_clamps_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headroom is ``cap - trees``: at the cap there is genuinely no room
    for another suite, so the clamp still reaches 0 -- a deferral, not a
    failure."""
    _patch_procs(monkeypatch, *_xdist_snapshot(7, 7, 7))  # 3 trees
    app = _build_app(tmp_path, trees_max=3)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 0
    assert result.clamped is True
    assert result.clamped_by == "host_load"


def test_tree_cap_does_not_clamp_below_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Headroom >= the requested limit leaves it untouched -- the governor
    only ever tightens."""
    _patch_procs(monkeypatch, *_xdist_snapshot(7, 7))  # 2 trees, headroom 6
    app = _build_app(tmp_path, trees_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.clamped_by is None


def test_tree_cap_bounds_per_pass_launches_on_idle_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even on an idle host a single pass may not launch more suites than
    the cap itself -- headroom is ``cap - 0 = cap``. Bounds the burst a
    large ``limit`` request would otherwise drop on the host at once."""
    _patch_procs(monkeypatch)  # empty snapshot: 0 trees
    app = _build_app(tmp_path, trees_max=4)

    result = app._apply_concurrency_governor(10)

    assert result.dispatch_limit == 4
    assert result.clamped is True
    assert result.clamped_by == "host_load"


def test_tree_headroom_event_carries_both_readings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deferral event must identify which term bound (``host_load_term``)
    and carry both measured counts, both caps, and the partial
    ``clamped_limit`` -- 'dispatched fewer than asked' stays explainable
    from events.db alone."""
    _patch_procs(monkeypatch, *_xdist_snapshot(7, 7, 7, 7, 7))  # 5 trees
    app = _build_app(tmp_path, trees_max=6)

    app._apply_concurrency_governor(5)

    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["clamped_by"] == "host_load"
    assert payload["host_load_term"] == "pytest_trees"
    assert payload["host_load_pytest_trees"] == 5
    assert payload["host_load_pytest_processes"] == 40
    assert payload["host_load_max_pytest_trees"] == 6
    assert payload["host_load_max_pytest_processes"] == 0
    assert payload["requested_limit"] == 5
    assert payload["clamped_limit"] == 1


def test_tree_cap_dry_run_clamps_but_suppresses_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run still applies the headroom clamp (a preview must match a
    live pass) but writes no durable event -- same discipline as the
    process brake and the open_pr_max/ci_headroom clamps."""
    _patch_procs(monkeypatch, *_xdist_snapshot(7, 7, 7))
    config = OrchestratorConfig(
        dispatch=DispatchConfig(
            host_load_max_pytest_processes=0,
            host_load_max_pytest_trees=3,
            default_limit=5,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 0
    assert result.clamped is True
    assert result.clamped_by == "host_load"
    assert query_events(app.paths.state_file, kind="dispatch_backpressure") == []


# ---------------------------------------------------------------------------
# The process brake still guards pathological width
# ---------------------------------------------------------------------------


def test_process_brake_dominates_tree_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One suite at pathological ``-n`` width is invisible to the tree cap
    (a single tree leaves ample tree headroom) -- the process brake is what
    catches it, and it still drops to 0."""
    procs = [_proc(100, 1, "pytest -n 64")]
    procs += [_proc(110 + i, 100, "python -u -c xdist") for i in range(20)]
    _patch_procs(monkeypatch, *procs)  # 1 tree, 21 processes
    app = _build_app(tmp_path, trees_max=8, processes_max=16)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_pytest_trees == 1
    assert result.dispatch_limit == 0
    assert result.clamped_by == "host_load"
    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert events[0]["payload"]["host_load_term"] == "pytest_processes"


def test_probe_runs_when_only_tree_cap_armed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill-switch granularity: process brake 0 alone must not also
    disable the tree governor -- the probe runs when EITHER knob is > 0
    (both 0 disables it entirely; pinned in test_host_load.py)."""
    calls = []
    monkeypatch.setattr(host_load, "list_host_processes", lambda: calls.append(1) or _lister())
    app = _build_app(tmp_path, trees_max=4, processes_max=0)

    result = app._apply_concurrency_governor(5)

    # Tree term armed alone still probes (the brake being off must not
    # also disable the governor).
    assert calls == [1]
    assert result.host_load_enabled is True
    assert result.dispatch_limit == 4


# ---------------------------------------------------------------------------
# Config: the new knob parses and validates like its sibling
# ---------------------------------------------------------------------------


def test_tree_knob_defaults_track_host_cpu_count() -> None:
    """On by default per #1903: unset, the tree governor sits at half this
    host's logical CPU count (one launch ≈ one suite, so the cap is 'suites
    the host absorbs')."""
    config = build_config_from_data({})
    assert config.dispatch.host_load_max_pytest_trees == (os.cpu_count() or 0) // 2


def test_tree_knob_accepts_int() -> None:
    config = build_config_from_data({"dispatch": {"host_load_max_pytest_trees": 4}})
    assert config.dispatch.host_load_max_pytest_trees == 4


def test_tree_knob_accepts_zero_as_kill_switch() -> None:
    config = build_config_from_data({"dispatch": {"host_load_max_pytest_trees": 0}})
    assert config.dispatch.host_load_max_pytest_trees == 0


@pytest.mark.parametrize("bad", ["4", 4.0, True])
def test_tree_knob_rejects_non_int(bad: object) -> None:
    with pytest.raises(ConfigError, match="host_load_max_pytest_trees.*must be an int"):
        build_config_from_data({"dispatch": {"host_load_max_pytest_trees": bad}})


def test_tree_knob_rejects_negative() -> None:
    with pytest.raises(ConfigError, match="host_load_max_pytest_trees.*must be >= 0"):
        build_config_from_data({"dispatch": {"host_load_max_pytest_trees": -1}})
