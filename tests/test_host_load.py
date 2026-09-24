"""Tests for host-load dispatch backpressure (issue #1843).

Two layers, mirroring the ``test_ci_headroom.py`` / ``test_ci_capacity_
backpressure.py`` split from issue #1770:

* ``pytest_tree_load`` / ``measure_host_load`` / the process listers --
  the measurement itself, exercised against fabricated process snapshots
  with zero subprocess use.
* The ``_apply_concurrency_governor`` wiring -- the clamp engages for
  every dispatch lane (not just fresh dispatch -- a rework launch spawns a
  real local suite too), fails open when the probe errors, writes a
  ``dispatch_backpressure`` event carrying the measured load, and honors
  the ``dispatch.host_load_max_pytest_processes`` kill switch.

``tests/conftest.py``'s autouse ``_no_real_host_load_probe`` fixture stubs
``charlie_work.host_load.list_host_processes`` to an empty snapshot for
every test in the suite; tests here that need a populated snapshot re-patch
the same module attribute inside their own body, which cleanly overrides
the default (``measure_host_load`` resolves it through module globals at
call time).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
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

# Bound at import time: conftest's autouse stub replaces the
# ``host_load.list_host_processes`` module attribute for the duration of
# every test, so tests of the dispatch function itself must reach the real
# object through this captured reference (its own ``sys.platform`` /
# ``quiesce.list_processes`` / ``_list_processes_posix`` lookups still
# resolve through module globals and stay patchable).
_real_list_host_processes = host_load.list_host_processes


def _proc(pid: int, ppid: int, command_line: str, name: str = "") -> quiesce.ProcessInfo:
    return quiesce.ProcessInfo(pid=pid, ppid=ppid, name=name, command_line=command_line)


def _procs(*procs: quiesce.ProcessInfo) -> tuple[quiesce.ProcessInfo, ...]:
    return tuple(procs)


def _lister(
    *procs: quiesce.ProcessInfo, error: str | None = None
) -> tuple[tuple[quiesce.ProcessInfo, ...], str | None]:
    return _procs(*procs), error


# ---------------------------------------------------------------------------
# pytest_tree_load: pure reduction of a snapshot to HostLoad
# ---------------------------------------------------------------------------


def test_pytest_tree_load_empty_snapshot() -> None:
    assert host_load.pytest_tree_load((), self_pid=999) == host_load.HostLoad(0, 0)


def test_pytest_tree_load_counts_single_root() -> None:
    load = host_load.pytest_tree_load(
        _procs(_proc(100, 1, "pytest -q --tb=short")),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


def test_pytest_tree_load_counts_descendants_as_tree_members() -> None:
    """xdist workers (``python -u -c ...``, no pytest token of their own)
    count toward the suite's tree, not as separate trees -- that is what
    makes the count track real suite fan-out."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -n 2"),
            _proc(101, 100, "python -u -c xdist_worker"),
            _proc(102, 100, "python -u -c xdist_worker"),
            _proc(103, 101, "python -c grandchild"),
            _proc(200, 1, "some unrelated service"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=4)


def test_pytest_tree_load_counts_disjoint_trees_once_each() -> None:
    """Two concurrent suites (the incident shape: CI on one runner, a
    worktree suite on another) are two trees; their members never
    double-count even though both snapshots share the host."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -q"),
            _proc(101, 100, "python -u -c worker_a"),
            _proc(200, 1, "python -m pytest tests/"),
            _proc(201, 200, "python -u -c worker_b"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=2, pytest_process_count=4)


def test_pytest_tree_load_nested_pytest_counts_once() -> None:
    """A pytest subprocess inside an outer pytest tree (a test that shells
    out to pytest) folds into the outer tree -- one suite, one tree."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -q"),
            _proc(101, 100, "python -u -c worker"),
            _proc(110, 101, "python -m pytest inner/"),
            _proc(111, 110, "python -u -c inner_worker"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=4)


def test_pytest_tree_load_recognizes_invocation_spellings() -> None:
    """Every spelling the fleet's suites actually take must match: bare
    pytest, python -m pytest, uv run pytest, a Scripts\\pytest.exe path,
    and a bash -c wrapper."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -q"),
            _proc(101, 1, "python -m pytest tests/"),
            _proc(102, 1, "uv run --extra dev pytest -q --tb=short"),
            _proc(103, 1, r'"C:\repo\.venv\Scripts\pytest.exe" tests/test_x.py'),
            _proc(104, 1, 'bash -c "cd repo && pytest -q"'),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=5, pytest_process_count=5)


def test_pytest_tree_load_ignores_non_invocation_matches() -> None:
    """``pytest.ini``, ``pytest-xdist`` (a pip-install cmdline), a test file
    named for pytest, and an unrelated token-embedded name are NOT suite
    roots -- a false positive here would permanently defer dispatch."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "editor pytest.ini"),
            _proc(101, 1, "pip install pytest-xdist"),
            _proc(102, 1, "vim tests/test_pytest.py"),
            _proc(103, 1, "mypytest_runner --serve"),
            _proc(104, 1, "python worker.py"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(0, 0)


def test_pytest_tree_load_excludes_the_tree_containing_self() -> None:
    """A measurement taken from inside a pytest suite (this project's own
    test run calling the function, or the pathological orchestrator-under-
    pytest deployment) must not count its own suite as the external load
    it guards against."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -q"),  # the suite we are inside
            _proc(101, 100, "python -u -c worker"),
            _proc(102, 101, "python test_body"),  # self_pid
            _proc(200, 1, "pytest -n 4"),  # a DIFFERENT suite -- still counts
            _proc(201, 200, "python -u -c other_worker"),
        ),
        self_pid=102,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=2)


def test_pytest_tree_load_excludes_topmost_ancestor_tree() -> None:
    """When self sits under nested pytest trees, the whole OUTERMOST tree
    is excluded -- a nested sibling pytest belongs to the same containing
    suite's load, not to external contention."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest outer"),
            _proc(110, 100, "python -m pytest inner"),  # nested root
            _proc(111, 110, "python test_body"),  # self_pid
            _proc(112, 100, "python -u -c outer_worker"),  # sibling of inner
            _proc(200, 1, "pytest other_suite"),  # unrelated -- counts
        ),
        self_pid=111,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


def test_pytest_tree_load_no_exclusion_when_self_not_under_pytest() -> None:
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -q"),
            _proc(200, 1, "charlie work"),  # the fleet process itself
        ),
        self_pid=200,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


def test_pytest_tree_load_survives_parent_cycles() -> None:
    """A ppid cycle in the snapshot (reused PIDs observed mid-scan) must
    terminate rather than hang the governor."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 101, "pytest -q"),
            _proc(101, 100, "python -u -c worker"),  # 100 <-> 101 cycle
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=2)


def test_pytest_tree_load_tolerates_blank_command_lines() -> None:
    """Win32_Process reports CommandLine=None for some system processes;
    the reducer must not choke on them."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(4, 0, "", "System"),
            _proc(100, 1, "pytest -q"),
        ),
        self_pid=999,
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


# ---------------------------------------------------------------------------
# measure_host_load: probe + fail-open + diagnostic event
# ---------------------------------------------------------------------------


def test_measure_host_load_returns_reading(tmp_path: Path) -> None:
    reading = host_load.measure_host_load(
        lister=lambda: _lister(_proc(100, 1, "pytest -q")),
        self_pid=999,
        diagnostic_state_path=tmp_path / "state.json",
        diagnostic_repo="repo",
    )
    assert reading == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


def test_measure_host_load_fails_open_on_lister_error(tmp_path: Path) -> None:
    """A failed snapshot returns None -- 'do not clamp' -- so a broken
    probe can never hard-stop dispatch."""
    reading = host_load.measure_host_load(
        lister=lambda: _lister(error="powershell exploded"),
        self_pid=999,
        diagnostic_state_path=tmp_path / "state.json",
        diagnostic_repo="repo",
    )
    assert reading is None


def test_measure_host_load_logs_unavailable_event(tmp_path: Path) -> None:
    """The fail-open path is not silent: a host_load_unavailable event
    carrying the reason lands in the diagnostic store."""
    state_file = tmp_path / "state.json"
    reading = host_load.measure_host_load(
        lister=lambda: _lister(error="powershell exploded"),
        self_pid=999,
        diagnostic_state_path=state_file,
        diagnostic_repo="repo",
    )
    assert reading is None
    events = query_events(state_file, kind="host_load_unavailable")
    assert len(events) == 1
    assert events[0]["repo"] == "repo"
    assert events[0]["payload"]["reason"] == "measurement_failed"
    assert "powershell exploded" in events[0]["payload"]["detail"]


def test_measure_host_load_rate_limits_repeated_failures(tmp_path: Path) -> None:
    """The governor calls the probe once per dispatch pass; a stuck probe
    must not write one warning row per pass forever (same dedup discipline
    as ci_headroom_unavailable, issue #1770 finding 2)."""
    state_file = tmp_path / "state.json"
    t0 = datetime.now(UTC)
    for _ in range(3):
        host_load.measure_host_load(
            lister=lambda: _lister(error="still broken"),
            self_pid=999,
            diagnostic_state_path=state_file,
            diagnostic_repo="repo",
            now=t0,
        )
    assert len(query_events(state_file, kind="host_load_unavailable")) == 1

    # Once the interval has elapsed the persistent failure re-reports --
    # 'still broken' must stay visible, just not per-pass.
    host_load.measure_host_load(
        lister=lambda: _lister(error="still broken"),
        self_pid=999,
        diagnostic_state_path=state_file,
        diagnostic_repo="repo",
        now=t0 + timedelta(minutes=host_load.DEFAULT_UNAVAILABLE_INTERVAL_MINUTES + 1),
    )
    assert len(query_events(state_file, kind="host_load_unavailable")) == 2


def test_measure_host_load_no_state_path_still_fails_open() -> None:
    """No diagnostic store configured: log-only, never raises, still None."""
    reading = host_load.measure_host_load(
        lister=lambda: _lister(error="broken"),
        self_pid=999,
    )
    assert reading is None


def test_measure_host_load_uses_module_lister_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``lister`` defaults to this module's ``list_host_processes`` looked
    up at call time -- the seam conftest's autouse stub and wiring tests
    both patch."""
    monkeypatch.setattr(
        host_load,
        "list_host_processes",
        lambda: _lister(_proc(100, 1, "pytest -q")),
    )
    reading = host_load.measure_host_load(self_pid=999)
    assert reading == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


# ---------------------------------------------------------------------------
# list_host_processes / _list_processes_posix: the probe itself
# ---------------------------------------------------------------------------


def test_list_host_processes_dispatches_to_quiesce_on_win32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = _lister(_proc(1, 0, "init"))
    monkeypatch.setattr(host_load.sys, "platform", "win32")
    monkeypatch.setattr(host_load.quiesce, "list_processes", lambda: sentinel)
    assert _real_list_host_processes() == sentinel


def test_list_host_processes_dispatches_to_procfs_elsewhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(host_load.sys, "platform", "linux")
    monkeypatch.setattr(host_load, "_list_processes_posix", lambda: _lister(_proc(1, 0, "init")))
    assert _real_list_host_processes() == _lister(_proc(1, 0, "init"))


def _write_procfs_entry(proc_root: Path, pid: int, ppid: int, cmdline: str) -> None:
    entry = proc_root / str(pid)
    entry.mkdir(parents=True)
    entry.joinpath("cmdline").write_bytes(cmdline.replace(" ", "\0").encode() + b"\0")
    # stat fields: pid (comm) state ppid ... -- comm can legally contain
    # spaces and parens, so ppid must be parsed after the LAST ')'.
    entry.joinpath("stat").write_text(f"{pid} (weird (name)) S {ppid} 0 0 0")


def test_list_processes_posix_reads_fabricated_procfs(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    _write_procfs_entry(proc_root, 100, 1, "pytest -q")
    _write_procfs_entry(proc_root, 101, 100, "python -u -c worker")
    # Non-pid entries (sys/, self, ...) are skipped; a vanished process is
    # skipped rather than failing the whole snapshot.
    (proc_root / "sys").mkdir()
    (proc_root / "200").mkdir()  # no cmdline/stat at all -> skipped

    procs, error = host_load._list_processes_posix(proc_root=proc_root)

    assert error is None
    by_pid = {p.pid: p for p in procs}
    assert by_pid[100].ppid == 1
    assert by_pid[100].command_line == "pytest -q"
    assert by_pid[101].ppid == 100
    assert 200 not in by_pid


def test_list_processes_posix_reports_missing_procfs(tmp_path: Path) -> None:
    procs, error = host_load._list_processes_posix(proc_root=tmp_path / "nope")
    assert procs == ()
    assert error is not None and "does not exist" in error


# ---------------------------------------------------------------------------
# Governor wiring: the host-load clamp in _apply_concurrency_governor
# ---------------------------------------------------------------------------


def _build_app(tmp_path: Path, host_load_max: int, **dispatch_kwargs: Any) -> OrchestratorApp:
    config = OrchestratorConfig(
        dispatch=DispatchConfig(
            host_load_max_pytest_processes=host_load_max,
            default_limit=5,
            **dispatch_kwargs,
        ),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def _patch_procs(monkeypatch: pytest.MonkeyPatch, *procs: quiesce.ProcessInfo) -> None:
    monkeypatch.setattr(host_load, "list_host_processes", lambda: _lister(*procs))


def _saturated_snapshot() -> list[quiesce.ProcessInfo]:
    """The incident shape from #1843: ~18 pytest processes across several
    trees (two CI suites at -n 8 fan-out, plus worktree suites)."""
    procs = [_proc(100, 1, "uv run --extra dev pytest -n auto -q --tb=short")]
    procs += [_proc(110 + i, 100, "python -u -c xdist") for i in range(8)]
    procs.append(_proc(200, 1, "pytest -n auto"))
    procs += [_proc(210 + i, 200, "python -u -c xdist") for i in range(8)]
    return procs


def test_host_load_clamps_dispatch_to_zero_when_saturated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Over the threshold -> dispatch_limit clamps to 0 with clamped_by
    naming host_load -- a deferral, not a failure; the next pass retries."""
    _patch_procs(monkeypatch, *_saturated_snapshot())
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_enabled is True
    assert result.host_load_pytest_processes == 18
    assert result.host_load_pytest_trees == 2
    assert result.dispatch_limit == 0
    assert result.clamped is True
    assert result.clamped_by == "host_load"


def test_host_load_applies_to_fresh_dispatch_lane_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_open_pr_backpressure=True (the fresh-dispatch call shape)
    clamps identically -- the term is not gated on that flag."""
    _patch_procs(monkeypatch, *_saturated_snapshot())
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(5, apply_open_pr_backpressure=True)

    assert result.dispatch_limit == 0
    assert result.clamped_by == "host_load"


def test_host_load_does_not_clamp_under_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single CI suite running alone is normal fleet activity -- the
    clamp exists for oversubscription, not for any nonzero load."""
    _patch_procs(
        monkeypatch,
        _proc(100, 1, "pytest -n 2"),
        _proc(101, 100, "python -u -c w1"),
        _proc(102, 100, "python -u -c w2"),
    )
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_pytest_processes == 3
    assert result.host_load_pytest_trees == 1
    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.clamped_by is None


def test_host_load_threshold_is_strictly_greater(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A count equal to the threshold does not defer -- 'exceeds' per the
    issue's wording, so the knob's value is the largest load that still
    permits a launch."""
    _patch_procs(
        monkeypatch,
        _proc(100, 1, "pytest -q"),
        _proc(200, 1, "pytest -q"),
    )
    app = _build_app(tmp_path, host_load_max=2)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_host_load_zero_disables_probe_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kill switch: host_load_max_pytest_processes=0 must not even
    spawn the process listing -- off means off, not 'probe then ignore'."""
    calls = []
    monkeypatch.setattr(host_load, "list_host_processes", lambda: calls.append(1) or _lister())
    app = _build_app(tmp_path, host_load_max=0)

    result = app._apply_concurrency_governor(5)

    assert calls == []
    assert result.host_load_enabled is False
    assert result.host_load_pytest_processes is None
    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_host_load_probe_skipped_when_limit_already_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass already clamped to 0 by an earlier term cannot launch
    anyway -- the probe (a real subprocess spawn) must not run. This is
    also what makes a saturated loop pay only ONE probe per pass: the
    wave-budget call clamps to 0, and the rework/fresh calls inside it
    then skip the probe on the same condition."""
    calls = []
    monkeypatch.setattr(host_load, "list_host_processes", lambda: calls.append(1) or _lister())
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(0)

    assert calls == []
    assert result.dispatch_limit == 0
    assert result.host_load_pytest_processes is None


def test_host_load_fails_open_when_probe_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken probe (PowerShell timeout, missing /proc) fails OPEN --
    dispatch proceeds -- and reports host_load_unavailable to this repo's
    own events.db, so 'deferral never fires' is diagnosable instead of
    silently absent."""
    monkeypatch.setattr(host_load, "list_host_processes", lambda: ((), "powershell exploded"))
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert result.host_load_pytest_processes is None
    events = query_events(app.paths.state_file, kind="host_load_unavailable")
    assert len(events) == 1
    assert events[0]["repo"] == app.repo_root.name
    assert events[0]["payload"]["reason"] == "measurement_failed"


def test_host_load_deferral_event_carries_measured_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion: each deferral emits an event with the measured
    load -- the dispatch_backpressure event names host_load and carries the
    process count, tree count, and threshold so '0 dispatched' is
    explainable from events.db alone."""
    _patch_procs(monkeypatch, *_saturated_snapshot())
    app = _build_app(tmp_path, host_load_max=8)

    app._apply_concurrency_governor(5)

    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["clamped_by"] == "host_load"
    assert payload["host_load_pytest_processes"] == 18
    assert payload["host_load_pytest_trees"] == 2
    assert payload["host_load_max_pytest_processes"] == 8
    assert payload["requested_limit"] == 5
    assert payload["clamped_limit"] == 0


def test_host_load_dry_run_clamps_but_suppresses_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dry_run still applies the clamp (a preview must match a live pass)
    but must not write the durable event -- same write-suppression
    discipline as the open_pr_max/ci_headroom clamps."""
    _patch_procs(monkeypatch, *_saturated_snapshot())
    config = OrchestratorConfig(
        dispatch=DispatchConfig(host_load_max_pytest_processes=8, default_limit=5),
        devin=DevinConfig(),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 0
    assert result.clamped is True
    assert result.clamped_by == "host_load"
    assert query_events(app.paths.state_file, kind="dispatch_backpressure") == []


def test_host_load_only_term_enabled_surfaces_in_dispatch_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo running ONLY the host-load term (every other governor knob
    at its default) must still see the clamp explained in a real
    ``dispatch()`` call's ``CommandResult.data`` -- the issue #1770
    finding-1 regression class (a hand-rolled ``or`` chain at the call
    sites silently dropped new terms), pinned at the dispatch boundary."""
    _patch_procs(monkeypatch, *_saturated_snapshot())
    app = _build_app(tmp_path, host_load_max=8)

    result = app.dispatch()

    assert result.ok is True
    # FakeGitHub's default issue (#123) is deferred by the clamp.
    assert result.data["selected_count"] == 0
    assert result.data["clamped_by"] == "host_load"
    assert result.data["host_load_pytest_processes"] == 18
    assert result.data["host_load_pytest_trees"] == 2
    assert result.data["host_load_max_pytest_processes"] == 8


# ---------------------------------------------------------------------------
# Config: the knob ships on, and 0 is the kill switch
# ---------------------------------------------------------------------------


def test_host_load_knob_defaults_to_host_cpu_count() -> None:
    """On by default per the issue's acceptance criteria: unset, the
    threshold is this host's logical CPU count ('more runnable test
    processes than cores' is the built-in saturation line)."""
    config = build_config_from_data({})
    assert config.dispatch.host_load_max_pytest_processes == (os.cpu_count() or 0)


def test_host_load_knob_accepts_int() -> None:
    config = build_config_from_data({"dispatch": {"host_load_max_pytest_processes": 24}})
    assert config.dispatch.host_load_max_pytest_processes == 24


def test_host_load_knob_accepts_zero_as_kill_switch() -> None:
    config = build_config_from_data({"dispatch": {"host_load_max_pytest_processes": 0}})
    assert config.dispatch.host_load_max_pytest_processes == 0


@pytest.mark.parametrize("bad", ["8", 8.0, True])
def test_host_load_knob_rejects_non_int(bad: object) -> None:
    with pytest.raises(ConfigError, match="host_load_max_pytest_processes.*must be an int"):
        build_config_from_data({"dispatch": {"host_load_max_pytest_processes": bad}})


def test_host_load_knob_rejects_negative() -> None:
    with pytest.raises(ConfigError, match="host_load_max_pytest_processes.*must be >= 0"):
        build_config_from_data({"dispatch": {"host_load_max_pytest_processes": -1}})
