"""Orchestrator-attribution scope for host-load measurement (issue #1943).

Sibling of ``tests/test_host_load.py`` (attachment point saturated) and
``tests/test_host_load_tree_headroom.py`` -- the same split pattern used for
the #1903 recalibration. This module covers the #1943 scoping layer:

* ``pytest_tree_load(scope_paths=...)`` counts only trees that are
  *orchestrator-attributable*: some member's -- or some member-ancestor's --
  command line references a managed path (the ``.var/charlie-work`` state-dir
  convention marker or a caller-supplied scope path such as a resolved
  worktrees/state root or a fleet-registered ``state_dir``).
* CI-runner suites (``C:\\actions-runners\\*``) and other unattributable
  trees feed neither count -- the incident: three swole runners mid-suite
  (~50 xdist processes) tripped the 48-process brake and zeroed every other
  repo's dispatch for 13+ hours on a sub-40%-CPU host.
* ``measure_host_load`` applies the scope end-to-end (built-in marker plus
  caller paths); ``scope_paths=None`` on ``pytest_tree_load`` preserves the
  pre-#1943 host-wide reading for diagnostics/tests.
* The governor-facing wiring: a CI-only snapshot reads 0 and never clamps,
  a mixed snapshot counts only the managed side, and a foreign repo's
  ``runtime.state_dir`` override stays attributable through the fleet
  registry seam (``fleet_registry.registered_state_dirs``).

Self-contained per the zero-cross-test-import guard
(``tests/test_zero_cross_test_import_guard.py``): the tiny helpers below
duplicate the sibling modules' ``_proc``/``_lister``/``_patch_procs`` rather
than importing them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHub
from charlie_work import fleet_registry, host_load, quiesce
from charlie_work.config import (
    ClaudeCodeConfig,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

# The managed-path spelling every default-layout repo shares -- the marker
# measure_host_load builds in. Tests spell it literally (the
# test_no_path_literals rules scan src/ only).
_STATE_MARKER = ".var/charlie-work"


def _proc(pid: int, ppid: int, command_line: str, name: str = "") -> quiesce.ProcessInfo:
    return quiesce.ProcessInfo(pid=pid, ppid=ppid, name=name, command_line=command_line)


def _procs(*procs: quiesce.ProcessInfo) -> tuple[quiesce.ProcessInfo, ...]:
    return tuple(procs)


def _lister(
    *procs: quiesce.ProcessInfo, error: str | None = None
) -> tuple[tuple[quiesce.ProcessInfo, ...], str | None]:
    return tuple(procs), error


def _patch_procs(monkeypatch: pytest.MonkeyPatch, *procs: quiesce.ProcessInfo) -> None:
    monkeypatch.setattr(host_load, "list_host_processes", lambda: _lister(*procs))


def _ci_runner_tree(
    root_pid: int = 400, launcher_pid: int = 300, workers: int = 8
) -> list[quiesce.ProcessInfo]:
    """Fabricate a GitHub Actions self-hosted-runner pytest tree -- the
    swole-runner shape behind #1943 (``C:\\actions-runners\\*\\_work``)."""
    return [
        _proc(launcher_pid, 1, r"C:\actions-runners\swole-2\_work\_tool\Runner.Worker.exe"),
        _proc(
            root_pid,
            launcher_pid,
            r"C:\actions-runners\swole-2\_work\mdls\mdls\.venv\Scripts\pytest.exe -n auto",
        ),
        *[_proc(root_pid + 10 + i, root_pid, "python -u -c xdist") for i in range(workers)],
    ]


def _build_app(
    tmp_path: Path,
    *,
    host_load_max: int = 0,
    trees_max: int = 0,
    fleet_dir_override: str | None = None,
    worktrees_dir: str | None = None,
    **dispatch_kwargs: Any,
) -> OrchestratorApp:
    """Build an app with explicit host-load knobs (both default 0 so each
    test opts into exactly the term it exercises)."""
    config = OrchestratorConfig(
        dispatch=DispatchConfig(
            host_load_max_pytest_processes=host_load_max,
            host_load_max_pytest_trees=trees_max,
            default_limit=5,
            **dispatch_kwargs,
        ),
        devin=DevinConfig(),
        claude_code=ClaudeCodeConfig(worktrees_dir=worktrees_dir),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(
        tmp_path, paths, config, FakeGitHub(), fleet_dir_override=fleet_dir_override
    )


# ---------------------------------------------------------------------------
# pytest_tree_load scope_paths: attribution semantics
# ---------------------------------------------------------------------------


def test_scoped_load_excludes_ci_runner_tree() -> None:
    """The #1943 incident shape: a CI-runner suite under
    ``C:\\actions-runners\\*`` is not orchestrator load -- nothing in any
    member's or ancestor's command line references a managed path."""
    load = host_load.pytest_tree_load(
        _procs(*_ci_runner_tree()),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(0, 0)


def test_scoped_load_counts_managed_worktree_tree() -> None:
    """A suite rooted at a managed worktree's venv interpreter counts in
    full -- root plus pathless xdist descendants."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(
                100,
                1,
                r"C:\repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -n 2",
            ),
            _proc(101, 100, "python -u -c w"),
            _proc(102, 100, "python -u -c w"),
        ),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=3)


def test_scoped_load_attributes_pathless_root_via_ancestor() -> None:
    """The dominant production shape: ``uv run pytest`` / ``python -m
    pytest`` roots carry no managed path in their own command line, so
    attribution walks the ancestry to the worker-harness launcher (whose
    ``--prompt-file`` names a state-dir path)."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(
                50,
                1,
                "devin --prompt-file /repo/.var/charlie-work/dispatches/s1/prompt.md --print",
            ),
            _proc(60, 50, "uv run --extra dev pytest -q"),
            _proc(61, 60, "python -u -c w"),
        ),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=2)


def test_scoped_load_ancestor_walk_spans_multiple_hops() -> None:
    """Attribution is not single-hop: a root several levels under the
    path-carrying launcher still counts (wrapper shells, env chains)."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(50, 1, "devin --prompt-file /repo/.var/charlie-work/dispatches/s1/p.md"),
            _proc(55, 50, "bash worker-entry.sh"),
            _proc(58, 55, "bash -c run-suite"),
            _proc(60, 58, "pytest -q"),
        ),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


def test_scoped_load_counts_only_managed_trees_in_mixed_snapshot() -> None:
    """The incident verbatim: a ~50-process CI fan-out beside managed
    suites -- only the managed side feeds either count."""
    procs = [
        *_ci_runner_tree(root_pid=400, launcher_pid=300, workers=50),
        _proc(
            100,
            1,
            r"C:\repo\.var\charlie-work\worktrees\wt-a\.venv\Scripts\pytest.exe -q",
        ),
        _proc(
            50,
            1,
            "devin --prompt-file /repo/.var/charlie-work/dispatches/s2/prompt.md --print",
        ),
        _proc(200, 50, "uv run pytest -n 2"),
        _proc(201, 200, "python -u -c w"),
    ]
    load = host_load.pytest_tree_load(procs, self_pid=999, scope_paths=(_STATE_MARKER,))
    assert load == host_load.HostLoad(pytest_tree_count=2, pytest_process_count=3)


def test_scoped_load_marker_requires_path_token_boundaries() -> None:
    """Near-miss spellings must not attribute: ``.var/charlie-work-legacy``
    (trailing extension) and ``my.var/charlie-work`` (leading garbage) are
    not the state-dir marker."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(
                100,
                1,
                r"C:\repo\.var\charlie-work-legacy\worktrees\wt\.venv\Scripts\pytest.exe -q",
            ),
            _proc(
                200,
                1,
                r"C:\repo2\my.var\charlie-work\wt\.venv\Scripts\pytest.exe -q",
            ),
        ),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(0, 0)


def test_scoped_load_caller_scope_path_attributes_override_layout() -> None:
    """A ``runtime.state_dir`` override outside ``.var/charlie-work`` is
    attributable only through a caller-supplied scope path -- the shape the
    governor passes as resolved roots / fleet-registered state_dirs."""
    procs = _procs(
        _proc(50, 1, "devin --prompt-file /opt/charlie-state/dispatches/s1/p.md --print"),
        _proc(60, 50, "python -m pytest -q"),
    )
    assert host_load.pytest_tree_load(
        procs, self_pid=999, scope_paths=("/opt/charlie-state", _STATE_MARKER)
    ) == host_load.HostLoad(1, 1)
    assert host_load.pytest_tree_load(
        procs, self_pid=999, scope_paths=(_STATE_MARKER,)
    ) == host_load.HostLoad(0, 0)


def test_scoped_load_caller_scope_path_requires_boundary() -> None:
    """Caller scope ``/opt/charlie-state`` must not attribute a cmdline
    under ``/opt/charlie-state-extra`` -- same token-boundary discipline as
    the built-in marker."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(60, 50, "python -m pytest -q"),
            _proc(50, 1, "devin --prompt-file /opt/charlie-state-extra/d/p.md --print"),
        ),
        self_pid=999,
        scope_paths=("/opt/charlie-state",),
    )
    assert load == host_load.HostLoad(0, 0)


def test_scoped_load_empty_scope_attributes_nothing() -> None:
    """An empty scope list is a real (if degenerate) scope -- nothing
    counts, matching the fail-quiet direction of the rest of the feature."""
    load = host_load.pytest_tree_load(
        _procs(_proc(100, 1, "pytest -q")),
        self_pid=999,
        scope_paths=(),
    )
    assert load == host_load.HostLoad(0, 0)


def test_unscoped_load_counts_everything() -> None:
    """``scope_paths=None`` preserves the pre-#1943 host-wide reading --
    the diagnostics/test seam the module keeps deliberately."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, "pytest -q"),
            *_ci_runner_tree(root_pid=400, launcher_pid=300, workers=2),
        ),
        self_pid=999,
        scope_paths=None,
    )
    assert load == host_load.HostLoad(pytest_tree_count=2, pytest_process_count=4)


def test_scoped_load_self_exclusion_still_applies() -> None:
    """Self-exclusion composes with scoping: the tree containing the
    measurer is dropped even when it is attributable."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(100, 1, r"C:\repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -q"),
            _proc(101, 100, "python test_body"),
            _proc(200, 1, r"C:\repo\.var\charlie-work\worktrees\wt2\.venv\Scripts\pytest.exe -q"),
        ),
        self_pid=101,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=1)


def test_scoped_load_nested_roots_still_merge() -> None:
    """#1918 merging composes with scoping: a nested managed root is one
    tree with its outer root, while a disjoint unattributable root still
    drops out."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(
                500,
                1,
                r"C:\repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -q",
            ),
            _proc(100, 500, "python -m pytest inner/"),  # nested, lower pid
            _proc(
                600,
                1,
                r"C:\actions-runners\swole-1\_work\r\r\.venv\Scripts\pytest.exe -q",
            ),
        ),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=2)


def test_scoped_load_merged_tree_attributes_when_any_member_carries_path() -> None:
    """A merged tree counts when ANY member/ancestor matches: a nested
    pathless inner root does not forfeit the attribution its outer
    root earned."""
    load = host_load.pytest_tree_load(
        _procs(
            _proc(500, 1, r"C:\repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -q"),
            _proc(100, 500, "python -m pytest inner/"),
            _proc(111, 100, "python -u -c inner_worker"),
        ),
        self_pid=999,
        scope_paths=(_STATE_MARKER,),
    )
    assert load == host_load.HostLoad(pytest_tree_count=1, pytest_process_count=3)


# ---------------------------------------------------------------------------
# measure_host_load: the governor-facing scoped reading
# ---------------------------------------------------------------------------


def test_measure_host_load_excludes_unattributable_trees(tmp_path: Path) -> None:
    """End-to-end: the probe the governor calls never counts a CI-runner
    suite -- the incident's 50-61 process readings now read 0."""
    reading = host_load.measure_host_load(
        lister=lambda: _lister(*_ci_runner_tree()),
        self_pid=999,
        diagnostic_state_path=tmp_path / "state.json",
        diagnostic_repo="repo",
    )
    assert reading == host_load.HostLoad(0, 0)


def test_measure_host_load_builtin_marker_counts_default_layout(tmp_path: Path) -> None:
    """The built-in state-dir marker covers every default-layout repo --
    no caller scope path needed for ``.var/charlie-work`` trees."""
    reading = host_load.measure_host_load(
        lister=lambda: _lister(
            _proc(
                100,
                1,
                r"C:\other-repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -q",
            )
        ),
        self_pid=999,
        diagnostic_state_path=tmp_path / "state.json",
        diagnostic_repo="repo",
    )
    assert reading == host_load.HostLoad(1, 1)


def test_measure_host_load_caller_scope_paths_cover_overrides(tmp_path: Path) -> None:
    """``scope_paths`` layers caller-supplied managed roots (resolved
    state/worktrees dirs, registered state_dirs) onto the marker -- the
    path a ``runtime.state_dir`` override moves state to."""
    override_state = tmp_path / "custom-state"
    reading = host_load.measure_host_load(
        lister=lambda: _lister(
            _proc(
                100,
                1,
                f"{override_state}/worktrees/wt/.venv/Scripts/pytest.exe -q",
            )
        ),
        self_pid=999,
        scope_paths=(override_state,),
        diagnostic_state_path=tmp_path / "state.json",
        diagnostic_repo="repo",
    )
    assert reading == host_load.HostLoad(1, 1)


def test_measure_host_load_scoped_and_fail_open_unchanged(tmp_path: Path) -> None:
    """Scoping does not touch the fail-open path: a lister error is still
    None + one host_load_unavailable event."""
    reading = host_load.measure_host_load(
        lister=lambda: _lister(error="powershell exploded"),
        self_pid=999,
        scope_paths=(tmp_path / "state",),
        diagnostic_state_path=tmp_path / "state.json",
        diagnostic_repo="repo",
    )
    assert reading is None
    events = query_events(tmp_path / "state.json", kind="host_load_unavailable")
    assert len(events) == 1


# ---------------------------------------------------------------------------
# registered_state_dirs: the fleet-registry scope seam
# ---------------------------------------------------------------------------


def _write_fleet_registry(fleet_root: Path, repos: dict[str, Any]) -> None:
    fleet_root.mkdir(parents=True, exist_ok=True)
    (fleet_root / "fleet.json").write_text(
        json.dumps({"version": 1, "repos": repos}), encoding="utf-8"
    )


def test_registered_state_dirs_returns_entry_state_dirs(tmp_path: Path) -> None:
    """Every registered repo's ``state_dir`` is returned; entries with a
    missing/blank value or a malformed shape are skipped."""
    fleet_root = tmp_path / "fleet"
    _write_fleet_registry(
        fleet_root,
        {
            "owner/a": {"state_dir": str(tmp_path / "s1")},
            "owner/b": {"state_dir": str(tmp_path / "s2")},
            "owner/c": {},
            "owner/d": {"state_dir": ""},
            "owner/e": "not-a-dict",
        },
    )
    assert fleet_registry.registered_state_dirs(str(fleet_root)) == (
        tmp_path / "s1",
        tmp_path / "s2",
    )


def test_registered_state_dirs_missing_registry_is_empty(tmp_path: Path) -> None:
    """No fleet.json (never registered, or a host with a single repo) ->
    empty scope contribution, never an error."""
    assert fleet_registry.registered_state_dirs(str(tmp_path / "nope")) == ()


def test_registered_state_dirs_corrupt_registry_is_empty(tmp_path: Path) -> None:
    """A corrupt registry contributes nothing rather than failing the
    governor's scope build."""
    fleet_root = tmp_path / "fleet"
    fleet_root.mkdir()
    (fleet_root / "fleet.json").write_text("{ not json", encoding="utf-8")
    assert fleet_registry.registered_state_dirs(str(fleet_root)) == ()


# ---------------------------------------------------------------------------
# Governor wiring: scoped counts drive the clamps
# ---------------------------------------------------------------------------


def test_governor_does_not_clamp_on_ci_runner_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The #1943 regression: 51 CI-runner processes above the process
    brake must not touch dispatch -- the scoped reading is 0 and no clamp
    or event fires."""
    _patch_procs(monkeypatch, *_ci_runner_tree(workers=50))
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_pytest_processes == 0
    assert result.host_load_pytest_trees == 0
    assert result.dispatch_limit == 5
    assert result.clamped is False
    assert query_events(app.paths.state_file, kind="dispatch_backpressure") == []


def test_governor_counts_only_managed_side_of_mixed_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI fan-out beside a small managed suite: the reading names the
    managed side only (3 procs, 1 tree) and the brake stays quiet."""
    procs = [
        *_ci_runner_tree(workers=50),
        _proc(
            100,
            1,
            r"C:\repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -n 2",
        ),
        _proc(101, 100, "python -u -c w"),
        _proc(102, 100, "python -u -c w"),
    ]
    _patch_procs(monkeypatch, *procs)
    app = _build_app(tmp_path, host_load_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_pytest_processes == 3
    assert result.host_load_pytest_trees == 1
    assert result.dispatch_limit == 5
    assert result.clamped is False


def test_governor_process_brake_still_binds_on_managed_fan_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The brake keeps its job on the managed side: one attributable suite
    at pathological width still hard-clamps to 0."""
    procs = [
        _proc(
            100,
            1,
            r"C:\repo\.var\charlie-work\worktrees\wt\.venv\Scripts\pytest.exe -n 64",
        ),
        *[_proc(110 + i, 100, "python -u -c xdist") for i in range(20)],
    ]
    _patch_procs(monkeypatch, *procs)
    app = _build_app(tmp_path, host_load_max=16, trees_max=8)

    result = app._apply_concurrency_governor(5)

    assert result.dispatch_limit == 0
    assert result.clamped_by == "host_load"
    events = query_events(app.paths.state_file, kind="dispatch_backpressure")
    assert events[0]["payload"]["host_load_term"] == "pytest_processes"


def test_governor_attributes_via_fleet_registered_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sibling repo whose ``state_dir`` sits outside ``.var/charlie-work``
    (a ``runtime.state_dir`` override) stays attributable through the
    fleet-registry seam the governor wires in -- its suite still counts."""
    fleet_root = tmp_path / "fleet"
    override_state = tmp_path / "sibling-state"
    _write_fleet_registry(fleet_root, {"owner/sibling": {"state_dir": str(override_state)}})
    _patch_procs(
        monkeypatch,
        _proc(50, 1, f"devin --prompt-file {override_state}/dispatches/s1/p.md --print"),
        _proc(60, 50, "uv run pytest -q"),
    )
    app = _build_app(tmp_path, trees_max=1, fleet_dir_override=str(fleet_root))

    result = app._apply_concurrency_governor(5)

    # The sibling's tree counts (trees=1 at cap 1 -> headroom 0).
    assert result.host_load_pytest_trees == 1
    assert result.dispatch_limit == 0
    assert result.clamped_by == "host_load"


def test_governor_attributes_pathless_root_via_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end for the dominant launch shape: a ``uv run pytest`` root
    under a worker-harness ancestor (``--prompt-file`` under this repo's
    own resolved state root) counts toward the governor's reading."""
    _patch_procs(
        monkeypatch,
        _proc(50, 1, "devin --prompt-file /repo/.var/charlie-work/dispatches/s1/p.md --print"),
        _proc(60, 50, "uv run pytest -q"),
    )
    app = _build_app(tmp_path, trees_max=1)

    result = app._apply_concurrency_governor(5)

    assert result.host_load_pytest_trees == 1
    assert result.dispatch_limit == 0
    assert result.clamped_by == "host_load"


def test_governor_attributes_via_worktrees_dir_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``claude_code.worktrees_dir`` override path of the governor's
    scope tuple. A suite whose only managed-path reference lives under the
    overridden worktrees root -- carrying neither the ``.var/charlie-work``
    marker nor any path under the resolved state root -- is attributable
    only through ``self._layout.worktrees``, the attribute reap_dispatch's
    ``scope_paths`` wires in. ``RuntimePaths.worktrees`` is deliberately
    override-blind (``paths.py``: "this member deliberately does NOT honour
    ``claude_code.worktrees_dir``"), so a swap to ``self.paths.worktrees``
    would silently read this tree as host noise and never clamp -- the
    regression this test pins."""
    # The override must live where no other managed-path marker reaches:
    # neither under the built-in ``.var/charlie-work`` convention marker
    # nor under this repo's resolved state root. Anchoring at the
    # filesystem root (``C:\`` / ``/``) guarantees that -- anything under
    # tmp_path inherits the marker here because the worktree itself sits
    # inside ``<repo>/.var/charlie-work/worktrees/``. The path never needs
    # to exist: scope matching is regex over command-line text, so nothing
    # stats it.
    override_worktrees = Path(tmp_path.anchor) / "wt-outside-state"
    assert _STATE_MARKER not in str(override_worktrees).replace("\\", "/")
    _patch_procs(
        monkeypatch,
        _proc(
            100,
            1,
            f"{override_worktrees}/wt-a/.venv/Scripts/pytest.exe -n 2",
        ),
        _proc(101, 100, "python -u -c w"),
    )
    app = _build_app(tmp_path, trees_max=1, worktrees_dir=str(override_worktrees))

    result = app._apply_concurrency_governor(5)

    # The override-dir suite counts (trees=1 at cap 1 -> headroom 0).
    assert result.host_load_pytest_trees == 1
    assert result.host_load_pytest_processes == 2
    assert result.dispatch_limit == 0
    assert result.clamped_by == "host_load"
