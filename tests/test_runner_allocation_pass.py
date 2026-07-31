"""Direct coverage of ``run_allocation_pass`` — the function that actuates.

Every other test reaches ``run_allocation_pass`` by patching it out and
asserting *that it was called*. Its own body — the early-return ladder, the
busy-runner error handling, the ``dry_run`` write guard — was exercised by
nothing. These tests drive the real function with a fake ``GitHub`` and a temp
``managed_root`` so the two safety properties CLAUDE.md calls out are covered at
the layer that composes them:

- **Never stop a busy listener.** The ``busy_error`` path pins a repo whose
  runner list is unreadable rather than parking anything in it.
- **Never actuate on a dry run.** The ``if not dry_run:`` guard around
  ``save_idle_streaks`` is the only thing keeping a simulated pass from
  advancing hysteresis state (issue #605).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import RunnerAllocationConfig
from charlie_work.github import GitHubError
from charlie_work.runner_allocation import SlotAction
from charlie_work.runner_allocation_pass import run_allocation_pass
from charlie_work.runner_slots import (
    ALLOCATION_STATE_FILENAME,
    load_allocation_stamp,
    load_idle_streaks,
)

CW = "Senkichi/charlie-work"
JC = "Senkichi/job-cannon"

PASS_INTERVAL = 300  # seconds; arbitrary but consistent allocation pass cadence


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


def _repo_from_path(path: str) -> str:
    """Extract "owner/name" from a ``repos/{repo}/actions/...`` API path."""
    return path.split("repos/", 1)[1].split("/actions", 1)[0]


def _run_id_from_path(path: str) -> int:
    """Extract the run id from a ``repos/{repo}/actions/runs/{id}/jobs`` path."""
    return int(path.split("/runs/", 1)[1].split("/", 1)[0])


class _FakeGitHub:
    """Minimal stand-in for ``GitHub.run()`` used only by the allocation pass.

    The pass addresses repos by explicit slug and only calls
    ``gh.run(["api", ...], json_output=True)``. This stub dispatches on the URL
    path so one instance can answer every repo's runners, runs, and jobs
    endpoints, and can be primed to raise ``GitHubError`` for a specific repo's
    runners list (the ``busy_error`` path that pins an unmeasurable repo).
    """

    def __init__(
        self,
        *,
        busy_runners: dict[str, set[str]] | None = None,
        runs: dict[str, list[dict[str, Any]]] | None = None,
        jobs: dict[int, list[dict[str, Any]]] | None = None,
        runners_error_repos: set[str] | None = None,
    ) -> None:
        self._busy_runners = busy_runners or {}
        self._runs = runs or {}
        self._jobs = jobs or {}
        self._runners_error_repos = runners_error_repos or set()
        self.calls: list[list[str]] = []

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        self.calls.append(args)
        path = args[1] if len(args) > 1 else ""

        if "/actions/runners" in path:
            repo = _repo_from_path(path)
            if repo in self._runners_error_repos:
                raise GitHubError(f"runners list unreadable for {repo}")
            names = self._busy_runners.get(repo, set())
            return {"runners": [{"name": n, "busy": True} for n in sorted(names)]}

        if "/actions/runs/" in path and "/jobs" in path:
            run_id = _run_id_from_path(path)
            return {"jobs": self._jobs.get(run_id, [])}

        if "/actions/runs?" in path:
            repo = _repo_from_path(path)
            return {"workflow_runs": self._runs.get(repo, [])}

        raise AssertionError(f"unexpected gh api call: {args}")


def _make_runner_dir(root: Path, name: str, repo: str, agent_name: str | None = None) -> Path:
    """Create a minimal runner directory that ``discover_runner_instances`` accepts."""
    path = root / name
    path.mkdir(parents=True)
    (path / "run.cmd").write_text("@echo off\n", encoding="utf-8")
    payload = json.dumps(
        {
            "agentName": agent_name or name,
            "gitHubUrl": f"https://github.com/{repo}",
            "workFolder": "_work",
        }
    )
    (path / ".runner").write_text(payload, encoding="utf-8-sig")
    return path


def _patch_liveness(monkeypatch: pytest.MonkeyPatch, running_dirs: set[str]) -> None:
    """Make discovery and actuation see ``running_dirs`` as having live listeners.

    ``is_runner_launched`` is the single liveness check both
    ``discover_runner_instances`` (to set ``running``) and ``apply_allocation``
    (to re-check before starting) call. Pointing it at a name set keeps the two
    calls in agreement without spawning real processes.
    """

    def _is_launched(runner_dir: Path) -> bool:
        return runner_dir.name in running_dirs

    monkeypatch.setattr("charlie_work.runner_slots.is_runner_launched", _is_launched)


def _allocation_state_path(fleet_dir: Path) -> Path:
    return fleet_dir / ALLOCATION_STATE_FILENAME


# --------------------------------------------------------------------------
# Early-return ladder
# --------------------------------------------------------------------------


def test_disabled_config_returns_skipped_without_touching_github() -> None:
    """``enabled=False`` short-circuits before any discovery or gh call."""
    gh = _FakeGitHub(runners_error_repos={CW})  # would raise if reached

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=False),
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    assert result.skipped is True
    assert result.plan is None
    assert any("disabled" in note for note in result.notes)
    assert gh.calls == []


def test_unresolvable_managed_root_returns_error_without_raising(
    tmp_path: Path,
) -> None:
    """No managed root and no fallback is a config error, not a crash."""
    fleet_dir = tmp_path / "fleet"
    result = run_allocation_pass(
        _FakeGitHub(),
        RunnerAllocationConfig(enabled=True),  # managed_root empty, no fallback
        fleet_dir_override=str(fleet_dir),
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is False
    assert result.error is not None
    assert "managed_root" in result.error

    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "prologue"
    assert stamp.full_pass_interval_seconds == PASS_INTERVAL
    assert stamp.skip_reason is not None
    assert "managed_root" in stamp.skip_reason


def test_nonexistent_managed_root_returns_error(tmp_path: Path) -> None:
    """A typo'd path fails loudly instead of silently allocating nothing."""
    fleet_dir = tmp_path / "fleet"
    result = run_allocation_pass(
        _FakeGitHub(),
        RunnerAllocationConfig(enabled=True, managed_root=str(tmp_path / "gone")),
        fleet_dir_override=str(fleet_dir),
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is False
    assert result.error is not None
    assert "does not exist" in result.error

    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "prologue"
    assert stamp.full_pass_interval_seconds == PASS_INTERVAL
    assert stamp.skip_reason is not None
    assert "does not exist" in stamp.skip_reason


def test_zero_discovered_instances_returns_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty managed root is a no-op with a visible note, not an error."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    fleet_dir = tmp_path / "fleet"

    result = run_allocation_pass(
        _FakeGitHub(),
        RunnerAllocationConfig(enabled=True, managed_root=str(root)),
        fleet_dir_override=str(fleet_dir),
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    assert result.skipped is True
    assert any("no configured runners" in note for note in result.notes)
    # A skip file is written so the doctor probe can tell this pass reached
    # allocation and found nothing, rather than assuming the daemon was stuck.
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "prologue"
    assert stamp.full_pass_interval_seconds == PASS_INTERVAL
    assert stamp.skip_reason is not None
    assert "no configured runners" in stamp.skip_reason


# --------------------------------------------------------------------------
# dry_run write guard
# --------------------------------------------------------------------------


def _converged_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, _FakeGitHub]:
    """One repo with one running listener and zero demand — a plan with no changes."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", CW)
    _patch_liveness(monkeypatch, {"cw-1"})

    gh = _FakeGitHub()  # no busy runners, no runs → demand 0
    return root, tmp_path / "fleet", gh


def test_dry_run_writes_no_state_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A preview must not advance hysteresis state (issue #605).

    The ``if not dry_run:`` guard around ``save_idle_streaks`` is the only thing
    keeping a simulated pass from changing later real decisions. Without it, a
    dry run that incremented the idle streak would let the next real pass park a
    slot based on slack the host never actually observed.
    """
    root, fleet_dir, gh = _converged_setup(tmp_path, monkeypatch)

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(fleet_dir),
        dry_run=True,
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    assert result.skipped is False
    assert result.results == ()
    assert not _allocation_state_path(fleet_dir).exists()


def test_real_run_with_converged_plan_writes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real pass with nothing to move still stamps the state file's source.

    The doctor probe in #595 reads ``load_allocation_stamp`` to tell an
    unattended pass from an operator's manual one. A converged host is the
    common case, so provenance must be written even when no slot moves.
    """
    root, fleet_dir, gh = _converged_setup(tmp_path, monkeypatch)

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(fleet_dir),
        dry_run=False,
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "prologue"
    assert stamp.updated_at is not None


def test_dry_run_with_planned_start_does_not_actuate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run threads ``dry_run=True`` into actuation and writes no state.

    With a parked runner and live demand, the plan wants a start. In preview the
    launch helper is called with ``dry_run=True`` (so it does not spawn) and the
    hysteresis file is not written. This is the composition-level check that the
    flag reaches both actuation and persistence, not just one of them.
    """
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", CW)
    _patch_liveness(monkeypatch, set())  # nothing running → plan will start

    gh = _FakeGitHub(
        runs={CW: [{"id": 1}]},
        jobs={1: [{"status": "in_progress", "labels": ["self-hosted"]}]},
    )

    launches: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        "charlie_work.runner_slots.launch_runner_listener",
        lambda path, dry_run=False: (
            launches.append((path, dry_run)),
            (True, "Would launch" if dry_run else "launched"),
        )[1],
    )

    fleet_dir = tmp_path / "fleet"
    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(fleet_dir),
        dry_run=True,
        source="cli",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    starts = [r for r in result.results if r.change.action is SlotAction.START]
    assert len(starts) == 1
    # The flag threaded through to the launch helper — no real spawn.
    assert launches == [(root / "cw-1", True)]
    assert not _allocation_state_path(fleet_dir).exists()


def test_real_run_with_planned_start_actuates_and_writes_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The positive half: a real run launches and persists provenance.

    Without this, the dry-run guard could silently disable actuation entirely
    and the suite would still be green.
    """
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", CW)
    _patch_liveness(monkeypatch, set())

    gh = _FakeGitHub(
        runs={CW: [{"id": 1}]},
        jobs={1: [{"status": "in_progress", "labels": ["self-hosted"]}]},
    )

    launches: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        "charlie_work.runner_slots.launch_runner_listener",
        lambda path, dry_run=False: (
            launches.append((path, dry_run)),
            (True, "launched"),
        )[1],
    )

    fleet_dir = tmp_path / "fleet"
    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(fleet_dir),
        dry_run=False,
        source="cli",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    assert result.started == 1
    assert launches == [(root / "cw-1", False)]
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "cli"


def _mature_slack_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, _FakeGitHub]:
    """One repo with two running listeners, zero demand, and a mature streak."""
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", CW)
    _make_runner_dir(root, "cw-2", CW)
    _patch_liveness(monkeypatch, {"cw-1", "cw-2"})

    fleet_dir = tmp_path / "fleet"
    from charlie_work.runner_slots import save_idle_streaks

    save_idle_streaks(
        fleet_dir, {CW: 3}, source="prologue", full_pass_interval_seconds=PASS_INTERVAL
    )
    assert load_idle_streaks(fleet_dir) == {CW: 3}

    gh = _FakeGitHub()
    return root, fleet_dir, gh


def test_dry_run_with_planned_park_does_not_actuate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preview must not call park_runner_slot with dry_run=False or write state."""
    root, fleet_dir, gh = _mature_slack_setup(tmp_path, monkeypatch)

    parks: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "charlie_work.runner_slots.park_runner_slot",
        lambda instance, *, dry_run=False: (
            parks.append((instance.name, dry_run)),
            (
                True,
                f"Would park {instance.name}" if dry_run else f"parked {instance.name}",
            ),
        )[1],
    )

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(
            enabled=True,
            managed_root=str(root),
            max_running_runners=1,
            min_running_per_repo=0,
            demand_idle_samples=3,
        ),
        fleet_dir_override=str(fleet_dir),
        dry_run=True,
        source="cli",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    assert result.parked == 2
    assert result.started == 0
    assert sorted(parks) == [("cw-1", True), ("cw-2", True)]
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "prologue"
    assert load_idle_streaks(fleet_dir) == {CW: 3}


def test_real_run_with_planned_park_actuates_and_writes_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real pass parks mature surplus slots, advancing both provenance and streak."""
    root, fleet_dir, gh = _mature_slack_setup(tmp_path, monkeypatch)

    parks: list[tuple[str, bool]] = []
    monkeypatch.setattr(
        "charlie_work.runner_slots.park_runner_slot",
        lambda instance, *, dry_run=False: (
            parks.append((instance.name, dry_run)),
            (
                True,
                f"Would park {instance.name}" if dry_run else f"parked {instance.name}",
            ),
        )[1],
    )

    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(
            enabled=True,
            managed_root=str(root),
            max_running_runners=1,
            min_running_per_repo=0,
            demand_idle_samples=3,
        ),
        fleet_dir_override=str(fleet_dir),
        dry_run=False,
        source="cli",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    assert result.parked == 2
    assert result.started == 0
    assert sorted(parks) == [("cw-1", False), ("cw-2", False)]
    stamp = load_allocation_stamp(fleet_dir)
    assert stamp is not None
    assert stamp.source == "cli"
    # The pre-actuation snapshot still saw two running slots and zero demand.
    assert load_idle_streaks(fleet_dir) == {CW: 4}


# --------------------------------------------------------------------------
# busy_error path — errors as values
# --------------------------------------------------------------------------


def test_unreadable_runner_list_pins_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo whose runner list cannot be read is pinned, not parked.

    Per the errors-as-values invariant, ``fetch_busy_runner_names`` returns the
    error as a value and the pass treats that repo as unmeasurable: the planner
    pins it at its current running count rather than risk parking a busy runner
    it cannot see. A regression that let a park slip through on an unreadable
    repo would abort a live CI job, and nothing else would catch it.
    """
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", CW)
    _make_runner_dir(root, "cw-2", CW)
    _make_runner_dir(root, "jc-1", JC)
    _make_runner_dir(root, "jc-2", JC)
    # CW has two running listeners; JC's are parked.
    _patch_liveness(monkeypatch, {"cw-1", "cw-2"})

    gh = _FakeGitHub(
        # CW's runners list is unreadable — the busy_error path.
        runners_error_repos={CW},
        # JC has queued self-hosted demand but no running listeners.
        runs={JC: [{"id": 10}]},
        jobs={10: [{"status": "queued", "labels": ["self-hosted"]}] * 4},
    )

    fleet_dir = tmp_path / "fleet"
    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(
            enabled=True,
            managed_root=str(root),
            max_running_runners=2,
            min_running_per_repo=1,
        ),
        fleet_dir_override=str(fleet_dir),
        dry_run=False,
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    plan = result.plan
    assert plan is not None

    # CW is pinned: no park changes target its runners.
    cw_parks = [c for c in plan.changes if c.action is SlotAction.PARK and c.repo == CW]
    assert cw_parks == [], "an unmeasurable repo must not have its runners parked"

    cw_target = next(t for t in plan.targets if t.repo == CW)
    assert cw_target.pinned is True
    assert cw_target.target == 2  # held at current running count
    assert cw_target.running == 2

    assert any("unmeasurable" in note for note in plan.notes)
    assert any(CW in note for note in plan.notes)


def test_unreadable_runner_list_holds_idle_streak_not_advances_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing data must not accrue toward parking a runner (hysteresis invariant).

    ``next_idle_streaks`` holds a repo's streak when its demand is
    unmeasurable. The pass feeds the error-demand through, so a transient API
    blip cannot mature the streak and trigger a park on the next real pass.
    """
    root = tmp_path / "actions-runners"
    root.mkdir()
    _make_runner_dir(root, "cw-1", CW)
    _make_runner_dir(root, "cw-2", CW)
    _patch_liveness(monkeypatch, {"cw-1", "cw-2"})

    fleet_dir = tmp_path / "fleet"
    # Seed a prior streak so we can tell hold-from-advance apart.
    from charlie_work.runner_slots import save_idle_streaks

    save_idle_streaks(
        fleet_dir, {CW: 2}, source="prologue", full_pass_interval_seconds=PASS_INTERVAL
    )
    assert load_idle_streaks(fleet_dir) == {CW: 2}

    gh = _FakeGitHub(runners_error_repos={CW})
    result = run_allocation_pass(
        gh,
        RunnerAllocationConfig(enabled=True, managed_root=str(root), max_running_runners=4),
        fleet_dir_override=str(fleet_dir),
        dry_run=False,
        source="prologue",
        full_pass_interval_seconds=PASS_INTERVAL,
    )

    assert result.ok is True
    # Streak held at 2, not advanced to 3 — missing data is not slack.
    assert load_idle_streaks(fleet_dir) == {CW: 2}
