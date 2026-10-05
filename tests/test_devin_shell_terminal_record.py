"""Issue #2052: devin-shell workers leave a durable terminal record.

Since #2049 every Worker's fate resolves through ``worker_fate.resolve_fate``
on per-Adapter evidence, and ``claude_code.launch_claude_worker`` already ran
``start_terminal_status_watcher`` (issue #773) so a dead claude worker leaves
``issue-<n>.claude.terminal.json``. Devin-shell's launcher ran no watcher, so
``resolve_fate`` had only liveness + the outcome file to work with. These
tests pin the follow-up:

- ``AdapterFateProfile.writes_terminal_record`` is declared True for every
  Popen-backed harness (devin-shell is the flip; the declaration, not an
  ``adapter_kind ==`` branch, is where the difference lives);
- ``launch_devin_session`` starts the same non-blocking watcher after a
  successful ``Popen``, so a process that exits leaves
  ``issue-<n>.devin.terminal.json`` -- atomically, newer than
  ``dispatched_at``, carrying the exit code and a copy of the worktree's
  ``.worker-outcome.json``;
- ``resolve_no_pr_orphan_fate`` consumes that record for a crashed exit and
  a throttled exit (the record is found adapter-agnostically by
  ``find_worker_terminal_status``'s ``issue-<n>.*.terminal.json`` glob).
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _devin_shell_fixtures import (
    _install_fake_create_worktree,
    _write_fake_devin,
)

from charlie_work import devin_shell, worker_fate
from charlie_work.devin_shell import launch_devin_session
from charlie_work.no_pr_orphan_fate import resolve_no_pr_orphan_fate
from charlie_work.process_utils import (
    find_worker_terminal_status,
    worker_terminal_status_path,
)
from charlie_work.worktree import WorktreeInfo

_CRASH_SCRIPT = "import sys\nsys.exit(3)\n"

_THROTTLE_DEATH_SCRIPT = (
    "import sys\n"
    'sys.stdout.write("API rate limit reached. Please try again later.\\n")\n'
    "sys.stdout.flush()\n"
    "sys.exit(1)\n"
)

# The worker writes its .worker-outcome.json (cwd == worktree) then exits 0:
# the watcher must copy it into the terminal record (issue #935 semantics).
_OUTCOME_SCRIPT = (
    "import json, pathlib\n"
    'pathlib.Path(".worker-outcome.json").write_text(\n'
    '    json.dumps({"push_succeeded": True, "pr_created": False, "head_sha": "abc123"}),\n'
    '    encoding="utf-8",\n'
    ")\n"
)


def _wait_for(predicate: Callable[[], bool], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _launch_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    issue_number: int,
    script_body: str,
) -> tuple[devin_shell.SessionRecord, Path, datetime]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    prompt_path = tmp_path / f"prompt-{issue_number}.md"
    prompt_path.write_text("do the thing", encoding="utf-8")
    _install_fake_create_worktree(monkeypatch, tmp_path)
    script = _write_fake_devin(tmp_path, script_body)
    dispatched_at = datetime.now(UTC)
    record = launch_devin_session(
        issue_number,
        f"agent/issue-{issue_number}",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        command_template=(sys.executable, str(script)),
    )
    return record, sessions_dir, dispatched_at


def test_devin_profile_declares_writes_terminal_record() -> None:
    """The profile, not an ``adapter_kind ==`` branch, carries the declaration.

    Mutating the flag off (see the skip test below) or forgetting to declare
    it on a Popen-backed harness is what this pins.
    """
    devin_profile = worker_fate.profile_for("devin")
    assert devin_profile is not None
    assert devin_profile.writes_terminal_record is True
    # Parity: every Popen-backed harness writes one; command/manual spawn
    # no worker process, so there is nothing to watch.
    assert worker_fate.profile_for("claude-code").writes_terminal_record is True
    assert worker_fate.profile_for("api").writes_terminal_record is True
    assert worker_fate.profile_for("command").writes_terminal_record is False
    assert worker_fate.profile_for("manual").writes_terminal_record is False


def test_devin_launch_leaves_terminal_record_newer_than_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: an exited devin worker leaves a fresh terminal record.

    The watcher writes ``issue-<n>.devin.terminal.json`` after ``Popen`` --
    launch must return without waiting for it (the file only appears once
    the spawned process actually exits).
    """
    record, sessions_dir, dispatched_at = _launch_worker(
        tmp_path, monkeypatch, 2052, _CRASH_SCRIPT
    )
    assert record.error is None
    assert record.pid is not None

    terminal_path = sessions_dir / "issue-2052.devin.terminal.json"
    assert terminal_path == worker_terminal_status_path(sessions_dir, 2052, "devin")
    assert _wait_for(terminal_path.is_file), "watcher never wrote the terminal record"

    terminal = find_worker_terminal_status(sessions_dir, 2052)
    assert terminal is not None
    assert terminal["pid"] == record.pid
    assert terminal["exit_code"] == 3
    ended_at = worker_fate.parse_iso_timestamp(terminal["ended_at"])
    assert ended_at is not None
    assert ended_at > dispatched_at
    assert terminal["duration_seconds"] >= 0


def test_devin_launch_terminal_record_copies_worker_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worktree's ``.worker-outcome.json`` is embedded at exit (#935).

    The outcome survives worktree teardown inside the durable record;
    ``worker_outcome_written_at`` carries the file's own mtime (B5).
    """
    record, sessions_dir, _dispatched_at = _launch_worker(
        tmp_path, monkeypatch, 2053, _OUTCOME_SCRIPT
    )
    assert record.error is None

    terminal_path = sessions_dir / "issue-2053.devin.terminal.json"
    assert _wait_for(terminal_path.is_file)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["exit_code"] == 0
    assert terminal["worker_outcome"]["push_succeeded"] is True
    assert terminal["worker_outcome"]["head_sha"] == "abc123"
    assert terminal["worker_outcome_written_at"]


def test_devin_review_launch_writes_terminal_record_without_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review launches write the record too -- exit code only (#1354 parity).

    A review checkout holds no ``.worker-outcome.json``, so the watcher is
    started with ``worktree_path=None``; ``find_worker_terminal_status``
    under the caller's ``reviews_dir`` still gives the review-verdict reaper
    a durable exit code keyed by PR number.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    prompt_path = tmp_path / "review-prompt.md"
    prompt_path.write_text("review the diff\n", encoding="utf-8")

    def fake_create_review_checkout(
        repo_root_arg: Path, pr_number_arg: int, head_sha_arg: str, *, reviews_dir: Path
    ) -> WorktreeInfo:
        checkout_path = reviews_dir / f"pr-{pr_number_arg}"
        checkout_path.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=checkout_path, branch=head_sha_arg, venv_junction=None)

    def fail_remove_review_checkout(*args: object, **kwargs: object) -> None:
        raise AssertionError("no review-checkout teardown on a successful launch")

    monkeypatch.setattr(devin_shell, "create_review_checkout", fake_create_review_checkout)
    monkeypatch.setattr(devin_shell, "remove_review_checkout", fail_remove_review_checkout)

    script = _write_fake_devin(tmp_path, "import sys\nsys.exit(1)\n")
    record = launch_devin_session(
        77,
        "agent/issue-7-fix",
        prompt_path,
        repo_root=repo_root,
        sessions_dir=reviews_dir,
        review=True,
        head_sha="a" * 40,
        command_template=(sys.executable, str(script)),
    )
    assert record.error is None

    terminal_path = reviews_dir / "issue-77.devin.terminal.json"
    assert _wait_for(terminal_path.is_file)
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["exit_code"] == 1
    # worktree_path=None: no outcome copy keys are written at all.
    assert "worker_outcome" not in terminal
    assert "worker_outcome_written_at" not in terminal


def test_devin_launch_skips_terminal_record_when_profile_undeclares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutation gate for the profile seam: flipping the declaration off must
    stop the record -- the watcher start consults the profile rather than
    being unconditional (and never branches on ``adapter_kind ==``)."""
    original = worker_fate.profile_for
    devin_profile = original("devin")
    assert devin_profile is not None
    flipped = replace(devin_profile, writes_terminal_record=False)
    monkeypatch.setattr(
        worker_fate,
        "profile_for",
        lambda kind: flipped if kind == "devin" else original(kind),
    )

    # A started watcher would poll every 0.05 s instead of 2 s, so a short
    # wait after the process dies is ten full poll intervals.
    monkeypatch.setattr("charlie_work.process_utils._TERMINAL_STATUS_POLL_INTERVAL_SECONDS", 0.05)
    record, sessions_dir, _ = _launch_worker(tmp_path, monkeypatch, 2054, _CRASH_SCRIPT)
    assert record.error is None
    assert record.pid is not None

    # Wait for the spawned process to die, then allow ten watcher poll
    # intervals -- had a watcher been started, the file would exist by then.
    assert _wait_for(lambda: not worker_fate.is_alive(record.pid, record.process_start_time)), (
        "fake devin process never exited"
    )
    time.sleep(0.5)
    assert not (sessions_dir / "issue-2054.devin.terminal.json").exists()
    assert find_worker_terminal_status(sessions_dir, 2054) is None


def test_resolve_fate_reads_devin_terminal_record_for_crashed_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a devin worker's fresh terminal record resolves Crashed.

    ``resolve_no_pr_orphan_fate`` is the sweep's real consumer: it finds the
    record via the adapter-agnostic glob and feeds its exit code through the
    freshness gate. ``basis.exit_code == 3`` can only come from the record,
    so it both was consumed and passed the ``dispatched_at`` freshness check.
    """
    record, sessions_dir, dispatched_at = _launch_worker(
        tmp_path, monkeypatch, 2055, _CRASH_SCRIPT
    )
    assert record.error is None
    assert _wait_for((sessions_dir / "issue-2055.devin.terminal.json").is_file)

    fate = resolve_no_pr_orphan_fate(
        issue_number=2055,
        entry={"adapter": "devin", "dispatched_at": _iso(dispatched_at)},
        terminal=find_worker_terminal_status(sessions_dir, 2055),
        worktree_path=None,
        worktree_outcome_raw=None,
        now=datetime.now(UTC),
    )

    assert isinstance(fate, worker_fate.Crashed)
    assert fate.basis.exit_code == 3
    assert fate.basis.stale == ()


def test_resolve_fate_reads_devin_terminal_record_for_throttled_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A throttled death resolves Throttled, still carrying the record's code.

    The persisted ``dead_worker_failure_kind`` stamp (written by the
    dead-session classifier from the log tail) supplies rule 6's failure; the
    fresh terminal record supplies the exit code.
    """
    record, sessions_dir, dispatched_at = _launch_worker(
        tmp_path, monkeypatch, 2056, _THROTTLE_DEATH_SCRIPT
    )
    assert record.error is None
    assert _wait_for((sessions_dir / "issue-2056.devin.terminal.json").is_file)

    fate = resolve_no_pr_orphan_fate(
        issue_number=2056,
        entry={
            "adapter": "devin",
            "dispatched_at": _iso(dispatched_at),
            "dead_worker_failure_kind": "rate_limited",
        },
        terminal=find_worker_terminal_status(sessions_dir, 2056),
        worktree_path=None,
        worktree_outcome_raw=None,
        now=datetime.now(UTC),
    )

    assert isinstance(fate, worker_fate.Throttled)
    assert fate.failure.kind == "rate_limited"
    assert fate.basis.exit_code == 1
    assert fate.basis.stale == ()


def test_devin_terminal_record_suppresses_log_tail_throttle_on_clean_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#656/#2022 parity: a completed devin worker's log tail is prose, not a
    provider error. The record (this pid, exit 0, embedded outcome) must veto
    log-tail classification -- the throttle marker quoted in the completion
    summary cannot arm a fleet cooldown. The same marker on a crashed session
    (exit != 0) still classifies, pinning the exit-code leg of the proof."""
    completing_script = (
        "import json, pathlib, sys\n"
        'sys.stdout.write("Session summary: fixed the rate limit regression.\\n")\n'
        "sys.stdout.flush()\n"
        'pathlib.Path(".worker-outcome.json").write_text(\n'
        '    json.dumps({"push_succeeded": True, "head_sha": "abc123"}), encoding="utf-8"\n'
        ")\n"
        "sys.exit(0)\n"
    )
    record, sessions_dir, _ = _launch_worker(tmp_path, monkeypatch, 2058, completing_script)
    assert record.error is None
    terminal_path = sessions_dir / "issue-2058.devin.terminal.json"
    assert _wait_for(terminal_path.is_file)

    kind, throttled_until = devin_shell.update_session_record_with_failure_classification(
        sessions_dir, 2058, config=None
    )
    # The marker text in the log tail would otherwise classify rate_limited;
    # the terminal record proves completion, so nothing is classified at all.
    assert kind is None
    assert throttled_until is None
    sidecar = json.loads((sessions_dir / "issue-2058.json").read_text(encoding="utf-8"))
    assert sidecar.get("failure_kind") is None

    # Same log tail, non-zero exit: the record no longer proves completion,
    # so the throttle signature is still honored.
    crashed_record, sessions_dir2, _ = _launch_worker(
        tmp_path / "second", monkeypatch, 2059, _THROTTLE_DEATH_SCRIPT
    )
    assert crashed_record.error is None
    assert _wait_for((sessions_dir2 / "issue-2059.devin.terminal.json").is_file)

    kind2, throttled_until2 = devin_shell.update_session_record_with_failure_classification(
        sessions_dir2, 2059, config=None
    )
    assert kind2 == "rate_limited"
    assert throttled_until2 is not None


def test_stale_devin_terminal_record_is_ignored_by_resolve_fate(tmp_path: Path) -> None:
    """Rule 1 still applies: a record whose ``ended_at`` predates the current
    ``dispatched_at`` is the previous attempt's and counts as absent -- no
    exit code, no outcome. This is the watcher-never-ran redispatch shape
    (e.g. orchestrator restart mid-session), pinned without timing
    dependence."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    dispatched_at = datetime.now(UTC)
    (sessions_dir / "issue-2057.devin.terminal.json").write_text(
        json.dumps(
            {
                "pid": 424242,
                "exit_code": 0,
                "started_at": _iso(dispatched_at - timedelta(minutes=10)),
                "ended_at": _iso(dispatched_at - timedelta(minutes=5)),
                "duration_seconds": 300.0,
            }
        ),
        encoding="utf-8",
    )

    fate = resolve_no_pr_orphan_fate(
        issue_number=2057,
        entry={"adapter": "devin", "dispatched_at": _iso(dispatched_at)},
        terminal=find_worker_terminal_status(sessions_dir, 2057),
        worktree_path=None,
        worktree_outcome_raw=None,
        now=datetime.now(UTC),
    )

    assert isinstance(fate, worker_fate.Crashed)
    # exit_code is None -- the stale record's was never consumed.
    assert fate.basis.exit_code is None
