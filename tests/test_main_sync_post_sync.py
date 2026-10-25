"""post_sync contract: partial fast-forward, defer + back-off, failure, rollback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _main_sync_fixtures import SyncRepos, make_sync_repos

from charlie_work.instrumentation import query_events
from charlie_work.layout import state_file_path
from charlie_work.main_sync import (
    BACKOFF_BASE_SECONDS,
    DEFER_ALARM_SECONDS,
    MARKER_FILENAME,
    SyncContext,
    sync_repo,
)
from charlie_work.main_sync_config import MainSyncConfig
from charlie_work.main_sync_state import load_sync_state
from charlie_work.write_gate import WriteGate

NOW = 1_800_000_000.0
REPO = "owner/repo"
SCRIPT = """\
import pathlib, sys
log, code = pathlib.Path(sys.argv[1]), int(sys.argv[2])
with log.open("a", encoding="utf-8") as handle:
    handle.write("run\\n")
if len(sys.argv) > 3:
    pathlib.Path(sys.argv[3]).write_text("touched by post_sync\\n", encoding="utf-8")
sys.exit(code)
"""


def _command(tmp_path: Path, code: int, *, touch: Path | None = None) -> tuple[str, ...]:
    script = tmp_path / "post_sync.py"
    script.write_text(SCRIPT, encoding="utf-8")
    argv = [sys.executable, str(script), str(tmp_path / "runs.log"), str(code)]
    return tuple(argv + ([str(touch)] if touch else []))


def _runs(tmp_path: Path) -> int:
    log = tmp_path / "runs.log"
    return len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0


def _ctx(
    tmp_path: Path, repos: SyncRepos, command: tuple[str, ...], *, now: float = NOW
) -> SyncContext:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return SyncContext(
        repo=REPO,
        repo_root=repos.clone,
        config=MainSyncConfig(post_sync_command=command, post_sync_when_paths=("uv.lock",)),
        fleet_dir=tmp_path / "fleet",
        now=now,
        write_gate=WriteGate(dry_run=False, state_path=state_file_path(state_dir), repo=REPO),
    )


def _events(tmp_path: Path, kind: str) -> list[dict]:
    return query_events(state_file_path(tmp_path / "state"), kind=kind)


def _marker(repos: SyncRepos) -> dict:
    path = repos.clone / ".git" / MARKER_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def test_untouched_range_lands_without_running_post_sync(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    tip = repos.push_commit("src/a.py", "x = 1\n")
    result = sync_repo(_ctx(tmp_path, repos, _command(tmp_path, 1)))
    assert (result.outcome, repos.head()) == ("advanced", tip)
    assert _runs(tmp_path) == 0


def test_exit_zero_keeps_the_advance(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("uv.lock", "v2\n")
    result = sync_repo(_ctx(tmp_path, repos, _command(tmp_path, 0)))
    assert (result.outcome, repos.head(), _runs(tmp_path)) == ("advanced", tip, 1)
    assert (_marker(repos)["from"], _marker(repos)["to"]) == (seed, tip)


def test_partial_fast_forward_then_defer_rolls_back_remainder_only(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    safe = repos.push_commit("src/a.py", "x = 1\n")
    repos.push_commit("uv.lock", "v2\n")
    tip = repos.push_commit("src/b.py", "y = 2\n")

    result = sync_repo(_ctx(tmp_path, repos, _command(tmp_path, 75)))

    assert (result.outcome, result.from_sha, result.to_sha) == ("deferred", safe, tip)
    assert repos.head() == safe  # the commit before the first uv.lock change stands
    assert (_marker(repos)["from"], _marker(repos)["to"]) == (seed, safe)
    state = load_sync_state(tmp_path / "fleet", REPO)
    assert (state.defer_count, state.failure_streak) == (1, 0)  # a defer is not a failure
    assert state.next_attempt_at == NOW + BACKOFF_BASE_SECONDS
    assert len(_events(tmp_path, "main_sync_deferred")) == 1
    assert len(_events(tmp_path, "main_sync_rolled_back")) == 1


def test_back_off_window_waits_without_running_post_sync(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    repos.push_commit("uv.lock", "v2\n")
    command = _command(tmp_path, 75)
    sync_repo(_ctx(tmp_path, repos, command))
    waiting = sync_repo(_ctx(tmp_path, repos, command, now=NOW + 10))
    assert (waiting.outcome, waiting.reason) == ("waiting", "backoff")
    assert _runs(tmp_path) == 1
    again = sync_repo(_ctx(tmp_path, repos, command, now=NOW + BACKOFF_BASE_SECONDS + 1))
    assert again.outcome == "deferred" and _runs(tmp_path) == 2
    state = load_sync_state(tmp_path / "fleet", REPO)
    assert state.next_attempt_at == NOW + BACKOFF_BASE_SECONDS + 1 + 2 * BACKOFF_BASE_SECONDS


def test_four_hours_of_deferral_alarms_once(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    repos.push_commit("uv.lock", "v2\n")
    command = _command(tmp_path, 75)
    sync_repo(_ctx(tmp_path, repos, command))
    late = sync_repo(_ctx(tmp_path, repos, command, now=NOW + DEFER_ALARM_SECONDS + 1))
    later = sync_repo(_ctx(tmp_path, repos, command, now=NOW + 2 * DEFER_ALARM_SECONDS))
    assert late.alarm == "deferred_4h"
    assert later.alarm is None
    assert [e["payload"]["reason"] for e in _events(tmp_path, "main_sync_alarm")] == [
        "deferred_4h"
    ]


def test_failure_rolls_back_counts_and_alarms_at_three_then_recovers(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("uv.lock", "v2\n")
    failing = _command(tmp_path, 1)

    results = [
        sync_repo(_ctx(tmp_path, repos, failing, now=NOW + offset)) for offset in (0, 301, 1000)
    ]

    assert [r.outcome for r in results] == ["failed", "failed", "failed"]
    assert [r.alarm for r in results] == [None, None, "post_sync_failing"]
    assert repos.head() == seed
    assert _marker(repos) == {}  # a rolled-back advance never writes one
    assert len(_events(tmp_path, "main_sync_failed")) == 3

    recovered = sync_repo(_ctx(tmp_path, repos, _command(tmp_path, 0), now=NOW + 10_000))
    assert (recovered.outcome, recovered.recovered, repos.head()) == ("advanced", True, tip)
    assert load_sync_state(tmp_path / "fleet", REPO).failure_streak == 0


def test_rollback_refused_alarms_immediately_and_keeps_the_marker(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("uv.lock", "v2\n")
    command = _command(tmp_path, 1, touch=repos.clone / "uv.lock")

    result = sync_repo(_ctx(tmp_path, repos, command))

    assert (result.outcome, result.reason, result.alarm) == (
        "failed",
        "rollback_refused",
        "rollback_refused",
    )
    assert repos.head() == tip
    assert (_marker(repos)["from"], _marker(repos)["to"]) == (seed, tip)
    assert len(_events(tmp_path, "main_sync_rollback_refused")) == 1
    assert load_sync_state(tmp_path / "fleet", REPO).rollback_alarmed
