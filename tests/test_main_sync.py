"""sync_repo against real repositories: advance, ref update, skips, events, marker."""

from __future__ import annotations

import json
from pathlib import Path

from _main_sync_fixtures import SyncRepos, git, make_sync_repos

from charlie_work.instrumentation import query_events
from charlie_work.layout import state_file_path
from charlie_work.main_sync import MARKER_FILENAME, SyncContext, sync_repo
from charlie_work.main_sync_config import MainSyncConfig
from charlie_work.main_sync_state import state_path
from charlie_work.write_gate import WriteGate

NOW = 1_800_000_000.0  # 2027-01-15 UTC
REPO = "owner/repo"


def _ctx(
    tmp_path: Path, root: Path, *, dry_run: bool = False, now: float = NOW, **config: object
) -> SyncContext:
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    return SyncContext(
        repo=REPO,
        repo_root=root,
        config=MainSyncConfig(**config),  # type: ignore[arg-type]
        fleet_dir=tmp_path / "fleet",
        now=now,
        write_gate=WriteGate(dry_run=dry_run, state_path=state_file_path(state_dir), repo=REPO),
    )


def _events(tmp_path: Path, kind: str) -> list[dict]:
    return query_events(state_file_path(tmp_path / "state"), kind=kind)


def _marker(repos: SyncRepos) -> dict | None:
    path = repos.clone / ".git" / MARKER_FILENAME
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_fast_forwards_default_branch_and_writes_marker(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    repos.push_commit("CLAUDE.md", "new rules\n")
    tip = repos.push_commit("src/a.py", "x = 1\n")

    result = sync_repo(_ctx(tmp_path, repos.clone))

    assert (result.outcome, result.from_sha, result.to_sha) == ("advanced", seed, tip)
    assert repos.head() == tip
    [event] = _events(tmp_path, "main_sync_advanced")
    assert event["payload"]["from_sha"] == seed and event["payload"]["commits"] == 2
    assert event["payload"]["ref_only"] is False
    marker = _marker(repos)
    assert marker is not None and marker["schema"] == 1
    assert (marker["from"], marker["to"], marker["commits"]) == (seed, tip, 2)
    assert marker["instruction_paths"] == ["CLAUDE.md"]
    assert marker["ts"].startswith("2027-01-15T")


def test_up_to_date_is_silent(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    result = sync_repo(_ctx(tmp_path, repos.clone))
    assert result.outcome == "up_to_date"
    assert _events(tmp_path, "main_sync_advanced") == []
    assert _events(tmp_path, "main_sync_skipped") == []
    assert _marker(repos) is None


def test_non_overlapping_dirty_files_still_fast_forward(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    tip = repos.push_commit("src/a.py", "x = 1\n")
    (repos.clone / "README.md").write_text("local\n", encoding="utf-8")
    assert sync_repo(_ctx(tmp_path, repos.clone)).outcome == "advanced"
    assert repos.head() == tip
    assert (repos.clone / "README.md").read_text(encoding="utf-8") == "local\n"


def test_overlapping_dirty_file_blocks_and_dedupes(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    repos.push_commit("README.md", "upstream\n")
    (repos.clone / "README.md").write_text("local\n", encoding="utf-8")

    first = sync_repo(_ctx(tmp_path, repos.clone))
    second = sync_repo(_ctx(tmp_path, repos.clone, now=NOW + 600))

    assert (first.outcome, first.reason, first.paths) == ("skipped", "blocked", ("README.md",))
    assert second.reason == "blocked"
    assert repos.head() == seed
    [event] = _events(tmp_path, "main_sync_skipped")  # once per sha pair, not per pass
    assert event["payload"]["paths"] == ["README.md"]


def test_untracked_blocker(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    repos.push_commit("new.txt", "upstream\n")
    (repos.clone / "new.txt").write_text("mine\n", encoding="utf-8")
    result = sync_repo(_ctx(tmp_path, repos.clone))
    assert (result.reason, result.paths) == ("blocked", ("new.txt",))


def test_diverged_skips(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    repos.push_commit("a.txt", "upstream\n")
    (repos.clone / "b.txt").write_text("local\n", encoding="utf-8")
    git(repos.clone, "add", "b.txt")
    git(repos.clone, "commit", "-q", "-m", "local work")
    assert sync_repo(_ctx(tmp_path, repos.clone)).reason == "diverged"


def test_unrelated_history_alarms_once(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    git(repos.upstream, "checkout", "-q", "--orphan", "reroot")
    (repos.upstream / "README.md").write_text("rewritten\n", encoding="utf-8")
    git(repos.upstream, "add", "README.md")
    git(repos.upstream, "commit", "-q", "-m", "re-rooted")
    git(repos.upstream, "push", "-q", "-f", "origin", "reroot:main")

    first = sync_repo(_ctx(tmp_path, repos.clone))
    second = sync_repo(_ctx(tmp_path, repos.clone, now=NOW + 600))

    assert (first.reason, first.alarm) == ("unrelated_history", "unrelated_history")
    assert (second.reason, second.alarm) == ("unrelated_history", None)
    [alarm] = _events(tmp_path, "main_sync_alarm")
    assert alarm["payload"]["reason"] == "unrelated_history"


def test_ref_only_update_on_feature_branch(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    git(repos.clone, "checkout", "-q", "-b", "feature")
    tip = repos.push_commit("a.txt", "a\n")

    result = sync_repo(_ctx(tmp_path, repos.clone))

    assert (result.outcome, result.ref_only) == ("advanced", True)
    assert git(repos.clone, "rev-parse", "refs/heads/main") == tip
    assert repos.head() == seed  # working tree untouched
    assert _marker(repos) is None  # files on disk did not change
    [event] = _events(tmp_path, "main_sync_advanced")
    assert event["payload"]["ref_only"] is True


def test_main_checked_out_in_linked_worktree_skips(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    git(repos.clone, "checkout", "-q", "-b", "feature")
    git(repos.clone, "worktree", "add", "-q", str(tmp_path / "wt"), "main")
    repos.push_commit("a.txt", "a\n")
    assert sync_repo(_ctx(tmp_path, repos.clone)).reason == "checked_out_elsewhere"


def test_linked_worktree_and_kill_switch_and_no_origin_are_silent(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    linked = tmp_path / "linked"
    git(repos.clone, "worktree", "add", "-q", "-b", "side", str(linked))
    repos.push_commit("a.txt", "a\n")

    assert sync_repo(_ctx(tmp_path, linked)).reason == "not_primary"
    assert sync_repo(_ctx(tmp_path, repos.clone, enabled=False)).reason == "disabled"
    git(repos.clone, "remote", "remove", "origin")
    assert sync_repo(_ctx(tmp_path, repos.clone)).reason == "no_origin"
    assert _events(tmp_path, "main_sync_skipped") == []


def test_dry_run_touches_nothing(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    seed = repos.head()
    tip = repos.push_commit("a.txt", "a\n")
    repos.fetch()  # dry run never fetches; classify sees the tracking ref as it is

    result = sync_repo(_ctx(tmp_path, repos.clone, dry_run=True))

    assert (result.outcome, result.reason, result.to_sha) == ("previewed", "ff", tip)
    assert repos.head() == seed
    assert _marker(repos) is None
    assert _events(tmp_path, "main_sync_advanced") == []
    assert not state_path(tmp_path / "fleet", REPO).exists()


def test_dry_run_does_not_fetch(tmp_path: Path) -> None:
    repos = make_sync_repos(tmp_path)
    repos.push_commit("a.txt", "a\n")
    assert sync_repo(_ctx(tmp_path, repos.clone, dry_run=True)).outcome == "up_to_date"
