"""Issue #2847: a transient salvage-push failure must not strand a dead worker's commit.

The orphan sweep tries ``salvage_push_stranded_commits`` once. When that push
times out, the commit stays in the worktree and the next redispatch used to
read it as divergent local work (``worktree_unsafe_local_commits``, a judgment
escalation). ``create_worktree`` now re-pushes it first, bounded by
``dispatch.salvage_push_retries``, and escalates exactly as before when the
budget is spent, the push is rejected, or the remote moved.

The #807/#257 refusal tests in ``test_worktree.py`` call ``create_worktree``
without a ``config``; they are the unchanged control for the no-retry path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from _worktree_fixtures import _clone_repo, _git, _init_bare_remote_and_clone

from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.paths import runtime_paths
from charlie_work.salvage_retry import (
    RETRY_COUNT_FIELD,
    SALVAGE_PUSH_FAILED,
    SALVAGE_PUSHED,
    is_rejection,
    latest_retryable_failure,
)
from charlie_work.state import append_event, load_state, save_state
from charlie_work.worktree import (
    RescueCapture,
    SalvagePushResult,
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    create_worktree,
)

ISSUE = 2847
BRANCH = "agent/issue-2847-stranded-salvage"
TIMEOUT_ERROR = "push_failed: command timed out after 60s"
REJECTED_ERROR = "push_failed: ! [rejected] BRANCH -> BRANCH (non-fast-forward)"


def _config(tmp_path: Path, **dispatch: Any) -> OrchestratorConfig:
    # Isolate post_mortem from the host's real Devin sessions.db (see
    # ``_unescalate_fixtures._app``): recovery's liveness probe reads it.
    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    if dispatch:
        config = dataclasses.replace(
            config, dispatch=dataclasses.replace(config.dispatch, **dispatch)
        )
    return config


def _state_file(repo: Path, config: OrchestratorConfig) -> Path:
    return runtime_paths(repo, config.runtime.state_dir).state_file


def _stranded_bed(tmp_path: Path) -> tuple[Path, Path, WorktreeInfo]:
    """Bare origin + clone + a worktree holding one committed, never-pushed commit."""
    remote, repo = _init_bare_remote_and_clone(tmp_path)
    info = create_worktree(repo, BRANCH, base_ref="origin/main", issue_number=ISSUE)
    (info.path / "work.txt").write_text("stranded work\n", encoding="utf-8")
    _git(info.path, "add", "work.txt")
    _git(info.path, "commit", "-m", "worker commit before death")
    return remote, repo, info


def _record_salvage_failure(state_file: Path, error: str) -> None:
    """What the orphan sweep leaves behind when its one salvage push fails."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state = append_event(
        load_state(state_file),
        SALVAGE_PUSH_FAILED,
        {
            "issue_number": ISSUE,
            "pr_number": None,
            "branch": BRANCH,
            "old_remote_sha": None,
            "commit_count": 1,
            "error": error,
        },
    )
    state["issues"][str(ISSUE)] = {
        "number": ISSUE,
        "status": "dispatched",
        "branch_name": BRANCH,
    }
    save_state(state_file, state)


def _redispatch(repo: Path, config: OrchestratorConfig, *, recovery: bool = True) -> WorktreeInfo:
    return create_worktree(
        repo,
        BRANCH,
        base_ref="origin/main",
        issue_number=ISSUE,
        config=config,
        recovery={"branch_name": BRANCH, "status": "dispatched"} if recovery else None,
    )


def _remote_sha(remote: Path) -> str | None:
    out = _git(remote, "ls-remote", str(remote), f"refs/heads/{BRANCH}").stdout.strip()
    return out.split()[0] if out else None


def _events(state_file: Path, kind: str) -> list[dict[str, Any]]:
    return [e["payload"] for e in load_state(state_file)["events"] if e["kind"] == kind]


def _used_attempts(state_file: Path) -> int | None:
    return load_state(state_file)["issues"][str(ISSUE)].get(RETRY_COUNT_FIELD)


class _FakePush:
    """Stands in for ``salvage_push_stranded_commits``; counts how often it is tried."""

    def __init__(self, result: SalvagePushResult) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> SalvagePushResult:
        self.calls += 1
        return self.result


def _patch_push(monkeypatch: pytest.MonkeyPatch, error: str) -> _FakePush:
    fake = _FakePush(SalvagePushResult(pushed=False, error=error, commit_count=1))
    monkeypatch.setattr("charlie_work.worktree.salvage_push_stranded_commits", fake)
    return fake


@pytest.fixture
def refusal_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable rescue capture so a refused reset surfaces as ``WorktreeUnsafeError``."""
    monkeypatch.setattr(
        "charlie_work.worktree._capture_worktree_work_to_rescue_ref",
        lambda *args, **kwargs: RescueCapture(
            ref_name=None, commit_sha=None, error="forced capture failure for refusal test"
        ),
    )


@pytest.mark.parametrize("recovery", [True, False], ids=["recovery", "fresh-dispatch"])
def test_transient_failure_is_retried_and_redispatch_proceeds_on_pushed_branch(
    tmp_path: Path, refusal_path: None, recovery: bool
) -> None:
    remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    assert _remote_sha(remote) is None

    redispatched = _redispatch(repo, config, recovery=recovery)

    # The commit reached origin and the worker resumes on it: same worktree,
    # same tip, same file -- not a fresh branch forked from base.
    assert _remote_sha(remote) == local_tip
    assert redispatched.path == info.path
    assert _git(redispatched.path, "rev-parse", "HEAD").stdout.strip() == local_tip
    assert (redispatched.path / "work.txt").read_text(encoding="utf-8") == "stranded work\n"
    pushed = _events(state_file, SALVAGE_PUSHED)
    assert [(p["issue_number"], p["branch"], p["retry_attempt"]) for p in pushed] == [
        (ISSUE, BRANCH, 1)
    ]
    # A landed push ends the episode, so a later stranding gets a fresh budget.
    assert _used_attempts(state_file) is None


def test_persistent_failure_escalates_as_before_once_retries_are_used_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    _remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    push = _patch_push(monkeypatch, TIMEOUT_ERROR)
    limit = config.dispatch.salvage_push_retries
    assert limit == 3

    # Attempts 1..limit-1 fail transiently: the launch is deferred through the
    # ordinary redispatch path (worktree_probe_failed), NOT escalated.
    for attempt in range(1, limit):
        with pytest.raises(WorktreeProbeFailedError, match=rf"retry {attempt}/{limit}"):
            _redispatch(repo, config)
        assert _used_attempts(state_file) == attempt
    # The last attempt fails too: retries are used up, so the guard refuses.
    with pytest.raises(WorktreeUnsafeError) as final:
        _redispatch(repo, config)
    assert final.value.kind == "worktree_unsafe_local_commits"
    assert _used_attempts(state_file) == limit
    assert push.calls == limit
    failures = _events(state_file, SALVAGE_PUSH_FAILED)
    assert [f.get("retry_attempt") for f in failures] == [None, 1, 2, 3]

    # Budget spent: no further push, the same escalation, nothing is lost.
    with pytest.raises(WorktreeUnsafeError):
        _redispatch(repo, config)
    assert push.calls == limit
    assert _used_attempts(state_file) == limit
    assert (info.path / "work.txt").exists()


def test_rejection_on_record_escalates_on_first_attempt_without_pushing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, REJECTED_ERROR)
    push = _patch_push(monkeypatch, TIMEOUT_ERROR)

    with pytest.raises(WorktreeUnsafeError) as raised:
        _redispatch(repo, config)

    assert raised.value.kind == "worktree_unsafe_local_commits"
    assert push.calls == 0
    assert _used_attempts(state_file) is None


def test_rejected_retry_escalates_immediately_and_is_never_retried_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    push = _patch_push(monkeypatch, REJECTED_ERROR)

    with pytest.raises(WorktreeUnsafeError):
        _redispatch(repo, config)
    assert push.calls == 1
    assert _used_attempts(state_file) == 1

    # The rejection is now the newest outcome on record: no second push.
    with pytest.raises(WorktreeUnsafeError):
        _redispatch(repo, config)
    assert push.calls == 1


def test_remote_that_moved_since_the_failure_is_a_second_writer_and_escalates(
    tmp_path: Path, refusal_path: None
) -> None:
    """Real git, no patched push: the retry re-probes the remote, finds work that
    is not the worker's, and pushes nothing.

    Fresh dispatch on purpose: recovery with a branch already on origin never
    reaches the refuse-to-reset guard (the rework attach snapshots the old tip
    and resets to origin), which is existing behavior this change leaves alone.
    """
    remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    # Someone else pushes to the branch after the failed salvage.
    other = tmp_path / "other-clone"
    _clone_repo(remote, other)
    _git(other, "checkout", "-b", BRANCH, "origin/main")
    (other / "foreign.txt").write_text("not the worker's\n", encoding="utf-8")
    _git(other, "add", "foreign.txt")
    _git(other, "commit", "-m", "foreign commit")
    _git(other, "push", "origin", BRANCH)
    foreign = _git(other, "rev-parse", "HEAD").stdout.strip()

    with pytest.raises(WorktreeUnsafeError) as raised:
        _redispatch(repo, config, recovery=False)

    assert raised.value.kind == "worktree_unsafe_local_commits"
    assert _remote_sha(remote) == foreign
    assert _events(state_file, SALVAGE_PUSHED) == []


def test_kill_switch_zero_restores_escalate_on_first_sight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path, salvage_push_retries=0)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    push = _patch_push(monkeypatch, TIMEOUT_ERROR)

    with pytest.raises(WorktreeUnsafeError):
        _redispatch(repo, config)

    assert push.calls == 0
    assert _used_attempts(state_file) is None


def test_no_recorded_salvage_failure_means_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    """Control: without a ``salvage_push_failed`` on record the guard decides
    alone, exactly as in the #807 tests."""
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    push = _patch_push(monkeypatch, TIMEOUT_ERROR)

    with pytest.raises(WorktreeUnsafeError):
        _redispatch(repo, config)

    assert push.calls == 0


def _event(kind: str, branch: str, error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"issue_number": ISSUE, "branch": branch}
    if error is not None:
        payload["error"] = error
    return {"at": "x", "kind": kind, "payload": payload}


@pytest.mark.parametrize(
    ("events", "retryable"),
    [
        ([_event(SALVAGE_PUSH_FAILED, BRANCH, TIMEOUT_ERROR)], True),
        ([], False),
        ([_event(SALVAGE_PUSH_FAILED, "agent/issue-1-other", TIMEOUT_ERROR)], False),
        ([_event(SALVAGE_PUSH_FAILED, BRANCH, REJECTED_ERROR)], False),
        ([_event(SALVAGE_PUSH_FAILED, BRANCH, "branch_ref_missing")], False),
        (
            [
                _event(SALVAGE_PUSH_FAILED, BRANCH, TIMEOUT_ERROR),
                _event(SALVAGE_PUSHED, BRANCH),
            ],
            False,
        ),
        (
            [
                _event(SALVAGE_PUSHED, BRANCH),
                _event(SALVAGE_PUSH_FAILED, BRANCH, TIMEOUT_ERROR),
            ],
            True,
        ),
    ],
    ids=[
        "timeout",
        "empty",
        "other-branch",
        "rejection",
        "not-a-push-failure",
        "superseded-by-success",
        "failure-after-success",
    ],
)
def test_latest_retryable_failure_reads_only_the_newest_outcome(
    events: list[dict[str, Any]], retryable: bool
) -> None:
    found = latest_retryable_failure({"events": events}, ISSUE, BRANCH)
    assert (found is not None) is retryable


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (REJECTED_ERROR, True),
        ("push_failed: remote: error: GH006: Protected branch update failed", True),
        ("push_failed: hint: Updates were rejected because the tip is behind", True),
        ("push_failed: ! [remote rejected] x -> x (fetch first)", True),
        (TIMEOUT_ERROR, False),
        ("push_failed: command exited 1", False),
        (None, False),
    ],
)
def test_is_rejection(error: str | None, expected: bool) -> None:
    assert is_rejection(error) is expected
