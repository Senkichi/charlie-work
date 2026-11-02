"""Issue #2847: a transient salvage-push failure must not strand a dead worker's commit.

The orphan sweep tries ``salvage_push_stranded_commits`` once. When that push
times out, the commit stays in the worktree and the next redispatch used to
read it as divergent local work (``worktree_unsafe_local_commits``, a judgment
escalation). ``create_worktree`` now re-pushes it first, bounded by
``dispatch.salvage_push_retries``, and escalates exactly as before when the
budget is spent, the push is rejected, or the remote moved. The result of the
retry's push is handled by cause: work already on origin attaches, a push
timeout or an unanswered probe defers, anything that means a second writer
falls through to the guards.

The #807/#257 refusal tests in ``test_worktree.py`` call ``create_worktree``
without a ``config``; they are the unchanged control for the no-retry path.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
from _worktree_fixtures import _clone_repo, _git, _init_bare_remote_and_clone

from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.paths import runtime_paths
from charlie_work import salvage_retry
from charlie_work import state as state_module
from charlie_work.salvage_retry import (
    FALL_THROUGH_SKIPS,
    ON_ORIGIN_SKIPS,
    RETRY_COUNT_FIELD,
    SALVAGE_PUSH_FAILED,
    SALVAGE_PUSHED,
    TRANSIENT_SKIPS,
    SalvageRetryOutcome,
    is_transient_push_failure,
    latest_retryable_failure,
    retry_failed_salvage_push,
    skip_kind,
)
from charlie_work.state import StateLockBusy, append_event, load_state, save_state, state_lock
from charlie_work.subprocess_runner import run_captured
from charlie_work.unescalate_reset_fields import UNESCALATE_ISSUE_RESET_FIELDS
from charlie_work.worktree import (
    RescueCapture,
    SalvagePushResult,
    WorktreeInfo,
    WorktreeProbeFailedError,
    WorktreeUnsafeError,
    create_worktree,
    salvage_push_stranded_commits,
)

ISSUE = 2847
BRANCH = "agent/issue-2847-stranded-salvage"
TIMEOUT_ERROR = "push_failed: command timed out after 60s"


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
    return _patch_result(monkeypatch, SalvagePushResult(pushed=False, error=error, commit_count=1))


def _patch_result(monkeypatch: pytest.MonkeyPatch, result: SalvagePushResult) -> _FakePush:
    fake = _FakePush(result)
    monkeypatch.setattr("charlie_work.worktree.salvage_push_stranded_commits", fake)
    return fake


def _reject_pushes(remote: Path) -> Path:
    """Make the bare remote refuse every push, the way a protected branch or a
    failing pre-receive gate does. Returns the hook so a test can lift it."""
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho 'rejected by test hook' >&2\nexit 1\n", newline="\n")
    hook.chmod(0o755)
    return hook


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


def test_rejection_recorded_by_the_real_push_path_is_never_retried(
    tmp_path: Path, refusal_path: None
) -> None:
    """Acceptance (c), end to end: the failure text is whatever the real sweep push
    records for a refused push, not a hand-written string."""
    remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    hook = _reject_pushes(remote)
    swept = salvage_push_stranded_commits(repo, BRANCH, info.path, base_ref="origin/main")
    # Production records ``command exited N``; the server's wording is not in it.
    assert swept.error == "push_failed: command exited 1"
    assert not is_transient_push_failure(swept.error)
    _record_salvage_failure(state_file, swept.error)
    hook.unlink()  # a retry that wrongly fired would land the commit now

    with pytest.raises(WorktreeUnsafeError) as raised:
        _redispatch(repo, config)

    assert raised.value.kind == "worktree_unsafe_local_commits"
    assert _remote_sha(remote) is None
    assert _used_attempts(state_file) is None


def test_rejected_retry_escalates_at_once_and_is_never_retried_again(
    tmp_path: Path, refusal_path: None
) -> None:
    """The recorded timeout earns one retry; the retry's own push is refused by the
    remote. That is not transient: escalate on first sight, do not defer."""
    remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    hook = _reject_pushes(remote)

    with pytest.raises(WorktreeUnsafeError) as raised:
        _redispatch(repo, config)

    assert raised.value.kind == "worktree_unsafe_local_commits"
    assert _used_attempts(state_file) == 1
    failures = _events(state_file, SALVAGE_PUSH_FAILED)
    assert [f.get("retry_attempt") for f in failures] == [None, 1]
    assert not is_transient_push_failure(failures[-1]["error"])

    # The refusal is now the newest outcome: no second push, even once the
    # remote would accept one.
    hook.unlink()
    with pytest.raises(WorktreeUnsafeError):
        _redispatch(repo, config)
    assert _remote_sha(remote) is None
    assert _used_attempts(state_file) == 1


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
        ([_event(SALVAGE_PUSH_FAILED, BRANCH, "push_failed: command exited 1")], False),
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
        "non-timeout-failure",
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
        (TIMEOUT_ERROR, True),
        ("push_failed: command timed out after 5s", True),
        ("push_failed: command exited 1", False),
        ("push_failed: ! [rejected] x -> x (non-fast-forward)", False),
        ("push_failed: remote: error: GH006: Protected branch update failed", False),
        ("command timed out after 60s", False),
        ("remote_probe_failed", False),
        ("", False),
        (None, False),
    ],
)
def test_only_a_push_timeout_is_transient(error: str | None, expected: bool) -> None:
    assert is_transient_push_failure(error) is expected


def test_transient_signature_is_what_run_captured_actually_records(tmp_path: Path) -> None:
    """The allow-list is anchored to the real producer, not to a string I typed: a
    command that times out reads transient, one that exits non-zero does not."""
    slept = run_captured(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, timeout_seconds=1
    )
    failed = run_captured(
        [sys.executable, "-c", "raise SystemExit(1)"], cwd=tmp_path, timeout_seconds=30
    )

    assert is_transient_push_failure(f"push_failed: {slept.error}") is True
    assert is_transient_push_failure(f"push_failed: {failed.error}") is False


# ---------------------------------------------------------------------------
# The retry's own result, handled by cause
# ---------------------------------------------------------------------------


def _ledger_push(
    state_file: Path, result: SalvagePushResult, *, limit: int = 3
) -> SalvageRetryOutcome:
    return retry_failed_salvage_push(
        state_file, ISSUE, BRANCH, limit=limit, push=_FakePush(result)
    )


def _failure_bed(tmp_path: Path) -> Path:
    state_file = tmp_path / "ledger" / "state.json"
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    return state_file


def _skipped(reason: str) -> SalvagePushResult:
    return SalvagePushResult(pushed=False, skip_reason=reason)


@pytest.mark.parametrize(
    ("result", "on_origin", "defer"),
    [
        *[(_skipped(k), False, True) for k in sorted(TRANSIENT_SKIPS)],
        *[(_skipped(k), True, False) for k in sorted(ON_ORIGIN_SKIPS)],
        *[(_skipped(k), False, False) for k in sorted(FALL_THROUGH_SKIPS)],
        (SalvagePushResult(pushed=True, new_remote_sha="a" * 40, commit_count=1), True, False),
        (SalvagePushResult(pushed=False, error=TIMEOUT_ERROR), False, True),
        (SalvagePushResult(pushed=False, error="push_failed: command exited 1"), False, False),
    ],
)
def test_outcome_is_decided_by_cause(
    tmp_path: Path, result: SalvagePushResult, on_origin: bool, defer: bool
) -> None:
    outcome = _ledger_push(_failure_bed(tmp_path), result)

    assert outcome.attempted is True
    assert (outcome.on_origin, outcome.defer) == (on_origin, defer)


def test_no_deferral_on_the_last_attempt_so_the_guards_get_the_final_word(tmp_path: Path) -> None:
    state_file = _failure_bed(tmp_path)
    probe_failed = _skipped("remote_probe_failed")

    first = _ledger_push(state_file, probe_failed, limit=2)
    last = _ledger_push(state_file, probe_failed, limit=2)

    assert (first.attempt, first.defer) == (1, True)
    assert (last.attempt, last.defer) == (2, False)
    assert _ledger_push(state_file, probe_failed, limit=2).attempted is False


def test_work_found_on_origin_is_recorded_as_landed_and_ends_the_episode(tmp_path: Path) -> None:
    state_file = _failure_bed(tmp_path)
    tip = "b" * 40

    outcome = _ledger_push(
        state_file, SalvagePushResult(pushed=False, skip_reason="up_to_date", old_remote_sha=tip)
    )

    assert outcome.on_origin is True
    landed = _events(state_file, SALVAGE_PUSHED)
    assert [(p["retry_attempt"], p["already_on_origin"], p["new_remote_sha"]) for p in landed] == [
        (1, True, tip)
    ]
    assert _used_attempts(state_file) is None
    # The stale timeout is no longer the newest outcome: nothing left to retry.
    assert _ledger_push(state_file, _skipped("up_to_date")).attempted is False


@pytest.mark.parametrize("recovery", [True, False], ids=["recovery", "fresh-dispatch"])
def test_work_already_on_origin_attaches_instead_of_resetting(
    tmp_path: Path, refusal_path: None, recovery: bool
) -> None:
    """The sweep's push landed and only its verification timed out. The retry finds
    the branch up to date: a fresh dispatch must attach to it, not reset to base."""
    remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    _git(info.path, "push", "origin", BRANCH)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)

    redispatched = _redispatch(repo, config, recovery=recovery)

    assert _remote_sha(remote) == local_tip
    assert redispatched.path == info.path
    assert _git(redispatched.path, "rev-parse", "HEAD").stdout.strip() == local_tip
    assert (redispatched.path / "work.txt").read_text(encoding="utf-8") == "stranded work\n"
    assert [p["already_on_origin"] for p in _events(state_file, SALVAGE_PUSHED)] == [True]
    assert _used_attempts(state_file) is None


def test_unanswered_probe_defers_instead_of_escalating_and_a_later_launch_recovers(
    tmp_path: Path, refusal_path: None
) -> None:
    """Real git: origin is unreachable for the retry's re-probe. That is a failed
    probe (``remote_probe_failed``), not a second writer -- defer, keep the commit,
    and push it on the next redispatch once origin answers."""
    remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "unreachable.git"))

    with pytest.raises(
        WorktreeProbeFailedError, match=r"retry 1/3 .*did not land \(remote_probe_failed\)"
    ):
        _redispatch(repo, config, recovery=False)

    assert _used_attempts(state_file) == 1
    assert (info.path / "work.txt").exists()
    assert _remote_sha(remote) is None

    _git(repo, "remote", "set-url", "origin", str(remote))
    redispatched = _redispatch(repo, config, recovery=False)

    assert _remote_sha(remote) == local_tip
    assert _git(redispatched.path, "rev-parse", "HEAD").stdout.strip() == local_tip
    assert [p["retry_attempt"] for p in _events(state_file, SALVAGE_PUSHED)] == [2]


@pytest.mark.parametrize("reason", sorted(TRANSIENT_SKIPS))
def test_transient_skip_defers_while_attempts_remain_then_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None, reason: str
) -> None:
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path, salvage_push_retries=2)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    push = _patch_result(monkeypatch, _skipped(reason))

    with pytest.raises(WorktreeProbeFailedError, match=rf"retry 1/2 .*\({reason}\)"):
        _redispatch(repo, config)
    with pytest.raises(WorktreeUnsafeError) as final:
        _redispatch(repo, config)

    assert final.value.kind == "worktree_unsafe_local_commits"
    assert (push.calls, _used_attempts(state_file)) == (2, 2)


@pytest.mark.parametrize(
    "reason", ["diverged", "remote_head_not_local", "live_writer_marker", "operator_claimed"]
)
def test_second_writer_skip_falls_through_to_the_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None, reason: str
) -> None:
    assert reason in FALL_THROUGH_SKIPS
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    push = _patch_result(monkeypatch, _skipped(reason))

    with pytest.raises(WorktreeUnsafeError) as raised:
        _redispatch(repo, config)

    assert raised.value.kind == "worktree_unsafe_local_commits"
    assert push.calls == 1
    assert [f.get("retry_attempt") for f in _events(state_file, SALVAGE_PUSH_FAILED)] == [None]


def _declared_skip_kinds() -> set[str]:
    """Every ``skip_reason=`` ``salvage_push_stranded_commits`` can return, read from
    its source. An expression this cannot read fails the test instead of passing it."""
    kinds: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(salvage_push_stranded_commits))):
        if not (isinstance(node, ast.keyword) and node.arg == "skip_reason"):
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            text = value.value
        elif (
            isinstance(value, ast.JoinedStr)
            and value.values
            and isinstance(value.values[0], ast.Constant)
        ):
            text = str(value.values[0].value)
        else:
            pytest.fail(f"cannot classify skip_reason expression: {ast.dump(value)}")
        kinds.add(skip_kind(text))
    return kinds


def test_every_skip_reason_has_exactly_one_cause_class() -> None:
    declared = _declared_skip_kinds()
    groups = (TRANSIENT_SKIPS, ON_ORIGIN_SKIPS, FALL_THROUGH_SKIPS)
    classified = set().union(*groups)

    # Positive control: the scan found the reasons it is supposed to find.
    assert {"remote_probe_failed", "up_to_date", "diverged", "invalid_branch"} <= declared
    assert sum(len(group) for group in groups) == len(classified), "a reason is in two classes"
    assert declared == classified, (
        f"unclassified: {declared - classified}; stale: {classified - declared}"
    )


# ---------------------------------------------------------------------------
# Fail-safe paths: bookkeeping trouble never blocks or loses a launch
# ---------------------------------------------------------------------------


class _LockedState:
    """``charlie_work.state`` with the advisory lock held by another writer."""

    def __getattr__(self, name: str) -> Any:
        return getattr(state_module, name)

    def state_lock(self, *args: Any, **kwargs: Any) -> Any:
        raise StateLockBusy("held by another writer")


def test_unavailable_state_lock_means_no_retry_and_the_guards_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    _remote, repo, _info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    push = _patch_push(monkeypatch, TIMEOUT_ERROR)
    monkeypatch.setattr(salvage_retry, "_state", _LockedState())

    with pytest.raises(WorktreeUnsafeError) as raised:  # NOT StateLockBusy
        _redispatch(repo, config)

    assert raised.value.kind == "worktree_unsafe_local_commits"
    assert push.calls == 0
    assert _used_attempts(state_file) is None


def test_failed_outcome_write_after_a_landed_push_still_reports_the_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, refusal_path: None
) -> None:
    remote, repo, info = _stranded_bed(tmp_path)
    config = _config(tmp_path)
    state_file = _state_file(repo, config)
    _record_salvage_failure(state_file, TIMEOUT_ERROR)
    local_tip = _git(info.path, "rev-parse", "HEAD").stdout.strip()

    def _disk_full(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(salvage_retry, "_record_outcome", _disk_full)

    # Fresh dispatch: it only attaches (rework=True) if the push was reported.
    redispatched = _redispatch(repo, config, recovery=False)

    assert _remote_sha(remote) == local_tip
    assert redispatched.path == info.path
    assert _git(redispatched.path, "rev-parse", "HEAD").stdout.strip() == local_tip
    assert (redispatched.path / "work.txt").read_text(encoding="utf-8") == "stranded work\n"
    assert _events(state_file, SALVAGE_PUSHED) == []  # the write failed, the push did not
    assert _used_attempts(state_file) == 1


# ---------------------------------------------------------------------------
# Un-escalate re-arms the retry budget
# ---------------------------------------------------------------------------


def test_unescalate_drops_the_retry_count_so_a_later_stranding_gets_a_fresh_budget(
    tmp_path: Path,
) -> None:
    from _unescalate_fixtures import _app

    assert RETRY_COUNT_FIELD in UNESCALATE_ISSUE_RESET_FIELDS
    app = _app(tmp_path)
    state_file = app.paths.state_file
    with state_lock(state_file):
        state = load_state(state_file)
        state["issues"][str(ISSUE)] = {
            "number": ISSUE,
            "status": "escalated",
            "branch_name": BRANCH,
            RETRY_COUNT_FIELD: 3,
        }
        state = append_event(
            state,
            SALVAGE_PUSH_FAILED,
            {"issue_number": ISSUE, "branch": BRANCH, "error": TIMEOUT_ERROR},
        )
        save_state(state_file, state)
    failed = SalvagePushResult(pushed=False, error=TIMEOUT_ERROR, commit_count=1)
    assert _ledger_push(state_file, failed).attempted is False  # budget spent before the re-arm

    result = app.unescalate(issue_number=ISSUE)

    assert result.ok is True
    assert RETRY_COUNT_FIELD not in load_state(state_file)["issues"][str(ISSUE)]
    again = _ledger_push(state_file, failed)
    assert (again.attempted, again.attempt) == (True, 1)
