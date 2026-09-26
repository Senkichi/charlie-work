"""Regression tests for issue #1684 — provider quota classification keyed on
the structured ``cognition.ai/errorKind`` trailer, not provider prose.

Root cause: both adapters carried an identical ``_QUOTA_EXHAUSTED_PATTERN``
regex that matched the literal word ``daily`` (``daily usage quota has been
exhausted|quota exceeded|usage limit``). When the Devin CLI changed the
sentence to ``weekly``, the classifier went silent: a zero-turn quota death
was recorded as an ordinary launch/session failure, burned
``review_dispatch_attempt_count`` (PR #1595: 3 launches in 18 minutes hit the
cap and escalated a PR nobody had reviewed), and never armed the reviewer
quota cooldown — so every pass relaunched into the same closed window.

The fix lives in one shared definition:
``throttle_signatures.match_quota_tail`` matches the structured
``"cognition.ai/errorKind": "resource_exhausted"`` trailer FIRST (stable
across provider wording), falling back to the config-driven, deliberately
period-agnostic prose markers in ``RuntimeConfig.quota_error_markers``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from charlie_work import claude_code, devin_shell
from charlie_work.config import (
    ConfigError,
    OrchestratorConfig,
    ReviewDispatchConfig,
    build_config_from_data,
)
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.throttle_signatures import (
    is_provider_throttle_failure,
    match_quota_tail,
    match_throttle_tail,
)
from charlie_work.workflow import (
    _classify_dead_sessions_and_update_throttle_state,
    _detect_and_handle_stalled_reviews,
    _reap_restore_rework_requested,
    _route_dead_worker_to_pre_review_rework,
)
from charlie_work.worker import WorkerView
from charlie_work.worktree import create_worktree
from charlie_work.write_gate import WriteGate

from _fakes_github import FakeGitHub
from _helpers import _init_git_repo
from _review_fixtures import _dispatch_reviews_app, _write_review_packet

# The exact message observed live 2026-09-17 on PR #1595 (issue #1684).
_WEEKLY_QUOTA_LOG = (
    "Error: Agent error: Your weekly usage quota has been exhausted. "
    "Visit https://usage.example.test to purchase on-demand usage or turn "
    "on auto-reload. (trace ID: abc123): {\n"
    '  "cognition.ai/errorKind": "resource_exhausted",\n'
    '  "cognition.ai/retryable": true\n'
    "}\n"
)

_RESOURCE_EXHAUSTED_TRAILER = '{"cognition.ai/errorKind": "resource_exhausted"}'


def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_bare_remote_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    remote = tmp_path / "remote"
    remote.mkdir(parents=True, exist_ok=True)
    _git(remote, "init", "--bare", "--initial-branch=main")
    clone = tmp_path / "clone"
    clone.mkdir(parents=True, exist_ok=True)
    _git(clone, "init", "--initial-branch=main")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test User")
    _git(clone, "config", "commit.gpgSign", "false")
    _git(clone, "remote", "add", "origin", str(remote))
    (clone / "README.md").write_text("hello\n", encoding="utf-8")
    _git(clone, "add", "README.md")
    _git(clone, "commit", "-m", "initial commit")
    _git(clone, "push", "-u", "origin", "main")
    return remote, clone


_DEFAULT_MARKERS = OrchestratorConfig().runtime.quota_error_markers


# ---------------------------------------------------------------------------
# match_quota_tail — the single shared classifier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        # The exact live weekly message (prose + structured trailer).
        (_WEEKLY_QUOTA_LOG, True),
        # An unknown/future period word still classifies: the structured
        # trailer wins whatever the sentence around it says.
        (
            "Error: Agent error: Your fortnightly usage quota has been "
            'exhausted. {"cognition.ai/errorKind": "resource_exhausted"}',
            True,
        ),
        # The trailer alone (no expected prose at all) still classifies —
        # the structured field is the primary signal.
        (_RESOURCE_EXHAUSTED_TRAILER, True),
        # Negative control: a DIFFERENT structured kind must NOT classify —
        # the matcher must not degrade to "any cognition.ai error".
        ('{"cognition.ai/errorKind": "internal"}', False),
        ('{"cognition.ai/errorKind": "permission_denied"}', False),
        # Legacy prose fallback (no trailer): the historical daily phrasing
        # plus period variants — the fallback is deliberately period-agnostic.
        ("Your daily usage quota has been exhausted.", True),
        ("Your weekly usage quota has been exhausted.", True),
        ("Your monthly usage quota has been exhausted.", True),
        ("quota exceeded", True),
        # Unrelated prose does not classify.
        ("Error: compilation failed in src/main.py", False),
        ("", False),
    ],
)
def test_match_quota_tail(tail: str, expected: bool) -> None:
    assert match_quota_tail(tail, _DEFAULT_MARKERS) is expected


def test_match_quota_tail_config_driven_marker() -> None:
    """Operators extend the prose fallback via ``quota_error_markers`` config —
    a new provider phrasing needs no code change."""
    tail = "Error: plan credit balance exhausted for this billing period"
    assert match_quota_tail(tail, _DEFAULT_MARKERS) is False
    assert match_quota_tail(tail, ("credit balance exhausted",)) is True


def test_exactly_one_quota_pattern_definition_in_src() -> None:
    """Acceptance: exactly one definition of the quota pattern exists in
    ``src/`` — the two hand-maintained ``_QUOTA_EXHAUSTED_PATTERN`` copies in
    ``devin_shell.py`` / ``claude_code.py`` are gone, and no source file may
    re-pin the period word."""
    src_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work"
    offenders = [
        path.name
        for path in src_dir.glob("*.py")
        if "_QUOTA_EXHAUSTED_PATTERN" in path.read_text(encoding="utf-8")
        or "daily usage quota" in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# Adapter classification (launch-time path) — both adapters share one helper
# ---------------------------------------------------------------------------


def _write_log(tmp_path: Path, name: str, text: str) -> Path:
    log_path = tmp_path / name
    log_path.write_text(text, encoding="utf-8")
    return log_path


def test_devin_classify_weekly_quota_log_is_quota_exhausted(tmp_path: Path) -> None:
    kind, throttled_until = devin_shell._classify_session_failure(
        _write_log(tmp_path, "issue-1.log", _WEEKLY_QUOTA_LOG)
    )
    assert kind == "quota_exhausted"
    assert throttled_until is not None


def test_claude_classify_weekly_quota_log_is_quota_exhausted(tmp_path: Path) -> None:
    kind, throttled_until = claude_code._classify_session_failure(
        _write_log(tmp_path, "issue-1.claude.log", _WEEKLY_QUOTA_LOG)
    )
    assert kind == "quota_exhausted"
    assert throttled_until is not None


def test_devin_classify_internal_error_kind_is_not_quota(tmp_path: Path) -> None:
    """Negative control at the classifier layer: a structured non-quota
    ``errorKind`` must not become ``quota_exhausted``."""
    kind, throttled_until = devin_shell._classify_session_failure(
        _write_log(
            tmp_path,
            "issue-1.log",
            'Error: Agent error: boom. {"cognition.ai/errorKind": "internal"}',
        )
    )
    assert kind is None
    assert throttled_until is None


def test_devin_update_session_record_persists_quota_kind(tmp_path: Path) -> None:
    """End-to-end at the sidecar layer: a weekly-quota log tail classifies the
    record ``quota_exhausted`` (not the caller's fallback) and returns a
    cooldown."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = _write_log(sessions_dir, "issue-42.log", _WEEKLY_QUOTA_LOG)
    record = devin_shell.SessionRecord(
        issue_number=42,
        branch="agent/issue-42",
        worktree_path="/tmp/wt-42",
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar = sessions_dir / "issue-42.json"
    sidecar.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    kind, throttled_until = devin_shell.update_session_record_with_failure_classification(
        sessions_dir,
        42,
        fallback_kind="launch_failed",
        config=OrchestratorConfig(),
    )
    assert kind == "quota_exhausted"
    assert throttled_until is not None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "quota_exhausted"


def test_claude_update_worker_record_persists_quota_kind(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = _write_log(sessions_dir, "issue-42.log", _WEEKLY_QUOTA_LOG)
    sidecar = sessions_dir / "issue-42.claude.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt-42",
                "prompt_path": "/tmp/prompt.md",
                "command": ["claude", "-p"],
                "pid": None,
                "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "log_path": str(log_path),
                "error": None,
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    kind, throttled_until = claude_code.update_worker_record_with_failure_classification(
        sessions_dir,
        42,
        fallback_kind="launch_failed",
        config=OrchestratorConfig(),
    )
    assert kind == "quota_exhausted"
    assert throttled_until is not None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "quota_exhausted"


def test_claude_update_worker_record_internal_kind_falls_back(tmp_path: Path) -> None:
    """Negative control through the sidecar path: a structured non-quota kind
    leaves the caller's fallback kind in place."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    log_path = _write_log(
        sessions_dir,
        "issue-42.log",
        'Error: Agent error: boom. {"cognition.ai/errorKind": "internal"}',
    )
    sidecar = sessions_dir / "issue-42.claude.json"
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": 42,
                "branch": "agent/issue-42",
                "worktree_path": "/tmp/wt-42",
                "prompt_path": "/tmp/prompt.md",
                "command": ["claude", "-p"],
                "pid": None,
                "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "log_path": str(log_path),
                "error": None,
                "adapter_kind": "claude-code",
            }
        ),
        encoding="utf-8",
    )

    kind, throttled_until = claude_code.update_worker_record_with_failure_classification(
        sessions_dir,
        42,
        fallback_kind="launch_failed",
        config=OrchestratorConfig(),
    )
    assert kind == "launch_failed"
    assert throttled_until is None


# ---------------------------------------------------------------------------
# dispatch_reviews launch-time quota_hit — the match_quota_tail OR-clause
# ---------------------------------------------------------------------------


def test_dispatch_reviews_quota_hit_on_structured_trailer_only(
    monkeypatch, tmp_path: Path
) -> None:
    """A launch failure whose error carries ONLY the structured
    ``cognition.ai/errorKind: resource_exhausted`` trailer — and no prose
    matching either marker list — must still set ``quota_hit``.

    Regression coverage for the ``or match_quota_tail(...)`` OR-clause
    ``dispatch_reviews`` gained under issue #1684: the pre-existing
    quota_hit tests reach ``quota_hit`` through the ``match_throttle_tail``
    prose branch (``"usage limit exceeded"`` matches
    ``throttle_error_markers``), so they cannot catch a regression that
    deletes the structured-trailer clause. This error string is built to
    miss every ``throttle_error_markers`` and every ``quota_error_markers``
    substring — only the ``match_quota_tail`` branch can fire. If that
    clause regresses, the launch failure is recorded as an ordinary
    per-PR failure instead of a global quota hit.
    """
    error_text = "Error: reviewer exited before first turn: " + _RESOURCE_EXHAUSTED_TRAILER
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs)
    _write_review_packet(tmp_path, 100, "sha-100")

    # Guard the premise: this error string must exercise ONLY the
    # structured-trailer path. Asserted here so a future marker-list
    # addition cannot silently turn this back into a prose-branch test.
    assert match_throttle_tail(error_text, app.config.runtime.throttle_error_markers)[0] is False
    assert not any(
        marker.lower() in error_text.lower() for marker in app.config.runtime.quota_error_markers
    )
    # The trailer alone classifies even with an empty prose fallback list.
    assert match_quota_tail(error_text, ()) is True

    def fake_launch(*args: Any, **kwargs: Any) -> claude_code.ClaudeWorkerRecord:
        return claude_code.ClaudeWorkerRecord(
            issue_number=kwargs.get("issue_number") or args[0],
            branch=kwargs.get("branch") or args[1],
            worktree_path="/fake/worktree",
            prompt_path="/fake/prompt.md",
            command=("claude", "-p", "--permission-mode", "plan"),
            pid=None,
            started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            log_path="/fake/log.log",
            error=error_text,
            process_start_time=1.0,
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    result = app.dispatch_reviews()
    state = load_state(app.paths.state_file)

    assert result.data["launched_count"] == 0
    assert result.data.get("quota_hit") is True
    assert state["reviewer_quota"]["throttled_until"] is not None
    # Claim rolled back — a global provider condition is not charged to the
    # PR as a failed dispatch.
    assert state["prs"]["100"].get("review_dispatch_status") is None


# ---------------------------------------------------------------------------
# Stalled-review reaper — a zero-turn quota death rolls back the attempt
# counter and arms the reviewer-quota cooldown
# ---------------------------------------------------------------------------


def _write_dead_devin_reviewer(
    reviews_dir: Path, pr_number: int, tmp_path: Path, log_text: str
) -> Path:
    """Fabricate a dead devin reviewer sidecar + log."""
    log_path = reviews_dir / f"issue-{pr_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    record = devin_shell.SessionRecord(
        issue_number=pr_number,
        branch=f"agent/issue-{pr_number}-fix",
        worktree_path=str(tmp_path / "worktrees" / f"issue-{pr_number}"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("devin", "--prompt-file", str(tmp_path / "prompt.md"), "--print"),
        pid=999999999,  # not a real live pid
        started_at=started,
        log_path=str(log_path),
        error=None,
        process_start_time=1.0,
    )
    sidecar_path = reviews_dir / f"issue-{pr_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return sidecar_path


def _seed_review_claim(tmp_path: Path, pr_number: int, attempt_count: int):
    repo_root = tmp_path / "repo"
    _init_git_repo(repo_root)
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(enabled=True))
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}), encoding="utf-8"
    )
    started = (datetime.now(UTC) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    with state_lock(state_file):
        state = load_state(state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": started,
            "reviewer_pid": 999999999,
            "reviewer_process_start_time": 1.0,
            "review_dispatch_attempt_count": attempt_count,
        }
        save_state(state_file, state)
    return repo_root, reviews_dir, config, state_file


def test_quota_dead_reviewer_rolls_back_attempt_and_arms_cooldown(tmp_path: Path) -> None:
    """The issue-#1684 acceptance case: a devin reviewer that dies zero-turn on
    the weekly quota wall (structured ``resource_exhausted`` trailer) must be
    classified as a provider throttle death — the attempt count is rolled
    back, the claim is cleared, and the reviewer-quota cooldown is armed so
    the next ``dispatch_reviews`` pass defers instead of relaunching into the
    same window."""
    repo_root, reviews_dir, config, state_file = _seed_review_claim(tmp_path, 100, 1)
    sidecar_path = _write_dead_devin_reviewer(reviews_dir, 100, tmp_path, _WEEKLY_QUOTA_LOG)

    stalled = _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    pr_state = state["prs"]["100"]

    # Attempt rolled back: a provider-wall death is not a PR-level outcome.
    assert pr_state.get("review_dispatch_attempt_count") == 0
    # Claim cleared — re-dispatchable once the quota window opens.
    assert pr_state.get("review_dispatch_status") is None
    # Reviewer-quota cooldown armed.
    quota = state.get("reviewer_quota", {})
    assert quota.get("throttled_until") is not None
    assert quota.get("consecutive_probe_failures") == 1
    # The dead sidecar is reaped so it cannot re-fire the backoff every pass.
    assert not sidecar_path.exists()
    assert any(
        entry.get("pr") == 100 and entry.get("reason") == "provider_throttled" for entry in stalled
    )


def test_non_quota_dead_reviewer_is_not_throttled(tmp_path: Path) -> None:
    """Negative control at the reaper layer: a dead reviewer whose log carries
    a structured non-quota errorKind is an ordinary failure — no rollback, no
    cooldown."""
    repo_root, reviews_dir, config, state_file = _seed_review_claim(tmp_path, 100, 1)
    _write_dead_devin_reviewer(
        reviews_dir,
        100,
        tmp_path,
        'Error: Agent error: internal failure. {"cognition.ai/errorKind": "internal"}\n',
    )

    _detect_and_handle_stalled_reviews(
        reviews_dir, state_file, config, repo_root, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    pr_state = state["prs"]["100"]
    assert pr_state.get("review_dispatch_attempt_count") == 1
    assert pr_state.get("review_dispatch_status") == "review_dispatch_failed"
    assert state.get("reviewer_quota") in (None, {})


# ---------------------------------------------------------------------------
# Worker lane — a zero-turn quota death does not burn rework/dispatch caps
# ---------------------------------------------------------------------------


def _write_dead_devin_worker(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path, log_text: str
) -> Path:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    record = devin_shell.SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return sidecar_path


def _classify_fixture(tmp_path: Path, issue_number: int, log_text: str):
    """Dead devin worker + empty worktree + active-label issue, no open PRs.

    Returns (repo_root, sessions_dir, state_file, gh, config, sidecar_path).
    """
    remote, repo_root = _init_bare_remote_and_clone(tmp_path / "repo")
    branch = f"agent/issue-{issue_number}"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sidecar_path = _write_dead_devin_worker(
        sessions_dir, issue_number, branch, info.path, log_text
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": []}), encoding="utf-8")
    config = OrchestratorConfig()
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": issue_number,
            "title": f"issue {issue_number}",
            "url": f"https://example.test/issues/{issue_number}",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    gh.prs = []
    return repo_root, sessions_dir, state_file, gh, config, sidecar_path


def test_dead_worker_quota_death_does_not_burn_redispatch_cap(tmp_path: Path) -> None:
    """A zero-turn worker death on ``resource_exhausted`` is a global provider
    condition: the fleet-wide cooldown is armed and the issue relabels to
    ready, but the death must NOT consume the issue's redispatch cap."""
    (
        _repo_root,
        sessions_dir,
        state_file,
        gh,
        config,
        sidecar_path,
    ) = _classify_fixture(tmp_path, 1684, _WEEKLY_QUOTA_LOG)

    reaped = _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    assert len(reaped) == 1
    assert reaped[0]["failure_kind"] == "quota_exhausted"
    assert not sidecar_path.exists()

    state = load_state(state_file)
    # Fleet-wide cooldown armed.
    assert state.get("throttled_until") is not None
    # Issue relabeled to ready — it redispatches once the window opens...
    assert (1684, config.labels.in_progress) in gh.labels_removed
    assert (1684, config.labels.ready) in gh.labels_added
    # ...but the death did not consume the redispatch cap.
    entry = state["issues"].get("1684", {})
    assert entry.get("redispatch_at") in (None, [])
    assert entry.get("status") != "escalated"


def test_dead_worker_non_throttle_death_counts_redispatch(tmp_path: Path) -> None:
    """Control for the cap gate: an ordinary dead worker still counts one
    redispatch entry, proving the gate is specific to provider-throttle
    kinds."""
    (
        _repo_root,
        sessions_dir,
        state_file,
        gh,
        config,
        _sidecar_path,
    ) = _classify_fixture(tmp_path, 1685, "Error: worker crashed unexpectedly\n")

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir, state_file, gh, config, write_gate=_wg(state_file)
    )

    state = load_state(state_file)
    entry = state["issues"].get("1685", {})
    assert len(entry.get("redispatch_at") or []) == 1


def _seed_dispatched_rework(tmp_path: Path, issue_number: int, pr_number: int, branch: str):
    """State for a rework worker: issue ``dispatched``, open PR with a live
    ``request_changes`` review decision."""
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {
                    str(issue_number): {
                        "status": "dispatched",
                        "dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    }
                },
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    pr_dir = tmp_path / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "reviewed_head_sha": "sha-live-head",
                "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )
    pr = {
        "number": pr_number,
        "headRefName": branch,
        "headRefOid": "sha-live-head",
        "state": "OPEN",
    }
    return state_file, {issue_number: [pr]}


def _worker_view(issue_number: int, branch: str, tmp_path: Path) -> WorkerView:
    return WorkerView(
        adapter_kind="devin",
        issue_number=issue_number,
        repo_key="",
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        process_start_time=None,
        log_path=str(tmp_path / f"issue-{issue_number}.log"),
        worktree_path=str(tmp_path / "wt" / f"issue-{issue_number}"),
        error=None,
        failure_kind=None,
        reclaimed=None,
        branch=branch,
    )


def test_rework_restore_quota_death_does_not_burn_caps(tmp_path: Path) -> None:
    """``_reap_restore_rework_requested``: a provider-throttle-classified death
    restores ``rework_requested`` but consumes neither ``redispatch_at`` nor
    ``worker_death_at``."""
    issue_number, pr_number, branch = 1684, 500, "agent/issue-1684"
    state_file, open_prs = _seed_dispatched_rework(tmp_path, issue_number, pr_number, branch)
    config = OrchestratorConfig()
    gh = FakeGitHub()

    _reap_restore_rework_requested(
        state_file,
        gh,
        config,
        open_prs,
        _worker_view(issue_number, branch, tmp_path),
        failure_kind="quota_exhausted",
        repo_root=None,
        write_gate=_wg(state_file),
    )

    entry = load_state(state_file)["issues"][str(issue_number)]
    assert entry["status"] == "rework_requested"
    assert entry.get("redispatch_at") in (None, [])
    assert entry.get("worker_death_at") in (None, [])


def test_rework_restore_stalled_death_counts_caps(tmp_path: Path) -> None:
    """Control for the rework-restore gate: an ordinary death still records
    both counters."""
    issue_number, pr_number, branch = 1685, 501, "agent/issue-1685"
    state_file, open_prs = _seed_dispatched_rework(tmp_path, issue_number, pr_number, branch)
    config = OrchestratorConfig()
    gh = FakeGitHub()

    _reap_restore_rework_requested(
        state_file,
        gh,
        config,
        open_prs,
        _worker_view(issue_number, branch, tmp_path),
        failure_kind="stalled",
        repo_root=None,
        write_gate=_wg(state_file),
    )

    entry = load_state(state_file)["issues"][str(issue_number)]
    assert entry["status"] == "rework_requested"
    assert len(entry.get("redispatch_at") or []) == 1
    assert len(entry.get("worker_death_at") or []) == 1


def test_pre_review_rework_quota_death_does_not_burn_cap(tmp_path: Path) -> None:
    """``_route_dead_worker_to_pre_review_rework``: with the cap already at
    ``max_auto_redispatch``, a provider-throttle death must not push the
    count over the edge — the issue routes to rework instead of escalating."""
    issue_number, pr_number = 1684, 500
    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"
    seeded = [
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
        for _ in range(config.watchdog.max_auto_redispatch)
    ]
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {str(issue_number): {"status": "dispatched", "redispatch_at": seeded}},
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    gh = FakeGitHub()

    _route_dead_worker_to_pre_review_rework(
        state_file,
        gh,
        config,
        {"number": pr_number},
        issue_number,
        "stale_empty_checks",
        failure_kind="quota_exhausted",
        write_gate=_wg(state_file),
    )

    entry = load_state(state_file)["issues"][str(issue_number)]
    assert entry["status"] == "rework_requested"
    assert entry.get("escalation_reason") is None


def test_pre_review_rework_stalled_death_escalates_at_cap(tmp_path: Path) -> None:
    """Control: the same cap-saturated entry escalates on an ordinary death —
    the quota exemption is specific to provider-throttle kinds."""
    issue_number, pr_number = 1685, 501
    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"
    seeded = [
        datetime.now(UTC).isoformat().replace("+00:00", "Z")
        for _ in range(config.watchdog.max_auto_redispatch)
    ]
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "issues": {str(issue_number): {"status": "dispatched", "redispatch_at": seeded}},
                "prs": {},
                "events": [],
            }
        ),
        encoding="utf-8",
    )
    gh = FakeGitHub()

    _route_dead_worker_to_pre_review_rework(
        state_file,
        gh,
        config,
        {"number": pr_number},
        issue_number,
        "stale_empty_checks",
        failure_kind="stalled",
        write_gate=_wg(state_file),
    )

    entry = load_state(state_file)["issues"][str(issue_number)]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "redispatch_cap_exceeded"


# ---------------------------------------------------------------------------
# Config — quota_error_markers is a validated, operator-extendable list
# ---------------------------------------------------------------------------


def test_quota_error_markers_loads_from_config() -> None:
    config = build_config_from_data(
        {"runtime": {"quota_error_markers": ["credit balance exhausted"]}}
    )
    assert config.runtime.quota_error_markers == ("credit balance exhausted",)


def test_quota_error_markers_rejects_non_list() -> None:
    with pytest.raises(ConfigError):
        build_config_from_data({"runtime": {"quota_error_markers": "credit balance"}})


def test_quota_error_markers_rejects_non_string_element() -> None:
    with pytest.raises(ConfigError):
        build_config_from_data({"runtime": {"quota_error_markers": [42]}})


@pytest.mark.parametrize(
    ("failure_kind", "expected"),
    [
        ("rate_limited", True),
        ("quota_exhausted", True),
        ("provider_auth", True),
        # Terminal, not a cooldown: escalates on first occurrence via
        # DETERMINISTIC_ESCALATION_FAILURE_KINDS — deliberately absent
        # from PROVIDER_THROTTLE_FAILURE_KINDS.
        ("provider_suspended", False),
        ("stalled", False),
        ("launch_failed", False),
        ("worker_blocked", False),
        (None, False),
    ],
)
def test_is_provider_throttle_failure(failure_kind: str | None, expected: bool) -> None:
    """The single-point-of-enforcement predicate every death-credit lane
    and the #1917 timed-reap exemption share — provider-throttle kinds
    True, ``provider_suspended`` (terminal, not a cooldown) False, and
    ``None`` (unclassified death) never a throttle kind."""
    assert is_provider_throttle_failure(failure_kind) is expected
