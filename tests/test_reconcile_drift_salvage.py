"""Worktree-salvage drift tests for ``reconcile.detect_drift`` /
``reconcile.apply_fixes``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1): completed
worktree log-tail classification, unpublished/dirty-worktree salvage, the
no-commits relabel, and the salvage fix lane (PR creation, labels, push
fallback).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from _reconcile_fixtures import (
    FakeGitHub,
    _init_bare_remote_and_clone,
    _issue,
    _setup_completed_worktree,
)
from _worktree_fixtures import _git
from charlie_work.config import OrchestratorConfig
from charlie_work.devin_shell import SessionRecord
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes,
    detect_drift,
)
from charlie_work.state import empty_state
from charlie_work.workflow import OrchestratorApp
from charlie_work.worktree import create_worktree


def _write_dead_session_sidecar(
    sessions_dir: Path, issue_number: int, branch: str, worktree_path: Path
) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    record = SessionRecord(
        issue_number=issue_number,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path="/tmp/prompt.md",
        command=("devin", "--prompt-file", "/tmp/prompt.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(sessions_dir / f"issue-{issue_number}.log"),
        error=None,
    )
    sidecar_path = sessions_dir / f"issue-{issue_number}.json"
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    (sessions_dir / f"issue-{issue_number}.claude.json").unlink(missing_ok=True)


# Adapter-kind -> sidecar filename suffix. Mirrors claude_code._ADAPTER_SIDECAR_SUFFIXES
# without importing it (keeps the test's failure surface independent of the adapter).
_ADAPTER_SIDECAR_SUFFIX = {"devin": "", "claude-code": ".claude", "api": ".api"}


def _write_dead_session_sidecar_for_adapter(
    sessions_dir: Path,
    issue_number: int,
    branch: str,
    worktree_path: Path,
    adapter_kind: str,
    log_text: str,
) -> Path:
    """Write a dead-session sidecar for any adapter kind, plus its log file.

    Unlike ``_write_dead_session_sidecar`` (devin-only, no log content), this
    also writes the log file with ``log_text`` so log-tail classification has
    real bytes to match against -- required for issue #656 regression coverage
    where the log must carry a throttle marker that *would* reclassify a
    non-completed session.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.log"
    log_path.write_text(log_text, encoding="utf-8")
    suffix = _ADAPTER_SIDECAR_SUFFIX[adapter_kind]
    sidecar_path = sessions_dir / f"issue-{issue_number}{suffix}.json"
    if adapter_kind == "devin":
        record = SessionRecord(
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
        sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    else:
        # claude-code / api share the ClaudeWorkerRecord on-disk shape; the
        # ``adapter_kind`` field disambiguates them (worker._from_claude_record
        # honors it so api sidecars surface as adapter_kind=="api").
        sidecar_path.write_text(
            json.dumps(
                {
                    "issue_number": issue_number,
                    "branch": branch,
                    "worktree_path": str(worktree_path),
                    "prompt_path": "/tmp/prompt.md",
                    "command": ["claude", "-p"],
                    "pid": None,
                    "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    "log_path": str(log_path),
                    "error": None,
                    "adapter_kind": adapter_kind,
                }
            ),
            encoding="utf-8",
        )
    return sidecar_path


@pytest.mark.parametrize("adapter_kind", ["devin", "claude-code", "api"])
def test_detect_drift_completed_worktree_skips_log_tail_throttle_classification(
    tmp_path: Path, adapter_kind: str
) -> None:
    """Issue #656 regression: a completed worktree's log-tail throttle markers
    must NOT emit ``provider_throttle_detected`` drift.

    This guards the three ``session_completed=True`` call sites in
    ``reconcile.detect_drift`` (one per adapter kind). The log file is seeded
    with ``"usage limit"`` -- a ``match_quota_tail`` / ``quota_error_markers``
    substring that, if log-tail classification ran, would return
    ``quota_exhausted`` plus a 24h ``throttled_until`` and emit a
    ``provider_throttle_detected`` drift item.
    The worktree inspection is ground truth the session completed, so
    ``session_completed=True`` must skip log-tail matching entirely.

    If ``session_completed=True`` is silently dropped from any of the three
    call sites, this test fails: a ``provider_throttle_detected`` drift item
    appears and the salvage drift (which proves the is_completed lane was
    taken) is shadowed by the throttle.
    """
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    issue_number = 656
    worktree_path, branch = _setup_completed_worktree(repo_root, issue_number)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    # Log tail that quotes a throttle marker in legitimate completion prose --
    # the exact false-positive shape observed live 2026-07-27.
    _write_dead_session_sidecar_for_adapter(
        sessions_dir,
        issue_number,
        branch,
        worktree_path,
        adapter_kind,
        log_text=(
            '## Summary\n\nFixed generic substrings ("rate limit", "usage limit") '
            "that legitimately appear in this codebase's rate-limit/quota domain.\n"
        ),
    )

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(issue_number, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    # The is_completed lane was taken: salvage drift is emitted.
    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == issue_number

    # The throttle must NOT fire despite the "usage limit" marker in the log --
    # session_completed=True skipped log-tail classification entirely.
    throttle = [d for d in drift if d.kind == "provider_throttle_detected"]
    assert not throttle, (
        f"completed {adapter_kind} session was reclassified from log tail despite "
        f"session_completed=True (issue #656 regression): {throttle}"
    )


def test_detect_drift_completed_unpublished_work_salvaged(tmp_path: Path) -> None:
    """Issue #252: dead session with clean, ahead worktree emits salvage drift."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 252)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 252, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(252, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == 252
    assert salvage[0].branch == branch
    assert salvage[0].base_branch == "main"
    assert salvage[0].remove_labels == (config.labels.in_progress,)
    assert salvage[0].add_labels == (config.labels.pr_open,)

    # No relabel-to-ready drift should be emitted
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert not relabel


def test_detect_drift_dirty_worktree_with_commits_salvaged(tmp_path: Path) -> None:
    """Issue #1130: dead session with a dirty worktree that has commits ahead
    of base emits salvage drift, not relabel-to-ready. The committed work is
    salvageable regardless of working-tree dirt (shim/scaffolding artifacts)."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 253, dirty=True)

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 253, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(253, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert len(salvage) == 1
    assert salvage[0].issue_number == 253
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert not relabel


def test_detect_drift_no_commits_relabels(tmp_path: Path) -> None:
    """Issue #252: dead session with no commits still relabels to ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    branch = "agent/issue-254"
    info = create_worktree(repo_root, branch, base_ref="origin/main")
    worktree_path = info.path

    sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
    _write_dead_session_sidecar(sessions_dir, 254, branch, worktree_path)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(254, [config.labels.in_progress])],
        repo_root=repo_root,
    )
    state = empty_state()

    drift = detect_drift(gh, state, config, repo_root=repo_root)

    salvage = [d for d in drift if d.kind == "session_unpublished_work_salvaged"]
    assert not salvage
    relabel = [d for d in drift if d.kind == "session_failed_relabeled"]
    assert len(relabel) == 1
    assert relabel[0].issue_number == 254


def test_apply_fixes_salvage_success_creates_pr_and_labels(tmp_path: Path) -> None:
    """Issue #252: apply_fixes pushes, creates a PR, and moves labels to pr_open."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 255)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(255, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=101,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=255,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    new_state = apply_fixes(gh, empty_state(), drift, config)

    # PR created
    assert len(gh.prs_created) == 1
    assert gh.prs_created[0]["head"] == branch
    assert gh.prs_created[0]["base"] == "main"

    # Branch pushed to remote
    remote_refs = _git(remote, "show-ref")
    assert "agent/issue-255" in remote_refs.stdout

    # Labels moved
    assert (255, config.labels.in_progress) in gh.labels_removed
    assert (255, config.labels.pr_open) in gh.labels_added

    # Event recorded as salvage
    events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert events[-1]["payload"]["kind"] == "session_unpublished_work_salvaged"


def test_apply_fixes_salvage_push_failure_fallback(tmp_path: Path) -> None:
    """Issue #252: a failed salvage push falls back to relabel-to-ready."""
    remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    worktree_path, branch = _setup_completed_worktree(repo_root, 256)

    config = OrchestratorConfig()
    gh = FakeGitHub(
        prs=[],
        issues=[_issue(256, [config.labels.in_progress])],
        repo_root=repo_root,
        pr_create_return=102,
    )

    drift = [
        DriftItem(
            kind="session_unpublished_work_salvaged",
            issue_number=256,
            pr_number=None,
            detail="salvage",
            fix_actions=("push", "pr_create"),
            remove_labels=(config.labels.in_progress,),
            add_labels=(config.labels.pr_open,),
            branch=branch,
            base_branch="main",
        )
    ]

    # Force push to fail
    import charlie_work.reconcile

    original_push_branch = charlie_work.reconcile.push_branch
    charlie_work.reconcile.push_branch = lambda repo, br, worktree_path=None: (
        False,
        "simulated push failure",
    )
    try:
        new_state = apply_fixes(gh, empty_state(), drift, config)
    finally:
        charlie_work.reconcile.push_branch = original_push_branch

    # No PR created, active label removed, ready label added
    assert not gh.prs_created
    assert (256, config.labels.in_progress) in gh.labels_removed
    assert (256, config.labels.ready) in gh.labels_added

    # Event recorded as failed relabel
    events = [e for e in new_state["events"] if e["kind"] == "reconcile"]
    assert events[-1]["payload"]["kind"] == "session_failed_relabeled"
    assert any("salvage_failed" in action for action in events[-1]["payload"]["fix_actions"])


def test_reconcile_dry_run_never_reaches_salvage_push_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1475: under ``dry_run=True`` the ``if fix and not dry_run and
    drift:`` gate in ``_reconcile_locked`` is the single enforcement point
    keeping ``apply_fixes`` -- and the ``push_branch`` call inside its
    ``session_unpublished_work_salvaged`` lane -- from issuing a real
    ``git push``.

    ``apply_fixes`` deliberately has no ``dry_run`` parameter (issue #1051
    chose caller-side single-point enforcement over dead-code threading), so
    the caller gate is the ONLY thing between a salvage drift item and a
    real push under dry-run. This test drives the real ``reconcile(fix=True)``
    entry point end-to-end with a live ``session_unpublished_work_salvaged``
    drift item, spying on ``push_branch`` itself -- NOT on
    ``apply_drift_fixes`` -- so that if the gate is ever dropped, the REAL
    ``apply_fixes`` runs its salvage lane and the spy records the push it was
    about to issue. Patching ``apply_drift_fixes`` instead would mask exactly
    the regression this test exists to catch.
    """
    # Stage the fixture outside tmp_path: under a nested agent worktree's
    # sandboxed TMPDIR, pytest's basetemp nests deep enough that the repo's
    # derived ``worktree add`` paths overflow git's internal $GIT_DIR buffer
    # (``fatal: '$GIT_DIR' too big``, ~260 chars) even with core.longpaths.
    # Mirrors tests/test_deescalation.py's ``_add_worktree_for_deescalation``
    # and tests/test_local_lane_worktree.py's ``repo`` fixture, which stage
    # under the shallow system temp dir for the same reason.
    local = os.environ.get("LOCALAPPDATA")
    staging_root = Path(local) / "Temp" if local else Path(tempfile.gettempdir())
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="cw-1475-", dir=staging_root))
    try:
        remote, repo_root = _init_bare_remote_and_clone(staging)
        issue_number = 1475
        worktree_path, branch = _setup_completed_worktree(repo_root, issue_number)

        sessions_dir = repo_root / ".var" / "charlie-work" / "dispatches" / "sessions"
        _write_dead_session_sidecar(sessions_dir, issue_number, branch, worktree_path)

        config = OrchestratorConfig()
        gh = FakeGitHub(
            prs=[],
            issues=[_issue(issue_number, [config.labels.in_progress])],
            repo_root=repo_root,
            pr_create_return=1475,
        )

        push_calls: list[tuple[object, ...]] = []

        def _spy_push_branch(*args: object, **kwargs: object) -> tuple[bool, None]:
            push_calls.append(args)
            return True, None

        # The leaf itself: reconcile.py's salvage lane calls the module-level
        # ``push_branch`` name with no dry_run threading.
        monkeypatch.setattr("charlie_work.reconcile.push_branch", _spy_push_branch)

        paths = runtime_paths(repo_root, config.runtime.state_dir)
        app = OrchestratorApp(repo_root, paths, config, gh, dry_run=True)
        result = app.reconcile(fix=True)

        # Drift was actually detected -- the gate engaged rather than the pass
        # silently no-opping.
        assert result.ok is True
        assert "dry-run" in result.message.lower()
        assert any(
            item["kind"] == "session_unpublished_work_salvaged" for item in result.data["drift"]
        ), "fixture must produce salvage drift for this test to be non-vacuous"

        # The dry-run gate held: no push, no PR, no label writes, nothing on
        # the remote.
        assert push_calls == [], (
            "push_branch was called under dry_run=True -- the `not dry_run` "
            "gate in _reconcile_locked has been lost (issue #1475): "
            f"{push_calls}"
        )
        assert gh.prs_created == []
        assert gh.labels_added == []
        assert gh.labels_removed == []
        assert branch not in _git(remote, "show-ref").stdout
    finally:
        shutil.rmtree(staging, ignore_errors=True)
