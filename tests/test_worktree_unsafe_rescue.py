"""``WorktreeUnsafeError`` classification, the live-writer deferral and rescue capture."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from _worktree_fixtures import _clone_repo, _git, _init_repo
from _worktree_shared_fixtures import _force_capture_failure, _init_repo_with_remote
from charlie_work.config import OrchestratorConfig
from charlie_work.host.fakes import FakeProcessProbe
from charlie_work import worktree as worktree_module
from charlie_work.worktree import (
    RESCUE_REF_PREFIX,
    RescueCapture,
    WORKTREE_UNSAFE_KINDS,
    WORKTREE_UNSAFE_KIND_LOCAL_COMMITS,
    WORKTREE_UNSAFE_KIND_SHIM_DIRT,
    WorktreeInfo,
    WorktreeUnsafeError,
    LiveWorkerRedispatchError,
    _worktree_unsafe_kind_from_reason,
    create_worktree,
    read_worktree_marker,
    remove_worktree,
    write_worktree_marker,
)

# --- Issue #1141: worktree_unsafe sweep must not escalate a LIVE worker ---


def _seed_live_writer_worktree(
    repo_root: Path,
    branch_name: str,
    *,
    marker_pid: int,
    session_id: str,
    sessions_dir: Path,
) -> WorktreeInfo:
    """Create a worktree (branch NOT pushed to origin), make it dirty in a
    worktree_unsafe-triggering way, and plant a writer marker + sidecar.

    This is the issue #1141 scenario: a live worker's transient working state
    (uncommitted modifications) in a worktree whose branch has not been pushed
    yet (killed-before-push recovery path).
    """
    info = create_worktree(repo_root, branch_name, base_ref="origin/main")
    # Worker-authored uncommitted modification — the "unsafe" dirt.
    (info.path / "worker_wip.txt").write_text("work in progress\n", encoding="utf-8")
    # Writer marker: records the live worker's pid + session id.
    write_worktree_marker(info.path, marker_pid, session_id)
    # Sidecar: gives the marker check (and the liveness gate) a recorded
    # session with a start-time fingerprint to corroborate.
    sidecar = sessions_dir / "issue-1141.json"
    sidecar.write_text(
        json.dumps({"session_id": session_id, "pid": marker_pid, "process_start_time": 1.0}),
        encoding="utf-8",
    )
    return info


def test_worktree_unsafe_defers_when_writer_marker_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Issue #1141: a dirty worktree whose writer marker has a LIVE pid must
    defer (LiveWorkerRedispatchError), not escalate (WorktreeUnsafeError).
    Dirt in a live worker's tree is normal working state, not residue.
    """
    remote, repo = _init_repo_with_remote(tmp_path)

    branch_name = "agent/issue-1141-live-writer"
    sessions_dir = repo / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    marker_pid = 424242

    info = _seed_live_writer_worktree(
        repo,
        branch_name,
        marker_pid=marker_pid,
        session_id="session-1141-live",
        sessions_dir=sessions_dir,
    )

    # The recovery record carries a DIFFERENT, dead worker_pid so the recovery
    # liveness probe (_probe_recovery_liveness) passes — simulating the bug
    # scenario where the probe missed the live writer (stale/recycled pid in
    # the record). The marker is the ground-truth liveness signal.
    recovery = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
    }

    # Only the marker's pid is alive; the recovery record's pid is dead.
    fake_host(probe=FakeProcessProbe({marker_pid: None}))

    with pytest.raises(LiveWorkerRedispatchError) as exc_info:
        create_worktree(
            repo,
            branch_name,
            base_ref="origin/main",
            recovery=recovery,
            sessions_dir=sessions_dir,
            config=OrchestratorConfig(),
            issue_number=1141,
        )

    assert exc_info.value.pid == marker_pid
    assert exc_info.value.probe_result == "live_writer_at_unsafe_evaluation"
    # No reap: the worktree and its dirty content must survive untouched.
    assert info.path.exists()
    assert (info.path / "worker_wip.txt").read_text(encoding="utf-8") == "work in progress\n"
    # The writer marker must not have been cleaned by the deferred path.
    assert read_worktree_marker(info.path) is not None

    # Clean up.
    remove_worktree(repo, info.path, branch=branch_name)


def test_worktree_unsafe_still_escalates_when_writer_marker_is_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_host
) -> None:
    """Issue #1141 inverse: a dirty worktree whose writer marker has a DEAD
    pid must still escalate (WorktreeUnsafeError) — death is established, so
    the dirt check is meaningful.
    """
    remote, repo = _init_repo_with_remote(tmp_path)

    branch_name = "agent/issue-1141-dead-writer"
    sessions_dir = repo / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    marker_pid = 424242

    info = _seed_live_writer_worktree(
        repo,
        branch_name,
        marker_pid=marker_pid,
        session_id="session-1141-dead",
        sessions_dir=sessions_dir,
    )

    recovery = {
        "branch_name": branch_name,
        "status": "dispatched",
        "worker_pid": 999999,
        "worker_process_start_time": 0.0,
    }

    # Every pid is dead — the marker is stale, the recovery record's pid is
    # stale. The marker check at entry cleans the stale marker; the dirt check
    # then escalates normally.
    fake_host(probe=FakeProcessProbe())
    _force_capture_failure(monkeypatch)

    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(
            repo,
            branch_name,
            base_ref="origin/main",
            recovery=recovery,
            sessions_dir=sessions_dir,
            config=OrchestratorConfig(),
            issue_number=1141,
        )

    # No reap occurred (WorktreeUnsafeError refuses the reset); the dirty
    # content survives.
    assert info.path.exists()
    assert (info.path / "worker_wip.txt").read_text(encoding="utf-8") == "work in progress\n"

    # Clean up.
    remove_worktree(repo, info.path, branch=branch_name)


# --- Issue #807: worktree_unsafe kind split at detection time ---


def test_worktree_unsafe_error_classifies_shim_dirt_from_uncommitted_modifications() -> None:
    """Issue #807: a ``WorktreeUnsafeError`` raised for uncommitted modifications
    (launch-shim dirt / adapter shim materialization) must carry
    ``kind="worktree_unsafe_shim_dirt"`` — the mechanical kind that stays in
    ``DETERMINISTIC_ESCALATION_FAILURE_KINDS`` and is eligible for auto-de-escalation.
    """
    exc = WorktreeUnsafeError("worktree has uncommitted modifications")
    assert exc.kind == "worktree_unsafe_shim_dirt"


def test_worktree_unsafe_error_classifies_local_commits_from_unpushed_commits() -> None:
    """Issue #807: a ``WorktreeUnsafeError`` raised for local commits not on the
    remote branch (genuine divergence) must carry
    ``kind="worktree_unsafe_local_commits"`` — the judgment kind that escalates
    immediately but is NOT eligible for auto-de-escalation.
    """
    exc = WorktreeUnsafeError("worktree has 1 local commit(s) not on remote branch")
    assert exc.kind == "worktree_unsafe_local_commits"


def test_worktree_unsafe_error_explicit_kind_overrides_reason_derivation() -> None:
    """Issue #807: an explicit ``kind`` argument takes precedence over the
    auto-derived kind from the reason string, so callers that already know the
    discriminator can pass it directly."""
    exc = WorktreeUnsafeError(
        "worktree has uncommitted modifications",
        kind="worktree_unsafe_local_commits",
    )
    assert exc.kind == "worktree_unsafe_local_commits"


def test_worktree_unsafe_error_rejects_unrecognized_explicit_kind() -> None:
    """Issue #807: an explicit ``kind`` that isn't one of ``WORKTREE_UNSAFE_KINDS``
    must fail closed with ``ValueError`` rather than being accepted silently. A
    stray/unrecognized kind would match neither
    ``DETERMINISTIC_ESCALATION_FAILURE_KINDS`` nor
    ``DETERMINISTIC_JUDGMENT_ESCALATION_FAILURE_KINDS``, so it would silently
    degrade to the ordinary redispatch-cap path instead of escalating.
    """
    with pytest.raises(ValueError):
        WorktreeUnsafeError(
            "worktree has uncommitted modifications",
            kind="worktree_unsafe_bogus",
        )


def test_worktree_unsafe_error_derived_kind_is_always_a_known_kind() -> None:
    """Issue #807: a reason string that matches neither known discriminator
    pattern must still derive a kind from ``WORKTREE_UNSAFE_KINDS`` (the
    fail-closed fallback), never an unrecognized or empty kind."""
    exc = WorktreeUnsafeError("worktree is in an unexpected state")
    assert exc.kind in WORKTREE_UNSAFE_KINDS


def test_worktree_unsafe_kind_fallback_is_local_commits_not_shim_dirt() -> None:
    """Issue #807 rework: a reason string that matches neither known pattern
    must fail closed toward ``WORKTREE_UNSAFE_KIND_LOCAL_COMMITS`` (judgment /
    human-needed), NOT ``WORKTREE_UNSAFE_KIND_SHIM_DIRT`` (mechanical /
    auto-clearable). Defaulting an unrecognized reason to the mechanical kind
    inverts this codebase's fail-closed convention and could silently reproduce
    the #807 bug for any future reason string that doesn't match the
    'local commit' or 'uncommitted modifications' substring checks — the
    de-escalation sweep would auto-clear it and return the issue to dispatch
    instead of escalating to a human.
    """
    # Direct function-level assertion on the fallback path.
    assert (
        _worktree_unsafe_kind_from_reason("worktree is in an unexpected state")
        == WORKTREE_UNSAFE_KIND_LOCAL_COMMITS
    )
    # End-to-end through WorktreeUnsafeError construction.
    exc = WorktreeUnsafeError("worktree is in an unexpected state")
    assert exc.kind == WORKTREE_UNSAFE_KIND_LOCAL_COMMITS


def test_worktree_unsafe_kind_shim_dirt_reason_classifies_as_shim_dirt() -> None:
    """Issue #807 rework: the 'uncommitted modifications' reason string (shim /
    adapter dirt) must still classify as ``WORKTREE_UNSAFE_KIND_SHIM_DIRT``
    after the fallback was flipped to ``WORKTREE_UNSAFE_KIND_LOCAL_COMMITS``.
    This guards against the fallback-flip silently misclassifying shim dirt as
    a judgment call, which would make every shim-dirt escalation
    non-auto-clearable."""
    assert (
        _worktree_unsafe_kind_from_reason("worktree has uncommitted modifications")
        == WORKTREE_UNSAFE_KIND_SHIM_DIRT
    )
    exc = WorktreeUnsafeError("worktree has uncommitted modifications")
    assert exc.kind == WORKTREE_UNSAFE_KIND_SHIM_DIRT


# --- Issue #849: rescue capture makes worktree_unsafe refusal recoverable ---


def test_rescue_capture_preserves_dirty_worktree_and_permits_reset(
    tmp_path: Path,
) -> None:
    """AC1: a dirty worktree is captured to a rescue ref and create_worktree
    then succeeds. The captured ref's content matches the pre-reset working
    tree byte-for-byte, including untracked files."""
    remote, repo = _init_repo_with_remote(tmp_path)

    branch_name = "agent/issue-849-rescue-ac1"
    info1 = create_worktree(repo, branch_name, base_ref="origin/main")

    # Worker-authored dirty content: a tracked modification + an untracked file.
    (info1.path / "README.md").write_text("modified by worker\n", encoding="utf-8")
    (info1.path / "worker_new_file.txt").write_text(
        "new untracked worker content\n", encoding="utf-8"
    )

    pre_reset_readme = (info1.path / "README.md").read_text(encoding="utf-8")
    pre_reset_new_file = (info1.path / "worker_new_file.txt").read_text(encoding="utf-8")

    # Re-dispatch: the stale dirty worktree triggers _raise_if_unsafe_to_reset,
    # which captures the work to a rescue ref and then permits the reset.
    info2 = create_worktree(repo, branch_name, base_ref="origin/main", issue_number=849)

    # create_worktree succeeded (no exception) and rescue_capture is populated.
    assert info2.rescue_capture is not None
    assert info2.rescue_capture.ref_name is not None
    assert info2.rescue_capture.error is None
    assert info2.rescue_capture.ref_name.startswith(RESCUE_REF_PREFIX)

    rescue_ref = info2.rescue_capture.ref_name

    # The captured ref's tree matches the pre-reset working tree byte-for-byte.
    rescued_readme = _git(repo, "show", f"{rescue_ref}:README.md").stdout
    assert rescued_readme == pre_reset_readme

    rescued_new_file = _git(repo, "show", f"{rescue_ref}:worker_new_file.txt").stdout
    assert rescued_new_file == pre_reset_new_file

    # Clean up.
    remove_worktree(repo, info2.path, branch=branch_name)


def test_rescue_capture_failure_still_raises_worktree_unsafe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: when capture fails, WorktreeUnsafeError is still raised and no
    reset occurs. Capture failure must never downgrade the safety property."""
    remote, repo = _init_repo_with_remote(tmp_path)

    branch_name = "agent/issue-849-rescue-ac2"
    info1 = create_worktree(repo, branch_name, base_ref="origin/main")

    # Worker-authored dirty content.
    (info1.path / "worker_wip.txt").write_text("work in progress\n", encoding="utf-8")

    # Force capture to fail.
    monkeypatch.setattr(
        "charlie_work.worktree._capture_worktree_work_to_rescue_ref",
        lambda *args, **kwargs: RescueCapture(
            ref_name=None, commit_sha=None, error="forced capture failure"
        ),
    )

    with pytest.raises(WorktreeUnsafeError, match="worktree has uncommitted source work"):
        create_worktree(repo, branch_name, base_ref="origin/main", issue_number=849)

    # No reset occurred — the dirty content survives untouched.
    assert info1.path.exists()
    assert (info1.path / "worker_wip.txt").read_text(encoding="utf-8") == "work in progress\n"

    # Clean up.
    remove_worktree(repo, info1.path, branch=branch_name)


def test_rescue_capture_excludes_scaffolding(
    tmp_path: Path,
) -> None:
    """AC3: injected_paths / materialize_dirs scaffolding is excluded from the
    captured content, consistent with _worker_authored_dirty."""
    remote, repo = _init_repo_with_remote(tmp_path)

    # Config with an injected path (e.g. a prompt file the orchestrator writes).
    from charlie_work.config import DispatchConfig, OrchestratorConfig

    config = OrchestratorConfig(dispatch=DispatchConfig(injected_paths=(".charlie-writer.json",)))

    branch_name = "agent/issue-849-rescue-ac3"
    info1 = create_worktree(repo, branch_name, base_ref="origin/main", config=config)

    # Worker-authored file (should be captured).
    (info1.path / "worker_authored.txt").write_text("worker content\n", encoding="utf-8")
    # Scaffolding file (should be excluded — HEAD version in the rescue tree).
    (info1.path / ".charlie-writer.json").write_text('{"injected": true}\n', encoding="utf-8")

    info2 = create_worktree(
        repo, branch_name, base_ref="origin/main", issue_number=849, config=config
    )

    assert info2.rescue_capture is not None
    assert info2.rescue_capture.ref_name is not None
    rescue_ref = info2.rescue_capture.ref_name

    # Worker-authored file IS in the rescue ref's tree.
    rescued_worker = _git(repo, "show", f"{rescue_ref}:worker_authored.txt").stdout
    assert rescued_worker == "worker content\n"

    # Scaffolding file is NOT in the rescue ref's tree with the injected
    # content — it has HEAD's version (which is absent, since the file was
    # untracked). The rescue tree should not contain .charlie-writer.json at all.
    show_result = subprocess.run(
        ["git", "show", f"{rescue_ref}:.charlie-writer.json"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert show_result.returncode != 0  # file does not exist in the rescue tree

    # Clean up.
    remove_worktree(repo, info2.path, branch=branch_name)


def test_rescue_capture_excludes_launcher_owned_shim_dirt(
    tmp_path: Path,
) -> None:
    """Issue #1391: launcher-owned shim dirt (``.devin/``, ``.git_worktree_dir/``,
    and a ``PR_BODY*.md`` scratch file) mixed with real tracked worker edits is
    excluded from the captured rescue tree, while the real worker edits are
    preserved.

    Calls ``_capture_worktree_work_to_rescue_ref`` directly so the exclusion
    pathspecs in the capture function are the only thing under test -- the
    ``_worker_authored_dirty`` matcher that also ignores launcher-owned dirt
    is not on this code path, so a regression in the capture exclusions cannot
    be masked by the dirty check refusing to trigger.
    """
    remote, repo = _init_repo_with_remote(tmp_path)

    branch_name = "agent/issue-1391-rescue-shim-dirt"
    info = create_worktree(repo, branch_name, base_ref="origin/main")

    # Real worker edits: a tracked modification + an untracked worker file.
    # These MUST appear in the rescue tree.
    worker_readme = "modified by worker\n"
    (info.path / "README.md").write_text(worker_readme, encoding="utf-8")
    worker_new = "real worker edit\n"
    (info.path / "worker_real_edit.txt").write_text(worker_new, encoding="utf-8")

    # Launcher-owned shim dirt. These MUST NOT appear in the rescue tree.
    (info.path / ".devin").mkdir(exist_ok=True)
    (info.path / ".devin" / "config.json").write_text('{"shim": true}\n', encoding="utf-8")
    (info.path / ".git_worktree_dir").mkdir(exist_ok=True)
    (info.path / ".git_worktree_dir" / "marker").write_text("shim marker\n", encoding="utf-8")
    # A PR_BODY*.md variant -- the glob exclusion must catch it.
    pr_body = "draft PR body\n"
    (info.path / "PR_BODY_1391.md").write_text(pr_body, encoding="utf-8")

    capture = worktree_module._capture_worktree_work_to_rescue_ref(
        repo, info.path, issue_number=1391
    )

    assert capture.error is None, f"capture failed: {capture.error}"
    assert capture.ref_name is not None
    assert capture.ref_name.startswith(RESCUE_REF_PREFIX)
    rescue_ref = capture.ref_name

    # Real worker edits ARE present in the rescue tree, byte-for-byte.
    assert _git(repo, "show", f"{rescue_ref}:README.md").stdout == worker_readme
    assert _git(repo, "show", f"{rescue_ref}:worker_real_edit.txt").stdout == worker_new

    # Launcher-owned shim paths are EXCLUDED from the rescue tree: ``git show``
    # for each shim path fails (non-zero exit) because the path does not exist
    # in the captured tree.
    for shim_path in (
        ".devin/config.json",
        ".git_worktree_dir/marker",
        "PR_BODY_1391.md",
    ):
        show_result = subprocess.run(
            ["git", "show", f"{rescue_ref}:{shim_path}"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        assert show_result.returncode != 0, (
            f"shim path {shim_path!r} leaked into the rescue tree; "
            f"git show returned {show_result.returncode} with stdout="
            f"{show_result.stdout!r}"
        )

    # Sanity: no path under a launcher-owned directory or matching the PR body
    # family appears anywhere in the captured tree.
    tree_paths = _git(repo, "ls-tree", "-r", "--name-only", rescue_ref).stdout.splitlines()
    leaked = [
        p
        for p in tree_paths
        if p.startswith(".devin/")
        or p.startswith(".git_worktree_dir/")
        or p.startswith("PR_BODY")
        or p in (".worker-pr-body.md", "_pr_body.md")
    ]
    assert leaked == [], f"launcher-owned shim paths leaked into rescue tree: {leaked}"

    # Clean up.
    remove_worktree(repo, info.path, branch=branch_name)


def test_rescue_capture_emits_event_retrievable_via_query_events(
    tmp_path: Path,
) -> None:
    """AC4: the emitted event records the rescue ref and is retrievable via
    query_events."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.instrumentation import query_events
    from charlie_work.paths import runtime_paths

    remote, repo = _init_repo_with_remote(tmp_path)

    config = OrchestratorConfig()
    paths = runtime_paths(repo, config.runtime.state_dir)
    state_file = paths.state_file

    branch_name = "agent/issue-849-rescue-ac4"
    info1 = create_worktree(repo, branch_name, base_ref="origin/main")

    # Worker-authored dirty content.
    (info1.path / "worker_wip.txt").write_text("rescue me\n", encoding="utf-8")

    info2 = create_worktree(
        repo, branch_name, base_ref="origin/main", issue_number=849, config=config
    )

    assert info2.rescue_capture is not None
    assert info2.rescue_capture.ref_name is not None
    rescue_ref = info2.rescue_capture.ref_name

    # The event is retrievable via query_events.
    events = query_events(state_file, kind="worktree_rescue_captured")
    assert len(events) >= 1
    rescue_events = [e for e in events if e.get("payload", {}).get("rescue_ref") == rescue_ref]
    assert len(rescue_events) == 1
    payload = rescue_events[0]["payload"]
    assert payload["rescue_ref"] == rescue_ref
    assert payload["issue_number"] == 849
    assert payload["commit_sha"] is not None

    # Clean up.
    remove_worktree(repo, info2.path, branch=branch_name)


def test_rework_reuse_capture_cleans_worktree_before_ff(
    tmp_path: Path,
) -> None:
    """Rework finding: rework=True + dirty-worktree + capture-succeeds +
    ff-only-succeeds-despite-dirt.

    The reuse-in-place branch captures dirty content to a rescue ref, then
    must clean the working tree before the ff-only merge. Without the clean,
    a dirty file the ff doesn't touch silently survives into the new work
    session and can be committed under the next worker's name — the exact
    hazard the "never commit tracked modifications the shim did not itself
    produce" invariant exists to prevent.

    This test asserts on the resulting worktree's on-disk state, not just on
    the rescue ref's content.
    """
    remote_repo = tmp_path / "remote"
    _init_repo(remote_repo)
    repo = tmp_path / "repo"
    _clone_repo(remote_repo, repo)

    branch_name = "agent/issue-849-rework-reuse-clean"
    info1 = create_worktree(repo, branch_name, base_ref="origin/main")
    # Push the branch so origin knows about it (required for rework reuse).
    _git(repo, "push", "origin", branch_name)

    # Worker-authored dirty content:
    # - A tracked modification to README.md (exists in initial commit; the
    #   ff-only below adds file2.txt and does NOT touch README.md, so without
    #   the clean the modification would survive the ff).
    # - An untracked file (also not touched by the ff).
    dirty_readme = "modified by worker\n"
    (info1.path / "README.md").write_text(dirty_readme, encoding="utf-8")
    (info1.path / "worker_untracked.txt").write_text(
        "untracked worker content\n", encoding="utf-8"
    )

    # Advance the branch on origin so the local worktree is behind and can
    # fast-forward. The new commit adds file2.txt — it does NOT touch
    # README.md, so the ff-only would succeed even with a dirty README.md.
    _git(remote_repo, "checkout", branch_name)
    (remote_repo / "file2.txt").write_text("remote change\n", encoding="utf-8")
    _git(remote_repo, "add", "file2.txt")
    _git(remote_repo, "commit", "-m", "add file2 on remote")
    remote_tip = _git(remote_repo, "rev-parse", "HEAD").stdout.strip()
    _git(remote_repo, "checkout", "main")

    # Re-dispatch with rework=True: the existing worktree is dirty, so
    # _capture_or_raise fires. Capture succeeds → the working tree is
    # cleaned → the ff-only merge proceeds on a clean tree.
    info2 = create_worktree(repo, branch_name, rework=True, issue_number=849)

    # Rescue capture is populated.
    assert info2.rescue_capture is not None
    assert info2.rescue_capture.ref_name is not None
    assert info2.rescue_capture.error is None
    rescue_ref = info2.rescue_capture.ref_name

    # The worktree was reused in-place and fast-forwarded to the origin tip.
    assert info2.path == info1.path
    assert _git(info2.path, "rev-parse", "HEAD").stdout.strip() == remote_tip

    # ON-DISK STATE: the dirty content is GONE from the working tree.
    # README.md is back to HEAD's version ("hello\n"), not the worker's
    # modification. Without the fix, this would still be "modified by worker\n".
    assert (info2.path / "README.md").read_text(encoding="utf-8") == "hello\n"

    # The untracked worker file is gone. Without the fix, it would survive.
    assert not (info2.path / "worker_untracked.txt").exists()

    # The ff-only brought in file2.txt from the remote.
    assert (info2.path / "file2.txt").read_text(encoding="utf-8") == "remote change\n"

    # The working tree is clean (no worker-authored dirt survives).
    status = _git(info2.path, "status", "--porcelain")
    assert status.stdout.strip() == ""

    # The rescue ref preserved the dirty content byte-for-byte.
    rescued_readme = _git(repo, "show", f"{rescue_ref}:README.md").stdout
    assert rescued_readme == dirty_readme
    rescued_untracked = _git(repo, "show", f"{rescue_ref}:worker_untracked.txt").stdout
    assert rescued_untracked == "untracked worker content\n"

    # Clean up.
    remove_worktree(repo, info2.path, branch=branch_name)
