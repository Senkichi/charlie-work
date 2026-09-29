"""Issue #1820: a ``closed`` local issue still blocks when its worker branch never merged.

``LocalFileGitHub.are_issues_open`` is the single point that resolves blocker
state for the local-file backend. ``state: closed`` frontmatter alone is not
proof the blocker's work reached HEAD -- a human can close the file before (or
without ever) merging the ``{branch_prefix}-<n>-*`` branch. These tests run a
real git repo: the closed blocker must keep counting as open while a
resolvable worker-branch tip is not an ancestor of HEAD, and must fall back to
the plain frontmatter verdict when no such branch exists (never dispatched, or
merged and the ref cleaned up -- the default lane's own end state under
``auto_merge.delete_branch``).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from _local_gate_helpers import drive_merge_gate
from charlie_work.backlog_reachability import _get_open_blockers_for_issue
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.labels import transition
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_lane import (
    branch_head_sha,
    is_ancestor,
    worktree_for_branch,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, env=env)
    assert result.returncode == 0, (args, result.stderr)
    return result


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit on ``main``.

    ``core.longpaths`` because pytest's basetemp under this repo's
    ``.var/worker-tmp`` pushes ``.git/objects`` paths past the Windows
    MAX_PATH limit; the ``GIT_*`` scrub stops an ambient ``GIT_DIR`` from
    redirecting the commands at the *outer* repo (same reasons as
    ``test_local_issues_loop_gates._init_repo``).
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    # Repo-local identity, matching every other real-git fixture in the
    # suite: ``git merge -m`` creates a commit and needs a committer, and
    # a fresh CI runner has no global user.name/user.email to fall back on.
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "test")
    _git(repo_root, "commit", "--allow-empty", "-m", "chore: seed")


def _write_issue(
    issues_dir: Path,
    number: int,
    *,
    state: str,
    body: str = "Body.",
    labels: tuple[str, ...] = (),
) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    labels_yaml = "[" + ", ".join(labels) + "]"
    path = issues_dir / f"{number:03d}_issue.md"
    path.write_text(f"---\nstate: {state}\nlabels: {labels_yaml}\n---\n{body}\n", encoding="utf-8")
    return path


def _commit_on_branch(repo_root: Path, branch: str, filename: str) -> None:
    _git(repo_root, "switch", "-c", branch)
    (repo_root / filename).write_text("x = 1\n", encoding="utf-8")
    _git(repo_root, "add", filename)
    _git(repo_root, "commit", "-m", "work")
    _git(repo_root, "switch", "main")


@pytest.fixture
def local_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A git repo whose issue #1 is closed and issue #2 is ``Blocked by #1``."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    _write_issue(issues_dir, 1, state="closed", body="Add helper.")
    _write_issue(issues_dir, 2, state="open", body="Blocked by #1\n\nUse the helper.")
    return repo_root, issues_dir


def test_closed_blocker_with_unmerged_worker_branch_still_counts_as_open(
    local_repo: tuple[Path, Path],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The issue's reproduction: ``state: closed`` + unmerged ``agent/issue-1-*``
    branch must still count as an open blocker."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    # The anomaly is surfaced once per client instance, not once per call --
    # are_issues_open runs several times per pass (prefetch + per-issue).
    with caplog.at_level("WARNING", logger="charlie_work.local_issues"):
        assert gh.are_issues_open([1]) == {1}
        gh.are_issues_open([1])
        gh.are_issues_open([1])
    warnings = [r for r in caplog.records if "worker branch is not merged" in r.getMessage()]
    assert len(warnings) == 1

    declared, open_blockers = _get_open_blockers_for_issue(gh, gh.issue_view(2))
    assert declared == [1]
    assert open_blockers == [1]


def test_closed_blocker_merged_branch_counts_as_closed(
    local_repo: tuple[Path, Path],
) -> None:
    """A worker branch merged into HEAD satisfies the blocker."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    _declared, open_blockers = _get_open_blockers_for_issue(gh, gh.issue_view(2))
    assert open_blockers == []


def test_closed_blocker_merged_then_branch_deleted_counts_as_closed(
    local_repo: tuple[Path, Path],
) -> None:
    """The default local merge lane's own end state: merged, then
    ``git branch -D`` under ``auto_merge.delete_branch``. The ref is gone, so
    there is nothing left to check -- dependents must not wedge forever."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    _git(repo_root, "branch", "-D", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()


def test_closed_issue_never_dispatched_falls_back_to_frontmatter(
    local_repo: tuple[Path, Path],
) -> None:
    """Closed-as-not-applicable: no worker branch was ever recorded, so
    ``state: closed`` keeps its historical verdict."""
    repo_root, issues_dir = local_repo
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()
    _declared, open_blockers = _get_open_blockers_for_issue(gh, gh.issue_view(2))
    assert open_blockers == []


def test_park_comment_branch_off_convention_still_blocks(
    local_repo: tuple[Path, Path],
) -> None:
    """The park comment names the branch exactly, so a worker branch outside
    the ``{prefix}-<n>-*`` convention is still honored."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "custom/helper-branch", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    comment = repo_root / "comment.md"
    comment.write_text(
        "Work for this issue is committed on branch `custom/helper-branch`. "
        "This repo has no remote.",
        encoding="utf-8",
    )
    gh.issue_comment(1, comment)

    assert gh.are_issues_open([1]) == {1}


def test_park_comment_branch_deleted_after_merge_counts_as_closed(
    local_repo: tuple[Path, Path],
) -> None:
    """A recorded branch name whose ref no longer resolves falls back to the
    frontmatter verdict -- merged-and-deleted cannot be told apart from
    deleted-unmerged at the ref level, and the former is the default lane's
    normal end state."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-1-add-helper", "helper.py")
    _git(repo_root, "merge", "--no-ff", "-m", "merge", "agent/issue-1-add-helper")
    _git(repo_root, "branch", "-D", "agent/issue-1-add-helper")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    comment = repo_root / "comment.md"
    comment.write_text(
        "Work for this issue is committed on branch `agent/issue-1-add-helper`. "
        "This repo has no remote.",
        encoding="utf-8",
    )
    gh.issue_comment(1, comment)

    assert gh.are_issues_open([1]) == set()


def test_configured_branch_prefix_drives_the_convention_scan(tmp_path: Path) -> None:
    """``github_client_for`` passes ``dispatch.branch_prefix`` through; a
    non-default prefix must scope the ``refs/heads/{prefix}-<n>-*`` scan."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    _write_issue(issues_dir, 1, state="closed")
    _commit_on_branch(repo_root, "bot/task-1-add-helper", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir, branch_prefix="bot/task")

    assert gh.are_issues_open([1]) == {1}
    # A client still on the default prefix sees no worker branch for #1.
    default_gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    assert default_gh.are_issues_open([1]) == set()


def test_sibling_issue_number_is_not_a_worker_branch(
    local_repo: tuple[Path, Path],
) -> None:
    """``agent/issue-12-x`` must not count as issue #1's worker branch."""
    repo_root, issues_dir = local_repo
    _commit_on_branch(repo_root, "agent/issue-12-other", "helper.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()


def test_open_issue_is_unaffected_by_the_merge_check(local_repo: tuple[Path, Path]) -> None:
    """An open issue reports open regardless of its branch state -- the extra
    check only re-arms *closed* issues."""
    repo_root, issues_dir = local_repo
    _write_issue(issues_dir, 3, state="open")
    _commit_on_branch(repo_root, "agent/issue-3-wip", "wip.py")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([3]) == {3}


def test_branch_pointing_at_base_counts_as_landed(local_repo: tuple[Path, Path]) -> None:
    """A worker branch whose tip adds nothing over its fork point is already
    contained in HEAD -- nothing to block on."""
    repo_root, issues_dir = local_repo
    _git(repo_root, "branch", "agent/issue-1-empty")
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    assert gh.are_issues_open([1]) == set()


# ---------------------------------------------------------------------------
# Issue #1972: local merge-gate rework must be capped.
#
# Before the fix, every conflict or suite failure in ``_local_merge_approved``
# routed the same record back to ``rework_requested`` via
# ``_local_route_merge_rework`` with no attempt bound -- a worker launch on
# every pass, forever (the uncapped cost spiral the issue describes). The fix
# persists one counter per failure kind on the local PR record
# (``local_merge_conflict_rework_attempts`` /
# ``local_suite_failed_rework_attempts``), routes to rework below the cap, and
# escalates with the distinct ``local_merge_rework_cap_exceeded`` reason +
# ``local_merge_rework_escalated`` event on the cap-th attempt.
#
# These tests drive real failing merge passes against a real git repo -- the
# AC asks for "N failing merge attempts" end to end, not unit-seam stubs.
# ---------------------------------------------------------------------------


@pytest.fixture
def lane_repo() -> Path:
    """A real git repo under the system temp dir, not ``tmp_path``.

    The merge gate attaches branch worktrees (``ensure_branch_worktree``), and
    ``git worktree add`` derives ``.git/worktrees/<name>`` paths plus a
    ``gitdir:`` back-pointer that overflow git's internal ``$GIT_DIR`` buffer
    at ``tmp_path`` depth under this repo's own worktree -- ``fatal: '$GIT_DIR'
    too big`` (same reason ``test_local_lane.repo`` uses ``tempfile.mkdtemp``).
    The conftest's ``_isolate_git_env`` whitelists the system temp dir.
    """
    return Path(tempfile.mkdtemp(prefix="cw-merge-gate-"))


def _commit_file(repo_root: Path, relpath: str, content: str, message: str) -> str:
    path = repo_root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo_root, "add", relpath)
    _git(repo_root, "commit", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def _make_branch(repo_root: Path, branch: str, relpath: str, content: str) -> str:
    """Branch off the current HEAD with one committed file; return its sha."""
    _git(repo_root, "checkout", "-b", branch)
    head = _commit_file(repo_root, relpath, content, f"feat: {relpath}")
    _git(repo_root, "checkout", "main")
    return head


def _lane_config(repo_root: Path, issues_dir: Path, **overrides: object) -> OrchestratorConfig:
    """Config through ``build_config_from_data`` so the local re-defaults apply."""
    data: dict = {
        # issues_dir is validated repo-root-relative -- always placed at
        # ``<repo>/docs/issues`` here.
        "local_issues": {"enabled": True, "issues_dir": "docs/issues"},
        # A suite that always passes in a nearly-empty scratch repo (``pytest``
        # would exit 5 -- "no tests collected" -- and defeat the gate for the
        # wrong reason); the suite-failure test overrides it.
        "dispatch": {"test_command": 'python -c "pass"'},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return build_config_from_data(data)


def _lane_app(
    repo_root: Path,
    issues_dir: Path,
    config: OrchestratorConfig | None = None,
) -> OrchestratorApp:
    cfg = config or _lane_config(repo_root, issues_dir)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    paths = runtime_paths(repo_root, cfg.runtime.state_dir)
    return OrchestratorApp(repo_root, paths, cfg, gh)


def _parked_lane_issue(
    app: OrchestratorApp, issues_dir: Path, issue_number: int, branch: str
) -> None:
    """The state ``park_unpublishable_work`` leaves: review-ready edge + entry."""
    labels = app.config.labels
    _write_issue(issues_dir, issue_number, state="open", labels=(labels.ready, labels.in_progress))
    transition(app.gh, labels, issue_number, "local_work_ready")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state.setdefault("issues", {})[str(issue_number)] = {
            "number": issue_number,
            "title": "Test issue",
            "status": "dispatched",
            "branch_name": branch,
        }
        save_state(app.paths.state_file, state)


def _adopt_and_approve(
    app: OrchestratorApp, issues_dir: Path, branch: str, head: str, issue_number: int = 7
) -> None:
    """Park the issue, adopt its branch into a lane record, approve the head."""
    _parked_lane_issue(app, issues_dir, issue_number, branch)
    app._local_review_packets()
    result = app.record_local_review(
        issue_number,
        "approved",
        reviewed_head=head,
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok, result.message


def _land_rework_and_reapprove(
    repo_root: Path,
    app: OrchestratorApp,
    issues_dir: Path,
    branch: str,
    relpath: str,
    content: str,
    issue_number: int = 7,
) -> str:
    """One rework cycle: commit on the branch worktree (as a rework worker
    would), rebuild the review packet, re-approve the new head. Returns the
    new head sha."""
    worktree = worktree_for_branch(repo_root, branch)
    assert worktree is not None, f"no worktree registered for {branch!r}"
    head = _commit_file(worktree, relpath, content, "rework")
    app._local_review_packets()
    result = app.record_local_review(
        issue_number,
        "approved",
        reviewed_head=head,
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok, result.message
    return head


class TestLocalMergeReworkCaps:
    """Issue #1972: repeated merge-gate failures escalate at the cap instead of
    routing back to rework forever."""

    def test_merge_conflict_routes_to_rework_below_cap_then_escalates(
        self, lane_repo: Path
    ) -> None:
        """cap = ``review.max_conflict_rework_attempts`` (2): attempt 1 routes
        to rework, the cap-th (2nd) conflict escalates."""
        _init_repo(lane_repo)
        issues_dir = lane_repo / "docs" / "issues"
        _commit_file(lane_repo, "shared.py", "v = 1\n", "seed shared")
        head = _make_branch(lane_repo, "agent/issue-7-x", "shared.py", "v = 2\n")
        app = _lane_app(lane_repo, issues_dir)
        _adopt_and_approve(app, issues_dir, "agent/issue-7-x", head)
        # Base moves with a conflicting edit AFTER approval.
        _commit_file(lane_repo, "shared.py", "v = 3\n", "conflicting base edit")

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "conflict"
        assert results[0]["routed_to"] == "rework"
        state = load_state_locked(app.paths.state_file)
        pr_entry = state["prs"]["7"]
        assert pr_entry["local_merge_conflict_rework_attempts"] == 1
        assert pr_entry["status"] == "rework_requested"
        assert state["issues"]["7"]["status"] == "rework_requested"

        # A rework cycle completes but leaves the conflict unresolved; the
        # packet re-mint (which resets the *remote* conflict counters) must
        # not touch the local counter.
        _land_rework_and_reapprove(
            lane_repo, app, issues_dir, "agent/issue-7-x", "rework.py", "r = 1\n"
        )
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["local_merge_conflict_rework_attempts"] == 1
        assert state["prs"]["7"]["status"] == "approved"

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "conflict"
        assert results[0]["routed_to"] == "escalated"
        state = load_state_locked(app.paths.state_file)
        pr_entry = state["prs"]["7"]
        assert pr_entry["status"] == "escalated"
        assert pr_entry["escalation_reason"] == "local_merge_rework_cap_exceeded"
        assert pr_entry["local_merge_conflict_rework_attempts"] == 2
        issue_entry = state["issues"]["7"]
        assert issue_entry["status"] == "escalated"
        assert issue_entry["escalation_reason"] == "local_merge_rework_cap_exceeded"
        assert issue_entry["reason_class"] == "mechanical"
        kinds = [e["kind"] for e in state["events"]]
        assert "local_merge_rework_escalated" in kinds
        # Mechanical escalation lands on the operator queue, not human-needed.
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.operator_queue in labels
        # The base never advanced and the lane now drops the record entirely.
        assert not is_ancestor(lane_repo, head, branch_head_sha(lane_repo, "main"))
        assert app._local_merge_approved() == []

    def test_suite_failure_routes_to_rework_below_cap_then_escalates(
        self, lane_repo: Path
    ) -> None:
        """The suite-failure lane reuses ``review.max_rework_cycles`` (2): the
        2nd failed suite run escalates with the same distinct reason."""
        _init_repo(lane_repo)
        issues_dir = lane_repo / "docs" / "issues"
        # The branch adds a failing test; the suite runs inside the branch
        # worktree, so pytest collects test_bad.py there.
        head = _make_branch(
            lane_repo, "agent/issue-7-x", "test_bad.py", "def test_x(): assert False\n"
        )
        config = _lane_config(lane_repo, issues_dir, dispatch={"test_command": "python -m pytest"})
        app = _lane_app(lane_repo, issues_dir, config=config)
        _adopt_and_approve(app, issues_dir, "agent/issue-7-x", head)

        # Async gate (issue #1974): the suite runs detached -- launch pass,
        # wait for the result file, then the resolving pass routes to rework.
        results = drive_merge_gate(app, 7)

        assert results[0]["outcome"] == "suite_failed"
        assert results[0]["routed_to"] == "rework"
        state = load_state_locked(app.paths.state_file)
        pr_entry = state["prs"]["7"]
        assert pr_entry["local_suite_failed_rework_attempts"] == 1
        assert pr_entry["status"] == "rework_requested"
        kinds = [e["kind"] for e in state["events"]]
        assert "local_suite_failed" in kinds
        # The suite counter is a separate field -- a suite failure must not
        # spend the conflict lane's budget.
        assert pr_entry.get("local_merge_conflict_rework_attempts", 0) == 0

        # Rework lands a commit but does not fix the failing test.
        _land_rework_and_reapprove(
            lane_repo, app, issues_dir, "agent/issue-7-x", "rework.py", "r = 1\n"
        )
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["local_suite_failed_rework_attempts"] == 1

        results = drive_merge_gate(app, 7)

        assert results[0]["outcome"] == "suite_failed"
        assert results[0]["routed_to"] == "escalated"
        state = load_state_locked(app.paths.state_file)
        pr_entry = state["prs"]["7"]
        assert pr_entry["status"] == "escalated"
        assert pr_entry["escalation_reason"] == "local_merge_rework_cap_exceeded"
        assert pr_entry["local_suite_failed_rework_attempts"] == 2
        issue_entry = state["issues"]["7"]
        assert issue_entry["status"] == "escalated"
        assert issue_entry["escalation_reason"] == "local_merge_rework_cap_exceeded"
        kinds = [e["kind"] for e in state["events"]]
        assert "local_merge_rework_escalated" in kinds
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.operator_queue in labels

    def test_counters_reset_on_successful_merge(self, lane_repo: Path) -> None:
        """A rework that actually resolves the conflict merges next pass; the
        counters restart at zero on the merged write."""
        _init_repo(lane_repo)
        issues_dir = lane_repo / "docs" / "issues"
        _commit_file(lane_repo, "shared.py", "v = 1\n", "seed shared")
        head = _make_branch(lane_repo, "agent/issue-7-x", "shared.py", "v = 2\n")
        app = _lane_app(lane_repo, issues_dir)
        _adopt_and_approve(app, issues_dir, "agent/issue-7-x", head)
        _commit_file(lane_repo, "shared.py", "v = 3\n", "conflicting base edit")

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "conflict"
        assert results[0]["routed_to"] == "rework"
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["local_merge_conflict_rework_attempts"] == 1

        # The rework converges shared.py on the base's content -- identical
        # changes merge cleanly -- and the packet re-mint keeps the counter.
        worktree = worktree_for_branch(lane_repo, "agent/issue-7-x")
        assert worktree is not None
        head = _commit_file(worktree, "shared.py", "v = 3\n", "rework: resolve the conflict")
        app._local_review_packets()
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["local_merge_conflict_rework_attempts"] == 1
        result = app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )
        assert result.ok, result.message

        results = drive_merge_gate(app, 7)

        assert results[0]["outcome"] in ("merged", "already_merged")
        state = load_state_locked(app.paths.state_file)
        pr_entry = state["prs"]["7"]
        assert pr_entry["status"] == "merged"
        assert pr_entry["local_merge_conflict_rework_attempts"] == 0
        assert pr_entry["local_suite_failed_rework_attempts"] == 0
