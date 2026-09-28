"""Tests for the local (no-remote) review/merge lane (issue #1844).

Two layers:

- ``charlie_work.local_lane`` -- the pure git/suite primitives, exercised
  against a real temporary git repository with NO origin remote.
- ``OrchestratorApp``'s local-lane delegates (``orchestration/local_lanes.py``)
  -- adoption, packet build, verdict recording, the merge gate, and rework
  routing, against the same real repo plus a real ``LocalFileGitHub``.

The contract under test: a finished worker branch parked at
``agent:review-ready`` is adopted into ``state["prs"]`` as a ``"local": True``
record, reviewed off ``git diff <base>...<branch>``, merged only after an
approved verdict + a green full-suite run inside the branch worktree, and
rejected/conflict/suite-failure outcomes route back through the existing
rework lane -- never to an operator gate.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from charlie_work.adapters import SessionDispatchResult, SessionRequest
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.labels import LabelConfig, transition
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_lane import (
    branch_diff,
    branch_head_sha,
    is_ancestor,
    is_local_pr_record,
    local_base_branch,
    local_pr_dict,
    local_pr_records,
    merge_branch_into_base,
    run_full_suite,
    suite_command_argv,
    synthesize_open_pr,
    worktree_for_branch,
)
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import detect_drift
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp


# ---------------------------------------------------------------------------
# Fixtures -- a real local git repo, no origin remote.
#
# Git repos live in ``tempfile.mkdtemp`` (the real system temp dir), not
# pytest's ``tmp_path``: ``tmp_path`` nests under this repo's own worktree
# (``.var/worker-tmp/...``) and the paths git derives for ``worktree add``
# (``<repo>/.git/worktrees/<name>`` and the target's ``gitdir:`` back-pointer)
# overflow git's internal ``$GIT_DIR`` buffer at that depth -- ``fatal:
# '$GIT_DIR' too big``. The conftest's ``_isolate_git_env`` already whitelists
# the system temp dir via ``GIT_CEILING_DIRECTORIES``.
# ---------------------------------------------------------------------------


@pytest.fixture
def repo() -> Path:
    root = Path(tempfile.mkdtemp(prefix="cw-lane-"))
    yield root


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo on ``main`` with one commit and NO origin remote."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.email", "test@example.test")
    _git(repo_root, "config", "user.name", "test")
    _git(repo_root, "commit", "--allow-empty", "-m", "chore: seed")


def _commit_file(repo_root: Path, relpath: str, content: str, message: str) -> str:
    path = repo_root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _git(repo_root, "add", relpath)
    _git(repo_root, "commit", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD").stdout.strip()


def _make_branch(repo_root: Path, branch: str, relpath: str, content: str) -> str:
    """Branch off main with one committed file; return the new head sha."""
    _git(repo_root, "checkout", "-b", branch)
    head = _commit_file(repo_root, relpath, content, f"feat: {relpath}")
    _git(repo_root, "checkout", "main")
    return head


def _write_issue(
    issues_dir: Path,
    number: int,
    *,
    slug: str = "issue",
    title: str = "Test issue",
    state: str = "open",
    labels: tuple[str, ...] = (),
) -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    labels_yaml = "[" + ", ".join(labels) + "]"
    path = issues_dir / f"{number:03d}_2026-09-17_{slug}.md"
    path.write_text(
        "---\n"
        f'title: "{title}"\n'
        f"state: {state}\n"
        f"labels: {labels_yaml}\n"
        'created: "2026-09-17"\n'
        'author: "test"\n'
        "---\n"
        "Body text.\n",
        encoding="utf-8",
    )
    return path


def _local_config(repo_root: Path, issues_dir: Path, **overrides) -> OrchestratorConfig:
    """Config through ``build_config_from_data`` so the local re-defaults apply."""
    data: dict = {
        # issues_dir is validated repo-root-relative -- the tests always place
        # it at ``<repo>/docs/issues``.
        "local_issues": {"enabled": True, "issues_dir": "docs/issues"},
        # A suite that always passes: the merge-gate tests need a suite that
        # succeeds in a nearly-empty scratch repo (``pytest`` would exit 5 --
        # "no tests collected" -- and defeat the gate for the wrong reason).
        # The suite-failure test overrides this with an explicit failure.
        # Bare "python" (not sys.executable): suite_command_argv shlex-splits
        # the command, which eats the backslashes in a Windows path.
        "dispatch": {"test_command": 'python -c "pass"'},
    }
    for section, values in overrides.items():
        data.setdefault(section, {}).update(values)
    return build_config_from_data(data)


def _app(
    repo_root: Path,
    issues_dir: Path,
    config: OrchestratorConfig | None = None,
) -> OrchestratorApp:
    cfg = config or _local_config(repo_root, issues_dir)
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    paths = runtime_paths(repo_root, cfg.runtime.state_dir)
    return OrchestratorApp(repo_root, paths, cfg, gh)


def _seed_issue_state(
    app: OrchestratorApp,
    issue_number: int,
    *,
    status: str = "dispatched",
    branch: str | None = None,
    title: str = "Test issue",
) -> None:
    """Write the issue-entry shape a parked worker leaves behind."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        issues = state.setdefault("issues", {})
        issues[str(issue_number)] = {
            "number": issue_number,
            "title": title,
            "status": status,
            **({"branch_name": branch} if branch else {}),
        }
        save_state(app.paths.state_file, state)


def _parked_issue(
    app: OrchestratorApp,
    issues_dir: Path,
    issue_number: int,
    branch: str,
) -> None:
    """Simulate ``park_unpublishable_work``: local_work_ready edge + state."""
    labels = app.config.labels
    _write_issue(issues_dir, issue_number, labels=(labels.ready, labels.in_progress))
    transition(app.gh, labels, issue_number, "local_work_ready")
    _seed_issue_state(app, issue_number, branch=branch)


# ---------------------------------------------------------------------------
# local_lane.py -- pure primitives
# ---------------------------------------------------------------------------


class TestLocalLanePrimitives:
    def test_branch_head_sha_and_base_branch(self, repo: Path) -> None:
        _init_repo(repo)
        head = _git(repo, "rev-parse", "HEAD").stdout.strip()

        assert branch_head_sha(repo, "main") == head
        assert branch_head_sha(repo, "does-not-exist") is None
        assert local_base_branch(repo) == "main"

    def test_branch_diff(self, repo: Path) -> None:
        _init_repo(repo)
        _make_branch(repo, "agent/issue-7-x", "feature.py", "x = 1\n")

        diff = branch_diff(repo, "main", "agent/issue-7-x")

        assert diff is not None
        assert "feature.py" in diff
        assert "+x = 1" in diff

    def test_branch_diff_missing_branch_returns_none(self, repo: Path) -> None:
        _init_repo(repo)
        assert branch_diff(repo, "main", "ghost-branch") is None

    def test_is_ancestor(self, repo: Path) -> None:
        _init_repo(repo)
        base = _git(repo, "rev-parse", "HEAD").stdout.strip()
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")

        assert is_ancestor(repo, base, head)
        assert not is_ancestor(repo, head, base)

    def test_suite_command_argv(self, repo: Path) -> None:
        argv = suite_command_argv("python -m pytest", repo)
        assert argv == ["python", "-m", "pytest", "-q", "--tb=short"]

    def test_run_full_suite_pass_and_fail(self, repo: Path) -> None:
        _init_repo(repo)
        ok = run_full_suite(repo, [sys.executable, "-c", "print('ok')"])
        assert ok.ok and ok.returncode == 0

        bad = run_full_suite(repo, [sys.executable, "-c", "import sys; sys.exit(3)"])
        assert not bad.ok and bad.returncode == 3


class TestMergeBranchIntoBase:
    def test_fast_forward(self, repo: Path) -> None:
        _init_repo(repo)
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")

        outcome = merge_branch_into_base(
            repo, "main", "agent/issue-7-x", worktrees_dir=repo / "wt"
        )

        assert outcome.status == "merged"
        assert outcome.fast_forward is True
        assert outcome.merged_sha == head
        assert branch_head_sha(repo, "main") == head

    def test_no_ff_on_diverged_base(self, repo: Path) -> None:
        _init_repo(repo)
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        # Base diverges: a non-conflicting commit on main.
        _commit_file(repo, "base.py", "b = 1\n", "chore: base moved")

        outcome = merge_branch_into_base(
            repo, "main", "agent/issue-7-x", worktrees_dir=repo / "wt"
        )

        assert outcome.status == "merged"
        assert outcome.fast_forward is False
        # Base now contains both changes via a real merge commit.
        assert is_ancestor(repo, head, branch_head_sha(repo, "main"))
        merge_wt = repo / "wt"
        assert not list(merge_wt.glob("*merge*")), "temporary merge worktree removed"

    def test_conflict_aborts_and_reports(self, repo: Path) -> None:
        _init_repo(repo)
        _commit_file(repo, "shared.py", "v = 1\n", "seed shared")
        _make_branch(repo, "agent/issue-7-x", "shared.py", "v = 2\n")
        _commit_file(repo, "shared.py", "v = 3\n", "conflicting base edit")

        outcome = merge_branch_into_base(
            repo, "main", "agent/issue-7-x", worktrees_dir=repo / "wt"
        )

        assert outcome.status == "conflict"
        assert "shared.py" in outcome.conflicted_paths
        # The merge was aborted -- no half-merged state on the base.
        status = _git(repo, "status", "--porcelain").stdout
        assert status == ""

    def test_already_merged(self, repo: Path) -> None:
        _init_repo(repo)
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        _git(repo, "merge", "--ff-only", "agent/issue-7-x")

        outcome = merge_branch_into_base(
            repo, "main", "agent/issue-7-x", worktrees_dir=repo / "wt"
        )

        assert outcome.status == "already"

    def test_missing_branch_is_error(self, repo: Path) -> None:
        _init_repo(repo)
        outcome = merge_branch_into_base(repo, "main", "ghost", worktrees_dir=repo / "wt")
        assert outcome.status == "error"

    def test_dirty_unrelated_base_checkout_still_merges(self, repo: Path) -> None:
        """Unrelated WIP/untracked files in the base checkout must not block.

        A local repo's main checkout legitimately carries worker scratch and
        operator WIP -- the lane would deadlock if any dirt deferred the merge.
        """
        _init_repo(repo)
        # Seed the file BEFORE branching so base stays the branch's ancestor;
        # the dirt below is uncommitted edits/untracked files on top.
        _commit_file(repo, "base.py", "b = 1\n", "seed base file")
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        # Unrelated dirt: an untracked file and a modified-but-uncommitted file
        # that the branch does not touch.
        (repo / "untracked.log").write_text("scratch", encoding="utf-8")
        (repo / "base.py").write_text("b = 2\n", encoding="utf-8")

        outcome = merge_branch_into_base(
            repo, "main", "agent/issue-7-x", worktrees_dir=repo / "wt"
        )

        assert outcome.status == "merged"
        assert outcome.fast_forward is True
        assert branch_head_sha(repo, "main") == head


class TestLocalRecordHelpers:
    def test_is_local_pr_record(self) -> None:
        assert is_local_pr_record({"local": True})
        assert not is_local_pr_record({"local": False})
        assert not is_local_pr_record({})
        assert not is_local_pr_record("nope")

    def test_local_pr_records_filters(self) -> None:
        state = {
            "prs": {
                "1": {"local": True, "status": "reviewing"},
                "2": {"status": "reviewing"},
                "3": {"local": True, "status": "merged"},
            }
        }
        records = local_pr_records(state)
        assert set(records) == {"1", "3"}

    def test_local_pr_dict_shape(self) -> None:
        record = {
            "number": 7,
            "issue_number": 7,
            "branch": "agent/issue-7-x",
            "headRefOid": "abc123",
            "baseRefName": "main",
            "title": "T",
            "local": True,
        }
        pr = local_pr_dict(record)
        assert pr["number"] == 7
        assert pr["headRefName"] == "agent/issue-7-x"
        assert pr["headRefOid"] == "abc123"
        assert pr["baseRefName"] == "main"
        assert pr["state"] == "OPEN"
        assert pr["local"] is True
        assert pr["isCrossRepository"] is False

    def test_synthesize_open_pr(self) -> None:
        record = {
            "number": 7,
            "branch": "agent/issue-7-x",
            "headRefOid": "abc",
            "baseRefName": "main",
            "status": "reviewing",
            "local": True,
        }
        pr = synthesize_open_pr(record)
        assert pr is not None
        assert pr["state"] == "OPEN"
        assert pr["headRefName"] == "agent/issue-7-x"
        assert pr["number"] == 7

    def test_synthesize_open_pr_terminal_records_excluded(self) -> None:
        for status in ("merged", "closed"):
            assert (
                synthesize_open_pr(
                    {
                        "number": 7,
                        "branch": "agent/issue-7-x",
                        "status": status,
                        "local": True,
                    }
                )
                is None
            )
        assert synthesize_open_pr({"number": 7, "status": "reviewing"}) is None


# ---------------------------------------------------------------------------
# Orchestration -- the lane itself, against a real repo + LocalFileGitHub
# ---------------------------------------------------------------------------


class TestAdoptionAndPackets:
    def test_parked_issue_adopted_and_packetized(self, repo: Path) -> None:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-thing", "feat.py", "x = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-thing")

        result = app._local_review_packets()

        assert result["adopted"] == [{"issue": 7, "branch": "agent/issue-7-thing", "head": head}]
        state = load_state_locked(app.paths.state_file)
        record = state["prs"]["7"]
        assert record["local"] is True
        assert record["status"] == "reviewing"
        assert record["headRefOid"] == head
        assert record["baseRefName"] == "main"
        # The issue entry mirrors the lane's status + branch.
        assert state["issues"]["7"]["status"] == "reviewing"
        assert state["issues"]["7"]["branch_name"] == "agent/issue-7-thing"
        # Packet files exist and carry the local diff.
        pr_dir = app.paths.prs / "pr-7"
        assert (pr_dir / "pr.json").is_file()
        assert (pr_dir / "diff.patch").is_file()
        assert "+x = 1" in (pr_dir / "diff.patch").read_text(encoding="utf-8")
        assert (pr_dir / "review-prompt.md").is_file()
        decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
        assert decision["decision"] == "pending"
        # Labels moved to the reviewing edge (pr_open + reviewing).
        current = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        labels = app.config.labels
        assert labels.reviewing in current
        assert labels.pr_open in current
        assert labels.review_ready not in current
        kinds = [e["kind"] for e in state["events"]]
        assert "local_review_adopted" in kinds

    def test_adoption_skips_issue_without_branch(self, repo: Path) -> None:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        app = _app(repo, issues_dir)
        labels = app.config.labels
        _write_issue(issues_dir, 8, labels=(labels.ready, labels.review_ready))
        _seed_issue_state(app, 8)  # no branch_name, no worktree

        result = app._local_review_packets()

        assert result["adopted"] == []
        assert result["skipped"] == [{"issue": 8, "reason": "no_committed_branch"}]
        state = load_state_locked(app.paths.state_file)
        assert "8" not in state.get("prs", {})
        kinds = [e["kind"] for e in state["events"]]
        assert "local_review_adopt_failed" in kinds

    def test_packet_phase_escalates_when_branch_deleted(self, repo: Path) -> None:
        """A local record whose branch vanished escalates, not silently skips."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()

        _git(repo, "branch", "-D", "agent/issue-7-x")

        result = app._local_review_packets()

        assert result["skipped"] == [{"issue": 7, "reason": "local_branch_missing"}]
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "escalated"
        assert state["prs"]["7"]["escalation_reason"] == "local_branch_missing"
        assert state["issues"]["7"]["status"] == "escalated"
        kinds = [e["kind"] for e in state["events"]]
        assert "local_review_packet_failed" in kinds

    def test_rework_head_move_rebuilds_packet(self, repo: Path) -> None:
        """A rework worker landing new commits triggers a packet rebuild."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()

        # Simulate rework: a new commit lands on the branch while the record
        # sits in rework_requested and the issue claims a (dead) dispatch.
        _git(repo, "checkout", "agent/issue-7-x")
        new_head = _commit_file(repo, "b.py", "b = 1\n", "fix: rework")
        _git(repo, "checkout", "main")
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["7"]["status"] = "rework_requested"
            state["issues"]["7"]["status"] = "dispatched"
            save_state(app.paths.state_file, state)

        result = app._local_review_packets()

        # Issue status "dispatched" + no live sidecar -> dead worker -> rebuild.
        assert result["packets"], "reworked head must trigger a packet rebuild"
        pr_dir = app.paths.prs / "pr-7"
        assert "+b = 1" in (pr_dir / "diff.patch").read_text(encoding="utf-8")
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["headRefOid"] == new_head
        assert state["prs"]["7"]["status"] == "reviewing"


class TestRecordLocalReview:
    def _adopted(self, repo: Path) -> tuple[OrchestratorApp, str]:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        return app, head

    def test_approved_verdict(self, repo: Path) -> None:
        app, head = self._adopted(repo)

        result = app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )

        assert result.ok, result.message
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "approved"
        assert state["prs"]["7"]["decision"] == "approved"
        assert state["issues"]["7"]["status"] == "approved"
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.pr_open in labels
        assert app.config.labels.reviewing not in labels

    def test_request_changes_routes_rework(self, repo: Path) -> None:
        app, head = self._adopted(repo)

        result = app.record_local_review(
            7,
            "request_changes",
            summary="Add a test for the new function.",
            reviewed_head=head,
            verdict_provenance="fresh_llm_review",
        )

        assert result.ok, result.message
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "request_changes"
        assert state["issues"]["7"]["status"] == "rework_requested"
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.needs_rework in labels
        # The rework brief was written for the rework dispatcher.
        assert (app.paths.prs / "pr-7" / "rework-prompt.md").is_file()
        # required_changes derived from the summary.
        decision = json.loads(
            (app.paths.prs / "pr-7" / "review-decision.json").read_text(encoding="utf-8")
        )
        assert decision["required_changes"] == ["Add a test for the new function."]

    def test_blocked_verdict_escalates(self, repo: Path) -> None:
        app, head = self._adopted(repo)

        result = app.record_local_review(
            7,
            "blocked",
            summary="Requires a product decision on scope.",
            reviewed_head=head,
            verdict_provenance="fresh_llm_review",
        )

        assert result.ok, result.message
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "blocked"
        assert state["issues"]["7"]["status"] == "blocked"
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.human_needed in labels

    def test_rejects_invalid_decision_and_provenance(self, repo: Path) -> None:
        app, head = self._adopted(repo)

        bad_decision = app.record_local_review(7, "lgtm", verdict_provenance="fresh_llm_review")
        assert not bad_decision.ok

        bad_provenance = app.record_local_review(7, "approved", verdict_provenance="made_up")
        assert not bad_provenance.ok

    def test_request_changes_requires_summary(self, repo: Path) -> None:
        app, head = self._adopted(repo)
        result = app.record_local_review(
            7,
            "request_changes",
            reviewed_head=head,
            verdict_provenance="fresh_llm_review",
        )
        assert not result.ok
        assert "summary" in result.message.lower()

    def test_rejects_wrong_head(self, repo: Path) -> None:
        app, head = self._adopted(repo)
        result = app.record_local_review(
            7,
            "approved",
            reviewed_head="0" * 40,
            verdict_provenance="fresh_llm_review",
        )
        assert not result.ok

    def test_rejects_non_local_record(self, repo: Path) -> None:
        app, head = self._adopted(repo)
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["prs"]["42"] = {"number": 42, "status": "reviewing"}
            save_state(app.paths.state_file, state)

        result = app.record_local_review(42, "approved", verdict_provenance="fresh_llm_review")
        assert not result.ok
        assert "not a local-lane record" in result.message


class TestLocalMergeGate:
    def _approved(self, repo: Path) -> tuple[OrchestratorApp, str, Path]:
        """Adopt + approve a branch; returns (app, head, issues_dir)."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        result = app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )
        assert result.ok, result.message
        return app, head, issues_dir

    def test_approved_branch_merges_and_issue_closes(self, repo: Path) -> None:
        app, head, _ = self._approved(repo)

        results = app._local_merge_approved()

        assert results[0]["outcome"] in ("merged", "already_merged")
        # Base advanced to the reviewed content.
        assert is_ancestor(repo, head, branch_head_sha(repo, "main"))
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "merged"
        assert state["prs"]["7"]["merged_sha"]
        # Issue state entry takes the remote lane's terminal convention:
        # "closed" (the merged label edge + issue_close run too).
        assert state["issues"]["7"]["status"] == "closed"
        # Issue closed + done label.
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.done in labels
        assert str(app.gh.issue_view(7)["state"]).upper() == "CLOSED"
        # Worker worktree released.
        assert worktree_for_branch(repo, "agent/issue-7-x") is None
        kinds = [e["kind"] for e in state["events"]]
        assert "local_suite_result" in kinds

    def test_suite_failure_routes_to_rework(self, repo: Path) -> None:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        # Branch adds a failing test -> suite gate must fail. The suite runs
        # inside the branch worktree, so pytest picks up test_bad.py there.
        head = _make_branch(repo, "agent/issue-7-x", "test_bad.py", "def test_x(): assert False\n")
        config = _local_config(repo, issues_dir, dispatch={"test_command": "python -m pytest"})
        app = _app(repo, issues_dir, config=config)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "suite_failed"
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "rework_requested"
        assert state["issues"]["7"]["status"] == "rework_requested"
        # Base NOT advanced.
        assert not is_ancestor(repo, head, branch_head_sha(repo, "main"))
        kinds = [e["kind"] for e in state["events"]]
        assert "local_suite_failed" in kinds
        assert "local_suite_result" in kinds

    def test_merge_defers_when_base_checkout_blocks(self, repo: Path) -> None:
        """A base checkout whose working tree would be clobbered defers the
        merge rather than failing or forcing it (issue #1844)."""
        app, head, _ = self._approved(repo)
        # main is checked out at ``repo``; an untracked file at a path the
        # branch adds makes ``git merge --ff-only`` refuse.
        (repo / "a.py").write_text("untracked = True\n", encoding="utf-8")

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "deferred"
        state = load_state_locked(app.paths.state_file)
        kinds = [e["kind"] for e in state["events"]]
        assert "local_merge_deferred" in kinds
        # The approval survives the deferral -- the record stays approved and
        # the base did not advance.
        assert state["prs"]["7"]["status"] == "approved"
        assert not is_ancestor(repo, head, branch_head_sha(repo, "main"))

    def test_merge_error_escalates_when_no_suite_command(self, repo: Path) -> None:
        """No resolvable suite command is a merge-gate infra failure: the lane
        escalates instead of merging unverified (issue #1844)."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        config = _local_config(repo, issues_dir, dispatch={"test_command": ""})
        app = _app(repo, issues_dir, config=config)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "error"
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "escalated"
        assert state["prs"]["7"]["escalation_reason"] == "local_merge_error"
        assert state["issues"]["7"]["status"] == "escalated"
        kinds = [e["kind"] for e in state["events"]]
        assert "local_merge_failed" in kinds
        # Base NOT advanced -- an unverified merge never lands.
        assert not is_ancestor(repo, head, branch_head_sha(repo, "main"))

    def test_merge_conflict_routes_to_rework(self, repo: Path) -> None:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        _commit_file(repo, "shared.py", "v = 1\n", "seed shared")
        head = _make_branch(repo, "agent/issue-7-x", "shared.py", "v = 2\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )
        # Base moves with a conflicting edit AFTER approval.
        _commit_file(repo, "shared.py", "v = 3\n", "conflicting base edit")

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "conflict"
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "rework_requested"
        assert state["issues"]["7"]["status"] == "rework_requested"
        labels = {lb["name"] for lb in app.gh.issue_view(7)["labels"]}
        assert app.config.labels.needs_rework in labels

    def test_no_merge_without_approval(self, repo: Path) -> None:
        """reviewing records must never reach the merge gate."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()

        results = app._local_merge_approved()

        assert results == []
        assert not is_ancestor(repo, head, branch_head_sha(repo, "main"))

    def test_auto_merge_disabled_skips_gate(self, repo: Path) -> None:
        """Kill switch: auto_merge.enabled=False holds approved work unmerged."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        config = _local_config(repo, issues_dir, auto_merge={"enabled": False})
        app = _app(repo, issues_dir, config=config)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )

        result = app._local_lane()

        assert result.ok
        assert not is_ancestor(repo, head, branch_head_sha(repo, "main"))
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["status"] == "approved"

    def test_diverged_base_merges_via_sync_then_ff(self, repo: Path) -> None:
        """Base moved (non-conflicting) post-approval: sync-merge then merge."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        app.record_local_review(
            7, "approved", reviewed_head=head, verdict_provenance="fresh_llm_review"
        )
        _commit_file(repo, "unrelated.py", "u = 1\n", "base moved")

        results = app._local_merge_approved()

        assert results[0]["outcome"] == "merged"
        main_head = branch_head_sha(repo, "main")
        assert is_ancestor(repo, head, main_head)


class TestLaneEntryPoint:
    def test_noop_on_publishing_backend(self, repo: Path) -> None:
        class _Publishing:
            publishes_pull_requests = True

        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        config = _local_config(repo, issues_dir)
        paths = runtime_paths(repo, config.runtime.state_dir)
        app = OrchestratorApp(repo, paths, config, _Publishing())

        result = app._local_lane()

        assert result.ok
        assert result.data == {"local": False}

    def test_full_pass_adopts_and_stages_for_review(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")

        # Never spawn a real reviewer process in tests.
        def _fake_launch(*args, **kwargs) -> ClaudeWorkerRecord:
            return ClaudeWorkerRecord(
                issue_number=kwargs.get("issue_number") or 7,
                branch=kwargs.get("branch") or "agent/issue-7-x",
                worktree_path="/fake/wt",
                prompt_path="/fake/prompt.md",
                command=("claude", "-p"),
                pid=4242,
                started_at="2026-09-24T00:00:00Z",
                log_path="/fake/log.log",
                error=None,
                process_start_time=1.0,
            )

        monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", _fake_launch)

        result = app._local_lane()

        assert result.ok
        assert result.data["adopted"]
        state = load_state_locked(app.paths.state_file)
        assert state["prs"]["7"]["review_dispatch_status"] == "review_dispatch_dispatched"
        assert state["prs"]["7"]["reviewer_pid"] == 4242


class TestReconcileSafety:
    def test_local_record_not_drifted_as_missing_pr(self, repo: Path) -> None:
        """Synthetic local PR entries keep reconcile from stripping labels."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        labels_cfg = LabelConfig()
        _write_issue(
            issues_dir,
            7,
            labels=(labels_cfg.pr_open, labels_cfg.reviewing),
        )
        config = _local_config(repo, issues_dir)
        gh = LocalFileGitHub(repo_root=repo, issues_dir=issues_dir)
        paths = runtime_paths(repo, config.runtime.state_dir)
        save_state(
            paths.state_file,
            {
                "issues": {"7": {"number": 7, "status": "reviewing", "title": "T"}},
                "prs": {
                    "7": {
                        "number": 7,
                        "local": True,
                        "issue_number": 7,
                        "branch": "agent/issue-7-x",
                        "headRefName": "agent/issue-7-x",
                        "headRefOid": "abc",
                        "baseRefName": "main",
                        "status": "reviewing",
                    }
                },
            },
        )

        drift = detect_drift(gh, load_state_locked(paths.state_file), config, repo_root=repo)

        kinds = {item.kind for item in drift}
        assert "state_pr_missing_on_github" not in kinds
        assert "issue_active_label_no_open_pr" not in kinds


class TestDispatchKillSwitch:
    def test_review_dispatch_disabled_does_not_claim(self, repo: Path) -> None:
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        config = _local_config(repo, issues_dir, review_dispatch={"enabled": False})
        app = _app(repo, issues_dir, config=config)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")

        result = app._local_lane()

        assert result.ok
        state = load_state_locked(app.paths.state_file)
        record = state["prs"]["7"]
        # Packet built (adoption is not gated on review_dispatch.enabled) but
        # no reviewer claim was laid.
        assert record["status"] == "reviewing"
        assert record.get("review_dispatch_status") is None


class TestDrainSuppressesLocalRework:
    """``loop(limit=0)`` -- the drain pass's forced budget (issue #1716).

    ``fleet stop --drain`` suppresses new launches by forcing
    ``app.loop(0)``; the remote rework/fresh lanes honor that via their
    ``candidates[:0]`` slice. ``_local_dispatch_rework`` is the local lane's
    only ``dispatch_sessions`` caller with no ``*_enabled`` kill switch of
    its own, so the explicit ``0`` threaded from ``_loop_body`` is its only
    drain signal -- without the gate a draining no-remote repo kept
    launching rework workers and the drain could never converge.
    """

    def _rework_pending(self, repo: Path) -> OrchestratorApp:
        """Park -> adopt -> ``request_changes``: lands issue 7 in
        ``rework_requested`` with ``rework-prompt.md`` written -- exactly the
        state ``_local_dispatch_rework`` selects on."""
        _init_repo(repo)
        issues_dir = repo / "docs" / "issues"
        head = _make_branch(repo, "agent/issue-7-x", "a.py", "a = 1\n")
        app = _app(repo, issues_dir)
        _parked_issue(app, issues_dir, 7, "agent/issue-7-x")
        app._local_review_packets()
        verdict = app.record_local_review(
            7,
            "request_changes",
            summary="Add coverage for the new path.",
            reviewed_head=head,
            verdict_provenance="fresh_llm_review",
        )
        assert verdict.ok, verdict.message
        state = load_state_locked(app.paths.state_file)
        assert state["issues"]["7"]["status"] == "rework_requested"
        assert (app.paths.prs / "pr-7" / "rework-prompt.md").is_file()
        return app

    @staticmethod
    def _spy_dispatch_sessions(
        monkeypatch: pytest.MonkeyPatch,
    ) -> list[SessionRequest]:
        calls: list[SessionRequest] = []

        def _fake(_repo_root, _manifest, _results, _settings, requests):
            calls.extend(requests)
            return [
                SessionDispatchResult(
                    issue_number=request.issue_number,
                    issue_title=request.issue_title,
                    prompt_path=str(request.prompt_path),
                    branch_name=request.branch_name,
                    adapter="claude-code",
                    ok=True,
                )
                for request in requests
            ]

        monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake)
        return calls

    def test_loop_zero_limit_suppresses_local_rework_dispatch(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = self._rework_pending(repo)
        calls = self._spy_dispatch_sessions(monkeypatch)

        result = app.loop(limit=0)

        assert result.ok, result.message
        # No session was dispatched and the issue stays queued for rework --
        # it is picked up by the first non-draining pass, not dropped.
        assert calls == []
        state = load_state_locked(app.paths.state_file)
        assert state["issues"]["7"]["status"] == "rework_requested"
        # The rest of the local lane still ran: drain suppresses launches,
        # not the reap/review bookkeeping that lets in-flight work finish.
        assert result.data["local_lane"]["rework_launches_suspended"] is True

    def test_loop_positive_limit_dispatches_local_rework(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: the same rework-pending state under a nonzero budget
        must dispatch -- otherwise the zero-limit test proves nothing."""
        app = self._rework_pending(repo)
        calls = self._spy_dispatch_sessions(monkeypatch)

        result = app.loop(limit=1)

        assert result.ok, result.message
        assert [request.issue_number for request in calls] == [7]
        assert calls[0].rework is True
        state = load_state_locked(app.paths.state_file)
        assert state["issues"]["7"]["status"] == "dispatched"

    def test_local_rework_claim_clears_dead_worker_failure_kind(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Issue #1917: the local rework claim drops the previous death's
        classification -- a stale provider-throttle stamp must not survive
        into the new dispatch epoch and exempt a later, genuinely
        different death."""
        app = self._rework_pending(repo)
        with state_lock(app.paths.state_file):
            state = load_state(app.paths.state_file)
            state["issues"]["7"]["dead_worker_failure_kind"] = "rate_limited"
            save_state(app.paths.state_file, state)
        calls = self._spy_dispatch_sessions(monkeypatch)

        result = app.loop(limit=1)

        assert result.ok, result.message
        assert [request.issue_number for request in calls] == [7]
        state = load_state_locked(app.paths.state_file)
        entry = state["issues"]["7"]
        assert entry["status"] == "dispatched"
        assert "dead_worker_failure_kind" not in entry
