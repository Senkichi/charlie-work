"""Tests for #1970: ``unescalate`` parks a finished local worker branch
for the local path instead of dropping the issue to the fresh-dispatch
baseline.

On a backend that cannot publish pull requests (``LocalFileGitHub``)
there is no PR to point a reviewer at -- the worker's branch IS the
deliverable. The no-live-PR branch of ``unescalate`` used to drop such
issues unconditionally: the next dispatch started a fresh worker that
either discarded the finished commits or escalated straight back through
#1944's diverged-branch check.

Now a non-empty ``branch_diff`` against the local base parks the issue
the same way ``park_unpublishable_work`` does at a dead worker --
``open_passive`` + the ``local_work_ready`` edge (``agent:review-ready``)
so ``_local_review_packets`` adopts the branch and mints a local review
record on the next pass. ``--requeue`` keeps the old drop for an operator
who wants a fresh worker anyway.

Covered here:

- happy path: label edge, status flip, branch preserved, event payload,
  and a following ``_local_review_packets`` pass adopting the branch;
- ``--requeue``: the old drop (``unescalated_requeued``) on identical git
  state;
- controls: branch at base (empty diff), recorded branch ref gone (diff
  failure), and a PR-capable backend on identical git state -- the
  capability probe, not the git state, selects the lane;
- dry-run reports ``local_work_ready`` without mutating labels or state;
- CLI wiring: ``--requeue`` reaches ``OrchestratorApp.unescalate``;
- the rework finding (PR #1987 review): an EXISTING local lane record for
  the issue is re-armed into the lane by the park (``local_pending``, or
  ``approved`` when a still-valid approval is on file, with a still-valid
  terminal verdict voided) instead of being stranded at ``open_passive``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _cli_fixtures import _FakeGitHub, _make_repo
from _unescalate_fixtures import _app, _events
from charlie_work import cli
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import build_config_from_data
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    PASSIVE_OPEN_STATUS,
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import CommandResult, OrchestratorApp


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


def _make_branch(repo_root: Path, branch: str) -> str:
    """Branch off main with one committed file; return the new head sha."""
    _git(repo_root, "checkout", "-b", branch)
    (repo_root / "feat.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo_root, "add", "feat.py")
    _git(repo_root, "commit", "-m", "feat: work")
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
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


def _local_app(repo_root: Path, issues_dir: Path) -> OrchestratorApp:
    config = build_config_from_data(
        {
            "local_issues": {"enabled": True, "issues_dir": "docs/issues"},
            # Isolate the post-mortem activity probe from any real
            # sessions.db on the host (same rationale as
            # _unescalate_fixtures._app).
            "post_mortem": {"db_path": str(repo_root / "missing-sessions.db")},
        }
    )
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)
    paths = runtime_paths(repo_root, config.runtime.state_dir)
    return OrchestratorApp(repo_root, paths, config, gh)


def _seed_escalated(app: OrchestratorApp, issue_number: int, *, branch: str | None) -> None:
    """The state entry an escalated local issue carries: sink status, the
    dispatch-recorded ``branch_name``, and a cleared-by-reset reason."""
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state.setdefault("issues", {})[str(issue_number)] = {
            "number": issue_number,
            "title": "Test issue",
            "status": "escalated",
            "escalation_reason": "session_failed_escalated",
            **({"branch_name": branch} if branch else {}),
        }
        save_state(app.paths.state_file, state)


def _label_names(gh: LocalFileGitHub, issue_number: int) -> set[str]:
    return {entry["name"] for entry in gh.issue_view(issue_number)["labels"]}


def _seed_reviewing_lane_record(
    app: OrchestratorApp,
    issues_dir: Path,
    issue_number: int,
    branch: str,
    head: str,
) -> None:
    """Drive the real adoption + packet build so ``prs[N]`` is a live lane
    record with a current packet (status ``reviewing``) -- the shape an
    already-in-flight local issue has before something escalates it."""
    labels = app.config.labels
    _write_issue(issues_dir, issue_number, labels=(labels.review_ready,))
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state.setdefault("issues", {})[str(issue_number)] = {
            "number": issue_number,
            "title": "Test issue",
            "status": PASSIVE_OPEN_STATUS,
            "branch_name": branch,
        }
        save_state(app.paths.state_file, state)
    built = app._local_review_packets()
    assert built["adopted"] == [{"issue": issue_number, "branch": branch, "head": head}]
    state = load_state(app.paths.state_file)
    assert state["prs"][str(issue_number)]["status"] == "reviewing"
    assert (app.paths.prs / f"pr-{issue_number}" / "review-prompt.md").is_file()


def _escalate_lane(
    app: OrchestratorApp,
    issues_dir: Path,
    issue_number: int,
    *,
    reason: str = "max_review_dispatch_attempts_exceeded",
) -> None:
    """The reviewer-attempt-cap escalation shape: record AND issue in the
    sink, ``agent:human-needed`` on the issue file, attempt budget spent."""
    labels = app.config.labels
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(issue_number)] = {
            **state["prs"][str(issue_number)],
            "status": "escalated",
            "escalation_reason": reason,
            "review_dispatch_attempt_count": (
                app.config.review_dispatch.max_review_dispatch_attempts
            ),
        }
        state["issues"][str(issue_number)] = {
            **state["issues"][str(issue_number)],
            "status": "escalated",
            "escalation_reason": reason,
        }
        save_state(app.paths.state_file, state)
    _write_issue(issues_dir, issue_number, labels=(labels.human_needed,))


# ---------------------------------------------------------------------------
# The #1970 regression: a finished local branch is parked, not dropped.
# ---------------------------------------------------------------------------


def test_unescalate_parks_finished_local_branch(tmp_path: Path) -> None:
    """Escalated issue, local backend, no PR, branch ahead of base:
    ``unescalate --issue`` leaves the issue ``open_passive`` with
    ``agent:review-ready`` -- and the next local-path pass adopts the
    branch into a local review record."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-1970-thing"
    head = _make_branch(repo_root, branch)
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 1970, labels=(labels.in_progress, labels.human_needed))
    _seed_escalated(app, 1970, branch=branch)

    result = app.unescalate(issue_number=1970)

    assert result.ok is True
    assert result.data["changed"] is True
    assert result.data["label_edge"] == "local_work_ready"
    assert result.data["parked_branch"] == branch

    # State: the open_passive placeholder keeps the branch locator --
    # adoption resolves the branch through the same field.
    state = load_state(app.paths.state_file)
    entry = state["issues"]["1970"]
    assert entry["status"] == PASSIVE_OPEN_STATUS
    assert entry["branch_name"] == branch
    assert "escalation_reason" not in entry

    # The local_work_ready edge applied: review_ready added, the rest of
    # the workflow labels stripped.
    assert _label_names(app.gh, 1970) == {labels.review_ready}

    # The event records the edge and the parked branch.
    unescalate_events = _events(state, "unescalate")
    assert len(unescalate_events) == 1
    payload = unescalate_events[0]["payload"]
    assert payload["label_edge"] == "local_work_ready"
    assert payload["parked_branch"] == branch
    assert payload["transitions"]["issue.status"] == ["escalated", PASSIVE_OPEN_STATUS]

    # The next local-path pass adopts the parked issue and mints a local
    # review record for the branch.
    adopt = app._local_review_packets()
    assert adopt["adopted"] == [{"issue": 1970, "branch": branch, "head": head}]
    state = load_state(app.paths.state_file)
    record = state["prs"]["1970"]
    assert record["local"] is True
    assert record["issue_number"] == 1970
    assert record["branch"] == branch
    assert record["headRefOid"] == head
    assert "local_review_adopted" in [e["kind"] for e in state["events"]]


def test_unescalate_requeue_drops_finished_local_branch(tmp_path: Path) -> None:
    """``--requeue`` on the identical git state keeps the old behavior:
    status dropped and ``unescalated_requeued`` -- the operator asked for
    a fresh worker, so the finished branch is left for the next dispatch
    to resolve, not parked for review."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-1970-thing"
    _make_branch(repo_root, branch)
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 1970, labels=(labels.in_progress, labels.human_needed))
    _seed_escalated(app, 1970, branch=branch)

    result = app.unescalate(issue_number=1970, requeue=True)

    assert result.ok is True
    assert result.data["changed"] is True
    assert result.data["label_edge"] == "unescalated_requeued"
    assert result.data["parked_branch"] is None

    state = load_state(app.paths.state_file)
    entry = state["issues"]["1970"]
    assert "status" not in entry

    # Every workflow label stripped, nothing added -- the issue is back
    # at the fresh-dispatch baseline.
    assert _label_names(app.gh, 1970) == set()

    unescalate_events = _events(state, "unescalate")
    assert len(unescalate_events) == 1
    assert unescalate_events[0]["payload"]["label_edge"] == "unescalated_requeued"
    assert unescalate_events[0]["payload"]["parked_branch"] is None

    # Nothing parked: a local-path pass finds no review-ready issue.
    assert app._local_review_packets()["adopted"] == []


# ---------------------------------------------------------------------------
# Controls: only provably-finished local work is parked.
# ---------------------------------------------------------------------------


def test_unescalate_drops_local_branch_with_no_commits(tmp_path: Path) -> None:
    """A branch that exists but carries no diff against the local base is
    empty scaffolding, not finished work -- the old drop stands."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-7-empty"
    _git(repo_root, "branch", branch)  # at main HEAD: zero commits ahead
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 7, labels=(labels.human_needed,))
    _seed_escalated(app, 7, branch=branch)

    result = app.unescalate(issue_number=7)

    assert result.ok is True
    assert result.data["label_edge"] == "unescalated_requeued"
    assert result.data["parked_branch"] is None
    state = load_state(app.paths.state_file)
    assert "status" not in state["issues"]["7"]
    assert labels.review_ready not in _label_names(app.gh, 7)


def test_unescalate_drops_when_recorded_branch_missing(tmp_path: Path) -> None:
    """A recorded ``branch_name`` whose ref no longer resolves cannot be
    adopted; ``branch_diff`` failing is not proof of work, so the issue
    drops like before."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 7, labels=(labels.human_needed,))
    _seed_escalated(app, 7, branch="agent/issue-7-gone")

    result = app.unescalate(issue_number=7)

    assert result.ok is True
    assert result.data["label_edge"] == "unescalated_requeued"
    assert result.data["parked_branch"] is None
    state = load_state(app.paths.state_file)
    assert "status" not in state["issues"]["7"]
    assert labels.review_ready not in _label_names(app.gh, 7)


def test_unescalate_drops_when_no_branch_recorded(tmp_path: Path) -> None:
    """An escalated issue with no ``branch_name`` and no convention-named
    worktree branch has nothing to park -- the old drop stands."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 7, labels=(labels.human_needed,))
    _seed_escalated(app, 7, branch=None)

    result = app.unescalate(issue_number=7)

    assert result.ok is True
    assert result.data["label_edge"] == "unescalated_requeued"
    state = load_state(app.paths.state_file)
    assert "status" not in state["issues"]["7"]
    assert labels.review_ready not in _label_names(app.gh, 7)


def test_unescalate_pr_capable_backend_still_drops(tmp_path: Path) -> None:
    """Identical git state (branch ahead of base) on a PR-capable backend
    keeps the drop -- the capability probe, not the presence of commits,
    selects the lane. On a remote backend the branch can be pushed and
    PRed; there is nothing to park."""
    _init_repo(tmp_path)
    branch = _make_branch(tmp_path, "agent/issue-8-x")
    app = _app(tmp_path)  # FakeGitHub: publishes_pull_requests defaults True
    _seed_escalated(app, 8, branch=branch)

    result = app.unescalate(issue_number=8)

    assert result.ok is True
    assert result.data["label_edge"] == "unescalated_requeued"
    assert result.data["parked_branch"] is None
    state = load_state(app.paths.state_file)
    assert "status" not in state["issues"]["8"]
    labels = app.config.labels
    assert (8, labels.review_ready) not in app.gh.labels_added
    assert (8, labels.human_needed) in app.gh.labels_removed


def test_unescalate_dry_run_reports_park_without_mutating(tmp_path: Path) -> None:
    """Dry-run computes the same local_work_ready transition but writes
    nothing: no status flip, no label edge, no event."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-1970-thing"
    _make_branch(repo_root, branch)
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 1970, labels=(labels.in_progress, labels.human_needed))
    _seed_escalated(app, 1970, branch=branch)

    result = app.unescalate(issue_number=1970, dry_run=True)

    assert result.ok is True
    assert result.data["changed"] is False
    assert result.data["label_edge"] == "local_work_ready"
    assert result.data["parked_branch"] == branch

    state = load_state(app.paths.state_file)
    assert state["issues"]["1970"]["status"] == "escalated"
    assert _events(state, "unescalate") == []
    assert _label_names(app.gh, 1970) == {labels.in_progress, labels.human_needed}


# ---------------------------------------------------------------------------
# CLI wiring: --requeue reaches OrchestratorApp.unescalate.
# ---------------------------------------------------------------------------


def test_cli_unescalate_requeue_flag_reaches_app(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "GitHub", _FakeGitHub)
    repo = _make_repo(tmp_path)
    captured: dict[str, object] = {}

    def _capture(
        self: OrchestratorApp,
        pr_number: int | None = None,
        issue_number: int | None = None,
        *,
        dry_run: bool = False,
        requeue: bool = False,
    ) -> CommandResult:
        captured.update(
            pr_number=pr_number,
            issue_number=issue_number,
            dry_run=dry_run,
            requeue=requeue,
        )
        return CommandResult(True, "ok", {})

    monkeypatch.setattr(OrchestratorApp, "unescalate", _capture)

    rc = cli.main(["--repo", str(repo), "unescalate", "--issue", "7", "--requeue"])

    assert rc == 0
    assert captured == {
        "pr_number": None,
        "issue_number": 7,
        "dry_run": False,
        "requeue": True,
    }

    captured.clear()
    rc = cli.main(["--repo", str(repo), "unescalate", "--issue", "7"])

    assert rc == 0
    assert captured["requeue"] is False


# ---------------------------------------------------------------------------
# PR #1987 rework finding: a local lane record that ALREADY exists for the
# issue must re-enter the lane under the park. The old behavior reset the
# record to ``open_passive`` -- a status no local lane phase consumes:
# adoption skips issues that already have a record, the packet pass sees a
# current packet and does not rebuild, and the reviewer/merge lanes only
# select ``reviewing``/``approved`` records. The issue sat at
# agent:review-ready forever -- a silent strand.
# ---------------------------------------------------------------------------


def test_unescalate_park_rearms_existing_lane_record(tmp_path: Path, monkeypatch) -> None:
    """Escalated lane record with a CURRENT packet + branch ahead of base:
    ``unescalate --issue`` parks the issue AND re-arms the record to
    ``local_pending`` so the next packet pass rebuilds it and the reviewer
    dispatcher can claim it -- the issue is not stranded."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-1970-thing"
    head = _make_branch(repo_root, branch)
    app = _local_app(repo_root, issues_dir)
    labels = app.config.labels
    _seed_reviewing_lane_record(app, issues_dir, 1970, branch, head)
    _escalate_lane(app, issues_dir, 1970)

    result = app.unescalate(issue_number=1970)

    assert result.ok is True
    assert result.data["label_edge"] == "local_work_ready"
    assert result.data["parked_branch"] == branch

    state = load_state(app.paths.state_file)
    record = state["prs"]["1970"]
    # Not open_passive: the record re-enters the lane at local_pending,
    # with the spent attempt budget and the escalation bookkeeping reset.
    assert record["status"] == "local_pending"
    assert record["review_dispatch_attempt_count"] == 0
    assert "escalation_reason" not in record
    assert record["branch"] == branch
    issue_entry = state["issues"]["1970"]
    assert issue_entry["status"] == PASSIVE_OPEN_STATUS
    assert _label_names(app.gh, 1970) == {labels.review_ready}
    transitions = _events(state, "unescalate")[0]["payload"]["transitions"]
    assert transitions["pr.status"] == ["escalated", "local_pending"]

    # The next packet pass rebuilds the existing record (no fresh
    # adoption) and re-enters it into review.
    rebuilt = app._local_review_packets()
    assert rebuilt["adopted"] == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["1970"]["status"] == "reviewing"

    # And the reviewer dispatcher really can claim it -- no strand.
    def _fake_launch(*args, **kwargs) -> ClaudeWorkerRecord:
        return ClaudeWorkerRecord(
            issue_number=1970,
            branch=branch,
            worktree_path="/fake/wt",
            prompt_path="/fake/prompt.md",
            command=("claude", "-p"),
            pid=4242,
            started_at="2026-09-29T00:00:00Z",
            log_path="/fake/log.log",
            error=None,
            process_start_time=1.0,
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", _fake_launch)
    dispatched = app._local_dispatch_reviewers()
    assert dispatched["claimed"] == [1970]
    assert dispatched["launched"][0]["pr"] == 1970


def test_unescalate_park_voids_still_valid_terminal_verdict(tmp_path: Path) -> None:
    """An escalated lane record carrying a request_changes verdict still
    valid at the branch head gets the same void the remote path performs:
    the verdict is archived to a pending stub so the rebuilt record is
    dispatchable -- a still-valid terminal decision would make
    ``_local_dispatch_reviewers`` skip it forever."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-1970-thing"
    head = _make_branch(repo_root, branch)
    app = _local_app(repo_root, issues_dir)
    _seed_reviewing_lane_record(app, issues_dir, 1970, branch, head)
    verdict = app.record_local_review(
        1970,
        "request_changes",
        summary="Add coverage for the new path.",
        reviewed_head=head,
        verdict_provenance="fresh_llm_review",
    )
    assert verdict.ok, verdict.message
    _escalate_lane(app, issues_dir, 1970)

    result = app.unescalate(issue_number=1970)

    assert result.ok is True
    assert result.data["parked_branch"] == branch
    assert result.data["verdict_voided"] is True
    state = load_state(app.paths.state_file)
    assert state["prs"]["1970"]["status"] == "local_pending"
    # The flat decision file is a pending stub again -- not the terminal
    # verdict that would make dispatch skip the record.
    decision = app._review_decision(1970)
    assert decision["decision"] == "pending"
    assert decision["reviewed_head_sha"] == head
    # The voided verdict is preserved in the rounds archive.
    assert (app.paths.prs / "pr-1970" / "rounds").is_dir()

    rebuilt = app._local_review_packets()
    assert rebuilt["adopted"] == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["1970"]["status"] == "reviewing"


def test_unescalate_park_restores_valid_approval_to_merge_lane(tmp_path: Path) -> None:
    """An escalated lane record whose approval is still valid at the
    branch head re-enters through ``approved`` -- the merge gate's own
    selection status -- instead of going back through review (the remote
    path's deliberate never-void-approved rule, applied to the local
    lane)."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    issues_dir = repo_root / "docs" / "issues"
    branch = "agent/issue-1970-thing"
    head = _make_branch(repo_root, branch)
    app = _local_app(repo_root, issues_dir)
    _seed_reviewing_lane_record(app, issues_dir, 1970, branch, head)
    verdict = app.record_local_review(
        1970,
        "approved",
        reviewed_head=head,
        verdict_provenance="fresh_llm_review",
    )
    assert verdict.ok, verdict.message
    _escalate_lane(app, issues_dir, 1970)

    result = app.unescalate(issue_number=1970)

    assert result.ok is True
    assert result.data["parked_branch"] == branch
    assert result.data["verdict_voided"] is False
    state = load_state(app.paths.state_file)
    record = state["prs"]["1970"]
    # ``_local_merge_approved`` selects exactly this status -- the record
    # is back in a lane instead of stranded.
    assert record["status"] == "approved"
    assert record["review_dispatch_attempt_count"] == 0
    assert "escalation_reason" not in record
    decision = app._review_decision(1970)
    assert decision["decision"] == "approved"
    assert decision["reviewed_head_sha"] == head
