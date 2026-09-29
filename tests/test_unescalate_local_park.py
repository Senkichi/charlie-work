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
- CLI wiring: ``--requeue`` reaches ``OrchestratorApp.unescalate``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _cli_fixtures import _FakeGitHub, _make_repo
from _unescalate_fixtures import _app, _events
from charlie_work import cli
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
