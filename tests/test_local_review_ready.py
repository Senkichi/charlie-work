"""Tests for the local-file issue source's review-ready hand-off.

Covers the "no-PR backend" salvage path end to end:

- ``LabelConfig.review_ready``'s membership in the workflow-label sets
  (``labels.py`` reads every label string from a ``LabelConfig`` instance,
  never a hardcoded literal).
- ``labels.transition`` applying the ``"local_work_ready"`` edge against a
  real ``LocalFileGitHub``.
- ``OrchestratorApp._is_dispatchable`` holding a review-ready issue out of
  dispatch, with a positive control.
- ``local_work_park.park_unpublishable_work`` directly: the capability-probe
  short-circuit (``publishes_pull_requests`` absent/True), the real park
  (labels + comment + event), the dry-run no-op, and the label-write-failure
  path.
- ``dead_worker_reap._attempt_salvage`` end to end against a real git repo
  with no origin remote and a real linked worktree, proving the local-file
  backend parks instead of pushing (``push_branch`` is monkeypatched to
  raise if reached), plus the flip side: a backend that DOES publish PRs
  must still reach ``push_branch``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import charlie_work.dead_worker_reap as dead_worker_reap
from charlie_work.config import LabelConfig, OrchestratorConfig, load_config
from charlie_work.dead_worker_reap import _attempt_salvage
from charlie_work.github_capabilities.pull_requests import MergedPRSearchResult
from charlie_work.labels import TransitionOutcome, transition
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.local_work_park import park_unpublishable_work
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate

# ---------------------------------------------------------------------------
# Inlined helpers -- self-contained per this repo's test-file convention
# (see test_salvage_dry_run_1418.py), not shared with the other two new
# files or with tests/test_local_issues.py (owned by another agent).
# ---------------------------------------------------------------------------


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote."""
    repo_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "--allow-empty",
            "-m",
            "chore: seed",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


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


def _wg(state_file: Path, *, dry_run: bool = False, repo: str = "test-repo") -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo=repo)


class _NoAttrGitHubDouble:
    """A GitHubLike double with NO ``publishes_pull_requests`` attribute at
    all -- the capability probe (``getattr(gh, "publishes_pull_requests",
    True)``) must default to True for this shape, exactly like every
    pre-local-issues test double in this repo."""


class _ExplicitPublishingGitHubDouble:
    publishes_pull_requests = True


class _PublishingGitHubDouble:
    """No ``publishes_pull_requests`` attribute (defaults True), but with
    the two GitHubLike members ``_salvage_already_landed`` calls before
    ``_attempt_salvage`` would reach the push-branch step."""

    def issue_view(self, number: int) -> dict:
        return {"number": number, "state": "OPEN"}

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str) -> MergedPRSearchResult:
        return MergedPRSearchResult([], ok=True)


# ---------------------------------------------------------------------------
# LabelConfig membership
# ---------------------------------------------------------------------------


def test_review_ready_label_membership() -> None:
    labels = LabelConfig()
    assert labels.review_ready in labels.terminal
    assert labels.review_ready in labels.all
    assert labels.review_ready in labels.workflow_labels
    assert labels.review_ready not in labels.active


# ---------------------------------------------------------------------------
# labels.transition against a real LocalFileGitHub
# ---------------------------------------------------------------------------


def test_transition_local_work_ready_adds_review_ready_keeps_ready(tmp_path: Path) -> None:
    labels_cfg = LabelConfig()
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1, labels=(labels_cfg.ready, labels_cfg.in_progress))
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)

    result = transition(gh, labels_cfg, 1, "local_work_ready")

    assert result.outcome is TransitionOutcome.APPLIED
    current_labels = {entry["name"] for entry in gh.issue_view(1)["labels"]}
    assert labels_cfg.review_ready in current_labels
    assert labels_cfg.ready in current_labels, "ready must survive -- it is not a workflow_label"
    assert labels_cfg.in_progress not in current_labels, "the active label must be removed"


# ---------------------------------------------------------------------------
# _is_dispatchable, with a positive control
# ---------------------------------------------------------------------------


def test_is_dispatchable_excludes_review_ready_with_control(tmp_path: Path) -> None:
    labels_cfg = LabelConfig()
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    parked_issue = {
        "number": 1,
        "state": "OPEN",
        "labels": [{"name": labels_cfg.ready}, {"name": labels_cfg.review_ready}],
    }
    control_issue = {
        "number": 2,
        "state": "OPEN",
        "labels": [{"name": labels_cfg.ready}],
    }

    assert app._is_dispatchable(parked_issue, operator_claimed=set()) is False
    assert app._is_dispatchable(control_issue, operator_claimed=set()) is True, (
        "control: the identical issue minus review_ready must be dispatchable -- "
        "otherwise the first assertion could be passing for an unrelated reason"
    )


# ---------------------------------------------------------------------------
# local_work_park.park_unpublishable_work, direct
# ---------------------------------------------------------------------------


def test_park_unpublishable_work_none_when_gh_has_no_attribute(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    write_gate = _wg(tmp_path / "state.json")

    result = park_unpublishable_work(
        _NoAttrGitHubDouble(), config, tmp_path, "agent/issue-1-x", 1, set(), None, write_gate
    )

    assert result is None


def test_park_unpublishable_work_none_when_gh_declares_true(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    write_gate = _wg(tmp_path / "state.json")

    result = park_unpublishable_work(
        _ExplicitPublishingGitHubDouble(),
        config,
        tmp_path,
        "agent/issue-1-x",
        1,
        set(),
        None,
        write_gate,
    )

    assert result is None


def test_park_unpublishable_work_parks_local_backend(tmp_path: Path) -> None:
    labels_cfg = LabelConfig()
    config = OrchestratorConfig()
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1, labels=(labels_cfg.ready, labels_cfg.in_progress))
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    state_file = tmp_path / "state.json"
    save_state(state_file, {})
    write_gate = _wg(state_file)

    ok, error = park_unpublishable_work(
        gh,
        config,
        tmp_path,
        "agent/issue-1-local",
        1,
        {labels_cfg.in_progress},
        "session_exited",
        write_gate,
    )

    assert (ok, error) == (True, None)
    issue = gh.issue_view(1)
    current_labels = {entry["name"] for entry in issue["labels"]}
    assert labels_cfg.review_ready in current_labels
    assert labels_cfg.ready in current_labels
    assert labels_cfg.in_progress not in current_labels
    assert len(issue["comments"]) == 1
    assert "agent/issue-1-local" in issue["comments"][0]["body"]

    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "local_work_ready"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["branch"] == "agent/issue-1-local"
    assert payload["issue_number"] == 1
    assert payload["label_write_ok"] is True


def test_park_unpublishable_work_dry_run_no_writes(tmp_path: Path) -> None:
    labels_cfg = LabelConfig()
    config = OrchestratorConfig()
    issues_dir = tmp_path / "docs" / "issues"
    issue_path = _write_issue(issues_dir, 1, labels=(labels_cfg.ready, labels_cfg.in_progress))
    before = issue_path.read_bytes()
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir, dry_run=True)
    state_file = tmp_path / "state.json"
    save_state(state_file, {})
    write_gate = _wg(state_file, dry_run=True)

    ok, error = park_unpublishable_work(
        gh,
        config,
        tmp_path,
        "agent/issue-1-local",
        1,
        {labels_cfg.in_progress},
        "session_exited",
        write_gate,
    )

    assert (ok, error) == (True, None)
    assert issue_path.read_bytes() == before, "dry-run must not touch the issue file at all"
    issue = gh.issue_view(1)
    assert issue["comments"] == []
    state = load_state(state_file)
    assert state["events"] == []


def test_park_unpublishable_work_label_write_failure(tmp_path: Path) -> None:
    """No issue file exists for the number being parked, so both the add and
    every remove in the ``local_work_ready`` edge fail against a real
    ``LocalFileGitHub`` -- no mocking needed to force the failure branch."""
    config = OrchestratorConfig()
    issues_dir = tmp_path / "docs" / "issues"
    issues_dir.mkdir(parents=True)
    gh = LocalFileGitHub(repo_root=tmp_path, issues_dir=issues_dir)
    state_file = tmp_path / "state.json"
    save_state(state_file, {})
    write_gate = _wg(state_file)

    ok, error = park_unpublishable_work(
        gh, config, tmp_path, "agent/issue-999-ghost", 999, set(), None, write_gate
    )

    assert ok is False
    assert error is not None and "label transition failed" in error

    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "local_work_ready"]
    assert len(events) == 1
    assert events[0]["payload"]["label_write_ok"] is False


# ---------------------------------------------------------------------------
# dead_worker_reap._attempt_salvage, end to end
# ---------------------------------------------------------------------------


def test_attempt_salvage_parks_local_backend_without_pushing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION CHECK target: if ``_attempt_salvage``'s
    ``park_unpublishable_work`` call (or its early-return-when-parked branch)
    were removed, this local-file-backend salvage would fall through to
    ``push_branch`` -- which is monkeypatched here to raise, so that
    regression fails this test loudly instead of silently attempting a push
    against a repo with no remote."""
    labels_cfg = LabelConfig()
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)

    branch = "agent/issue-1-local-work"
    worktree_path = tmp_path / "wt"
    subprocess.run(
        ["git", "worktree", "add", "-b", branch, str(worktree_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    (worktree_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "feature.txt"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", "feat: work"],
        cwd=worktree_path,
        check=True,
        capture_output=True,
        text=True,
    )

    config_file = repo_root / "orchestrator.config.yaml"
    config_file.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")
    config = load_config(config_file)

    issues_dir = repo_root / config.local_issues.issues_dir
    _write_issue(issues_dir, 1, labels=(labels_cfg.ready, labels_cfg.in_progress))
    gh = LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)

    state_file = tmp_path / "state.json"
    save_state(state_file, {})
    write_gate = _wg(state_file)

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("push_branch must not be called for a no-PR backend")

    monkeypatch.setattr(dead_worker_reap, "push_branch", _boom)

    ok, error = _attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo_root,
        worktree_path=worktree_path,
        branch=branch,
        base_ref="HEAD",
        issue_number=1,
        active_labels={labels_cfg.in_progress},
        issue_labels={labels_cfg.ready, labels_cfg.in_progress},
        state_file=state_file,
        failure_kind="session_exited",
        write_gate=write_gate,
    )

    assert (ok, error) == (True, None)

    issue = gh.issue_view(1)
    current_labels = {entry["name"] for entry in issue["labels"]}
    assert labels_cfg.review_ready in current_labels
    assert labels_cfg.ready in current_labels
    assert labels_cfg.in_progress not in current_labels
    assert len(issue["comments"]) == 1
    assert branch in issue["comments"][0]["body"]

    state = load_state(state_file)
    events = [e for e in state["events"] if e["kind"] == "local_work_ready"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["branch"] == branch
    assert payload["issue_number"] == 1
    assert payload["label_write_ok"] is True


def test_attempt_salvage_calls_push_branch_when_gh_publishes_prs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flip side of the control above: a gh double with no
    ``publishes_pull_requests`` attribute defaults to True (per
    ``local_work_park.publishes_pull_requests``'s documented default), so
    ``park_unpublishable_work`` must return ``None`` and ``_attempt_salvage``
    must fall through to ``push_branch`` -- proving the local-file
    short-circuit above is conditional on the capability flag, not
    unconditional."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    config = OrchestratorConfig()
    state_file = tmp_path / "state.json"
    save_state(state_file, {})
    write_gate = _wg(state_file)

    calls: list[tuple[Path, str]] = []

    def _fake_push_branch(
        repo_root_arg: Path,
        branch_arg: str,
        *,
        worktree_path: Path | None = None,
        dry_run: bool = False,
    ) -> tuple[bool, str | None]:
        calls.append((repo_root_arg, branch_arg))
        return False, "x"

    monkeypatch.setattr(dead_worker_reap, "push_branch", _fake_push_branch)

    ok, error = _attempt_salvage(
        gh=_PublishingGitHubDouble(),
        config=config,
        repo_root=repo_root,
        worktree_path=repo_root,
        branch="agent/issue-5-ghost",
        base_ref="HEAD",
        issue_number=5,
        active_labels=set(),
        issue_labels=set(),
        state_file=state_file,
        failure_kind=None,
        write_gate=write_gate,
    )

    assert calls, (
        "push_branch was never called -- park_unpublishable_work must have parked instead"
    )
    assert (ok, error) == (False, "x")
