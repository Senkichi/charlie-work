"""Shared local-lane test fixtures (a real git repo, no origin remote).

Extracted from ``test_local_lane.py`` so other test modules can reuse them
without a cross-test-module import (issue #2039).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from charlie_work.adapters import SessionDispatchResult, SessionRequest
from charlie_work.config import OrchestratorConfig, build_config_from_data
from charlie_work.labels import transition
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
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


def new_repo_root() -> Path:
    """A fresh repo root in the real system temp dir (see the note above)."""
    return Path(tempfile.mkdtemp(prefix="cw-lane-"))


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
    transition(app.gh, labels, issue_number, "local_work_ready", state_path=app.paths.state_file)
    _seed_issue_state(app, issue_number, branch=branch)


def rework_pending_app(repo: Path) -> OrchestratorApp:
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


def spy_dispatch_sessions(monkeypatch: pytest.MonkeyPatch) -> list[SessionRequest]:
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
