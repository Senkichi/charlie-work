"""Shared local-park scaffold for the issue-#1971 dead-dispatched backstop tests.

Hoisted out of ``test_orphan_sweep_backstop_local_park.py`` when
``test_local_park_deferral_bounds.py`` needed the same repo + state
builders: the no-cross-test-import guard
(``tests/test_zero_cross_test_import_guard.py``, issue #1284) requires
shared helpers to live in a ``tests/_*.py`` module, never a
``test_*.py -> test_*.py`` import.

The helpers build the pieces every test in both files needs: a fresh git
repo with no origin, a ``local_issues`` config whose worktrees root sits
outside ``tmp_path``'s deep nesting, a dead ``dispatched`` state entry
whose drift marker is already past the ``dead_dispatched_reap_minutes``
backstop, and thin wrappers to run the orphan sweep and read back state
events / label names.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _host_fixtures import host_probe

from charlie_work.config import OrchestratorConfig, load_config
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import resolved_layout
from charlie_work.state import load_state, save_state
from charlie_work.worktree import worktree_path_for_branch
from charlie_work.write_gate import WriteGate


def _git(repo_root: Path, *args: str, cwd: Path | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd or repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote."""
    repo_root.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "core.longpaths", "true")
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "--allow-empty",
        "-m",
        "chore: seed",
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


# Explicit fixture name: importing modules bring the (private) symbol into
# their namespace so pytest registers ``shallow_wts`` there — a public
# function name would collide with the fixture parameter in test
# signatures (ruff F811).
@pytest.fixture(name="shallow_wts")
def _shallow_wts(tmp_path: Path) -> Iterator[Path]:
    """A worktrees root OUTSIDE ``tmp_path``'s deep nesting (see
    test_orphan_sweep_local_park.py for why)."""
    root = Path(tempfile.mkdtemp(prefix="cwwt1971-"))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _local_config(repo_root: Path, worktrees_dir: Path) -> OrchestratorConfig:
    config_file = repo_root / "orchestrator.config.yaml"
    config_file.write_text(
        "local_issues:\n"
        "  enabled: true\n"
        "claude_code:\n"
        f'  worktrees_dir: "{worktrees_dir.as_posix()}"\n',
        encoding="utf-8",
    )
    return load_config(config_file)


def _add_worktree_commit(repo_root: Path, config: OrchestratorConfig, branch: str) -> Path:
    worktrees_dir = resolved_layout(config, repo_root).worktrees
    worktree_path = worktree_path_for_branch(repo_root, branch, worktrees_dir)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo_root, "worktree", "add", "-b", branch, str(worktree_path))
    (worktree_path / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(repo_root, "add", "feature.txt", cwd=worktree_path)
    _git(
        repo_root,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-m",
        "feat: work",
        cwd=worktree_path,
    )
    return worktree_path


def _seed_dead_dispatched(
    state_file: Path,
    issue_number: int,
    branch: str,
    *,
    armed_drift_minutes_ago: int = 120,
) -> None:
    """A dead dispatched entry whose drift armed long enough ago that the
    ``dead_dispatched_reap_minutes`` backstop (default 60 min) is due to
    escalate it on this very pass."""
    state = load_state(state_file)
    old = (
        (datetime.now(UTC) - timedelta(minutes=armed_drift_minutes_ago))
        .isoformat()
        .replace("+00:00", "Z")
    )
    state["issues"][str(issue_number)] = {
        "status": "dispatched",
        "worker_pid": 424242,
        "worker_process_start_time": 1700000000.0,
        "dispatched_at": "2026-09-17T00:00:00Z",
        "branch_name": branch,
        "orphan_flagged_at": old,
        "orphan_drift_at": old,
        "orphan_drift_fingerprint": '{"dead": true}',
    }
    save_state(state_file, state)


def _run_sweep(
    sessions_dir: Path,
    state_file: Path,
    config: OrchestratorConfig,
    gh,
    write_gate: WriteGate,
    fleet_dir: Path,
    monkeypatch,
) -> None:
    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    with host_probe(monkeypatch, alive=False):
        _detect_and_handle_orphaned_workers(
            sessions_dir,
            state_file,
            config,
            gh,
            write_gate=write_gate,
            fleet_dir_override=str(fleet_dir),
        )


def _events(state_file: Path, kind: str, issue_number: int | None = None):
    state = load_state(state_file)
    out = []
    for event in state.get("events", []):
        if event.get("kind") != kind:
            continue
        if (
            issue_number is not None
            and event.get("payload", {}).get("issue_number") != issue_number
        ):
            continue
        out.append(event)
    return out


def _label_names(gh: LocalFileGitHub, issue_number: int) -> set[str]:
    return {entry["name"] for entry in gh.issue_view(issue_number)["labels"]}


def _sessions_dir(tmp_path: Path) -> Path:
    sessions_dir = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir
