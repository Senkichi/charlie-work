"""Base selection for the collect-only gate (issue #2123)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _script_loader import load_script_module

REPO = Path(__file__).resolve().parents[1]
script = load_script_module(REPO / "scripts" / "collect_gate_base_sha.py", "collect_gate_base_sha")
WORKFLOW = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(repo: Path, name: str, body: str) -> None:
    (repo / name).write_text(body, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", f"add {name}")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _commit(tmp_path, "t.py", "def test_a(): pass\ndef test_b(): pass\n")
    return tmp_path


def test_base_is_merge_ref_first_parent_not_stale_pr_base(repo: Path) -> None:
    stale_base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "switch", "-c", "pr")
    _commit(repo, "pr.py", "def test_pr(): pass\n")
    _git(repo, "switch", "main")
    # Another PR lands on main after "pr" branched and removes a leaf.
    _commit(repo, "t.py", "def test_a(): pass\n")
    base_tip = _git(repo, "rev-parse", "HEAD")
    # Simulate actions/checkout of refs/pull/N/merge.
    _git(repo, "switch", "--detach", "main")
    _git(repo, "merge", "--no-ff", "-m", "merge", "pr")

    resolved = script.merge_base_parent(str(repo))

    assert resolved == base_tip
    assert resolved != stale_base
    assert "def test_b" not in _git(repo, "show", f"{resolved}:t.py")


def test_non_merge_head_fails_loudly(repo: Path) -> None:
    with pytest.raises(script.NotAMergeCommitError, match="expected exactly 2"):
        script.merge_base_parent(str(repo))


def test_main_exit_code_and_empty_stdout_on_non_merge(repo: Path, capsys) -> None:
    assert script.main(["x", str(repo)]) == 1
    assert capsys.readouterr().out == ""


def test_workflow_collects_base_at_merge_ref_parent() -> None:
    job = WORKFLOW[WORKFLOW.index("  collect-only-gate:") :]
    assert "fetch-depth: 2" in job
    assert "scripts/collect_gate_base_sha.py" in job
    assert "git worktree add ./base-tree ${{ github.event.pull_request.base.sha }}" not in job
