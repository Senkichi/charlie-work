"""Tests for ``scripts/git_push_lint_hook.py``'s ``_is_git_push`` detector (#1309).

Split out of ``tests/test_git_push_lint_hook.py`` for the Track-1 shoulder
(issue #1577): that module was a saturated ``test_module`` attachment point
(42 bound members vs the live 41.0 boundary). This sibling carries the
``_is_git_push`` seam -- the token-based git-push detection the push hook
contributes on top of ``worker_stop_gate``'s reused machinery.

The detector is exercised against real shlex output (no mocking) -- the
token-based parsing is the push hook's own contribution and must be correct
against real shell syntax.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from _script_loader import load_script_module

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = REPO_ROOT / "scripts" / "git_push_lint_hook.py"


def _load_module() -> ModuleType:
    return load_script_module(_SCRIPT_PATH, "git_push_lint_hook_under_test")


@pytest.fixture()
def hook(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A fresh module instance per test."""
    return _load_module()


# ---------------------------------------------------------------------------
# _is_git_push -- positive cases (must fire).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git push",
        "git push origin main",
        "git push --force",
        "git push --force-with-lease origin agent/issue-1309",
        "git push -u origin my-branch",
        "git push --tags",
        "git -C /repo push",
        "git -c core.quotePath=false push",
        "git --git-dir=/repo/.git push",
        "git --git-dir /repo/.git push",
        "git --work-tree=/repo push",
        "cd /repo && git push",
        "cd /repo && git push origin main",
        "git status; git push",
        "git fetch && git push",
        "GIT_PAGER=cat git push",
        "env GIT_PAGER=cat git push",
        "/usr/bin/git push",
        "/usr/bin/git push origin main",
        "git.exe push",
        "git push origin HEAD:refs/heads/main",
    ],
)
def test_is_git_push_positive(hook: ModuleType, command: str) -> None:
    assert hook._is_git_push(command) is True


# ---------------------------------------------------------------------------
# _is_git_push -- negative cases (must NOT fire).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git commit -m 'feat: add thing'",
        "git commit -m 'git push'",
        "git fetch",
        "git fetch origin",
        "git pull",
        "git pull origin main",
        "git add -A",
        "git log --oneline",
        "git diff",
        "git diff --cached",
        "git remote add origin git@github.com:foo/bar.git",
        "git remote -v",
        "git branch",
        "git checkout main",
        "git merge-base HEAD origin/main",
        "git rev-parse --show-toplevel",
        "git config user.email test@example.com",
        "echo git push",
        'echo "git push"',
        "echo 'running git push now'",
        "git status; echo git push",
        "git-push",
        "git-push --force",
        "gitlog push",
        "gh pr merge 123",
        "ruff check .",
        "uv run pytest",
        "",
        "   ",
    ],
)
def test_is_git_push_negative(hook: ModuleType, command: str) -> None:
    assert hook._is_git_push(command) is False


# ---------------------------------------------------------------------------
# _is_git_push -- quoted-mention trap (the merge_preflight_hook's first
# version fell into this; the push hook must not).
# ---------------------------------------------------------------------------


def test_is_git_push_quoted_mention_in_commit_message(hook: ModuleType) -> None:
    command = 'git commit -m "fix: handle git push failures"'
    assert hook._is_git_push(command) is False


def test_is_git_push_heredoc_does_not_swallow_subsequent_command(hook: ModuleType) -> None:
    # shlex does not understand heredocs, but the quoted body is at least
    # tokenized as individual words. The key property: `git commit` is the
    # first git invocation, and its subcommand is `commit`, not `push`.
    command = "git commit -m 'docs: mention git push' && git push"
    assert hook._is_git_push(command) is True


# ---------------------------------------------------------------------------
# _is_git_push -- unparseable command (unbalanced quotes) falls back to regex.
# ---------------------------------------------------------------------------


def test_is_git_push_unbalanced_quotes_falls_back_to_regex(hook: ModuleType) -> None:
    # shlex raises ValueError on unbalanced quotes; the regex fallback
    # matches `git push` in the raw text. This is less precise (it can
    # match inside quoted strings) but unparseable input is rare and the
    # Stop gate still backstops.
    assert hook._is_git_push('git push "unterminated') is True
    assert hook._is_git_push('git status "unterminated') is False
