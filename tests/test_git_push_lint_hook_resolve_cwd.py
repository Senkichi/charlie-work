"""Tests for ``scripts/git_push_lint_hook.py``'s cwd resolution (#1468).

Split out of ``tests/test_git_push_lint_hook.py`` for the Track-1 shoulder
(issue #1577): that module was a saturated ``test_module`` attachment point
(42 bound members vs the live 41.0 boundary). This sibling carries the
cwd-resolution seam -- ``_resolve_cwd`` plus the ``_leading_cd_target``
parser it delegates to: payload.cwd fallback, the ``cd <path> <sep>`` prefix
override, and the fail-closed fallbacks for relative / ``~`` / underivable
cd targets. The ``main()``-level integration tests for the same override
stayed with the ``main()`` seam in ``tests/test_git_push_lint_hook.py``.
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
# _resolve_cwd
# ---------------------------------------------------------------------------


def test_resolve_cwd_uses_payload_cwd(hook: ModuleType, tmp_path: Path) -> None:
    # No leading `cd` prefix -> fall back to payload.cwd.
    result = hook._resolve_cwd({"cwd": str(tmp_path)}, "git push")
    assert result == tmp_path


def test_resolve_cwd_falls_back_to_path_cwd(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = hook._resolve_cwd({}, "git push")
    assert result == tmp_path


def test_resolve_cwd_ignores_non_string_cwd(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = hook._resolve_cwd({"cwd": 123}, "git push")
    assert result == tmp_path


def test_resolve_cwd_no_command_falls_back_to_payload_cwd(
    hook: ModuleType, tmp_path: Path
) -> None:
    # Backward-compat: command defaults to "" -> payload.cwd wins.
    result = hook._resolve_cwd({"cwd": str(tmp_path)})
    assert result == tmp_path


# ---------------------------------------------------------------------------
# _resolve_cwd -- leading `cd <path> <sep>` prefix overrides payload.cwd
# (regression for #1468: a subagent whose Bash tool cwd resets to the
# session default is hooked with payload.cwd pointing at that default, not
# the worktree the command text `cd`s into).
# ---------------------------------------------------------------------------


def test_resolve_cwd_cd_prefix_overrides_payload_cwd(hook: ModuleType, tmp_path: Path) -> None:
    other = tmp_path / "worktree"
    other.mkdir()
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    # Use the posix form of the path inside the command -- the real repro
    # (`cd "C:/Users/.../worktree" && git push`) uses forward slashes, and
    # posix shlex strips unquoted backslashes (a pre-existing limitation
    # shared with `_is_git_push`'s tokenizer).
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, f"cd {other.as_posix()} && git push")
    assert result == other


def test_resolve_cwd_cd_prefix_semicolon_separator(hook: ModuleType, tmp_path: Path) -> None:
    other = tmp_path / "worktree"
    other.mkdir()
    result = hook._resolve_cwd({}, f"cd {other.as_posix()} ; git push")
    assert result == other


def test_resolve_cwd_cd_prefix_quoted_path_with_spaces(hook: ModuleType, tmp_path: Path) -> None:
    other = tmp_path / "path with spaces"
    other.mkdir()
    result = hook._resolve_cwd({}, f'cd "{other.as_posix()}" && git push')
    assert result == other


def test_resolve_cwd_cd_chain_takes_last_target(hook: ModuleType, tmp_path: Path) -> None:
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    result = hook._resolve_cwd({}, f"cd {a.as_posix()} && cd {b.as_posix()} && git push")
    assert result == b


def test_resolve_cwd_cd_dash_falls_back(hook: ModuleType, tmp_path: Path) -> None:
    # `cd -` (previous directory) -- target not derivable, fall back to
    # payload.cwd to preserve fail-closed.
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "cd - && git push")
    assert result == payload_cwd


def test_resolve_cwd_cd_no_path_falls_back(hook: ModuleType, tmp_path: Path) -> None:
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    # `cd` with no path argument -- ambiguous, fall back.
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "cd && git push")
    assert result == payload_cwd


def test_resolve_cwd_cd_without_separator_falls_back(hook: ModuleType, tmp_path: Path) -> None:
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    # `cd /x git push` (no separator) is not a `cd <path> <sep>` form --
    # fall back rather than guessing.
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "cd /x git push")
    assert result == payload_cwd


def test_resolve_cwd_non_cd_first_token_uses_payload_cwd(hook: ModuleType, tmp_path: Path) -> None:
    # `git -C /x push` does NOT start with `cd` -- payload.cwd wins. The
    # `-C` form is a separate (hypothetical) vector not in this fix's
    # scope; the regression is the `cd <worktree> && git push` form.
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "git -C /x push")
    assert result == payload_cwd


# ---------------------------------------------------------------------------
# _resolve_cwd -- non-absolute cd targets fall back to payload.cwd
# (rework for #1468: the owner's fail-closed-on-ambiguity mandate applied
# to relative and ~-prefixed cd targets, not just the `cd -`/`cd -P` forms
# `_leading_cd_target` already rejects).
# ---------------------------------------------------------------------------


def test_resolve_cwd_relative_cd_target_falls_back_to_payload_cwd(
    hook: ModuleType, tmp_path: Path
) -> None:
    # `cd ../other && git push` -- a relative target resolves against a
    # base directory the hook cannot know (it is spawned at a fixed
    # project-root cwd while payload.cwd is dynamic), so trusting it
    # silently reintroduces #1468's own defect class. Fall back to
    # payload.cwd rather than guessing the base.
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "cd ../other && git push")
    assert result == payload_cwd


def test_resolve_cwd_tilde_cd_target_falls_back_to_payload_cwd(
    hook: ModuleType, tmp_path: Path
) -> None:
    # `cd ~/x && git push` -- shlex never expands `~`, so `Path("~/x")`
    # is a nonexistent path whose lookup surfaces as a spurious
    # fail-closed deny. Fall back to payload.cwd.
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "cd ~/x && git push")
    assert result == payload_cwd


def test_resolve_cwd_bare_tilde_cd_target_falls_back_to_payload_cwd(
    hook: ModuleType, tmp_path: Path
) -> None:
    # `cd ~ && git push` -- same as ~/x: `Path("~")` is not absolute and
    # not expanded by shlex, so fall back.
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()
    result = hook._resolve_cwd({"cwd": str(payload_cwd)}, "cd ~ && git push")
    assert result == payload_cwd


# ---------------------------------------------------------------------------
# _leading_cd_target -- direct unit tests for the parser.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("cd /repo && git push", Path("/repo")),
        ("cd /repo ; git push", Path("/repo")),
        ("cd /repo & git push", Path("/repo")),
        ("cd /a && cd /b && git push", Path("/b")),
        ('cd "/path with spaces" && git push', Path("/path with spaces")),
        ("cd /repo && git push origin main", Path("/repo")),
    ],
)
def test_leading_cd_target_parses(hook: ModuleType, command: str, expected: Path) -> None:
    assert hook._leading_cd_target(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "",
        "git push",
        "git -C /repo push",
        "cd - && git push",
        "cd -P /repo && git push",
        "cd && git push",
        "cd /repo git push",  # no separator
        "echo cd /repo && git push",  # cd not in command position
        'git push "unterminated',  # unparseable -> None
    ],
)
def test_leading_cd_target_returns_none(hook: ModuleType, command: str) -> None:
    assert hook._leading_cd_target(command) is None
