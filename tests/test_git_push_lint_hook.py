"""Tests for ``scripts/git_push_lint_hook.py`` (#1309).

The W2 follow-up PreToolUse hook that lints before ``git push`` leaves the
machine. Reuses ``worker_stop_gate``'s changed-set derivation and scoped-ruff
machinery -- the tests here verify the push hook's own logic (git-push
detection, deny/allow decisions, fail-open/fail-closed contract), not the
reused machinery (which has its own coverage in ``test_worker_stop_gate.py``).

``_load_stop_gate`` is monkeypatched to return a lightweight stand-in module
so no real ``git``/``ruff`` subprocesses run. The ``_is_git_push`` detector is
exercised against real shlex output (no mocking) -- the token-based parsing is
the push hook's own contribution and must be correct against real shell syntax.
"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _stdin(payload: dict[str, Any]) -> io.StringIO:
    return io.StringIO(json.dumps(payload))


def _bash_payload(command: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"tool_name": "Bash", "tool_input": {"command": command}}
    payload.update(extra)
    return payload


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


# ---------------------------------------------------------------------------
# main() -- non-push and non-Bash paths (must allow / leave undecided).
# ---------------------------------------------------------------------------


def test_main_non_bash_tool_allows(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(hook.sys, "stdin", _stdin({"tool_name": "edit", "tool_input": {}}))
    rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_bash_non_push_allows(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git status")))
    rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_malformed_stdin_allows(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO("not json"))
    rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_empty_stdin_allows(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(""))
    rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_non_dict_tool_input_allows(
    hook: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        _stdin({"tool_name": "Bash", "tool_input": "not a dict"}),
    )
    rc = hook.main()
    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# main() -- push + lint check (subprocess mocked via _load_stop_gate).
# ---------------------------------------------------------------------------


def _fake_stop_gate(
    *,
    changed: tuple[Any, ...] = (),
    ruff_block: bool = False,
    ruff_reason: str = "",
    repo_root_path: Path | None = None,
) -> ModuleType:
    """A minimal stand-in for worker_stop_gate with just the functions
    the push hook calls."""
    import types

    fake = types.ModuleType("fake_stop_gate")

    class _ChangedFile:
        def __init__(self, path: str, deleted: bool = False) -> None:
            self.path = path
            self.deleted = deleted

    class _GateResult:
        def __init__(self, block: bool, reason: str = "") -> None:
            self.block = block
            self.reason = reason

    fake.ChangedFile = _ChangedFile  # type: ignore[attr-defined]
    fake.GateResult = _GateResult  # type: ignore[attr-defined]

    def _repo_root(cwd: Path) -> Path:
        return repo_root_path or cwd

    def _all_changed_files(repo_root: Path) -> tuple[Any, ...]:
        return changed

    def _run_ruff(repo_root: Path, py_files: tuple[str, ...]) -> Any:
        return _GateResult(block=ruff_block, reason=ruff_reason)

    fake._repo_root = _repo_root  # type: ignore[attr-defined]
    fake._all_changed_files = _all_changed_files  # type: ignore[attr-defined]
    fake._run_ruff = _run_ruff  # type: ignore[attr-defined]
    return fake


def test_main_push_no_changes_allows(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: _fake_stop_gate(changed=()))
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_push_no_py_files_allows(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = _fake_stop_gate(
        changed=(type("CF", (), {"path": "notes.txt", "deleted": False})(),),
    )
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_push_ruff_check_failure_denies(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = _fake_stop_gate(
        changed=(type("CF", (), {"path": "src/dirty.py", "deleted": False})(),),
        ruff_block=True,
        ruff_reason="ruff check failed:\nE501 line too long",
    )
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0  # blocking is via JSON, not exit code
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "ruff check failed" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_push_ruff_format_failure_denies(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = _fake_stop_gate(
        changed=(type("CF", (), {"path": "src/dirty.py", "deleted": False})(),),
        ruff_block=True,
        ruff_reason="ruff format --check failed:\nwould reformat dirty.py",
    )
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "ruff format" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_push_ruff_passes_allows(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = _fake_stop_gate(
        changed=(type("CF", (), {"path": "src/clean.py", "deleted": False})(),),
        ruff_block=False,
    )
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


def test_main_push_deleted_py_file_skipped_allows(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    # A deleted .py file must not be in the ruff scope -- the same
    # behavior as worker_stop_gate's _evaluate (cf.deleted is filtered).
    fake = _fake_stop_gate(
        changed=(type("CF", (), {"path": "src/gone.py", "deleted": True})(),),
    )
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


# ---------------------------------------------------------------------------
# main() -- fail-closed on internal error during a confirmed push.
# ---------------------------------------------------------------------------


def test_main_push_repo_root_error_fails_closed(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    def _boom():
        raise RuntimeError("git binary missing")

    monkeypatch.setattr(hook, "_load_stop_gate", _boom)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0  # blocking is via JSON, not exit code
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "failing closed" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_push_all_changed_files_error_fails_closed(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = _fake_stop_gate()

    def _boom(repo_root: Path) -> tuple[Any, ...]:
        raise RuntimeError("git status failed")

    fake._all_changed_files = _boom  # type: ignore[attr-defined]
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "failing closed" in out["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_push_ruff_subprocess_error_fails_closed(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    fake = _fake_stop_gate(
        changed=(type("CF", (), {"path": "src/dirty.py", "deleted": False})(),),
    )

    def _boom(repo_root: Path, py_files: tuple[str, ...]) -> Any:
        raise RuntimeError("ruff crashed")

    fake._run_ruff = _boom  # type: ignore[attr-defined]
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(tmp_path))))

    rc = hook.main()

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "failing closed" in out["hookSpecificOutput"]["permissionDecisionReason"]


# ---------------------------------------------------------------------------
# Fail-open fallback: branch-base ambiguity narrows to working-tree-only
# (inherited from worker_stop_gate._committed_diff_files, not re-implemented).
# This test verifies the integration: the push hook does NOT block when
# the committed-diff surface cannot be derived -- it silently narrows to
# working-tree-only scope, same as the Stop gate.
# ---------------------------------------------------------------------------


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def test_main_push_detached_head_fail_open_narrows_to_working_tree(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """On a detached HEAD (no branch to define 'diverged from'),
    _committed_diff_files returns () -- the push hook must not block on
    the ambiguity, only on actual ruff failures in the working-tree scope."""
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(["init"], cwd=root)
    _run_git(["config", "user.email", "test@example.com"], cwd=root)
    _run_git(["config", "user.name", "Test"], cwd=root)
    (root / "README.md").write_text("placeholder\n", encoding="utf-8")
    _run_git(["add", "README.md"], cwd=root)
    _run_git(["commit", "-m", "init"], cwd=root)
    # Detach HEAD so _committed_diff_files short-circuits to ().
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    _run_git(["checkout", "--detach", head_sha], cwd=root)
    # Add a clean .py file to the working tree -- ruff should pass.
    (root / "clean.py").write_text("x = 1\n", encoding="utf-8")

    # Load the REAL worker_stop_gate (not a fake) to verify the fail-open
    # integration. Monkeypatch _run_ruff to avoid shelling out to ruff.
    real_gate = hook._load_stop_gate()
    monkeypatch.setattr(
        real_gate, "_run_ruff", lambda repo_root, py_files: real_gate.GateResult(block=False)
    )
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: real_gate)
    monkeypatch.setattr(hook.sys, "stdin", _stdin(_bash_payload("git push", cwd=str(root))))

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""


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


# ---------------------------------------------------------------------------
# main() -- the cd-prefix override scopes the lint check to the cd target,
# not payload.cwd (the issue's specified regression test).
# ---------------------------------------------------------------------------


def test_main_push_cd_prefix_scopes_lint_to_cd_target(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """A ``cd <other-repo> && git push`` command scopes the lint check to
    ``<other-repo>``, not ``payload.cwd`` (#1468). Verifies the cwd handed
    to ``worker_stop_gate._repo_root`` is the cd target."""
    other = tmp_path / "worktree"
    other.mkdir()
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()

    captured: list[Path] = []

    def _capture_repo_root(cwd: Path) -> Path:
        captured.append(cwd)
        return cwd

    fake = _fake_stop_gate(changed=())
    fake._repo_root = _capture_repo_root  # type: ignore[attr-defined]
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        _stdin(_bash_payload(f"cd {other.as_posix()} && git push", cwd=str(payload_cwd))),
    )

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
    assert captured == [other], (
        f"expected _repo_root resolved from cd prefix to {other}, got {captured}"
    )


def test_main_push_no_cd_uses_payload_cwd(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Without a cd prefix, payload.cwd is used (the pre-fix behavior)."""
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()

    captured: list[Path] = []

    def _capture_repo_root(cwd: Path) -> Path:
        captured.append(cwd)
        return cwd

    fake = _fake_stop_gate(changed=())
    fake._repo_root = _capture_repo_root  # type: ignore[attr-defined]
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        _stdin(_bash_payload("git push", cwd=str(payload_cwd))),
    )

    rc = hook.main()

    assert rc == 0
    assert captured == [payload_cwd]


def test_main_push_tilde_cd_target_falls_back_to_payload_cwd(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """``cd ~/x && git push`` falls back to ``payload.cwd`` rather than
    producing a GateError-driven deny (#1468 rework). ``shlex`` never
    expands ``~``, so ``Path("~/x")`` is a nonexistent path; trusting it
    would hand ``_repo_root`` a path with no repo, surfacing as a spurious
    fail-closed deny. Instead ``payload.cwd`` wins and the lint check
    proceeds normally (here: no changes -> allow)."""
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()

    captured: list[Path] = []

    def _capture_repo_root(cwd: Path) -> Path:
        captured.append(cwd)
        return cwd

    fake = _fake_stop_gate(changed=())
    fake._repo_root = _capture_repo_root  # type: ignore[attr-defined]
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        _stdin(_bash_payload("cd ~/x && git push", cwd=str(payload_cwd))),
    )

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == "", (
        "tilde cd target must not produce a deny -- it falls back to payload.cwd"
    )
    assert captured == [payload_cwd], (
        f"expected _repo_root resolved from payload.cwd to {payload_cwd}, got {captured}"
    )


def test_main_push_relative_cd_target_falls_back_to_payload_cwd(
    hook: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """``cd ../other && git push`` falls back to ``payload.cwd`` (#1468
    rework). A relative target resolves against an unknown base (the
    hook's spawn cwd, not ``payload.cwd``); trusting it would scope the
    lint check to the wrong repo. ``payload.cwd`` wins instead."""
    payload_cwd = tmp_path / "session-default"
    payload_cwd.mkdir()

    captured: list[Path] = []

    def _capture_repo_root(cwd: Path) -> Path:
        captured.append(cwd)
        return cwd

    fake = _fake_stop_gate(changed=())
    fake._repo_root = _capture_repo_root  # type: ignore[attr-defined]
    monkeypatch.setattr(hook, "_load_stop_gate", lambda: fake)
    monkeypatch.setattr(
        hook.sys,
        "stdin",
        _stdin(_bash_payload("cd ../other && git push", cwd=str(payload_cwd))),
    )

    rc = hook.main()

    assert rc == 0
    assert capsys.readouterr().out.strip() == ""
    assert captured == [payload_cwd]


# ---------------------------------------------------------------------------
# .claude/settings.json wiring.
# ---------------------------------------------------------------------------


def test_claude_settings_wires_git_push_lint_hook() -> None:
    settings_path = REPO_ROOT / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    pre_tool_use = settings["hooks"]["PreToolUse"]

    # The Bash matcher entry must contain the git push lint hook.
    bash_entry = None
    for entry in pre_tool_use:
        if entry["matcher"] == "Bash":
            bash_entry = entry
            break
    assert bash_entry is not None, "no Bash matcher in PreToolUse hooks"

    commands = [h["command"] for h in bash_entry["hooks"]]
    push_hook_cmds = [c for c in commands if "git_push_lint_hook" in c]
    assert len(push_hook_cmds) == 1, (
        f"expected exactly one git_push_lint_hook command, got {push_hook_cmds}"
    )
    assert push_hook_cmds[0].strip().endswith("|| true"), (
        "git push lint hook must fail open on crash"
    )
    assert "scripts/git_push_lint_hook.py" in push_hook_cmds[0]
