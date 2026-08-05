"""Tests for the advisory PreToolUse hook that wires ``charlie merge-check``.

Issue #894. The hook lives in ``.claude/hooks/merge_check.py`` and is registered
in ``.claude/settings.json``. It runs on every Bash tool call, but only acts on
``gh pr merge`` commands for PRs whose head branch matches the configured worker
prefix. The hook is advisory: it always emits ``permissionDecision: allow`` and
surfaces the ``merge-check`` verdict in ``systemMessage``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK_SCRIPT = REPO_ROOT / ".claude" / "hooks" / "merge_check.py"
SETTINGS_FILE = REPO_ROOT / ".claude" / "settings.json"


def _run_hook(
    stdin_payload: dict,
    *,
    env: dict[str, str] | None = None,
    project_dir: Path | None = None,
) -> tuple[int, str, str]:
    """Run the hook script with the given stdin and environment.

    Returns ``(returncode, stdout, stderr)``. The hook is always expected to
    exit 0 because it is advisory-only.
    """
    full_env = {**os.environ, **(env or {})}
    if project_dir is not None:
        full_env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    result = subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(stdin_payload),
        text=True,
        capture_output=True,
        encoding="utf-8",
        env=full_env,
    )
    return result.returncode, result.stdout, result.stderr


def _make_fake_gh(tmp_path: Path, head_ref: str) -> Path:
    """Create a fake ``gh`` executable that returns the given ``headRefName``."""
    script = tmp_path / "fake_gh.py"
    script.write_text(
        "import json, sys\n"
        "pr = next((a for a in sys.argv[1:] if a.isdigit()), '0')\n"
        f"print(json.dumps({{'headRefName': {head_ref!r}}}))\n",
        encoding="utf-8",
    )
    return script


def _make_fake_charlie(
    tmp_path: Path,
    *,
    ok: bool,
    reason: str = "approved_at_head",
) -> Path:
    """Create a fake ``charlie`` executable that returns a merge-check verdict."""
    script = tmp_path / "fake_charlie.py"
    script.write_text(
        "import json, sys\n"
        "pr = next((a for a in sys.argv[1:] if a.isdigit()), '0')\n"
        f"ok = {ok}\n"
        f"reason = {reason!r}\n"
        "msg = f'PR #{pr}: ' + ('approved at head' if ok else 'not authorized')\n"
        "data = {'authorized': ok, 'reason': reason}\n"
        "print(json.dumps({'ok': ok, 'message': msg, 'data': data}))\n",
        encoding="utf-8",
    )
    return script


def _parse_hook_output(stdout: str) -> dict:
    """Return the last JSON object printed by the hook."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON found in hook stdout: {stdout!r}")


@pytest.fixture
def base_env(tmp_path: Path) -> dict[str, str]:
    """Environment with fake ``gh`` and ``charlie`` binaries.

    The fake ``gh`` returns a worker branch by default; tests that want a
    non-worker branch override ``FAKE_GH_HEAD`` before creating the fake or set
    it directly. The fake ``charlie`` is not used until ``FAKE_CHARLIE_OK`` is
    set.
    """
    gh = _make_fake_gh(tmp_path, os.environ.get("FAKE_GH_HEAD", "agent/issue-123-fix"))
    charlie = _make_fake_charlie(
        tmp_path,
        ok=os.environ.get("FAKE_CHARLIE_OK", "true").lower() == "true",
        reason=os.environ.get("FAKE_CHARLIE_REASON", "approved_at_head"),
    )
    return {
        "GH_BIN": str(gh),
        "CHARLIE_BIN": str(charlie),
    }


def test_settings_file_exists_and_registers_bash_hook() -> None:
    """The hook must be declared in ``.claude/settings.json``."""
    assert SETTINGS_FILE.exists()
    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    assert "PreToolUse" in settings
    bash_hooks = [h for h in settings["PreToolUse"] if h.get("matcher") == "Bash"]
    assert bash_hooks
    command = bash_hooks[0]["hooks"][0]["command"]
    assert ".claude/hooks/merge_check.py" in command


def test_hook_allows_non_bash_tools() -> None:
    """Only Bash tool calls are inspected."""
    output = _run_hook({"tool_name": "Read", "tool_input": {"file_path": "x"}})
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "systemMessage" not in result


def test_hook_allows_non_merge_bash_commands() -> None:
    """Ordinary Bash commands pass through without a message."""
    output = _run_hook({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "systemMessage" not in result


def test_hook_skips_non_worker_branch(tmp_path: Path, base_env: dict[str, str]) -> None:
    """PRs on non-worker branches are not checked, so legitimate direct merges stay quiet."""
    gh = _make_fake_gh(tmp_path, "fix/123")
    env = {**base_env, "GH_BIN": str(gh)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 759 --squash"}},
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "skipping" in (result.get("systemMessage") or "").lower()


def test_hook_uses_config_branch_prefix(tmp_path: Path, base_env: dict[str, str]) -> None:
    """The worker prefix is read from config, not hardcoded."""
    config_dir = tmp_path / "custom-repo"
    config_dir.mkdir()
    (config_dir / "orchestrator.config.yaml").write_text(
        "dispatch:\n  branch_prefix: custom/prefix\n",
        encoding="utf-8",
    )
    gh = _make_fake_gh(config_dir, "custom/prefix-123-fix")
    charlie = _make_fake_charlie(tmp_path, ok=True, reason="approved_at_head")
    env = {
        **base_env,
        "GH_BIN": str(gh),
        "CHARLIE_BIN": str(charlie),
    }
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 456"}},
        env=env,
        project_dir=config_dir,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "authorized" in (result.get("systemMessage") or "").lower()


def test_hook_surfaces_unauthorized_worker_pr(tmp_path: Path, base_env: dict[str, str]) -> None:
    """A worker PR that fails ``merge-check`` is surfaced but still allowed."""
    gh = _make_fake_gh(tmp_path, "agent/issue-593-fix")
    charlie = _make_fake_charlie(tmp_path, ok=False, reason="no_decision")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(charlie)}
    output = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "gh pr merge 759 --squash --delete-branch"},
        },
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    msg = result.get("systemMessage", "")
    assert "NOT authorized" in msg
    assert "no_decision" in msg


def test_hook_surfaces_authorized_worker_pr(tmp_path: Path, base_env: dict[str, str]) -> None:
    """A worker PR that passes ``merge-check`` is surfaced as authorized."""
    gh = _make_fake_gh(tmp_path, "agent/issue-894-fix")
    charlie = _make_fake_charlie(tmp_path, ok=True, reason="approved_at_head")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(charlie)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 894"}},
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    msg = result.get("systemMessage", "")
    assert "authorized" in msg.lower()
    assert "not authorized" not in msg.lower()


def test_hook_allows_unparseable_pr_number() -> None:
    """A ``gh pr merge`` with no parseable PR number is allowed with a note."""
    output = _run_hook({"tool_name": "Bash", "tool_input": {"command": "gh pr merge some-branch"}})
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "could not parse" in (result.get("systemMessage") or "").lower()


def test_hook_allows_when_gh_fails(tmp_path: Path, base_env: dict[str, str]) -> None:
    """If ``gh pr view`` fails, the hook is fail-open at the advisory level."""
    script = tmp_path / "fake_gh_fail.py"
    script.write_text(
        "import sys\nprint('no such pull request', file=sys.stderr)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    env = {**base_env, "GH_BIN": str(script)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 999"}},
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "could not read" in (result.get("systemMessage") or "").lower()
