"""Tests for the enforcing PreToolUse hook that wires ``charlie merge-check``.

Issue #894. The hook lives in ``.claude/hooks/merge_check.py`` and is registered
in ``.claude/settings.json``. It runs on every Bash tool call, but only acts on
``gh pr merge`` commands for PRs whose head branch matches the configured worker
prefix. For worker PRs the hook enforces ``merge-check``: an unauthorized PR is
denied. Infrastructure failures outside the ``merge-check`` verdict itself are
fail-open and allowed with a note; an unparseable ``merge-check`` result for a
worker PR is denied, matching the command's deliberate fail-closed contract.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
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

    Returns ``(returncode, stdout, stderr)``. The hook always exits 0; the
    ``permissionDecision`` field inside the JSON output carries the verdict.
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
    """Create a fake ``gh`` executable that returns the given ``headRefName``.

    The fake supports ``gh pr view --json number,headRefName`` (with or without
    a positional selector) and ``gh pr merge --help`` (delegated to the real
    ``gh`` binary on ``PATH``). Numeric selectors override the default PR number;
    non-numeric selectors must exactly match ``head_ref`` or the fake fails.
    """
    script = tmp_path / "fake_gh.py"
    number_match = re.search(r"\d+", head_ref)
    default_number = int(number_match.group()) if number_match else 123
    real_gh = shutil.which("gh") or "gh"
    script.write_text(
        "import json, shutil, subprocess, sys\n"
        f"HEAD_REF = {head_ref!r}\n"
        f"DEFAULT_PR_NUMBER = {default_number}\n"
        f"REAL_GH = {real_gh!r}\n"
        "\n"
        "def _help():\n"
        "    result = subprocess.run([REAL_GH] + sys.argv[1:])\n"
        "    sys.exit(result.returncode)\n"
        "\n"
        "def _view():\n"
        "    args = sys.argv[1:]\n"
        "    fields_arg = 'headRefName'\n"
        "    if '--json' in args:\n"
        "        idx = args.index('--json')\n"
        "        if idx + 1 < len(args):\n"
        "            fields_arg = args[idx + 1]\n"
        "        args = args[:idx] + args[idx + 2:]\n"
        "    selector = None\n"
        "    for a in args:\n"
        "        if a not in ('pr', 'view'):\n"
        "            selector = a\n"
        "            break\n"
        "    pr_number = DEFAULT_PR_NUMBER\n"
        "    if selector is not None:\n"
        "        if selector.isdigit():\n"
        "            pr_number = int(selector)\n"
        "        elif selector == HEAD_REF:\n"
        "            pass\n"
        "        else:\n"
        "            print(f'no pull request found for {selector}', file=sys.stderr)\n"
        "            sys.exit(1)\n"
        "    fields = fields_arg.split(',')\n"
        "    data = {}\n"
        "    if 'number' in fields:\n"
        "        data['number'] = pr_number\n"
        "    if 'headRefName' in fields:\n"
        "        data['headRefName'] = HEAD_REF\n"
        "    print(json.dumps(data))\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    if '--help' in sys.argv and 'merge' in sys.argv:\n"
        "        _help()\n"
        "    elif 'view' in sys.argv:\n"
        "        _view()\n"
        "    else:\n"
        "        print('unknown gh command', file=sys.stderr)\n"
        "        sys.exit(1)\n",
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
        "payload = {'ok': ok, 'message': msg, 'data': data}\n"
        "print(json.dumps(payload, indent=2, sort_keys=True, default=str))\n",
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
    """Environment with fake ``gh`` and ``charlie`` binaries and an isolated fleet dir."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    gh = _make_fake_gh(tmp_path, os.environ.get("FAKE_GH_HEAD", "agent/issue-123-fix"))
    charlie = _make_fake_charlie(
        tmp_path,
        ok=os.environ.get("FAKE_CHARLIE_OK", "true").lower() == "true",
        reason=os.environ.get("FAKE_CHARLIE_REASON", "approved_at_head"),
    )
    return {
        "GH_BIN": str(gh),
        "CHARLIE_BIN": str(charlie),
        "CHARLIE_WORK_FLEET_DIR": str(fleet_dir),
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
    """The worker prefix is read from the layered config, not hardcoded."""
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


def test_hook_uses_global_layer_for_branch_prefix(
    tmp_path: Path, base_env: dict[str, str]
) -> None:
    """The hook reads the global fleet layer, like the orchestrator does."""
    repo = tmp_path / "repo"
    repo.mkdir()
    global_fleet_dir = tmp_path / "global-fleet"
    global_fleet_dir.mkdir()
    (global_fleet_dir / "config.yaml").write_text(
        "dispatch:\n  branch_prefix: global/prefix\n",
        encoding="utf-8",
    )
    gh = _make_fake_gh(repo, "global/prefix-123-fix")
    charlie = _make_fake_charlie(tmp_path, ok=True, reason="approved_at_head")
    env = {
        **base_env,
        "GH_BIN": str(gh),
        "CHARLIE_BIN": str(charlie),
        "CHARLIE_WORK_FLEET_DIR": str(global_fleet_dir),
    }
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 456"}},
        env=env,
        project_dir=repo,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "authorized" in (result.get("systemMessage") or "").lower()


def test_hook_denies_when_branch_prefix_config_is_invalid(
    tmp_path: Path, base_env: dict[str, str]
) -> None:
    """A broken config does not silently fall back to a hardcoded worker prefix."""
    repo = tmp_path / "broken-repo"
    repo.mkdir()
    (repo / "orchestrator.config.yaml").write_text(
        "unknown_section:\n  key: value\n",
        encoding="utf-8",
    )
    gh = _make_fake_gh(repo, "agent/issue-123-fix")
    env = {**base_env, "GH_BIN": str(gh)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 123"}},
        env=env,
        project_dir=repo,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "cannot load worker branch prefix" in (result.get("systemMessage") or "").lower()


def test_hook_surfaces_unauthorized_worker_pr(tmp_path: Path, base_env: dict[str, str]) -> None:
    """A worker PR that fails ``merge-check`` is denied."""
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
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    msg = result.get("systemMessage", "")
    assert "NOT authorized" in msg
    assert "no_decision" in msg
    assert "PR #759" in msg
    assert result["hookSpecificOutput"].get("permissionDecisionReason") == msg


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
    assert "PR #894" in msg


def test_hook_allows_unresolvable_pr_selector(base_env: dict[str, str]) -> None:
    """A ``gh pr merge`` whose selector cannot be resolved is allowed with a note."""
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge some-branch"}},
        env=base_env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert "could not resolve" in (result.get("systemMessage") or "").lower()


def test_hook_allows_when_gh_fails(tmp_path: Path, base_env: dict[str, str]) -> None:
    """If ``gh pr view`` fails, the hook is fail-open."""
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
    assert "could not resolve" in (result.get("systemMessage") or "").lower()


def test_hook_denies_when_merge_check_cannot_be_parsed(
    tmp_path: Path, base_env: dict[str, str]
) -> None:
    """If the merge-check preflight produces no usable JSON for a worker PR, the hook denies."""
    gh = _make_fake_gh(tmp_path, "agent/issue-123-fix")
    bad_charlie = tmp_path / "bad_charlie.py"
    bad_charlie.write_text("print('not json')", encoding="utf-8")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(bad_charlie)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 123"}},
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "merge-check" in (result.get("systemMessage") or "").lower()


def test_hook_denies_when_merge_check_crashes(tmp_path: Path, base_env: dict[str, str]) -> None:
    """A merge-check that exits non-zero with no JSON is denied for a worker PR."""
    gh = _make_fake_gh(tmp_path, "agent/issue-123-fix")
    crash_charlie = tmp_path / "crash_charlie.py"
    crash_charlie.write_text("import sys\nsys.exit(1)\n", encoding="utf-8")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(crash_charlie)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 123"}},
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "merge-check produced no stdout" in (result.get("systemMessage") or "")


def test_hook_resolves_pr_from_current_branch(tmp_path: Path, base_env: dict[str, str]) -> None:
    """``gh pr merge`` with no positional selector uses the current branch's PR."""
    repo = tmp_path / "worker-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-b", "agent/issue-456-fix"], cwd=repo, check=True)
    gh = _make_fake_gh(repo, "agent/issue-456-fix")
    charlie = _make_fake_charlie(tmp_path, ok=True, reason="approved_at_head")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(charlie)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge"}},
        env=env,
        project_dir=repo,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    msg = result.get("systemMessage", "")
    assert "authorized" in msg.lower()
    assert "PR #456" in msg


def test_hook_denies_no_argument_on_worker_branch_when_pr_view_fails(
    tmp_path: Path, base_env: dict[str, str]
) -> None:
    """A no-argument ``gh pr merge`` on a worker branch with no resolvable PR is denied."""
    repo = tmp_path / "worker-repo-fail"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-b", "agent/issue-789-fix"], cwd=repo, check=True)
    script = tmp_path / "fake_gh_fail.py"
    script.write_text(
        "import sys\nprint('no pull request found', file=sys.stderr)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    env = {**base_env, "GH_BIN": str(script)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge"}},
        env=env,
        project_dir=repo,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert (
        "could not resolve a pr for current worker branch"
        in (result.get("systemMessage") or "").lower()
    )


def test_hook_allows_no_argument_on_non_worker_branch(
    tmp_path: Path, base_env: dict[str, str]
) -> None:
    """A no-argument ``gh pr merge`` on a non-worker branch is not checked."""
    repo = tmp_path / "fix-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", "-b", "fix/321"], cwd=repo, check=True)
    script = tmp_path / "fake_gh_fail.py"
    script.write_text(
        "import sys\nprint('no pull request found', file=sys.stderr)\nsys.exit(1)\n",
        encoding="utf-8",
    )
    env = {**base_env, "GH_BIN": str(script)}
    output = _run_hook(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge"}},
        env=env,
        project_dir=repo,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert (
        "could not resolve a pr for the current branch"
        in (result.get("systemMessage") or "").lower()
    )


def test_hook_parses_quoted_value_flags(tmp_path: Path, base_env: dict[str, str]) -> None:
    """Quoted multi-word flag values do not break PR-number extraction."""
    gh = _make_fake_gh(tmp_path, "agent/issue-123-fix")
    charlie = _make_fake_charlie(tmp_path, ok=False, reason="no_decision")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(charlie)}
    output = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": 'gh pr merge --body "merge this thing" 759 --squash'},
        },
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "PR #759" in (result.get("systemMessage") or "")


def test_hook_parses_numeric_value_flag(tmp_path: Path, base_env: dict[str, str]) -> None:
    """A numeric flag value (e.g. ``--body "123"``) is not mistaken for the PR number."""
    gh = _make_fake_gh(tmp_path, "agent/issue-123-fix")
    charlie = _make_fake_charlie(tmp_path, ok=False, reason="no_decision")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(charlie)}
    output = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": 'gh pr merge --body "123" 759'},
        },
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "PR #759" in (result.get("systemMessage") or "")


def test_hook_parses_repo_short_flag(tmp_path: Path, base_env: dict[str, str]) -> None:
    """The ``-R`` value flag is skipped so the positional PR number is still found."""
    gh = _make_fake_gh(tmp_path, "agent/issue-123-fix")
    charlie = _make_fake_charlie(tmp_path, ok=False, reason="no_decision")
    env = {**base_env, "GH_BIN": str(gh), "CHARLIE_BIN": str(charlie)}
    output = _run_hook(
        {
            "tool_name": "Bash",
            "tool_input": {"command": "gh pr merge -R owner/repo 759"},
        },
        env=env,
    )
    assert output[0] == 0
    result = _parse_hook_output(output[1])
    assert result["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "PR #759" in (result.get("systemMessage") or "")


def test_hook_value_flags_match_gh_help() -> None:
    """The parser's value-flag set matches the installed ``gh`` help output."""
    spec = importlib.util.spec_from_file_location("merge_check", HOOK_SCRIPT)
    merge_check_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge_check_module)  # type: ignore[union-attr]

    flags = merge_check_module._gh_pr_merge_value_flags(REPO_ROOT)
    expected = {
        "-A",
        "--author-email",
        "-b",
        "--body",
        "-F",
        "--body-file",
        "-t",
        "--subject",
        "-R",
        "--repo",
        "--match-head-commit",
    }
    assert expected <= flags
    assert "--match-title" not in flags
    assert "--match-head-sha" not in flags


def test_hook_parses_real_merge_check_json_output(tmp_path: Path, monkeypatch, capsys) -> None:
    """The real ``charlie --json merge-check`` output is a single pretty-printed JSON doc.

    The hook parser must consume the whole stdout as one document, not scan
    lines, because ``print_result`` emits ``indent=2`` multi-line JSON. This
    test runs the real CLI, captures its stdout, and feeds that output through
    the hook's merge-check parser.
    """
    from charlie_work import cli

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)

    pr_dir = repo / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "sha-abc123"}),
        encoding="utf-8",
    )

    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    class FakeGitHub:
        def __init__(self, *, repo_root=None, dry_run=False, runtime=None, **kwargs):
            self.repo_root = repo_root
            self.dry_run = dry_run

        def pr_view(self, number: int):
            return {
                "number": number,
                "headRefName": "agent/issue-456-fix",
                "headRefOid": "sha-abc123",
                "state": "OPEN",
            }

        def name_with_owner(self):
            return "owner/repo"

    monkeypatch.setattr(cli, "GitHub", FakeGitHub)

    returncode = cli.main(["--json", "--repo", str(repo), "merge-check", "456"])
    assert returncode == 0

    cli_stdout = capsys.readouterr().out
    assert json.loads(cli_stdout)

    # Replay the captured CLI output through the hook parser. This is exactly
    # the shape the real ``charlie merge-check --json`` writes to stdout.
    replay = tmp_path / "replay_charlie_stdout.py"
    replay.write_text(f"import sys\nsys.stdout.write({cli_stdout!r})\n", encoding="utf-8")
    monkeypatch.setenv("CHARLIE_BIN", str(replay))

    spec = importlib.util.spec_from_file_location("merge_check", HOOK_SCRIPT)
    merge_check_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(merge_check_module)  # type: ignore[union-attr]
    try:
        verdict = merge_check_module._run_merge_check(456, repo)
    finally:
        # ``configure_logging`` opens a log file in the fleet dir; close it so
        # pytest's temp-path cleanup can remove the directory on Windows.
        logging.shutdown()

    assert verdict["ok"] is True
    assert verdict["data"]["authorized"] is True
    assert verdict["data"]["reason"] == "approved_at_head"
