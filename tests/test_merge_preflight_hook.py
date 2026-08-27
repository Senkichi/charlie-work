"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. ``_load_fleet_roots`` and ``_run_merge_check`` are
monkeypatched for ``_decide``/``main`` tests; ``_run_merge_check``'s own test
patches ``charlie_work.cli.main`` (imported lazily inside the function).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work import merge_preflight_hook as hook

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# _parse_gh_merge_targets
# ---------------------------------------------------------------------------


def test_parse_plain_merge() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge 123 --squash")
    assert targets == [{"pr": 123, "repo": None, "cd_cwd": None}]


def test_parse_dash_r_repo() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge -R owner/repo 123")
    assert targets == [{"pr": 123, "repo": "owner/repo", "cd_cwd": None}]


def test_parse_repo_equals() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge --repo=owner/repo 123")
    assert targets == [{"pr": 123, "repo": "owner/repo", "cd_cwd": None}]


def test_parse_repo_url_flag() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge --repo https://github.com/o/r 55")
    assert targets == [{"pr": 55, "repo": "o/r", "cd_cwd": None}]


def test_parse_pr_url_form() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge https://github.com/o/r/pull/55")
    assert targets == [{"pr": 55, "repo": "o/r", "cd_cwd": None}]


def test_parse_no_number_current_branch() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge --squash")
    assert targets == [{"pr": None, "repo": None, "cd_cwd": None}]


def test_parse_multiple_chained_invocations() -> None:
    command = "gh pr merge 1 --squash && gh pr merge -R o/r 2"
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [
        {"pr": 1, "repo": None, "cd_cwd": None},
        {"pr": 2, "repo": "o/r", "cd_cwd": None},
    ]


def test_parse_unbalanced_quote_yields_pr_none() -> None:
    targets = hook._parse_gh_merge_targets('gh pr merge 123 "unterminated')
    assert targets == [{"pr": None, "repo": None, "cd_cwd": None}]


def test_parse_no_merge_command_yields_empty_list() -> None:
    targets = hook._parse_gh_merge_targets("gh pr list")
    assert targets == []


# ---------------------------------------------------------------------------
# _repo_for_cwd
# ---------------------------------------------------------------------------


def test_repo_for_cwd_exact_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, root) == "o/repo"


def test_repo_for_cwd_nested_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "sub" / "dir"
    nested.mkdir(parents=True)
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, nested) == "o/repo"


def test_repo_for_cwd_sibling_prefix_does_not_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sibling = tmp_path / "repo-other"
    root.mkdir()
    sibling.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, sibling) is None


def test_repo_for_cwd_unrelated_cwd_returns_none(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    unrelated = tmp_path / "elsewhere"
    root.mkdir()
    unrelated.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, unrelated) is None


# ---------------------------------------------------------------------------
# _decide — Bash
# ---------------------------------------------------------------------------


def test_decide_bash_fleet_pr_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "not_approved"))
    reason = hook._decide("Bash", {"command": "gh pr merge -R o/repo 42 --squash"}, tmp_path)
    assert reason is not None
    assert "not_approved" in reason
    assert "#42" in reason


def test_decide_bash_fleet_pr_allowed_on_passing_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide("Bash", {"command": "gh pr merge -R o/repo 42 --squash"}, tmp_path)
    assert reason is None


def test_decide_bash_out_of_fleet_repo_skips_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide("Bash", {"command": "gh pr merge -R other/repo 42 --squash"}, tmp_path)
    assert reason is None
    assert calls == []


def test_decide_bash_no_pr_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(
        hook, "_run_merge_check", lambda repo_root, pr: (True, "should not be called")
    )
    reason = hook._decide("Bash", {"command": "gh pr merge --squash"}, root)
    assert reason is not None
    assert "o/repo" in reason


def test_decide_bash_unreadable_registry_denies_mentioning_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # None = registry exists but cannot be read/parsed -> fail closed.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    reason = hook._decide("Bash", {"command": "gh pr merge 1 --squash"}, tmp_path)
    assert reason is not None
    assert "registry" in reason.lower()


def test_decide_bash_unreadable_registry_denies_even_with_explicit_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreadable registry cannot confirm ANY repo is outside the fleet.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    reason = hook._decide("Bash", {"command": "gh pr merge -R other/repo 1 --squash"}, tmp_path)
    assert reason is not None
    assert "registry" in reason.lower()


def test_decide_bash_genuinely_empty_fleet_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # {} = no registry file / no repos: nothing fleet-managed, nothing to guard.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {})
    assert hook._decide("Bash", {"command": "gh pr merge 1 --squash"}, tmp_path) is None
    assert (
        hook._decide("Bash", {"command": "gh pr merge -R other/repo 1 --squash"}, tmp_path) is None
    )


def test_decide_mcp_unreadable_registry_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression for the review finding on PR #1195: the MCP path must honor
    # the same fail-closed contract as the Bash path when the registry is
    # unreadable — otherwise a corrupt fleet.json silently disables
    # enforcement for MCP-initiated merges.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    called: list[Any] = []
    monkeypatch.setattr(hook, "_run_merge_check", lambda *a: called.append(a) or (True, ""))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "Senkichi", "repo": "charlie-work", "pullNumber": 5},
        tmp_path,
    )
    assert reason is not None
    assert "registry" in reason.lower()
    assert called == []


def test_decide_mcp_genuinely_empty_fleet_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "r", "pullNumber": 5},
        tmp_path,
    )
    assert reason is None


def test_load_fleet_roots_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from charlie_work import layout

    registry = tmp_path / "fleet.json"
    monkeypatch.setattr(layout, "fleet_registry_path", lambda override=None: registry)

    # Missing file: no fleet configured -> {}.
    assert hook._load_fleet_roots() == {}

    # Corrupt file: read failure -> None (fail closed).
    registry.write_text("{not json", encoding="utf-8")
    assert hook._load_fleet_roots() is None

    # Non-dict JSON: also a read failure.
    registry.write_text("[]", encoding="utf-8")
    assert hook._load_fleet_roots() is None

    # Well-formed registry: owner/name lowercased -> repo_root Path.
    registry.write_text(
        '{"version": 1, "repos": {"Owner/Repo": {"repo_root": "C:/x/repo"}}}',
        encoding="utf-8",
    )
    roots = hook._load_fleet_roots()
    assert roots == {"owner/repo": Path("C:/x/repo")}


# ---------------------------------------------------------------------------
# _decide — MCP merge_pull_request
# ---------------------------------------------------------------------------


def test_decide_mcp_fleet_repo_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "head_moved"))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo", "pullNumber": 7},
        tmp_path,
    )
    assert reason is not None
    assert "head_moved" in reason
    assert "#7" in reason


def test_decide_mcp_fleet_repo_allowed_on_passing_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo", "pullNumber": 7},
        tmp_path,
    )
    assert reason is None


def test_decide_mcp_non_fleet_owner_repo_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "other", "repo": "repo", "pullNumber": 7},
        tmp_path,
    )
    assert reason is None
    assert calls == []


def test_decide_mcp_missing_pull_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo"},
        tmp_path,
    )
    assert reason is not None


def test_decide_mcp_non_int_pull_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "repo", "pullNumber": "not-an-int"},
        tmp_path,
    )
    assert reason is not None


# ---------------------------------------------------------------------------
# main() — stdin protocol
# ---------------------------------------------------------------------------


def _run_main(
    monkeypatch: pytest.MonkeyPatch, stdin_text: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str]:
    monkeypatch.setattr(hook.sys, "stdin", io.StringIO(stdin_text))
    rc = hook.main()
    captured = capsys.readouterr()
    return rc, captured.out


def test_main_malformed_json_rc0_no_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rc, out = _run_main(monkeypatch, "{not json", capsys)
    assert rc == 0
    assert out == ""


def test_main_benign_bash_no_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    rc, out = _run_main(monkeypatch, payload, capsys)
    assert rc == 0
    assert out == ""


def test_main_merge_shaped_bash_decide_raises_denies_fail_closed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> str | None:
        raise RuntimeError("registry blew up")

    monkeypatch.setattr(hook, "_decide", _boom)
    payload = json.dumps(
        {"tool_name": "Bash", "tool_input": {"command": "gh pr merge 1 --squash"}}
    )
    rc, out = _run_main(monkeypatch, payload, capsys)
    assert rc == 0
    assert out.strip() != ""
    decision = json.loads(out)
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "RuntimeError" in decision["hookSpecificOutput"]["permissionDecisionReason"]


def test_main_non_merge_bash_internals_raise_rc0_no_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> str | None:
        raise RuntimeError("should not matter, command is not merge-shaped")

    monkeypatch.setattr(hook, "_decide", _boom)
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "ls -la"}})
    rc, out = _run_main(monkeypatch, payload, capsys)
    assert rc == 0
    assert out == ""


# ---------------------------------------------------------------------------
# _run_merge_check
# ---------------------------------------------------------------------------


class _FakeCliModule:
    """Stand-in for ``charlie_work.cli`` with a patchable ``main`` attribute."""


def test_run_merge_check_rc_zero_is_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import charlie_work.cli as cli_module

    monkeypatch.setattr(cli_module, "main", lambda argv: 0)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is True
    assert isinstance(detail, str)


def test_run_merge_check_rc_nonzero_is_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import charlie_work.cli as cli_module

    monkeypatch.setattr(cli_module, "main", lambda argv: 2)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is False


def test_run_merge_check_system_exit_is_not_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import charlie_work.cli as cli_module

    def _raise_system_exit(argv: list[str]) -> int:
        raise SystemExit(3)

    monkeypatch.setattr(cli_module, "main", _raise_system_exit)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is False


def test_run_merge_check_runtime_error_denies_with_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import charlie_work.cli as cli_module

    def _raise_runtime_error(argv: list[str]) -> int:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "main", _raise_runtime_error)
    ok, detail = hook._run_merge_check(tmp_path, 1)
    assert ok is False
    assert "RuntimeError" in detail


# ---------------------------------------------------------------------------
# Token-based detection regressions
# ---------------------------------------------------------------------------


def test_parse_quoted_mention_is_not_an_invocation() -> None:
    # The first version of this hook denied its own feature commit: the
    # commit message described the guarded command, and raw-text regex
    # matched the mention.
    command = 'git commit -m "Intercepts Bash gh pr merge and the MCP merge tool"'
    assert hook._parse_gh_merge_targets(command) == []


def test_parse_quoted_tokens_still_detected() -> None:
    # Shell-quoting individual words must not evade detection: shlex strips
    # the quotes, so the consecutive-token match still fires.
    targets = hook._parse_gh_merge_targets('"gh" pr merge 55')
    assert targets == [{"pr": 55, "repo": None, "cd_cwd": None}]


def test_parse_env_prefixed_invocation_detected() -> None:
    targets = hook._parse_gh_merge_targets("GH_TOKEN=x gh pr merge 7")
    assert targets == [{"pr": 7, "repo": None, "cd_cwd": None}]


def test_decide_mcp_bool_pull_number_in_fleet_repo_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # bool is a subclass of int; pullNumber=true must not preflight PR #1.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/r": tmp_path})
    called: list[Any] = []
    monkeypatch.setattr(hook, "_run_merge_check", lambda *a: called.append(a) or (True, ""))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "r", "pullNumber": True},
        tmp_path,
    )
    assert reason is not None
    assert "cannot determine PR number" in reason
    assert called == []


# ---------------------------------------------------------------------------
# Round-2 fixes (#1195): GH_REPO env override and interspersed global flags
# ---------------------------------------------------------------------------


def test_decide_bash_gh_repo_env_override_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer-mandated bypass: GH_REPO=owner/repo was invisible to the
    # old parser, so a merge into a fleet repo from a cwd outside every fleet
    # root sailed through undecided. cwd is deliberately outside the fleet
    # root to prove the target came from GH_REPO, not from cwd resolution.
    outside = tmp_path / "outside"
    outside.mkdir()
    fleet_root = tmp_path / "repo"
    fleet_root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"senkichi/charlie-work": fleet_root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "not_approved"))
    reason = hook._decide(
        "Bash",
        {"command": "GH_REPO=senkichi/charlie-work gh pr merge 5"},
        outside,
    )
    assert reason is not None
    assert "not_approved" in reason
    assert "#5" in reason


@pytest.mark.parametrize(
    "command",
    [
        "gh pr -R o/r merge 5",
        "gh -R o/r pr merge 5",
        "gh --repo=o/r pr merge 5",
    ],
)
def test_parse_interspersed_global_flags(command: str) -> None:
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [{"pr": 5, "repo": "o/r", "cd_cwd": None}]


def test_parse_gh_repo_env_overridden_by_explicit_flag() -> None:
    # Same precedence as gh itself: explicit -R/--repo beats GH_REPO.
    targets = hook._parse_gh_merge_targets("GH_REPO=a/b gh pr merge 5 -R x/y")
    assert targets == [{"pr": 5, "repo": "x/y", "cd_cwd": None}]


def test_parse_gh_repo_env_via_env_prefix() -> None:
    targets = hook._parse_gh_merge_targets("env GH_REPO=a/b gh pr merge 7")
    assert targets == [{"pr": 7, "repo": "a/b", "cd_cwd": None}]


def test_parse_multi_invocation_with_gh_repo_env() -> None:
    command = "gh pr merge 5; GH_REPO=o/r gh pr merge 6"
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [
        {"pr": 5, "repo": None, "cd_cwd": None},
        {"pr": 6, "repo": "o/r", "cd_cwd": None},
    ]


def test_gh_pr_merge_regex_matches_interspersed_flags() -> None:
    assert hook._GH_PR_MERGE.search("gh -R o/r pr merge 5")
    assert hook._GH_PR_MERGE.search("gh pr -R o/r merge 5")


def test_gh_pr_merge_regex_does_not_match_unrelated_or_across_separator() -> None:
    assert hook._GH_PR_MERGE.search("gh pr list; git merge main") is None
    assert hook._GH_PR_MERGE.search("git merge main") is None


def test_parse_gh_pr_view_merge_is_not_an_invocation() -> None:
    # "merge" here is an argument to "view", not the merge subcommand.
    assert hook._parse_gh_merge_targets("gh pr view merge") == []


def test_parse_quoted_mention_of_gh_pr_merge_is_not_an_invocation() -> None:
    command = "git commit -m 'about gh pr merge'"
    assert hook._parse_gh_merge_targets(command) == []


# ---------------------------------------------------------------------------
# Round-3 fixes (#1195): value-taking flags before the PR number, and the
# .claude/settings.json wiring that activates the hook.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "expected_pr"),
    [
        ("gh pr merge -t 42 1195", 1195),
        ("gh pr merge --subject 42 1195", 1195),
        ("gh pr merge --match-head-commit abc123 8", 8),
        ("gh pr merge --squash 5", 5),
        ("gh pr merge -s -d 5", 5),
        ("gh pr merge --body=hello 7", 7),
        ("gh pr merge 1195 -t 42", 1195),
    ],
)
def test_parse_known_flags_do_not_shift_the_pr_number(command: str, expected_pr: int) -> None:
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [{"pr": expected_pr, "repo": None, "cd_cwd": None}]


def test_parse_unknown_flag_before_pr_number_is_ambiguous() -> None:
    # -x is not in _MERGE_VALUE_FLAGS or _MERGE_BOOLEAN_FLAGS, so whether it
    # consumes "42" as a value is unknowable here. Reading "42" as the PR
    # would validate the wrong pull request (round-3 review finding) -- the
    # invocation must fail closed instead.
    targets = hook._parse_gh_merge_targets("gh pr merge -x 42 1195")
    assert targets == [{"pr": None, "repo": None, "cd_cwd": None}]


def test_parse_unknown_flag_suppresses_pr_url_form_too() -> None:
    command = "gh pr merge --frobnicate 9 https://github.com/o/r/pull/3"
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [{"pr": None, "repo": None, "cd_cwd": None}]


def test_decide_bash_ambiguous_pr_denies_without_running_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})

    def _must_not_run(repo_root: Path, pr: int) -> tuple[bool, str]:
        raise AssertionError("merge-check must not run when the PR number is ambiguous")

    monkeypatch.setattr(hook, "_run_merge_check", _must_not_run)
    reason = hook._decide("Bash", {"command": "gh pr merge -x 42 1195"}, root)
    assert reason is not None
    assert "o/repo" in reason


def test_claude_settings_wires_merge_preflight_hook() -> None:
    settings_path = REPO_ROOT / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    pre_tool_use = settings["hooks"]["PreToolUse"]
    matchers = {entry["matcher"]: entry for entry in pre_tool_use}
    assert set(matchers) == {"Bash", "mcp__github__merge_pull_request"}

    module_ref = f"-m {hook.__name__}"
    for matcher, entry in matchers.items():
        # The Bash matcher may carry additional PreToolUse hooks (e.g. the
        # git-push lint hook, #1309); find the merge_preflight_hook among
        # them rather than assuming it is the only one.
        command_hooks = entry["hooks"]
        merge_hooks = [h for h in command_hooks if module_ref in h.get("command", "")]
        assert len(merge_hooks) == 1, (
            f"{matcher} matcher must wire exactly one {module_ref} hook, got {len(merge_hooks)}"
        )
        command_hook = merge_hooks[0]
        assert command_hook["type"] == "command"
        command = command_hook["command"]
        assert command.strip().endswith("|| true"), f"{matcher} hook must fail open on crash"


# ---------------------------------------------------------------------------
# #1252 defect 2: heredoc bodies must not trigger the merge gate
# ---------------------------------------------------------------------------


def test_parse_heredoc_body_with_merge_prose_is_not_an_invocation() -> None:
    # The observed incident: ``gh issue create`` was denied because the issue
    # body, written via heredoc, contained "gh pr merge 123" as prose.
    command = "gh issue create --body-file - <<'EOF'\nThis report describes\ngh pr merge 123\nbehavior.\nEOF"
    assert hook._parse_gh_merge_targets(command) == []


def test_parse_heredoc_body_unquoted_delimiter_is_not_an_invocation() -> None:
    command = "gh issue create --body-file - <<EOF\ngh pr merge 123\nEOF"
    assert hook._parse_gh_merge_targets(command) == []


def test_parse_heredoc_strip_dash_preserves_real_merge_after_body() -> None:
    # A real merge after a heredoc body must still be detected.
    command = (
        "gh issue create --body-file - <<'EOF'\ngh pr merge 999\nEOF\ngh pr merge 42 --squash"
    )
    targets = hook._parse_gh_merge_targets(command)
    assert len(targets) == 1
    assert targets[0]["pr"] == 42


def test_parse_heredoc_with_command_continuation_on_same_line() -> None:
    # ``cat <<EOF && gh pr merge 42``: the merge is on the same line as the
    # heredoc start, so it runs AFTER the heredoc closes and must be detected.
    command = "cat <<'EOF'\nbody\ngh pr merge 999\nEOF\ngh pr merge 42 --squash"
    targets = hook._parse_gh_merge_targets(command)
    assert len(targets) == 1
    assert targets[0]["pr"] == 42


def test_parse_heredoc_inside_double_quotes_not_stripped() -> None:
    # A << inside a double-quoted string is not a heredoc; the quoted body
    # is a single shlex token and must not match.
    command = 'echo "a << b gh pr merge 123"'
    assert hook._parse_gh_merge_targets(command) == []


def test_parse_unclosed_heredoc_left_intact_fail_closed() -> None:
    # A << with no closing delimiter is not stripped — the raw tokens flow
    # through and the merge gate fires (fail-closed), rather than silently
    # dropping a potentially real merge.
    command = "cat <<EOF\ngh pr merge 123"
    targets = hook._parse_gh_merge_targets(command)
    # The body is NOT stripped, so the merge invocation is detected.
    assert len(targets) == 1
    assert targets[0]["pr"] == 123


def test_parse_heredoc_strip_dash_delimiter() -> None:
    # <<- strips leading tabs from the delimiter line.
    command = "cat <<-EOF\n\tbody\ngh pr merge 999\n\tEOF\ngh pr merge 7"
    targets = hook._parse_gh_merge_targets(command)
    assert len(targets) == 1
    assert targets[0]["pr"] == 7


def test_decide_bash_heredoc_body_does_not_deny_issue_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The live incident: a ``gh issue create`` with a heredoc body containing
    # merge prose was denied. After the fix, it must pass through undecided.
    # The cwd is INSIDE a fleet repo so the old code (without heredoc
    # stripping) would resolve the "merge" from the heredoc body to this
    # repo and run merge-check on PR #123 — the test must fail against the
    # unfixed code to exercise the fix.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    command = (
        "gh issue create --body-file - <<'EOF'\n"
        "This report describes gh pr merge 123 behavior.\n"
        "EOF"
    )
    reason = hook._decide("Bash", {"command": command}, root)
    assert reason is None
    assert calls == []


# ---------------------------------------------------------------------------
# #1252 defect 1: repo resolved from command's effective cwd, not hook cwd
# ---------------------------------------------------------------------------


def test_parse_cd_absolute_path_sets_cd_cwd(tmp_path: Path) -> None:
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {target_dir.as_posix()} && gh pr merge 1679", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["pr"] == 1679
    assert targets[0]["cd_cwd"] == target_dir.resolve()


def test_parse_cd_relative_path_resolved_against_cwd(tmp_path: Path) -> None:
    (tmp_path / "sub" / "repo").mkdir(parents=True)
    targets = hook._parse_gh_merge_targets("cd sub/repo && gh pr merge 5", tmp_path)
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] == (tmp_path / "sub" / "repo").resolve()


def test_parse_no_cd_yields_cd_cwd_none(tmp_path: Path) -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge 5", tmp_path)
    assert targets == [{"pr": 5, "repo": None, "cd_cwd": None}]


def test_parse_cd_in_subshell_does_not_leak(tmp_path: Path) -> None:
    # A cd inside (...) must not affect commands after the subshell.
    target_dir = tmp_path / "inner"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"(cd {target_dir.as_posix()} && ls); gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_in_subshell_applies_inside(tmp_path: Path) -> None:
    # A cd inside (...) applies to merge invocations inside the subshell.
    target_dir = tmp_path / "inner"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"(cd {target_dir.as_posix()} && gh pr merge 5)", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] == target_dir.resolve()


def test_parse_cd_not_in_command_position_ignored(tmp_path: Path) -> None:
    # ``echo cd /path`` must not update effective cwd — cd is an argument.
    targets = hook._parse_gh_merge_targets(
        f"echo cd {tmp_path.as_posix()} && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_multiple_cds_track_effective_cwd(tmp_path: Path) -> None:
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {dir_a.as_posix()} && gh pr merge 1; cd {dir_b.as_posix()} && gh pr merge 2",
        tmp_path,
    )
    assert len(targets) == 2
    assert targets[0]["cd_cwd"] == dir_a.resolve()
    assert targets[1]["cd_cwd"] == dir_b.resolve()


def test_decide_bash_cd_resolves_correct_fleet_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The core defect: a session anchored in charlie-work merging a job-cannon
    # PR by number was checked against charlie-work's PR. After the fix, the
    # cd'd directory determines the repo.
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root.mkdir()
    job_cannon_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    # Session cwd is charlie-work, but the merge runs in job-cannon.
    reason = hook._decide(
        "Bash",
        {"command": f"cd {job_cannon_root.as_posix()} && gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is None
    assert calls == [(job_cannon_root, 1679)]


def test_decide_bash_cd_to_non_fleet_dir_skips_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the cd'd directory is not a fleet repo, the merge is out of scope —
    # do NOT fall back to the hook cwd (the cd is authoritative).
    charlie_root = tmp_path / "charlie-work"
    outside = tmp_path / "outside"
    charlie_root.mkdir()
    outside.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"senkichi/charlie-work": charlie_root})
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "Bash",
        {"command": f"cd {outside.as_posix()} && gh pr merge 5 --squash"},
        charlie_root,
    )
    assert reason is None
    assert calls == []


def test_decide_bash_no_cd_warns_on_hook_cwd_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No --repo and no cd: the hook infers from its own cwd and warns.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide("Bash", {"command": "gh pr merge 5 --squash"}, root)
    assert reason is None
    stderr = capsys.readouterr().err
    assert "inferring repo from hook cwd" in stderr


def test_decide_bash_cd_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With a cd, the repo is resolved confidently — no warning.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide(
        "Bash", {"command": f"cd {root.as_posix()} && gh pr merge 5 --squash"}, tmp_path
    )
    assert reason is None
    stderr = capsys.readouterr().err
    assert "inferring repo from hook cwd" not in stderr


def test_decide_bash_explicit_repo_does_not_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With an explicit --repo, no warning.
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {"o/repo": root})
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (True, "ok"))
    reason = hook._decide("Bash", {"command": "gh pr merge -R o/repo 5 --squash"}, tmp_path)
    assert reason is None
    stderr = capsys.readouterr().err
    assert "inferring repo from hook cwd" not in stderr


def test_decide_bash_cd_denied_on_failed_merge_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The cd-resolved repo must still be checked: a failing merge-check denies.
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root.mkdir()
    charlie_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    monkeypatch.setattr(hook, "_run_merge_check", lambda repo_root, pr: (False, "not_approved"))
    reason = hook._decide(
        "Bash",
        {"command": f"cd {job_cannon_root.as_posix()} && gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is not None
    assert "not_approved" in reason
    assert "#1679" in reason
    assert "senkichi/job-cannon" in reason


# ---------------------------------------------------------------------------
# #1252 review finding: cd in a pipeline or backgrounded command runs in a
# subshell — its cwd change must not persist to later && / ; -joined commands
# in the same chain. Without this, ``echo x | cd <repo> && gh pr merge N``
# and ``cd <repo> & gh pr merge N`` reopen the exact wrong-repo merge-check
# bypass this hook closes.
# ---------------------------------------------------------------------------


def test_parse_cd_piped_does_not_persist_cd_cwd(tmp_path: Path) -> None:
    # ``echo x | cd <repo> && gh pr merge N``: the cd is the right side of a
    # pipe, so it runs in a subshell. The ``&&`` joins the *pipeline* (not
    # the cd) to the merge, so the merge runs in the original cwd, not the
    # cd'd directory. cd_cwd must be None.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"echo x | cd {target_dir.as_posix()} && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_backgrounded_does_not_persist_cd_cwd(tmp_path: Path) -> None:
    # ``cd <repo> & gh pr merge N``: the ``&`` backgrounds the cd (subshell),
    # so the merge runs in the original cwd. cd_cwd must be None.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(f"cd {target_dir.as_posix()} & gh pr merge 5", tmp_path)
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_left_side_of_pipe_does_not_persist(tmp_path: Path) -> None:
    # ``cd <repo> | cat; gh pr merge N``: the left side of a pipe also runs
    # in a subshell, so the cd does not persist past the pipeline.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {target_dir.as_posix()} | cat; gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_in_multi_pipe_does_not_persist(tmp_path: Path) -> None:
    # A cd in any segment of a multi-stage pipeline is subshell-scoped.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"echo a | echo b | cd {target_dir.as_posix()} && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_before_pipeline_persists_after_pipeline(tmp_path: Path) -> None:
    # ``cd <repo> && echo x | cat && gh pr merge N``: the cd is before the
    # pipeline (joined by ``&&``), so it persists. The pipeline (echo | cat)
    # runs in subshells but does not undo the prior cd. The merge after the
    # pipeline runs in the cd'd directory.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"cd {target_dir.as_posix()} && echo x | cat && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] == target_dir.resolve()


def test_parse_cd_piped_with_semicolon_does_not_persist(tmp_path: Path) -> None:
    # ``echo x | cd <repo>; gh pr merge N``: the semicolon ends the pipeline,
    # reverting the subshell-scoped cd. The merge runs in the original cwd.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"echo x | cd {target_dir.as_posix()}; gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_parse_cd_in_subshell_with_pipe_does_not_leak(tmp_path: Path) -> None:
    # ``(echo x | cd <repo>) && gh pr merge N``: the pipe is inside a
    # subshell, so the cd is doubly contained. The merge after ``)`` runs in
    # the original cwd.
    target_dir = tmp_path / "job-cannon"
    target_dir.mkdir()
    targets = hook._parse_gh_merge_targets(
        f"(echo x | cd {target_dir.as_posix()}) && gh pr merge 5", tmp_path
    )
    assert len(targets) == 1
    assert targets[0]["cd_cwd"] is None


def test_decide_bash_cd_piped_does_not_check_wrong_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The wrong-repo bypass this hook closes: a session anchored in
    # charlie-work runs ``echo x | cd job-cannon && gh pr merge 1679``. The
    # cd is piped (subshell), so the merge actually runs in charlie-work.
    # The hook must check charlie-work's PR #1679, NOT job-cannon's. If the
    # cd were wrongly attributed, merge-check would run against job-cannon
    # (the wrong repo), silently bypassing the gate for charlie-work.
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root.mkdir()
    job_cannon_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "Bash",
        {"command": f"echo x | cd {job_cannon_root.as_posix()} && gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is None
    # The merge runs in charlie-work (the hook cwd, since the piped cd does
    # not persist), so merge-check is called on charlie_root, not
    # job_cannon_root.
    assert calls == [(charlie_root, 1679)], (
        f"merge-check must run against charlie-work (hook cwd), not job-cannon "
        f"(piped cd); got {calls}"
    )


def test_decide_bash_cd_backgrounded_does_not_check_wrong_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Same bypass via backgrounding: ``cd job-cannon & gh pr merge 1679``.
    # The cd is backgrounded (subshell), so the merge runs in charlie-work.
    charlie_root = tmp_path / "charlie-work"
    job_cannon_root = tmp_path / "job-cannon"
    charlie_root.mkdir()
    job_cannon_root.mkdir()
    monkeypatch.setattr(
        hook,
        "_load_fleet_roots",
        lambda: {
            "senkichi/charlie-work": charlie_root,
            "senkichi/job-cannon": job_cannon_root,
        },
    )
    calls: list[Any] = []
    monkeypatch.setattr(
        hook,
        "_run_merge_check",
        lambda repo_root, pr: calls.append((repo_root, pr)) or (True, "ok"),
    )
    reason = hook._decide(
        "Bash",
        {"command": f"cd {job_cannon_root.as_posix()} & gh pr merge 1679 --squash"},
        charlie_root,
    )
    assert reason is None
    assert calls == [(charlie_root, 1679)], (
        f"merge-check must run against charlie-work (hook cwd), not job-cannon "
        f"(backgrounded cd); got {calls}"
    )
