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


# ---------------------------------------------------------------------------
# _parse_gh_merge_targets
# ---------------------------------------------------------------------------


def test_parse_plain_merge() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge 123 --squash")
    assert targets == [{"pr": 123, "repo": None}]


def test_parse_dash_r_repo() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge -R owner/repo 123")
    assert targets == [{"pr": 123, "repo": "owner/repo"}]


def test_parse_repo_equals() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge --repo=owner/repo 123")
    assert targets == [{"pr": 123, "repo": "owner/repo"}]


def test_parse_repo_url_flag() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge --repo https://github.com/o/r 55")
    assert targets == [{"pr": 55, "repo": "o/r"}]


def test_parse_pr_url_form() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge https://github.com/o/r/pull/55")
    assert targets == [{"pr": 55, "repo": "o/r"}]


def test_parse_no_number_current_branch() -> None:
    targets = hook._parse_gh_merge_targets("gh pr merge --squash")
    assert targets == [{"pr": None, "repo": None}]


def test_parse_multiple_chained_invocations() -> None:
    command = "gh pr merge 1 --squash && gh pr merge -R o/r 2"
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [
        {"pr": 1, "repo": None},
        {"pr": 2, "repo": "o/r"},
    ]


def test_parse_unbalanced_quote_yields_pr_none() -> None:
    targets = hook._parse_gh_merge_targets('gh pr merge 123 "unterminated')
    assert targets == [{"pr": None, "repo": None}]


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
    assert targets == [{"pr": 55, "repo": None}]


def test_parse_env_prefixed_invocation_detected() -> None:
    targets = hook._parse_gh_merge_targets("GH_TOKEN=x gh pr merge 7")
    assert targets == [{"pr": 7, "repo": None}]


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
    assert targets == [{"pr": 5, "repo": "o/r"}]


def test_parse_gh_repo_env_overridden_by_explicit_flag() -> None:
    # Same precedence as gh itself: explicit -R/--repo beats GH_REPO.
    targets = hook._parse_gh_merge_targets("GH_REPO=a/b gh pr merge 5 -R x/y")
    assert targets == [{"pr": 5, "repo": "x/y"}]


def test_parse_gh_repo_env_via_env_prefix() -> None:
    targets = hook._parse_gh_merge_targets("env GH_REPO=a/b gh pr merge 7")
    assert targets == [{"pr": 7, "repo": "a/b"}]


def test_parse_multi_invocation_with_gh_repo_env() -> None:
    command = "gh pr merge 5; GH_REPO=o/r gh pr merge 6"
    targets = hook._parse_gh_merge_targets(command)
    assert targets == [
        {"pr": 5, "repo": None},
        {"pr": 6, "repo": "o/r"},
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
