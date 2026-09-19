"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

``_parse_gh_merge_targets`` / ``_GH_PR_MERGE``: merge-invocation
detection, PR-number and ``-R``/``GH_REPO`` resolution, flag handling,
and the mention-vs-invocation token rules.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

import pytest

from charlie_work import merge_preflight_hook as hook

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


# ---------------------------------------------------------------------------
# Round-2 fixes (#1195): GH_REPO env override and interspersed global flags
# ---------------------------------------------------------------------------


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
# Round-3 fixes (#1195): value-taking flags before the PR number
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
