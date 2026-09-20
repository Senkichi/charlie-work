"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

``_parse_gh_merge_targets`` heredoc handling (#1252 defect 2):
heredoc bodies must not trigger the merge gate, while real merges
after a heredoc (or inside an unclosed one) still must.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work import merge_preflight_hook as hook

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
