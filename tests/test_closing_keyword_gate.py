"""Tests for the closing-keyword CI gate (issue #790).

Covers the pure scanning function (`find_unexpected_closing_references`),
the refactored shared negation-aware scanning primitive it shares with
`linked_issue_number` (`iter_unnegated_closing_keyword_matches`), the new
`GitHub.pr_commits` REST wrapper, and the CLI command that wires them
together for `charlie closing-keyword-check --pr N`.

The centerpiece is `test_pr788_actual_commit_message_is_detected_as_a_violation`:
a pinned regression fixture using PR #788's own real, merged commit message
text (reconstructed via ``git show 24d3aceae758da81641a898d05314c62a92e2608
--format=%B -s``) as a failing positive control. PR #788 fixed the
negation-aware matcher for charlie-work's own label-transition binding, but
its own commit message contained an unnegated "Fixes #649" inside a quoted
illustrative example and GitHub's native auto-close-on-merge acted on it,
closing issue #649 on merge even though the PR body itself
(`closingIssuesReferences`) was clean. That incident is exactly what issue
#790 exists to prevent.
"""

from __future__ import annotations

import argparse

import pytest

from charlie_work import cli as cli_module
from charlie_work.closing_keyword_gate import (
    UnexpectedClosingReference,
    find_unexpected_closing_references,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.github import (
    CLOSING_KEYWORD_PR_FIELDS,
    GitHub,
    GitHubRunResult,
    iter_unnegated_closing_keyword_matches,
)
from charlie_work.github import linked_issue_number

# --- find_unexpected_closing_references: core scanning behavior ---


def test_clean_when_only_the_intended_issue_is_referenced() -> None:
    findings = find_unexpected_closing_references(
        pr_body="Fixes #999999",
        commit_messages=["fix: do the thing\n\nFixes #999999"],
        intended_issue_number=999999,
    )
    assert findings == []


def test_flags_body_reference_to_an_issue_other_than_the_intended_one() -> None:
    findings = find_unexpected_closing_references(
        pr_body="Fixes #999999\n\nAlso closes #999998 as a drive-by.",
        commit_messages=[],
        intended_issue_number=999999,
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.issue_number == 999998
    assert finding.source == "pr body"
    assert finding.matched_text.lower() == "closes #999998"


def test_flags_commit_message_reference_to_an_issue_other_than_the_intended_one() -> None:
    findings = find_unexpected_closing_references(
        pr_body="Fixes #999999",
        commit_messages=["fix: unrelated cleanup\n\nFixes #999997 as well"],
        intended_issue_number=999999,
    )
    assert len(findings) == 1
    assert findings[0].issue_number == 999997
    assert findings[0].source == "commit #1"


def test_multiple_commits_are_each_scanned_and_indexed_from_one() -> None:
    findings = find_unexpected_closing_references(
        pr_body="Fixes #999999",
        commit_messages=["wip", "fix: closes #999996 by mistake", "fixup"],
        intended_issue_number=999999,
    )
    assert len(findings) == 1
    assert findings[0].source == "commit #2"
    assert findings[0].issue_number == 999996


def test_negated_reference_is_not_flagged() -> None:
    # "does not fix #999993" must not be treated as a live closing reference
    # -- same negation guard linked_issue_number already applies (issue #781).
    findings = find_unexpected_closing_references(
        pr_body="",
        commit_messages=["chore: note that this does not fix #999993"],
        intended_issue_number=None,
    )
    assert findings == []


def test_intended_none_flags_every_unnegated_reference() -> None:
    # A cross-repository/fork PR (or a same-repo PR with no resolvable
    # branch/keyword binding at all) has no trusted target to exempt
    # anything against -- mirrors linked_issue_number's fail-closed posture
    # for unknown provenance: nothing is exempt when nothing is intended.
    findings = find_unexpected_closing_references(
        pr_body="Fixes #999995",
        commit_messages=["Resolves #999994"],
        intended_issue_number=None,
    )
    assert {finding.issue_number for finding in findings} == {999995, 999994}


def test_unexpected_closing_reference_is_a_frozen_dataclass() -> None:
    # CLAUDE.md invariant: config/value objects are frozen dataclasses.
    finding = UnexpectedClosingReference(
        issue_number=999995, source="pr body", matched_text="fix #999995"
    )
    with pytest.raises(AttributeError):
        finding.issue_number = 999994  # type: ignore[misc]


def test_shares_scanning_primitive_with_linked_issue_number() -> None:
    # The gate must not diverge from linked_issue_number's own negation
    # guard -- both are backed by the same iter_unnegated_closing_keyword_matches
    # generator in github.py (refactored out of _first_unnegated_closing_keyword_match
    # specifically so these two consumers cannot drift onto separate regexes).
    # Padding matches the existing repo regression test's construction
    # (test_linked_issue_number_negation_does_not_shadow_later_genuine_match):
    # without it, the second match's 32-char negation lookback window reaches
    # back across the sentence boundary and treats it as negated too.
    padded_body = "does not fix #999993. " + ("x" * 50) + " Fixes #999999"

    matches = list(iter_unnegated_closing_keyword_matches(padded_body))
    assert [int(match.group(1)) for match in matches] == [999999]

    assert (
        linked_issue_number(
            {"body": padded_body}, is_cross_repository=False, branch_prefix="agent/issue"
        )
        == 999999
    )

    findings = find_unexpected_closing_references(
        pr_body=padded_body, commit_messages=[], intended_issue_number=None
    )
    assert [finding.issue_number for finding in findings] == [999999]


# --- Pinned regression: PR #788's actual merged commit message ---

# Reconstructed verbatim via:
#   git show 24d3aceae758da81641a898d05314c62a92e2608 --format=%B -s
# DO NOT edit this string to "clean it up" -- its exact wording (the quoted
# "Fixes #649" mid-sentence, with no negation word in its 32-char lookback
# window) is the whole point of the fixture.
_PR_788_COMMIT_MESSAGE = """fix(github): guard closing-keyword binding against negation, defang outbound reviewer prose (#788)

_CLOSING_KEYWORD_REF matched a closing keyword even inside a negated phrase
("does not fix #649" bound to issue 649 the same as "Fixes #649"). Two
independent failure modes exist: GitHub's own auto-close (not controllable
from this codebase) and charlie-work's own linked_issue_number label-transition
binding (fixable by guarding the match).

Inbound: linked_issue_number now rejects a closing-keyword match preceded by
a negation word/contraction (not/never/without/cannot/n't) within a 32-char
lookback window, scanning forward past a negated match instead of giving up
on the field so a later genuine match still binds. Biased toward the safe
direction: a missed binding leaves current label state; a false binding
silently marks live work done.

Outbound: a new defang_closing_keywords() rewrites "<keyword> #N" to
"<keyword> issue N" (number stays legible, binding syntax removed) wherever
reviewer-authored prose is embedded in a rework brief a worker reads and
copies into its own PR body/commit -- text this codebase cannot re-check with
linked_issue_number's guard once it leaves. Applied in
_render_required_changes_section (both the structured-findings and
summary-fallback tiers) and in _write_rework_prompt's dispatch_note slot --
a second, independent template interpolation point in prompts/rework.md that
carries the same reviewer summary text and was found leaking the raw keyword
through an integration test before this commit.

Addresses issue 781
"""


def test_pr788_actual_commit_message_is_detected_as_a_violation() -> None:
    # PR #788's declared target was issue #781 ("Addresses issue 781" is not
    # a closing keyword, so it doesn't bind via linked_issue_number either --
    # its branch fix/false-close-hardening carries no issue number, so 781
    # is the value an operator/CLI would supply or that a future stricter
    # convention would resolve; the point under test is orthogonal to how
    # 781 was determined). GitHub's native auto-close does not care about an
    # "Addresses" trailer -- it acts on any unnegated closing keyword,
    # anywhere, including inside this quoted illustrative example -- and
    # closed issue #649 on merge.
    findings = find_unexpected_closing_references(
        pr_body="",
        commit_messages=[_PR_788_COMMIT_MESSAGE],
        intended_issue_number=781,
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.issue_number == 649
    assert finding.source == "commit #1"
    assert finding.matched_text.lower() == "fixes #649"


def test_pr788_commit_message_also_flagged_when_target_is_unresolved() -> None:
    # Same fixture, no declared target at all (intended_issue_number=None) --
    # still exactly one finding, issue #649. Proves the detection does not
    # depend on knowing the "right" answer in advance.
    findings = find_unexpected_closing_references(
        pr_body="",
        commit_messages=[_PR_788_COMMIT_MESSAGE],
        intended_issue_number=None,
    )
    assert [finding.issue_number for finding in findings] == [649]


# --- GitHub.pr_commits: REST wrapper (not gh pr view --json commits) ---


def test_pr_commits_extracts_raw_message_from_rest_shape(monkeypatch, tmp_path) -> None:
    def fake_run(self, args, *, json_output=False, allow_failure=False):
        assert args[0] == "api"
        assert "pulls/999992/commits" in args[1]
        return [
            {"sha": "abc123", "commit": {"message": "fix: thing\n\nFixes #999999"}},
            {"sha": "def456", "commit": {"message": "wip"}},
        ]

    monkeypatch.setattr(GitHub, "run", fake_run)
    gh = GitHub(tmp_path)

    commits = gh.pr_commits(999992)

    assert commits is not None
    assert [c["commit"]["message"] for c in commits] == ["fix: thing\n\nFixes #999999", "wip"]


def test_pr_commits_returns_none_on_failure(monkeypatch, tmp_path) -> None:
    def fake_run(self, args, *, json_output=False, allow_failure=False):
        return GitHubRunResult(ok=False, returncode=1, stdout="", stderr="boom", error="boom")

    monkeypatch.setattr(GitHub, "run", fake_run)
    gh = GitHub(tmp_path)

    assert gh.pr_commits(999992) is None


# --- CLI wiring: charlie closing-keyword-check --pr N ---


class _FakeGitHubForCLI:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def pr_view(self, number: int, *, fields: str | None = None):
        return {
            "number": number,
            "body": "Fixes #999999",
            "headRefName": "agent/issue-999999-do-thing",
            "isCrossRepository": False,
        }

    def pr_commits(self, number: int):
        return [{"commit": {"message": "fix: unrelated cleanup\n\nFixes #999997 as well"}}]


def _cli_args(pr: int) -> argparse.Namespace:
    return argparse.Namespace(repo=None, config=None, fleet_dir=None, dry_run=False, pr=pr)


def test_cli_closing_keyword_check_fails_on_unexpected_reference(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(cli_module, "GitHub", _FakeGitHubForCLI)

    result = cli_module.run_closing_keyword_check_command(_cli_args(999992))

    assert result.ok is False
    assert result.data["intended_issue_number"] == 999999
    assert result.data["findings"] == [
        {"issue_number": 999997, "source": "commit #1", "matched_text": "Fixes #999997"}
    ]
    assert "999997" in result.message


def test_cli_closing_keyword_check_passes_when_clean(monkeypatch, tmp_path) -> None:
    class FakeGitHubClean(_FakeGitHubForCLI):
        def pr_commits(self, number: int):
            return [{"commit": {"message": "fix: unrelated cleanup\n\nFixes #999999 as well"}}]

    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(cli_module, "GitHub", FakeGitHubClean)

    result = cli_module.run_closing_keyword_check_command(_cli_args(999992))

    assert result.ok is True
    assert result.data["findings"] == []


def test_cli_closing_keyword_check_reports_fetch_failure(monkeypatch, tmp_path) -> None:
    class FakeGitHubNoCommits(_FakeGitHubForCLI):
        def pr_commits(self, number: int):
            return None

    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(cli_module, "GitHub", FakeGitHubNoCommits)

    result = cli_module.run_closing_keyword_check_command(_cli_args(999992))

    assert result.ok is False
    assert "commits" in result.message


# --- Regression: CI failure on run 30609781476 (statusCheckRollup) ---
#
# `_FakeGitHubForCLI` above stubs `GitHub.pr_view` entirely and returns every
# field the gate could ever want regardless of what was actually requested --
# it would pass even if `run_closing_keyword_check_command` asked `gh` for
# every PR_VIEW_FIELDS entry, including `statusCheckRollup`. That is exactly
# what shipped and broke CI: the real `GitHub.pr_view()` requested
# `statusCheckRollup`, which GraphQL: Resource not accessible by integration
# under the default Actions GITHUB_TOKEN (observed live, run 30609781476,
# before `checks: read` had been granted at all). Rather than grant that
# scope and re-run to see whether it's sufficient for this nested
# connection, the fix removes the need for the field entirely. The tests
# below exercise the REAL `GitHub.pr_view()` -> `GitHub.run()` call path
# (only the lowest-level subprocess wrapper is mocked) so a regression back
# to the wide query fails loudly here instead of only in CI.


def test_closing_keyword_pr_fields_excludes_statuscheckrollup() -> None:
    # Guards the query contract directly: whatever CLOSING_KEYWORD_PR_FIELDS
    # is defined as, it must never grow to include the field that triggered
    # the token-scope failure -- widening it back is the exact regression
    # this issue is about.
    requested = set(CLOSING_KEYWORD_PR_FIELDS.split(","))
    assert "statusCheckRollup" not in requested
    assert requested == {"title", "body", "headRefName", "isCrossRepository"}


def test_cli_closing_keyword_check_queries_narrow_pr_view_fields_end_to_end(
    monkeypatch, tmp_path
) -> None:
    def fake_run(self, args, *, json_output=False, allow_failure=False):
        if args[:2] == ["pr", "view"]:
            fields = set(args[args.index("--json") + 1].split(","))
            # This is the load-bearing assertion: a fake that unconditionally
            # returns every field (like _FakeGitHubForCLI above) can't catch
            # a regression to the wide PR_VIEW_FIELDS query -- this one
            # inspects the actual `gh pr view --json <fields>` argv the CLI
            # builds, the same argv `gh` itself would reject the
            # statusCheckRollup portion of under a restricted Actions token.
            assert fields == {"title", "body", "headRefName", "isCrossRepository"}, (
                "closing-keyword-check must query CLOSING_KEYWORD_PR_FIELDS only -- "
                f"got {sorted(fields)}, which would re-trigger the statusCheckRollup "
                "GraphQL failure from run 30609781476"
            )
            return {
                "title": "",
                "body": "Fixes #999999",
                "headRefName": "agent/issue-999999-do-thing",
                "isCrossRepository": False,
            }
        if args[:1] == ["api"]:
            return [{"commit": {"message": "fix: unrelated cleanup\n\nFixes #999997 as well"}}]
        raise AssertionError(f"unexpected gh invocation in this test: {args}")

    monkeypatch.setattr(GitHub, "run", fake_run)
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    # Deliberately NOT monkeypatching cli_module.GitHub here -- the real
    # GitHub class must be constructed so its real pr_view()/pr_commits()
    # methods build the argv under test, with only the subprocess-level
    # run() mocked out.

    result = cli_module.run_closing_keyword_check_command(_cli_args(999992))

    assert result.ok is False
    assert result.data["findings"] == [
        {"issue_number": 999997, "source": "commit #1", "matched_text": "Fixes #999997"}
    ]
