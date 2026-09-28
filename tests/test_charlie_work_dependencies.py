"""Issue-dependency detection: prose-only body references and the GitHub dependency fetch path.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from _fakes_github import FakeGitHub
from charlie_work import github as github_module


def test_detect_prose_only_dependencies_do_not_dispatch_before() -> None:
    """Test detection of 'do not dispatch before' pattern (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "Do not dispatch before P2-T2/P2-T3 have landed."
    assert detect_prose_only_dependencies(body) is True

    body = "DO NOT DISPATCH BEFORE #123 merges"
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_task_references() -> None:
    """Test detection of task references like P2-T3 (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "This depends on P2-T2 and P2-T3."
    assert detect_prose_only_dependencies(body) is True

    body = "Wait for P1-T5 to complete first."
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_wait_for_pr() -> None:
    """Test detection of 'wait for PR' pattern (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "Wait for this PR to merge before starting."
    assert detect_prose_only_dependencies(body) is True

    body = "wait for that PR to land"
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_with_structured_blockers() -> None:
    """Test that issues with structured blockers are handled correctly (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "Blocked by #123"
    # Has structured blockers, so prose-only detection returns False
    # (doesn't match the prose patterns)
    assert detect_prose_only_dependencies(body) is False
    assert parse_blockers(body) == [123]

    # Test case with both prose and structured blockers
    body = "Do not dispatch before P2-T2. Blocked by #123"
    # Has prose pattern, so detection returns True
    # But caller will check parse_blockers and see structured blockers exist
    assert detect_prose_only_dependencies(body) is True
    assert parse_blockers(body) == [123]


def test_detect_prose_only_dependencies_no_match() -> None:
    """Test that normal issue bodies don't trigger false positives (issue #225)."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "This is a normal issue with no dependencies."
    assert detect_prose_only_dependencies(body) is False

    body = "Fix the bug in the authentication module."
    assert detect_prose_only_dependencies(body) is False

    body = ""
    assert detect_prose_only_dependencies(body) is False


def test_detect_prose_only_dependencies_descriptive_task_refs_no_match() -> None:
    """REQUIRED 1 negative tests: bare/descriptive task-marker mentions must NOT match.

    Plan-generated issue bodies routinely mention task markers descriptively
    (e.g. 'implements P2-T4', title suffix '(P2-T4)', 'this task is P2-T3 of
    the plan'). These must not fire the detector — only dependency-context uses
    should match (REQUIRED 1, PR #230 rework).
    """
    from charlie_work.github import detect_prose_only_dependencies

    # "implements P2-T4" — descriptive, not a dependency declaration
    body = "This implements P2-T4 of the expiry plan."
    assert detect_prose_only_dependencies(body) is False

    # Commit-style title with inline task marker — descriptive reference in prose
    body = "fix(expiry): thread careers-match (P2-T4)"
    assert detect_prose_only_dependencies(body) is False

    # "this task is P2-T3 of the plan" — describes what the issue is, not what it depends on
    body = "This task is P2-T3 of the plan."
    assert detect_prose_only_dependencies(body) is False


def test_detect_prose_only_dependencies_ignores_fenced_code_block() -> None:
    """Issue #1819: dependency-shaped prose inside a fenced code block is
    quoted/example code, not the issue author's own declaration — the
    function previously scanned the raw body with no fence awareness at
    all, so a code sample quoting "wait until the PR merges"-style text
    parked the issue under ``agent:prose-only-deps``."""
    from charlie_work.github import detect_prose_only_dependencies

    # Pattern 3 ("wait for ... PR/merge") quoted inside a fenced sample.
    body = (
        "This parser parks issues on prose like:\n\n"
        "```python\n"
        "# example: wait for this PR to merge before starting\n"
        "# or 'depends on P2-T2' in a comment\n"
        "```\n"
    )
    assert detect_prose_only_dependencies(body) is False

    body = "```\nDo not dispatch before P2-T2 lands.\n```\n"
    assert detect_prose_only_dependencies(body) is False


def test_detect_prose_only_dependencies_outside_fenced_block_still_fires() -> None:
    """The fenced-code exclusion must not over-suppress: a genuine
    dependency declaration in prose still parks the issue."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "```text\nexample: wait for this PR to merge\n```\n\nDepends on P2-T2 now.\n"
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_section_after_fenced_block_still_scanned() -> None:
    """Issue #1819: ``_scan_blocker_sections`` re-runs the fence scan on the
    stripped body, so the strip must remove the closing fence line too —
    an orphaned closer would be re-read as an opening fence and swallow a
    real 'Blocked by' section after the code block."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "```\ncode\n```\n## Blocked by\n- https://example.com/unreadable\n"
    assert parse_blockers(body) == []
    assert detect_prose_only_dependencies(body) is True


# --- Issue #1949: issue-number ordering prose -------------------------------


def test_detect_prose_only_dependencies_issue_ref_ordering() -> None:
    """Issue #1949: ordering prose next to a same-repo issue number is a
    prose-only dependency — ``parse_blockers`` reads no blocker from these
    shapes, so the issue used to dispatch immediately, out of order."""
    from charlie_work.github import detect_prose_only_dependencies, parse_blockers

    body = "Do this after #12 lands."
    assert parse_blockers(body) == []
    assert detect_prose_only_dependencies(body) is True

    body = "Wait for #12 to merge first."
    assert parse_blockers(body) == []
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_issue_ref_ordering_variants() -> None:
    """Issue #1949: the ordering-phrase family — 'wait for'/'requires' + ref
    (no completion verb required) and 'before/until/after' + ref + verb. A
    multi-ref run contributes every ref it names."""
    from charlie_work.github import detect_prose_only_dependencies

    assert detect_prose_only_dependencies("Wait for #12.") is True
    assert detect_prose_only_dependencies("This requires #12.") is True
    assert detect_prose_only_dependencies("Start only after #12 and #13 land.") is True
    assert detect_prose_only_dependencies("Hold until the fix in #12 is done.") is True
    assert detect_prose_only_dependencies("Land this before #12 ships.") is True
    # "Depends on #N" is a structured declaration that parse_blockers reads —
    # it only counts as prose-only when the parser's own guards void it (a
    # foreign ref in the clause makes the clause describe other issues;
    # parking for a human is the fail-safe).
    assert detect_prose_only_dependencies("Depends on #5, see #6 for context") is True


def test_detect_prose_only_dependencies_issue_ref_declared_edge_not_flagged() -> None:
    """Issue #1949 acceptance: the same prose whose ref is already declared
    as a blocker is not prose-only — the blocker-set comparison is what keeps
    legitimate bodies passing."""
    from charlie_work.github import detect_prose_only_dependencies

    assert (
        detect_prose_only_dependencies("Do this after #12 lands.\n\n## Blocked by\n- #12\n")
        is False
    )
    # An inline "Blocked by #N" declaration covers the prose mention too.
    assert detect_prose_only_dependencies("Wait for #12 to merge first. Blocked by #12") is False
    # "Depends on #N" is itself a structured declaration — it reads cleanly,
    # so it is never prose-only.
    assert detect_prose_only_dependencies("This depends on #12.") is False
    # A *different* declared blocker does not cover the prose edge on #12.
    assert (
        detect_prose_only_dependencies("Wait for #12 to merge first.\n\n## Blocked by\n- #34\n")
        is True
    )
    # A multi-ref run with one undeclared member still flags.
    assert (
        detect_prose_only_dependencies("Wait for #12 and #13 to merge.\n\n## Blocked by\n- #12\n")
        is True
    )


def test_detect_prose_only_dependencies_issue_ref_ordering_quoted() -> None:
    """Issue #1949: the quoting guards ``parse_blockers`` applies — fenced
    code block, inline code span, double-quote span, blockquote line — apply
    to the new issue-ref ordering patterns too."""
    from charlie_work.github import detect_prose_only_dependencies

    # Mutation anchor: unquoted, the same ordering prose flags.
    assert detect_prose_only_dependencies("Wait for #12 to merge first.") is True
    assert (
        detect_prose_only_dependencies(
            "Example prose that parks an issue:\n\n```\nWait for #12 to merge first.\n```\n"
        )
        is False
    )
    assert (
        detect_prose_only_dependencies("The lint phrase is `wait for #12 to merge first`.")
        is False
    )
    assert (
        detect_prose_only_dependencies(
            'Authors write "wait for #12 to merge first" to park a ticket.'
        )
        is False
    )
    assert (
        detect_prose_only_dependencies("> Wait for #12 to merge first.\nNormal body line.")
        is False
    )
    # A match that STARTS inside a quoted span is quoted prose even when a
    # second ref outside the span extends the match (stale-comment shape from
    # the rollout corpus: the comment says "Ignored until #575 lands").
    assert (
        detect_prose_only_dependencies(
            'The comment says "Ignored until #575 lands" — #575 has landed. Update it.'
        )
        is False
    )


def test_detect_prose_only_dependencies_existing_patterns_quoted() -> None:
    """Issue #1949: the pre-existing patterns get the same quoting guards —
    an issue that quotes the do-not-dispatch phrase to describe the detector
    must not park itself (drafting the issue body tripped this)."""
    from charlie_work.github import detect_prose_only_dependencies

    assert (
        detect_prose_only_dependencies('The detector fires on "do not dispatch before" in a body.')
        is False
    )
    assert (
        detect_prose_only_dependencies("The detector fires on `do not dispatch before` in a body.")
        is False
    )
    assert (
        detect_prose_only_dependencies("> do not dispatch before P2-T2 lands.\nNotes below.")
        is False
    )


def test_detect_prose_only_dependencies_quoted_match_skipped_not_fatal() -> None:
    """A quoted ordering match is skipped, not terminal for the scan: a later
    genuine declaration in the same body still flags."""
    from charlie_work.github import detect_prose_only_dependencies

    body = "The lint watches for `wait for #12 to merge`.\n\nWait for #34 to merge first."
    assert detect_prose_only_dependencies(body) is True


def test_detect_prose_only_dependencies_ordering_tightening() -> None:
    """Issue #1949 rollout tightening — each case below was a real false
    positive in the fleet's open-issue corpus:

    - an ``owner/repo#N`` reference is not a *same-repo* issue reference;
    - bare past-tense narration ("after PR #920 merged") reports history, it
      does not order this issue;
    - a long/comma-separated gap between "requires"/"depends on" and the ref
      is incidental mention;
    - "does not wait for" is an explicit non-dependency;
    - "incomplete" contains "complete" but is not a completion verb;
    - noun-form "merge" ("#990's merge", "the #1500 merge") names the merge
      event, it does not order the issue behind it;
    - "this merged" reads "is merged" inside "this" — auxiliaries are
      word-anchored (rework finding).
    """
    from charlie_work.github import detect_prose_only_dependencies

    # Cross-repo ordering — spec scopes the detector to same-repo refs.
    assert (
        detect_prose_only_dependencies("Install only after Senkichi/charlie-work#1540 has landed.")
        is False
    )
    # Past-tense narration (all observed as FPs in the fleet corpus).
    assert detect_prose_only_dependencies("Observed after PR #920 merged.") is False
    assert detect_prose_only_dependencies("The bug was forked before #312 merged.") is False
    # The auxiliary alternative is word-anchored: "is" inside "this" is not
    # the auxiliary ("thIS merged" used to read as "is merged").
    assert detect_prose_only_dependencies("Observed after PR #920 in this merged state.") is False
    # Auxiliary + participle still reads as ordering, not narration.
    assert detect_prose_only_dependencies("Do not start until #12 is merged.") is True
    assert detect_prose_only_dependencies("Do not start until #12 has landed.") is True
    # Incidental requires/depends-on separated by a comma clause.
    assert (
        detect_prose_only_dependencies(
            "This requires a clean tree, careful sequencing, and review (#357)."
        )
        is False
    )
    # Explicit negation.
    assert detect_prose_only_dependencies("This issue does not wait for #207.") is False
    # "incomplete" is not the verb "complete".
    assert (
        detect_prose_only_dependencies("Check after #9 whether the result is incomplete") is False
    )
    # Noun-form "merge" is the event noun, not a completion verb
    # (rollout step 3 — charlie-work#1504, job-cannon#991 FPs):
    # a possessive clitic or a determiner directly before the ref marks it.
    assert detect_prose_only_dependencies("Verified against main after #990's merge.") is False
    assert detect_prose_only_dependencies("Field created_at after the #1500 merge.") is False
    # …but a verb "merge"/"merges" still flags (plural-subject ordering, and
    # a possessive on the ref ordering its dependents' merge).
    assert detect_prose_only_dependencies("Dispatch after #336 and #329 merge.") is True
    assert detect_prose_only_dependencies("Hold until #12's dependents merge.") is True
    # A genuine verb later in the sub-clause is not swallowed by the noun.
    assert detect_prose_only_dependencies("Hold until the #12 merge completes.") is True
    # A determiner before the ref does not make "merge" a noun: an
    # intervening subject ("the #12 PR merges" — "the" marks the PR) or
    # verb agreement under a determiner ("the #12 merges") is still
    # ordering prose. The noun token must directly follow the ref run or
    # the possessive clitic (round-3 rework repro cases).
    assert detect_prose_only_dependencies("Start after the #12 PR merges.") is True
    assert detect_prose_only_dependencies("Do not start until the #12 fix merges.") is True
    assert detect_prose_only_dependencies("Hold until the #12 work merges.") is True
    assert detect_prose_only_dependencies("Land after the #12 change merges.") is True
    assert detect_prose_only_dependencies("Hold until the #12 PR merges and CI is green.") is True
    assert detect_prose_only_dependencies("Ship after the #12 merges.") is True


def test_detect_prose_only_dependencies_ordering_negation_boundary() -> None:
    """Issue #1949 rework: ``_NEGATED_ORDERING_RE`` is word-anchored — a word
    that merely *ends* in a negation token is not a negation — while real
    negations (``not``/``n't``/``never``/``cannot``/``no longer``) still
    suppress the modal-verb shapes, and the negation-insensitive
    ``before/until/after`` shape is unaffected (negation_sensitive flag
    mutations killed in both directions)."""
    from charlie_work.github import detect_prose_only_dependencies

    # "casino" ends in "no" — not a negation; the ordering still flags.
    assert detect_prose_only_dependencies("Skip the casino wait for #12 to finish.") is True
    # "no longer" suppression on the negation-sensitive wait-for shape.
    assert detect_prose_only_dependencies("It will no longer wait for #12.") is False
    # "cannot" keeps suppressing — it matched via a "not" substring before
    # the boundary anchor; it is now an explicit negation alternative.
    assert detect_prose_only_dependencies("This issue cannot wait for #12.") is False
    assert detect_prose_only_dependencies("This issue doesn't wait for #12.") is False
    # The before/until/after shape is NOT negation-sensitive: an earlier
    # "not" in the clause does not free the ordering edge — not even a
    # negation immediately before the keyword ("not until" still orders).
    assert detect_prose_only_dependencies("Do not merge before #12 lands.") is True
    assert detect_prose_only_dependencies("It is not until #12 lands that we can ship.") is True


def test_github_dependencies_404_tolerance(tmp_path: Path) -> None:
    """Test that 404 errors from dependencies API are handled gracefully (feature not available)."""
    from charlie_work.github import get_github_issue_dependencies

    class FakeGitHubWith404(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate 404 response (feature not available)
            self.dependencies_response = {"message": "Not Found", "status": 404}

    fake_gh = FakeGitHubWith404()
    result = get_github_issue_dependencies(fake_gh, 123)
    assert result == []


def test_github_dependencies_transient_error_fail_open(tmp_path: Path) -> None:
    """Test that transient errors from dependencies API fail open with warning."""
    from charlie_work.github import get_github_issue_dependencies
    from unittest.mock import patch

    class FakeGitHubWithTransientError(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate transient error (None return)
            self.dependencies_response = None

    fake_gh = FakeGitHubWithTransientError()

    # get_github_issue_dependencies moved to charlie_work.github_capabilities.issues
    # in L07 (issue #1591); its module logger moved with it, so the patch target
    # follows the symbol. The function object is still re-exported from
    # charlie_work.github (imported above), only its logger binding relocated.
    with patch("charlie_work.github_capabilities.issues.logger") as mock_logger:
        result = get_github_issue_dependencies(fake_gh, 123)
        assert result == []
        # Should have logged a warning about the transient error
        assert any(
            "returned None" in str(call.args[0]) for call in mock_logger.warning.call_args_list
        )


def test_github_dependencies_successful_parse(tmp_path: Path) -> None:
    """Test that successful dependencies API responses are parsed correctly."""
    from charlie_work.github import get_github_issue_dependencies

    class FakeGitHubWithDependencies(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate successful response with dependencies
            self.dependencies_response = [
                {"number": 100, "url": "https://example.test/issues/100"},
                {"number": 200, "url": "https://example.test/issues/200"},
            ]

    fake_gh = FakeGitHubWithDependencies()
    result = get_github_issue_dependencies(fake_gh, 123)
    assert result == [100, 200]


def test_get_github_issue_dependencies_caches_successful_result_per_pass(
    monkeypatch, tmp_path: Path
) -> None:
    """Issue #870: get_github_issue_dependencies() made one live `gh api`
    call per invocation, with zero caching -- called once per ready issue
    from _get_open_blockers, and _get_open_blockers itself is called once per
    issue from *two* separate places (_filter_blocked_issues and
    _summarize_issue), so every ready issue's dependencies were fetched
    twice per status() call. A successful resolution for a given issue
    number must now cost exactly one live call per pass.

    Uses the real GitHub (not FakeGitHub, whose `run()` has its own
    dependencies_response stand-in and doesn't exercise the production
    _list_cache path) with subprocess.run mocked.
    """
    from charlie_work.github import get_github_issue_dependencies

    calls: list[str] = []

    def fake_run(command, **kwargs):
        # command: ["gh", "api", "repos/{owner}/{repo}/issues/<N>/dependencies/blocked_by"]
        calls.append(command[2])
        payload = json.dumps([{"number": 200}])
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=payload, stderr="")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)
    gh = github_module.GitHub(repo_root=tmp_path)

    first = get_github_issue_dependencies(gh, 100)
    second = get_github_issue_dependencies(gh, 100)

    assert first == [200]
    assert second == [200]
    assert len(calls) == 1  # second call is a pure cache hit, no live fetch

    # A different issue number is a distinct cache key -- costs a fresh read.
    get_github_issue_dependencies(gh, 101)
    assert len(calls) == 2

    # invalidate_list_cache() (called once per orchestrator pass) must force
    # a fresh read on the next call for the same issue number.
    gh.invalidate_list_cache()
    get_github_issue_dependencies(gh, 100)
    assert len(calls) == 3


def test_github_dependencies_unexpected_type_fail_open(tmp_path: Path) -> None:
    """Test that unexpected return types from dependencies API fail open with warning."""
    from charlie_work.github import get_github_issue_dependencies
    from unittest.mock import patch

    class FakeGitHubWithUnexpectedType(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            # Simulate unexpected return type
            self.dependencies_response = "unexpected string"

    fake_gh = FakeGitHubWithUnexpectedType()

    # See the transient-error test above: the moved function's logger now lives
    # at charlie_work.github_capabilities.issues.logger (L07, issue #1591).
    with patch("charlie_work.github_capabilities.issues.logger") as mock_logger:
        result = get_github_issue_dependencies(fake_gh, 123)
        assert result == []
        # Should have logged a warning about the unexpected type
        assert any(
            "unexpected type" in str(call.args[0]) for call in mock_logger.warning.call_args_list
        )
