"""Exemption-claim escape for the janitor's no-op rework gate (issue #2281).

The test-adequacy gate's own prescribed remedy -- a ``Test-exempt:``
trailer on an empty commit -- produces no diff delta, so the patch-id
comparison alone classified the remedy as a no-op and escalated the gate's
own fix (swole #487 / PR #490). The escape in
``no_op_rework_body._trailer_exempt_escape_warning`` compares the live
exemption claim against the claim at the reviewed head and satisfies the
gate when the claim is new. Split out of
``tests/test_janitor_no_op_rework.py`` to keep that file under the
file-size-ratchet cap; helpers are inlined per the #1284 self-containment
rule enforced by ``tests/test_zero_cross_test_import_guard.py``.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _janitor_fixtures import _config, _green_checks, _green_pr

from charlie_work.config import OrchestratorConfig, TestAdequacyConfig
from charlie_work.janitor import (
    _calculate_patch_id,
    no_op_escape_needs_pr_commits,
    run_janitor,
)
from charlie_work.test_adequacy_exempt import newly_claimed_exempt_reason


def _adequacy_on_config() -> OrchestratorConfig:
    """The janitor fixture config with the test-adequacy gate enabled.

    ``_config()`` leaves ``TestAdequacyConfig.enabled`` at its False default;
    the issue #2281 exemption-claim escape is only live while the gate that
    sanctions the marker channel runs.
    """
    config = _config()
    return replace(config, test_adequacy=replace(config.test_adequacy, enabled=True))


def _commit(sha: str, message: str) -> dict:
    """One ``pulls/{n}/commits`` REST entry: top-level ``sha`` + ``commit.message``."""
    return {"sha": sha, "commit": {"message": message}}


def _body_sha256(body: str | None) -> str:
    """The ``reviewed_body_sha256`` wire contract (issue #1939): SHA-256 of the
    PR body with CRLF/CR line endings normalized to LF, ``None`` treated as
    the empty string. Computed here rather than imported so the tests pin the
    contract, not the implementation that happens to produce it."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_NO_OP_DIFF = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""

_UNCHANGED_BODY = "Closes #123.\n\nTests: added unit tests for the search path."


def _no_op_pr_state(diff: str) -> dict:
    """A request_changes verdict whose diff matches the live one exactly.

    Mirrors the test-adequacy auto-reject shape from the issue report:
    the reviewed head predates the rework commits, the patch-id is
    unchanged (the rework is trailer-only), and the body baseline matches
    the live body so the #1939 escape is not in play.
    """
    return {
        "decision": "request_changes",
        "reviewed_head_sha": "aaa111",
        "reviewed_patch_id": _calculate_patch_id(diff),
        "reviewed_body_sha256": _body_sha256(_UNCHANGED_BODY),
    }


def test_no_op_rework_exempt_trailer_commit_escapes_gate(tmp_path: Path) -> None:
    """Issue #2281: an empty commit carrying a valid ``Test-exempt:`` trailer
    is the test-adequacy gate's own remedy -- it produces no diff delta, so
    the patch-id comparison alone flags it as a no-op (swole #487 / PR
    #490). A live exemption claim the reviewed head did not already make is
    substantive rework and must satisfy the gate.

    MUTATION CHECK: MUST FAIL against the pre-fix implementation, which had
    no exemption-claim channel next to the #1939 body escape."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)
    pr_commits = [
        _commit("aaa111", "feat: add feature without tests"),
        _commit("bbb222", "docs: progress note"),
        _commit(
            "ccc333",
            "chore: claim test-adequacy exemption\n\nTest-exempt: pure refactor, no behavior change",
        ),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=pr_commits,
    )

    assert verdict.ok is True, (
        f"Expected the exemption-claim escape to satisfy the gate, got {verdict.failures}"
    )
    assert not verdict.is_no_op_rework
    assert not any("unchanged since request_changes verdict" in f for f in verdict.failures)
    assert any("test-adequacy" in w and "#2281" in w for w in verdict.warnings)


def test_no_op_rework_empty_commit_without_trailer_still_blocks(tmp_path: Path) -> None:
    """Positive control (issue #2281 acceptance): an empty rework commit
    that carries NO exemption claim is a genuine no-op -- the patch-id
    failure must still fire."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)
    pr_commits = [
        _commit("aaa111", "feat: add feature without tests"),
        _commit("ccc333", "chore: empty rework commit"),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=pr_commits,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_trailer_at_reviewed_head_still_blocks(tmp_path: Path) -> None:
    """Issue #2281 acceptance: an exemption already claimed AT the reviewed
    head must not escape -- the trailer did not arrive with the rework."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)
    pr_commits = [
        _commit(
            "aaa111",
            "feat: add feature without tests\n\nTest-exempt: claimed before the verdict",
        ),
        _commit("ccc333", "chore: empty rework commit"),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=pr_commits,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_body_marker_already_claimed_still_blocks(tmp_path: Path) -> None:
    """Issue #2281 acceptance: when the unchanged body already carried the
    marker at the reviewed head, a new trailer commit adds no new claim --
    the escape must stay shut."""
    body = f"{_UNCHANGED_BODY}\n\nTest-exempt: claimed at verdict time"
    pr_state = {
        "decision": "request_changes",
        "reviewed_head_sha": "aaa111",
        "reviewed_patch_id": _calculate_patch_id(_NO_OP_DIFF),
        "reviewed_body_sha256": _body_sha256(body),
    }
    pr = _green_pr(headRefOid="ccc333", body=body)
    pr_commits = [
        _commit("aaa111", "feat: add feature without tests"),
        _commit("ccc333", "chore: empty rework\n\nTest-exempt: claimed at verdict time"),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=pr_commits,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_exempt_escape_needs_enabled_gate(tmp_path: Path) -> None:
    """The ``Test-exempt:`` marker is only a sanctioned channel while the
    test-adequacy gate runs; with ``enabled=False`` a trailer is vacuous and
    the rework stays a no-op."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)
    pr_commits = [
        _commit("aaa111", "feat: add feature without tests"),
        _commit("ccc333", "chore: claim exemption\n\nTest-exempt: gate disabled here"),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _config(),  # test_adequacy.enabled stays at its False default
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=pr_commits,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_exempt_escape_fails_closed_without_commits(tmp_path: Path) -> None:
    """No commit list (fetch failed or caller skipped it) means newness is
    undeterminable -- fail closed, same convention as a missing body
    baseline."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=None,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_exempt_escape_fails_closed_on_rewritten_history(tmp_path: Path) -> None:
    """A ``reviewed_head_sha`` absent from the live commit list (force-push
    or rebase replaced the reviewed history) makes the claim's newness
    undeterminable -- fail closed."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)
    pr_commits = [
        _commit("zzz999", "feat: rewritten base commit"),
        _commit("ccc333", "chore: claim exemption\n\nTest-exempt: after a force-push"),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=pr_state,
        pr_commits=pr_commits,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


def test_no_op_rework_exempt_escape_reads_reviewed_head_from_decision(tmp_path: Path) -> None:
    """The escape's anchor is the verdict's ``reviewed_head_sha`` (read
    file-first like the ``decision`` field above it, falling back to the
    pr_state mirror), not whichever commit happens to be newest."""
    pr_state = _no_op_pr_state(_NO_OP_DIFF)
    decision = dict(pr_state, reviewed_head_sha="bbb222")
    pr = _green_pr(headRefOid="ccc333", body=_UNCHANGED_BODY)
    pr_commits = [
        _commit("aaa111", "feat: add feature without tests"),
        _commit("bbb222", "chore: claim exemption\n\nTest-exempt: claimed at review time"),
        _commit("ccc333", "chore: empty rework commit"),
    ]

    verdict = run_janitor(
        pr,
        _green_checks(),
        _adequacy_on_config(),
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=_NO_OP_DIFF,
        review_decision=decision,
        pr_commits=pr_commits,
    )

    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)


# --- no_op_escape_needs_pr_commits: the caller-side fetch predicate -----------
#
# Truth table over the three inputs the predicate reads: adequacy-gate
# enabled, the resolved decision, and (the head-not-advanced arm added in
# the #2281 rework so an escalated, head-frozen request_changes PR does not
# cost a REST call every loop pass) the live head vs the verdict's
# ``reviewed_head_sha``.

_RC = {"decision": "request_changes", "reviewed_head_sha": "aaa111"}


@pytest.mark.parametrize(
    ("enabled", "decision", "live_head", "expected"),
    [
        # Escape live: gate on + request_changes + head advanced.
        (True, dict(_RC), "ccc333", True),
        # Head unreadable/unknown cannot prove the escape dead -> still fetch.
        (True, dict(_RC), None, True),
        (True, dict(_RC), "", True),
        # A verdict without reviewed_head_sha cannot prove the escape dead
        # (the gate-side fallback reads pr_state's anchor, which this
        # predicate cannot see) -> still fetch.
        (True, {"decision": "request_changes"}, "ccc333", True),
        # Head still pinned at the reviewed commit: no post-verdict commit
        # exists to carry a claim, so the escape can never fire -> no fetch.
        (True, dict(_RC), "aaa111", False),
        # Non-request_changes decisions: nothing for the escape to compare.
        (True, {"decision": "approved"}, "ccc333", False),
        (True, {"decision": "blocked"}, "ccc333", False),
        (True, {"decision": "missing"}, "ccc333", False),
        (True, {}, "ccc333", False),
        (True, None, "ccc333", False),
        # Gate disabled: the marker channel is unsanctioned -> never fetch.
        (False, dict(_RC), "ccc333", False),
        (False, dict(_RC), "aaa111", False),
        (False, {"decision": "approved"}, "ccc333", False),
        (False, None, None, False),
    ],
)
def test_no_op_escape_needs_pr_commits_truth_table(
    enabled: bool,
    decision: dict[str, Any] | None,
    live_head: str | None,
    expected: bool,
) -> None:
    """``no_op_escape_needs_pr_commits`` returns True only when the
    exemption-claim escape is live: gate enabled + request_changes verdict
    + head not provably still at the reviewed commit."""
    config = TestAdequacyConfig(enabled=enabled, exempt_marker="Test-exempt:")
    assert no_op_escape_needs_pr_commits(config, decision, live_head) is expected


# --- newly_claimed_exempt_reason: direct malformed-input coverage --------------


def test_newly_claimed_exempt_reason_returns_post_head_claim() -> None:
    """Positive control: a trailer first appearing after the reviewed head
    is returned verbatim."""
    commits = [
        _commit("aaa111", "feat: add feature without tests"),
        _commit("bbb222", "chore: rework\n\nTest-exempt: pure refactor"),
    ]
    assert (
        newly_claimed_exempt_reason("Closes #123.", commits, "aaa111", "Test-exempt:")
        == "pure refactor"
    )


@pytest.mark.parametrize(
    ("body", "commits", "reviewed_head", "marker"),
    [
        # Empty commit list: no evidence either way.
        ("Closes #123.", [], "aaa111", "Test-exempt:"),
        # Empty reviewed_head_sha: no anchor to compare against.
        (
            "Closes #123.",
            [_commit("aaa111", "x\n\nTest-exempt: claim")],
            "",
            "Test-exempt:",
        ),
        # Empty marker: no sanctioned channel exists.
        (
            "Closes #123.",
            [_commit("aaa111", "x"), _commit("bbb222", "y\n\nTest-exempt: claim")],
            "aaa111",
            "",
        ),
        # Reviewed head absent from the list (force-push/rebase replaced the
        # reviewed history): newness is undeterminable.
        (
            "Closes #123.",
            [_commit("zzz999", "x"), _commit("bbb222", "y\n\nTest-exempt: claim")],
            "aaa111",
            "Test-exempt:",
        ),
        # Claim already present at the reviewed head is not new.
        (
            "Closes #123.",
            [
                _commit("aaa111", "x\n\nTest-exempt: old claim"),
                _commit("bbb222", "chore: rework"),
            ],
            "aaa111",
            "Test-exempt:",
        ),
        # No claim anywhere in the window.
        (
            "Closes #123.",
            [_commit("aaa111", "x"), _commit("bbb222", "chore: rework")],
            "aaa111",
            "Test-exempt:",
        ),
    ],
)
def test_newly_claimed_exempt_reason_fails_closed(
    body: str,
    commits: list,
    reviewed_head: str,
    marker: str,
) -> None:
    assert newly_claimed_exempt_reason(body, commits, reviewed_head, marker) == ""


def test_newly_claimed_exempt_reason_tolerates_malformed_commit_entries() -> None:
    """Non-Mapping entries and non-Mapping ``commit`` payloads degrade to an
    empty message rather than raising -- matching the REST shape FakeGitHub
    and ``GitHub.pr_commits`` can produce around API edge cases."""
    marker = "Test-exempt:"
    commits = [
        "not-a-mapping",
        42,
        None,
        _commit("aaa111", "feat: base"),
        {"sha": "bbb222"},  # no commit key
        {"sha": "ccc333", "commit": None},  # commit: null
        {"sha": "ddd444", "commit": "not-a-mapping"},  # commit non-mapping
        _commit("eee555", "chore: rework\n\nTest-exempt: still found"),
    ]
    assert newly_claimed_exempt_reason("", commits, "aaa111", marker) == "still found"
    # A malformed entry that still carries the head sha anchors the split.
    head_is_malformed = [
        {"sha": "aaa111", "commit": None},
        _commit("bbb222", "chore: rework\n\nTest-exempt: after malformed head"),
    ]
    assert (
        newly_claimed_exempt_reason("", head_is_malformed, "aaa111", marker)
        == "after malformed head"
    )
