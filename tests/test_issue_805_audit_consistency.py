"""Consistency tests for the issue #805 unescalation audit document.

The audit in ``docs/audits/issue-805-unescalation-audit.md`` adjudicates 7
manually-unescalated merged PRs.  Issue #805 acceptance criterion 1 explicitly
permits three verdicts — SPURIOUS, SUBSTANTIVE, INSUFFICIENT-EVIDENCE — and states
that "a manufactured SPURIOUS is not" acceptable.

A row is only honestly SPURIOUS when this audit independently verified that no
unaddressed substantive finding shipped.  Citing the operator's self-report as
this audit's own verification is a manufactured SPURIOUS — exactly what the
issue disallows.  These tests guard the document's internal consistency so a
future edit cannot silently regress the fix from PR #1181's rework.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

AUDIT_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "audits" / "issue-805-unescalation-audit.md"
)

# The 7 PRs in scope, per issue #805.
EXPECTED_PRS = {540, 531, 503, 584, 585, 637, 630}

VALID_VERDICTS = {"SPURIOUS", "SUBSTANTIVE", "INSUFFICIENT-EVIDENCE"}


def _read_audit() -> str:
    return AUDIT_PATH.read_text(encoding="utf-8")


def _parse_table_verdicts(text: str) -> dict[int, str]:
    """Extract ``{pr_number: verdict}`` from the summary table."""
    verdicts: dict[int, str] = {}
    for line in text.splitlines():
        # Table rows look like:
        # | 540 | #480  | max_review_... | SPURIOUS | `load_ledger`... | Yes — rework |
        # Verdict is the 4th column — skip PR, Issue, Escalation reason.
        m = re.match(r"^\|\s*(\d{3})\s*\|[^|]*\|[^|]*\|\s*(\w[\w-]*)\s*\|", line)
        if m:
            pr = int(m.group(1))
            verdict = m.group(2)
            if pr in EXPECTED_PRS and verdict in VALID_VERDICTS:
                verdicts[pr] = verdict
    return verdicts


def _parse_section_verdicts(text: str) -> dict[int, str]:
    """Extract ``{pr_number: verdict}`` from per-PR section headers.

    Section headers look like: ``### PR 585 (issue #485) — INSUFFICIENT-EVIDENCE``
    """
    verdicts: dict[int, str] = {}
    for m in re.finditer(
        r"^### PR (\d{3}) \(issue #\d+\)\s*—\s*(\w[\w-]*)\s*$",
        text,
        re.MULTILINE,
    ):
        pr = int(m.group(1))
        verdict = m.group(2)
        if pr in EXPECTED_PRS and verdict in VALID_VERDICTS:
            verdicts[pr] = verdict
    return verdicts


def _section_text(text: str, pr: int) -> str:
    """Return the body of the per-PR section for *pr* (up to the next ``---``)."""
    pattern = rf"### PR {pr} \(issue #\d+\).*?^---"
    m = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    assert m is not None, f"No section found for PR {pr}"
    return m.group(0)


@pytest.fixture(scope="module")
def audit_text() -> str:
    return _read_audit()


class TestAuditCoverage:
    """All 7 PRs must be adjudicated in both the table and the per-PR sections."""

    def test_all_7_prs_in_table(self, audit_text: str) -> None:
        verdicts = _parse_table_verdicts(audit_text)
        assert set(verdicts) == EXPECTED_PRS, (
            f"Table is missing PRs: {EXPECTED_PRS - set(verdicts)}"
        )

    def test_all_7_prs_have_sections(self, audit_text: str) -> None:
        verdicts = _parse_section_verdicts(audit_text)
        assert set(verdicts) == EXPECTED_PRS, (
            f"Sections missing for PRs: {EXPECTED_PRS - set(verdicts)}"
        )

    def test_table_and_section_verdicts_agree(self, audit_text: str) -> None:
        table = _parse_table_verdicts(audit_text)
        sections = _parse_section_verdicts(audit_text)
        assert table == sections, f"Table verdicts {table} != section verdicts {sections}"


class TestVerdictHonesty:
    """A SPURIOUS verdict must be backed by this audit's independent verification.

    The review finding from PR #1181: row 585 presented the operator's self-report
    as this audit's independent verification — a manufactured SPURIOUS, which
    issue #805 explicitly disallows.  A row is only honestly SPURIOUS when the
    audit section contains its own ``git show`` verification (the method in
    step 3 of the issue), not when it cites an operator review as the basis.
    """

    @pytest.mark.parametrize("pr", sorted(EXPECTED_PRS))
    def test_spurious_row_has_independent_verification(self, audit_text: str, pr: int) -> None:
        verdicts = _parse_section_verdicts(audit_text)
        if verdicts.get(pr) != "SPURIOUS":
            pytest.skip(f"PR {pr} is {verdicts.get(pr)}, not SPURIOUS")
        section = _section_text(audit_text, pr)
        # The audit's independent verification method is ``git show <sha>:<path>``.
        # A section that only cites the operator's self-report without its own
        # git-show check is a manufactured SPURIOUS.
        assert "git show" in section, (
            f"PR {pr} is SPURIOUS but the section has no independent "
            f"``git show`` verification — citing an operator review as "
            f"this audit's verification is a manufactured SPURIOUS "
            f"(issue #805 acceptance criterion 1)."
        )

    def test_pr_585_is_insufficient_evidence(self, audit_text: str) -> None:
        """PR 585 must be INSUFFICIENT-EVIDENCE, not a manufactured SPURIOUS.

        The automated reviewer produced no verdict (session limit after 1 turn,
        0 tool calls), and this audit did not independently re-verify the
        docs-only PR's claims.  Per issue #805, INSUFFICIENT-EVIDENCE is the
        honest verdict.
        """
        verdicts = _parse_section_verdicts(audit_text)
        assert verdicts[585] == "INSUFFICIENT-EVIDENCE", (
            f"PR 585 verdict is {verdicts[585]}, expected INSUFFICIENT-EVIDENCE"
        )

    def test_pr_585_section_does_not_claim_independent_verification(self, audit_text: str) -> None:
        """The 585 section must not present operator self-report as audit verification."""
        section = _section_text(audit_text, 585)
        # The section must explicitly state the audit did NOT independently verify.
        assert "did not independently" in section.lower(), (
            "PR 585 section must state the audit did not independently verify, "
            "not present the operator's self-report as this audit's verification."
        )


class TestHeadlineConsistency:
    """The headline count must be consistent with the table verdicts."""

    def test_headline_mentions_insufficient_evidence_count(self, audit_text: str) -> None:
        """If any row is INSUFFICIENT-EVIDENCE, the headline must say so."""
        verdicts = _parse_table_verdicts(audit_text)
        ie_count = sum(1 for v in verdicts.values() if v == "INSUFFICIENT-EVIDENCE")
        headline = audit_text.split("## Headline result")[1].split("##")[0]
        if ie_count > 0:
            assert "INSUFFICIENT-EVIDENCE" in headline, (
                f"{ie_count} row(s) are INSUFFICIENT-EVIDENCE but the headline "
                f"does not mention it — the count is inconsistent."
            )
            assert "confirmed" in headline.lower(), (
                "Headline must qualify the count as 'confirmed' when some rows "
                "are INSUFFICIENT-EVIDENCE (indeterminate), not claim all 7."
            )

    def test_headline_spurious_count_matches_table(self, audit_text: str) -> None:
        """The headline's SPURIOUS count must match the table."""
        verdicts = _parse_table_verdicts(audit_text)
        spurious_count = sum(1 for v in verdicts.values() if v == "SPURIOUS")
        headline = audit_text.split("## Headline result")[1].split("##")[0]
        # The headline says "6 adjudicated SPURIOUS" — check it matches.
        m = re.search(r"(\d+)\s+adjudicated SPURIOUS", headline)
        assert m is not None, "Headline must state the SPURIOUS count"
        assert int(m.group(1)) == spurious_count, (
            f"Headline says {m.group(1)} SPURIOUS, table has {spurious_count}"
        )
