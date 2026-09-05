"""Recorded-vs-collected assertion gate (issue #1621).

Under load, a full ``pytest --junit-xml=<file>`` run can exit 0 with 0
failures and 0 errors yet write a well-formed junit file whose ``<testcase>``
count (and ``<testsuite tests=...>`` attribute) falls short of
``pytest --collect-only`` -- a contiguous alphabetic tail of ``tests/``
silently omitted, with nothing in the repo asserting that the number of tests
*recorded* equals the number *collected*.  This gate closes that gap: it
counts the ``<testcase>`` elements in a junit XML document and the tests
reported by a collect-only run, and fails when the two counts differ.

This is **enforcement**, not evidence: the CI step that runs it is part of the
already-required "Tests" job, so a mismatch fails the merge gate without any
new required-check name or coordinated three-place config edit.

Relationship to the #1538 collect-only gate
-------------------------------------------

The #1538 gate compares *leaf-name multisets* from ``pytest --collect-only``
output at base vs head to approve verbatim test relocations; its parser
(:func:`charlie_work.collect_only_gate.parse_collect_only_output`) extracts
one node ID per line from the **non-``-q``** collect-only format and splits
each into a leaf name.  This gate does **not** reparse node IDs into leaf
names -- it only needs a *count*, so it reads the count pytest itself reports
(the ``N tests collected`` summary line when present, else the sum of the
``file: count`` lines that ``-q`` produces, else a node-ID line count).  That
is a different operation on a different facet of the same output, not a
second leaf-name parser; when #1538 merges, the collected count here can be
cross-checked against ``sum(collect_leaf_names(output)[0].values())`` for
defence in depth, but no divergence is possible today because #1538 is not on
``main``.

This module is the **pure scanning logic** -- no I/O, no git, no subprocess.
The CLI command layer (:mod:`charlie_work.junit_recorded_gate_command`) owns
the file reads and the exit-code decision, following the same split as
:mod:`charlie_work.collect_only_gate` / :mod:`charlie_work.mojibake_gate`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data model (frozen, per CLAUDE.md invariant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JunitCount:
    """The testcase count extracted from a junit XML document.

    ``testcase_count`` is the number of ``<testcase>`` elements across every
    ``<testsuite>`` -- the figure the issue (#1621) names as "recorded".
    ``suite_tests_attr`` is the sum of the ``tests=`` attributes on every
    ``<testsuite>`` element (pytest's own recorded total).  The two should
    agree; a mismatch between them is an *internal* junit inconsistency
    (``internal_consistency_ok=False``) reported separately from the
    recorded-vs-collected comparison, because it indicates the junit writer
    itself lost records rather than the suite being truncated mid-run.
    """

    testcase_count: int
    suite_tests_attr: int
    internal_consistency_ok: bool


@dataclass(frozen=True)
class JunitRecordedFinding:
    """A single discrepancy found by the recorded-vs-collected gate.

    ``kind`` is one of:

    * ``"recorded_vs_collected"`` -- the junit ``<testcase>`` count differs
      from the collect-only count (the load-truncation failure mode #1621
      documents).
    * ``"junit_internal"`` -- the junit ``<testcase>`` count differs from the
      ``<testsuite tests=...>`` attribute sum, an internal junit inconsistency
      surfaced for diagnosis independent of the collected comparison.
    * ``"empty_junit"`` -- the junit document parsed but contained zero
      ``<testcase>`` elements (and zero ``tests`` attribute), which is almost
      always a wiring error (wrong file, empty run) rather than truncation.
    * ``"empty_collect"`` -- the collect-only output yielded zero tests, which
      means collection itself failed or the wrong output was captured.
    """

    kind: str
    detail: str


@dataclass(frozen=True)
class JunitRecordedResult:
    """The full output of the recorded-vs-collected comparison.

    ``ok`` is ``True`` only when the recorded and collected counts are equal
    AND the junit document is internally consistent AND neither side is
    emptily zero.  ``findings`` is the tuple of
    :class:`JunitRecordedFinding` discrepancies; an empty tuple means the
    gate passed.
    """

    recorded: int
    collected: int
    suite_tests_attr: int
    findings: tuple[JunitRecordedFinding, ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when the gate passed (no findings)."""
        return len(self.findings) == 0


# ---------------------------------------------------------------------------
# Junit counting -- pure function over the XML text
# ---------------------------------------------------------------------------


def count_junit_testcases(junit_xml: str) -> JunitCount:
    """Count ``<testcase>`` elements and sum ``<testsuite tests=...>`` attrs.

    Parses *junit_xml* with :mod:`xml.etree.ElementTree` and counts every
    ``testcase`` element anywhere in the tree (pytest-xdist can emit multiple
    ``<testsuite>`` blocks, one per worker, so a root-level walk is required
    rather than reading a single suite).  The ``tests`` attribute on each
    ``<testsuite>`` is summed separately as pytest's own recorded total.

    Returns a :class:`JunitCount` with ``internal_consistency_ok=False`` when
    the element count and the attribute sum disagree -- a signal that the
    junit writer itself lost records, independent of the collected comparison.

    Never raises on a malformed document: a parse failure returns a zero
    count with ``internal_consistency_ok=False``, which the comparison surfaces
    as an ``empty_junit`` / ``recorded_vs_collected`` finding rather than a
    Python traceback.  This matches the errors-as-values invariant: the
    command layer turns a zero recorded count into a non-zero exit, not an
    exception.
    """
    try:
        root = ET.fromstring(junit_xml)
    except ET.ParseError:
        return JunitCount(
            testcase_count=0,
            suite_tests_attr=0,
            internal_consistency_ok=False,
        )
    testcase_count = sum(1 for _ in root.iter("testcase"))
    suite_tests_attr = 0
    for suite in root.iter("testsuite"):
        attr = suite.get("tests")
        if attr is None:
            continue
        try:
            suite_tests_attr += int(attr)
        except ValueError:
            continue
    # When every suite lacks a tests attribute, fall back to treating the
    # element count as the attribute sum so the internal-consistency check
    # does not spuriously fire on a non-pytest junit variant.
    if suite_tests_attr == 0 and testcase_count > 0:
        suite_tests_attr = testcase_count
    internal_ok = testcase_count == suite_tests_attr
    return JunitCount(
        testcase_count=testcase_count,
        suite_tests_attr=suite_tests_attr,
        internal_consistency_ok=internal_ok,
    )


# ---------------------------------------------------------------------------
# Collected counting -- pure function over collect-only output text
# ---------------------------------------------------------------------------

# ``N tests collected`` (optionally followed by `` in Xs``).  Matches the
# summary line pytest emits at the end of a non-``-q`` collect-only run, and
# the equivalent ``N tests in M seconds`` / ``N tests deselected`` forms that
# appear in some pytest configurations.  Anchored to the start of the line so
# a test path containing the substring cannot match.
_COLLECTED_SUMMARY_RE = re.compile(
    r"^\s*(\d+)\s+tests?\s+(?:collected|in\s+[\d.]+s|deselected)",
    re.IGNORECASE,
)

# ``path/to/test_x.py: N`` -- the per-file line pytest 9.x emits in ``-q``
# collect-only mode (a compact ``file: count`` format with no individual node
# IDs).  The path may contain colons only after a drive letter on Windows
# (``C:\...``), but pytest emits repo-relative forward-slash paths here, so
# the last ``:`` is the count separator.
_FILE_COUNT_RE = re.compile(r"^.+?:\s*(\d+)\s*$")


def count_collected_tests(collect_output: str) -> int:
    """Count the tests pytest reports collected from collect-only output.

    Three formats are recognised, in priority order:

    1. **Summary line** (``N tests collected``, optionally `` in Xs``): the
       authoritative count pytest itself computed.  Present at the end of a
       non-``-q`` ``--collect-only`` run.  The first matching summary line
       wins; pytest emits exactly one.
    2. **``-q`` per-file lines** (``tests/test_x.py: N``): pytest 9.x's ``-q``
       mode produces a compact ``file: count`` format with *no* summary line,
       so the counts are summed across every matching line.  This is the
       format the issue (#1621) asks CI to capture.
    3. **Node-ID lines** (lines containing ``::``): the non-``-q`` per-test
       listing, counted directly.  Used only when neither of the above is
       present.

    Returns 0 when no recognised line is found (a collection failure or a
    wrongly-captured file), which the comparison surfaces as an
    ``empty_collect`` finding rather than a silent pass.
    """
    file_count_total = 0
    file_count_lines = 0
    node_id_lines = 0
    for raw_line in collect_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _COLLECTED_SUMMARY_RE.match(line)
        if m:
            return int(m.group(1))
        m = _FILE_COUNT_RE.match(line)
        if m:
            file_count_total += int(m.group(1))
            file_count_lines += 1
            continue
        if "::" in line:
            node_id_lines += 1
    if file_count_lines > 0:
        return file_count_total
    return node_id_lines


# ---------------------------------------------------------------------------
# Comparison -- the recorded-vs-collected assertion
# ---------------------------------------------------------------------------


def compare_recorded_vs_collected(
    junit_xml: str,
    collect_output: str,
) -> JunitRecordedResult:
    """Compare the junit ``<testcase>`` count to the collect-only count.

    The gate **fails** (returns a result with ``ok=False``) when:

    * the recorded ``<testcase>`` count differs from the collected count --
      the load-truncation failure mode #1621 documents (exit 0, zero
      failures, a plausible-looking junit that silently omits a tail); or
    * the junit document is internally inconsistent (``<testcase>`` count !=
      ``<testsuite tests=...>`` sum) -- surfaced as a separate finding so a
      junit-writer bug is not masked by an accidental count equality; or
    * either side is emptily zero -- a wiring error (wrong file, failed
      collection) rather than a real pass.

    Gate inputs are diff-derived, never hand-typed (graft E, rule #9): both
    counts come directly from pytest's own output; no hardcoded test count or
    file list exists anywhere in this function.
    """
    junit = count_junit_testcases(junit_xml)
    collected = count_collected_tests(collect_output)

    findings: list[JunitRecordedFinding] = []

    if junit.testcase_count == 0 and junit.suite_tests_attr == 0:
        findings.append(
            JunitRecordedFinding(
                kind="empty_junit",
                detail=(
                    "junit document parsed but contained 0 <testcase> "
                    "elements and a 0 tests attribute -- likely a wrong "
                    "file path or an empty run, not truncation"
                ),
            )
        )
    if collected == 0:
        findings.append(
            JunitRecordedFinding(
                kind="empty_collect",
                detail=(
                    "collect-only output yielded 0 tests -- collection "
                    "failed or the wrong output was captured"
                ),
            )
        )
    if not junit.internal_consistency_ok:
        findings.append(
            JunitRecordedFinding(
                kind="junit_internal",
                detail=(
                    f"junit <testcase> count ({junit.testcase_count}) != "
                    f"<testsuite tests=...> attribute sum "
                    f"({junit.suite_tests_attr}) -- the junit writer itself "
                    f"lost records, independent of the collected comparison"
                ),
            )
        )
    if junit.testcase_count != collected:
        findings.append(
            JunitRecordedFinding(
                kind="recorded_vs_collected",
                detail=(
                    f"recorded <testcase> count ({junit.testcase_count}) != "
                    f"collected count ({collected}) -- "
                    f"{abs(junit.testcase_count - collected)} test(s) "
                    f"{'missing from junit' if junit.testcase_count < collected else 'extra in junit'}; "
                    f"under load this is the truncated-tail failure mode "
                    f"of issue #1621 (exit 0, zero failures, silent omission)"
                ),
            )
        )

    return JunitRecordedResult(
        recorded=junit.testcase_count,
        collected=collected,
        suite_tests_attr=junit.suite_tests_attr,
        findings=tuple(findings),
    )


# ---------------------------------------------------------------------------
# Report rendering (for CI step summary and stdout)
# ---------------------------------------------------------------------------


def render_gate_report(result: JunitRecordedResult) -> str:
    """Render the gate's findings as a human-readable report.

    Printed to stdout and the CI step summary.  When the gate passes, a brief
    one-line summary is produced instead of a findings list.
    """
    if result.ok:
        return (
            f"junit-recorded-check: PASSED ({result.recorded} <testcase> "
            f"elements == {result.collected} collected; suite tests attr "
            f"sum={result.suite_tests_attr})"
        )

    lines: list[str] = [
        "## Recorded-vs-collected gate (issue #1621)",
        "",
        f"Recorded (junit `<testcase>` count): {result.recorded}",
        f"Collected (pytest --collect-only): {result.collected}",
        f"Suite `tests=` attribute sum: {result.suite_tests_attr}",
        "",
        f"**{len(result.findings)} finding(s):**",
        "",
    ]
    for finding in result.findings:
        lines.append(f"- **{finding.kind}**: {finding.detail}")
    lines.append("")
    lines.append(
        "_This gate is enforcement (a step of the required 'Tests' job). "
        "A junit file that records fewer tests than were collected fails "
        "the merge even when pytest exits 0 with zero failures -- the "
        "truncated-tail failure mode of issue #1621._"
    )
    return "\n".join(lines)
