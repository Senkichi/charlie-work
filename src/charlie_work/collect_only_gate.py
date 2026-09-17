"""Collect-only gate: leaf-name multiset equality for verbatim test relocations (issue #1538).

Every candidate design for the god-object paydown Track-1 precondition specified
``pytest --collect-only -q`` full node-ID set-equality (base vs head).  A verbatim
test relocation necessarily changes each moved node's module-path component
(``tests/test_foo.py::test_x`` -> ``tests/test_foo_split.py::test_x``), so full
node-ID equality fails on exactly the splits this gate exists to approve.  This
defect was shared by all three competing designs and was caught by the mechanics
judge (graft K), not proposed by any one of them.

The fix (graft K): compare the **multiset of leaf test-name components** --
function name, plus class name where present, plus parametrize id -- after
stripping the module-path prefix from each collected node ID, base vs head.
Additionally assert that every leaf name removed from the source module
reappears in a sibling module somewhere under the same ``tests/`` tree.  This
second clause is what defeats class-wrapping (renaming a function inside a class
to dodge the multiset check) and in-place deletion (a leaf that just vanishes).

Gate inputs are diff-derived, never hand-typed (graft E, global rule #9): the
two collected sets come directly from running ``pytest --collect-only``
against base and head; nothing about which tests moved is enumerated by hand
anywhere in the gate's code.  The issue (#1538) says ``--collect-only -q``
because it was written against an older pytest where ``-q`` still produced
one-node-ID-per-line; this repo's pytest (9.x in uv.lock) changed ``-q`` to a
compact ``file: count`` format that does not include individual node IDs, so
the CI workflow uses ``--collect-only`` (without ``-q``) to get the
one-node-ID-per-line format this parser expects.

The gate must be positive-controlled before it is trusted (graft I): during the
pilot (#1542), a deliberately wrong split (one leaf dropped, not relocated) must
make the collect-only diff non-empty.  If it does not, the gate is broken and
the pilot stops.  This control is exercised in #1542, not built here, but the
gate's interface must support running it (i.e. it must fail loudly on a
genuinely missing leaf, not just on a renamed one).

This module is the **pure scanning logic** -- no I/O, no git, no subprocess.
The CLI command layer (:mod:`charlie_work.collect_only_gate_command`) owns the
file reads and the exit-code decision, following the same split as
:mod:`charlie_work.mojibake_gate` / :mod:`charlie_work.private_slug_gate`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data model (frozen, per CLAUDE.md invariant)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectOnlyFinding:
    """A single discrepancy found by the collect-only gate.

    ``kind`` is one of:

    * ``"added"`` -- a leaf name present at head but not at base (net-new test).
    * ``"removed"`` -- a leaf name present at base but not at head (deleted test).
    * ``"count_mismatch"`` -- a leaf name present at both base and head but with
      different multiplicities (e.g. a new module reuses a common leaf name
      such as ``test_init``).
    * ``"missing_sibling"`` -- a leaf name removed from a source module under
      ``tests/`` that did not reappear in any sibling module under ``tests/``
      at head (graft K's second clause: defeats class-wrapping and in-place
      deletion).

    ``base_count`` / ``head_count`` carry the leaf's multiplicity on each side
    for the clause-1 kinds (``added`` / ``removed`` / ``count_mismatch``), so a
    ``count_mismatch``'s direction is derived from data, never inferred from
    text.  They are ``None`` for ``missing_sibling``, whose relevant counts
    (removed-from-module and sibling reappearances) live in ``detail``.
    """

    kind: str
    leaf_name: str
    source_module: str | None = None
    detail: str = ""
    base_count: int | None = None
    head_count: int | None = None

    @property
    def fails_gate(self) -> bool:
        """Whether this finding fails the gate (Scope table, issue #1538).

        The verdict is evaluated per finding, never from net totals
        (amendment 2026-09-17): ``removed``, ``missing_sibling``, and
        ``count_mismatch`` where head < base fail; ``added`` and
        ``count_mismatch`` where head > base are reported but pass.  A
        ``count_mismatch`` missing its counts fails closed (direction
        unknown), as does any unrecognised kind.
        """
        if self.kind == "added":
            return False
        if self.kind == "count_mismatch":
            return (
                self.base_count is None
                or self.head_count is None
                or self.head_count < self.base_count
            )
        return True  # "removed", "missing_sibling", and any unknown kind


@dataclass(frozen=True)
class CollectOnlyResult:
    """The full output of the collect-only gate for one base-vs-head comparison.

    ``base_leaf_counts`` / ``head_leaf_counts`` are the multisets (``Counter``)
    of leaf names at base and head.  ``base_module_leaves`` /
    ``head_module_leaves`` map each module path to its leaf-name multiset, used
    by the sibling-reappearance check.  ``findings`` is the tuple of
    :class:`CollectOnlyFinding` discrepancies -- both enforced failures and
    reported-only rows (the Scope table distinguishes them per finding kind).
    """

    base_leaf_counts: Counter[str]
    head_leaf_counts: Counter[str]
    base_module_leaves: dict[str, Counter[str]] = field(default_factory=dict)
    head_module_leaves: dict[str, Counter[str]] = field(default_factory=dict)
    findings: tuple[CollectOnlyFinding, ...] = ()

    @property
    def failures(self) -> tuple[CollectOnlyFinding, ...]:
        """The findings that fail the gate (Scope table, per finding kind)."""
        return tuple(f for f in self.findings if f.fails_gate)

    @property
    def ok(self) -> bool:
        """``True`` when the gate passed (no enforced findings).

        Reported-only findings (``added``, ``count_mismatch`` with head >
        base) do not fail the gate -- the verdict is the disjunction of
        per-finding verdicts, never a net-totals comparison (a head that
        drops one leaf and adds another must still fail).
        """
        return len(self.failures) == 0


# ---------------------------------------------------------------------------
# Operator exemption (issue #1686)
# ---------------------------------------------------------------------------
#
# The gate's verdict is fail-closed by design: a deleted test, a rename, or a
# shrunken multiplicity fails the required check on every applicable PR.
# Some of those failures are legitimate -- the PR deleted a feature and its
# tests, renamed a misnamed test, changed parametrization ids, or
# consolidated duplicates -- so the gate needs a controlled escape hatch.
#
# The escape hatch is an OPERATOR-APPLIED PR LABEL (the configured
# ``labels.collect_gate_exempt`` name), resolved against the PR's live
# labels at gate run time -- never the ``github.event.pull_request.labels``
# snapshot, a PR-body line, a commit trailer, or any file in the tree, all
# of which are worker-authored content and would let a PR self-grant the
# exemption. Workers have no GitHub token, so only an operator can apply
# the label; the two-step operator flow is "apply the label, re-run the
# failed job".
#
# An active exemption waives only the gate's enforced verdict: findings are
# computed identically, every waived finding is still printed (kind + leaf),
# and the exit status flips to success. Nothing about how findings are
# computed or reported changes -- the label never suppresses output.


@dataclass(frozen=True)
class CollectGateExemption:
    """The resolved operator-exemption verdict for one gate run (issue #1686).

    ``label`` is the configured exemption label name (from
    ``LabelConfig.collect_gate_exempt``, never re-declared). ``active`` is
    True only when the live labels query confirmed the label is present on
    the PR -- a failed, empty, or malformed query resolves to ``active=False``
    (fail closed). ``detail`` is the human-readable reason for the
    resolution, printed in the gate output so the outcome is never silent.
    """

    label: str
    active: bool
    detail: str


def resolve_collect_gate_exemption(
    *,
    exemption_label: str,
    pr_number: int | None,
    pr_labels: set[str] | None,
    query_error: str | None = None,
) -> CollectGateExemption | None:
    """Resolve the operator-exemption verdict for a gate run.

    Returns ``None`` when no resolution was attempted at all (``pr_number``
    is ``None`` -- the caller supplied no ``--pr``): the gate then behaves
    exactly as it did before #1686, with no exemption mention in the output.

    When a PR number was supplied the verdict is always resolved, but the
    resolution can only *grant* on positive evidence: ``pr_labels`` is the
    set of label names on the PR as returned by a live API query, and
    ``None`` means the query itself failed -- which resolves to
    ``active=False`` (fail closed) with the reason carried in ``detail``.
    Never raises.
    """
    if pr_number is None:
        return None
    if pr_labels is None:
        return CollectGateExemption(
            label=exemption_label,
            active=False,
            detail=(
                f"labels query failed for PR #{pr_number}"
                + (f": {query_error}" if query_error else "")
                + " -- exemption not granted (fail closed)"
            ),
        )
    if exemption_label in pr_labels:
        return CollectGateExemption(
            label=exemption_label,
            active=True,
            detail=f"label `{exemption_label}` present on PR #{pr_number}",
        )
    return CollectGateExemption(
        label=exemption_label,
        active=False,
        detail=f"label `{exemption_label}` not present on PR #{pr_number}",
    )


# Single-line machine-readable record the gate command appends to its stdout
# message whenever an exemption was evaluated. The Actions job log preserves
# it verbatim (prefixed with the runner timestamp), and the review packet
# reads it back through ``repos/{owner}/{repo}/actions/jobs/{id}/logs`` --
# the only read-only, head-pinned channel that carries what the gate waived
# on a given head (Actions check runs expose no ``output`` fields, and the
# check-run annotations surface is capped and file-diagnostic shaped).
EXEMPTION_LOG_MARKER = "COLLECT-GATE-EXEMPTION v1 "

# The check-run name the ``collect-only-gate`` job reports under (the job's
# ``name:`` in .github/workflows/ci.yml). The review packet locates the
# gate's Actions job through this name on the PR's head-pinned check list,
# then reads the job's log for EXEMPTION_LOG_MARKER. It is a wire contract
# shared between ci.yml and the packet builder --
# ``test_collect_only_gate.py`` asserts both ends of it stay equal.
COLLECT_ONLY_GATE_CHECK_NAME = "Collect-only gate"


def exemption_log_marker(
    exemption: CollectGateExemption,
    waived: Sequence[CollectOnlyFinding],
    head_sha: str | None = None,
) -> str:
    """Render the single-line log marker for one evaluated exemption.

    ``waived`` is the set of findings the exemption waived (the gate's
    enforced findings when ``exemption.active`` -- ``()`` otherwise). The
    payload stays compact (kind + leaf + source module per finding) so the
    line survives intact in the job log.

    ``head_sha`` binds the record to the head the gate ran against. The
    review packet verifies it against the PR's live ``headRefOid`` before
    trusting the record -- a label applied for head A legitimately remains
    applied on head B (the issue forbids stripping it on ``synchronize``),
    so evidence must provably describe THIS head or be ignored.
    """
    payload = {
        "v": 1,
        "label": exemption.label,
        "active": exemption.active,
        "detail": exemption.detail,
        "head_sha": head_sha,
        "waived": [
            {
                "kind": f.kind,
                "leaf_name": f.leaf_name,
                "source_module": f.source_module,
            }
            for f in waived
        ],
    }
    return EXEMPTION_LOG_MARKER + json.dumps(payload, separators=(",", ":"), sort_keys=True)


def parse_exemption_log_marker(log_text: str) -> dict[str, Any] | None:
    """Extract the exemption payload from Actions job-log text.

    Scans for the LAST line carrying :data:`EXEMPTION_LOG_MARKER` (a job log
    can legitimately contain earlier output before the marker) and returns
    its parsed JSON payload when it is a ``v: 1`` object. Returns ``None``
    on no marker or any malformed payload -- callers treat ``None`` as
    "waived findings unavailable", never as an exemption verdict.
    """
    payload: Any = None
    for line in log_text.splitlines():
        idx = line.find(EXEMPTION_LOG_MARKER)
        if idx == -1:
            continue
        try:
            candidate = json.loads(line[idx + len(EXEMPTION_LOG_MARKER) :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
    if not isinstance(payload, dict) or payload.get("v") != 1:
        return None
    return payload


# ---------------------------------------------------------------------------
# Parsing -- pure functions over collect-only output text
# ---------------------------------------------------------------------------


def parse_collect_only_output(output: str) -> list[str]:
    """Parse ``pytest --collect-only -q`` output into a list of node IDs.

    Each line of *output* is examined.  A line is a node ID when it contains
    ``::`` (pytest separates the module path from test items with ``::``).
    Summary lines (``N tests collected``, ``no tests collected in ...``) and
    blank lines do not contain ``::`` and are skipped.

    Windows line endings (``\\r\\n``) and trailing whitespace are stripped.
    Backslashes in module paths are normalized to forward slashes so the same
    test produces the same module path on Windows and POSIX.

    Never raises -- a malformed line simply yields no node ID.
    """
    node_ids: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.rstrip("\r\n").strip()
        if not line:
            continue
        if "::" not in line:
            continue  # summary line, warning, or file-level entry without tests
        # Normalize backslashes to forward slashes in the module-path component
        # so the same test produces the same module path cross-platform.
        # Only the part before the first ``::`` is a filesystem path; the rest
        # is pytest's test-item naming which uses ``::`` and ``[]`` and never
        # contains a filesystem separator.
        idx = line.index("::")
        module_part = line[:idx].replace("\\", "/")
        leaf_part = line[idx:]
        node_ids.append(module_part + leaf_part)
    return node_ids


def extract_leaf_name(node_id: str) -> tuple[str, str] | None:
    """Split a node ID into ``(module_path, leaf_name)``.

    The module path is everything before the first ``::``.  The leaf name is
    everything after the first ``::`` -- function name, plus class name(s) where
    present, plus parametrize id.  This is the multiset element the gate
    compares (graft K): stripping the module-path prefix means a verbatim
    relocation (same leaf name, different module path) produces the same
    multiset element.

    Returns ``None`` if the line does not contain ``::`` (not a valid node ID).
    """
    idx = node_id.find("::")
    if idx == -1:
        return None
    module_path = node_id[:idx]
    leaf_name = node_id[idx + 2 :]
    return module_path, leaf_name


def collect_leaf_names(
    output: str,
) -> tuple[Counter[str], dict[str, Counter[str]]]:
    """Parse collect-only output and return ``(leaf_counts, module_leaves)``.

    ``leaf_counts`` is the multiset (``Counter``) of all leaf names across
    every module.  ``module_leaves`` maps each module path to the multiset of
    leaf names within that module.  Both are used by
    :func:`compare_collect_only`: ``leaf_counts`` for the multiset-equality
    check (clause 1), ``module_leaves`` for the sibling-reappearance check
    (clause 2).
    """
    node_ids = parse_collect_only_output(output)
    leaf_counts: Counter[str] = Counter()
    module_leaves: dict[str, Counter[str]] = {}
    for nid in node_ids:
        parts = extract_leaf_name(nid)
        if parts is None:
            continue
        module_path, leaf_name = parts
        leaf_counts[leaf_name] += 1
        module_leaves.setdefault(module_path, Counter())[leaf_name] += 1
    return leaf_counts, module_leaves


# ---------------------------------------------------------------------------
# Comparison -- the two clauses (graft K)
# ---------------------------------------------------------------------------

# The ``tests/`` prefix that sibling modules must live under.  A module path
# starting with this string is a test module under the tests tree.  The check
# is a simple prefix test (not a glob): pytest's ``--collect-only -q`` emits
# paths relative to the invocation root, and this repo's tests live under
# ``tests/``.  A path like ``tests/sub/test_foo.py`` is correctly included.
_TESTS_PREFIX = "tests/"


def _is_under_tests(module_path: str) -> bool:
    """Return ``True`` if *module_path* is under the ``tests/`` tree."""
    return module_path.startswith(_TESTS_PREFIX)


def compare_collect_only(base_output: str, head_output: str) -> CollectOnlyResult:
    """Compare leaf-name multisets from base and head collect-only output.

    Two clauses (graft K):

    1. **Multiset equality**: the multiset of all leaf names (function name +
       class name(s) + parametrize id, module-path stripped) is compared at
       base and head.  A verbatim relocation (same leaf name, different
       module path) produces no finding; every leaf whose multiplicity
       differs produces exactly one finding -- ``added`` (absent at base),
       ``removed`` (absent at head), or ``count_mismatch`` (present at both
       with different counts, direction carried in ``base_count`` /
       ``head_count``).

    2. **Sibling reappearance**: every leaf name removed from a source module
       under ``tests/`` must reappear in a sibling module under ``tests/`` at
       head.  This defeats class-wrapping (renaming a function inside a class
       in the same module, which changes the leaf name) and in-place deletion
       (a leaf that just vanishes from a module without reappearing anywhere).

    The verdict follows the issue's Scope table (amendment 2026-09-17),
    evaluated per finding via :attr:`CollectOnlyFinding.fails_gate`, never
    from net totals: ``removed``, ``missing_sibling``, and ``count_mismatch``
    where head < base fail; ``added`` and ``count_mismatch`` where head >
    base are reported but pass.  This is what keeps the gate satisfiable
    alongside the test-adequacy gate (ordinary PRs are REQUIRED to add
    tests): a pure-addition PR reports ``added`` findings and passes, while a
    head that drops one leaf and adds a differently named one still fails on
    the ``removed`` finding.  Unlike the AST-equivalence gate (#1541,
    evidence only), this gate is **enforcement** -- the CLI command exits
    non-zero on failure, and the job is a required check.

    Gate inputs are diff-derived, never hand-typed (graft E, rule #9): the two
    collected sets come from parsing ``pytest --collect-only -q`` output.  No
    hardcoded list of moved test names exists anywhere in this function.
    """
    base_leaf_counts, base_module_leaves = collect_leaf_names(base_output)
    head_leaf_counts, head_module_leaves = collect_leaf_names(head_output)

    findings: list[CollectOnlyFinding] = []

    # --- Clause 1: multiset equality (leaf names, module-path stripped) ---
    # One finding per leaf whose multiplicity differs, classified so the kind
    # matches the issue's Scope table exactly: absent at base -> ``added``;
    # absent at head -> ``removed``; present at both with different counts ->
    # ``count_mismatch``.  Both counts ride on the finding as data so a
    # count_mismatch's direction (head < base fails, head > base is reported
    # only) is derived from fields, never inferred from text.
    for leaf in sorted(set(base_leaf_counts) | set(head_leaf_counts)):
        bc = base_leaf_counts.get(leaf, 0)
        hc = head_leaf_counts.get(leaf, 0)
        if bc == hc:
            continue
        if bc == 0:
            kind = "added"
            detail = f"leaf name present at head but not at base (base count=0, head count={hc})"
        elif hc == 0:
            kind = "removed"
            detail = f"leaf name present at base but not at head (base count={bc}, head count=0)"
        else:
            kind = "count_mismatch"
            detail = f"base count={bc}, head count={hc}"
        findings.append(
            CollectOnlyFinding(
                kind=kind,
                leaf_name=leaf,
                base_count=bc,
                head_count=hc,
                detail=detail,
            )
        )

    # --- Clause 2: sibling reappearance (graft K's second clause) ---
    # For each module under tests/ at base, for each leaf removed from that
    # module (base count > head count), assert the leaf reappears in some
    # OTHER module under tests/ at head.
    for module_path in sorted(base_module_leaves):
        if not _is_under_tests(module_path):
            continue
        base_mod = base_module_leaves[module_path]
        head_mod = head_module_leaves.get(module_path, Counter())
        removed_from_module = base_mod - head_mod  # Counter diff: positive only
        for leaf in sorted(removed_from_module):
            removed_count = removed_from_module[leaf]
            # Count this leaf's appearances in sibling modules under tests/
            # at head (i.e. modules != module_path, under tests/).
            sibling_count = 0
            for other_path, other_leaves in head_module_leaves.items():
                if other_path == module_path:
                    continue
                if not _is_under_tests(other_path):
                    continue
                sibling_count += other_leaves.get(leaf, 0)
            if sibling_count < removed_count:
                findings.append(
                    CollectOnlyFinding(
                        kind="missing_sibling",
                        leaf_name=leaf,
                        source_module=module_path,
                        detail=(
                            f"leaf removed from {module_path} "
                            f"(removed count={removed_count}) but only "
                            f"{sibling_count} reappearance(s) in sibling "
                            f"modules under tests/"
                        ),
                    )
                )

    return CollectOnlyResult(
        base_leaf_counts=base_leaf_counts,
        head_leaf_counts=head_leaf_counts,
        base_module_leaves=base_module_leaves,
        head_module_leaves=head_module_leaves,
        findings=tuple(findings),
    )


# ---------------------------------------------------------------------------
# Report rendering (for CI step summary and stdout)
# ---------------------------------------------------------------------------


def _render_finding_line(finding: CollectOnlyFinding) -> str:
    """Render one finding as a markdown bullet."""
    if finding.kind == "missing_sibling":
        return (
            f"- **{finding.kind}**: `{finding.leaf_name}` removed from "
            f"`{finding.source_module}` -- {finding.detail}"
        )
    return f"- **{finding.kind}**: `{finding.leaf_name}` -- {finding.detail}"


def render_gate_report(
    result: CollectOnlyResult,
    exemption: CollectGateExemption | None = None,
) -> str:
    """Render the gate's findings as a human-readable report.

    Unlike the AST-equivalence gate's review packet (evidence, not enforcement),
    this report is the gate's **failure output** -- it is printed to stdout and
    the CI step summary when the gate fails, and the command exits non-zero.
    When the gate passes, a brief summary is produced instead -- including any
    reported-only findings (``added``, ``count_mismatch`` with head > base),
    which are listed as passing rows, not failures.

    ``exemption`` (issue #1686) is the resolved operator-exemption verdict,
    or ``None`` when none was evaluated -- in which case the report is
    byte-identical to the pre-#1686 output. When evaluated, the report
    states the resolution explicitly: an ACTIVE exemption marks the failing
    findings as waived (they are still listed, kind + leaf -- the label
    suppresses the verdict, never the output); an inactive one states why
    the exemption was not applied; an active one with nothing to waive says
    so, making a stale label visible.
    """
    base_total = sum(result.base_leaf_counts.values())
    head_total = sum(result.head_leaf_counts.values())

    def _exemption_note() -> str:
        assert exemption is not None
        return f"Operator exemption `{exemption.label}`: {exemption.detail}."

    if result.ok or (exemption is not None and exemption.active):
        waived = exemption is not None and exemption.active and bool(result.failures)
        if not result.findings:
            line = (
                f"collect-only gate: PASSED ({base_total} leaf names at base, "
                f"{head_total} at head; multisets match, all removed leaves "
                f"reappeared in siblings under tests/)"
            )
            if exemption is not None:
                line += (
                    f"\nexemption: {_exemption_note()} "
                    "Nothing to waive -- no enforced findings on this run "
                    "(a label left on a clean PR is stale and can be removed)."
                )
            return line
        reported = [f for f in result.findings if not f.fails_gate]
        if waived:
            assert exemption is not None
            lines = [
                f"collect-only gate: PASSED ({base_total} leaf names at base, "
                f"{head_total} at head; {len(result.failures)} enforced "
                f"finding(s) WAIVED by operator exemption label "
                f"`{exemption.label}`)",
                "",
            ]
            lines.extend(_render_finding_line(f) + "  *(waived)*" for f in result.failures)
        else:
            lines = [
                f"collect-only gate: PASSED ({base_total} leaf names at base, "
                f"{head_total} at head; no enforced findings)",
            ]
        if exemption is not None:
            lines.append("")
            note = f"exemption: {_exemption_note()}"
            if exemption.active and not result.failures:
                note += (
                    " Nothing to waive -- no enforced findings on this run "
                    "(a label left on a clean PR is stale and can be removed)."
                )
            lines.append(note)
        if reported:
            lines.append("")
            lines.append(f"{len(reported)} reported finding(s) (pass, not enforced):")
            lines.extend(_render_finding_line(f) for f in reported)
        return "\n".join(lines)

    failures = result.failures
    reported = [f for f in result.findings if not f.fails_gate]

    lines = [
        "## Collect-only gate (issue #1538)",
        "",
        f"Base: {base_total} leaf names",
        f"Head: {head_total} leaf names",
        "",
        f"**{len(failures)} failing finding(s):**",
        "",
    ]
    lines.extend(_render_finding_line(f) for f in failures)

    if reported:
        lines.append("")
        lines.append(f"{len(reported)} reported finding(s) (pass, not enforced):")
        lines.append("")
        lines.extend(_render_finding_line(f) for f in reported)

    if exemption is not None:
        lines.append("")
        lines.append(f"exemption: {_exemption_note()}")

    lines.append("")
    lines.append(
        "_This gate is enforcement (a required check), scoped per finding "
        "(issue #1538): `removed`, `missing_sibling`, and `count_mismatch` "
        "with head < base fail; `added` and `count_mismatch` with head > "
        "base are reported but pass. A verbatim test relocation (same leaf "
        "name, different module path) passes; a rename, deletion, dropped "
        "multiplicity, or class-wrapping dodge fails._"
    )
    return "\n".join(lines)
