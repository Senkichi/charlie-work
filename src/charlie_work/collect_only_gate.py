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

from collections import Counter
from dataclasses import dataclass, field


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
      different multiplicities (e.g. a parametrize case was dropped or added).
    * ``"missing_sibling"`` -- a leaf name removed from a source module under
      ``tests/`` that did not reappear in any sibling module under ``tests/``
      at head (graft K's second clause: defeats class-wrapping and in-place
      deletion).
    """

    kind: str
    leaf_name: str
    source_module: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class CollectOnlyResult:
    """The full output of the collect-only gate for one base-vs-head comparison.

    ``base_leaf_counts`` / ``head_leaf_counts`` are the multisets (``Counter``)
    of leaf names at base and head.  ``base_module_leaves`` /
    ``head_module_leaves`` map each module path to its leaf-name multiset, used
    by the sibling-reappearance check.  ``findings`` is the tuple of
    :class:`CollectOnlyFinding` discrepancies; an empty tuple means the gate
    passed.
    """

    base_leaf_counts: Counter[str]
    head_leaf_counts: Counter[str]
    base_module_leaves: dict[str, Counter[str]] = field(default_factory=dict)
    head_module_leaves: dict[str, Counter[str]] = field(default_factory=dict)
    findings: tuple[CollectOnlyFinding, ...] = ()

    @property
    def ok(self) -> bool:
        """``True`` when the gate passed (no findings)."""
        return len(self.findings) == 0


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
       class name(s) + parametrize id, module-path stripped) must be identical
       at base and head.  A verbatim relocation (same leaf name, different
       module path) passes this clause; a rename, addition, or deletion does
       not.

    2. **Sibling reappearance**: every leaf name removed from a source module
       under ``tests/`` must reappear in a sibling module under ``tests/`` at
       head.  This defeats class-wrapping (renaming a function inside a class
       in the same module, which changes the leaf name) and in-place deletion
       (a leaf that just vanishes from a module without reappearing anywhere).

    The gate **fails** (returns a result with ``ok=False``) if either clause
    is violated.  Unlike the AST-equivalence gate (#1541, evidence only), this
    gate is **enforcement** -- the CLI command exits non-zero on failure, and
    the job is a required check.

    Gate inputs are diff-derived, never hand-typed (graft E, rule #9): the two
    collected sets come from parsing ``pytest --collect-only -q`` output.  No
    hardcoded list of moved test names exists anywhere in this function.
    """
    base_leaf_counts, base_module_leaves = collect_leaf_names(base_output)
    head_leaf_counts, head_module_leaves = collect_leaf_names(head_output)

    findings: list[CollectOnlyFinding] = []

    # --- Clause 1: multiset equality (leaf names, module-path stripped) ---
    added = head_leaf_counts - base_leaf_counts
    removed = base_leaf_counts - head_leaf_counts
    for leaf in sorted(added):
        findings.append(
            CollectOnlyFinding(
                kind="added",
                leaf_name=leaf,
                detail=(
                    f"leaf name present at head but not at base "
                    f"(head count={head_leaf_counts[leaf]}, base count=0)"
                ),
            )
        )
    for leaf in sorted(removed):
        findings.append(
            CollectOnlyFinding(
                kind="removed",
                leaf_name=leaf,
                detail=(
                    f"leaf name present at base but not at head "
                    f"(base count={base_leaf_counts[leaf]}, head count=0)"
                ),
            )
        )
    # Count mismatches: leaf present at both but with different multiplicity.
    common_keys = set(base_leaf_counts) & set(head_leaf_counts)
    for leaf in sorted(common_keys):
        bc = base_leaf_counts[leaf]
        hc = head_leaf_counts[leaf]
        if bc != hc:
            findings.append(
                CollectOnlyFinding(
                    kind="count_mismatch",
                    leaf_name=leaf,
                    detail=f"base count={bc}, head count={hc}",
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


def render_gate_report(result: CollectOnlyResult) -> str:
    """Render the gate's findings as a human-readable report.

    Unlike the AST-equivalence gate's review packet (evidence, not enforcement),
    this report is the gate's **failure output** -- it is printed to stdout and
    the CI step summary when the gate fails, and the command exits non-zero.
    When the gate passes, a brief one-line summary is produced instead.
    """
    base_total = sum(result.base_leaf_counts.values())
    head_total = sum(result.head_leaf_counts.values())

    if result.ok:
        return (
            f"collect-only gate: PASSED ({base_total} leaf names at base, "
            f"{head_total} at head; multisets match, all removed leaves "
            f"reappeared in siblings under tests/)"
        )

    lines: list[str] = [
        "## Collect-only gate (issue #1538)",
        "",
        f"Base: {base_total} leaf names",
        f"Head: {head_total} leaf names",
        "",
        f"**{len(result.findings)} finding(s):**",
        "",
    ]

    for finding in result.findings:
        if finding.kind == "missing_sibling":
            lines.append(
                f"- **{finding.kind}**: `{finding.leaf_name}` removed from "
                f"`{finding.source_module}` -- {finding.detail}"
            )
        else:
            lines.append(f"- **{finding.kind}**: `{finding.leaf_name}` -- {finding.detail}")

    lines.append("")
    lines.append(
        "_This gate is enforcement (a required check). A verbatim test "
        "relocation (same leaf name, different module path) passes; a "
        "rename, addition, deletion, or class-wrapping dodge does not._"
    )
    return "\n".join(lines)
