"""Tests for the collect-only gate (issue #1538).

Covers:

* :func:`parse_collect_only_output` -- parsing ``pytest --collect-only -q``
  output into node IDs (filtering summary lines, normalizing paths).
* :func:`extract_leaf_name` -- splitting a node ID into (module_path,
  leaf_name), stripping the module-path prefix.
* :func:`collect_leaf_names` -- aggregating leaf names into multisets per
  module and overall.
* :func:`compare_collect_only` -- the two clauses (graft K):
  - Clause 1: leaf-name multiset comparison (verbatim relocation produces no
    finding; rename / addition / deletion / count-mismatch produce one
    finding per differing leaf).
  - Clause 2: sibling reappearance (leaf removed from a module under tests/
    must reappear in a sibling under tests/).
* The Scope table (issue #1538, amendment 2026-09-17): one test per row --
  ``removed``, ``missing_sibling``, and ``count_mismatch`` with head < base
  fail; ``added`` and ``count_mismatch`` with head > base are reported but
  pass. Includes the pure-addition PR case and the scoping mutation control
  (a head that drops one leaf and adds a differently named one must fail --
  it goes red if the verdict is ever computed from net totals or ``removed``
  is made non-fatal).
* Mutation control: reverting the multiset comparison to full-node-ID equality
  makes the gate reject a verbatim relocation (proves the fix actually fixes
  graft K's defect).
* Rule #9 compliance: no hardcoded test-name list in the gate source.
* The CLI command ``charlie collect-only-check``.
* :func:`render_gate_report` -- report rendering for pass and fail cases.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path

import yaml

from charlie_work.collect_only_gate import (
    CollectOnlyFinding,
    CollectOnlyResult,
    collect_leaf_names,
    compare_collect_only,
    extract_leaf_name,
    parse_collect_only_output,
    render_gate_report,
)
from charlie_work.collect_only_gate_command import (
    run_collect_only_check_command,
)


# ---------------------------------------------------------------------------
# parse_collect_only_output
# ---------------------------------------------------------------------------


def test_parse_collect_only_output_basic() -> None:
    """Node IDs are extracted one per line; summary lines are skipped."""
    output = (
        "tests/test_foo.py::test_a\n"
        "tests/test_foo.py::TestClass::test_b\n"
        "tests/test_bar.py::test_c[1]\n"
        "3 tests collected\n"
    )
    ids = parse_collect_only_output(output)
    assert ids == [
        "tests/test_foo.py::test_a",
        "tests/test_foo.py::TestClass::test_b",
        "tests/test_bar.py::test_c[1]",
    ]


def test_parse_collect_only_output_skips_blank_and_summary() -> None:
    """Blank lines and summary lines (no ``::``) are skipped."""
    output = "\ntests/test_foo.py::test_a\n\nno tests collected in 0.01s\n"
    ids = parse_collect_only_output(output)
    assert ids == ["tests/test_foo.py::test_a"]


def test_parse_collect_only_output_strips_carriage_returns() -> None:
    """Windows line endings (\\r\\n) are stripped."""
    output = "tests/test_foo.py::test_a\r\n"
    ids = parse_collect_only_output(output)
    assert ids == ["tests/test_foo.py::test_a"]


def test_parse_collect_only_output_normalizes_backslashes() -> None:
    """Backslashes in module paths are normalized to forward slashes."""
    output = "tests\\test_foo.py::test_a\n"
    ids = parse_collect_only_output(output)
    assert ids == ["tests/test_foo.py::test_a"]


def test_parse_collect_only_output_empty() -> None:
    """Empty output yields no node IDs."""
    assert parse_collect_only_output("") == []
    assert parse_collect_only_output("no tests collected\n") == []


# ---------------------------------------------------------------------------
# extract_leaf_name
# ---------------------------------------------------------------------------


def test_extract_leaf_name_function() -> None:
    """A bare function: module path + function name."""
    assert extract_leaf_name("tests/test_foo.py::test_a") == (
        "tests/test_foo.py",
        "test_a",
    )


def test_extract_leaf_name_class_method() -> None:
    """A class method: module path + Class::method."""
    assert extract_leaf_name("tests/test_foo.py::TestClass::test_b") == (
        "tests/test_foo.py",
        "TestClass::test_b",
    )


def test_extract_leaf_name_parametrized() -> None:
    """A parametrized test: module path + function[id]."""
    assert extract_leaf_name("tests/test_foo.py::test_c[1]") == (
        "tests/test_foo.py",
        "test_c[1]",
    )


def test_extract_leaf_name_nested_class() -> None:
    """A nested class method: module path + Outer::Inner::method."""
    assert extract_leaf_name("tests/test_foo.py::Outer::Inner::test_d") == (
        "tests/test_foo.py",
        "Outer::Inner::test_d",
    )


def test_extract_leaf_name_no_double_colon() -> None:
    """A line without ``::`` is not a valid node ID."""
    assert extract_leaf_name("no tests collected") is None


# ---------------------------------------------------------------------------
# collect_leaf_names
# ---------------------------------------------------------------------------


def test_collect_leaf_names_basic() -> None:
    """Leaf names are counted into an overall multiset and per-module multisets."""
    output = (
        "tests/test_foo.py::test_a\n"
        "tests/test_foo.py::TestClass::test_b\n"
        "tests/test_bar.py::test_a\n"
    )
    leaf_counts, module_leaves = collect_leaf_names(output)
    assert leaf_counts == Counter({"test_a": 2, "TestClass::test_b": 1})
    assert module_leaves["tests/test_foo.py"] == Counter({"test_a": 1, "TestClass::test_b": 1})
    assert module_leaves["tests/test_bar.py"] == Counter({"test_a": 1})


def test_collect_leaf_names_parametrize_counts() -> None:
    """Parametrized test cases are distinct leaf names in the multiset."""
    output = "tests/test_foo.py::test_c[1]\ntests/test_foo.py::test_c[2]\n"
    leaf_counts, _ = collect_leaf_names(output)
    assert leaf_counts == Counter({"test_c[1]": 1, "test_c[2]": 1})


# ---------------------------------------------------------------------------
# compare_collect_only -- clause 1: multiset equality (graft K)
# ---------------------------------------------------------------------------


def test_verbatim_relocation_passes() -> None:
    """A verbatim test relocation (same leaf name, different module path) passes.

    This is the core fix for graft K: full-node-ID equality would reject this
    (the module path changed), but leaf-name multiset equality accepts it.
    """
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo_split.py::test_a\ntests/test_foo_split.py::test_b\n"
    result = compare_collect_only(base, head)
    assert result.ok is True
    assert result.findings == ()


def test_no_change_passes() -> None:
    """Identical base and head output passes."""
    output = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    result = compare_collect_only(output, output)
    assert result.ok is True


def test_renamed_leaf_fails() -> None:
    """A renamed leaf (function name changed) fails the multiset check."""
    base = "tests/test_foo.py::test_a\n"
    head = "tests/test_foo.py::test_a_renamed\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    kinds = {f.kind for f in result.findings}
    assert "removed" in kinds
    assert "added" in kinds


def test_added_leaf_is_reported() -> None:
    """A net-new leaf (addition without removal) is an ``added`` finding.

    The finding is still emitted -- the Scope table reports it -- but it does
    not fail the gate (see the Scope-table section below).
    """
    base = "tests/test_foo.py::test_a\n"
    head = "tests/test_foo.py::test_a\ntests/test_foo.py::test_new\n"
    result = compare_collect_only(base, head)
    added = [f for f in result.findings if f.kind == "added"]
    assert len(added) == 1
    assert added[0].leaf_name == "test_new"
    assert added[0].base_count == 0
    assert added[0].head_count == 1


def test_removed_leaf_fails() -> None:
    """A deleted leaf (removal without re-addition) fails the multiset check."""
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    assert any(f.kind == "removed" and f.leaf_name == "test_b" for f in result.findings)


def test_dropped_parametrize_case_fails() -> None:
    """A dropped parametrize case is a ``removed`` finding and fails.

    The parametrize id is part of the leaf name, so ``test_c[2]`` vanishing
    is a removal, not a count_mismatch on ``test_c``.
    """
    base = "tests/test_foo.py::test_c[1]\ntests/test_foo.py::test_c[2]\n"
    head = "tests/test_foo.py::test_c[1]\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    assert any(f.kind == "removed" and f.leaf_name == "test_c[2]" for f in result.findings)


# ---------------------------------------------------------------------------
# compare_collect_only -- clause 2: sibling reappearance (graft K)
# ---------------------------------------------------------------------------


def test_sibling_reappearance_passes() -> None:
    """A leaf removed from a module that reappears in a sibling under tests/ passes."""
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\ntests/test_bar.py::test_b\n"
    result = compare_collect_only(base, head)
    assert result.ok is True


def test_missing_sibling_reappearance_fails() -> None:
    """A leaf removed from a module that does NOT reappear in a sibling fails.

    This is the in-place deletion case (graft K's second clause): ``test_b``
    vanishes from ``tests/test_foo.py`` and does not reappear in any sibling
    under ``tests/``. The multiset check (clause 1) catches this too, but the
    sibling-reappearance finding provides the specific diagnostic.
    """
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    missing = [f for f in result.findings if f.kind == "missing_sibling"]
    assert len(missing) == 1
    assert missing[0].leaf_name == "test_b"
    assert missing[0].source_module == "tests/test_foo.py"


def test_class_wrapping_dodge_fails() -> None:
    """Class-wrapping (renaming a function inside a class in the same module) fails.

    The leaf name changes from ``test_foo`` to ``TestClass::test_foo``, so the
    multiset check (clause 1) catches it. The sibling-reappearance check
    (clause 2) also catches it: ``test_foo`` was removed from the module and
    did not reappear in a sibling (the new ``TestClass::test_foo`` is a
    different leaf name).
    """
    base = "tests/test_a.py::test_foo\n"
    head = "tests/test_a.py::TestClass::test_foo\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    kinds = {f.kind for f in result.findings}
    # Clause 1 catches the rename (removed + added).
    assert "removed" in kinds
    assert "added" in kinds
    # Clause 2 catches the missing sibling reappearance.
    missing = [f for f in result.findings if f.kind == "missing_sibling"]
    assert len(missing) == 1
    assert missing[0].leaf_name == "test_foo"


def test_move_outside_tests_fails_sibling_check() -> None:
    """A leaf moved from tests/ to a non-tests/ path fails the sibling check.

    The multiset is unchanged (same leaf name), so clause 1 passes. But clause 2
    fails: the leaf was removed from a module under tests/ and did not reappear
    in a sibling UNDER tests/ (it went to src/).
    """
    base = "tests/test_foo.py::test_a\n"
    head = "src/test_foo.py::test_a\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    missing = [f for f in result.findings if f.kind == "missing_sibling"]
    assert len(missing) == 1
    assert missing[0].leaf_name == "test_a"
    assert missing[0].source_module == "tests/test_foo.py"


def test_multiple_leaves_one_missing_sibling() -> None:
    """When multiple leaves move but one doesn't reappear, only that one is flagged."""
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\ntests/test_foo.py::test_c\n"
    # test_a and test_b moved to siblings; test_c vanished.
    head = "tests/test_bar.py::test_a\ntests/test_baz.py::test_b\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    missing = [f for f in result.findings if f.kind == "missing_sibling"]
    assert len(missing) == 1
    assert missing[0].leaf_name == "test_c"


# ---------------------------------------------------------------------------
# Scope table (issue #1538, amendment 2026-09-17) -- one test per row.
#
# The verdict is evaluated PER FINDING, never from net totals:
#   - FAIL: ``removed``; ``missing_sibling``; ``count_mismatch`` head < base.
#   - PASS (still reported): ``added``; ``count_mismatch`` head > base.
# ---------------------------------------------------------------------------


def test_scope_removed_row_fails() -> None:
    """Scope row ``removed``: a leaf present at base and absent at head fails."""
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    removed = [f for f in result.findings if f.kind == "removed"]
    assert len(removed) == 1
    assert removed[0].leaf_name == "test_b"
    assert removed[0].fails_gate is True
    assert result.failures == tuple(f for f in result.findings if f.fails_gate)


def test_scope_missing_sibling_row_fails() -> None:
    """Scope row ``missing_sibling``: a leaf that leaves its module and
    reappears nowhere under tests/ fails."""
    base = "tests/test_foo.py::test_a\n"
    head = "src/test_foo.py::test_a\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    missing = [f for f in result.findings if f.kind == "missing_sibling"]
    assert len(missing) == 1
    assert missing[0].fails_gate is True


def test_scope_count_mismatch_shrunk_row_fails() -> None:
    """Scope row ``count_mismatch`` head < base: a dropped multiplicity fails.

    ``test_init`` exists in two modules at base and only one at head -- the
    leaf is present on both sides, so this is a count_mismatch (not a
    removal), and the shrunk direction makes it fail.
    """
    base = "tests/test_foo.py::test_init\ntests/test_bar.py::test_init\n"
    head = "tests/test_foo.py::test_init\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    mismatches = [f for f in result.findings if f.kind == "count_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].leaf_name == "test_init"
    # Direction is carried as data, not inferred from text.
    assert mismatches[0].base_count == 2
    assert mismatches[0].head_count == 1
    assert mismatches[0].fails_gate is True


def test_scope_added_row_passes() -> None:
    """Scope row ``added``: a leaf present at head but not at base is
    reported but does not fail."""
    base = "tests/test_foo.py::test_a\n"
    head = "tests/test_foo.py::test_a\ntests/test_foo.py::test_new\n"
    result = compare_collect_only(base, head)
    assert result.ok is True
    assert result.failures == ()
    added = [f for f in result.findings if f.kind == "added"]
    assert len(added) == 1
    assert added[0].leaf_name == "test_new"
    assert added[0].fails_gate is False


def test_scope_count_mismatch_grown_row_passes() -> None:
    """Scope row ``count_mismatch`` head > base: a grown multiplicity is
    reported but does not fail.

    The issue's example: a new module reuses a common leaf name such as
    ``test_init``, so its multiplicity grows from 1 to 2.
    """
    base = "tests/test_foo.py::test_init\n"
    head = "tests/test_foo.py::test_init\ntests/test_bar.py::test_init\n"
    result = compare_collect_only(base, head)
    assert result.ok is True
    assert result.failures == ()
    mismatches = [f for f in result.findings if f.kind == "count_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].leaf_name == "test_init"
    assert mismatches[0].base_count == 1
    assert mismatches[0].head_count == 2
    assert mismatches[0].fails_gate is False


def test_pure_addition_pr_passes() -> None:
    """A PR that only adds tests -- the normal feature/bugfix shape -- passes.

    This is what keeps the gate satisfiable alongside the test-adequacy
    gate (ordinary PRs are REQUIRED to add tests). This PR itself adds
    ``tests/test_collect_only_gate.py``, so it is the first real
    pure-addition case: the gate's own job must conclude success on it.
    """
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = (
        "tests/test_foo.py::test_a\n"
        "tests/test_foo.py::test_b\n"
        "tests/test_new_module.py::test_x\n"
        "tests/test_new_module.py::test_y[param]\n"
        "tests/test_new_module.py::TestNew::test_z\n"
    )
    result = compare_collect_only(base, head)
    assert result.ok is True
    assert result.failures == ()
    # The additions are still reported, just not enforced.
    kinds = {f.kind for f in result.findings}
    assert kinds == {"added"}
    assert len(result.findings) == 3


def test_relocation_plus_new_test_passes() -> None:
    """Net-new tests alongside a relocation are allowed.

    The moved leaf's multiset entry is unchanged and it reappears in a
    sibling under tests/; the net-new leaf is an ``added`` finding. The two
    are evaluated independently per leaf -- the addition does not mask a
    removal, and the removal check does not block the addition.
    """
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\ntests/test_bar.py::test_b\ntests/test_bar.py::test_new\n"
    result = compare_collect_only(base, head)
    assert result.ok is True
    added = [f for f in result.findings if f.kind == "added"]
    assert [f.leaf_name for f in added] == ["test_new"]


def test_scoping_mutation_control_drop_one_add_one_fails() -> None:
    """Scoping mutation control: a head that drops one leaf and adds a
    differently named one must FAIL.

    Mutated TOWARD the forbidden outcome (issue #1538, amendment
    2026-09-17): base and head have the SAME total leaf count (2 == 2), so
    any verdict computed from net totals passes this case -- which is
    exactly the dodge the gate exists for. This test goes red if the
    verdict is ever computed from net totals, or if ``removed`` is made
    non-fatal.
    """
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\ntests/test_foo.py::test_x\n"
    result = compare_collect_only(base, head)

    # Totals are equal -- a net-totals verdict would (wrongly) pass.
    assert sum(result.base_leaf_counts.values()) == sum(result.head_leaf_counts.values())

    assert result.ok is False
    kinds = {f.kind for f in result.findings}
    assert "removed" in kinds  # test_b dropped -- the failing half
    assert "added" in kinds  # test_x added -- reported, does not rescue the gate
    assert any(f.kind == "removed" and f.leaf_name == "test_b" for f in result.failures)


def test_fails_gate_unknown_kind_fails_closed() -> None:
    """An unrecognised finding kind fails closed (Scope table default).

    Only ``added`` and ``count_mismatch`` with head > base are scoped as
    passing; anything the table does not name as passing is enforced.
    """
    finding = CollectOnlyFinding(kind="surprise_new_kind", leaf_name="test_a")
    assert finding.fails_gate is True


def test_fails_gate_count_mismatch_missing_counts_fails_closed() -> None:
    """A ``count_mismatch`` without direction data fails closed.

    The direction must come from ``base_count``/``head_count`` fields; if
    they are absent the finding cannot be classified as the passing row, so
    it fails rather than silently passing.
    """
    finding = CollectOnlyFinding(kind="count_mismatch", leaf_name="test_a")
    assert finding.fails_gate is True


# ---------------------------------------------------------------------------
# Mutation control: reverting to full-node-ID equality rejects a verbatim
# relocation (proves the fix actually fixes graft K's defect)
# ---------------------------------------------------------------------------


def test_mutation_full_node_id_equality_rejects_verbatim_relocation() -> None:
    """Mutation control: full-node-ID set equality rejects a verbatim relocation.

    This test simulates the defect (graft K): if the comparison used full node-ID
    set equality instead of leaf-name multiset equality, a verbatim relocation
    (same leaf name, different module path) would be rejected. The test proves
    the fix is load-bearing by showing the naive comparison fails on exactly
    the case the gate exists to approve.

    The mutation: replace ``compare_collect_only`` with a full-node-ID set
    comparison. The verbatim relocation below must PASS under the leaf-name
    multiset comparison (the fix) and FAIL under the full-node-ID set
    comparison (the defect).
    """
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo_split.py::test_a\ntests/test_foo_split.py::test_b\n"

    # The fix (leaf-name multiset equality): PASSES.
    result = compare_collect_only(base, head)
    assert result.ok is True, (
        "Leaf-name multiset equality must pass a verbatim relocation. "
        "If this fails, the gate has regressed to the full-node-ID defect (graft K)."
    )

    # The defect (full-node-ID set equality): FAILS.
    base_ids = set(parse_collect_only_output(base))
    head_ids = set(parse_collect_only_output(head))
    full_node_id_equal = base_ids == head_ids
    assert full_node_id_equal is False, (
        "Full-node-ID set equality must reject a verbatim relocation "
        "(the module path changed). This is graft K's defect -- the test "
        "proves the leaf-name multiset fix is load-bearing."
    )


# ---------------------------------------------------------------------------
# Rule #9 compliance: no hardcoded test-name list in the gate source
# ---------------------------------------------------------------------------


def test_no_hardcoded_test_name_list_in_gate_source() -> None:
    """Rule #9: the gate module contains no hardcoded list of test names.

    The gate's inputs are diff-derived (graft E, rule #9): the two collected
    sets come from parsing ``pytest --collect-only -q`` output. No hardcoded
    list of moved test names should exist in the gate's code. This test
    verifies the gate module does not contain a literal list of specific test
    names that could be a hand-maintained moved-test set.
    """
    import charlie_work.collect_only_gate as gate_mod

    source = Path(gate_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=gate_mod.__file__)

    # Collect all string-literal lists (list/tuple/set of string constants).
    suspicious: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            elements = node.elts
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in elements):
                # A list of 2+ string literals that look like test names is
                # suspicious. The gate's only string-literal lists are the
                # finding-kind constants (single-element tuples), not test names.
                values = [e.value for e in elements]
                joined = " ".join(values)
                if len(values) >= 2 and "test" in joined.lower():
                    suspicious.append(str(values))

    assert not suspicious, (
        f"Gate source contains suspicious string-literal lists that look like "
        f"hardcoded test names: {suspicious}. The moved-test set must be "
        f"diff-derived (graft E, rule #9), never hand-typed."
    )


# ---------------------------------------------------------------------------
# render_gate_report
# ---------------------------------------------------------------------------


def test_render_gate_report_pass() -> None:
    """A passing result renders a brief PASSED summary."""
    result = CollectOnlyResult(
        base_leaf_counts=Counter({"test_a": 1}),
        head_leaf_counts=Counter({"test_a": 1}),
    )
    report = render_gate_report(result)
    assert "PASSED" in report
    assert "1" in report  # leaf count


def test_render_gate_report_fail() -> None:
    """A failing result renders findings with leaf names and details."""
    result = CollectOnlyResult(
        base_leaf_counts=Counter({"test_a": 1, "test_b": 1}),
        head_leaf_counts=Counter({"test_a": 1}),
        findings=(
            CollectOnlyFinding(
                kind="missing_sibling",
                leaf_name="test_b",
                source_module="tests/test_foo.py",
                detail="leaf removed from tests/test_foo.py",
            ),
        ),
    )
    report = render_gate_report(result)
    assert "FAILED" not in report  # render_gate_report uses markdown, not "FAILED"
    assert "missing_sibling" in report
    assert "test_b" in report
    assert "tests/test_foo.py" in report
    assert "enforcement" in report.lower()


# ---------------------------------------------------------------------------
# CLI command (charlie collect-only-check)
# ---------------------------------------------------------------------------


def _make_cli_args(
    tmp_path: Path,
    *,
    base_collect: str = "base_collect.txt",
    head_collect: str = "head_collect.txt",
    output: str | None = None,
) -> argparse.Namespace:
    """Build the argparse namespace for ``collect-only-check``."""
    return argparse.Namespace(
        command="collect-only-check",
        base_collect=base_collect,
        head_collect=head_collect,
        output=output,
        repo=None,
        config=None,
        fleet_dir=None,
        dry_run=True,
    )


def _apply_cli_mocks(monkeypatch, tmp_path: Path) -> None:
    """Mock ``cli.bootstrap_command`` to return a context rooted at *tmp_path*."""
    from charlie_work import cli as cli_module

    def mock_bootstrap(args):
        from charlie_work.config import OrchestratorConfig
        from charlie_work.github import GitHub
        from charlie_work.paths import RuntimePaths

        return cli_module.CommandContext(
            repo_root=tmp_path,
            config=OrchestratorConfig(),
            paths=RuntimePaths.__new__(RuntimePaths),
            gh=GitHub(repo_root=tmp_path, runtime=None, dry_run=True),
        )

    monkeypatch.setattr(cli_module, "bootstrap_command", mock_bootstrap)


def test_cli_collect_only_check_passes_verbatim_relocation(monkeypatch, tmp_path: Path) -> None:
    """The CLI command passes a verbatim relocation (same leaf names, different modules)."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "base_collect.txt").write_text(
        "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n", encoding="utf-8"
    )
    (tmp_path / "head_collect.txt").write_text(
        "tests/test_foo_split.py::test_a\ntests/test_foo_split.py::test_b\n",
        encoding="utf-8",
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path))
    assert result.ok is True
    assert result.data["base_leaf_count"] == 2
    assert result.data["head_leaf_count"] == 2
    assert result.data["findings"] == []


def test_cli_collect_only_check_fails_on_missing_leaf(monkeypatch, tmp_path: Path) -> None:
    """The CLI command fails loudly on a genuinely missing leaf (positive control support)."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "base_collect.txt").write_text(
        "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n", encoding="utf-8"
    )
    (tmp_path / "head_collect.txt").write_text("tests/test_foo.py::test_a\n", encoding="utf-8")
    result = run_collect_only_check_command(_make_cli_args(tmp_path))
    assert result.ok is False
    assert len(result.data["findings"]) > 0
    kinds = {f["kind"] for f in result.data["findings"]}
    assert "removed" in kinds


def test_cli_collect_only_check_passes_pure_addition(monkeypatch, tmp_path: Path) -> None:
    """The CLI command passes a pure-addition head (Scope table: ``added`` is
    reported, not enforced)."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "base_collect.txt").write_text("tests/test_foo.py::test_a\n", encoding="utf-8")
    (tmp_path / "head_collect.txt").write_text(
        "tests/test_foo.py::test_a\ntests/test_new_module.py::test_b\n", encoding="utf-8"
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path))
    assert result.ok is True
    assert result.data["failing_count"] == 0
    assert len(result.data["findings"]) == 1
    finding = result.data["findings"][0]
    assert finding["kind"] == "added"
    assert finding["leaf_name"] == "test_b"
    assert finding["fails"] is False


def test_cli_collect_only_check_fails_on_missing_collect_file(monkeypatch, tmp_path: Path) -> None:
    """A missing collect file returns ok=False (fail-closed), not a crash."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    result = run_collect_only_check_command(_make_cli_args(tmp_path))
    assert result.ok is False
    assert "not found" in result.message or "collect file" in result.message.lower()


def test_cli_collect_only_check_writes_output_file(monkeypatch, tmp_path: Path) -> None:
    """``--output`` writes the gate report to the named file."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "base_collect.txt").write_text("tests/test_foo.py::test_a\n", encoding="utf-8")
    (tmp_path / "head_collect.txt").write_text("tests/test_foo.py::test_a\n", encoding="utf-8")
    args = _make_cli_args(tmp_path, output="gate_report.txt")
    result = run_collect_only_check_command(args)
    assert result.ok is True
    report = (tmp_path / "gate_report.txt").read_text(encoding="utf-8")
    assert "PASSED" in report


# ---------------------------------------------------------------------------
# CI workflow shell compatibility (PR #1595 rework -- the #1624 bug class)
# ---------------------------------------------------------------------------

_CI_YML = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def test_collect_only_gate_job_steps_use_bash_shell() -> None:
    """The collect-only-gate job's bash-syntax steps must declare ``shell: bash``.

    The ``Collect tests at head`` step uses the bash ``|| true`` idiom
    (``true`` is not a pwsh command), the ``Collect tests at base`` step is a
    bash script (``set -e``, ``trap``, a subshell), and the
    ``Run collect-only gate`` step uses bash ``\\`` line continuations in a
    multi-line ``run: |`` block.  The job runs on ``windows-latest``, whose
    default shell is pwsh -- pwsh cannot parse ``\\`` at end of line (it
    treats it as a literal backslash, then sees ``--base-collect`` on the next
    line as a bare unary ``--`` operator and throws ``ParserError`` -- the
    #1624 failure, reproduced on this job's first run, 35260669565).  All
    three steps must therefore explicitly set ``shell: bash`` (Git Bash is
    always available on GitHub-hosted Windows runners).

    This is the mutation-checkable regression guard for the #1595 rework:
    reverting any of the three ``shell: bash`` lines makes the test fail.
    """
    assert _CI_YML.exists(), f"ci.yml not found at {_CI_YML}"
    workflow = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))

    job = workflow["jobs"]["collect-only-gate"]
    by_name = {s["name"]: s for s in job["steps"] if "name" in s}

    head_step = by_name["Collect tests at head"]
    assert head_step.get("shell") == "bash", (
        "ci.yml 'Collect tests at head' step must set shell: bash -- the "
        "|| true idiom is bash, not pwsh (true is not a pwsh command)"
    )

    base_step = by_name["Collect tests at base"]
    assert base_step.get("shell") == "bash", (
        "ci.yml 'Collect tests at base' step must set shell: bash -- the "
        "run block is a bash script (set -e, trap, subshell) that pwsh "
        "cannot parse"
    )

    gate_step = by_name["Run collect-only gate"]
    assert gate_step.get("shell") == "bash", (
        "ci.yml 'Run collect-only gate' step must set shell: bash -- the "
        "multi-line run block uses bash \\ line continuations that pwsh "
        "cannot parse (ParserError: Missing expression after unary '--')"
    )
