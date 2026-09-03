"""Tests for the collect-only gate (issue #1538).

Covers:

* :func:`parse_collect_only_output` -- parsing ``pytest --collect-only -q``
  output into node IDs (filtering summary lines, normalizing paths).
* :func:`extract_leaf_name` -- splitting a node ID into (module_path,
  leaf_name), stripping the module-path prefix.
* :func:`collect_leaf_names` -- aggregating leaf names into multisets per
  module and overall.
* :func:`compare_collect_only` -- the two clauses (graft K):
  - Clause 1: leaf-name multiset equality (verbatim relocation passes, rename
    / addition / deletion / count-mismatch fail).
  - Clause 2: sibling reappearance (leaf removed from a module under tests/
    must reappear in a sibling under tests/).
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


def test_added_leaf_fails() -> None:
    """A net-new leaf (addition without removal) fails the multiset check."""
    base = "tests/test_foo.py::test_a\n"
    head = "tests/test_foo.py::test_a\ntests/test_foo.py::test_new\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    assert any(f.kind == "added" and f.leaf_name == "test_new" for f in result.findings)


def test_removed_leaf_fails() -> None:
    """A deleted leaf (removal without re-addition) fails the multiset check."""
    base = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
    head = "tests/test_foo.py::test_a\n"
    result = compare_collect_only(base, head)
    assert result.ok is False
    assert any(f.kind == "removed" and f.leaf_name == "test_b" for f in result.findings)


def test_count_mismatch_fails() -> None:
    """A parametrize case dropped or added (count mismatch) fails."""
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
