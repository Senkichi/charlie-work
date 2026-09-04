"""Tests for the recorded-vs-collected gate (issue #1621).

Covers:

* :func:`count_junit_testcases` -- counting ``<testcase>`` elements and
  summing ``<testsuite tests=...>`` attributes (single + multiple suites,
  xdist-style multi-suite, missing attributes, malformed XML).
* :func:`count_collected_tests` -- the three collect-only formats: the
  ``N tests collected`` summary line, the ``-q`` ``file: count`` per-file
  format (summed), and the non-``-q`` node-ID-per-line format (counted).
* :func:`compare_recorded_vs_collected` -- the recorded-vs-collected
  assertion: pass on equality, fail on truncation (recorded < collected),
  fail on internal junit inconsistency, fail on empty sides.
* :func:`render_gate_report` -- report rendering for pass and fail.
* The CLI command ``charlie junit-recorded-check``: pass, fail, missing
  files, --output.
* Rule #9 compliance: no hardcoded test count in the gate source.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

from charlie_work.junit_recorded_gate import (
    compare_recorded_vs_collected,
    count_collected_tests,
    count_junit_testcases,
    render_gate_report,
)
from charlie_work.junit_recorded_gate_command import (
    run_junit_recorded_check_command,
)


# ---------------------------------------------------------------------------
# count_junit_testcases
# ---------------------------------------------------------------------------


def test_count_junit_single_suite() -> None:
    """A single <testsuite> with N <testcase> elements counts N."""
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuites name="pytest tests">'
        '<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="3" time="0.1">'
        '<testcase classname="t" name="a" time="0.01" />'
        '<testcase classname="t" name="b" time="0.01" />'
        '<testcase classname="t" name="c" time="0.01" />'
        "</testsuite></testsuites>"
    )
    result = count_junit_testcases(xml)
    assert result.testcase_count == 3
    assert result.suite_tests_attr == 3
    assert result.internal_consistency_ok is True


def test_count_junit_multi_suite_xdist() -> None:
    """Multiple <testsuite> blocks (xdist) are summed across the tree."""
    xml = (
        "<testsuites>"
        '<testsuite name="pytest" tests="2" errors="0" failures="0" skipped="0">'
        '<testcase classname="t" name="a" time="0" />'
        '<testcase classname="t" name="b" time="0" />'
        "</testsuite>"
        '<testsuite name="pytest" tests="2" errors="0" failures="0" skipped="0">'
        '<testcase classname="t" name="c" time="0" />'
        '<testcase classname="t" name="d" time="0" />'
        "</testsuite>"
        "</testsuites>"
    )
    result = count_junit_testcases(xml)
    assert result.testcase_count == 4
    assert result.suite_tests_attr == 4
    assert result.internal_consistency_ok is True


def test_count_junit_internal_inconsistency_detected() -> None:
    """testcase count != testsuite tests attr -> internal_consistency_ok False."""
    # tests attribute says 5 but only 3 testcase elements exist -- the
    # junit writer lost records (the #1621 failure shape).
    xml = (
        '<testsuites><testsuite name="pytest" tests="5" errors="0" failures="0" skipped="0">'
        '<testcase classname="t" name="a" time="0" />'
        '<testcase classname="t" name="b" time="0" />'
        '<testcase classname="t" name="c" time="0" />'
        "</testsuite></testsuites>"
    )
    result = count_junit_testcases(xml)
    assert result.testcase_count == 3
    assert result.suite_tests_attr == 5
    assert result.internal_consistency_ok is False


def test_count_junit_missing_tests_attr_falls_back_to_element_count() -> None:
    """A suite without a tests attr does not spuriously flag inconsistency."""
    xml = (
        '<testsuites><testsuite name="pytest">'
        '<testcase classname="t" name="a" time="0" />'
        '<testcase classname="t" name="b" time="0" />'
        "</testsuite></testsuites>"
    )
    result = count_junit_testcases(xml)
    assert result.testcase_count == 2
    assert result.suite_tests_attr == 2
    assert result.internal_consistency_ok is True


def test_count_junit_malformed_xml_returns_zero() -> None:
    """Malformed XML returns a zero count (fail-closed), not a raised exception."""
    result = count_junit_testcases("<not><closed>")
    assert result.testcase_count == 0
    assert result.suite_tests_attr == 0
    assert result.internal_consistency_ok is False


# ---------------------------------------------------------------------------
# count_collected_tests
# ---------------------------------------------------------------------------


def test_count_collected_summary_line() -> None:
    """The 'N tests collected' summary line is the authoritative count."""
    output = (
        "tests/test_foo.py::test_a\n"
        "tests/test_foo.py::test_b\n"
        "tests/test_foo.py::test_c\n"
        "3 tests collected in 0.05s\n"
    )
    assert count_collected_tests(output) == 3


def test_count_collected_q_file_count_format() -> None:
    """The -q 'file: count' format is summed across every per-file line."""
    # pytest 9.x -q format: no summary line, one 'file: count' line per file.
    output = "tests/test_foo.py: 10\ntests/test_bar.py: 5\ntests/test_baz.py: 241\n"
    assert count_collected_tests(output) == 256


def test_count_collected_q_format_matches_real_repo_total() -> None:
    """The -q per-file sum reproduces the issue's observed collected total."""
    # A representative slice of the real `pytest --collect-only -q` output
    # (pytest 9.x -q format). The gate sums the per-file counts.
    output = (
        "tests/test_worktree.py: 241\ntests/test_charlie_work.py: 11\ntests/test_janitor.py: 1\n"
    )
    assert count_collected_tests(output) == 253


def test_count_collected_node_id_format_no_summary() -> None:
    """Non-q node-ID lines without a summary are counted directly."""
    output = (
        "tests/test_foo.py::test_a\n"
        "tests/test_foo.py::TestClass::test_b\n"
        "tests/test_bar.py::test_c[1]\n"
    )
    assert count_collected_tests(output) == 3


def test_count_collected_empty_output_returns_zero() -> None:
    """Empty / unrecognised output yields 0 (fail-closed)."""
    assert count_collected_tests("") == 0
    assert count_collected_tests("no tests collected\n") == 0
    assert count_collected_tests("some random stderr noise\n") == 0


def test_count_collected_summary_line_wins_over_file_counts() -> None:
    """When both a summary line and file counts are present, the summary wins."""
    output = "tests/test_foo.py: 10\ntests/test_bar.py: 5\n7 tests collected in 0.1s\n"
    assert count_collected_tests(output) == 7


# ---------------------------------------------------------------------------
# compare_recorded_vs_collected
# ---------------------------------------------------------------------------


_JUNIT_3 = (
    '<testsuites><testsuite name="pytest" tests="3" errors="0" failures="0" skipped="0">'
    '<testcase classname="t" name="a" time="0" />'
    '<testcase classname="t" name="b" time="0" />'
    '<testcase classname="t" name="c" time="0" />'
    "</testsuite></testsuites>"
)
_COLLECT_3 = "tests/test_foo.py: 3\n"


def test_compare_passes_when_recorded_equals_collected() -> None:
    """Equal recorded and collected counts pass the gate."""
    result = compare_recorded_vs_collected(_JUNIT_3, _COLLECT_3)
    assert result.ok is True
    assert result.recorded == 3
    assert result.collected == 3
    assert result.findings == ()


def test_compare_fails_on_truncated_tail() -> None:
    """recorded < collected is the #1621 truncation failure mode -> fail."""
    # Collected 6245, junit recorded only 5269 (the L09 reviewer run 1 shape).
    collect = "tests/test_big.py: 6245\n"
    junit = (
        '<testsuites><testsuite name="pytest" tests="5269" errors="0" '
        'failures="0" skipped="0">'
        + "".join(f'<testcase classname="t" name="t{i}" time="0" />' for i in range(5269))
        + "</testsuite></testsuites>"
    )
    result = compare_recorded_vs_collected(junit, collect)
    assert result.ok is False
    assert result.recorded == 5269
    assert result.collected == 6245
    kinds = {f.kind for f in result.findings}
    assert "recorded_vs_collected" in kinds
    assert "missing from junit" in next(
        f.detail for f in result.findings if f.kind == "recorded_vs_collected"
    )


def test_compare_fails_on_internal_junit_inconsistency() -> None:
    """testcase count != testsuite tests attr surfaces a junit_internal finding."""
    # recorded elements == collected (3 == 3), but the tests attr says 5 --
    # the junit writer lost records in its own bookkeeping.
    junit = (
        '<testsuites><testsuite name="pytest" tests="5" errors="0" failures="0" skipped="0">'
        '<testcase classname="t" name="a" time="0" />'
        '<testcase classname="t" name="b" time="0" />'
        '<testcase classname="t" name="c" time="0" />'
        "</testsuite></testsuites>"
    )
    result = compare_recorded_vs_collected(junit, _COLLECT_3)
    assert result.ok is False
    kinds = {f.kind for f in result.findings}
    assert "junit_internal" in kinds


def test_compare_fails_on_empty_junit() -> None:
    """An empty / malformed junit surfaces empty_junit and recorded_vs_collected."""
    result = compare_recorded_vs_collected("<not><closed>", _COLLECT_3)
    assert result.ok is False
    kinds = {f.kind for f in result.findings}
    assert "empty_junit" in kinds
    assert "recorded_vs_collected" in kinds


def test_compare_fails_on_empty_collect() -> None:
    """An empty collect-only output surfaces empty_collect and recorded_vs_collected."""
    result = compare_recorded_vs_collected(_JUNIT_3, "")
    assert result.ok is False
    kinds = {f.kind for f in result.findings}
    assert "empty_collect" in kinds
    assert "recorded_vs_collected" in kinds


def test_compare_extra_in_junit_fails() -> None:
    """recorded > collected also fails (extra records, not just truncation)."""
    collect = "tests/test_foo.py: 2\n"
    junit = (
        '<testsuites><testsuite name="pytest" tests="3" errors="0" failures="0" skipped="0">'
        '<testcase classname="t" name="a" time="0" />'
        '<testcase classname="t" name="b" time="0" />'
        '<testcase classname="t" name="c" time="0" />'
        "</testsuite></testsuites>"
    )
    result = compare_recorded_vs_collected(junit, collect)
    assert result.ok is False
    assert "extra in junit" in next(
        f.detail for f in result.findings if f.kind == "recorded_vs_collected"
    )


# ---------------------------------------------------------------------------
# render_gate_report
# ---------------------------------------------------------------------------


def test_render_report_pass() -> None:
    """A passing result renders a one-line PASSED summary."""
    result = compare_recorded_vs_collected(_JUNIT_3, _COLLECT_3)
    report = render_gate_report(result)
    assert "PASSED" in report
    assert "3" in report


def test_render_report_fail_lists_findings() -> None:
    """A failing result renders the findings list with counts."""
    result = compare_recorded_vs_collected(_JUNIT_3, "tests/test_foo.py: 10\n")
    report = render_gate_report(result)
    assert "FAILED" not in report  # render_gate_report uses markdown headings
    assert "recorded_vs_collected" in report
    assert "Recorded" in report
    assert "Collected" in report


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def _make_cli_args(
    tmp_path: Path,
    *,
    junit: str = "pytest-junit.xml",
    collect: str = "collected.txt",
    output: str | None = None,
) -> argparse.Namespace:
    """Build the argparse namespace for ``junit-recorded-check``."""
    return argparse.Namespace(
        command="junit-recorded-check",
        junit=junit,
        collect=collect,
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


def test_cli_passes_when_recorded_equals_collected(monkeypatch, tmp_path: Path) -> None:
    """The CLI command passes when junit testcase count == collected count."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "pytest-junit.xml").write_text(_JUNIT_3, encoding="utf-8")
    (tmp_path / "collected.txt").write_text(_COLLECT_3, encoding="utf-8")
    result = run_junit_recorded_check_command(_make_cli_args(tmp_path))
    assert result.ok is True
    assert result.data["recorded"] == 3
    assert result.data["collected"] == 3
    assert result.data["findings"] == []


def test_cli_fails_on_truncation(monkeypatch, tmp_path: Path) -> None:
    """The CLI command fails when recorded < collected (truncated tail)."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "pytest-junit.xml").write_text(_JUNIT_3, encoding="utf-8")
    (tmp_path / "collected.txt").write_text("tests/test_foo.py: 10\n", encoding="utf-8")
    result = run_junit_recorded_check_command(_make_cli_args(tmp_path))
    assert result.ok is False
    kinds = {f["kind"] for f in result.data["findings"]}
    assert "recorded_vs_collected" in kinds


def test_cli_fails_on_missing_junit_file(monkeypatch, tmp_path: Path) -> None:
    """A missing junit file returns ok=False (fail-closed), not a crash."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "collected.txt").write_text(_COLLECT_3, encoding="utf-8")
    result = run_junit_recorded_check_command(_make_cli_args(tmp_path))
    assert result.ok is False
    assert "not found" in result.message or "junit" in result.message.lower()


def test_cli_fails_on_missing_collect_file(monkeypatch, tmp_path: Path) -> None:
    """A missing collect file returns ok=False (fail-closed), not a crash."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "pytest-junit.xml").write_text(_JUNIT_3, encoding="utf-8")
    result = run_junit_recorded_check_command(_make_cli_args(tmp_path))
    assert result.ok is False
    assert "not found" in result.message or "collect" in result.message.lower()


def test_cli_writes_output_file(monkeypatch, tmp_path: Path) -> None:
    """``--output`` writes the gate report to the named file."""
    _apply_cli_mocks(monkeypatch, tmp_path)
    (tmp_path / "pytest-junit.xml").write_text(_JUNIT_3, encoding="utf-8")
    (tmp_path / "collected.txt").write_text(_COLLECT_3, encoding="utf-8")
    args = _make_cli_args(tmp_path, output="gate_report.txt")
    result = run_junit_recorded_check_command(args)
    assert result.ok is True
    report = (tmp_path / "gate_report.txt").read_text(encoding="utf-8")
    assert "PASSED" in report


# ---------------------------------------------------------------------------
# Rule #9 compliance: no hardcoded test count in the gate source
# ---------------------------------------------------------------------------


def test_no_hardcoded_test_count_in_gate_source() -> None:
    """The gate source contains no hardcoded expected test count (rule #9).

    Both counts are derived from pytest's own output at runtime; a literal
    like ``6245`` or ``5959`` in the gate logic would be a brittle hardcoded
    list. The issue numbers and example counts appear only in docstrings and
    comments (which ``ast.get_docstring`` / string literals would surface),
    so we scan only call positions -- but a simpler guard is to assert no
    integer literal > 1000 appears in a function body outside a docstring.
    """
    import charlie_work.junit_recorded_gate as mod

    tree = ast.parse(ast.unparse(ast.parse(open(mod.__file__, encoding="utf-8").read())))
    # Walk only function bodies (not module docstrings, not standalone string
    # literals that are docstrings) for large integer literals.
    big_literals: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, int):
                    if child.value > 1000:
                        big_literals.append(child.value)
    assert big_literals == [], (
        f"gate source contains large integer literals {big_literals} -- "
        f"a hardcoded test count would be brittle (rule #9)"
    )
