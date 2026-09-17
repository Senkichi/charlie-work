"""Regression tests for the gh ``--json`` field-list lint matcher (issue #1609).

The repo-wide scan lives in ``tests/test_doctor.py::
test_gh_field_lists_use_constants_no_inline_literals``; these tests are its
positive controls for the single-positional-list call shape
``gh.run([...], json_output=True)`` -- the shape the matcher never inspected
before #1609. They feed the shared matcher (``_field_list_lint.
_find_gh_field_list_violations``) synthetic modules so a vacuous pass against
that shape can never recur.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _field_list_lint import _find_gh_field_list_violations


def test_field_list_lint_detects_single_positional_list_shape() -> None:
    """Regression test for issue #1609: the single-positional-list call shape
    ``gh.run([...], json_output=True)`` must be inspected, not skipped.

    Before the fix, ``node.args`` held one ``ast.List`` for this shape, the
    positional loop saw no ``ast.Constant``, and ``node.keywords`` had no
    ``args=`` entry, so the call was skipped entirely -- the lint passed
    vacuously. This test feeds the matcher a synthetic module using that shape
    with an inline ``"number"`` literal after ``--json`` and asserts a
    violation is reported (the positive control this test has never had).
    """
    synthetic = (
        "def probe(gh):\n"
        "    return gh.run(\n"
        '        ["pr", "list", "--state", "all", "--limit", "1", "--json", "number"],\n'
        "        json_output=True,\n"
        "    )\n"
    )
    tree = ast.parse(synthetic, filename="<synthetic>")
    probe_path = Path("<synthetic>")

    violations = _find_gh_field_list_violations(tree, probe_path)

    assert len(violations) == 1, (
        f"expected one violation for the single-positional-list shape, got {violations}"
    )
    _file_str, lineno, literal = violations[0]
    assert literal == "number"
    assert lineno == 3  # the "number" literal's line in the synthetic source


def test_field_list_lint_passes_single_positional_list_with_constant() -> None:
    """Positive-control complement to issue #1609: when the single-positional-list
    shape references a constant (``PROBE_NUMBER_FIELDS``) instead of an inline
    literal, the matcher must report NO violation -- the element after ``--json``
    is an ``ast.Name``, not an ``ast.Constant``.
    """
    synthetic = (
        "from charlie_work.github import PROBE_NUMBER_FIELDS\n"
        "\n"
        "def probe(gh):\n"
        "    return gh.run(\n"
        '        ["pr", "list", "--state", "all", "--limit", "1", "--json", PROBE_NUMBER_FIELDS],\n'
        "        json_output=True,\n"
        "    )\n"
    )
    tree = ast.parse(synthetic, filename="<synthetic>")
    probe_path = Path("<synthetic>")

    violations = _find_gh_field_list_violations(tree, probe_path)

    assert violations == [], f"expected no violation when a constant is used, got {violations}"
