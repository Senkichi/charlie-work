"""Issue #3038: never-under-admit controls for the AC5 write-site text gate.

The AC5 scan in ``test_verdict_provenance_enforcement`` gates its
``record_decision`` half on ``_may_write_review_decision`` so the scan
stops parsing and walking every file under ``src/``. A gate must be proven
incapable of silently dropping a write site the AST matcher finds. The
controls live in their own module because
``test_verdict_provenance_enforcement.py`` is over the file-size ratchet's
cap (issue #1442), which does not allow new members, and the gate machinery
was hoisted to ``_verdict_decision_scan`` (issue #1284's no-cross-test-import
invariant).
"""

from __future__ import annotations

import ast

import pytest

from _verdict_decision_scan import (
    _may_write_review_decision,
    _review_decision_write_sites,
)


# The same never-under-admit controls the issue #2931 gate carries for the
# record_review half of the scan, against ``_review_decision_write_sites``
# (both call shapes it matches: bare Name and attribute).
@pytest.mark.parametrize(
    ("source", "has_site"),
    [
        pytest.param(
            "def f(pr_dir) -> None:\n    record_decision(pr_dir, {})\n", True, id="bare_name"
        ),
        pytest.param(
            "from charlie_work.review_decision import record_decision\n"
            "def f(pr_dir):\n"
            "    record_decision(pr_dir, {}, None)\n",
            True,
            id="bare_name_in_function",
        ),
        pytest.param(
            "def f(pr_dir) -> None:\n    _wf.record_decision(pr_dir, {})\n",
            True,
            id="attribute_form",
        ),
        pytest.param(
            "# — note\ndef f(pr_dir) -> None:\n    _wf.record_decision(pr_dir, {})\n",
            True,
            id="non_ascii_file",
        ),
        # The parser NFKC-normalizes identifiers: a fullwidth ``r`` is still
        # a ``record_decision`` name, though the raw text lacks the needle.
        pytest.param(
            "def f(pr_dir) -> None:\n    _wf.ｒecord_decision(pr_dir, {})\n",
            True,
            id="fullwidth_identifier",
        ),
        pytest.param("app.record_review(1)\n", False, id="other_name"),
        pytest.param("def f():\n    return 1\n", False, id="unrelated"),
    ],
)
def test_review_decision_text_gate_never_drops_a_site_the_ast_matcher_finds(
    source: str, has_site: bool
) -> None:
    # Control: the matcher sees what the case claims, so the gate is compared
    # against a real positive/negative rather than against the author's label.
    checked, exempted = _review_decision_write_sites(ast.parse(source))
    assert bool(checked or exempted) is has_site
    assert _may_write_review_decision(source) is has_site


def test_review_decision_text_gate_admits_the_exempt_site_it_must_still_collect() -> None:
    """The exempted ``merge_authorize`` site is collected, not skipped: the
    scan's exemption census (len == 1) depends on seeing it."""
    source = "def merge_authorize(pr_dir):\n    _wf.record_decision(pr_dir, payload, None)\n"
    checked, exempted = _review_decision_write_sites(ast.parse(source))
    assert checked == []
    assert len(exempted) == 1, (checked, exempted)
    assert _may_write_review_decision(source) is True
