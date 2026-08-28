"""Static guard for issue #1460: zero line-count logic in the attachment-
budget dispatch clause / review-packet codepath.

This package's binding operator constraint (``model.py``'s module docstring)
is member counts on attachment points, never line counts. This test scans
the SOURCE TEXT of ``review_delta.py`` in full, plus the exact source of the
new #1460 symbols in ``baseline.py`` and ``workflow.py`` (via
``inspect.getsource``, not the whole file -- ``workflow.py`` legitimately
carries #1445's line-count-based over-cap probe elsewhere, which is a
different, unrelated feature this guard must not false-positive on).

Forbidden tokens are line-count-metric IDENTIFIERS (``line_count``,
``cap_lines``, ``LOC``, ``lines_added``, ``added_lines``, ``line_threshold``,
``line_cap``) -- not ``splitlines()`` or ``len()`` themselves, both of which
are legitimately used by ``reconstruct_baseline_head_text`` for diff-hunk
text manipulation (array indexing/bounds, not a size metric fed to a cap
decision).
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from charlie_work.attachment_contracts import baseline, review_delta
from charlie_work.workflow import OrchestratorApp, render_attachment_budget_section

_FORBIDDEN = re.compile(
    r"line_count|cap_lines|\bLOC\b|lines_added|added_lines|line_threshold|line_cap",
    re.IGNORECASE,
)

_REVIEW_DELTA_PATH = Path(review_delta.__file__)


def _assert_clean(label: str, text: str) -> None:
    matches = _FORBIDDEN.findall(text)
    assert not matches, f"{label} contains line-count token(s): {matches}"


def test_review_delta_module_has_no_line_count_tokens() -> None:
    _assert_clean("review_delta.py", _REVIEW_DELTA_PATH.read_text(encoding="utf-8"))


def test_baseline_new_symbols_have_no_line_count_tokens() -> None:
    for fn in (baseline.effective_ceiling, baseline.bump_ack_is_external, baseline.new_bumps):
        _assert_clean(f"baseline.{fn.__name__}", inspect.getsource(fn))


def test_workflow_new_symbols_have_no_line_count_tokens() -> None:
    _assert_clean(
        "OrchestratorApp._build_attachment_budget_value",
        inspect.getsource(OrchestratorApp._build_attachment_budget_value),
    )
    _assert_clean(
        "OrchestratorApp._build_attachment_budget_section",
        inspect.getsource(OrchestratorApp._build_attachment_budget_section),
    )
    _assert_clean(
        "render_attachment_budget_section",
        inspect.getsource(render_attachment_budget_section),
    )


def test_review_delta_never_imports_ast_or_scan_tree() -> None:
    """No AST inspection, no full-tree scan -- pure text/data on already-
    fetched diff/baseline text, per the module's own docstring contract."""
    text = _REVIEW_DELTA_PATH.read_text(encoding="utf-8")
    assert "import ast" not in text
    assert "scan_tree" not in text
