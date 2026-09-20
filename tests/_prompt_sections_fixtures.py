"""Shared values/helpers for the prompt-sections test modules.

Hoisted verbatim out of ``tests/test_prompt_sections.py`` (issue #1570, Track 1
shoulder) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

from charlie_work.markdown_fence import fenced_block
from charlie_work.prompt_test_command import prompt_test_command_values
from charlie_work.prompts import render_prompt

# The test-command placeholders every worker/rework writer supplies. Taken from the
# production function (no repo, no override -> the unresolved form) so these hand-built
# fixtures cannot drift from what the real writers pass; strict rendering refuses a
# template whose ``$targeted_test_command`` / ``$full_suite_command`` nothing supplies.
TEST_COMMAND_VALUES = dict(prompt_test_command_values("", None))

ISSUE_VALUES = {
    "issue_number": 123,
    "issue_title": "Fix search",
    "issue_url": "https://example.test/issues/123",
    "issue_body": "Body text",
    "issue_body_block": fenced_block("Body text", "md"),
    "branch_name": "agent/issue-123-fix-search",
    "issue_comments": "",
    # Issue #1444: the module-map section value. Empty here because these
    # tests render against a tmp_path with no src/charlie_work tree; the
    # real writer (``_write_worker_prompt``) derives it from the live tree.
    "module_map": "",
    # Issue #1460: the attachment-budget dispatch clause. Empty for the same
    # reason module_map is empty above.
    "attachment_budget": "",
    "pr_number": 456,
    "pr_title": "fix: search is broken",
    "pr_url": "https://example.test/pull/456",
    "dispatch_note": "Fix the typo in the search function.",
    "dispatch_note_block": fenced_block("Fix the typo in the search function.", "md"),
    "required_changes_section": "",
    **TEST_COMMAND_VALUES,
}


def _render_worker_with_sections(template_name: str, variants: tuple[str, ...] = ()) -> str:
    """Render a worker prompt with section variables merged in.

    `render_prompt` now handles section resolution internally, so this helper
    just passes the issue values directly. ``variants`` selects section overlays
    (the slash-command ``skills`` loop is one; the default is the plain git/gh loop).
    """
    return render_prompt(template_name, ISSUE_VALUES, variants=variants)
