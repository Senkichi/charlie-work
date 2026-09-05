"""Tests for the devin-shell reviewer harness sanitizer (issue #1513).

Extracted from ``tests/test_devin_shell.py`` as part of the attachment-contracts
ratchet remedy (issue #1616): ``test_devin_shell.py`` exceeded its baselined
ceiling by one member, and the over-ceiling test is the #1535 reviewer-harness
sanitizer test, which belongs with the harness-routing subject rather than the
adapter's launch/reap surface.
"""

from __future__ import annotations

from charlie_work.devin_shell import (
    DEFAULT_COMMAND_TEMPLATE,
    _REVIEW_COMMAND_TEMPLATE,
    _sanitize_review_command_template,
)


def test_sanitize_review_command_template_strips_permission_mode() -> None:
    """Issue #1513: a reviewer launch must never carry ``--permission-mode
    dangerous`` regardless of what ``DevinConfig.command`` (a field shared
    with worker dispatch) supplies. Sanitizing the worker default must
    reproduce the documented ``_REVIEW_COMMAND_TEMPLATE`` constant exactly,
    and the flag must be stripped in both its "flag + value token" form and
    its "--permission-mode=value" form, with nothing appended in its place
    (unlike claude_code's reviewer sanitizer, which pins to ``plan``)."""
    assert _sanitize_review_command_template(DEFAULT_COMMAND_TEMPLATE) == _REVIEW_COMMAND_TEMPLATE
    assert _sanitize_review_command_template(
        ("devin", "--permission-mode=dangerous", "--permission-mode", "dangerous", "--print")
    ) == ("devin", "--print")
    # No occurrence at all is a no-op.
    assert _sanitize_review_command_template(("devin", "--print")) == ("devin", "--print")
