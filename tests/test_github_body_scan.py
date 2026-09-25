"""Facade re-export seam for ``github_body_scan`` (issue #1819 rework).

The body-scanning domain (issue-mention, blocker-declaration, and
prose-dependency scanners plus the shared ``_fenced_block_ranges`` fence
model) was extracted from the over-cap ``github.py`` monolith into
``charlie_work.github_body_scan``. ``github.py`` re-exports the three public
scanners so every ``from charlie_work.github import ...`` caller keeps
working; these tests pin that seam so the re-export cannot silently drop a
name or drift to a second implementation.
"""

from __future__ import annotations

import charlie_work.github as github_module
import charlie_work.github_body_scan as body_scan


def test_github_reexports_body_scanners_by_identity() -> None:
    """``charlie_work.github.X`` is ``charlie_work.github_body_scan.X`` — the
    facade binds the same objects, not look-alike redefinitions."""
    for name in (
        "issue_numbers_mentioned_by_pr",
        "parse_blockers",
        "detect_prose_only_dependencies",
    ):
        assert getattr(github_module, name) is getattr(body_scan, name), name
        assert getattr(github_module, name).__module__ == "charlie_work.github_body_scan"


def test_body_scanners_still_importable_from_github() -> None:
    """The legacy ``from charlie_work.github import X`` import path still
    resolves for all downstream callers that use it."""
    from charlie_work.github import (  # noqa: F401
        detect_prose_only_dependencies,
        issue_numbers_mentioned_by_pr,
        parse_blockers,
    )
