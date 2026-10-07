"""Linked-issue label reads for the **Merge path**, served from the per-pass cache.

``gh.issue_list(state="open")`` is cached for the whole pass (the dispatch and
reap scans already pay for it), so a label check that finds the issue there
costs zero GitHub calls. An issue the list does not show (closed, past the list
limit, or an unreadable list) returns ``None`` and the caller falls back to its
live ``issue_view`` read -- the cache only ever removes calls, it never turns a
miss into "no labels".
"""

from __future__ import annotations

from typing import Any

from ..github import GitHubError, label_names


def open_issue_labels(gh: Any, issue_number: int) -> set[str] | None:
    """Label names of open issue ``issue_number`` from the cached list, else ``None``."""
    try:
        listed = gh.issue_list(state="open")
    except (GitHubError, ValueError):
        return None
    if not isinstance(listed, list):
        return None
    for issue in listed:
        if isinstance(issue, dict) and issue.get("number") == issue_number and "labels" in issue:
            return label_names(issue)
    return None
