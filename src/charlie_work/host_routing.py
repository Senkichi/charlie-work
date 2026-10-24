"""Host routing: the ``host:<name>`` issue label (spec C2, interface I-7).

Two orchestrators on two hosts can serve one repository. Every issue belongs to
exactly one of them, or to neither:

- an issue labelled ``<prefix><host>`` belongs to the orchestrator whose
  ``dispatch.host`` is ``<host>`` (case-insensitive);
- an issue with no host label belongs to the orchestrator whose
  ``dispatch.default_host`` is true;
- an issue with two different host labels belongs to neither, so no host claims
  it twice. It shows up in the ``other_host`` backlog bin on both.

With ``dispatch.host`` unset (the kill switch) or an empty ``labels.host_prefix``
every issue is local, which is exactly the behaviour before host routing.
Dispatch, the backlog classifier, reconcile, finalize and the review/merge lane
all ask :func:`owned_here`, so the rule lives in one place.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .github import GitHubError
from .issue_linking import linked_issue_number

if TYPE_CHECKING:
    from .config import OrchestratorConfig


def issue_hosts(labels: Iterable[str], prefix: str) -> tuple[str, ...]:
    """The distinct hosts named by ``<prefix><host>`` labels, lower-cased and sorted.

    Empty when the issue has no host label or ``prefix`` is empty (routing off).
    """
    if not prefix:
        return ()
    folded = prefix.lower()
    hosts = {
        label[len(prefix) :].strip().lower()
        for label in labels
        if label.lower().startswith(folded)
    }
    return tuple(sorted(host for host in hosts if host))


def owned_here(labels: Iterable[str], config: OrchestratorConfig) -> bool:
    """True when the issue carrying ``labels`` belongs to this orchestrator."""
    host = config.dispatch.host
    prefix = config.labels.host_prefix
    if host is None or not prefix:
        return True
    hosts = issue_hosts(labels, prefix)
    if not hosts:
        return config.dispatch.default_host
    return hosts == (host.strip().lower(),)


def _label_names(item: Mapping[str, Any]) -> set[str]:
    return {str(label.get("name") or "") for label in item.get("labels") or []}


def foreign_issue_numbers(
    issues: Iterable[Mapping[str, Any]], config: OrchestratorConfig
) -> frozenset[int]:
    """Numbers of the issues in ``issues`` that another host owns (CW-2).

    Reconcile uses this to leave those issues, and the PRs bound to them, alone.
    Empty when routing is off.
    """
    if config.dispatch.host is None:
        return frozenset()
    return frozenset(
        int(issue["number"])
        for issue in issues
        if issue.get("number") is not None and not owned_here(_label_names(issue), config)
    )


def local_issues(issues: list[dict[str, Any]], config: OrchestratorConfig) -> list[dict[str, Any]]:
    """``issues`` without the ones another host owns; the same list when routing is off."""
    if config.dispatch.host is None:
        return issues
    return [issue for issue in issues if owned_here(_label_names(issue), config)]


def local_prs(app: Any, prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The PRs whose linked issue this host owns (the review/merge lane, CW-2).

    A PR inherits its linked issue's owner. A PR with no linked open issue (an
    operator PR, or a stale branch) counts as unlabelled, so the default host
    keeps it. When the open-issue fetch fails the default host keeps every PR,
    which is today's behaviour, and any other host keeps none: it never reviews
    or merges a PR it cannot prove it owns.
    """
    config = app.config
    if config.dispatch.host is None or not prs:
        return prs
    try:
        open_issues = app.gh.issue_list(state="open")
    except GitHubError:
        return prs if config.dispatch.default_host else []
    labels_by_issue = {
        int(issue["number"]): _label_names(issue)
        for issue in open_issues
        if issue.get("number") is not None
    }
    kept = []
    for pr in prs:
        number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
        )
        if owned_here(labels_by_issue.get(number, set()), config):
            kept.append(pr)
    return kept
