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
Intake, dispatch, the backlog classifier, reconcile, finalize and the
review/merge lane all ask :func:`owned_here`, and the unauthorized-merge
tripwire asks its positive-proof sibling :func:`owned_by_another_host`, so
the rule lives in one place.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from .github import GitHubError, build_branch_issue_validator_from_issues
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


def owned_by_another_host(labels: Iterable[str], config: OrchestratorConfig) -> bool:
    """True when exactly one host, and it is not this one, provably owns the issue.

    The positive-proof direction of :func:`owned_here`, for controls that must
    not act on a maybe. Routing off (``dispatch.host`` unset or an empty
    ``labels.host_prefix``) means nothing is foreign. An issue with no host
    label belongs to the default host, which is provable foreign ownership
    only to a non-default host. An issue carrying two or more distinct host
    labels belongs to *neither* host, so no single other host provably owns it
    and this returns False -- the caller must fail towards keeping it in
    scope, not towards dropping it.
    """
    host = config.dispatch.host
    prefix = config.labels.host_prefix
    if host is None or not prefix:
        return False
    hosts = issue_hosts(labels, prefix)
    if len(hosts) > 1:
        return False
    if not hosts:
        return not config.dispatch.default_host
    return hosts[0] != host.strip().lower()


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

    Linkage resolves identically to reap_loop's per-PR scan (issue #2822
    review): the stale-branch guard ``branch_issue_validator`` is built from
    the already-fetched open-issue list, so a stale ``agent/issue-<n>`` branch
    naming a closed issue N is not keyed to N — the resolver falls through to
    the PR body's closing keyword. Without it the two surfaces could disagree
    and the default host could review or merge a PR whose body closes another
    host's issue while the owning host drops it.
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
    branch_issue_validator = build_branch_issue_validator_from_issues(open_issues)
    kept = []
    for pr in prs:
        number = linked_issue_number(
            pr,
            is_cross_repository=pr.get("isCrossRepository"),
            branch_prefix=config.dispatch.branch_prefix,
            branch_issue_validator=branch_issue_validator,
        )
        if owned_here(labels_by_issue.get(number, set()), config):
            kept.append(pr)
    return kept


def drop_foreign_findings(
    app: Any, findings: list[dict[str, Any]], remembered_prs: set[int]
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    """``findings`` minus those whose linked issue another host provably owns (#2910).

    For a control that audits merges that already happened, where the other
    host did the reviewing and so left no evidence in this host's state. Each
    finding names its linked issue under ``issue`` (``int | None``) and its PR
    under ``pr``.

    Returns ``(kept, proven)``: the findings to still report, and a
    ``{pr: issue}`` map of the findings this call newly proved foreign, which
    the caller records durably and feeds back as ``remembered_prs`` on later
    passes so the proof is not re-paid. A remembered PR is dropped without
    any read: foreign ownership was proven when the merge was found, and a
    merge's owner is a fact about when it merged, so a later relabel does not
    re-open it.

    The proof is per-issue and exact -- deliberately not an issue-list
    snapshot, for two reasons:

    - **The list read is capped.** ``gh.issue_list`` goes through
      ``gh issue list --limit 500``, so a backlog issue older than the 500
      newest issues is invisible to it -- precisely the shape that made this
      control's false positive return on a long-lived repo (reconcile had to
      retire the same capped read for the same reason, issue #762).
      ``gh.issue_view(number)`` answers one issue by number and has no
      window.
    - **It fails towards reporting.** :func:`local_prs` answers "may this
      host act on the PR?", so on an unreadable issue list a non-default
      host drops every PR. A finding is evidence of a possible bypass, so it
      is dropped only on positive proof: the issue was read by number and
      :func:`owned_by_another_host` says exactly one other host owns it. A
      finding with no linked issue, a lookup that fails or answers the wrong
      payload, or an unreadable issue all stay.

    With ``dispatch.host`` unset (routing off) the list is returned untouched
    and no issue read is made.

    Ownership is :func:`owned_by_another_host`'s rule, which is deliberately
    stricter than :func:`foreign_issue_numbers`' on one point: an issue
    carrying two or more distinct host labels belongs to *neither* host, so
    no other host provably owns it and the finding is kept -- the tripwire
    would rather report one merge too many than leave a merge no host owns
    unaudited everywhere.
    """
    if app.config.dispatch.host is None:
        return findings, {}
    kept: list[dict[str, Any]] = []
    proven: dict[int, int] = {}
    # Per-pass memo over issue_view calls (each is a subprocess): several
    # findings can name the same issue, and ``None`` marks a looked-up issue
    # whose ownership could not be proven this pass -- it is not re-fetched
    # for a second finding naming it.
    labels_by_issue: dict[int, frozenset[str] | None] = {}
    for finding in findings:
        pr = finding.get("pr")
        issue = finding.get("issue")
        if isinstance(pr, int) and pr in remembered_prs:
            continue  # proven foreign on an earlier pass; durable, so no re-read
        if not isinstance(issue, int) or isinstance(issue, bool):
            kept.append(finding)
            continue
        if issue not in labels_by_issue:
            try:
                view = app.gh.issue_view(issue)
            except (GitHubError, ValueError):
                # GitHubError: the real read failed (gh down, issue gone).
                # ValueError: a test double signals a missing issue the same
                # way. Either way the proof is absent, so the finding stays.
                view = None
            labels_by_issue[issue] = (
                frozenset(_label_names(view))
                if isinstance(view, dict) and view.get("number") == issue
                else None
            )
        labels = labels_by_issue[issue]
        if labels is not None and owned_by_another_host(labels, app.config):
            if isinstance(pr, int) and not isinstance(pr, bool):
                proven[pr] = issue
            continue
        kept.append(finding)
    return kept, proven
