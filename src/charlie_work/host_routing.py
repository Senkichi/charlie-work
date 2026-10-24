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

from collections.abc import Iterable
from typing import TYPE_CHECKING

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
