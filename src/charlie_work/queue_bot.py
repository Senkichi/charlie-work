"""Merge-queue bot PR detection (Aviator parallel mode).

Extracted from ``reconcile.py`` so the predicate does not grow that over-cap
module (file-size ratchet, issue #1442); ``reconcile`` re-exports it.
"""

from __future__ import annotations

from typing import Any

from .config import OrchestratorConfig


def is_queue_bot_pr(pr: dict[str, Any], config: OrchestratorConfig) -> bool:
    """Return True if ``pr`` was created by the configured merge-queue bot.

    Aviator parallel mode creates draft PRs on ``mq-bot-*`` branches to test
    queued PR combinations (``mq-tmp-*`` is a separate, PR-less internal
    branch Aviator uses for fast-forward checks -- it never gets a PR at
    all, so it can't reach this filter regardless). These PRs must be
    invisible to the fleet: they are not fleet-owned, have no linked issue,
    and their lifecycle is entirely Aviator's. The ``branch_prefix`` filter
    already excludes them from most paths (dispatch, merge-train, broadcast
    sync), but the reconcile PR loop and the mergequeue detectors enumerate
    ALL PRs and would otherwise emit spurious ``merged_outside_orchestrator``
    / ``closed_unmerged_pr_*`` drift items and pollute ``state["prs"]`` with
    entries the fleet did not create.

    This is the single predicate for that exclusion. It checks the PR's
    ``author.login`` against ``auto_merge.queue_bot_login`` (e.g.
    ``aviator-app[bot]``). When the config key is unset, no PR is excluded.
    """
    queue_bot_login = config.auto_merge.queue_bot_login
    if not queue_bot_login:
        return False
    author = pr.get("author")
    if isinstance(author, dict):
        return author.get("login") == queue_bot_login
    return False
