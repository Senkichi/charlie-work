"""Local-only-repo (``local_issues.enabled``) gating for ``heartbeat_check.py`` (issue #1861).

A ``local_issues.enabled`` repo (e.g. ``local/mdls``) has no GitHub remote,
so every ``gh <issue|pr> list -R <slug>`` can only ever fail with "Could
not resolve to a Repository" -- a permanent spurious ANOMALY per beat per
gh-based check. This module holds the pieces `heartbeat_check.py`'s
gh-based checks use to skip cleanly instead: the shared skip-line detail,
the layered-config precedence resolver, and a helper per check whose
local-only-repo branch needs more than the bare one-liner
(``report.ok(check, LOCAL_ONLY_SKIP_DETAIL); return``) the simple checks
use directly.

Loaded from `heartbeat_check.py` via ``importlib`` from the sibling script
path (see that file's ``_load_heartbeat_local_repo``), never a bare
``import`` -- matching how ``git_push_lint_hook.py`` loads
``worker_stop_gate.py``: ``scripts/`` is not a package and is deliberately
kept off ``sys.path`` by the test harness (``tests/_script_loader.py``).
Extracted verbatim out of `heartbeat_check.py` (file-size ratchet, #1879)
with no behavior change -- every helper here reproduces exactly the
statements (and their order) the call site used to run inline.

Stdlib-only, same constraint as `heartbeat_check.py` itself
(scripts/README.md): no `charlie_work` or third-party imports.
"""

from __future__ import annotations

from typing import Any

# ``charlie_work.layout.GLOBAL_CONFIG_FILENAME`` mirrored under the
# stdlib-only invariant (scripts/README.md): the fleet-wide config layer
# ``load_repos`` consults for ``local_issues.enabled`` when a repo's own
# config does not set the key.
FLEET_GLOBAL_CONFIG_FILENAME = "config.yaml"

# Skip detail emitted by every gh-based check on a ``local_issues.enabled``
# repo. The check reports this OK line instead of calling gh: visible,
# uniform with the one-line-per-check contract, and never an anomaly.
LOCAL_ONLY_SKIP_DETAIL = "skipped: local-only repo (local_issues.enabled; no GitHub remote)"


def local_issues_enabled(repo_config: dict[str, Any], fleet_config: dict[str, Any]) -> bool:
    """Effective ``local_issues.enabled`` under layered-config precedence.

    Mirrors ``global_config.load_layered_config``'s per-key precedence for
    this one knob (stdlib-only reimplementation -- the script cannot import
    the package): the repo's own config wins when it sets
    ``local_issues.enabled``; otherwise the fleet layer's value applies; a
    layer that does not set the key falls through to the next. Anything
    that is not a mapping or not literally ``True`` reads as disabled --
    an unreadable or ambiguous config cannot prove the repo is local, so it
    is treated as GitHub-backed and the gh-based checks still run (the same
    posture ``fleet_registry`` documents for its runner count: fall through
    to the GitHub path on unclassifiable config).
    """
    for layer in (repo_config, fleet_config):
        section = layer.get("local_issues")
        if isinstance(section, dict) and "enabled" in section:
            return section["enabled"] is True
    return False


def skip_dispatch_coverage(
    report: Any,
    check: str,
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
) -> None:
    """``check_dispatch_coverage``'s local-only-repo report/carry-forward (verbatim move).

    The caller still runs ``check_dispatch_throttle``/``check_in_progress_staleness``
    directly (they live in `heartbeat_check.py`, which imports *this*
    module, so importing back here would cycle) -- this covers only the
    ``report.ok``/state-carry pair that used to sit ahead of those calls.
    """
    report.ok(check, LOCAL_ONLY_SKIP_DETAIL)
    # Nothing gh-derived was measured this beat; carry the last real
    # beat's snapshot forward untouched, same as skip_delta.
    new_repo_state["dispatchable_issues"] = prev_repo_state.get("dispatchable_issues", [])


def skip_in_progress_staleness(
    report: Any,
    check: str,
    prev_map: dict[str, str],
    new_repo_state: dict[str, Any],
) -> None:
    """``check_in_progress_staleness``'s local-only-repo branch (verbatim move).

    The in-progress set is gh-derived (`check_dispatch_coverage`'s
    ``gh issue list``), which cannot run on a local-only repo. Skipping
    rather than deriving it from state.json's issue mirror on purpose: the
    mirror lags (observed: a merged issue still carried
    ``agent:in-progress`` a day after its file went ``state: closed``), so
    deriving here would manufacture exactly the kind of permanent false
    ANOMALY the #1861 gate exists to remove.
    """
    new_repo_state["in_progress"] = prev_map
    report.ok(check, LOCAL_ONLY_SKIP_DETAIL)


def skip_review_liveness(report: Any, check: str) -> None:
    """``check_review_liveness``'s local-only-repo branch (verbatim move).

    Local-lane review claims live under the same ``prs/pr-<n>/`` packet
    dirs, but the "still open" set this check filters them against comes
    from ``gh pr list`` -- unrunnable here.
    """
    report.ok(check, LOCAL_ONLY_SKIP_DETAIL)


def skip_merge_flow(
    report: Any,
    check: str,
    prev_repo_state: dict[str, Any],
    new_repo_state: dict[str, Any],
) -> None:
    """``check_merge_flow``'s local-only-repo branch (verbatim move).

    Carries the delta snapshot forward untouched, same as ``skip_delta``:
    nothing was measured this beat.
    """
    new_repo_state["mergequeue_count"] = prev_repo_state.get("mergequeue_count")
    new_repo_state["mergequeue_unchanged_streak"] = prev_repo_state.get(
        "mergequeue_unchanged_streak", 0
    )
    new_repo_state["last_merged_at"] = prev_repo_state.get("last_merged_at")
    report.ok(check, LOCAL_ONLY_SKIP_DETAIL)
