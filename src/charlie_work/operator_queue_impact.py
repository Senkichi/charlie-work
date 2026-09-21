"""Operator-queue impact measurement and edge-detection (issue #1768).

Replaces the level-triggered ``_maybe_emit_operator_queue_depth`` raw-count
gauge (which fired every pass once any operator-queue item existed,
regardless of change) with an edge-triggered signal keyed on how much of
the automated-ready backlog is transitively blocked behind the sink
(``agent:operator-queue`` / ``agent:human-needed`` / reviewer-``blocked``)
roots, plus the oldest root's age -- not a plain count of roots. A single
root can transitively block the majority of a small repo's backlog (the
"fresh-eyes" case: 1 root, 17 of 21 open issues) while a raw root count
never reflects that; conversely many roots with shallow/no dependents is
lower urgency than a raw count implies.

See the originating investigation (issue #1768) for the full root-cause
analysis and the design options considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .config import OrchestratorConfig
from .github import GitHubError, GitHubLike, label_names, parse_blockers


@dataclass(frozen=True)
class OperatorQueueImpact:
    """Transitive blast radius of the current sink roots.

    ``blocked_ready_issue_numbers`` is the set of *automated-ready* open
    issues (carrying ``config.labels.ready``) reachable from ``roots`` by
    following body-declared blocker edges (``Blocked by #N`` / ``Depends
    on #N``, ``github.parse_blockers``) forward: an issue is "reachable"
    when it declares one of the roots as a blocker, or declares an issue
    that does, transitively. Traversal follows the full open-issue graph
    regardless of label so a chain through a non-ready intermediate issue
    is not cut short; only the *reported* count and number list are
    restricted to ready-labelled issues, matching the operational question
    ("how much of the dispatchable backlog is stuck"), not raw graph
    reachability.

    ``oldest_root_age_days`` is supplied by the caller (it comes from
    ``state.json``'s per-issue ``terminal_since``, not from GitHub) rather
    than computed here, so this module stays a pure GitHub/label reader
    with no ``state.json`` shape knowledge.

    ``observed`` (issue #1768 review findings 1/2) is False when the
    ``gh.issue_list`` fetch failed or came back ambiguously empty --
    mirroring ``classify_backlog_reachability``'s ``observed`` field for
    the identical fetch. Callers MUST treat ``observed=False`` as
    "unknown", never as a genuine zero-impact reading: the other fields
    are still populated (roots, zero counts) so a caller that forgets the
    check gets an inert-looking-but-wrong result rather than a crash, but
    the whole point of this field is that they should not forget the
    check.
    """

    roots: tuple[int, ...]
    blocked_ready_issue_numbers: tuple[int, ...]
    blocked_ready_count: int
    oldest_root_age_days: float | None
    observed: bool = True


def compute_operator_queue_impact(
    gh: GitHubLike,
    config: OrchestratorConfig,
    roots: set[int],
    *,
    oldest_root_age_days: float | None = None,
) -> OperatorQueueImpact:
    """Compute the transitive automated-ready impact of the sink roots.

    Reuses ``gh.issue_list(state="open")`` -- the same unfiltered fetch
    ``classify_backlog_reachability`` already makes every pass -- and
    ``parse_blockers`` (body-text only, zero GitHub calls). Deliberately
    does NOT also union GitHub-native dependencies
    (``get_github_issue_dependencies``, as ``_get_open_blockers_for_issue``
    does): that call is a per-issue fetch, already paid for only the
    subset of issues reachability's dependency-gate arm actually reaches,
    not the whole open backlog. Walking every open issue through it here
    would add real new GitHub API calls -- exactly what this rewrite must
    not do (issue #1768: "reusing the dependency/blocker data the dispatch
    pass already computes ... no new API calls per pass"). The client's
    list cache is cleared once per ``loop()`` pass, not per call
    (``repo_meta.invalidate_list_cache``), so calling ``issue_list`` again
    here is a cache hit whenever ``classify_backlog_reachability`` already
    ran this pass, and a single new list call (not an N+1) otherwise.

    Fails **unobserved**, not open, on a failed or ambiguously-empty fetch
    (issue #1768 review findings 1/2) -- never raises, matching
    ``classify_backlog_reachability``'s advisory contract, but also never
    reports a fetch failure as a genuine zero. ``GitHubClient.run``
    (reached through ``issue_list`` -> ``_list_json``) *raises*
    ``GitHubError`` on a ``gh`` timeout, a missing binary, or a
    non-zero/unparseable exit; this used to be a pure ``state.json`` dict
    scan with zero network I/O, so an uncaught ``GitHubError`` here would
    crash an otherwise-fully-completed loop pass over what is meant to be
    an advisory signal. A ``gh`` call that returns successfully with an
    empty list is *equally* ambiguous with a failed one at this layer (the
    same reasoning ``classify_backlog_reachability`` documents for the
    identical fetch), so it is treated the same way. An empty *root* set,
    by contrast, is unambiguous -- there is nothing to measure, not a
    measurement that failed -- and is reported as a normal, observed,
    zero-impact result.
    """
    sorted_roots = tuple(sorted(roots))
    if not roots:
        return OperatorQueueImpact((), (), 0, oldest_root_age_days, observed=True)

    try:
        issues = gh.issue_list(state="open")
    except GitHubError:
        return OperatorQueueImpact(sorted_roots, (), 0, oldest_root_age_days, observed=False)
    if not issues:
        return OperatorQueueImpact(sorted_roots, (), 0, oldest_root_age_days, observed=False)

    ready_label = config.labels.ready
    dependents: dict[int, set[int]] = {}
    ready_numbers: set[int] = set()
    for issue in issues:
        number = issue.get("number")
        if number is None:
            continue
        number = int(number)
        if ready_label in label_names(issue):
            ready_numbers.add(number)
        for blocker in parse_blockers(issue.get("body") or ""):
            dependents.setdefault(blocker, set()).add(number)

    seen: set[int] = set()
    frontier: list[int] = list(roots)
    while frontier:
        current = frontier.pop()
        for dependent in dependents.get(current, ()):
            if dependent in seen or dependent in roots:
                continue
            seen.add(dependent)
            frontier.append(dependent)

    blocked_ready = tuple(sorted(seen & ready_numbers))
    return OperatorQueueImpact(
        sorted_roots, blocked_ready, len(blocked_ready), oldest_root_age_days, observed=True
    )


# Bucket edges for the oldest-root-age edge-detection dimension, in days.
# Deliberately hardcoded rather than a config knob: this is a
# reporting-resolution constant (how coarse a change has to be to count as
# "different enough to re-alert"), not an operator-tunable business
# threshold like ``operator_queue_depth_threshold`` is.
_AGE_BUCKET_EDGES_DAYS: tuple[float, ...] = (1.0, 3.0, 7.0, 14.0, 30.0)

# Bounded low-rate reminder interval (hours) for a materially-unchanged but
# still-over-threshold queue, per issue #1768 AC1 ("a bounded low-rate
# durable-marker-driven reminder"). Hardcoded for the same reason as the age
# buckets above.
LOW_RATE_REMINDER_HOURS: float = 24.0


def age_bucket_label(age_days: float | None) -> str:
    """Coarse staleness bucket for the oldest root's age.

    Used only for edge-detection (did the bucket change), never as a
    displayed metric on its own -- the raw ``oldest_root_age_days`` is what
    gets emitted and shown to the operator.
    """
    if age_days is None:
        return "unknown"
    for edge in _AGE_BUCKET_EDGES_DAYS:
        if age_days < edge:
            return f"<{edge:g}d"
    return f">={_AGE_BUCKET_EDGES_DAYS[-1]:g}d"


def operator_queue_impact_signature(
    *,
    root_issue_numbers: list[int] | tuple[int, ...],
    over_threshold: bool,
    age_bucket: str,
) -> dict[str, Any]:
    """The comparable subset of fields edge-detection keys on.

    Deliberately excludes ``blocked_ready_count``'s exact value (only
    whether it is over/under the configured threshold) and the raw age
    (only its bucket) -- those are the two dimensions AC1 asks for
    ("a configured impact-threshold crossing" / "age bucket crossing"),
    not "any numeric wobble."
    """
    return {
        "root_issue_numbers": sorted(root_issue_numbers),
        "over_threshold": over_threshold,
        "age_bucket": age_bucket,
    }


def should_fire_operator_queue_impact(
    baseline: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    now: datetime,
    qualifies: bool = True,
    reminder_hours: float = LOW_RATE_REMINDER_HOURS,
) -> bool:
    """Edge-detection gate (issue #1768), mirroring the #817
    ``_filter_fleet_health_transitions`` transition-filter shape at
    repo-level scope instead of per-worker scope.

    Fires when:
      (a) there is no recorded baseline yet AND ``qualifies`` -- the first
          observation of a non-empty root set is itself a material change
          from "no known state" (this is what makes a brand-new
          single-root queue, e.g. the fresh-eyes shape, fire on its very
          first occurrence even though a lone root would never cross a
          raw-count threshold), but only when that first observation
          actually carries impact;
      (b) the root set changed AND ``qualifies``;
      (c) the impact-vs-threshold side flipped; or
      (d) the age bucket changed.

    When none of those hold but the queue is still over threshold, fires
    once more per ``reminder_hours`` elapsed since ``baseline["alerted_at"]``
    -- the bounded "still true, no change" visibility AC1 permits, driven
    from a durable marker (never re-derived by rescanning events.db, per
    the policy this issue also establishes).

    ``qualifies`` (issue #1768 review finding 7) is normally
    ``over_threshold or blocked_ready_count > 0``, computed by the caller
    (this pure function does not see the raw count -- see
    ``operator_queue_impact_signature``'s docstring for why). Gating arms
    (a)/(b) on it stops a brand-new sink arrival or an unrelated root-set
    change with zero transitive impact from producing a
    zero-information warning by itself. Arms (c)/(d) are never gated on
    it: an ``over_threshold`` flip is meaningful in either direction
    (arriving at or clearing the threshold implies ``qualifies`` was true
    for at least one side), and an age-bucket crossing only exists when a
    root -- and therefore some prior qualifying state -- is already
    present. Defaults to ``True`` so this pure function's own direct unit
    tests are unaffected by the parameter's addition; the emitter always
    passes the real computed value.
    """
    if baseline is None:
        return qualifies
    if current["root_issue_numbers"] != baseline.get("root_issue_numbers"):
        return qualifies
    if current["over_threshold"] != baseline.get("over_threshold"):
        return True
    if current["age_bucket"] != baseline.get("age_bucket"):
        return True
    if not current["over_threshold"]:
        return False
    alerted_at = baseline.get("alerted_at")
    if not alerted_at:
        return True
    try:
        alerted_dt = datetime.fromisoformat(str(alerted_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return True
    return (now - alerted_dt).total_seconds() >= reminder_hours * 3600.0
