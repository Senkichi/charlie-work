"""Blocker-prefetch and branch-issue-validator delegates moved out of ``OrchestratorApp``.

Track 2 Phase B, L04 batch 1 (design doc
``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``,
Sections 3.1/3.2). Bodies relocated verbatim from ``charlie_work.workflow``;
``workflow_delegation._install_delegates`` re-attaches each top-level ``def``
unwrapped onto ``OrchestratorApp``. ``linked_issue_number`` and
``_authorized_override_matches`` are reached through ``_wf.``:
``linked_issue_number`` is patched on ``charlie_work.workflow`` by the suite
(Tier D, via the ``workflow_mod`` alias), and ``_authorized_override_matches``
is a ``charlie_work.workflow`` module-level def. ``_wf.`` is also the seam for
the shared state primitives ``state_lock``/``load_state`` used by
``_drain_local_blocker_patch_equiv`` (issue #1967), matching
``github_ops_unlinked_prs``. All other free names are imported directly (no
test patches them on ``charlie_work.workflow``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import charlie_work.workflow as _wf
from charlie_work.github import (
    GitHubError,
    build_branch_issue_validator,
    get_github_issue_dependencies,
    parse_blockers,
)
from charlie_work.instrumentation import query_events
from charlie_work.local_issues import drain_patch_equiv_satisfied
from charlie_work.state import StateLockBusy

logger = logging.getLogger(__name__)


def _make_branch_issue_validator(self) -> Callable[[int], bool] | None:
    """Build a validator for branch-name-derived issue numbers (issue #1229).

    Facade re-export of ``github.build_branch_issue_validator`` -- the
    single-point-of-enforcement constructor lives there so the open-number
    extraction is shared with ``build_branch_issue_validator_from_issues``
    (used by ``reconcile.detect_drift``'s snapshot-derived validator) and
    cannot diverge between the rework-routing call sites and the
    module-level sweeps (``_detect_and_handle_orphaned_workers``,
    ``_classify_dead_sessions_and_update_throttle_state``). See
    ``github.build_branch_issue_validator``'s docstring for the full
    rationale (fail-open on API outage, ``_LIST_LIMIT`` tradeoff,
    per-pass caching).

    Issue #1229 rework note: the prior module-level
    ``_build_branch_issue_validator`` delegate was a pure one-line
    pass-through to ``github.build_branch_issue_validator``. It was
    collapsed into its real home (``github.py``) rather than relocated to
    ``checks.py`` because ``github.py`` already imports from ``checks.py``
    at module load (``from .checks import _run_id_from_link``), so
    ``checks.py`` importing ``build_branch_issue_validator`` back at
    module level would create a circular import; a lazy import would
    couple a CI-check-classification module to an unrelated concern for
    zero behavioral benefit. Calling ``github.build_branch_issue_validator``
    directly shrinks the workflow monolith more than relocating the
    delegate would, and this method remains as the in-class facade so
    the nine ``self._make_branch_issue_validator()`` call sites are
    unchanged.
    """
    return build_branch_issue_validator(self.gh)


def _prefetch_blocker_data(self, issues: list[dict[str, Any]]) -> None:
    """Warm the GitHub client's per-pass cache for blocker lookups.

    ``status()`` runs ``_get_open_blockers`` once per ready issue via both
    ``_filter_blocked_issues`` and ``_summarize_issue`` -- and until issue
    #870, each call issued a live, uncached `gh api .../dependencies/
    blocked_by` request plus one live `gh issue view` per declared
    blocker, entirely serially. Measured on the live fleet registry: 62
    ready issues drove 169s of a 184s `fleet status --json` run through
    this exact path, including confirmed duplicate fetches (issues #887/
    #888 each fetched twice; their shared blocker #886 fetched 4 times).

    This method does the equivalent fetching *once* per unique resource
    before any of the unmodified per-issue code below runs:

      1. Fetch all ready issues' GitHub-native dependencies in a single
         batched GraphQL query when the GitHub client supports it. The
         client falls back to per-issue REST calls when the query cannot
         be built (no git remote, no gh auth, etc.).
      2. Union those dependencies with each issue's body-declared
         blockers (pure Python, no I/O) into one deduplicated set of
         blocker issue numbers.
      3. Resolve open/closed state for that whole set in a single
         ``are_issues_open`` call, which itself batches cache misses in
         one GraphQL query when possible (see github.py).

    Both fetch layers cache into ``self.gh._list_cache``, so every
    subsequent call to ``_get_open_blockers`` -- unchanged, still called
    once per issue from two separate call sites -- resolves entirely
    from the warm cache with no further network calls. Structuring it
    this way (fetch each unique resource exactly once, then let the
    existing serial consumers read a warm cache) avoids relying on a
    cache race between concurrent callers to naturally dedupe: nothing
    here fetches the same issue number from two threads at once.

    Must be called once per ``status()`` invocation, before the first
    call to ``_get_open_blockers`` (directly or via
    ``_filter_blocked_issues`` / ``_summarize_issue``). Harmless to skip
    (callers just fall back to the prior, slower, uncached behavior) but
    never harmful to call twice -- the second call is a no-op cache hit.
    """
    issue_numbers = [int(issue["number"]) for issue in issues]
    if not issue_numbers:
        return

    if hasattr(self.gh, "issue_dependencies"):
        try:
            deps_by_number = self.gh.issue_dependencies(issue_numbers)
        except (GitHubError, OSError, ValueError, TypeError):
            # The batch method should fall back internally, but if it
            # raises for any reason, fall back to the per-issue function.
            deps_by_number = {
                number: get_github_issue_dependencies(self.gh, number) for number in issue_numbers
            }
    else:
        # Test doubles and older GitHub-like objects without the batch method.
        deps_by_number = {
            number: get_github_issue_dependencies(self.gh, number) for number in issue_numbers
        }

    all_blockers: set[int] = set()
    for issue in issues:
        issue_number = int(issue["number"])
        all_blockers.update(parse_blockers(issue.get("body", "")))
        all_blockers.update(deps_by_number.get(issue_number, []))

    if all_blockers:
        self.gh.are_issues_open(sorted(all_blockers))


def _drain_local_blocker_patch_equiv(self) -> None:
    """Emit ``local_blocker_satisfied_by_patch_equivalence`` once per satisfied issue.

    Issue #1967. ``LocalFileGitHub.are_issues_open`` has no event channel,
    so when its patch-equivalence fallback is what lets a closed issue stop
    blocking, the client collects the issue on ``_patch_equiv_satisfied``
    instead; this drains that map in one locked read-modify-write -- the
    same batch-after-the-work shape ``_record_unlinked_pr_skips`` uses --
    and records one event per issue. Dedupe against ``events.db``
    (``query_events`` on kind+issue_number) rather than the client-side
    ``_patch_equiv_noted`` set alone, because the fleet loop rebuilds the
    client every pass: without the durable check a surviving branch ref
    would refire the signal on every pass. No-op on the remote backend
    (``drain_patch_equiv_satisfied`` returns ``{}``).

    Failure here is by-value: an informational notice must never outrank
    the pass's real work, so lock contention or a write error is logged
    and swallowed (identical best-effort shape to
    ``_record_unlinked_pr_skips``).
    """
    pending = drain_patch_equiv_satisfied(self.gh)
    if not pending:
        return
    try:
        with _wf.state_lock(self.paths.state_file):
            state = _wf.load_state(self.paths.state_file)
            emitted = False
            for issue_number, (branches, base) in sorted(pending.items()):
                if query_events(
                    self.paths.state_file,
                    kind="local_blocker_satisfied_by_patch_equivalence",
                    issue_number=issue_number,
                ):
                    continue
                state = self._record_event(
                    state,
                    "local_blocker_satisfied_by_patch_equivalence",
                    {
                        "issue_number": issue_number,
                        "branch": branches[0],
                        "branches": list(branches),
                        "base": base,
                    },
                )
                emitted = True
            if emitted:
                self.write_gate.save_state(state)
    except (OSError, ValueError, StateLockBusy) as exc:
        logger.warning(
            "could not record local_blocker_satisfied_by_patch_equivalence for %d issue(s): %s",
            len(pending),
            exc,
        )
