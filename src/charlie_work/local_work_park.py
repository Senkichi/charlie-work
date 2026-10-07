"""Salvage for a backend with no pull requests: the branch IS the deliverable.

``dead_worker_reap._attempt_salvage`` exists to publish completed-but-
unpublished work: push the branch, open a PR. A repo on the local-file issue
source (``local_issues``) has no remote, so neither step can happen --
``push_branch`` would fail, the reaper would relabel the issue ``ready``, and
the redispatch would trip ``worktree_unsafe`` on the very commits the worker
made, recording a worker that SUCCEEDED as an escalation.

For such a backend "published" means: the commits stay on the branch, the
issue moves to the terminal ``review_ready`` label (out of dispatch, and out
of the reclaim lane, which only acts on *active* labels), and a comment on the
issue names the branch for the human who will review and merge it.

Lives outside ``dead_worker_reap`` on purpose: that module is a cap-exempt
moved unit whose symbol set and size are pinned by
``tests/test_dead_worker_reap_split.py``; it carries only the call.

The same capability predicate also gates the other PR-shaped per-pass
``loop()`` lanes (issue #1810): the in-loop reconcile and main-CI reclaim
delegates in ``orchestration/state_pr_capability_lanes.py`` consult
``publishes_pull_requests`` directly.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .config import OrchestratorConfig
from .dead_dispatched_timer import (
    LOCAL_PARK_DEFER_FIELDS,
    LOCAL_PARK_DEFER_MAX_PASSES,
    clear_local_park_deferral,
    dead_dispatched_reap_due,
    defer_or_expire_local_park,
)
from .github import GitHubError, GitHubLike, label_names
from .labels import TransitionOutcome
from .local_lane import branch_diff_result, local_base_branch, probe_branch_ref
from .local_noop_rework_rearm import rearm_no_op_local_rework
from .paths import resolved_layout, runtime_paths
from .rework_prompts import _write_text_atomic
from .state import PASSIVE_OPEN_STATUS, load_state, load_state_locked, state_lock
from .worker_fate import persisted_failure
from .worktree import inspect_worktree_state, read_worker_outcome, worktree_path_for_branch
from .write_gate import WriteGate

if TYPE_CHECKING:
    from .worker import WorkerView

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalParkResult:
    """Verdict from the dead-worker salvage probe
    (:func:`salvage_dead_worker_commits` / :func:`park_salvageable_local_orphan`).

    Issue #1971 split "the lane found nothing" into the two answers its
    callers must not conflate: ``no_commits`` is a *proved* empty verdict
    (the probe ran and the branch carries no delta, or the branch ref
    provably does not exist -- callers proceed to reclaim/escalate), while
    ``probe_failed`` is an *inconclusive* verdict (git probing erred on a
    branch that may exist -- callers defer and retry next pass rather than
    discarding possibly-salvageable work). ``parked`` means the dead worker's
    committed work was published this call -- parked ``review_ready`` on a
    no-PR backend, pushed + PR'd (or already landed, so skipped) on a
    PR-capable one; ``park_failed`` means salvageable commits were found but
    the publish write failed and must be retried.
    """

    status: Literal["parked", "park_failed", "no_commits", "probe_failed"]
    error: str | None = None


def publishes_pull_requests(gh: GitHubLike) -> bool:
    """Whether ``gh`` can host a PR. Absent attribute means yes.

    A capability probe rather than an ``isinstance(gh, LocalFileGitHub)``
    check so the real client and every existing test double keep the default
    without declaring anything, and a future no-PR backend opts in with one
    class attribute.
    """
    return bool(getattr(gh, "publishes_pull_requests", True))


def park_unpublishable_work(
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    branch: str,
    issue_number: int,
    active_labels: set[str],
    failure_kind: str | None,
    write_gate: WriteGate,
) -> tuple[bool, str | None] | None:
    """Park a dead session's committed branch for human review.

    Returns ``None`` when ``gh`` publishes pull requests -- the caller carries
    on with push + PR. Otherwise returns ``_attempt_salvage``'s own
    ``(ok, error)`` contract, where ``ok`` means "handled, do not redispatch".

    A failed label write returns ``ok=False``. Unlike the PR path there is no
    durable artifact (an open PR) to make a later pass skip this issue, so
    reporting success with the active label still in place would strand it as
    in-progress forever; falling through to the caller's redispatch cap is the
    loud outcome. Errors come back as values, never raised.
    """
    if publishes_pull_requests(gh):
        return None

    # Issue #2094: a rework worker that died without moving the branch must
    # re-arm (or escalate), never park -- nothing selects a parked issue whose
    # head equals the verdict's reviewed head. The classification rides along
    # so a provider-throttled death re-arms without burning the no-op cap.
    rearmed = rearm_no_op_local_rework(
        gh, config, repo_root, branch, issue_number, write_gate, failure_kind
    )
    if rearmed is not None:
        return rearmed

    result = write_gate.transition(gh, config.labels, issue_number, "local_work_ready")
    label_ok = write_gate.dry_run or result.outcome is TransitionOutcome.APPLIED
    if label_ok and not write_gate.dry_run:
        _post_branch_comment(gh, config, repo_root, branch, issue_number)

    state_file = write_gate.state_path
    with state_lock(state_file):
        state = load_state(state_file)
        if label_ok:
            issue_entry = (state.get("issues") or {}).get(str(issue_number))
            if isinstance(issue_entry, dict) and issue_entry.get("status") == "dispatched":
                # Issue #1923: parking is not only a label change -- the state
                # entry must leave the dispatched lane in the same write. Left
                # at "dispatched", the state/PID orphan sweep keeps the dead
                # worker's entry: it re-derives the no-open-PR drift every
                # pass, and the ``dead_dispatched_reap_minutes`` backstop
                # eventually force-escalates the parked work to the operator
                # queue (the mdls #144 sequence this issue reports).
                # ``open_passive`` is the sweep's own converged placeholder
                # for "work published, no live worker": it is
                # ORCHESTRATOR_OWNED (``issue_status_normalized`` skips it),
                # stays in ACTIVE_STATE_STATUSES (issue-close convergence
                # still applies), and the local review/merge lane adopts on
                # the ``review_ready`` label rather than the status string.
                # The drift markers are popped -- the same fields
                # ``dispatch_state`` pops when a real dispatch claims the
                # issue -- because the drift they armed is resolved here.
                issue_entry = {**issue_entry, "status": PASSIVE_OPEN_STATUS}
                issue_entry.pop("orphan_flagged_at", None)
                issue_entry.pop("orphan_drift_fingerprint", None)
                issue_entry.pop("orphan_drift_at", None)
                clear_local_park_deferral(issue_entry)
                state["issues"][str(issue_number)] = issue_entry
        state = write_gate.append_event(
            state,
            "local_work_ready",  # event-consumer: audit-only -- the actionable signal is the review_ready label applied inline just above (it holds the issue out of dispatch and is what the operator sees); this event is the audit record of which branch was parked and whether the label write landed
            {
                "issue_number": issue_number,
                "branch": branch,
                "failure_kind": failure_kind,
                "removed_labels": sorted(active_labels),
                "label_write_ok": label_ok,
            },
        )
        write_gate.save_state(state)
    if not label_ok:
        return False, f"review-ready label transition failed: {result.outcome.value}"
    return True, None


def _post_branch_comment(
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    branch: str,
    issue_number: int,
) -> None:
    """Tell the reviewer which branch holds the work. Best-effort.

    The label is the state; the comment is a convenience. A failure here is
    logged and does not fail the park. Only the two failure families the
    operation can legitimately produce are caught -- a programming error must
    not hide behind a warning.
    """
    body_dir = runtime_paths(repo_root, config.runtime.state_dir).issues / f"issue-{issue_number}"
    # Issue #1844: when the local review/merge lane is enabled (the default),
    # ``agent:review-ready`` is the lane's *input* -- the next loop pass adopts
    # the branch, reviews it, runs the suite, and merges with no operator
    # action. The comment describes which mode is actually armed so the
    # operator never reads "you must merge this by hand" on a repo whose lane
    # will do it automatically.
    lane_enabled = config.review_dispatch.enabled and config.auto_merge.enabled
    body_text = (
        f"Work for this issue is committed on branch `{branch}`. This repo has no "
        "remote, so nothing was pushed and no PR exists: the local review lane "
        "will pick the branch up on the next pass -- review, full test suite, "
        "and merge into the base branch all run automatically. No action is "
        "needed unless the lane escalates."
        if lane_enabled
        else (
            f"Work for this issue is committed on branch `{branch}`. This repo has no "
            "remote, so nothing was pushed and no PR exists, and the local "
            "review/merge lane is disabled (review_dispatch.enabled and "
            "auto_merge.enabled): review the branch, merge it, and close the "
            "issue."
        )
    )
    try:
        body_dir.mkdir(parents=True, exist_ok=True)
        body_path = body_dir / "review-ready-comment.md"
        _write_text_atomic(body_path, body_text)
        gh.issue_comment(issue_number, body_path)
    except (OSError, GitHubError):
        logger.warning("review-ready comment post failed issue=%d", issue_number, exc_info=True)


def park_salvageable_local_orphan(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    worktrees_dir: Path | None,
    state: dict[str, Any],
    issue_number: int,
    issue: dict[str, Any],
    active_labels: set[str],
    issue_labels: set[str],
    state_file: Path,
    worker_outcome: dict[str, Any] | None,
    write_gate: WriteGate,
) -> LocalParkResult | None:
    """Issue #1923: park a no-PR-backend orphan's committed branch for review.

    On a ``local_issues`` backend the worker's branch IS the deliverable --
    there is no remote to push to and no PR to open. The caller's reclaim
    (strip active labels, re-add ``ready``) is therefore not "requeue the
    issue": it hands the issue back to dispatch, where the next worker
    launch dies on the dead worker's own commits (``worktree_unsafe``) and
    the issue escalates to the operator queue as if the worker had failed.
    This is the state/PID orphan sweep's half of the salvage gate the
    sidecar-based dead-worker lane already runs (``dead_worker_reap.py``:
    ``inspection.ahead_count > 0`` -> ``_attempt_salvage``) -- both loci now
    route through the same ``_attempt_salvage`` -> ``park_unpublishable_work``
    path, so a parked issue applies ``agent:review-ready`` and advances off
    ``dispatched`` in one step regardless of which lane found it.

    Returns ``None`` when this lane does not apply and the caller must
    proceed to its normal handling:

    * a PR-capable backend -- committed-but-unpushed recovery there is
      ``salvage_push_stranded_commits`` + ``_open_pr_for_orphaned_branch``,
      unchanged;
    * no resolvable ``repo_root`` -- the worktree cannot be inspected and
      the park itself needs it.

    Otherwise returns a :class:`LocalParkResult` -- ``parked`` when the issue
    was handled this call (parked, or the salvage skip fired because the
    work already landed), ``park_failed`` when salvageable commits were
    found but the park write failed, ``no_commits`` when the probe *proved*
    nothing salvageable exists, and ``probe_failed`` (issue #1971) when the
    probe was inconclusive. The two empty-verdict statuses differ in what
    the caller may do next: ``no_commits`` permits reclaim/escalation,
    ``probe_failed`` must defer it -- a transient git failure can never be
    the reason committed work is discarded.

    The worktree-missing case still probes the branch ref directly: on a
    no-remote repo the branch is the deliverable, so a reclaimed worktree
    must not strand committed work that still exists in the main checkout.
    The fallback measures a content delta (``branch_diff_result`` against the
    local base), not a bare commit count.

    The probe+publish tail lives in :func:`salvage_dead_worker_commits` — the
    backend-agnostic "dead worker with commits" seam every dead-worker locus
    shares (issue #2262); this function adds the no-PR-backend gate on top.
    """
    if publishes_pull_requests(gh) or repo_root is None:
        return None
    return salvage_dead_worker_commits(
        gh=gh,
        config=config,
        repo_root=repo_root,
        worktrees_dir=worktrees_dir,
        state=state,
        issue_number=issue_number,
        issue=issue,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=state_file,
        worker_outcome=worker_outcome,
        write_gate=write_gate,
    )


def salvage_dead_worker_commits(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    worktrees_dir: Path | None,
    state: dict[str, Any],
    issue_number: int,
    issue: dict[str, Any],
    active_labels: set[str],
    issue_labels: set[str],
    state_file: Path,
    worker_outcome: dict[str, Any] | None,
    write_gate: WriteGate,
) -> LocalParkResult | None:
    """Issue #2262: the single "dead worker with commits" probe+publish seam.

    Every dead-worker locus that would otherwise requeue the issue routes
    committed work through here. The probe is backend-agnostic: inspect the
    worker's worktree for commits ahead of base (falling back to a branch-ref
    content diff when the worktree is gone), then hand committed work to
    ``_attempt_salvage`` — which parks it review-ready via
    ``park_unpublishable_work`` on a no-PR backend, or pushes the branch and
    opens a PR on a PR-capable one. The capability decision therefore lives
    in exactly one place; a locus that forgets it cannot reintroduce the
    "requeue committed local work into a worktree reset" gap.

    This function takes ``state_lock`` itself (inside ``_attempt_salvage`` /
    ``park_unpublishable_work``) and does git + label I/O, so it must run
    OUTSIDE ``state_lock`` — call it in a pre-lock phase and carry the
    verdict in, the same pre-lock/two-phase pattern the dead-worker sweep
    uses (``decide_pre`` / ``apply_requests_pre``).

    Returns ``None`` only when no probe is possible (``repo_root``
    unresolvable) — the caller proceeds to its normal handling. Otherwise a
    :class:`LocalParkResult`: ``parked`` (published — never requeue),
    ``park_failed`` / ``probe_failed`` (defer and retry; a transient failure
    must never be the reason committed work is discarded), ``no_commits``
    (proved empty — the caller's normal reclaim/requeue applies).
    """
    # Deferred: workflow.py imports this module for the sweep call site, so
    # a top-level ``import charlie_work.workflow`` would cycle. Attribute
    # access through the module object also keeps suite patches on
    # ``charlie_work.workflow._attempt_salvage`` / ``.slugify`` live.
    import charlie_work.workflow as _wf

    if repo_root is None:
        return None
    entry = (state.get("issues") or {}).get(str(issue_number))
    branch = entry.get("branch_name") if isinstance(entry, dict) else None
    if not branch:
        branch = (
            f"{config.dispatch.branch_prefix}-{issue_number}-"
            f"{_wf.slugify(str(issue.get('title') or 'work'))}"
        )
    worktree_path = (
        worktree_path_for_branch(repo_root, branch, worktrees_dir)
        if worktrees_dir is not None
        else None
    )

    ahead_count = 0
    resolved_base_ref: str | None = None
    if worktree_path is not None and worktree_path.is_dir():
        inspection = inspect_worktree_state(
            worktree_path,
            config.dispatch.base_ref,
            config.dispatch.injected_paths,
            config.dispatch.materialize_dirs,
        )
        ahead_count = inspection.ahead_count
        resolved_base_ref = inspection.resolved_base_ref
    if ahead_count <= 0:
        # Branch-ref fallback: the worktree is gone (reclaimed) or
        # uninspectable while the branch still carries the worker's
        # commits. A non-empty diff against the local base means real work
        # is salvageable. ``branch_diff_result`` returns no diff for ANY git
        # error, though -- including a missing branch ref (a determinate "no
        # work" answer) and transient failures alike -- so on failure
        # disambiguate with the ref probe: a provably absent ref is
        # ``no_commits``, anything else is ``probe_failed`` (issue #1971: an
        # inconclusive probe must defer the reap, never read as no-work),
        # carrying git's own stderr so the deferral/escalation reason names
        # the actual failure (missing base ref, no merge base, ...).
        base_branch = resolved_base_ref or local_base_branch(repo_root) or "HEAD"
        diff, diff_error = branch_diff_result(repo_root, base_branch, branch)
        if diff is None:
            ref_exists, ref_error = probe_branch_ref(repo_root, branch)
            if ref_exists is False:
                return LocalParkResult("no_commits")
            ref_note = "exists" if ref_exists else f"unverifiable ({ref_error})"
            return LocalParkResult("probe_failed", f"{diff_error} (branch ref {ref_note})")
        if diff:
            ahead_count = 1
            resolved_base_ref = base_branch
    if ahead_count <= 0:
        return LocalParkResult("no_commits")

    salvaged, salvage_error = _wf._attempt_salvage(
        gh=gh,
        config=config,
        repo_root=repo_root,
        worktree_path=worktree_path if worktree_path is not None else repo_root,
        branch=branch,
        base_ref=resolved_base_ref or config.dispatch.base_ref,
        issue_number=issue_number,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=state_file,
        failure_kind=(persisted_failure(entry).kind if isinstance(entry, dict) else None),
        issue_title=issue.get("title"),
        issue=issue,
        worker_outcome=worker_outcome,
        write_gate=write_gate,
    )
    if salvaged:
        return LocalParkResult("parked")
    return LocalParkResult("park_failed", salvage_error)


def _defer_probe_failure(
    write_gate: WriteGate, issue_number: int, *, reason: str, now: datetime
) -> bool:
    """Count one inconclusive-probe pass on the state entry; True while deferring.

    Runs pre-lock in the reclaim lane, so it takes ``state_lock`` itself and
    persists the counter (and the fingerprinted ``orphaned_worker_drift``
    audit event) for the next pass. Delegates to the same
    :func:`defer_or_expire_local_park` the in-lock backstop uses, so the two
    lanes share ONE budget (``LOCAL_PARK_DEFER_MAX_PASSES``, keyed on the
    sweep's ``now`` so a pass is never counted twice) and one event per
    distinct reason.
    """
    state_file = write_gate.state_path
    events: list[tuple[str, dict[str, Any]]] = []
    with state_lock(state_file):
        state = load_state(state_file)
        entry = (state.get("issues") or {}).get(str(issue_number))
        if not isinstance(entry, dict):
            return True
        entry = dict(entry)
        deferred = defer_or_expire_local_park(
            entry,
            issue_number=issue_number,
            reason=reason,
            now=now,
            orphan_drift_at=entry.get("orphan_drift_at"),
            sweep_events=events,
        )
        state["issues"][str(issue_number)] = entry
        for _kind, payload in events:  # defer_or_expire_local_park emits only drift events
            state = write_gate.append_event(state, "orphaned_worker_drift", payload)
        write_gate.save_state(state)
    return deferred


def _reset_probe_deferral(write_gate: WriteGate, state: dict[str, Any], issue_number: int) -> None:
    """Drop stale deferral bookkeeping once a pass gets a conclusive verdict.

    ``state`` is the sweep's pre-lock snapshot and only gates the lock: the
    common case (no counter on the entry) costs no lock or write.
    """
    snapshot = (state.get("issues") or {}).get(str(issue_number))
    if not isinstance(snapshot, dict) or not any(f in snapshot for f in LOCAL_PARK_DEFER_FIELDS):
        return
    state_file = write_gate.state_path
    with state_lock(state_file):
        fresh = load_state(state_file)
        entry = (fresh.get("issues") or {}).get(str(issue_number))
        if isinstance(entry, dict):
            entry = dict(entry)
            clear_local_park_deferral(entry)
            fresh["issues"][str(issue_number)] = entry
            write_gate.save_state(fresh)


def park_or_reclaim_local_orphan(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    worktrees_dir: Path | None,
    state: dict[str, Any],
    issue_number: int,
    issue: dict[str, Any],
    active_labels: set[str],
    issue_labels: set[str],
    state_file: Path,
    worker_outcome: dict[str, Any] | None,
    write_gate: WriteGate,
    reclaim_results: dict[int, dict[str, Any]],
    park_verdicts: dict[int, LocalParkResult],
    now: datetime,
) -> bool:
    """Issue #1923: the no-open-PR sweep tail -- park salvageable work, else reclaim.

    Runs :func:`park_salvageable_local_orphan` (whose docstring carries the
    full gate contract), records its verdict in ``park_verdicts`` -- the
    ``dead_dispatched_reap_minutes`` backstop consults the same verdicts to
    decide whether a timed escalation may proceed (issue #1971) -- and
    returns True when the caller must ``continue`` without reclaiming.

    ``parked`` continues because the work is safely held on the branch.
    ``probe_failed`` continues WHILE the bounded deferral budget remains
    (:func:`_defer_probe_failure`): the probe could not disprove committed
    work, and reclaiming (strip active, re-add ``ready``, redispatch) on a
    transient git failure is the discard path issue #1971 removes; the
    active label stays in place, an ``orphaned_worker_drift`` event names
    the probe error, and the probe retries next pass. Once
    ``LOCAL_PARK_DEFER_MAX_PASSES`` consecutive passes fail, the failure is
    deterministic (a missing base ref, no merge base) and would wedge the
    entry forever, so the reclaim runs and carries the probe error.

    On any other outcome -- a PR-capable backend, a proven ``no_commits``
    verdict, a failed park, or a spent probe budget -- the sweep's normal
    reclaim runs here instead: strip the active labels, re-add ``ready``,
    record the outcome in ``reclaim_results``. A failed park or spent probe
    budget additionally stamps ``salvage_failed``/``salvage_error`` onto the
    recorded reclaim so the ``session_failed_relabeled`` event carries *why*
    an issue with possibly-committed work fell through -- the loud outcome,
    matching the sibling lane's salvage-failure contract.

    Lives here rather than inline in ``_detect_and_handle_orphaned_workers``
    because ``workflow.py`` sits over its file-size ratchet mark -- the same
    reason ``orphaned_worker_sweep.py`` exists (#1911).
    """
    park_result = park_salvageable_local_orphan(
        gh=gh,
        config=config,
        repo_root=repo_root,
        worktrees_dir=worktrees_dir,
        state=state,
        issue_number=issue_number,
        issue=issue,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=state_file,
        worker_outcome=worker_outcome,
        write_gate=write_gate,
    )
    if park_result is not None:
        park_verdicts[issue_number] = park_result
    if park_result is not None and park_result.status == "parked":
        return True
    probe_error: str | None = None
    if park_result is not None and park_result.status == "probe_failed":
        probe_error = park_result.error or park_result.status
        if _defer_probe_failure(write_gate, issue_number, reason=probe_error, now=now):
            return True
    else:
        _reset_probe_deferral(write_gate, state, issue_number)

    needs_ready = config.labels.ready not in issue_labels
    # Issue #2226: route through the WriteGate's canonical seam so the
    # return to ``ready`` lands in events.db bound to this repo; the
    # conditional add preserves the no-redundant-write contract.
    label_result = write_gate.apply_issue_labels(
        gh,
        config.labels,
        issue_number,
        add=(config.labels.ready,) if needs_ready else (),
        remove=sorted(active_labels),
        to_state="ready",
        cause="session_failed_relabeled",
    )
    label_write_ok = label_result.ok
    reclaim_results[issue_number] = {
        "removed_labels": sorted(active_labels),
        "added_ready": needs_ready,
        "label_write_ok": label_write_ok,
    }
    if park_result is not None and park_result.status == "park_failed":
        # Issue #1923: carry the park failure onto the relabel event so the
        # event stream shows this issue had salvageable commits that could
        # not be parked.
        reclaim_results[issue_number]["salvage_failed"] = True
        reclaim_results[issue_number]["salvage_error"] = park_result.error
    elif probe_error is not None:
        # Issue #1971: the deferral budget is spent -- the reclaim proceeds
        # (a deterministic probe failure must not wedge the entry) and the
        # event carries git's own error, never a bare status.
        reclaim_results[issue_number]["salvage_failed"] = True
        reclaim_results[issue_number]["salvage_error"] = (
            f"branch probe failed {LOCAL_PARK_DEFER_MAX_PASSES} consecutive passes: {probe_error}"
        )
        # The verdict is spent: recording it would make the in-lock backstop
        # re-defer an issue this pass already reclaimed.
        park_verdicts.pop(issue_number, None)
        _reset_probe_deferral(write_gate, state, issue_number)
    return False


def park_backstop_due_local_orphans(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Any,
    worktrees_dir: Path | None,
    state: dict[str, Any],
    state_file: Path,
    no_pr_orphans: Iterable[int],
    issues_by_number: Mapping[int, dict[str, Any]],
    worker_outcomes: Mapping[int, dict[str, Any] | None],
    reclaim_results: Mapping[int, dict[str, Any]],
    escalations: Sequence[Collection[int]],
    park_verdicts: Mapping[int, "LocalParkResult"],
    dead_dispatched_reap_minutes: float,
    now: datetime,
    write_gate: WriteGate,
) -> dict[int, str]:
    """Issue #1971: park-probe the no-PR orphans the timed backstop is due to reap.

    ``orphaned_worker_sweep.maybe_reap_dead_dispatched_worker`` is a pure
    timer running inside the sweep's ``state_lock``: a dead issue whose
    ``orphan_drift_at`` expired escalates without checking whether the
    worker's branch carries commits. The reclaim loop
    (:func:`park_or_reclaim_local_orphan`) already park-probes every orphan
    that still carries an ACTIVE label, but it skips labelless and
    terminal-only issues -- exactly the stale ``dispatched`` entries the
    backstop escalates. On a no-PR backend the branch IS the deliverable,
    so every backstop-due orphan the earlier lanes did not handle gets the
    same :func:`park_salvageable_local_orphan` attempt here, BEFORE the
    lock: the park takes ``state_lock`` and does git/label I/O, so it can
    never run inside the locked classification.

    Skipped (the backstop keeps its existing semantics for each):

    * issues already handled this pass -- a recorded park verdict
      (``park_verdicts``), a completed reclaim (``reclaim_results``), or a
      same-pass escalation (``escalations``: worker-blocked, zero-artifact,
      or cross-repo -- each already stripped the active labels and applied
      ``human_needed``, which a park's ``local_work_ready`` edge would
      otherwise silently strip);
    * issues carrying a ``config.labels.terminal`` label -- ``done``,
      ``human_needed``, ``operator_queue``, ``review_ready`` are answers
      another lane already gave, and parking would strip them;
    * entries the timer predicate does not mark due -- checked first on the
      sweep's pre-lock snapshot (cheap filter) and re-verified on a fresh
      locked read immediately before the probe, since the in-lock backstop
      re-checks on fresh state too.

    Returns ``{issue_number: reason}`` for every issue whose park attempt
    failed (``park_failed``) or whose branch probe was inconclusive
    (``probe_failed``) -- the in-lock backstop consults the map and defers
    the escalation one pass rather than discarding possibly-salvageable
    work. A ``parked`` verdict flips the state entry off ``dispatched`` so
    the backstop's status re-verify skips it; a ``no_commits`` verdict (or
    a defensive ``None``) falls through and the backstop escalates exactly
    as it does today.

    No-ops (returns empty) on a PR-capable backend or a missing
    ``repo_root`` -- committed-but-unpushed recovery there is
    ``salvage_push_stranded_commits`` + ``_open_pr_for_orphaned_branch``,
    unchanged.
    """
    if not isinstance(repo_root, Path) or publishes_pull_requests(gh):
        return {}
    escalated = {number for group in escalations for number in group}
    deferred: dict[int, str] = {}
    for issue_number in no_pr_orphans:
        prior_verdict = park_verdicts.get(issue_number)
        if prior_verdict is not None:
            if prior_verdict.status in ("park_failed", "probe_failed"):
                deferred[issue_number] = prior_verdict.error or prior_verdict.status
            continue
        if issue_number in reclaim_results or issue_number in escalated:
            continue
        issue = issues_by_number.get(issue_number)
        if issue is None:
            continue
        issue_labels = label_names(issue)
        if issue_labels & config.labels.terminal:
            continue
        entry = (state.get("issues") or {}).get(str(issue_number))
        if not isinstance(entry, dict) or not dead_dispatched_reap_due(
            state=state,
            entry=entry,
            dead_dispatched_reap_minutes=dead_dispatched_reap_minutes,
            now=now,
        ):
            continue
        # Stale-snapshot edge: ``state`` is the sweep's pre-lock snapshot,
        # but the in-lock backstop re-checks due-ness on a fresh load, and
        # the earlier pre-lock lanes (reclaim, park, live-handoff finalize)
        # may have rewritten the entry under their own ``state_lock``s.
        # Re-verify on the current entry before the probe/park I/O so the
        # drain never fires on fields a cheaper lane already resolved this
        # pass -- the same fresh read ``park_labelless_dead_local_session``
        # takes for its status check.
        fresh = load_state_locked(state_file)
        fresh_entry = (fresh.get("issues") or {}).get(str(issue_number))
        if not isinstance(fresh_entry, dict) or not dead_dispatched_reap_due(
            state=fresh,
            entry=fresh_entry,
            dead_dispatched_reap_minutes=dead_dispatched_reap_minutes,
            now=now,
        ):
            continue
        park_result = park_salvageable_local_orphan(
            gh=gh,
            config=config,
            repo_root=repo_root,
            worktrees_dir=worktrees_dir,
            state=fresh,
            issue_number=issue_number,
            issue=issue,
            active_labels=issue_labels & config.labels.active,
            issue_labels=issue_labels,
            state_file=state_file,
            worker_outcome=worker_outcomes.get(issue_number),
            write_gate=write_gate,
        )
        if park_result is not None and park_result.status in (
            "park_failed",
            "probe_failed",
        ):
            deferred[issue_number] = park_result.error or park_result.status
    return deferred


def park_labelless_dead_local_session(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Any,
    issue_labels: set[str],
    worker: WorkerView,
    write_gate: WriteGate,
) -> bool:
    """Issue #1971: park a dead session whose active label is already gone.

    The dead-session reap's ``if not active_labels: continue`` gate is right
    on a PR-capable backend -- an issue carrying only terminal labels has
    nothing to reclaim -- but on a no-PR backend it discards real work: a
    session whose active label is already gone while its state entry still
    reads ``dispatched`` and its branch still carries commits is picked up
    later by the ``dead_dispatched_reap_minutes`` backstop and escalated,
    the branch never offered for review. When the state entry is still
    ``dispatched`` this runs the same :func:`park_salvageable_local_orphan`
    probe the state-keyed sweep uses: park salvageable work, leave a proved
    no-commits verdict alone, and let an inconclusive probe defer to a
    later pass.

    ``worker.branch`` -- the session sidecar's recorded branch -- backfills a
    missing ``branch_name`` on the state entry (the state-keyed sweep can
    only synthesize ``{prefix}-{n}-{slug}``, which misses pre-slug or
    renamed branches). ``worker.worktree_path`` gates the
    ``.worker-outcome.json`` read on being a real directory so a
    missing/empty path never probes the caller's cwd (cw#1771/#660).
    Returns True when the issue was parked this call.
    """
    if not isinstance(repo_root, Path) or publishes_pull_requests(gh):
        return False
    if issue_labels & config.labels.terminal:
        # A terminal label (done / human-needed / operator-queue /
        # review-ready) is already an answer some lane gave; the park's
        # ``local_work_ready`` edge would strip it.
        return False
    issue_number = worker.issue_number
    state_file = write_gate.state_path
    entry = (load_state_locked(state_file).get("issues") or {}).get(str(issue_number))
    if not isinstance(entry, dict) or entry.get("status") != "dispatched":
        return False
    if worker.branch and not entry.get("branch_name"):
        entry = {**entry, "branch_name": worker.branch}
    # Path("")/Path(".") name themselves "" -- an empty sidecar
    # worktree_path must not resolve the outcome probe to cwd.
    worktree_path = Path(worker.worktree_path or "")
    result = park_salvageable_local_orphan(
        gh=gh,
        config=config,
        repo_root=repo_root,
        worktrees_dir=resolved_layout(config, repo_root).worktrees,
        state={"issues": {str(issue_number): entry}},
        issue_number=issue_number,
        issue={},
        active_labels=set(),
        issue_labels=issue_labels,
        state_file=state_file,
        worker_outcome=(
            read_worker_outcome(worktree_path)
            if worktree_path.name and worktree_path.is_dir()
            else None
        ),
        write_gate=write_gate,
    )
    return result is not None and result.status == "parked"
