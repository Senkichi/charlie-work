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
from pathlib import Path
from typing import Any

from .config import OrchestratorConfig
from .github import GitHubError, GitHubLike
from .labels import TransitionOutcome
from .local_lane import branch_diff, local_base_branch
from .paths import runtime_paths
from .rework_prompts import _write_text_atomic
from .state import PASSIVE_OPEN_STATUS, load_state, state_lock
from .worktree import inspect_worktree_state, worktree_path_for_branch
from .write_gate import WriteGate

logger = logging.getLogger(__name__)


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
) -> tuple[bool, str | None] | None:
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
    proceed to its normal reclaim:

    * a PR-capable backend -- committed-but-unpushed recovery there is
      ``salvage_push_stranded_commits`` + ``_open_pr_for_orphaned_branch``,
      unchanged;
    * no resolvable ``repo_root`` -- the worktree cannot be inspected and
      the park itself needs it;
    * no provably salvageable commits -- ``inspect_worktree_state`` reports
      ahead_count 0, including UNKNOWN (missing worktree dir or a failed
      git probe): uncertainty is never proof of work, and the redispatch
      dying ``worktree_unsafe`` on real commits is the loud outcome.

    The worktree-missing case still probes the branch ref directly: on a
    no-remote repo the branch is the deliverable, so a reclaimed worktree
    must not strand committed work that still exists in the main checkout.
    The fallback measures a content delta (``branch_diff`` against the
    local base), not a bare commit count.

    Returns ``(True, None)`` when the issue was handled -- parked, or the
    salvage skip fired because the work already landed -- and the caller
    must ``continue`` without reclaiming. Returns ``(False, error)`` when
    the park itself failed; the caller falls through to the normal reclaim
    so the failure stays loud (redispatch cap / dead-dispatched backstop),
    matching the sibling lane's salvage-failure contract.
    """
    # Deferred: workflow.py imports this module for the sweep call site, so
    # a top-level ``import charlie_work.workflow`` would cycle. Attribute
    # access through the module object also keeps suite patches on
    # ``charlie_work.workflow._attempt_salvage`` / ``.slugify`` live.
    import charlie_work.workflow as _wf

    if publishes_pull_requests(gh) or repo_root is None:
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
        # is salvageable; a None diff (missing ref / probe failure) reads
        # as no work and falls through to the caller's reclaim.
        base_branch = resolved_base_ref or local_base_branch(repo_root) or "HEAD"
        if branch_diff(repo_root, base_branch, branch):
            ahead_count = 1
            resolved_base_ref = base_branch
    if ahead_count <= 0:
        return None

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
        failure_kind=(entry.get("dead_worker_failure_kind") if isinstance(entry, dict) else None),
        issue_title=issue.get("title"),
        issue=issue,
        worker_outcome=worker_outcome,
        write_gate=write_gate,
    )
    return salvaged, salvage_error


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
) -> bool:
    """Issue #1923: the no-open-PR sweep tail -- park salvageable work, else reclaim.

    Runs :func:`park_salvageable_local_orphan` (whose docstring carries the
    full gate contract). Returns ``True`` when the issue was parked -- the
    caller ``continue``s without reclaiming. On any other outcome -- a
    PR-capable backend, no salvageable commits, or a failed park -- the
    sweep's normal reclaim runs here instead: strip the active labels,
    re-add ``ready``, record the outcome in ``reclaim_results``. A failed
    park additionally stamps ``salvage_failed``/``salvage_error`` onto the
    recorded reclaim so the ``session_failed_relabeled`` event carries *why*
    an issue with committed work fell through -- the loud outcome, matching
    the sibling lane's salvage-failure contract.

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
    if park_result is not None and park_result[0]:
        return True

    needs_ready = config.labels.ready not in issue_labels
    label_write_ok = True
    for label in sorted(active_labels):
        if not gh.remove_issue_label(issue_number, label):
            label_write_ok = False
    if needs_ready:
        if not gh.add_issue_label(issue_number, config.labels.ready):
            label_write_ok = False
    reclaim_results[issue_number] = {
        "removed_labels": sorted(active_labels),
        "added_ready": needs_ready,
        "label_write_ok": label_write_ok,
    }
    if park_result is not None:
        # Issue #1923: carry the park failure onto the relabel event so the
        # event stream shows this issue had salvageable commits that could
        # not be parked.
        reclaim_results[issue_number]["salvage_failed"] = True
        reclaim_results[issue_number]["salvage_error"] = park_result[1]
    return False
