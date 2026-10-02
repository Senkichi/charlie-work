"""Salvage and PR-opening effects for the dead-worker sweep.

Moved verbatim from the retired ``dead_worker_reap`` module: opening a PR for a
salvaged worker branch (issues #252, #956, #1130), the already-landed check, and the
orphaned-branch PR fallback.
"""

from __future__ import annotations


from pathlib import Path
from typing import Any

from ..closing_reference import (
    ValidationResult,
    probe_closing_link,
    validate_closing_reference,
)
from ..config import (
    OrchestratorConfig,
)
from ..github import (
    GitHubLike,
    PR_CLOSING_ISSUES_FIELDS,
)
from ..instrumentation import log_event
from ..labels import TransitionOutcome, apply_issue_labels
from ..local_work_park import park_unpublishable_work
from ..pr_create_retry import create_pr_with_retry
from ..state import (
    load_state,
    state_lock,
)
from ..worktree import (
    push_branch,
    resolve_base_branch_name,
    summarize_branch_work,
)
from ..salvage_superseded import check_salvage_superseded, salvage_skip_event_kind
from ..write_gate import WriteGate, require_write_gate


def _safe_repo_slug(gh: GitHubLike) -> str:
    """Return the ``owner/repo`` slug, or ``"?"`` if the lookup fails.

    ``name_with_owner()`` raises ``GitHubError`` on failure (offline, gh
    missing, etc.); this is used only to qualify a closing-reference line, so
    a lookup failure must not stop salvage-PR creation. Mirrors
    ``reconcile._repo_slug``.
    """
    try:
        return gh.name_with_owner()
    except Exception:
        return "?"


def _dispatching_repo_name(gh: GitHubLike, repo_root: Path) -> str:
    """Return the repo-name segment of the dispatching repo (issue #1244).

    Prefers ``gh.name_with_owner()`` (``owner/repo``) so the name matches
    the fleet registry's keys.  Falls back to ``repo_root.name`` (the
    directory name) when the GitHub lookup fails (offline, gh missing) —
    the directory name is usually the same as the GitHub repo name, and a
    mismatch only means the scope gate cannot attribute the issue, which
    is the safe direction (pass, not block) *for that gate*.

    That "safe direction" reasoning does not carry over unchanged to this
    return value's second, opposite-polarity consumer:
    ``cross_repo_gate.py``'s :func:`~charlie_work.cross_repo_gate._find_owning_repo`
    (added by the #1756-#1758 positive-evidence redesign) uses this same
    name to *exclude* the dispatching repo's own registered entry from the
    sibling-repo search. There, a mismatch (this fallback returning a
    deployment directory name like ``charlie-work-daemon`` that does not
    match the registry's ``charlie-work`` key) would let the dispatching
    repo's own entry be searched as a "sibling" and escalate the repo
    against itself — a *block*, not a pass. ``_find_owning_repo`` guards
    against exactly this by also excluding a managed-roots entry whose
    resolved root is (or contains) the dispatching repo's actual
    ``repo_root``, independent of whatever name this function returns
    (review finding 3) — so a caller adding a third name-keyed consumer of
    this value should not assume a mismatch is automatically safe there
    too; check which direction that consumer's decision points first.
    """
    try:
        nwo = gh.name_with_owner()
        parts = nwo.rsplit("/", 1)
        return parts[1] if len(parts) == 2 else nwo
    except Exception:
        return repo_root.name


def _open_salvage_pr(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    issue_title: str | None = None,
    source_description: str = "worker branch",
    state_file: Path | None = None,
    worker_outcome: dict[str, Any] | None = None,
) -> tuple[int | None, str | None, ValidationResult | None]:
    """Open a PR for a salvaged worker branch and move issue labels toward ``pr_open``.

    Returns ``(pr_number, error, closing_ref)``. ``pr_number`` is the created
    PR number, or ``None`` when the PR could not be created. ``error`` is
    ``None`` when both the PR and the label swap succeeded; otherwise it
    describes the first failure encountered (a missing ``repo_root``, a
    failed PR create, or a label write failure after the PR was created).
    ``closing_ref`` is the `~charlie_work.closing_reference.ValidationResult`
    from canonicalizing the closing-reference line before the PR was
    created, or ``None`` when PR creation never reached that far (missing
    ``repo_root``).

    cw#1263: the body's ``Closes #N`` line is validated/canonicalized via
    `closing_reference.validate_closing_reference` before ``gh.pr_create``
    ever sees it -- this is the sole point where both salvage/orphan-recovery
    callers (`_attempt_salvage`, `_open_pr_for_orphaned_branch`) create a PR,
    so routing the fixed-up body through here covers both without a second
    call site to keep in sync. After a successful create, GitHub's own
    ``closingIssuesReferences`` resolution is queried and compared against
    ``issue_number``; a mismatch is logged (``pr_closing_ref_unlinked``) but
    never blocks the return -- this is the only verification surface that
    would catch it, since GitHub's own auto-close resolution can diverge from
    the text charlie-work wrote even when that text looks correct.

    cw#1771: ``worker_outcome`` is the dict `~charlie_work.worktree.read_worker_outcome`
    returned for this branch, when the caller has one. When it carries
    non-empty ``pr_title``/``pr_body`` strings, those are used verbatim (the
    worker's own drafted PR content, per the DEFAULT ``push_pr_outcome.md``
    contract -- workers no longer attempt ``gh pr create`` themselves) --
    still routed through ``validate_closing_reference`` below so a missing or
    wrong closing line is corrected the same as a synthesized body. Title/body
    synthesis (the ``"Salvaged work for #N"`` boilerplate below) is the
    fallback for the genuine crash case: a worker that died before writing an
    outcome file at all, so there is nothing drafted to prefer.
    """
    if repo_root is None:
        return None, "repo_root is required to open a salvage PR", None

    base_branch = resolve_base_branch_name(repo_root, base_ref)

    drafted_title: str | None = None
    drafted_body: str | None = None
    if isinstance(worker_outcome, dict):
        candidate_title = worker_outcome.get("pr_title")
        if isinstance(candidate_title, str) and candidate_title.strip():
            drafted_title = candidate_title.strip()
        candidate_body = worker_outcome.get("pr_body")
        if isinstance(candidate_body, str) and candidate_body.strip():
            # Non-empty is checked via .strip(), but the value used is the
            # unstripped original (unlike drafted_title above) -- a PR body
            # is markdown, so worker-authored leading/trailing structure
            # (blank lines around a heading, a trailing signature block) is
            # preserved rather than collapsed. GitHub renders incidental
            # leading/trailing whitespace as a cosmetic no-op either way.
            drafted_body = candidate_body

    if drafted_title is not None:
        title = drafted_title
    else:
        title = (
            f"Salvaged work for #{issue_number}: {issue_title}"
            if issue_title
            else f"Salvaged work for issue #{issue_number}"
        )

    if drafted_body is not None:
        body = drafted_body
    else:
        # The body must satisfy the same janitor gate as a worker-authored one
        # (`review.require_tests_or_rationale`). A fixed boilerplate string cannot:
        # it carries no rationale token, so every salvage PR failed a gate on text
        # the orchestrator itself wrote. Derive the rationale from the worker's own
        # commit log instead of injecting the gate's keywords -- a branch with no
        # commits still yields no summary, and still correctly fails.
        body = (
            f"Closes #{issue_number}\n\nSalvaged by the orchestrator from a {source_description}."
        )
        # Pass the RESOLVED base branch, not the raw ``base_ref``. The orphaned-branch
        # lane (``_open_pr_for_orphaned_branch``) sources ``base_ref`` straight from
        # ``config.dispatch.base_ref``, whose default is ``""`` and which the live
        # config leaves unset -- so production reaches here with the empty sentinel.
        # ``require_valid_rev("")`` raises, ``summarize_branch_work`` returns "", and
        # the body falls back to boilerplate that cannot pass the janitor gate: the
        # exact defect this code exists to fix, on the lane that hits it most.
        summary = summarize_branch_work(
            repo_root,
            branch,
            base_branch,
            test_path_globs=config.test_adequacy.test_path_globs,
        )
        if summary:
            body = f"{body}\n\n{summary}"

    closing_ref = validate_closing_reference(body, issue_number, repo=_safe_repo_slug(gh), gh=gh)
    body = closing_ref.body
    if closing_ref.changed and state_file is not None:
        log_event(
            state_file,
            "pr_closing_ref_rewritten",
            {
                "issue_number": issue_number,
                "findings": list(closing_ref.findings),
                "source": source_description,
            },
        )

    # cw#1273: every gh.pr_create call site routes through the bounded outer
    # retry + duplicate-PR guard instead of calling gh.pr_create directly.
    retry_result = create_pr_with_retry(
        gh,
        head=branch,
        base=base_branch,
        title=title,
        body=body,
        max_retries=config.runtime.pr_create_retry_max_attempts,
        base_seconds=config.runtime.pr_create_retry_base_seconds,
    )
    pr_number = retry_result.pr_number
    if pr_number is None:
        return (
            None,
            retry_result.error or "gh pr create failed or returned no PR number",
            closing_ref,
        )

    # `pr_number` is falsy (0) under `dry_run`, where no real PR was opened and
    # a `gh pr view 0` call would be both wasted and nonsensical -- only probe
    # a real, truthy PR number.
    if pr_number and state_file is not None:
        # cw#1868: probe_closing_link re-probes across GitHub's asynchronous
        # closing-keyword indexing, so a just-created PR is not logged as
        # unlinked on a read that precedes the index. It returns None when the
        # query itself failed: a transient `gh` failure collapses to the same
        # empty result as a real miss, and this event exists to be acted on.
        linked_numbers = probe_closing_link(
            gh, pr_number, issue_number, fields=PR_CLOSING_ISSUES_FIELDS
        )
        if linked_numbers is not None and issue_number not in linked_numbers:
            log_event(
                state_file,
                "pr_closing_ref_unlinked",
                {
                    "issue_number": issue_number,
                    "pr_number": pr_number,
                    "linked_issue_numbers": sorted(linked_numbers),
                },
            )

    # Issue #2226: the label write routes through the canonical seam so the
    # PR-open transition lands in events.db as ``lifecycle_transition``; the
    # conditional add preserves the no-redundant-write contract.
    # ``pr_number`` is falsy (0) under ``dry_run`` — the same discriminator the
    # probe above uses — so ``state_path=None`` then, keeping dry-run free of
    # both the label write's intent and the lifecycle event.
    # write-gate-exempt(issue=2226): ~15 callers and no write_gate param; dry-run suppression via falsy pr_number sentinel.
    label_result = apply_issue_labels(
        gh,
        config.labels,
        issue_number,
        add=(config.labels.pr_open,) if config.labels.pr_open not in issue_labels else (),
        remove=sorted(active_labels),
        to_state="pr_open",
        state_path=state_file if pr_number else None,
        pr_number=pr_number or None,
        cause="pr_salvage",
    )
    if label_result.outcome is TransitionOutcome.PARTIAL_FAILURE:
        return pr_number, "PR created but label write failed", closing_ref

    return pr_number, None, closing_ref


def _salvage_already_landed(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    branch: str,
    base_ref: str,
    issue_number: int,
    issue: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    """Return ``(already_landed, reason)`` if salvage should be skipped.

    Issue #1221 / #1241: a dead session's snapshot (issue, pr_number, branch)
    can be stale by the time staleness trips and salvage fires -- the linked
    issue may have been closed and/or its PR merged by an operator or sibling
    worker inside the staleness threshold window, and (the #1241 race) the
    branch's commits may already be reachable from origin/main via a merge
    commit whose tree differs from the salvage head's tree. Re-check LIVE
    terminal state at fire time instead of trusting the snapshot.

    This is now a thin delegate to the shared single enforcement point
    ``salvage_superseded.check_salvage_superseded`` so the workflow salvage
    lane and the reconcile salvage lane cannot diverge on which checks fire
    or in which order. The shared check covers:

    1. the linked issue is CLOSED (``issue`` carries ``state`` from the
       caller's ``gh.issue_view`` -- one call, already made; if ``issue`` is
       None the shared check fetches it via ``gh.issue_view`` so the
       closed-issue check still fires on the reconcile lane, which had not
       fetched).
    2. a PR binding to this issue is MERGED (``gh.merged_prs_for_issue`` -- one
       call). A failed search (``ok=False``) is treated as "unknown", which
       falls through to opening the PR; a human reviews salvage PRs anyway.
    3. the salvage branch's tree contributes an empty diff against current main
       (``salvage_branch_empty_diff`` -- a fetch + two rev-parse calls). This
       is the belt-and-suspenders for the case where (1)/(2) miss (e.g. a
       squash-merge that closed the issue but whose PR search lags, or work
       landed via a sibling branch). Fails safe (returns False) on git error.
    4. (#1241) the salvage branch's tip is an ANCESTOR of origin/main
       (``salvage_branch_reachable_from_main`` -- a fetch + ``git merge-base
       --is-ancestor``). This catches the #1241 race that (3) misses: a merge
       commit incorporated the salvage head while main advanced with other
       commits, so the trees differ (empty-diff reads "not empty") but the
       salvage head carries nothing new (ancestry reads "already on main").
       Fails open on git error.

    ``reason`` is a short string identifying which check fired, recorded in the
    skip event (``salvage_skipped_already_landed`` for reasons 1-3,
    ``salvage_skipped_superseded`` for reason 4) for diagnosis.
    """
    return check_salvage_superseded(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        issue=issue,
    )


def _attempt_salvage(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path,
    worktree_path: Path,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    state_file: Path,
    failure_kind: str | None,
    issue_title: str | None = None,
    issue: dict[str, Any] | None = None,
    worker_outcome: dict[str, Any] | None = None,
    write_gate: WriteGate,
) -> tuple[bool, str | None]:
    """Push a completed branch and open a PR, then move labels to ``pr_open``.

    Returns ``(ok, error)``. Errors are recorded as values and never raised.
    ``ok`` is ``True`` once the PR is created, even if the label swap failed;
    in that case ``error`` describes the label failure and the
    ``session_salvaged`` event records ``label_write_ok=False``.

    ``ok`` is also ``True`` (with ``error=None``) when salvage is *skipped*
    because the work already landed (issue #1221): the dead session's snapshot
    can be stale, so before opening a PR we re-check live terminal state. A
    skip emits ``salvage_skipped_already_landed`` instead of opening a vestigial
    duplicate PR, and the caller treats it as "handled" (no redispatch).

    ``worker_outcome`` (cw#1771) is passed straight through to
    ``_open_salvage_pr``, which prefers its drafted ``pr_title``/``pr_body``
    over synthesis when present. Callers pass the outcome file read from the
    ISSUE's actual worktree directory -- never a repo_root fallback used when
    that directory does not exist -- so a stray `.worker-outcome.json` at the
    main checkout root is never mistaken for this branch's drafted content.

    NOTE (issue #1326): the ``push_branch`` call below threads
    ``dry_run=write_gate.dry_run`` so a dry-run invocation does not issue a
    real ``git push``. This is explicit-threading (mirroring
    ``_reconcile_locked``'s convention) rather than a 7th WriteGate primitive
    -- gating an external git push through WriteGate was an open design
    question (see issue #1326's remedy section) and the simplest correct
    fix is to thread the flag directly. Downstream state writes and label
    transitions remain gated by ``write_gate``; the ``gh pr create`` in
    ``_open_salvage_pr`` is gated at the ``GitHub`` client sink level.
    """
    write_gate = require_write_gate(write_gate)
    already_landed, skip_reason = _salvage_already_landed(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        issue=issue,
    )
    if already_landed:
        with state_lock(state_file):
            state = load_state(state_file)
            # Issue #282: preserve the liveness fingerprint so the recovery
            # path can verify the worker is dead before the worktree is
            # reclaimed.
            # Issue #1241: the event kind is mapped from the skip reason so the
            # new reachability skip (``commits_reachable``) emits
            # ``salvage_skipped_superseded`` while the #1221 reasons keep their
            # existing ``salvage_skipped_already_landed`` event.
            state = write_gate.append_event(
                state,
                salvage_skip_event_kind(
                    skip_reason
                ),  # event-consumer: audit-only -- kind resolves to one of two registered literals (salvage_skipped_already_landed / salvage_skipped_superseded), both in _LEVEL_BY_KIND; the actionable state mutation (worktree reap) happens in the caller, this event is the observable skip record (issue #1241)
                {
                    "issue_number": issue_number,
                    "failure_kind": failure_kind,
                    "reason": skip_reason,
                    # The skip path does NOT remove labels -- label cleanup is the
                    # reconcile lane's job. Record the active labels at skip time
                    # for diagnosis (what state the issue was in), not as removed.
                    "active_labels": sorted(active_labels),
                },
            )
            write_gate.save_state(state)
        return True, None

    # No-PR backend (local-file issues): the branch is the deliverable -- see
    # ``local_work_park``. After the already-landed skip, before any push.
    parked = park_unpublishable_work(
        gh, config, repo_root, branch, issue_number, active_labels, failure_kind, write_gate
    )
    if parked is not None:
        return parked

    push_ok, push_error = push_branch(
        repo_root, branch, worktree_path=worktree_path, dry_run=write_gate.dry_run
    )
    if not push_ok:
        return False, push_error

    pr_number, pr_error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title=issue_title,
        source_description="completed-but-unpublished worker worktree",
        state_file=state_file,
        worker_outcome=worker_outcome,
    )
    if pr_number is None:
        return False, pr_error or "gh pr create failed or returned no PR number"

    with state_lock(state_file):
        state = load_state(state_file)
        # Issue #282: preserve the liveness fingerprint so the recovery path
        # can verify the worker is dead before the worktree is reclaimed.
        state = write_gate.append_event(
            state,
            "session_salvaged",
            {
                "issue_number": issue_number,
                "failure_kind": failure_kind,
                "removed_labels": sorted(active_labels),
                "pr_number": pr_number,
                "label_write_ok": pr_error is None,
                "label_error": pr_error,
            },
        )
        write_gate.save_state(state)
    return True, pr_error


def _open_pr_for_orphaned_branch(
    *,
    gh: GitHubLike,
    config: OrchestratorConfig,
    repo_root: Path | None,
    branch: str,
    base_ref: str,
    issue_number: int,
    active_labels: set[str],
    issue_labels: set[str],
    issue_title: str | None = None,
    state_file: Path | None = None,
    worker_outcome: dict[str, Any] | None = None,
) -> tuple[int | None, str | None, ValidationResult | None]:
    """Open a PR for a branch that the worker pushed but could not create a PR for.

    Returns ``(pr_number, error, closing_ref)``. Errors are recorded as
    values and never raised. This is the orchestrator-side recovery for
    issue #935: workers are unauthenticated in their environment, so after
    pushing a completed branch they cannot run ``gh pr create``. The
    orchestrator, which is authenticated, creates the PR and moves the issue
    labels toward ``pr_open``. See `_open_salvage_pr` for the closing-
    reference validation and post-create verification this delegates to.

    ``worker_outcome`` (cw#1771) is the caller's pre-read
    ``.worker-outcome.json`` for this issue -- this is the dominant,
    by-design case the outcome-file contract exists for (not a crash), so its
    drafted ``pr_title``/``pr_body`` are almost always present here. See
    `_open_salvage_pr` for how they are preferred over synthesis.
    """
    return _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch=branch,
        base_ref=base_ref,
        issue_number=issue_number,
        active_labels=active_labels,
        issue_labels=issue_labels,
        issue_title=issue_title,
        source_description="worker branch that could not open a PR",
        state_file=state_file,
        worker_outcome=worker_outcome,
    )
