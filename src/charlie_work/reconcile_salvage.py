"""The ``session_unpublished_work_salvaged`` drift-fix lane for ``reconcile.apply_fixes``.

Extracted from ``reconcile.py`` during the issue #2226 rework: the branch's
push -> supersession-check -> PR-create -> label-swap body (~225 lines)
pushed ``reconcile.py`` past its file-size ratchet mark, so the lane lives
here as its own module. ``reconcile.apply_fixes`` delegates to
:func:`apply_unpublished_work_salvage` and keeps ownership of the reconcile
event emission; this module owns everything up to producing the updated
``DriftItem``.

Import direction: ``reconcile`` -> ``reconcile_salvage`` only. Names this
module needs back from ``reconcile`` (``DriftItem``, ``_repo_slug``) are
imported lazily inside the function body, mirroring ``reconcile.py``'s own
lazy-import convention for cycle-breaking (e.g. its ``from .worker import
_alive_review_worker_issue_numbers``).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .closing_reference import probe_closing_link, validate_closing_reference
from .config import OrchestratorConfig
from .github import PR_CLOSING_ISSUES_FIELDS, GitHubLike
from .instrumentation import log_event
from .labels import apply_issue_labels
from .local_work_park import publishes_pull_requests
from .pr_create_retry import create_pr_with_retry
from .salvage_superseded import check_salvage_superseded, salvage_skip_event_kind
from .worktree import push_branch, summarize_branch_work

if TYPE_CHECKING:
    from charlie_work.reconcile import DriftItem


def apply_unpublished_work_salvage(
    gh: GitHubLike,
    config: OrchestratorConfig,
    item: DriftItem,
    *,
    state_path: Path | None,
) -> DriftItem:
    """Apply the ``session_unpublished_work_salvaged`` drift fix.

    Issue #252: push the completed branch, create a PR, and move labels to
    ``pr_open``; if any step fails, fall back to the normal relabel-to-ready
    path. Returns the (possibly rebuilt) ``DriftItem`` -- ``apply_fixes``
    folds it into the reconcile event unchanged.

    ``repo`` on the emitted ``lifecycle_transition`` events binds to
    ``gh.repo_root`` -- the repo this lane actually pushes to and labels in
    (issue #2226).
    """
    from charlie_work.reconcile import DriftItem, _repo_slug

    if item.issue_number is not None and item.branch and item.base_branch:
        repo_root = getattr(gh, "repo_root", None)
        repo_name = repo_root.name if repo_root is not None else None
        salvage_ok = False
        salvage_error = "repo_root not available"
        pr_number = None
        if repo_root is not None:
            # Issue #1241: before pushing/opening a PR, re-check LIVE
            # terminal state through the shared single enforcement
            # point (``check_salvage_superseded``). This salvage lane
            # is the second of the two salvage paths (the workflow
            # lane is the other) and previously had NO supersession
            # check -- it opened a vestigial duplicate PR whenever the
            # work had already landed through a sibling merge while
            # the dead session's snapshot still looked stranded. On a
            # skip, do NOT push, do NOT open a PR, and do NOT relabel
            # to ready (the work already landed -- redispatching would
            # loop). Label convergence is left to the closed-issue /
            # merged-PR drift kinds that fire alongside this one. The
            # skip is recorded as an observable event plus a
            # fix_action on the reconcile event.
            superseded, skip_reason = check_salvage_superseded(
                gh=gh,
                config=config,
                repo_root=repo_root,
                branch=item.branch,
                base_ref=item.base_branch,
                issue_number=item.issue_number,
            )
            if superseded:
                if state_path is not None:
                    log_event(
                        state_path,
                        salvage_skip_event_kind(
                            skip_reason
                        ),  # event-consumer: audit-only -- kind resolves to one of two registered literals (salvage_skipped_already_landed / salvage_skipped_superseded), both in _LEVEL_BY_KIND; the actionable label convergence happens in the sibling closed-issue / merged-PR drift kinds, this event is the observable skip record (issue #1241)
                        {
                            "issue_number": item.issue_number,
                            "reason": skip_reason,
                            "branch": item.branch,
                        },
                    )
                item = DriftItem(
                    kind=item.kind,
                    issue_number=item.issue_number,
                    pr_number=item.pr_number,
                    detail=item.detail,
                    fix_actions=item.fix_actions + (f"salvage_skipped: {skip_reason}",),
                    remove_labels=(),
                    add_labels=(),
                    branch=item.branch,
                    base_branch=item.base_branch,
                )
                # Skip the push/PR/relabel block: mark salvage_ok True
                # with no PR so the ``if salvage_ok`` label-swap block
                # below runs against empty remove/add label sets (a
                # no-op) rather than falling through to the
                # relabel-to-ready fallback. The work already landed;
                # redispatch is wrong.
                salvage_ok = True
                salvage_error = None
            else:
                push_ok, push_error = push_branch(repo_root, item.branch)
                if push_ok:
                    # ``publishes_pull_requests`` is the salvage seam's
                    # capability probe (issue #2262): a no-PR backend's
                    # null-object ``pr_create`` exists but can never publish,
                    # so skip the PR block rather than retrying a write that
                    # structurally cannot land.
                    can_publish_pr = publishes_pull_requests(gh) and (
                        getattr(gh, "pr_create", None) is not None
                    )
                    if can_publish_pr:
                        # Same janitor body gate as a worker-authored PR --
                        # boilerplate alone can never satisfy it. Derive the
                        # rationale from the worker's own commit log rather
                        # than injecting the gate's keywords.
                        salvage_body = (
                            f"Closes #{item.issue_number}\n\n"
                            "Salvaged by the orchestrator from a completed-but-unpublished "
                            "worker worktree."
                        )
                        branch_summary = summarize_branch_work(
                            repo_root,
                            item.branch,
                            item.base_branch,
                            test_path_globs=config.test_adequacy.test_path_globs,
                        )
                        if branch_summary:
                            salvage_body = f"{salvage_body}\n\n{branch_summary}"
                        # cw#1263: canonicalize/validate the closing-reference
                        # line the same way workflow.py's `_open_salvage_pr`
                        # does, via the shared `closing_reference` module.
                        # `workflow.py` imports `reconcile.py` (for
                        # `apply_fixes`/`detect_drift`), so importing
                        # `workflow._open_salvage_pr` back into this module
                        # would cycle -- the standalone third module is what
                        # lets both salvage-body builders share one
                        # implementation without either importing the other.
                        closing_ref = validate_closing_reference(
                            salvage_body, item.issue_number, repo=_repo_slug(gh), gh=gh
                        )
                        salvage_body = closing_ref.body
                        if closing_ref.changed and state_path is not None:
                            log_event(
                                state_path,
                                "pr_closing_ref_rewritten",
                                {
                                    "issue_number": item.issue_number,
                                    "findings": list(closing_ref.findings),
                                    "source": "session_unpublished_work_salvaged",
                                },
                            )
                        # cw#1273: route through the bounded outer retry
                        # + duplicate-PR guard instead of calling
                        # gh.pr_create directly, matching workflow.py's
                        # _open_salvage_pr (the other pr_create call site).
                        #
                        # cw#1771: unlike workflow.py's _open_salvage_pr,
                        # this lane deliberately does NOT read
                        # worker_outcome for a drafted title/body. This
                        # branch fires for unpushed work -- the worker
                        # died (or was reaped) before ever running
                        # `git push` -- so any pr_title/pr_body the
                        # worker drafted would describe a push that
                        # never happened and may reference state (a PR
                        # number, a verified head sha) that is false at
                        # the head this code just pushed on the
                        # worker's behalf. Synthesis-only is correct
                        # here; do not "fix" this to match the
                        # clean-handoff lane.
                        retry_result = create_pr_with_retry(
                            gh,
                            head=item.branch,
                            base=item.base_branch,
                            title=f"Salvaged work for issue #{item.issue_number}",
                            body=salvage_body,
                            max_retries=config.runtime.pr_create_retry_max_attempts,
                            base_seconds=config.runtime.pr_create_retry_base_seconds,
                        )
                        pr_number = retry_result.pr_number
                    if pr_number is not None:
                        salvage_ok = True
                        # `pr_number` is falsy (0) under `dry_run`, where no
                        # real PR was opened -- only probe a real, truthy PR
                        # number (mirrors workflow.py::_open_salvage_pr).
                        if pr_number and state_path is not None:
                            # cw#1868: settled across GitHub's indexing
                            # race; None = failed query, never a miss (see
                            # dead_worker_reap._open_salvage_pr).
                            linked_numbers = probe_closing_link(
                                gh,
                                pr_number,
                                item.issue_number,
                                fields=PR_CLOSING_ISSUES_FIELDS,
                            )
                            if (
                                linked_numbers is not None
                                and item.issue_number not in linked_numbers
                            ):
                                log_event(
                                    state_path,
                                    "pr_closing_ref_unlinked",
                                    {
                                        "issue_number": item.issue_number,
                                        "pr_number": pr_number,
                                        "linked_issue_numbers": sorted(linked_numbers),
                                    },
                                )
                    else:
                        salvage_error = (
                            "gh pr create failed or returned no PR number"
                            if can_publish_pr
                            else "backend does not publish pull requests"
                        )
                else:
                    salvage_error = push_error or "git push failed"

        if salvage_ok:
            # write-gate-exempt(issue=2226): salvage lane is out-of-wave raw territory (no write_gate param; reachable only when fix and not dry_run — the gh client owns dry-run).
            label_result = apply_issue_labels(
                gh,
                config.labels,
                item.issue_number,
                add=item.add_labels,
                remove=item.remove_labels,
                state_path=state_path,
                repo=repo_name,
                pr_number=pr_number,
                cause=item.kind,
            )
            label_ok = label_result.ok
            fix_actions = list(item.fix_actions)
            if not label_ok:
                fix_actions.append("label_write_failed: true")
            item = DriftItem(
                kind=item.kind,
                issue_number=item.issue_number,
                pr_number=pr_number,
                detail=item.detail,
                fix_actions=tuple(fix_actions),
                remove_labels=item.remove_labels,
                add_labels=item.add_labels,
                branch=item.branch,
                base_branch=item.base_branch,
            )
        else:
            # Fallback: treat as session_failed_relabeled and add ready label
            # write-gate-exempt(issue=2226): salvage lane is out-of-wave raw territory (no write_gate param; reachable only when `fix and not dry_run`).
            label_result = apply_issue_labels(
                gh,
                config.labels,
                item.issue_number,
                add=(config.labels.ready,) if config.labels.ready not in item.add_labels else (),
                remove=item.remove_labels,
                to_state="ready",
                state_path=state_path,
                repo=repo_name,
                cause="salvage_failed_fallback",
            )
            label_ok = label_result.ok
            fix_actions = list(item.fix_actions)
            fix_actions.append(f"salvage_failed: {salvage_error}")
            if not label_ok:
                fix_actions.append("label_write_failed: true")
            item = DriftItem(
                kind="session_failed_relabeled",
                issue_number=item.issue_number,
                pr_number=None,
                reason="salvage_failed_fallback",
                detail=item.detail,
                fix_actions=tuple(fix_actions),
                remove_labels=item.remove_labels,
                add_labels=(config.labels.ready,),
                branch=item.branch,
                base_branch=item.base_branch,
            )
    return item
