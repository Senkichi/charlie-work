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
``publishes_pull_requests`` directly, and the dispatch worker-token probe
reaches it through ``worker_github_token_findings_if_publishing`` below.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import OrchestratorConfig
from .env_sanitize import WorkerTokenFinding, worker_github_token_findings
from .github import GitHubError, GitHubLike
from .labels import TransitionOutcome
from .paths import runtime_paths
from .state import load_state, state_lock
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


def worker_github_token_findings_if_publishing(
    config: OrchestratorConfig, gh: GitHubLike
) -> list[WorkerTokenFinding]:
    """``worker_github_token_findings`` gated on PR capability (issue #1810).

    A backend that cannot publish pull requests never has a worker push a
    branch or open a PR, so a missing scoped worker GitHub token is not a
    defect there: return no findings, which skips the escalation path the
    findings feed in ``_dispatch_impl`` (the ``worker_token_missing``
    warning plus the ``require_worker_github_token`` deferral) instead of
    reporting a defect that cannot exist on such a repo.
    """
    if not publishes_pull_requests(gh):
        return []
    return worker_github_token_findings(config)


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
    try:
        body_dir.mkdir(parents=True, exist_ok=True)
        body_path = body_dir / "review-ready-comment.md"
        body_path.write_text(
            f"Work for this issue is committed on branch `{branch}`. This repo has no "
            "remote, so nothing was pushed and no PR exists: review the branch, merge it, "
            "and close the issue.",
            encoding="utf-8",
        )
        gh.issue_comment(issue_number, body_path)
    except (OSError, GitHubError):
        logger.warning("review-ready comment post failed issue=%d", issue_number, exc_info=True)
