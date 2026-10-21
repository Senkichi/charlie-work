"""Local-git fallback for a PR diff GitHub refuses or fails to serve (issue #1807).

``PullRequests.pr_diff`` reads a PR's diff with the ``vnd.github.v3.diff``
media type. For PRs of more than 300 changed files GitHub refuses to render
that diff at all, answering (observed live on PR #2160, a 461-file mechanical
ledger split):

    {"message": "Sorry, the diff exceeded the maximum number of files (300).
    Consider using 'List pull requests files' API or locally cloning the
    repository instead.", "errors": [{"resource": "PullRequest", "field":
    "diff", "code": "too_large"}]}

Every other fetch failure -- network blip, rate limit, token outage -- fails
the same way. ``pr_diff`` used to collapse all of these to ``""``, the same
value a genuinely-empty diff produces, so ``review()`` wrote an empty
``diff.patch`` into the packet and the #1251 empty-diff gate then skipped
review dispatch on every pass, forever, with no error event and no escalation
(#1807's report: >100 ``review_dispatch_skipped_empty_diff`` warnings and a
review that never ran).

Two pieces live here:

- ``base_ref_from_rest_pr`` extracts ``baseRefName`` from the REST
  ``GET pulls/{number}`` JSON body, so the fallback can diff against the
  PR's real base branch without a GraphQL read.
- ``local_pr_diff`` builds the identical unified diff from the
  orchestrator's own checkout (recipe above).
- ``emit_unavailable_event`` records the ``review_packet_diff_unavailable``
  warning event when even the fallback fails (or cannot run), so the
  condition lands in ``events.db`` instead of collapsing into an empty
  packet. Only a genuinely-empty *and successfully fetched* diff keeps
  producing the (correct, #1251) skip-forever behavior.

The subprocess defaults in this module correspond to a "publisher" host ``git fetch origin pull/N/head`` lands the PR
  head at ``FETCH_HEAD`` (through ``run_git_with_retry``, the same
  network-touching git seam every other fetch uses), and
  ``git diff --merge-base origin/<baseRefName> FETCH_HEAD`` renders the same
  merge-base diff with no file cap. This is exactly the recipe the issue
  prescribes. ``pull/{number}/head`` is stable read-only API: GitHub
  advertises and serves it even for PRs whose head ref (or the whole PR
  branch) has since been deleted.

The subprocess defaults in this module correspond to a "publisher" host
port shape (CLAUDE.md's status indicator answer): ``git fetch`` against the
remote is the only potentially slow call and it goes through
``run_git_with_retry``'s bounded retry loop; the local ``git diff`` is a
pure read against refs the fetch just landed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..git_retry import run_git_with_retry
from ..subprocess_runner import RunResult, run_captured

logger = logging.getLogger(__name__)

#: The warning event emitted when a diff is unavailable from both the remote
#: and the local checkout (registered in ``event_levels/``).
EVENT_KIND = "review_packet_diff_unavailable"

#: ``git fetch`` of one PR head ref: bounded so a stalled remote cannot hang
#: a repo pass. Generous -- a very large PR head can take a while to land --
#: but finite.
GIT_FETCH_TIMEOUT_SECONDS = 300

#: ``git diff --merge-base`` against the fetched head: a pure local read.
GIT_DIFF_TIMEOUT_SECONDS = 120


def base_ref_from_rest_pr(payload: Any) -> str | None:
    """Extract the PR's ``baseRefName`` from a REST ``pulls/{number}`` body.

    Tolerant by design: anything that is not a PR-shaped dict (a transport
    failure value, aGraphQL body, empty stdout) returns ``None``; the caller
    treats that as "fallback cannot determine the base branch" rather than
    guessing ``main``.
    """
    if not isinstance(payload, dict):
        return None
    base = payload.get("base")
    if not isinstance(base, dict):
        return None
    ref = base.get("ref")
    if isinstance(ref, str) and ref.strip():
        return ref
    return None


def _git_error_text(result: RunResult, fallback: str) -> str:
    """One-line failure explanation for a git ``RunResult``.

    Mirrors ``subprocess_runner.command_failure_message``'s preference for
    the specific stderr over the generic ``.error`` ("command exited N").
    """
    text = result.stderr.strip() or (result.error or "").strip()
    return text or fallback


def local_pr_diff(repo_root: Path, pr_number: int, base_ref: str) -> str | None:
    """Build a PR's merge-base diff from the orchestrator's own checkout.

    Returns the diff text (possibly ``""`` when the local merge-base diff is
    genuinely empty -- same shape as a fetched diff file), or ``None`` when
    the fallback could not produce a diff. Never raises: both subprocesses
    are the repo's standard captured-run seams, which already turn failures
    into values.

    ``repo_root`` is the PR's *repository* checkout, i.e. the main checkout,
    not a worker worktree -- the orchestrator's authoritative clone.
    """
    fetch = run_git_with_retry(
        ["git", "fetch", "origin", f"pull/{pr_number}/head"],
        cwd=repo_root,
        timeout_seconds=GIT_FETCH_TIMEOUT_SECONDS,
    )
    if not fetch.ok:
        logger.warning(
            "local_pr_diff #%d: git fetch origin pull/%d/head failed (%s)",
            pr_number,
            pr_number,
            _git_error_text(fetch, "command exited non-zero"),
        )
        return None
    diff = run_captured(
        ["git", "diff", "--merge-base", f"origin/{base_ref}", "FETCH_HEAD"],
        cwd=repo_root,
        timeout_seconds=GIT_DIFF_TIMEOUT_SECONDS,
    )
    if not diff.ok:
        logger.warning(
            "local_pr_diff #%d: git diff --merge-base origin/%s FETCH_HEAD failed (%s)",
            pr_number,
            base_ref,
            _git_error_text(diff, "command exited non-zero"),
        )
        return None
    return diff.stdout


def emit_unavailable_event(
    repo_root: Path | None,
    state_dir: str | None,
    pr_number: int,
    error: str,
) -> None:
    """Record ``review_packet_diff_unavailable`` for a PR whose diff is
    unreachable from both the remote and this checkout (issue #1807).

    Follows ``outbound_body_guard``'s capability-layer event precedent:
    ``repo_root``+``state_dir`` locate the state file, the level registry
    entry supplies the severity, and the whole thing is best-effort --
    instrumentation must never break the ``pr_diff`` return path, whose
    contract is "errors come back as values". ``error`` is the rendered
    transport error text (GitHub's ``too_large`` message, a network
    failure, ...) so the event names the actual cause instead of a generic
    "diff unavailable".
    """
    if repo_root is None:
        return
    try:
        # Deferred: ``paths`` pulls in ``config``, which imports
        # ``charlie_work.github`` -- an import cycle at module load for
        # this package. Mirrors ``outbound_body_guard._state_path_for``.
        from ..instrumentation import log_event
        from .. import layout
        from ..paths import runtime_paths

        state_path = runtime_paths(
            repo_root, state_dir or layout.DEFAULT_STATE_DIR, check_phantom=False
        ).state_file
        # write-gate-exempt(issue=1807): observability-only event write, mirroring outbound_body_guard.py's own stage-2 raw log_event
        log_event(
            state_path,
            EVENT_KIND,
            {"pr_number": pr_number, "error": error[:500]},
            repo=repo_root.name,
        )
    except Exception:  # noqa: BLE001 -- telemetry only
        logger.debug(
            "review_packet_diff_unavailable event for PR #%d could not be recorded",
            pr_number,
            exc_info=True,
        )
