"""Orchestrator-side repair of a PR-body-only closing-keyword gate failure (issue #2108).

CI's Lint job runs the closing-keyword gate (issue #790), which fails a PR whose
*body* (or a commit message) holds an unnegated closing keyword aimed at an issue
other than the PR's declared target. A failure that lives in the PR body is not
something a rework worker can fix -- workers carry no ``gh`` token (issue #502),
so they cannot edit the body -- and routing it to rework just burns the no-op
rework cap until the PR escalates (the #2005 / PR #2021 incident). The
orchestrator holds the token, so it applies the gate's own suggested rewrite
(``closed #60`` -> ``closed issue 60``) and re-runs the failed Lint run.

This module is the pure/IO-light half: classification, the scan, the rewrite,
and the edit + rerun mechanics. State, events, and escalation are the caller's
(``orchestration/state_ci_gate_body_autofix.py``). It never raises: every
failure comes back as a value on ``AutofixResult``.

A failure ``via commit message`` is deliberately *not applicable* -- fixing it
needs a history rewrite, which stays worker rework -- and so is any failure whose
failing steps include anything besides the closing-keyword gate.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .checks import _is_failing_run
from .closing_keyword_gate import (
    UnexpectedClosingReference,
    exclude_base_reachable_commits,
    find_unexpected_closing_references,
)
from .github import (
    CLOSING_KEYWORD_PR_FIELDS,
    GitHubError,
    GitHubRunResult,
    defang_closing_keywords,
)
from .issue_linking import iter_unnegated_closing_keyword_matches, linked_issue_number

logger = logging.getLogger(__name__)

# Prefix of the CI step name (``.github/workflows/ci.yml``, "Closing-keyword gate
# (issue 790)"). A prefix, not the full name, so the issue suffix can change.
CLOSING_KEYWORD_STEP_PREFIX = "Closing-keyword gate"

# Per-head cap on autofix attempts. The gate and this module share one scan, so a
# repeat failure on the same head means the scan and CI disagree; bound the loop.
MAX_AUTOFIX_ATTEMPTS_PER_HEAD = 2

# ``AutofixResult.reason`` of a HELD outcome caused by a transient scan fetch failure.
SCAN_UNAVAILABLE_REASON = "scan_unavailable"


class AutofixOutcome(str, Enum):
    NOT_APPLICABLE = "not_applicable"  # route as before (worker rework)
    FIXED = "fixed"  # body edited and/or Lint re-run requested
    HELD = "held"  # transient (run still in progress, scan fetch failed); retry next pass
    FAILED = "failed"  # could not repair; caller escalates (never rework)


@dataclass(frozen=True)
class AutofixResult:
    outcome: AutofixOutcome
    reason: str = ""
    body_edited: bool = False
    rewritten: tuple[str, ...] = ()
    run_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ClosingReferenceScan:
    intended: int | None
    body: str
    body_findings: tuple[UnexpectedClosingReference, ...]
    commit_findings: tuple[UnexpectedClosingReference, ...]


def rewrite_unexpected_body_references(body: str, *, intended: int) -> str:
    """Defang every unnegated closing reference in ``body`` except the declared target's.

    ``defang_closing_keywords`` rewrites unconditionally, so applying it to the
    whole body would also strip the intended ``Closes #N`` and unlink the PR from
    its own issue. Rewrite per match, over the same
    ``iter_unnegated_closing_keyword_matches`` scan the gate uses.
    """
    out: list[str] = []
    cursor = 0
    for match in iter_unnegated_closing_keyword_matches(body):
        if int(match.group(1)) == intended:
            continue
        out.append(body[cursor : match.start()])
        out.append(defang_closing_keywords(match.group(0)))
        cursor = match.end()
    out.append(body[cursor:])
    return "".join(out)


def failing_runs_are_closing_keyword_gate_only(
    checks: Sequence[Mapping[str, Any]] | None,
    failed_check_names: Sequence[str],
    fetch_job: Callable[[int], dict[str, Any] | None],
) -> tuple[int, ...] | None:
    """Return the workflow run ids to re-run when the closing-keyword gate is the *only* failing step.

    ``None`` means not provable (a failing job we cannot inspect, a failing step
    other than the gate, or no failing run found at all) -- the caller keeps the
    existing rework routing. At least one failing job must have been verified, so
    an empty ``checks`` can never read as "gate-only".
    """
    run_ids: set[int] = set()
    verified = 0
    for check in checks or ():
        if check.get("name") not in failed_check_names or not _is_failing_run(dict(check)):
            continue
        job_id = check.get("databaseId")
        run_id = check.get("runId")
        if not isinstance(job_id, int) or not isinstance(run_id, int):
            return None
        job = fetch_job(job_id)
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list):
            return None
        failed = [
            str(step.get("name") or "")
            for step in steps
            if isinstance(step, dict) and str(step.get("conclusion") or "").lower() == "failure"
        ]
        if len(failed) != 1 or not failed[0].startswith(CLOSING_KEYWORD_STEP_PREFIX):
            return None
        run_ids.add(run_id)
        verified += 1
    return tuple(sorted(run_ids)) if verified else None


def scan_pr_closing_references(
    gh: Any, pr_number: int, *, branch_prefix: str
) -> ClosingReferenceScan | None:
    """Re-run the gate's own scan against the live PR; ``None`` if any fetch fails.

    Mirrors ``closing_keyword_gate_command.run_closing_keyword_check_command``
    (same field set, same live-merge-base commit exclusion, same
    ``linked_issue_number`` target resolution) so the repair rewrites exactly
    what the gate would flag. Fails closed: an unresolvable surface is ``None``,
    never an empty finding list.
    """
    pr = gh.pr_view(pr_number, fields=CLOSING_KEYWORD_PR_FIELDS)
    commits = gh.pr_commits(pr_number)
    if not pr or commits is None:
        return None
    base_ref, head_sha = pr.get("baseRefName"), pr.get("headRefOid")
    comparison = gh.compare(str(base_ref), str(head_sha)) if base_ref and head_sha else None
    merge_base = comparison.get("merge_base_commit") if comparison else None
    merge_base_sha = merge_base.get("sha") if isinstance(merge_base, dict) else None
    if not isinstance(merge_base_sha, str) or not merge_base_sha:
        return None
    scanned = exclude_base_reachable_commits(commits, merge_base_sha=merge_base_sha)
    intended = linked_issue_number(
        pr, is_cross_repository=pr.get("isCrossRepository"), branch_prefix=branch_prefix
    )
    findings = find_unexpected_closing_references(
        pr_body=str(pr.get("body") or ""),
        commit_messages=[str((c.get("commit") or {}).get("message") or "") for c in scanned],
        intended_issue_number=intended,
    )
    return ClosingReferenceScan(
        intended=intended,
        body=str(pr.get("body") or ""),
        body_findings=tuple(f for f in findings if f.source == "pr body"),
        commit_findings=tuple(f for f in findings if f.source != "pr body"),
    )


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def autofix_closing_keyword_pr_body(
    gh: Any,
    *,
    pr_number: int,
    issue_number: int,
    run_ids: Sequence[int],
    branch_prefix: str,
    body_path: Path,
    rerun_already_running: Callable[[str], bool],
) -> AutofixResult:
    """Rewrite the PR body's stray closing references, then re-run the failed Lint run(s).

    Precondition (caller-checked): the closing-keyword gate is the only failing
    step of every run in ``run_ids``. Idempotent across passes: a body already
    fixed (by us on an earlier pass whose rerun was refused, or by anyone else)
    skips the edit and goes straight to the rerun.
    """
    try:
        scan = scan_pr_closing_references(gh, pr_number, branch_prefix=branch_prefix)
    except GitHubError as exc:  # pr_view raises; pr_commits/compare return None
        logger.warning("closing-keyword autofix: scan fetch failed for #%s: %s", pr_number, exc)
        scan = None
    if scan is None:
        # A fetch failure says nothing about the body, and an escalation here is
        # terminal (Lint stays red, so the mechanical de-escalation sweep cannot
        # clear it). Hold and retry next pass instead.
        return AutofixResult(AutofixOutcome.HELD, SCAN_UNAVAILABLE_REASON)
    if scan.commit_findings:
        return AutofixResult(AutofixOutcome.NOT_APPLICABLE, "commit_message_finding")
    if scan.intended is None and scan.body_findings:
        return AutofixResult(AutofixOutcome.FAILED, "no_declared_target")
    if scan.body_findings and scan.intended != issue_number:
        # The gate resolves its target without the orchestrator's branch-issue
        # validator, so on a stale ``agent/issue-N`` branch it can name a different
        # issue than the one ``review()`` bound the PR to. Defanging the body's
        # ``Closes #M`` there would unlink the PR from its real lane; escalate.
        return AutofixResult(AutofixOutcome.FAILED, "declared_target_mismatch")

    rewritten = tuple(f.matched_text for f in scan.body_findings)
    if scan.body_findings:
        assert scan.intended is not None
        new_body = rewrite_unexpected_body_references(scan.body, intended=scan.intended)
        try:
            _write_text_atomic(body_path, new_body)
            gh.pr_edit(pr_number, body_path)
        except (GitHubError, OSError) as exc:
            logger.warning("closing-keyword autofix: body edit failed for #%s: %s", pr_number, exc)
            return AutofixResult(AutofixOutcome.FAILED, f"body_edit_failed: {exc}")

    errors: list[str] = []
    triggered: list[int] = []
    for run_id in run_ids:
        result = gh.run(["run", "rerun", str(run_id), "--failed"], allow_failure=True)
        if isinstance(result, GitHubRunResult) and not result.ok:
            errors.append(result.error or f"gh run rerun {run_id} exited {result.returncode}")
        elif isinstance(result, GitHubRunResult) or isinstance(result, str):
            triggered.append(run_id)  # a str is the dry-run echo; treat as success
        else:
            errors.append(f"unexpected result from gh run rerun {run_id}: {result!r}")
    edited = bool(scan.body_findings)
    if errors and not triggered and all(rerun_already_running(e) for e in errors):
        return AutofixResult(AutofixOutcome.HELD, errors[0], edited, rewritten)
    if errors:
        return AutofixResult(
            AutofixOutcome.FAILED, "rerun_failed: " + "; ".join(errors), edited, rewritten
        )
    return AutofixResult(AutofixOutcome.FIXED, "", edited, rewritten, tuple(triggered))
