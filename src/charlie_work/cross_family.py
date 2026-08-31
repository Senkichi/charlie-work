"""Cross-family adversarial review for the automated auto-gate path — run a
NON-Claude model (e.g. codex via the Devin CLI) against a spec or PR and
capture its findings for the review packet.

See ``rescue_review.py`` for the sibling implementation used by the rescue
tier (issue #555); the two were split in the role-config Phase 2 cleanup
because only the rescue tier survives that refactor. This module is the
auto-gate-only remainder: report reuse across passes, async launch/reap
(issue #1078), and verdict parsing for the auto-verdict gate.

Rationale: the orchestrator's own adversarial review is Claude-family end to
end, so it shares Claude's blind spots. A different model family surfaces
what same-family redundancy cannot. See the standing practice note in memory
(feedback_cross_family_devin_review): findings are LEADS, not verdicts — the
cross-family model over-escalates severity, so every finding must be verified
against live code before it is accepted, and it never gates a merge on its
own.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .env_sanitize import sanitize_env
from .process_utils import get_process_start_time, is_pid_alive, kill_process_tree
from .rescue_review import (
    CrossFamilyResult,
    CrossFamilyVerdict,
    LEGACY_VACUOUS_SUMMARY,
    _find_json_verdict,
    _VERDICT_RE,
    extract_head_ref_oid,
    extract_report_body,
    render_command,
    report_body_is_valid,
)
from .subprocess_runner import no_console_window_kwargs

logger = logging.getLogger(__name__)

# Local copy of rescue_review._CAVEAT, byte-identical by construction: needed
# only to build the report header in this module's own local _report() copy
# below. _report (like _fail below it) is a local duplicate rather than a
# cross-module import of a private rescue_review helper -- see _fail's own
# comment for the rationale, which applies identically here since both are
# called from the kept async launch/reap block.
_CAVEAT = (
    "> Findings below are **leads, not verdicts** — this cross-family model "
    "over-escalates severity. Verify each against the live code before folding "
    "it in, reject over-escalations with a reason, and never let this report "
    "gate a merge on its own."
)


def report_is_reusable(text: str, current_head_sha: str | None) -> bool:
    """Return True if a stored cross-family report can be reused as-is.

    This is the *single* definition of "reusable", shared by the two callers
    that must agree on it (issue #1081):

    - ``workflow._cross_family_for_pr`` uses it to decide whether to reuse the
      stored report or re-run the cross-family model.
    - ``workflow.OrchestratorApp.loop``'s same-head packet skip uses it to
      decide whether the packet is stale and ``review()`` must re-run.

    They were previously separate: the skip considered only the packet head SHA
    and the prompt-template digest, so a PR whose head never moved skipped
    ``review()`` forever and the regeneration below was never reached. The
    report stayed unusable permanently. Keeping one predicate is what makes
    "the skip fires" and "the report would be reused" the same question — if
    they can disagree, the disagreement *is* the stall.

    A report is reusable only if all three hold:

    - it is not a failure stub (``(UNAVAILABLE)`` header, written by ``_fail``
      on timeout/non-zero exit),
    - its body is a semantically real review (``report_body_is_valid``), and
    - it was generated against exactly ``current_head_sha``.

    The head comparison requires BOTH sides to be known, and is not a bare
    equality. A bare ``extract_head_ref_oid(text) == current_head_sha`` reads
    as if it already rejects a report with no head SHA, but it silently returns
    True when the caller *also* has no head — ``None == None`` — so a
    header-less report would be judged reusable precisely when the head is
    unknowable. That is the same indeterminate-comparison-collapsing-into-the-
    permissive-branch shape #1079 closed one layer up, and ``headRefOid`` being
    absent is the very scenario #1081 was filed about. An unadjudicable head
    must fail closed.
    """
    if not text.strip():
        return False
    if "(UNAVAILABLE)" in text.splitlines()[0]:
        return False
    if not report_body_is_valid(extract_report_body(text)):
        return False
    stored_head_sha = extract_head_ref_oid(text)
    if stored_head_sha is None or current_head_sha is None:
        return False
    return stored_head_sha == current_head_sha


# ---------------------------------------------------------------------------
# Issue #1078: asynchronous launch/reap for cross-family review
#
# The fleet dispatcher iterates repositories sequentially. A synchronous
# ``run_cross_family_review`` call blocks for up to ``timeout_seconds`` (600s
# in production), which can consume 2x the ``full_pass_interval_seconds`` (300s)
# and starve later repos' lanes. The pair below splits the blocking across two
# fleet passes:
#
#   pass N:   ``launch_cross_family_review`` — Popen, write marker, return
#             immediately with ``pending=True``.
#   pass N+1: ``reap_cross_family_review`` — check marker, poll process,
#             collect stdout or kill on timeout, write the report.
#
# The marker (``.pending.json``) stores the PID, launch timestamp, timeout,
# and temp-file paths so the reaper can identify the process across passes.
# Process identity is validated via ``get_process_start_time`` to avoid acting
# on a recycled PID.
# ---------------------------------------------------------------------------


def _pending_marker_path(report_path: Path) -> Path:
    """Sidecar marker path derived from the report path."""
    return report_path.with_suffix(report_path.suffix + ".pending.json")


def _stdout_tmp_path(report_path: Path) -> Path:
    """Temp stdout path for the async subprocess output."""
    return report_path.with_suffix(report_path.suffix + ".stdout.tmp")


def _stderr_tmp_path(report_path: Path) -> Path:
    """Temp stderr path for the async subprocess output."""
    return report_path.with_suffix(report_path.suffix + ".stderr.tmp")


def _cleanup_pending(marker_path: Path, stdout_path: Path, stderr_path: Path) -> None:
    """Remove the marker and temp output files left by a pending review."""
    for p in (marker_path, stdout_path, stderr_path):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def _report(model: str, body: str, head_ref_oid: str | None = None) -> str:
    header = f"# Cross-family adversarial review — `{model}`"
    if head_ref_oid:
        header += f"\n\n<!-- PR head SHA: {head_ref_oid} -->"
    return f"{header}\n\n{_CAVEAT}\n\n---\n\n{body.strip()}\n"


def _fail(
    report_path: Path,
    model: str,
    reason: str,
    *,
    partial: str = "",
    returncode: int | None = None,
) -> CrossFamilyResult:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    stub = f"# Cross-family adversarial review — `{model}` (UNAVAILABLE)\n\n> {reason}\n"
    if partial.strip():
        stub += f"\n---\n\nPartial output before failure:\n\n{partial.strip()}\n"
    report_path.write_text(stub, encoding="utf-8")
    return CrossFamilyResult(
        ok=False, report_path=str(report_path), model=model, returncode=returncode, error=reason
    )


def launch_cross_family_review(
    *,
    model: str,
    command: Sequence[str] | str,
    repo_root: Path,
    prompt_text: str,
    prompt_path: Path,
    report_path: Path,
    timeout_seconds: int,
    dry_run: bool = False,
    head_ref_oid: str | None = None,
    popen: Callable[..., subprocess.Popen] | None = None,
) -> CrossFamilyResult:
    """Launch a cross-family review asynchronously via ``Popen``.

    Writes the prompt, starts the subprocess with stdout/stderr redirected
    to temp files, and writes a ``.pending.json`` marker so a later
    ``reap_cross_family_review`` call can collect the result. Returns
    immediately with ``pending=True`` — the report file is NOT yet written.

    Never raises: launch failures (OSError) are returned as not-ok, non-pending
    results with a failure stub written to ``report_path``, matching
    ``run_cross_family_review``'s contract.

    The in-process retry that ``run_cross_family_review`` performs on
    transient provider errors is deliberately dropped here — a retry with a
    90s sleep would re-introduce the blocking this function exists to
    eliminate. The per-head regeneration budget (``max_regen_attempts``)
    already bounds retries across passes.
    """
    if dry_run:
        return CrossFamilyResult(
            ok=False,
            report_path=str(report_path),
            model=model,
            error="DRY-RUN: cross-family review not executed",
        )

    # Staleness warning — mirrors run_cross_family_review.
    if report_path.exists() and report_path.stat().st_size > 0:
        old_text = report_path.read_text(encoding="utf-8")
        old_head_sha = extract_head_ref_oid(old_text)
        if old_head_sha and head_ref_oid and old_head_sha != head_ref_oid:
            old_body = extract_report_body(old_text)
            if report_body_is_valid(old_body):
                logger.warning(
                    "Cross-family report staleness detected: overwriting report "
                    "for PR head %s with new report for PR head %s.",
                    old_head_sha[:12],
                    head_ref_oid[:12],
                )

    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    rendered = render_command(command, {"model": model, "prompt_path": str(prompt_path)})
    env = sanitize_env(repo_root)
    stdout_path = _stdout_tmp_path(report_path)
    stderr_path = _stderr_tmp_path(report_path)
    marker_path = _pending_marker_path(report_path)

    spawn = popen if popen is not None else subprocess.Popen
    try:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(stdout_path, "w", encoding="utf-8") as stdout_file,
            open(stderr_path, "w", encoding="utf-8") as stderr_file,
        ):
            proc = spawn(
                rendered,
                cwd=str(repo_root),
                stdout=stdout_file,
                stderr=stderr_file,
                shell=isinstance(rendered, str),
                env=env,
                **no_console_window_kwargs(),
            )
    except OSError as exc:
        return _fail(report_path, model, f"cross-family runner failed to start: {exc}")

    expected_start_time = get_process_start_time(proc.pid)
    marker = {
        "pid": proc.pid,
        "started_at": time.time(),
        "timeout_seconds": timeout_seconds,
        "model": model,
        "report_path": str(report_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "head_ref_oid": head_ref_oid,
        "expected_start_time": expected_start_time,
    }
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    return CrossFamilyResult(
        ok=False,
        report_path=str(report_path),
        model=model,
        pending=True,
        error="cross-family review launched, pending",
    )


def reap_cross_family_review(
    *,
    report_path: Path,
) -> CrossFamilyResult | None:
    """Reap a pending cross-family review launched by ``launch_cross_family_review``.

    Returns:
        - ``None`` if no pending marker exists (no review in flight).
        - ``CrossFamilyResult(pending=True)`` if the process is still running
          and within the timeout.
        - ``CrossFamilyResult(ok=True, ...)`` if the process exited and stdout
          is a valid review; the report file is written.
        - ``CrossFamilyResult(ok=False, ...)`` if the process timed out, exited
          with empty/blocked output, or the marker is corrupted; a failure
          stub is written to ``report_path``.

    Marker and temp files are cleaned up in all terminal cases (ok, fail,
    timeout). They are left in place while pending.
    """
    marker_path = _pending_marker_path(report_path)
    if not marker_path.exists():
        return None

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Corrupted marker — clean up and treat as no pending review.
        _cleanup_pending(marker_path, _stdout_tmp_path(report_path), _stderr_tmp_path(report_path))
        return None

    pid = marker.get("pid")
    started_at = marker.get("started_at")
    timeout_seconds = marker.get("timeout_seconds", 600)
    model = marker.get("model", "unknown")
    head_ref_oid = marker.get("head_ref_oid")
    expected_start_time = marker.get("expected_start_time")
    stdout_path = Path(marker.get("stdout_path", str(_stdout_tmp_path(report_path))))
    stderr_path = Path(marker.get("stderr_path", str(_stderr_tmp_path(report_path))))

    if pid is None or started_at is None:
        _cleanup_pending(marker_path, stdout_path, stderr_path)
        return None

    alive = is_pid_alive(pid, expected_start_time)
    elapsed = time.time() - float(started_at)

    if alive and elapsed < float(timeout_seconds):
        return CrossFamilyResult(
            ok=False,
            report_path=str(report_path),
            model=model,
            pending=True,
            error="cross-family review still running",
        )

    if alive and elapsed >= float(timeout_seconds):
        # Timeout — kill the process tree and write a failure stub. Pass
        # expected_start_time so kill_process_tree re-checks process identity
        # before killing: if this pid was recycled by an unrelated process
        # after the reviewer subprocess exited, killing it here would kill
        # that unrelated process instead.
        kill_process_tree(pid, expected_start_time)
        partial = ""
        try:
            partial = stdout_path.read_text(encoding="utf-8")
        except (OSError, ValueError):
            partial = ""
        _cleanup_pending(marker_path, stdout_path, stderr_path)
        return _fail(
            report_path,
            model,
            f"cross-family review timed out after {int(elapsed)}s",
            partial=partial,
        )

    # Process has exited — collect stdout and validate.
    stdout = ""
    try:
        stdout = stdout_path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        pass

    _cleanup_pending(marker_path, stdout_path, stderr_path)

    if not stdout.strip() or not report_body_is_valid(extract_report_body(stdout)):
        return _fail(report_path, model, "cross-family review produced empty or blocked report")

    # Write the final report with the standard header.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_report(model, stdout, head_ref_oid), encoding="utf-8")
    return CrossFamilyResult(
        ok=True,
        report_path=str(report_path),
        model=model,
        returncode=0,
    )


_BLOCKER_OR_MAJOR_RE = re.compile(
    r"\*\*(?:BLOCKER|MAJOR)\*\*|^#+\s*(?:BLOCKER|MAJOR)\b",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MalformedCrossFamilyVerdict:
    """A cross-family report whose output could not be trusted as a verdict.

    Returned by ``parse_cross_family_verdict`` instead of a content-free
    ``CrossFamilyVerdict`` (issue #784) in two cases:

    - The legacy Markdown-only path found a BLOCKER/MAJOR marker but could
      not extract any summary text -- a genuinely vacuous report.
    - A JSON verdict block declared ``decision: "request_changes"`` with an
      empty/missing ``required_changes`` -- a contract violation that must
      not silently degrade into the (possibly better) legacy-path result.

    Callers must treat this the same as an absent verdict: never increment a
    rework counter and never dispatch rework against it. ``raw_body`` and
    ``reason`` are carried for diagnosis (event logging, operator
    inspection) -- this type exists so that outcome is distinguishable from
    a report that was simply absent or UNAVAILABLE (``None``).
    """

    raw_body: str
    reason: str


def parse_cross_family_verdict(
    report_text: str,
) -> CrossFamilyVerdict | MalformedCrossFamilyVerdict | None:
    """Parse a cross-family report into a :class:`CrossFamilyVerdict`.

    Prefers a JSON verdict block (see ``_find_json_verdict``) when the
    report contains one, so ``required_changes`` carries the reviewer's
    itemized findings into the recorded verdict. Falls back to the legacy
    Markdown-only parse — unchanged from before this function accepted JSON
    — for any report without one, so every historical report (and any
    reviewer session that doesn't emit the new block) continues to produce
    byte-identical ``decision``/``summary`` output.

    Two fail-safes govern how the JSON block is trusted:

    - A ``**BLOCKER**``/``**MAJOR**`` marker anywhere in the Markdown findings
      always overrides a JSON block claiming ``"approved"`` — this verdict
      auto-records and can unblock the merge lane, so a downgrade the model
      contradicts in its own findings is never trusted silently.
    - ``required_changes`` is mandatory (per the template) whenever the JSON
      block's *own* ``decision`` is ``"request_changes"``. A block that
      violates that contract (empty or missing list) is not trusted at all
      (issue #784): it returns :class:`MalformedCrossFamilyVerdict` directly
      rather than silently falling through to the legacy path, which would
      let a reviewer dodge the structured contract just by leaving the list
      empty. This check is keyed on the JSON block's own declared decision,
      not the post-fail-safe-override one, so a JSON ``"approved"`` verdict
      that gets overridden to ``request_changes`` by a Markdown severity
      marker is unaffected -- that path never promised itemization.

    Returns ``None`` when the report is absent, UNAVAILABLE, or semantically
    invalid (so the caller skips the PR rather than recording a wrong
    verdict). Returns :class:`MalformedCrossFamilyVerdict` when the report
    asserts blocking findings but delivers nothing usable — callers must
    treat this identically to ``None`` (skip, don't record, don't count
    against any rework budget) but may use it for diagnostics.
    """
    if not report_text or not report_text.strip():
        return None
    first_line = report_text.strip().splitlines()[0]
    if "(UNAVAILABLE)" in first_line:
        return None
    body = extract_report_body(report_text)
    if not report_body_is_valid(body):
        return None

    has_blocker_or_major = bool(_BLOCKER_OR_MAJOR_RE.search(body))

    json_verdict = _find_json_verdict(body)
    if json_verdict is not None:
        if json_verdict.decision == "request_changes" and not json_verdict.required_changes:
            return MalformedCrossFamilyVerdict(
                raw_body=body,
                reason="json_verdict_request_changes_missing_required_changes",
            )
        decision = json_verdict.decision
        if has_blocker_or_major and decision == "approved":
            decision = "request_changes"
        return CrossFamilyVerdict(
            decision=decision,
            summary=json_verdict.summary,
            required_changes=json_verdict.required_changes,
        )

    # Legacy path: no JSON verdict block at all. Extract the verdict section
    # as the summary — logic unchanged from the pre-JSON parser.
    summary = ""
    verdict_match = _VERDICT_RE.search(body)
    if verdict_match:
        # Take the text after the verdict marker up to the next blank line
        # or end of body.
        after = body[verdict_match.end() :].strip()
        # Strip leading markdown punctuation like "**" or "---"
        after = after.lstrip("*-# \n")
        # Take up to ~500 chars of the first paragraph.
        lines: list[str] = []
        for line in after.splitlines():
            stripped = line.strip()
            if not stripped or stripped == "---":
                break
            lines.append(stripped)
        summary = " ".join(lines)[:500]
    if has_blocker_or_major:
        # No hardcoded placeholder here (issue #784): a BLOCKER/MAJOR marker
        # with no extractable summary is genuinely content-free, and
        # CrossFamilyVerdict.__post_init__ raises on exactly that shape.
        try:
            return CrossFamilyVerdict(decision="request_changes", summary=summary)
        except ValueError:
            return MalformedCrossFamilyVerdict(
                raw_body=body,
                reason="blocker_or_major_with_no_extractable_summary",
            )
    return CrossFamilyVerdict(
        decision="approved",
        summary=summary or "Cross-family review found no BLOCKER or MAJOR findings",
    )


__all__ = [
    "CrossFamilyResult",
    "LEGACY_VACUOUS_SUMMARY",
    "MalformedCrossFamilyVerdict",
    "launch_cross_family_review",
    "reap_cross_family_review",
    "report_is_reusable",
    "parse_cross_family_verdict",
]
