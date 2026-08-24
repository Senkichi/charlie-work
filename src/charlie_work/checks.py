from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import InfraBlockedConfig


@dataclass(frozen=True)
class CheckSummary:
    required: tuple[str, ...]
    passed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...]
    infra_failed: tuple[str, ...]
    # Issue #1383: required checks that failed because of a fleet-wide
    # infrastructure condition (Actions budget/runner outage) rather than
    # the PR's code. Distinct from ``infra_failed`` (per-PR
    # CANCELLED/INFRA_FAILURE/TIMED_OUT, issue #841): infra_blocked is a
    # fleet-wide condition escalated once per window, not per-PR, and never
    # routed to rework. Populated by the data-boundary enrichment
    # (``_enrich_checks_infra_blocked`` in ``workflow.py``) which rewrites a
    # FAILURE check's state to the ``INFRA_BLOCKED`` marker before
    # ``summarize_checks`` ever sees it.
    infra_blocked: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return (
            not self.pending
            and not self.failed
            and not self.missing
            and not self.infra_failed
            and not self.infra_blocked
            and not self.unavailable
        )


class _CheckClassification(Enum):
    """Single-check-run classification used by all check consumers.

    Each call site maps these values to its own output shape:
    ``summarize_checks`` aggregates them into ``CheckSummary`` buckets;
    ``_is_failing_run`` returns ``True`` only for ``FAIL``;
    ``_is_infra_run`` returns ``True`` for ``CANCELLED`` or ``INFRA``;
    ``_non_required_check_findings`` in ``workflow.py`` reports ``CANCELLED``
    separately and treats ``INFRA`` as a failure in the informational context.
    """

    PASS = "pass"
    PENDING = "pending"
    EMPTY = "empty"  # null/empty state+bucket, treated as pending
    SKIPPED = "skipped"  # SKIPPED/NEUTRAL — legitimate non-outcomes
    CANCELLED = "cancelled"  # CANCELLED (infra hiccup, reported distinctly)
    INFRA = "infra"  # INFRA_FAILURE or TIMED_OUT
    # Issue #1383: a FAILURE conclusion reclassified at the data boundary as
    # a fleet-wide infrastructure block (billing/runner outage). Routed to
    # ``CheckSummary.infra_blocked``, never to rework.
    INFRA_BLOCKED = "infra_blocked"
    FAIL = "fail"  # FAILURE or any other unrecognized terminal state


def _classify_check_run(check: dict[str, Any]) -> _CheckClassification:
    """Classify a single check run into a shared enum.

    ``state`` carries the canonical conclusion (``gh pr checks`` exposes it
    directly and the GraphQL fallback reconstructs it from status/conclusion).
    ``bucket`` is used only as the pass/pending alternatives it is documented
    to provide (see ``PR_CHECKS_FIELDS`` in ``github.py``), never as an
    independent requirement.

    The ``INFRA_BLOCKED`` marker state is set by the data-boundary enrichment
    in ``workflow.py`` (``_enrich_checks_infra_blocked``) when a FAILURE
    check's job shows structural evidence of a non-started job (zero steps /
    instant-fail) or a config-listed billing annotation -- issue #1383. It is
    never produced by GitHub's API directly, so an unenriched check list
    (e.g. from a unit test) cannot land here by accident.
    """
    state = str(check.get("state") or "").upper()
    bucket = str(check.get("bucket") or "").lower()

    if state == "SUCCESS" or bucket == "pass":
        return _CheckClassification.PASS
    if state in {"PENDING", "QUEUED", "IN_PROGRESS", "REQUESTED"} or bucket == "pending":
        return _CheckClassification.PENDING
    if not state and not bucket:
        return _CheckClassification.EMPTY
    if state in {"SKIPPED", "NEUTRAL"}:
        # Legitimate non-outcomes (path-filtered/matrix-conditional
        # jobs) — never a failure.
        return _CheckClassification.SKIPPED
    if state == "CANCELLED":
        return _CheckClassification.CANCELLED
    if state == "INFRA_BLOCKED":
        return _CheckClassification.INFRA_BLOCKED
    if state in {"INFRA_FAILURE", "TIMED_OUT"}:
        return _CheckClassification.INFRA
    return _CheckClassification.FAIL


def summarize_checks(
    checks: list[dict[str, Any]] | None, required: tuple[str, ...]
) -> CheckSummary:
    if checks is None:
        # Command-level failure (gh returned no parseable check list). Every
        # required check is reported as unavailable, not missing, so callers can
        # distinguish a broken gh CLI from genuinely missing checks.
        return CheckSummary(
            required=required,
            passed=(),
            pending=(),
            failed=(),
            missing=(),
            infra_failed=(),
            infra_blocked=(),
            unavailable=required,
        )

    # Group all runs by name (multiple runs can share the same name, e.g., matrix legs)
    by_name: dict[str, list[dict[str, Any]]] = {}
    for check in checks:
        name = str(check.get("name") or "")
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(check)

    passed: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    infra_failed: list[str] = []
    infra_blocked: list[str] = []

    for name in required:
        runs = by_name.get(name)
        if not runs:
            missing.append(name)
            continue

        # Aggregate all runs for this name using worst-of semantics:
        # - failed if ANY run failed
        # - pending if any run is pending and none failed
        # - passed only if ALL runs passed
        name_failed = False
        name_pending = False
        name_infra_failed = False
        name_infra_blocked = False

        for check in runs:
            classification = _classify_check_run(check)

            if classification == _CheckClassification.PASS:
                # This run passed - continue checking other runs
                continue
            if classification in {_CheckClassification.PENDING, _CheckClassification.EMPTY}:
                # Not yet completed - the whole check name is pending unless a later run fails
                name_pending = True
            elif classification == _CheckClassification.SKIPPED:
                # Legitimate non-outcomes (path-filtered/matrix-conditional
                # jobs) — never a failure.
                continue
            elif classification == _CheckClassification.INFRA_BLOCKED:
                name_infra_blocked = True
            elif classification in {_CheckClassification.CANCELLED, _CheckClassification.INFRA}:
                name_infra_failed = True
            elif classification == _CheckClassification.FAIL:
                name_failed = True

        if name_failed:
            failed.append(name)
        elif name_infra_blocked:
            infra_blocked.append(name)
        elif name_infra_failed:
            infra_failed.append(name)
        elif name_pending:
            pending.append(name)
        else:
            passed.append(name)

    return CheckSummary(
        required=required,
        passed=tuple(passed),
        pending=tuple(pending),
        failed=tuple(failed),
        missing=tuple(missing),
        infra_failed=tuple(infra_failed),
        infra_blocked=tuple(infra_blocked),
        unavailable=(),
    )


# Matches the workflow-run-id segment of a GitHub Actions check link, e.g.
# https://github.com/OWNER/REPO/actions/runs/RUN_ID/job/JOB_ID (optionally
# followed by a query string or #fragment, e.g. "?check_suite_focus=true").
_ACTIONS_RUN_LINK_RE = re.compile(r"/actions/runs/(\d+)/job/")


def _run_id_from_link(link: str | None) -> int | None:
    """Derive a GitHub Actions workflow run id from a check's ``link`` field.

    ``gh pr checks --json`` has no ``runId`` field, so the run id (needed to
    call ``gh run rerun RUN_ID`` for flake-aware debounce, issue #391) must be
    parsed out of the check's link. Only GitHub Actions check links match;
    external status checks may have arbitrary or empty links. Never raises —
    returns ``None`` for anything that doesn't match.
    """
    if not link:
        return None
    match = _ACTIONS_RUN_LINK_RE.search(link)
    if not match:
        return None
    return int(match.group(1))


@dataclass(frozen=True)
class CheckDebounceResult:
    """Result of classifying required-check failures for flake-aware debounce.

    Fields:
        rerun_run_ids: Distinct workflow run ids that should be retried because
            one or more required checks failed for the first time on the
            current head SHA.
        check_rerun_attempts: New value for the PR state key
            ``check_rerun_attempts`` keyed by head SHA and then check name.
        definitive_failed: Required checks that have already consumed their
            one rerun attempt on this head and are still failing.
    """

    rerun_run_ids: tuple[int, ...] = ()
    check_rerun_attempts: dict[str, Any] = field(default_factory=dict)
    definitive_failed: tuple[str, ...] = ()


def classify_check_failures(
    checks: list[dict[str, Any]] | None,
    required: tuple[str, ...],
    pr_state: dict[str, Any] | None,
    head_sha: str | None,
    *,
    record_attempts: bool = True,
) -> CheckDebounceResult:
    """Classify required-check failures for flake-aware debounce.

    The returned ``check_rerun_attempts`` is a nested dict that should be stored
    in the PR's state entry under the key ``check_rerun_attempts``. It is keyed
    by head SHA, then by required check name, with a list of run ids for which a
    rerun has already been attempted.

    Rules:
      * A passing required check clears the attempt marker for the current head.
      * A new head resets all attempt markers (we keep only the current head).
      * A failing required check whose run id has not been attempted on the
        current head is a first failure; when ``record_attempts`` is ``True``
        the run id is added to the attempts and returned in ``rerun_run_ids``.
      * A failing required check whose run id has already been attempted is
        definitive and is returned in ``definitive_failed``.
      * If a failed required check has no parseable GitHub Actions run id, we
        cannot auto-rerun it and it is treated as definitive.
    """
    if not required:
        return CheckDebounceResult()

    summary = summarize_checks(checks, required)

    # Existing attempts keyed by head SHA.
    attempts_state: dict[str, Any] = dict((pr_state or {}).get("check_rerun_attempts") or {})
    current_attempts: dict[str, Any]
    if head_sha:
        current_attempts = dict(attempts_state.get(head_sha) or {})
    else:
        current_attempts = {}

    # Clear markers for required checks that are now passing on this head.
    passing: set[str] = set(summary.passed)
    for name in list(current_attempts):
        if name in passing:
            del current_attempts[name]

    # Group raw check runs by name, filtering to required checks only.
    runs_by_name: dict[str, list[dict[str, Any]]] = {}
    if checks:
        for check in checks:
            name = str(check.get("name") or "")
            if name in required:
                runs_by_name.setdefault(name, []).append(check)

    rerun_run_ids: list[int] = []
    definitive_failed: list[str] = []

    for name in summary.failed:
        # Collect the failing Actions run ids for this required check.
        failed_run_ids: set[int] = set()
        for check in runs_by_name.get(name, []):
            if not _is_failing_run(check):
                continue
            run_id: int | None = check.get("runId")
            if not isinstance(run_id, int):
                run_id = _run_id_from_link(check.get("link"))
            if isinstance(run_id, int):
                failed_run_ids.add(run_id)

        if not failed_run_ids:
            # No parseable Actions run id (e.g. external status check) — cannot
            # auto-rerun, so treat as definitive immediately.
            definitive_failed.append(name)
            continue

        attempted_run_ids: set[int] = set(current_attempts.get(name, []))
        new_run_ids = failed_run_ids - attempted_run_ids

        if not new_run_ids:
            # All failing run ids have already been retried for this head.
            definitive_failed.append(name)
            continue

        if record_attempts:
            rerun_run_ids.extend(sorted(new_run_ids))
            current_attempts[name] = sorted(attempted_run_ids | new_run_ids)
        else:
            # Caller has other janitor blockers; do not consume a rerun attempt
            # but treat it as definitive for this pass.
            definitive_failed.append(name)

    if head_sha:
        attempts_state = {head_sha: current_attempts}
    else:
        # If we cannot associate attempts with a head, drop them to avoid
        # cross-head contamination.
        attempts_state = {}

    return CheckDebounceResult(
        rerun_run_ids=tuple(sorted(set(rerun_run_ids))),
        check_rerun_attempts=attempts_state,
        definitive_failed=tuple(definitive_failed),
    )


def _is_failing_run(check: dict[str, Any]) -> bool:
    """Return True if a single check run represents a code failure (not pending/infra/skipped)."""
    return _classify_check_run(check) == _CheckClassification.FAIL


def _is_infra_run(check: dict[str, Any]) -> bool:
    """Return True if a single check run represents an infrastructure failure.

    The counterpart to `_is_failing_run`: `_CheckClassification.CANCELLED`
    and `_CheckClassification.INFRA` are infra conditions, not code failures.
    Routes through `_classify_check_run` -- the same shared classifier
    `summarize_checks` uses to populate `CheckSummary.infra_failed` -- so a
    run whose `bucket` resolves it to PASS/PENDING before `state` is ever
    consulted (issue #985) is no longer misclassified as infra here while
    the aggregator already treats it as passing/pending.
    """
    return _classify_check_run(check) in {
        _CheckClassification.CANCELLED,
        _CheckClassification.INFRA,
    }


# Setup/bootstrap step names that do not count as "the job ran real work".
# Structural definition (the meaning of "zero non-setup steps"), not the
# config-listed infra-annotation set issue #1383 keeps in config -- so this
# stays a code constant, not an operator-tunable list.
_SETUP_STEP_NAMES: frozenset[str] = frozenset(
    {"set up job", "checkout", "initialize", "complete job"}
)


def _parse_actions_ts(value: Any) -> datetime | None:
    """Parse a GitHub Actions ISO-8601 timestamp (``Z`` or offset) to aware datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _job_duration_seconds(job: dict[str, Any]) -> float | None:
    """Seconds between a job's ``started_at`` and ``completed_at``, or None."""
    started = _parse_actions_ts(job.get("started_at"))
    completed = _parse_actions_ts(job.get("completed_at"))
    if started is None or completed is None:
        return None
    delta = (completed - started).total_seconds()
    if delta < 0:
        return None
    return delta


def is_infra_blocked_check(
    job: dict[str, Any],
    annotations: list[dict[str, Any]] | None,
    config: InfraBlockedConfig,
) -> bool:
    """Classify a FAILED job as ``infra_blocked`` (fleet-wide infra), not code.

    Issue #1383: a required check that failed because the Actions budget
    exhausted / no runner was available carries zero signal about the PR's
    code. Routing it to rework burns caps on no-op cycles and escalates a
    healthy PR. This classifier is the single source of truth for the
    structural + annotation signals that distinguish such a failure from a
    genuine code FAILURE, applied at the check-ingestion data boundary
    (``_enrich_checks_infra_blocked`` in ``workflow.py``) BEFORE the
    rework-routing decision.

    Signals (preferred order -- structural over string where the API exposes
    it, per the issue):
      1. Annotation match: any annotation whose ``message`` contains a
         case-insensitive substring from ``config.annotation_patterns``. A
         billing annotation is authoritative infrastructure evidence even
         when step data is present, so this wins independently.
      2. Zero non-setup steps: the job has no ``steps`` key at all, an
         empty steps list, or only setup bootstrap steps -- the runner
         never started the actual work (billing lapse / runner never
         picked up the job). A missing ``steps`` key is treated as zero
         steps, restoring the pre-#1383 ``is_infrastructure_failure``
         behavior for the absent-key shape (round-2 #1383 review).

    ``config.instant_fail_seconds`` is a reserved knob with no current
    behavioral effect (see its config docstring).

    Returns ``False`` for any non-FAILURE conclusion (the function is only
    meaningful for checks already known to have failed). Never raises: a
    malformed job (missing/typed-wrong fields) degrades to ``False``, so an
    unenrichable check falls back to ordinary ``failed`` routing rather than
    crashing the ingestion path.
    """
    if not isinstance(job, dict):
        return False
    if not config.enabled:
        return False
    if str(job.get("conclusion") or "").upper() != "FAILURE":
        return False

    # Signal 1: config-listed annotation substring (case-insensitive).
    patterns = config.annotation_patterns
    if patterns:
        raw_annotations = annotations if isinstance(annotations, list) else []
        for annotation in raw_annotations:
            if not isinstance(annotation, dict):
                continue
            message = str(annotation.get("message") or "").lower()
            if any(pattern.lower() in message for pattern in patterns if pattern):
                return True

    # Signal 2: zero non-setup steps (runner never started the work). A
    # missing ``steps`` key is treated as zero steps -- the Actions API
    # omits the array for some check-run shapes, and the pre-#1383
    # ``is_infrastructure_failure`` returned True for that case (a FAILURE
    # job with no step data never started the work). Restoring that
    # behavior keeps this classifier behavior-preserving for the
    # absent-key shape that the prior Signal 3 duration gate alone missed
    # when timestamps were also absent (issue #1383 round-2 review). The
    # config docstring's "0 disables the timing signal; zero-step-alone
    # still classifies" contract is honored: a missing/empty steps array
    # classifies regardless of ``instant_fail_seconds``.
    steps = job.get("steps")
    if steps is None:
        return True
    if isinstance(steps, list):
        non_setup = [
            s
            for s in steps
            if isinstance(s, dict) and str(s.get("name") or "").lower() not in _SETUP_STEP_NAMES
        ]
        if len(steps) == 0 or len(non_setup) == 0:
            return True

    return False


@dataclass(frozen=True)
class InfraRerunResult:
    """Result of classifying required-check infrastructure failures for auto-rerun.

    Mirrors `CheckDebounceResult` but for CANCELLED/INFRA_FAILURE/TIMED_OUT
    required checks (issue #841): unlike a genuine code FAILURE, an infra
    failure has no code-fix rework path, so the caller escalates to a human
    once a check's run ids are exhausted rather than dispatching rework.

    Fields:
        rerun_run_ids: Distinct workflow run ids that should be retried via
            `gh run rerun RUN_ID` because the required check they belong to
            has not yet exhausted its attempt cap on the current head SHA.
        infra_rerun_attempts: New value for the PR state key
            ``infra_rerun_attempts``, keyed by head SHA, then required check
            name, then run id (as a string), storing the number of rerun
            attempts made so far for that run id.
        definitive_failed: Required checks whose infra failure will not be
            retried this pass -- either every failing run id has already
            reached the attempt cap, no run id could be parsed from the
            check's link (so it cannot be auto-retried at all), or the
            caller declined to record attempts (other janitor blockers are
            present this pass, mirroring `classify_check_failures`).
    """

    rerun_run_ids: tuple[int, ...] = ()
    infra_rerun_attempts: dict[str, Any] = field(default_factory=dict)
    definitive_failed: tuple[str, ...] = ()


def classify_infra_failures(
    checks: list[dict[str, Any]] | None,
    required: tuple[str, ...],
    pr_state: dict[str, Any] | None,
    head_sha: str | None,
    *,
    record_attempts: bool = True,
    attempt_cap: int = 2,
) -> InfraRerunResult:
    """Classify required-check infrastructure failures for bounded auto-rerun.

    A job-level `timeout-minutes` kill on this repo's self-hosted runners
    reports CANCELLED, not TIMED_OUT (verified across the REST Jobs API, REST
    Checks API, and GraphQL statusCheckRollup). `summarize_checks` already
    buckets that into `infra_failed`, which correctly blocks merge
    (`CheckSummary.ready`) -- but until this function, nothing ever retried
    or escalated it: `classify_check_failures` only iterates `summary.failed`
    (a code push can't fix an infra kill), so an infra-failed PR sat blocked
    forever with only a diagnostic `merge_failed_attempt_alarm` event
    (issue #841).

    Benign supersede-cancels (ci.yml's own `concurrency: cancel-in-progress`
    on a superseded PR-branch run, or the "Reclaim superseded main CI runs"
    workflow / `cancel_superseded_runs()` cancelling queued main runs) are
    unreachable here by construction: `gh pr checks` is scoped to the PR's
    CURRENT head SHA only, so a CANCELLED entry for an old, superseded head
    is invisible by the time this function runs on a later pass. There is no
    per-check head_sha field to compare against a branch tip, so no runtime
    discriminator is possible or needed here (mirrors the existing
    worst-of-aggregation comment a few lines up in `summarize_checks`, made
    for a different purpose).

    Rules (mirrors `classify_check_failures`, count-based instead of
    set-based because `gh run rerun` reuses the same run id across retries --
    verified live: two production reruns both landed as ``run_attempt: 2`` on
    the SAME run id, never a new run id):
      * A passing required check clears the attempt marker for the current head.
      * A new head resets all attempt markers (we keep only the current head).
      * An infra-failed required check whose run id has been retried fewer
        than ``attempt_cap`` times on the current head is eligible; when
        ``record_attempts`` is True the attempt count is incremented and the
        run id is returned in ``rerun_run_ids``.
      * An infra-failed required check whose every run id has reached
        ``attempt_cap`` is definitive -- the caller escalates instead of
        retrying again.
      * An infra-failed required check with no parseable GitHub Actions run
        id cannot be auto-retried and is treated as definitive immediately.
    """
    if not required:
        return InfraRerunResult()

    summary = summarize_checks(checks, required)

    # Existing attempts keyed by head SHA.
    attempts_state: dict[str, Any] = dict((pr_state or {}).get("infra_rerun_attempts") or {})
    current_attempts: dict[str, Any]
    if head_sha:
        current_attempts = dict(attempts_state.get(head_sha) or {})
    else:
        current_attempts = {}

    # Clear markers for required checks that are now passing on this head.
    passing: set[str] = set(summary.passed)
    for name in list(current_attempts):
        if name in passing:
            del current_attempts[name]

    # Group raw check runs by name, filtering to required checks only.
    runs_by_name: dict[str, list[dict[str, Any]]] = {}
    if checks:
        for check in checks:
            name = str(check.get("name") or "")
            if name in required:
                runs_by_name.setdefault(name, []).append(check)

    rerun_run_ids: list[int] = []
    definitive_failed: list[str] = []

    for name in summary.infra_failed:
        # Collect the infra-failing Actions run ids for this required check.
        infra_run_ids: set[int] = set()
        for check in runs_by_name.get(name, []):
            if not _is_infra_run(check):
                continue
            run_id: int | None = check.get("runId")
            if not isinstance(run_id, int):
                run_id = _run_id_from_link(check.get("link"))
            if isinstance(run_id, int):
                infra_run_ids.add(run_id)

        if not infra_run_ids:
            # No parseable Actions run id (e.g. external status check) -- cannot
            # auto-rerun, so treat as definitive immediately.
            definitive_failed.append(name)
            continue

        if not record_attempts:
            # Caller has other janitor blockers; do not consume a rerun attempt
            # but treat it as definitive for this pass.
            definitive_failed.append(name)
            continue

        name_attempts: dict[str, int] = dict(current_attempts.get(name) or {})
        eligible_run_ids = {
            run_id for run_id in infra_run_ids if name_attempts.get(str(run_id), 0) < attempt_cap
        }

        if not eligible_run_ids:
            # Every infra-failing run id for this check has reached the
            # attempt cap on this head -- definitive, escalate instead.
            definitive_failed.append(name)
            continue

        rerun_run_ids.extend(sorted(eligible_run_ids))
        for run_id in eligible_run_ids:
            name_attempts[str(run_id)] = name_attempts.get(str(run_id), 0) + 1
        current_attempts[name] = name_attempts

    if head_sha:
        attempts_state = {head_sha: current_attempts}
    else:
        # If we cannot associate attempts with a head, drop them to avoid
        # cross-head contamination.
        attempts_state = {}

    return InfraRerunResult(
        rerun_run_ids=tuple(sorted(set(rerun_run_ids))),
        infra_rerun_attempts=attempts_state,
        definitive_failed=tuple(definitive_failed),
    )
