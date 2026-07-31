from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CheckSummary:
    required: tuple[str, ...]
    passed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...]
    infra_failed: tuple[str, ...]
    unavailable: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return (
            not self.pending
            and not self.failed
            and not self.missing
            and not self.infra_failed
            and not self.unavailable
        )


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

        for check in runs:
            state = str(check.get("state") or "").upper()
            bucket = str(check.get("bucket") or "").lower()

            if state == "SUCCESS" or bucket == "pass":
                # This run passed - continue checking other runs
                continue
            elif state in {"PENDING", "QUEUED", "IN_PROGRESS", "REQUESTED"} or bucket == "pending":
                name_pending = True
            elif not state and not bucket:
                # Null/empty state and bucket means the check-run hasn't populated yet - classify as pending
                name_pending = True
            elif state == "FAILURE":
                # FAILURE state indicates code failure - highest priority
                name_failed = True
            elif state == "CANCELLED":
                # CANCELLED state indicates infrastructure failure (e.g., billing lapse, runner death)
                # Note: Signal-3's head-unchanged qualifier is omitted here because checks are evaluated
                # at the current head, so cancellations in scope are genuine infrastructure failures
                name_infra_failed = True
            elif state == "INFRA_FAILURE":
                # INFRA_FAILURE is a marker state set by the GitHub adapter enrichment layer
                # to indicate jobs with zero steps or billing annotations (signals 1 and 2 from #210)
                name_infra_failed = True
            elif state == "TIMED_OUT":
                # A genuine step/job timeout is an infrastructure condition, not a
                # code failure (mirrors CANCELLED/INFRA_FAILURE just above). On
                # this repo's self-hosted runners, a `timeout-minutes` kill is
                # observed to report CANCELLED rather than TIMED_OUT (issue #841),
                # but TIMED_OUT is a documented GitHub Actions conclusion value
                # and must not fall through to the code-failure catch-all below.
                name_infra_failed = True
            else:
                # Any other failure state.
                name_failed = True

        if name_failed:
            failed.append(name)
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
    """Return True if a single check run represents a code failure (not pending/infra)."""
    state = str(check.get("state") or "").upper()
    bucket = str(check.get("bucket") or "").lower()

    if state == "SUCCESS" or bucket == "pass":
        return False
    if state in {"PENDING", "QUEUED", "IN_PROGRESS", "REQUESTED"} or bucket == "pending":
        return False
    if not state and not bucket:
        return False
    if state in {"CANCELLED", "INFRA_FAILURE", "TIMED_OUT"}:
        return False
    return True


def _is_infra_run(check: dict[str, Any]) -> bool:
    """Return True if a single check run represents an infrastructure failure.

    The counterpart to `_is_failing_run`: CANCELLED, INFRA_FAILURE, and
    TIMED_OUT are infra conditions, not code failures (see the matching
    branches in `summarize_checks`'s per-run loop above).
    """
    state = str(check.get("state") or "").upper()
    return state in {"CANCELLED", "INFRA_FAILURE", "TIMED_OUT"}


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
