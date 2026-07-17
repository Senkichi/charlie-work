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
            else:
                # Any other failure state (TIMED_OUT, etc.)
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
    if state == "CANCELLED" or state == "INFRA_FAILURE":
        return False
    return True
