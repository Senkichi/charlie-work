from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckSummary:
    required: tuple[str, ...]
    passed: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...]
    infra_failed: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.pending and not self.failed and not self.missing and not self.infra_failed


def summarize_checks(checks: list[dict[str, Any]], required: tuple[str, ...]) -> CheckSummary:
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
                # unless it's due to a superseding push (which we can't detect from this data alone)
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
    )
