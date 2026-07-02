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

    @property
    def ready(self) -> bool:
        return not self.pending and not self.failed and not self.missing


def summarize_checks(checks: list[dict[str, Any]], required: tuple[str, ...]) -> CheckSummary:
    by_name = {str(check.get("name") or ""): check for check in checks}
    passed: list[str] = []
    pending: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    for name in required:
        check = by_name.get(name)
        if check is None:
            missing.append(name)
            continue
        state = str(check.get("state") or "").upper()
        conclusion = str(check.get("conclusion") or "").upper()
        bucket = str(check.get("bucket") or "").lower()
        if state == "SUCCESS" or conclusion == "SUCCESS" or bucket == "pass":
            passed.append(name)
        elif state in {"PENDING", "QUEUED", "IN_PROGRESS", "REQUESTED"} or bucket == "pending":
            pending.append(name)
        else:
            failed.append(name)
    return CheckSummary(
        required=required,
        passed=tuple(passed),
        pending=tuple(pending),
        failed=tuple(failed),
        missing=tuple(missing),
    )
