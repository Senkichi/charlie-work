"""Shared value coercion and row helpers for the rollup derivation handlers."""

from __future__ import annotations

import re
from typing import Any

Row = tuple[str, dict[str, Any]]


def _int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _flag(value: Any) -> int | None:
    return None if value is None else int(bool(value))


def _ints(values: Any) -> list[int]:
    if not isinstance(values, list):
        return []
    return [n for n in (_int(v) for v in values) if n is not None]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def reason_group(reason: Any) -> str | None:
    """Group a ``review_verdict_missed`` reason: tokens stay, free text is prefix-grouped.

    ``died_mid_session`` / ``launch_failed`` are enum-like and kept verbatim; anything
    else is lowercased, cut at the first ``:`` and reduced to its first two words with
    numbers masked, so ``"PR #2087 is MERGED"`` and ``"PR #2090 is MERGED"`` collapse.
    """
    if not isinstance(reason, str) or not reason.strip():
        return None
    if re.fullmatch(r"[a-z0-9_]+", reason):
        return reason
    head = re.sub(r"#?\d+", "#", reason.split(":", 1)[0].strip().lower())
    return " ".join(head.split()[:2]) or None


def _refs(ev: dict) -> tuple[Any, Any]:
    p = ev["payload"]
    return p.get("issue_number", ev["issue_number"]), p.get("pr_number", ev["pr_number"])


def _milestone(ev: dict, milestone: str, issue: Any, pr: Any, approx: bool = False) -> Row:
    return (
        "issue_milestones",
        {
            "issue": _int(issue),
            "pr": _int(pr),
            "milestone": milestone,
            "event_kind": ev["kind"],
            "approx": int(approx),
        },
    )
