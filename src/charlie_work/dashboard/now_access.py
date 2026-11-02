"""Tolerant accessors over a status-snapshot ``data`` dict (shared by the Now model)."""

from __future__ import annotations

from typing import Any

from .now_types import RepoRead


def snapshot_data(repo: RepoRead) -> dict[str, Any]:
    return repo.snapshot.data or {}


def dict_list(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = data.get(key)
    return [x for x in raw if isinstance(x, dict)] if isinstance(raw, list) else []


def as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def pos_int(value: Any) -> int | None:
    """A positive int (issue/PR numbers), else ``None`` -- never a fake ``0``."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def label_set(issue: dict[str, Any]) -> set[str]:
    raw = issue.get("labels")
    return {x for x in raw if isinstance(x, str)} if isinstance(raw, list) else set()
