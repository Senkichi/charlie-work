"""Per-repo main-sync counters: skip dedupe, post_sync back-off and alarm latches.

One JSON file per repo under ``<fleet_dir>/main-sync/``, written atomically. A missing,
unreadable or mistyped file loads as a fresh state, so a bad file costs at most one
repeated event, never a crash (docs/specs/main-sync.md, "Observability and alarms").
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

STATE_DIRNAME = "main-sync"
_OPTIONAL_TIMES = frozenset({"defer_since", "next_attempt_at"})


@dataclass(frozen=True)
class RepoSyncState:
    """``last_skip_key`` dedupes skip events per ``reason:local:origin``.

    ``failure_streak``, ``defer_*`` and ``next_attempt_at`` drive the post_sync
    back-off; each ``*_alarmed`` flag makes its alarm fire once until the next
    success clears the whole state. ``budget_skipped`` puts the repo first next pass.
    """

    last_skip_key: str = ""
    failure_streak: int = 0
    failure_alarmed: bool = False
    defer_since: float | None = None
    defer_count: int = 0
    next_attempt_at: float | None = None
    defer_alarmed: bool = False
    history_alarmed: bool = False
    rollback_alarmed: bool = False
    budget_skipped: bool = False

    @property
    def alarmed(self) -> bool:
        return (
            self.failure_alarmed
            or self.defer_alarmed
            or self.history_alarmed
            or self.rollback_alarmed
        )


def state_path(fleet_dir: Path, repo: str) -> Path:
    return fleet_dir / STATE_DIRNAME / f"{repo.replace('/', '__')}.json"


def _valid(name: str, value: object, default: object) -> bool:
    if name in _OPTIONAL_TIMES:
        return value is None or (isinstance(value, int | float) and not isinstance(value, bool))
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, type(default))


def load_state(fleet_dir: Path, repo: str) -> RepoSyncState:
    try:
        raw = json.loads(state_path(fleet_dir, repo).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return RepoSyncState()
    if not isinstance(raw, dict):
        return RepoSyncState()
    default = RepoSyncState()
    values = {f.name: raw.get(f.name, getattr(default, f.name)) for f in fields(RepoSyncState)}
    if not all(_valid(name, value, getattr(default, name)) for name, value in values.items()):
        return RepoSyncState()
    return RepoSyncState(**values)


def save_state(fleet_dir: Path, repo: str, state: RepoSyncState) -> None:
    path = state_path(fleet_dir, repo)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, asdict(state))
    except OSError as exc:
        logger.warning("main-sync: could not save state for %s: %s", repo, exc)
