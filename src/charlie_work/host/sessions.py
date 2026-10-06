"""Host port: live worker / reviewer session counts.

Leaf module (stdlib only at top). Each Real method late-binds to the exact
attribute its consumer reached before the port existed, so every existing
patch (``workflow._count_live_sessions``, ``workflow.count_fleet_live_sessions``,
``workflow.count_fleet_live_reviews``, ``dispatch_selection._count_live_reviews``,
``worker.iter_workers``, ``worktree._own_live_session_pids``) keeps
intercepting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SessionCounter(Protocol):
    def live_workers(self, sessions_dir: Path, state_file: Path | None = None) -> int: ...

    def fleet_live_workers(self, fleet_dir_override: str | None) -> tuple[int, list[str]]: ...

    def live_reviews(self, reviews_dir: Path, state_file: Path | None = None) -> int: ...

    def fleet_live_reviews(self, fleet_dir_override: str | None) -> tuple[int, list[str]]: ...

    def live_issue_numbers(self, sessions_dir: Path) -> set[int]: ...

    def live_session_pids(self, sessions_dir: Path) -> dict[str, int]: ...


class RealSessionCounter:
    def live_workers(self, sessions_dir: Path, state_file: Path | None = None) -> int:
        from .. import workflow

        return workflow._count_live_sessions(sessions_dir, state_file)

    def fleet_live_workers(self, fleet_dir_override: str | None) -> tuple[int, list[str]]:
        from .. import workflow

        return workflow.count_fleet_live_sessions(fleet_dir_override)

    def live_reviews(self, reviews_dir: Path, state_file: Path | None = None) -> int:
        from .. import dispatch_selection

        return dispatch_selection._count_live_reviews(reviews_dir, state_file)

    def fleet_live_reviews(self, fleet_dir_override: str | None) -> tuple[int, list[str]]:
        from .. import workflow

        return workflow.count_fleet_live_reviews(fleet_dir_override)

    def live_issue_numbers(self, sessions_dir: Path) -> set[int]:
        from ..worker import iter_workers

        return {w.issue_number for w in iter_workers(sessions_dir) if w.is_alive()}

    def live_session_pids(self, sessions_dir: Path) -> dict[str, int]:
        from ..worktree import _own_live_session_pids

        return _own_live_session_pids(sessions_dir)
