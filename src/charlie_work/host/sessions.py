"""Host port: live worker / reviewer session counts.

Leaf module (stdlib only at top). Each Real method resolves the underlying
primitive's module attribute at call time (``live_session_count`` /
``fleet_registry`` / ``worker`` / ``worktree``), so a patch on the primitive's
own module still reaches every consumer -- the ``workflow.*`` /
``dispatch_selection.*`` delegate layer the Real used to route through was
deleted in issue #2235.
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
        from .. import live_session_count

        return live_session_count.count_live_sessions(
            sessions_dir, state_file, live_session_count.WORKER_LANE
        )

    def fleet_live_workers(self, fleet_dir_override: str | None) -> tuple[int, list[str]]:
        from .. import fleet_registry

        return fleet_registry.count_fleet_live_sessions(fleet_dir_override)

    def live_reviews(self, reviews_dir: Path, state_file: Path | None = None) -> int:
        from .. import live_session_count

        return live_session_count.count_live_sessions(
            reviews_dir, state_file, live_session_count.REVIEW_LANE
        )

    def fleet_live_reviews(self, fleet_dir_override: str | None) -> tuple[int, list[str]]:
        from .. import fleet_registry

        return fleet_registry.count_fleet_live_reviews(fleet_dir_override)

    def live_issue_numbers(self, sessions_dir: Path) -> set[int]:
        from ..worker import iter_workers

        return {w.issue_number for w in iter_workers(sessions_dir) if w.is_alive()}

    def live_session_pids(self, sessions_dir: Path) -> dict[str, int]:
        from ..worktree import _own_live_session_pids

        return _own_live_session_pids(sessions_dir)
