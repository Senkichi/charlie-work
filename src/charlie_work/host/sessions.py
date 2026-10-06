"""Host port: live worker / reviewer session counts.

Leaf module (stdlib only at top). Each Real method late-binds to the exact
attribute its consumer reached before the port existed, so every existing
patch (``workflow._count_live_sessions``, ``workflow.count_fleet_live_sessions``,
``workflow.count_fleet_live_reviews``, ``dispatch_selection._count_live_reviews``)
keeps intercepting. The module-level ``count_fleet_live_sessions`` below is
what ``workflow.count_fleet_live_sessions`` is bound to -- the end of that
late-binding chain (issue #2230).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class SessionCounter(Protocol):
    def live_workers(self, sessions_dir: Path, state_file: Path | None = None) -> int: ...

    def fleet_live_workers(self, fleet_dir_override: str | None) -> tuple[int, list[str]]: ...

    def live_reviews(self, reviews_dir: Path, state_file: Path | None = None) -> int: ...

    def fleet_live_reviews(self, fleet_dir_override: str | None) -> tuple[int, list[str]]: ...


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


def count_fleet_live_sessions(fleet_dir_override: str | None) -> tuple[int, list[str]]:
    """Late-binding facade over ``fleet_registry.count_fleet_live_sessions``.

    ``charlie_work.workflow`` re-exports this under the same name -- the
    attribute ``RealSessionCounter.fleet_live_workers`` resolves at call
    time. A plain ``from fleet_registry import ...`` re-export on workflow
    would freeze fleet_registry's function object at import time, so a
    patch against ``fleet_registry.count_fleet_live_sessions`` -- the
    attribute the supervise call site used to reach -- would stop biting.
    Resolving the attribute at call time keeps both patch surfaces live:
    patching ``workflow.count_fleet_live_sessions`` intercepts the port's
    lookup, patching ``fleet_registry.count_fleet_live_sessions``
    intercepts inside this body. Lives here rather than in ``workflow.py``
    because that file is over the per-module size cap (issue #1442
    ratchet); moved here in the #2230 rework.
    """
    from .. import fleet_registry

    return fleet_registry.count_fleet_live_sessions(fleet_dir_override)
