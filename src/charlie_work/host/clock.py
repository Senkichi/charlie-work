"""Host port: wall-clock and monotonic time.

Leaf module (stdlib only). ``format_utc`` is the single definition of the
orchestrator's UTC timestamp format; ``state.utc_now`` renders through it.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Timezone-aware UTC wall-clock time."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds, for measuring intervals."""
        ...


class RealClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


def format_utc(moment: datetime) -> str:
    """Render ``moment`` as the second-resolution ``...Z`` stamp used in state."""
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
