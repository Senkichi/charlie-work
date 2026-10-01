"""Host port: process liveness.

Leaf module (stdlib only at top). The Real late-binds to
``process_utils.is_pid_alive`` / ``get_process_start_time`` at call time, so a
patch of either primitive still reaches every consumer.
"""

from __future__ import annotations

from typing import Protocol


class ProcessProbe(Protocol):
    def is_alive(self, pid: int | None, start_time: float | None = None) -> bool:
        """True when ``pid`` is a live process matching ``start_time`` (if given)."""
        ...

    def start_time(self, pid: int) -> float | None:
        """Process start-time fingerprint for ``pid``, or None if unknown."""
        ...


class RealProcessProbe:
    def is_alive(self, pid: int | None, start_time: float | None = None) -> bool:
        if pid is None or pid <= 0:
            return False
        from .. import process_utils as pu

        return pu.is_pid_alive(pid, start_time)

    def start_time(self, pid: int) -> float | None:
        from .. import process_utils as pu

        return pu.get_process_start_time(pid)
