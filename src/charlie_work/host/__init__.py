"""Host ports: seams onto what the host provides (clock, process liveness; more per slice).

Leaf package: stdlib imports only at module top, so it cannot create import
cycles. ``current()`` is the single registry read by both ``OrchestratorApp``
and non-app modules; tests swap it through the ``fake_host`` fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock, RealClock
from .liveness import ProcessProbe, RealProcessProbe


@dataclass(frozen=True)
class HostPorts:
    clock: Clock
    probe: ProcessProbe


REAL = HostPorts(clock=RealClock(), probe=RealProcessProbe())

_ACTIVE: HostPorts = REAL


def current() -> HostPorts:
    return _ACTIVE
