"""Host ports: seams onto what the host provides (clock, process liveness, session counts, worker/reviewer launch).

Leaf package: stdlib imports only at module top, so it cannot create import
cycles. ``current()`` is the single registry read by both ``OrchestratorApp``
and non-app modules; tests swap it through the ``fake_host`` fixture.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock, RealClock
from .launch import (
    DecideLauncher,
    ForemanPassLauncher,
    RealDecideLauncher,
    RealForemanPassLauncher,
    RealReviewLauncher,
    RealWorkerLauncher,
    ReviewLauncher,
    WorkerLauncher,
)
from .liveness import ProcessProbe, RealProcessProbe
from .sessions import RealSessionCounter, SessionCounter


@dataclass(frozen=True)
class HostPorts:
    clock: Clock
    probe: ProcessProbe
    sessions: SessionCounter
    launch: ReviewLauncher
    worker_launch: WorkerLauncher
    decide_launch: DecideLauncher
    foreman_pass_launch: ForemanPassLauncher


REAL = HostPorts(
    clock=RealClock(),
    probe=RealProcessProbe(),
    sessions=RealSessionCounter(),
    launch=RealReviewLauncher(),
    worker_launch=RealWorkerLauncher(),
    decide_launch=RealDecideLauncher(),
    foreman_pass_launch=RealForemanPassLauncher(),
)

_ACTIVE: HostPorts = REAL


def current() -> HostPorts:
    return _ACTIVE
