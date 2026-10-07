"""``dashboard:`` config section (ADR-0008, ADR-0007 field-metadata validation).

One dashboard process serves the whole machine, so the section is host-wide-only
(``OrchestratorConfig.dashboard`` carries ``HostWideOnly``): a per-repo config
that sets it is rejected by ``global_config.load_layered_config``. It lives here
rather than in ``config.py`` to keep that over-cap monolith from growing; this
module must not import ``config`` (it imports only ``config_validation``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from ..config_validation import AtLeastOne, InRange, NonEmpty, Typed

DASHBOARD_SECTION = "dashboard"


@dataclass(frozen=True)
class DashboardConfig:
    """Knobs for ``charlie dashboard`` (read-only; never writes fleet state)."""

    # Kill switch: new features ship ON; False makes ``charlie dashboard`` refuse to serve.
    enabled: Annotated[bool, Typed] = True
    # Loopback by default: the dashboard has no auth, so it must not listen publicly
    # unless the operator opts in explicitly.
    host: Annotated[str, Typed, NonEmpty] = "127.0.0.1"
    port: Annotated[int, Typed, InRange(1, 65535)] = 8765
    # Browser-side refresh cadence.
    poll_interval_seconds: Annotated[int, Typed, AtLeastOne] = 20
    # Cadence of the read-only collector that samples the fleet sources.
    collector_interval_seconds: Annotated[int, Typed, AtLeastOne] = 30
    # Cadence of the rollup pass that rebuilds the derived dashboard.db tables.
    rollup_interval_seconds: Annotated[int, Typed, AtLeastOne] = 120
