"""ci-fleet's host kinds (host I/O, leak drain) carry the levels its design assigns.

ci-fleet logs through this package's sink, and ``test_event_kind_registry_exhaustive``
fails on any of its kinds without an entry. This file pins the *level* of each:
the eight that need an operator's eye are warnings, the rest are info.
"""

from __future__ import annotations

import pytest

from charlie_work.instrumentation import _LEVEL_BY_KIND

WARNING = (
    "host_reboot_due",
    "host_leak_warn",
    "host_io_fallback",
    "host_io_hold",
    "host_io_exclusions_applied",
    "host_io_exclusions_failed",
    "host_io_exclusions_pending",
    "host_io_verify_failed",
)
INFO = (
    "host_io_unprovisioned",
    "host_io_invalid",
    "host_io_converged",
    "host_io_reverted",
    "host_io_converge_failed",
    "host_io_rollout_complete",
    "host_io_finalize_requested",
    "host_io_ab_result",
    "host_drain_started",
    "host_drained",
    "host_drain_cleared",
    "host_probe_stale",
)


@pytest.mark.parametrize("kind", WARNING)
def test_ci_fleet_host_kind_is_a_warning(kind: str) -> None:
    assert _LEVEL_BY_KIND.get(kind) == "warning"


@pytest.mark.parametrize("kind", INFO)
def test_ci_fleet_host_kind_is_info(kind: str) -> None:
    assert _LEVEL_BY_KIND.get(kind) == "info"
