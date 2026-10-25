"""The launchd half of ``fleet_dispatch.probe_fleet_watchdog`` (spec C2, CW-3).

On the Mac the fleet supervisor is restarted by a launchd job (the hosts repo
installs it as a LaunchDaemon) instead of the ``charlie-fleet-pass`` scheduled
task. The job label is host provisioning, not charlie-work config, so the
launcher exports it as ``CHARLIE_WORK_LAUNCHD_LABEL`` (for example
``com.senkichi.hosts.charlie-fleet``). The result keeps ``WatchdogProbe``'s
tri-state: True armed, False disabled, None unknown (label unset, job not
loaded, or ``launchctl`` failed).
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

from .subprocess_runner import RunResult, run_captured

LAUNCHD_LABEL_ENV = "CHARLIE_WORK_LAUNCHD_LABEL"
LAUNCHD_DOMAIN = "system"


def probe_launchd_job(
    *,
    run_command: Callable[..., RunResult] = run_captured,
    timeout_seconds: int = 15,
) -> tuple[bool | None, str]:
    """Return ``(armed, detail)`` for the launchd job named by ``CHARLIE_WORK_LAUNCHD_LABEL``."""
    label = os.environ.get(LAUNCHD_LABEL_ENV, "").strip()
    if not label:
        return None, f"{LAUNCHD_LABEL_ENV} is unset; launchd job unknown"
    target = f"{LAUNCHD_DOMAIN}/{label}"
    loaded = run_command(["launchctl", "print", target], cwd=None, timeout_seconds=timeout_seconds)
    if not loaded.ok:
        return None, f"launchd job {target} is not loaded (launchctl print failed)"
    listing = run_command(
        ["launchctl", "print-disabled", LAUNCHD_DOMAIN],
        cwd=None,
        timeout_seconds=timeout_seconds,
    )
    if not listing.ok:
        return None, f"launchctl print-disabled {LAUNCHD_DOMAIN} failed"
    match = re.search(rf'"{re.escape(label)}"\s*=>\s*(\w+)', listing.stdout)
    if match and match.group(1).lower() in ("disabled", "true"):
        return False, f"launchd job {target} is disabled"
    return True, f"launchd job {target} is loaded and enabled"
