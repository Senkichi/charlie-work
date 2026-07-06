from __future__ import annotations

import os
import sys
from pathlib import Path


def fleet_dir(*, override: str | None = None) -> Path:
    """Return the user-level fleet directory for charlie-work.

    Platform-specific defaults:
    - Windows (win32): %LOCALAPPDATA%\\charlie-work\\
    - POSIX: ${XDG_STATE_HOME:-~/.local/state}/charlie-work/

    The override parameter (or CHARLIE_WORK_FLEET_DIR env var) allows
    test isolation without hardcoding platform-specific paths in fixtures.

    Args:
        override: Optional path string to use instead of the platform default.
                  If None, checks CHARLIE_WORK_FLEET_DIR env var, then uses
                  the platform default.

    Returns:
        Path to the fleet directory.
    """
    if override is not None:
        return Path(override)

    env_override = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    if env_override:
        return Path(env_override)

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))

    return base / "charlie-work"
