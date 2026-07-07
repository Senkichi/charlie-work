from __future__ import annotations

from typing import Any


def emit_digest(notify_config: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Emit a consolidated attention digest for a fleet pass.

    This is a stub for #166 integration. When #166 lands, this will be replaced
    with the actual notifier implementation that sends notifications via the
    configured channel (email, Slack, etc.).

    Args:
        notify_config: The notify configuration section from GlobalConfig.
        events: A list of attention events aggregated from all repos in the fleet pass.

    Returns:
        A digest dict with metadata about the emitted notification.
    """
    # Stub implementation for #170 (will be replaced by #166)
    return {
        "events": events,
        "count": len(events),
        "emitted": False,  # Will be True when #166 lands
        "stub": True,  # Flag to indicate this is the stub
    }
