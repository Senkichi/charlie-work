"""Scoped ``deescalation`` config-section parsing (issues #1314, #1477).

``build_config_from_data`` deliberately parses ONLY the three keys below out
of the ``deescalation`` section rather than routing the whole section through
``_build_section``: the section was previously 100% inert (never passed into
``OrchestratorConfig`` -- always defaulted), so full-section parsing would
silently (a) hard-reject unknown keys for any live config that already has a
``deescalation:`` block with extra/typo'd keys (self-deploy-brick risk) and
(b) flip ``enabled`` / ``interval_minutes`` defaults for configs that set
those keys expecting them to be ignored. Full-section activation is a
separate, explicitly-reviewed change with operator notification; unknown keys
and pre-existing ``enabled``/``interval_minutes`` overrides are silently
ignored, same as before #1314.

Extracted from ``config.py`` (file-size ratchet, #1442) so the over-cap
monolith gains only a call site; ``config.py`` imports the helper back so the
facade surface is unchanged. The error messages are byte-identical to the
inline versions they replaced --
``tests/test_issue_1314_operator_queue_followups.py`` pins them (including
the ``operator_queue_depth_threshold`` unit parenthetical from #1768).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _deescalation_nonneg_int(
    data: Mapping[str, Any], key: str, *, qualifier: str = ""
) -> int | None:
    """Validate one optional non-negative-int knob in the ``deescalation`` section.

    Returns the value (or ``None`` when absent, which callers translate into
    "keep the dataclass default"). ``qualifier`` is the extra clause spliced
    between the key name and the predicate in the error message -- used by
    ``operator_queue_depth_threshold`` to name the unit it counts (#1768).
    ``ConfigError`` is imported lazily to avoid a circular import
    (``config.py`` imports this module; this module needs ``ConfigError``
    from ``config.py``) -- the same pattern
    ``capacity_starvation_escalation.parse_runner_capacity_escalation`` uses.
    """
    from .config import ConfigError

    value = data.get(key)
    prefix = f"config section 'deescalation' key '{key}' {qualifier}"
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConfigError(f"{prefix}must be an int, got {type(value).__name__}")
    if value is not None and value < 0:
        raise ConfigError(f"{prefix}must be >= 0, got {value}")
    return value


def parse_deescalation_overrides(deescalation_data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the scoped ``deescalation`` knobs and return dataclass overrides.

    The returned dict is spliced into ``DeescalationConfig(**...)`` by
    ``build_config_from_data``: only keys the operator actually set appear,
    so every other field keeps its dataclass default.
    """
    oqr_interval = _deescalation_nonneg_int(
        deescalation_data, "operator_queue_review_interval_minutes"
    )
    oq_threshold = _deescalation_nonneg_int(
        deescalation_data,
        "operator_queue_depth_threshold",
        qualifier="(blocked-ready-issue count, not root-issue count -- see issue #1768) ",
    )
    irr_window = _deescalation_nonneg_int(
        deescalation_data, "identical_reason_recurrence_window_minutes"
    )
    overrides: dict[str, Any] = {}
    if oqr_interval is not None:
        overrides["operator_queue_review_interval_minutes"] = oqr_interval
    if oq_threshold is not None:
        overrides["operator_queue_depth_threshold"] = oq_threshold
    if irr_window is not None:
        overrides["identical_reason_recurrence_window_minutes"] = irr_window
    return overrides
