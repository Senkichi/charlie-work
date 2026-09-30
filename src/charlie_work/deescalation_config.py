"""Scoped ``deescalation`` config-section projection (issues #1314, #1477).

``build_config_from_data`` deliberately validates ONLY the keys ``DeescalationConfig``
annotates with rules out of the ``deescalation`` section rather than routing the whole
section through ``validate_section``: the section was previously 100% inert (never
passed into ``OrchestratorConfig`` -- always defaulted), so full-section parsing would
silently (a) hard-reject unknown keys for any live config that already has a
``deescalation:`` block with extra/typo'd keys (self-deploy-brick risk) and
(b) flip ``enabled`` / ``interval_minutes`` defaults for configs that set
those keys expecting them to be ignored. Full-section activation is a
separate, explicitly-reviewed change with operator notification; unknown keys
and pre-existing ``enabled``/``interval_minutes`` overrides are silently
ignored, same as before #1314.

The scope is derived, not listed: a ``DeescalationConfig`` field is in scope
exactly when it carries field metadata (``Annotated[..., Typed, ...]``), so
widening the activated surface is an explicit, reviewable annotation on the
field and never a second list to keep in sync.

Extracted from ``config.py`` (file-size ratchet, #1442).
``tests/test_issue_1314_operator_queue_followups.py`` pins the behaviour
(including the ``operator_queue_depth_threshold`` unit note from #1768).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .config_validation import field_specs


def project_scoped_keys(cls: type, section_data: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of ``section_data`` naming a field of ``cls`` that carries metadata.

    Everything else in the raw section (unknown keys, un-annotated knobs) is
    dropped, which is what keeps the pre-#1314 "silently ignored" contract.
    """
    scoped = {spec.name for spec in field_specs(cls) if spec.markers}
    return {k: v for k, v in section_data.items() if k in scoped}
