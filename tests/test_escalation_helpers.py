"""Direct unit tests for escalation.py's pure (edge, reason_class) -> label
helpers (issue #1266: mechanical-escalation routing to an operator queue).

Imports are deliberately from ``charlie_work.escalation`` directly, never
``charlie_work.workflow`` -- ``tests/test_escalation_split.py``'s AC5 scan
(``test_facade_reexports_every_name_consumers_reach_through_workflow``) only
matches references reached *through* ``charlie_work.workflow`` (an
``ImportFrom`` of that module, a ``workflow.<name>``/``wf.<name>`` attribute
access, or the string-dotted monkeypatch form). A direct
``charlie_work.escalation`` import is invisible to that scan by construction,
so this file cannot perturb AC5's derived "who needs a facade re-export"
census.
"""

from __future__ import annotations

import pytest

from charlie_work.config import LabelConfig
from charlie_work.escalation import (
    _escalation_edge,
    _escalation_label,
    _repair_reason_class,
)

# ---------------------------------------------------------------------------
# _escalation_edge: full (edge, reason_class) matrix
# ---------------------------------------------------------------------------

# The two edges with a mechanical counterpart in _MECHANICAL_ESCALATION_EDGES,
# plus "blocked" -- the judgment-only edge with deliberately no counterpart
# (see _escalation_edge's own docstring: "there is deliberately no 'blocked
# but mechanical' cell in this mapping").
_EDGE_MATRIX = [
    ("escalated", "mechanical", "operator_queued"),
    ("escalated", "judgment", "escalated"),
    ("redispatch_escalated", "mechanical", "redispatch_operator_queued"),
    ("redispatch_escalated", "judgment", "redispatch_escalated"),
    ("blocked", "mechanical", "blocked"),
    ("blocked", "judgment", "blocked"),
]


@pytest.mark.parametrize("edge,reason_class,expected", _EDGE_MATRIX)
def test_escalation_edge_matrix(edge: str, reason_class: str, expected: str) -> None:
    assert _escalation_edge(edge, reason_class) == expected


def test_escalation_edge_unrecognized_edge_passes_through_for_judgment() -> None:
    """An edge with no mechanical counterpart at all (not one of the two keys
    in _MECHANICAL_ESCALATION_EDGES) is a no-op passthrough for judgment,
    same as "blocked" above -- this isn't a special case in the
    implementation, just a different edge name exercising the same identity
    branch."""
    assert _escalation_edge("some_other_edge", "judgment") == "some_other_edge"


def test_escalation_edge_unrecognized_edge_passes_through_for_mechanical() -> None:
    """The .get(edge, edge) fallback in _MECHANICAL_ESCALATION_EDGES means an
    edge absent from that table passes through unchanged even for
    reason_class="mechanical" -- there is no implicit third mechanical
    counterpart invented for an edge nobody registered one for."""
    assert _escalation_edge("some_other_edge", "mechanical") == "some_other_edge"


def test_escalation_edge_invalid_reason_class_raises_value_error() -> None:
    """escalation_reason_class's fail-loud contract: an unrecognized
    reason_class must raise, not silently fall through to the identity
    return, so a typo at a call site is caught immediately rather than
    producing an escalation the de-escalation sweep can never recognize."""
    with pytest.raises(ValueError, match=r"invalid escalation reason_class: 'bogus'"):
        _escalation_edge("escalated", "bogus")


def test_escalation_edge_invalid_reason_class_error_message_matches_state_py() -> None:
    """Pin the exact message format state.py's escalation_reason_class raises,
    since callers (or an operator reading a traceback) depend on the
    !r-repr'd value being present verbatim."""
    with pytest.raises(ValueError) as excinfo:
        _escalation_edge("escalated", "not-a-real-class")
    assert str(excinfo.value) == "invalid escalation reason_class: 'not-a-real-class'"


# ---------------------------------------------------------------------------
# _escalation_label: edge -> the label a labels.py transition actually adds
# ---------------------------------------------------------------------------


def test_escalation_label_escalated_is_human_needed() -> None:
    labels = LabelConfig()
    assert _escalation_label(labels, "escalated") == labels.human_needed


def test_escalation_label_operator_queued_is_operator_queue() -> None:
    labels = LabelConfig()
    assert _escalation_label(labels, "operator_queued") == labels.operator_queue


def test_escalation_label_redispatch_escalated_is_human_needed() -> None:
    labels = LabelConfig()
    assert _escalation_label(labels, "redispatch_escalated") == labels.human_needed


def test_escalation_label_redispatch_operator_queued_is_operator_queue() -> None:
    labels = LabelConfig()
    assert _escalation_label(labels, "redispatch_operator_queued") == labels.operator_queue


def test_escalation_label_blocked_is_human_needed() -> None:
    labels = LabelConfig()
    assert _escalation_label(labels, "blocked") == labels.human_needed


# ---------------------------------------------------------------------------
# Composition: _escalation_label(labels, _escalation_edge(edge, reason_class))
# is the actual (status, reason_class) -> label mapping every call site
# resolves through end to end.
# ---------------------------------------------------------------------------

_LABEL_MATRIX = [
    ("escalated", "mechanical", "operator_queue"),
    ("escalated", "judgment", "human_needed"),
    ("redispatch_escalated", "mechanical", "operator_queue"),
    ("redispatch_escalated", "judgment", "human_needed"),
    ("blocked", "mechanical", "human_needed"),
    ("blocked", "judgment", "human_needed"),
]


@pytest.mark.parametrize("edge,reason_class,label_attr", _LABEL_MATRIX)
def test_escalation_edge_and_label_compose_to_the_expected_label(
    edge: str, reason_class: str, label_attr: str
) -> None:
    labels = LabelConfig()
    resolved_edge = _escalation_edge(edge, reason_class)
    assert _escalation_label(labels, resolved_edge) == getattr(labels, label_attr)


# ---------------------------------------------------------------------------
# _repair_reason_class: label-repair target reason_class for an escalated
# issue's state entry
# ---------------------------------------------------------------------------


def test_repair_reason_class_none_entry_defaults_to_judgment() -> None:
    assert _repair_reason_class(None) == "judgment"


def test_repair_reason_class_missing_reason_class_defaults_to_judgment() -> None:
    """A pre-#797 escalation (predating the reason_class field) must not be
    silently assumed mechanical."""
    assert _repair_reason_class({}) == "judgment"


def test_repair_reason_class_invalid_stored_value_defaults_to_judgment() -> None:
    """A legacy/corrupt reason_class value falls back to judgment -- the same
    fail-closed direction as a missing one, never assumed mechanical."""
    assert _repair_reason_class({"reason_class": "not-a-real-class"}) == "judgment"


def test_repair_reason_class_mechanical_is_preserved() -> None:
    assert _repair_reason_class({"reason_class": "mechanical"}) == "mechanical"


def test_repair_reason_class_judgment_is_preserved() -> None:
    assert _repair_reason_class({"reason_class": "judgment"}) == "judgment"


def test_repair_reason_class_deescalation_cap_notified_overrides_to_judgment() -> None:
    """Issue #1266: once _deescalate_mechanical_issue's cap-exhaustion branch
    has already moved an issue off operator_queue onto human_needed, both
    label-repair consumers must treat it as judgment going forward even
    though the stored reason_class is still "mechanical" -- otherwise a
    repair sweep would re-apply operator_queue to an issue the auto-clear
    sweep has already given up on."""
    entry = {
        "reason_class": "mechanical",
        "deescalation_cap_notified_at": "2026-08-01T00:00:00+00:00",
    }
    assert _repair_reason_class(entry) == "judgment"
