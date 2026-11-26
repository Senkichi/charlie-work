# ruff: noqa: F811  (the imported ``fleet`` fixture is re-bound as a test parameter)
"""Rollup classification: a level file is enough to ignore a kind (issue #3093)."""

from __future__ import annotations

from types import MappingProxyType

import pytest
from _dashboard_rollup_fixtures import (  # noqa: F401  (fleet is a pytest fixture)
    ALPHA,
    NOW,
    fleet,
)

from charlie_work import instrumentation
from charlie_work.dashboard import rollup
from charlie_work.dashboard.rollup_derive import (
    HANDLERS,
    KNOWN_IGNORED,
    SWEEP_SUFFIX,
    is_classified,
)
from charlie_work.instrumentation import (
    _LEVEL_BY_KIND,
    _load_level_registry,
    has_level_entry,
)

_NEW_KIND = "brand_new_leveled_kind"


def test_level_file_kind_without_handler_is_classified_with_no_dashboard_edit(
    fleet, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: adding only ``<kind>.level`` classifies the kind in the rollup."""
    assert _NEW_KIND not in HANDLERS and _NEW_KIND not in KNOWN_IGNORED
    fleet.emit(fleet.alpha, "2026-10-01T10:30:00Z", _NEW_KIND, {"x": 1})
    before = rollup.run_rollup(fleet.sources(), NOW)
    assert {s.source: s for s in before.sources}[ALPHA].unclassified == {_NEW_KIND: 1}

    levels_dir = tmp_path / "event_levels"
    levels_dir.mkdir()
    (levels_dir / f"{_NEW_KIND}.level").write_text("info\n# routine bookkeeping\n")
    patched = {**_LEVEL_BY_KIND, **_load_level_registry(levels_dir)}
    monkeypatch.setattr(instrumentation, "_LEVEL_BY_KIND", MappingProxyType(patched))

    assert is_classified(_NEW_KIND)
    assert is_classified(_NEW_KIND + SWEEP_SUFFIX)
    after = rollup.run_rollup(fleet.sources(), NOW)
    assert after.errors == ()
    assert {s.source: s for s in after.sources}[ALPHA].unclassified == {}


def test_kind_with_no_level_file_handler_or_residual_entry_stays_unclassified() -> None:
    kind = "brand_new_unregistered_kind"
    assert not has_level_entry(kind)
    assert not is_classified(kind)
    assert not is_classified(kind + SWEEP_SUFFIX)


def test_sweep_of_a_handled_kind_does_not_inherit_classification() -> None:
    """A handled kind has a level file too, but its ``_sweep`` needs its own handler."""
    for kind in HANDLERS:
        if kind.endswith(SWEEP_SUFFIX):
            continue
        sweep = kind + SWEEP_SUFFIX
        if sweep not in HANDLERS:
            assert not is_classified(sweep), sweep
    assert "dispatch" in HANDLERS and "dispatch_sweep" not in HANDLERS
    assert not is_classified("dispatch_sweep")


def test_known_ignored_holds_only_kinds_without_a_level_file() -> None:
    with_level = sorted(k for k in KNOWN_IGNORED if k in _LEVEL_BY_KIND)
    assert not with_level, (
        f"drop these KNOWN_IGNORED entries; the level file suffices: {with_level}"
    )
    assert not KNOWN_IGNORED.keys() & HANDLERS.keys()
    assert all(reason.strip() for reason in KNOWN_IGNORED.values())


def test_has_level_entry_is_exact_membership() -> None:
    assert has_level_entry("api_budget_refused")
    assert not has_level_entry("api_budget_refused" + SWEEP_SUFFIX)
    assert not has_level_entry("no_such_kind_anywhere")
