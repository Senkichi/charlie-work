"""TIS-CW-6: ``model:<tier>`` label parsing, model families and tier selection (pure)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from charlie_work.config import LabelConfig, load_config
from charlie_work.model_tier import (
    NO_ENTRY,
    RESTRICTED,
    model_in_tier,
    model_tiers,
    select_tier_entry,
)
from charlie_work.role_chain import RoleEntry

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
SONNET = RoleEntry("claude-code", "claude-sonnet-5-5")
SWE = RoleEntry("devin-shell", "swe-2")
OPUS = RoleEntry("claude-code", "claude-opus-5-5")
OPUS_ALIAS = RoleEntry("claude-code", "opus")
CHAIN = (SONNET, SWE, OPUS, OPUS_ALIAS)


@pytest.mark.parametrize(
    ("labels", "prefix", "expected"),
    [
        (["automated-ready", "model:opus"], "model:", ("opus",)),
        (["Model: Opus ", "model:opus"], "model:", ("opus",)),
        (["model:sonnet", "model:opus"], "model:", ("opus", "sonnet")),
        (["model:", "priority:critical"], "model:", ()),
        (["model:opus"], "", ()),
        (["tier/opus"], "tier/", ("opus",)),
    ],
)
def test_model_tiers(labels: list[str], prefix: str, expected: tuple[str, ...]) -> None:
    assert model_tiers(labels, prefix) == expected


@pytest.mark.parametrize(
    ("model", "tier", "expected"),
    [
        ("claude-opus-5-5", "opus", True),
        ("claude-opus-5-5", "Opus", True),
        ("claude-opus-5-5", "claude", True),
        ("claude-sonnet-5-5", "opus", False),
        ("opus", "opus", True),
        ("swe-2", "swe", True),
        # A family is a whole token, never a substring.
        ("gpt-5-codex", "code", False),
        # An empty id is the harness default: it belongs to no family.
        ("", "opus", False),
        ("claude-opus-5-5", "", False),
    ],
)
def test_model_in_tier(model: str, tier: str, expected: bool) -> None:
    assert model_in_tier(model, tier) is expected


def test_first_entry_of_the_family_is_selected() -> None:
    selection, reason = select_tier_entry(CHAIN, "opus", {}, NOW)
    assert reason is None
    assert selection is not None
    assert (selection.entry, selection.index, selection.is_fallback) == (OPUS, 2, True)
    assert selection.chain == CHAIN


def test_a_restricted_entry_moves_to_the_next_of_its_family() -> None:
    ledger = {OPUS.key: NOW + timedelta(hours=1)}
    selection, reason = select_tier_entry(CHAIN, "opus", ledger, NOW)
    assert reason is None
    assert selection is not None
    assert (selection.entry, selection.index) == (OPUS_ALIAS, 3)


def test_an_expired_restriction_restricts_nothing() -> None:
    selection, _ = select_tier_entry(CHAIN, "opus", {OPUS.key: NOW}, NOW)
    assert selection is not None
    assert selection.entry == OPUS


def test_every_family_entry_restricted_is_a_fallback() -> None:
    later = NOW + timedelta(hours=1)
    ledger = {OPUS.key: later, OPUS_ALIAS.key: later}
    assert select_tier_entry(CHAIN, "opus", ledger, NOW) == (None, RESTRICTED)


def test_a_family_with_no_entry_is_a_fallback() -> None:
    assert select_tier_entry(CHAIN, "haiku", {}, NOW) == (None, NO_ENTRY)


def test_the_primary_family_is_index_zero() -> None:
    selection, _ = select_tier_entry(CHAIN, "sonnet", {}, NOW)
    assert selection is not None
    assert (selection.index, selection.is_fallback) == (0, False)


def test_the_prefix_defaults_on_and_empty_turns_it_off(tmp_path: Path) -> None:
    assert LabelConfig().model_tier_prefix == "model:"
    config_file = tmp_path / "orchestrator.config.yaml"
    config_file.write_text('labels:\n  model_tier_prefix: ""\n', encoding="utf-8")
    assert load_config(config_file).labels.model_tier_prefix == ""
