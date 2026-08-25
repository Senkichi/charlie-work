"""Hand-computed expectations for classify_ledger (structural, no allowlist)."""

from __future__ import annotations

from charlie_work.attachment_contracts.ledger import classify_ledger


def test_empty_members_not_a_ledger() -> None:
    assert classify_ledger([]) is False


def test_contiguous_v1_to_v5_is_a_ledger() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v3", "_migrate_v4", "_migrate_v5"]
    # 5/5 match, dominance 100%, diffs all 1 -> strictly increasing, contiguous.
    assert classify_ledger(members) is True


def test_single_skipped_number_still_a_ledger() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v4", "_migrate_v5"]
    # 4/4 match, dominance 100%, diffs 1,2,1 -> gap of 1 skipped number tolerated.
    assert classify_ledger(members) is True


def test_gap_of_two_breaks_ledger() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v5"]
    # diffs 1,3 -> 3 > max step of 2 -> not a ledger.
    assert classify_ledger(members) is False


def test_non_numbered_intruder_dilutes_dominance_below_threshold() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v3", "helper", "other_helper"]
    # 3 matches (>= floor of 3), but dominant count 3 / total 5 = 60% < 80% -> caught.
    assert classify_ledger(members) is False


def test_below_match_floor_even_with_full_dominance() -> None:
    members = ["_migrate_v1", "_migrate_v2", "helper"]
    # Only 2 matches < MIN_MATCHING_MEMBERS(3) -> fails on the floor, not dominance.
    assert classify_ledger(members) is False


def test_duplicate_number_breaks_strict_increase() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v2"]
    # sorted numbers [1, 2, 2] -> diff of 0 is not >= 1 -> not strictly increasing.
    assert classify_ledger(members) is False


def test_dominance_exactly_at_threshold_passes() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v3", "_migrate_v4", "_apply_m1"]
    # dominant count 4 / total 5 = 0.8, not < 0.8 -> passes; diffs 1,1,1 -> contiguous.
    assert classify_ledger(members) is True


def test_different_prefix_dilution_below_threshold() -> None:
    members = ["_migrate_v1", "_migrate_v2", "_migrate_v3", "_apply_m1"]
    # dominant "_migrate_v" count 3 / total 4 = 0.75 < 0.8 -> not a ledger.
    assert classify_ledger(members) is False


def test_names_without_trailing_digits_never_match() -> None:
    members = ["step_one", "step_two", "step_three", "step_four"]
    assert classify_ledger(members) is False
