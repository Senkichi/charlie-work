"""Issue #2086: role-chain config, pure selection, and the fleet quota ledger.

App-level lane tests live in ``test_role_waterfall_workers.py`` and
``test_role_waterfall_reviewers.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from charlie_work import role_chain, role_quota_ledger, role_selection
from charlie_work.config import ConfigError, build_config_from_data
from charlie_work.role_chain import RoleEntry
from charlie_work.role_selection import select_role_entry

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _z(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# --- config parse / validation ------------------------------------------------


def test_default_roles_have_length_one_chains() -> None:
    config = build_config_from_data({})
    assert config.worker.fallbacks == ()
    assert config.reviewer.fallbacks == ()
    assert config.worker.chain == (RoleEntry(config.worker.harness, config.worker.model),)
    assert len(config.reviewer.chain) == 1


def test_fallbacks_parse_into_frozen_entries_in_order() -> None:
    config = build_config_from_data(
        {
            "worker": {
                "harness": "devin-shell",
                "model": "swe-2",
                "fallbacks": [
                    {"harness": "claude-code", "model": "claude-sonnet-5-5"},
                    {"harness": "devin-shell", "model": "swe-1-6"},
                ],
            },
            "reviewer": {
                "harness": "claude-code",
                "model": "claude-opus-5-5",
                "fallbacks": [{"harness": "devin-shell", "model": "swe-2", "effort": "high"}],
            },
        }
    )
    assert [e.key for e in config.worker.chain] == [
        ("devin-shell", "swe-2"),
        ("claude-code", "claude-sonnet-5-5"),
        ("devin-shell", "swe-1-6"),
    ]
    assert config.reviewer.chain[1] == RoleEntry("devin-shell", "swe-2", "high")
    with pytest.raises(AttributeError):
        config.worker.fallbacks[0].model = "x"  # type: ignore[misc]


# Explicit ids keep every leaf's node id as it was when ``match`` was a loose substring
# (the collect-only gate treats a renamed leaf as a deletion); ``match`` itself is now
# anchored to the key path the section validator reports (ADR-0007).
@pytest.mark.parametrize(
    ("section", "fallbacks", "match"),
    [
        pytest.param(
            "worker",
            [{"harness": "not-a-harness", "model": "m"}],
            r"^worker\.fallbacks\[0\]\.harness: expected one of ",
            id="worker-fallbacks0-harness",
        ),
        pytest.param(
            "reviewer",
            [{"harness": "manual", "model": "m"}],
            r"^reviewer\.fallbacks\[0\]\.harness: expected one of ",
            id="reviewer-fallbacks1-harness",
        ),
        pytest.param(
            "worker",
            [{"harness": "devin-shell", "model": "swe-2"}],
            r"^worker\.fallbacks\[0\]: expected a \(harness, model\) pair not already in",
            id="worker-fallbacks2-duplicate",
        ),
        pytest.param(
            "worker",
            [
                {"harness": "claude-code", "model": "a"},
                {"harness": "claude-code", "model": "a"},
            ],
            r"^worker\.fallbacks\[1\]: expected a \(harness, model\) pair not already in",
            id="worker-fallbacks3-duplicate",
        ),
        pytest.param(
            "worker",
            [{"harness": "claude-code", "model": str(i)} for i in range(4)],
            r"^worker\.fallbacks: expected at most 3 entries, got 4 entries$",
            id="worker-fallbacks4-at most 3",
        ),
        pytest.param(
            "worker",
            {"harness": "claude-code"},
            r"^worker\.fallbacks: expected list of mappings, got ",
            id="worker-fallbacks5-must be a list",
        ),
        pytest.param(
            "worker",
            ["claude-code"],
            r"^worker\.fallbacks\[0\]: expected mapping, got 'claude-code'",
            id="worker-fallbacks6-must be a mapping",
        ),
        pytest.param(
            "worker",
            [{"harness": "claude-code", "effort": "high"}],
            r"^worker\.fallbacks\[0\]: expected known keys \(valid: harness, model\), "
            r"got unknown key\(s\) effort$",
            id="worker-fallbacks7-unknown key",
        ),
        pytest.param(
            "worker",
            [{"harness": "claude-code", "model": 5}],
            r"^worker\.fallbacks\[0\]\.model: expected string, got 5 \(int\)$",
            id="worker-fallbacks8-must be a string",
        ),
        pytest.param(
            "worker",
            [{"harness": "manual"}],
            r"^worker\.fallbacks\[0\]\.harness: expected a harness that records sessions",
            id="worker-fallbacks9-cannot be part of a role chain",
        ),
    ],
)
def test_bad_fallbacks_raise_config_error(section: str, fallbacks: object, match: str) -> None:
    primary = (
        {"harness": "devin-shell", "model": "swe-2"}
        if section == "worker"
        else {"harness": "claude-code", "model": "claude-opus-5-5"}
    )
    with pytest.raises(ConfigError, match=match):
        build_config_from_data({section: {**primary, "fallbacks": fallbacks}})


@pytest.mark.parametrize("section", ["worker", "reviewer"])
def test_null_fallbacks_normalize_to_an_empty_chain(section: str) -> None:
    """``fallbacks: null`` builds ``()`` (what ``chain_of`` expects), never ``None``."""
    role = getattr(build_config_from_data({section: {"fallbacks": None}}), section)
    assert role.fallbacks == ()
    assert len(role.chain) == 1


def test_rescue_roles_keep_fallbacks_unparsed() -> None:
    """Known gap carried over from #2088: rescue.* stores ``fallbacks`` as written."""
    raw = [{"harness": "not-a-harness"}]
    config = build_config_from_data({"rescue": {"worker": {"fallbacks": raw}}})
    assert config.rescue.worker.fallbacks == raw


def test_reviewer_fallback_harness_must_be_review_capable() -> None:
    """``command`` is a worker harness but cannot review: reviewer chains reject it."""
    with pytest.raises(ConfigError, match=r"^reviewer\.fallbacks\[0\]\.harness: expected one of "):
        build_config_from_data({"reviewer": {"fallbacks": [{"harness": "command"}]}})


def test_same_family_fallback_warns_but_loads(caplog: pytest.LogCaptureFixture) -> None:
    role_chain._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="charlie_work.role_chain"):
        config = build_config_from_data(
            {
                "worker": {
                    "harness": "devin-shell",
                    "model": "swe-2",
                    "fallbacks": [{"harness": "claude-code", "model": "claude-sonnet-5-5"}],
                },
                "reviewer": {"harness": "claude-code", "model": "claude-opus-5-5"},
            }
        )
    assert len(config.worker.chain) == 2
    assert any("anthropic-family" in r.getMessage() for r in caplog.records)


def test_cross_family_chains_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    role_chain._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger="charlie_work.role_chain"):
        build_config_from_data(
            {
                "worker": {
                    "harness": "devin-shell",
                    "model": "swe-2",
                    "fallbacks": [{"harness": "devin-shell", "model": "swe-1-6"}],
                },
                "reviewer": {"harness": "claude-code", "model": "claude-opus-5-5"},
            }
        )
    assert not [r for r in caplog.records if "family" in r.getMessage()]


# --- pure selection -------------------------------------------------------------

CHAIN = (
    RoleEntry("devin-shell", "swe-2"),
    RoleEntry("claude-code", "claude-sonnet-5-5"),
    RoleEntry("devin-shell", "swe-1-6"),
)


def test_select_empty_ledger_picks_primary() -> None:
    entry, skipped = select_role_entry(CHAIN, {}, NOW)
    assert entry == CHAIN[0]
    assert skipped == ()


def test_select_skips_restricted_entries_in_order() -> None:
    until = NOW + timedelta(hours=2)
    entry, skipped = select_role_entry(CHAIN, {CHAIN[0].key: until}, NOW)
    assert entry == CHAIN[1]
    assert [(s.index, s.entry, s.until) for s in skipped] == [(0, CHAIN[0], until)]

    entry, skipped = select_role_entry(CHAIN, {CHAIN[0].key: until, CHAIN[1].key: until}, NOW)
    assert entry == CHAIN[2]
    assert [s.index for s in skipped] == [0, 1]


def test_select_expired_restriction_returns_to_primary() -> None:
    entry, skipped = select_role_entry(CHAIN, {CHAIN[0].key: NOW}, NOW)
    assert entry == CHAIN[0]
    assert skipped == ()


def test_select_all_restricted_returns_none() -> None:
    later = NOW + timedelta(minutes=5)
    ledger = {e.key: later + timedelta(minutes=i) for i, e in enumerate(CHAIN)}
    entry, skipped = select_role_entry(CHAIN, ledger, NOW)
    assert entry is None
    assert len(skipped) == 3
    selection = role_selection.build_selection(CHAIN, ledger, NOW)
    assert selection.exhausted
    assert selection.earliest_until == later
    assert selection.report_fields()["chain_retry_at"] == _z(later)


def test_select_ignores_other_models_of_the_same_harness() -> None:
    entry, _ = select_role_entry(
        CHAIN, {("devin-shell", "some-other-model"): NOW + timedelta(hours=1)}, NOW
    )
    assert entry == CHAIN[0]


def test_length_one_chain_never_reads_the_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> dict:
        raise AssertionError("length-1 chain must not read the ledger")

    monkeypatch.setattr(role_quota_ledger, "load_restrictions", _boom)
    selection = role_selection.select_for_launch((CHAIN[0],))
    assert selection.entry == CHAIN[0]
    assert selection.index == 0
    assert selection.chain_report_fields() == {}


# CHAIN[0] is devin-shell/swe-2 (adapter "devin"); CHAIN[1] is claude-code (adapter "claude-code").
QUOTA = {"reason": "quota_exhausted", "adapter_kind": "devin"}


def test_window_covered_for_a_quota_window_attributable_to_a_skipped_entry() -> None:
    until = NOW + timedelta(hours=2)
    selection = role_selection.build_selection(CHAIN, {CHAIN[0].key: until}, NOW)
    assert role_selection.window_covered(_z(until), selection, **QUOTA)
    assert role_selection.window_covered(
        _z(until - timedelta(hours=1)), selection, reason="rate_limited", adapter_kind="devin"
    )


@pytest.mark.parametrize(
    ("per_repo", "kwargs"),
    [
        pytest.param(None, QUOTA, id="no-window"),
        pytest.param(timedelta(hours=3), QUOTA, id="outlasts-the-skipped-restriction"),
        pytest.param(timedelta(hours=1), {"reason": None, "adapter_kind": None}, id="unstamped"),
        pytest.param(
            timedelta(hours=1), {"reason": None, "adapter_kind": "devin"}, id="operator-hold"
        ),
        pytest.param(
            timedelta(hours=1),
            {"reason": "provider_auth", "adapter_kind": "devin"},
            id="dead-credential",
        ),
        pytest.param(
            timedelta(hours=1),
            {"reason": "quota_exhausted", "adapter_kind": None},
            id="no-adapter",
        ),
        pytest.param(
            timedelta(hours=1),
            {"reason": "quota_exhausted", "adapter_kind": "claude-code"},
            id="adapter-of-no-skipped-entry",
        ),
    ],
)
def test_window_covered_refuses_windows_the_ledger_does_not_explain(
    per_repo: timedelta | None, kwargs: dict[str, str | None]
) -> None:
    until = NOW + timedelta(hours=2)
    selection = role_selection.build_selection(CHAIN, {CHAIN[0].key: until}, NOW)
    window = None if per_repo is None else _z(NOW + per_repo)
    assert not role_selection.window_covered(window, selection, **kwargs)


def test_window_covered_ignores_restrictions_on_entries_selection_did_not_skip() -> None:
    # The primary is free; only a LATER entry is restricted. No skipped entry
    # exists, so no per-repo window is explained -- even a short, stamped one.
    later_only = role_selection.build_selection(
        CHAIN, {CHAIN[2].key: NOW + timedelta(hours=9)}, NOW
    )
    assert later_only.skipped == ()
    window = _z(NOW + timedelta(hours=1))
    assert not role_selection.window_covered(window, later_only, **QUOTA)
    assert not role_selection.window_covered(
        window, later_only, reason="quota_exhausted", adapter_kind="devin"
    )


def test_window_covered_never_for_a_length_one_chain_or_an_exhausted_chain() -> None:
    until = NOW + timedelta(hours=2)
    single = role_selection.build_selection(CHAIN[:1], {}, NOW)
    assert not role_selection.window_covered(_z(until), single, **QUOTA)
    ledger = {e.key: until for e in CHAIN}
    exhausted = role_selection.build_selection(CHAIN, ledger, NOW)
    assert not role_selection.window_covered(_z(until), exhausted, **QUOTA)


# --- ledger ---------------------------------------------------------------------


def test_ledger_path_is_in_the_fleet_dir(tmp_path: Path) -> None:
    # conftest points CHARLIE_WORK_FLEET_DIR at tmp_path / "fleet".
    assert role_quota_ledger.ledger_path() == tmp_path / "fleet" / "role_quota_ledger.json"


def test_ledger_records_monotonically_and_atomically(tmp_path: Path) -> None:
    later = NOW + timedelta(hours=3)
    earlier = NOW + timedelta(hours=1)
    assert role_quota_ledger.record_restriction(
        "devin-shell", "swe-2", later, reason="rate_limited", source="t"
    )
    assert not role_quota_ledger.record_restriction(
        "devin-shell", "swe-2", earlier, reason="rate_limited", source="t"
    )
    assert role_quota_ledger.load_restrictions() == {("devin-shell", "swe-2"): later}
    path = role_quota_ledger.ledger_path()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert not list(path.parent.glob("*.tmp"))


def test_ledger_tolerates_a_corrupt_file() -> None:
    path = role_quota_ledger.ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert role_quota_ledger.load_restrictions() == {}
    assert role_quota_ledger.record_restriction(
        "claude-code", "m", NOW + timedelta(hours=1), reason="quota_exhausted", source="t"
    )


def test_classified_death_records_only_stamped_throttle_kinds() -> None:
    until = _z(NOW + timedelta(hours=1))
    stamped = {role_quota_ledger.SESSION_ROLE_KEY: {"harness": "devin-shell", "model": "swe-2"}}
    assert not role_quota_ledger.record_classified_death({}, "rate_limited", until, source="t")
    assert not role_quota_ledger.record_classified_death(stamped, "stalled", until, source="t")
    assert not role_quota_ledger.record_classified_death(
        stamped, "provider_auth", until, source="t"
    )
    assert role_quota_ledger.load_restrictions() == {}
    assert role_quota_ledger.record_classified_death(stamped, "quota_exhausted", until, source="t")
    assert set(role_quota_ledger.load_restrictions()) == {("devin-shell", "swe-2")}
