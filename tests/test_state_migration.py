"""Load-bearing regression tests: production state.json shapes must survive.

The fixture `tests/fixtures/state_production_redacted.json` is a redacted copy of a
real job-cannon `.var/charlie-work/state.json` (titles/bodies/URLs/paths
scrubbed; numbers, statuses, event kinds, and timestamps kept verbatim — see
docs/design/extraction-dossier.md section 3 "De facto state.json schema"). These
tests prove the extracted `state.py` module can round-trip a file written by the
OLD in-repo orchestrators without dropping fields it doesn't know about.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from charlie_work.state import append_event, empty_state, load_state, save_state

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "state_production_redacted.json"


def _load_fixture_raw() -> dict:
    with FIXTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _copied_fixture(tmp_path: Path) -> Path:
    # load_state can quarantine unparseable files in place; never point it at the
    # checked-in fixture itself.
    dest = tmp_path / "state.json"
    shutil.copyfile(FIXTURE_PATH, dest)
    return dest


def test_fixture_is_valid_json_and_nonempty() -> None:
    raw = _load_fixture_raw()

    assert raw["version"] == 1
    assert len(raw["issues"]) > 0
    assert len(raw["prs"]) > 0
    assert len(raw["events"]) > 0


def test_load_state_preserves_top_level_keys(tmp_path: Path) -> None:
    raw = _load_fixture_raw()
    state_path = _copied_fixture(tmp_path)

    loaded = load_state(state_path)

    assert loaded["version"] == raw["version"]
    assert loaded["generated_at"] == raw["generated_at"]
    assert set(loaded["issues"].keys()) == set(raw["issues"].keys())
    assert set(loaded["prs"].keys()) == set(raw["prs"].keys())
    assert len(loaded["events"]) == len(raw["events"])


def test_load_state_preserves_issue_fields(tmp_path: Path) -> None:
    raw = _load_fixture_raw()
    state_path = _copied_fixture(tmp_path)

    loaded = load_state(state_path)

    for number, expected in raw["issues"].items():
        assert loaded["issues"][number] == expected


def test_load_state_preserves_pr_fields_including_optional_ones(tmp_path: Path) -> None:
    raw = _load_fixture_raw()
    state_path = _copied_fixture(tmp_path)

    loaded = load_state(state_path)

    for number, expected in raw["prs"].items():
        assert loaded["prs"][number] == expected

    # Production PR schema drift (see dossier §3): "decision", "cross_family_ok",
    # and "cross_family_report" are all optional and must not be required by the
    # loader. Confirm at least one PR in the fixture exercises the absent case
    # (pr-497: reviewed but never got a terminal decision — the dossier's
    # documented "review_packet clobbers decision" case) and one exercises the
    # present case.
    without_decision = [pr for pr in raw["prs"].values() if "decision" not in pr]
    with_cross_family = [pr for pr in raw["prs"].values() if "cross_family_ok" in pr]
    assert without_decision, "fixture must retain a PR with no decision field"
    assert with_cross_family, "fixture must retain a PR with cross_family fields"


def test_load_state_preserves_event_order_and_kinds(tmp_path: Path) -> None:
    raw = _load_fixture_raw()
    state_path = _copied_fixture(tmp_path)

    loaded = load_state(state_path)

    assert loaded["events"] == raw["events"]

    observed_kinds = {event["kind"] for event in loaded["events"]}
    # Every event kind seen in production must still be present in the fixture
    # after loading — this is the "at least one of every distinct event kind"
    # guarantee from the task spec.
    assert observed_kinds == {
        "intake",
        "dispatch",
        "review_packet",
        "record_review",
        "spec_review",
    }


def test_load_state_tolerates_event_payload_schema_drift(tmp_path: Path) -> None:
    """dispatch/review_packet payloads gained fields mid-history in production.

    Per the dossier: `dispatch` payload gained `failed_issue_numbers` partway
    through; `review_packet` gained `cross_family_ok`/`cross_family_reused`
    later still. Both old- and new-shape payloads must survive a load/save
    round trip unchanged, since state.py stores events opaquely.
    """
    state_path = _copied_fixture(tmp_path)
    loaded = load_state(state_path)

    dispatch_payloads = [e["payload"] for e in loaded["events"] if e["kind"] == "dispatch"]
    old_shape = [p for p in dispatch_payloads if "failed_issue_numbers" not in p]
    new_shape = [p for p in dispatch_payloads if "failed_issue_numbers" in p]
    assert old_shape, "fixture must retain a pre-drift dispatch payload"
    assert new_shape, "fixture must retain a post-drift dispatch payload"

    review_packet_payloads = [
        e["payload"] for e in loaded["events"] if e["kind"] == "review_packet"
    ]
    old_rp_shape = [p for p in review_packet_payloads if "cross_family_ok" not in p]
    new_rp_shape = [p for p in review_packet_payloads if "cross_family_ok" in p]
    assert old_rp_shape, "fixture must retain a pre-drift review_packet payload"
    assert new_rp_shape, "fixture must retain a post-drift review_packet payload"


def test_save_then_load_round_trips_unknown_top_level_field(tmp_path: Path) -> None:
    """setdefault-based loading must never drop fields it doesn't recognize.

    The fixture carries a synthetic `_fixture_unknown_field` that current
    state.py neither reads nor writes. If load_state/save_state ever switched
    from mutate-and-return to an allowlist/schema-validated shape, this is the
    regression that would catch silently dropped production data.
    """
    raw = _load_fixture_raw()
    assert "_fixture_unknown_field" in raw, "fixture must carry a deliberately unknown field"

    state_path = _copied_fixture(tmp_path)
    loaded = load_state(state_path)
    assert loaded["_fixture_unknown_field"] == raw["_fixture_unknown_field"]

    save_state(state_path, loaded)
    reloaded = load_state(state_path)

    assert reloaded["_fixture_unknown_field"] == raw["_fixture_unknown_field"]
    assert reloaded["issues"] == raw["issues"]
    assert reloaded["prs"] == raw["prs"]
    assert reloaded["events"] == raw["events"]


def test_save_state_refreshes_generated_at_but_keeps_everything_else(tmp_path: Path) -> None:
    raw = _load_fixture_raw()
    state_path = _copied_fixture(tmp_path)
    loaded = load_state(state_path)

    save_state(state_path, loaded)
    reloaded = load_state(state_path)

    # save_state always stamps a fresh generated_at (see state.py); production
    # data predates "now", so this must have moved forward, not been dropped.
    assert reloaded["generated_at"] != raw["generated_at"]
    assert reloaded["issues"] == raw["issues"]
    assert reloaded["prs"] == raw["prs"]
    assert reloaded["events"] == raw["events"]


def test_append_event_caps_at_200_keeping_newest() -> None:
    """append_event bounds the audit trail to the most recent 200 entries.

    This is the truncation the durable per-PR rework counter had to be moved
    off of (see test_rework_cap_survives_event_log_truncation) — pin the exact
    contract so the cap and the newest-retained invariant can't silently drift.
    """
    state = empty_state()
    for index in range(205):
        append_event(state, "dispatch", {"seq": index})

    events = state["events"]
    assert len(events) == 200
    # Oldest five (seq 0-4) were evicted; the newest is retained and last.
    assert events[0]["payload"]["seq"] == 5
    assert events[-1]["payload"]["seq"] == 204
    assert [event["payload"]["seq"] for event in events] == list(range(5, 205))


def test_append_event_below_cap_keeps_all() -> None:
    state = empty_state()
    for index in range(3):
        append_event(state, "intake", {"seq": index})

    assert [event["payload"]["seq"] for event in state["events"]] == [0, 1, 2]


def test_old_orchestrator_state_file_loads_cleanly_end_to_end(tmp_path: Path) -> None:
    """A state.json written by an OLD in-repo orchestrator loads without error,
    without quarantine, and without raising in the NEW extracted package."""
    state_path = _copied_fixture(tmp_path)

    loaded = load_state(state_path)

    # No corrupt-quarantine sibling file should have been created.
    quarantined = list(tmp_path.glob("state.json.corrupt-*"))
    assert quarantined == []

    assert isinstance(loaded["issues"], dict)
    assert isinstance(loaded["prs"], dict)
    assert isinstance(loaded["events"], list)
    assert loaded["version"] == 1
