"""Tests for baseline.py: round-trip, bump validation (G4), ratchet, tamper."""

from __future__ import annotations

import json

from charlie_work.attachment_contracts.baseline import (
    check_tamper,
    compare,
    dumps,
    entries_of,
    generate,
    load,
    loads,
    validate_bump,
)
from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    BaselineEntry,
    Bump,
    SaturationVerdict,
)


def _point(identity: str, count: int, file: str | None = None) -> AttachmentPoint:
    return AttachmentPoint(
        kind="class",
        identity=identity,
        file=file or f"src/{identity}.py",
        members=tuple(f"m{i}" for i in range(count)),
    )


def _verdict(
    identity: str, count: int, boundary: float = 6.0, saturated: bool = True
) -> SaturationVerdict:
    return SaturationVerdict(
        point=_point(identity, count),
        saturated=saturated,
        q3=3.0,
        iqr=2.0,
        boundary=boundary,
        population=4,
    )


# ---------------------------------------------------------------------------
# generate() + round-trip (dumps/loads, load)
# ---------------------------------------------------------------------------


def test_generate_includes_only_saturated_points() -> None:
    verdicts = (
        _verdict("a", 10, saturated=True),
        _verdict("b", 2, saturated=False),
    )
    doc = generate(
        verdicts,
        generated_by="charlie_work.attachment_contracts 0.1.0",
        generated_at="2026-08-24T00:00:00Z",
        floor=4,
    )
    assert len(doc["entries"]) == 1
    assert doc["entries"][0]["identity"] == "a"
    assert doc["version"] == 1
    assert doc["floor"] == 4


def test_generate_entries_sorted_by_kind_file_identity() -> None:
    v_b = _verdict("b_point", 10, saturated=True)
    v_a = _verdict("a_point", 10, saturated=True)
    doc = generate((v_b, v_a), generated_by="x", generated_at="t", floor=4)
    idents = [e["identity"] for e in doc["entries"]]
    assert idents == ["a_point", "b_point"]


def test_dumps_is_deterministic() -> None:
    verdicts = (_verdict("z", 10), _verdict("a", 10))
    doc = generate(verdicts, generated_by="x", generated_at="t", floor=4)
    text1 = dumps(doc)
    text2 = dumps(doc)
    assert text1 == text2
    assert text1.endswith("\n")
    # indent=1, sorted keys: top-level keys alphabetical.
    parsed = json.loads(text1)
    assert list(parsed.keys()) == sorted(parsed.keys())
    # entries sorted (kind, file, identity) regardless of input order.
    assert [e["identity"] for e in parsed["entries"]] == ["a", "z"]


def test_round_trip_generate_dumps_loads_entries_of() -> None:
    verdicts = (_verdict("a", 10, boundary=6.0),)
    doc = generate(verdicts, generated_by="x", generated_at="t", floor=4)
    text = dumps(doc)
    reloaded = loads(text)
    entries = entries_of(reloaded)
    assert len(entries) == 1
    assert entries[0] == BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0, bumps=()
    )


def test_round_trip_preserves_bumps() -> None:
    bump = Bump(to=20, reason="temporary spike", actor="worker", ack="issue#999")
    entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0, bumps=(bump,)
    )
    doc = {
        "version": 1,
        "generated_by": "x",
        "generated_at": "t",
        "floor": 4,
        "entries": [
            {
                "kind": entry.kind,
                "identity": entry.identity,
                "file": entry.file,
                "member_count": entry.member_count,
                "boundary": entry.boundary,
                "bumps": [
                    {"to": bump.to, "reason": bump.reason, "actor": bump.actor, "ack": bump.ack}
                ],
            }
        ],
    }
    reloaded = loads(dumps(doc))
    entries = entries_of(reloaded)
    assert entries[0].bumps == (bump,)


def test_load_rejects_wrong_version() -> None:
    text = json.dumps(
        {"version": 2, "generated_by": "x", "generated_at": "t", "floor": 4, "entries": []}
    )
    import pytest

    with pytest.raises(Exception):
        loads(text)


def test_load_from_path(tmp_path) -> None:
    verdicts = (_verdict("a", 10),)
    doc = generate(verdicts, generated_by="x", generated_at="t", floor=4)
    path = tmp_path / ".attachment-budgets.json"
    path.write_text(dumps(doc), encoding="utf-8")
    reloaded = load(path)
    assert entries_of(reloaded)[0].identity == "a"


# ---------------------------------------------------------------------------
# validate_bump (G4)
# ---------------------------------------------------------------------------


def test_worker_bump_without_ack_rejected() -> None:
    bump = Bump(to=20, reason="spike", actor="worker", ack="")
    error = validate_bump(bump)
    assert error is not None
    assert "G4" in error


def test_worker_bump_with_ack_accepted() -> None:
    bump = Bump(to=20, reason="spike", actor="worker", ack="issue#42")
    assert validate_bump(bump) is None


def test_interactive_bump_without_ack_accepted() -> None:
    bump = Bump(to=20, reason="reviewed and approved", actor="interactive", ack="")
    assert validate_bump(bump) is None


def test_bump_without_reason_rejected() -> None:
    bump = Bump(to=20, reason="   ", actor="interactive", ack="")
    error = validate_bump(bump)
    assert error is not None
    assert "reason" in error


# ---------------------------------------------------------------------------
# compare(): block finding, ratchet-down, bump raises ceiling
# ---------------------------------------------------------------------------


def _doc_with_entries(*entries: BaselineEntry) -> dict[str, object]:
    return {
        "version": 1,
        "generated_by": "x",
        "generated_at": "t",
        "floor": 4,
        "entries": [
            {
                "kind": e.kind,
                "identity": e.identity,
                "file": e.file,
                "member_count": e.member_count,
                "boundary": e.boundary,
                "bumps": [
                    {"to": b.to, "reason": b.reason, "actor": b.actor, "ack": b.ack}
                    for b in e.bumps
                ],
            }
            for e in entries
        ],
    }


def test_compare_blocks_point_above_baseline_with_no_bump() -> None:
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 15, saturated=True),)

    findings, ratcheted = compare(current, doc)

    assert len(findings) == 1
    assert findings[0].severity == "block"
    assert findings[0].identity == "a"
    ratcheted_entries = entries_of(ratcheted)
    assert ratcheted_entries[0].member_count == 10  # unchanged, still blocked


def test_compare_ratchets_down_when_point_shrinks() -> None:
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 5, boundary=6.0, saturated=True),)

    findings, ratcheted = compare(current, doc)

    assert findings == []
    ratcheted_entries = entries_of(ratcheted)
    assert len(ratcheted_entries) == 1
    assert ratcheted_entries[0].member_count == 5


def test_compare_no_longer_saturated_drops_from_baseline() -> None:
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 3, boundary=6.0, saturated=False),)

    findings, ratcheted = compare(current, doc)

    assert findings == []
    assert entries_of(ratcheted) == ()


def test_compare_bump_raises_effective_ceiling() -> None:
    bump = Bump(to=20, reason="reviewed", actor="interactive", ack="")
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0, bumps=(bump,)
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 15, boundary=6.0, saturated=True),)

    findings, ratcheted = compare(current, doc)

    assert findings == []
    ratcheted_entries = entries_of(ratcheted)
    assert ratcheted_entries[0].member_count == 10
    assert ratcheted_entries[0].bumps == (bump,)


def test_compare_newly_saturated_point_added_without_finding() -> None:
    doc = _doc_with_entries()
    current = (_verdict("new_point", 10, boundary=6.0, saturated=True),)

    findings, ratcheted = compare(current, doc)

    assert findings == []
    ratcheted_entries = entries_of(ratcheted)
    assert len(ratcheted_entries) == 1
    assert ratcheted_entries[0].identity == "new_point"


# ---------------------------------------------------------------------------
# check_tamper()
# ---------------------------------------------------------------------------


def test_tamper_detects_hand_raised_member_count() -> None:
    # Baseline claims 50 members with no bump justifying it; actual point
    # currently has only 10 -- classic hand-edit-the-JSON tamper.
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=50, boundary=6.0
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 10, boundary=6.0, saturated=True),)

    findings = check_tamper(current, doc)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "tamper" in findings[0].message


def test_tamper_clean_when_covered_by_matching_bump() -> None:
    bump = Bump(to=50, reason="reviewed spike", actor="interactive", ack="")
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=50, boundary=6.0, bumps=(bump,)
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 10, boundary=6.0, saturated=True),)

    findings = check_tamper(current, doc)

    assert findings == []


def test_tamper_flags_invalid_worker_bump_even_if_count_consistent() -> None:
    bump = Bump(to=10, reason="spike", actor="worker", ack="")  # G4 violation
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0, bumps=(bump,)
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 10, boundary=6.0, saturated=True),)

    findings = check_tamper(current, doc)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "G4" in findings[0].message


def test_tamper_no_finding_when_matches_and_no_bumps() -> None:
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    doc = _doc_with_entries(baseline_entry)
    current = (_verdict("a", 10, boundary=6.0, saturated=True),)

    findings = check_tamper(current, doc)

    assert findings == []
