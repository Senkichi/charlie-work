"""Tests for baseline.py: round-trip, bump validation (G4), ratchet, tamper."""

from __future__ import annotations

import json

from charlie_work.attachment_contracts.baseline import (
    KIND_STATS_KEY,
    TamperError,
    check_ratchet_tamper,
    check_tamper,
    compare,
    dumps,
    entries_of,
    generate,
    kind_stats_of,
    load,
    loads,
    validate_bump,
    with_kind_stats,
)
from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    BaselineEntry,
    Bump,
    KindStats,
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


def test_interactive_bump_without_ack_rejected() -> None:
    # Round-2 review finding #10: round-1 validated the worker branch only,
    # so a worker that mislabeled itself `actor="interactive"` bypassed the
    # ack requirement entirely (`Bump(actor="interactive", ack="")` used to
    # pass). Both actors now require the same shape-checked ack, closing the
    # mislabel vector: there is nothing to gain by claiming "interactive".
    bump = Bump(to=20, reason="reviewed and approved", actor="interactive", ack="")
    error = validate_bump(bump)
    assert error is not None
    assert "G4" in error


def test_interactive_bump_with_shaped_ack_accepted() -> None:
    bump = Bump(to=20, reason="reviewed and approved", actor="interactive", ack="handle:senkichi")
    assert validate_bump(bump) is None


def test_interactive_bump_with_junk_ack_rejected() -> None:
    # A mislabeled worker gains nothing: a junk ack is rejected for
    # actor="interactive" exactly as it is for actor="worker".
    bump = Bump(to=20, reason="reviewed and approved", actor="interactive", ack="x")
    error = validate_bump(bump)
    assert error is not None
    assert "G4" in error


def test_worker_bump_with_junk_ack_rejected() -> None:
    # Round-2 review finding #10: ack was validated for non-emptiness only,
    # so a junk ack like "x" passed. Now the shape must look like an
    # external reference.
    bump = Bump(to=20, reason="spike", actor="worker", ack="x")
    error = validate_bump(bump)
    assert error is not None
    assert "G4" in error


def test_worker_bump_with_issue_number_ack_accepted() -> None:
    bump = Bump(to=20, reason="spike", actor="worker", ack="#42")
    assert validate_bump(bump) is None


def test_worker_bump_with_owner_repo_issue_ack_accepted() -> None:
    bump = Bump(to=20, reason="spike", actor="worker", ack="owner/repo#42")
    assert validate_bump(bump) is None


def test_worker_bump_with_url_ack_accepted() -> None:
    bump = Bump(to=20, reason="spike", actor="worker", ack="https://example.com/issues/1")
    assert validate_bump(bump) is None


def test_worker_bump_with_dispatch_handle_ack_accepted() -> None:
    bump = Bump(to=20, reason="spike", actor="worker", ack="dispatch:abc123")
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
    bump = Bump(to=20, reason="reviewed", actor="interactive", ack="handle:senkichi")
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


def test_compare_preserves_unknown_top_level_keys() -> None:
    # Round-2 review finding #11: `compare()` (the engine behind
    # `baseline --ratchet`) previously rebuilt the document from a fixed key
    # allowlist, silently dropping an operator-set "mode" key on every
    # routine ratchet -- reverting the PreToolUse hook's enforce mode back to
    # "advise" with no finding.
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    doc = {**_doc_with_entries(baseline_entry), "mode": "enforce"}
    current = (_verdict("a", 5, boundary=6.0, saturated=True),)

    _findings, ratcheted = compare(current, doc)

    assert ratcheted["mode"] == "enforce"


def test_compare_new_saturated_point_with_no_baseline_entry_blocks() -> None:
    # Round-2 review finding #13: compare() is only ever called once a
    # baseline document already exists (check_tree / `baseline --ratchet`
    # both guard on the file being present) -- the true freeze-on-adopt
    # case ("no baseline anywhere yet") never reaches compare() at all, it
    # is handled entirely by generate(). So a currently-saturated point with
    # no matching entry here is a brand-new god-object, not an adoption
    # artifact, and must block -- silently freezing it (the old behavior)
    # let a fresh 50-method class enter completely unchecked.
    doc = _doc_with_entries()
    current = (_verdict("new_point", 10, boundary=6.0, saturated=True),)

    findings, ratcheted = compare(current, doc)

    assert len(findings) == 1
    assert findings[0].severity == "block"
    assert findings[0].identity == "new_point"
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
    # Finding #10: interactive bumps require a shaped ack too, now.
    bump = Bump(to=50, reason="reviewed spike", actor="interactive", ack="handle:senkichi")
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


# ---------------------------------------------------------------------------
# check_ratchet_tamper(): closes the raise-to-match laundering gap (finding #1)
# ---------------------------------------------------------------------------


def test_ratchet_tamper_detects_raise_to_match_laundering() -> None:
    # The empirical proof case from the round-1 review: baseline hand-raised
    # from 134 to 135 in lockstep with real growth, no bump -- both
    # `compare()` and `check_tamper()` are blind to this; only a diff against
    # the PREVIOUS committed baseline can see it.
    previous = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="OrchestratorApp",
            file="src/x.py",
            member_count=134,
            boundary=5.0,
        )
    )
    current = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="OrchestratorApp",
            file="src/x.py",
            member_count=135,
            boundary=5.0,
        )
    )

    findings = check_ratchet_tamper(previous, current)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "tamper" in findings[0].message
    assert "134" in findings[0].message and "135" in findings[0].message


def test_ratchet_tamper_clean_when_unchanged() -> None:
    entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    previous = _doc_with_entries(entry)
    current = _doc_with_entries(entry)

    assert check_ratchet_tamper(previous, current) == []


def test_ratchet_tamper_clean_on_legitimate_ratchet_down() -> None:
    previous = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    current = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=5, boundary=6.0)
    )

    assert check_ratchet_tamper(previous, current) == []


def test_ratchet_tamper_clean_for_a_brand_new_entry() -> None:
    previous = _doc_with_entries()
    current = _doc_with_entries(
        BaselineEntry(kind="class", identity="new", file="src/n.py", member_count=10, boundary=6.0)
    )

    assert check_ratchet_tamper(previous, current) == []


def test_ratchet_tamper_no_findings_when_no_previous_document() -> None:
    current = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=999, boundary=6.0)
    )

    assert check_ratchet_tamper(None, current) == []


def test_ratchet_tamper_bump_does_not_excuse_a_member_count_raise() -> None:
    # Even a validly-acked bump does NOT justify the member_count FIELD
    # itself rising -- legitimate bump usage raises the ceiling while leaving
    # member_count untouched (see test_compare_bump_raises_effective_ceiling).
    # A rise in member_count is tamper regardless of bumps.
    bump = Bump(to=20, reason="reviewed", actor="interactive", ack="handle:senkichi")
    previous = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    current = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="a",
            file="src/a.py",
            member_count=15,
            boundary=6.0,
            bumps=(bump,),
        )
    )

    findings = check_ratchet_tamper(previous, current)

    assert len(findings) == 1


# ---------------------------------------------------------------------------
# loads(): duplicate (kind, file, identity) entries are rejected (finding #7)
# ---------------------------------------------------------------------------


def test_loads_rejects_duplicate_identity_entries() -> None:
    doc = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="FakeGitHub",
            file="tests/test_worker.py",
            member_count=6,
            boundary=5.0,
        ),
        BaselineEntry(
            kind="class",
            identity="FakeGitHub",
            file="tests/test_worker.py",
            member_count=6,
            boundary=5.0,
        ),
    )
    text = dumps(doc)

    import pytest

    with pytest.raises(TamperError):
        loads(text)


# ---------------------------------------------------------------------------
# loads(): malformed field extraction surfaces as TamperError, never a bare
# KeyError/ValueError escape (finding #12)
# ---------------------------------------------------------------------------


def test_loads_rejects_entry_missing_required_key() -> None:
    doc = {
        "version": 1,
        "generated_by": "x",
        "generated_at": "t",
        "floor": 4,
        "entries": [{"kind": "class", "identity": "a", "file": "src/a.py", "boundary": 6.0}],
    }
    text = json.dumps(doc)

    import pytest

    with pytest.raises(TamperError):
        loads(text)


def test_loads_rejects_entry_with_non_numeric_member_count() -> None:
    doc = {
        "version": 1,
        "generated_by": "x",
        "generated_at": "t",
        "floor": 4,
        "entries": [
            {
                "kind": "class",
                "identity": "a",
                "file": "src/a.py",
                "member_count": "not-a-number",
                "boundary": 6.0,
                "bumps": [],
            }
        ],
    }
    text = json.dumps(doc)

    import pytest

    with pytest.raises(TamperError):
        loads(text)


def test_loads_rejects_bump_missing_required_key() -> None:
    doc = {
        "version": 1,
        "generated_by": "x",
        "generated_at": "t",
        "floor": 4,
        "entries": [
            {
                "kind": "class",
                "identity": "a",
                "file": "src/a.py",
                "member_count": 10,
                "boundary": 6.0,
                "bumps": [{"to": 20, "actor": "worker"}],  # missing "reason"
            }
        ],
    }
    text = json.dumps(doc)

    import pytest

    with pytest.raises(TamperError):
        loads(text)


def test_loads_allows_same_identity_in_different_files() -> None:
    doc = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="FakeGitHub",
            file="tests/_fakes_github.py",
            member_count=45,
            boundary=5.0,
        ),
        BaselineEntry(
            kind="class",
            identity="FakeGitHub",
            file="tests/_reconcile_fixtures.py",
            member_count=10,
            boundary=5.0,
        ),
    )
    text = dumps(doc)

    reloaded = loads(text)
    assert len(entries_of(reloaded)) == 2


# ---------------------------------------------------------------------------
# kind_stats: frozen per-kind Tukey fence (issue #1614)
# ---------------------------------------------------------------------------


def _verdict_kind(
    identity: str, count: int, kind: str, boundary: float, saturated: bool = True
) -> SaturationVerdict:
    return SaturationVerdict(
        point=AttachmentPoint(
            kind=kind,  # type: ignore[arg-type]
            identity=identity,
            file=f"src/{identity}.py",
            members=tuple(f"m{i}" for i in range(count)),
        ),
        saturated=saturated,
        q3=boundary - 3.0,
        iqr=2.0,
        boundary=boundary,
        population=4,
    )


def test_generate_persists_kind_stats_derived_from_verdicts() -> None:
    verdicts = (
        _verdict_kind("big", 20, "class", boundary=17.0, saturated=True),
        _verdict_kind("bigmod", 50, "test_module", boundary=40.0, saturated=True),
        _verdict_kind("small", 2, "class", boundary=17.0, saturated=False),
    )
    doc = generate(verdicts, generated_by="x", generated_at="t", floor=4)
    stats = kind_stats_of(doc)
    assert set(stats) == {"class", "test_module"}
    assert stats["class"] == KindStats(kind="class", q3=14.0, iqr=2.0, boundary=17.0, population=4)
    assert stats["test_module"].boundary == 40.0


def test_kind_stats_of_empty_when_absent() -> None:
    doc = _doc_with_entries()
    assert kind_stats_of(doc) == {}


def test_kind_stats_round_trips_through_dumps_loads() -> None:
    verdicts = (_verdict_kind("big", 20, "class", boundary=17.0, saturated=True),)
    doc = generate(verdicts, generated_by="x", generated_at="t", floor=4)
    reloaded = loads(dumps(doc))
    assert kind_stats_of(reloaded)["class"].boundary == 17.0


def test_dumps_does_not_add_kind_stats_null_for_old_baseline() -> None:
    # A pre-#1614 baseline (no kind_stats) must not gain a ``"kind_stats": null``
    # entry on a round-trip dump -- that would silently change every old
    # baseline's diff and break check_tree's "absent means fall back to live".
    old = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    text = dumps(old)
    assert KIND_STATS_KEY not in json.loads(text)
    assert kind_stats_of(loads(text)) == {}


def test_loads_rejects_malformed_kind_stats_missing_field() -> None:
    doc = _doc_with_entries()
    doc[KIND_STATS_KEY] = {"class": {"q3": 8.0, "iqr": 6.0}}  # missing boundary, population
    import pytest

    with pytest.raises(TamperError):
        loads(dumps(doc))


def test_loads_rejects_kind_stats_not_object() -> None:
    doc = _doc_with_entries()
    doc[KIND_STATS_KEY] = "not-an-object"
    import pytest

    with pytest.raises(TamperError):
        loads(json.dumps(doc))


def test_compare_preserves_kind_stats_verbatim_on_ratchet() -> None:
    # Issue #1614: a ratchet may not recompute or raise the frozen fence.
    baseline_entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=17.0
    )
    doc = {
        **_doc_with_entries(baseline_entry),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    # Shrink the point -> ratchet down; kind_stats must be unchanged.
    current = (_verdict_kind("a", 5, "class", boundary=17.0, saturated=True),)
    _findings, ratcheted = compare(current, doc)
    assert kind_stats_of(ratcheted)["class"].boundary == 17.0
    # And the ratcheted document carries the same kind_stats object verbatim.
    assert ratcheted[KIND_STATS_KEY] == doc[KIND_STATS_KEY]


def test_with_kind_stats_recomputes_from_verdicts() -> None:
    doc = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    verdicts = (_verdict_kind("big", 20, "class", boundary=15.5, saturated=True),)
    refrozen = with_kind_stats(doc, verdicts)
    assert kind_stats_of(refrozen)["class"].boundary == 15.5
    # Input document is not mutated.
    assert kind_stats_of(doc)["class"].boundary == 17.0


def test_check_ratchet_tamper_detects_raised_frozen_boundary() -> None:
    # Issue #1614: a ratchet may lower entries and may not raise the frozen
    # boundary. A hand-edit that loosened the fence is tamper.
    previous = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    current = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 20.0, "iqr": 2.0, "boundary": 23.0, "population": 8}},
    }
    findings = check_ratchet_tamper(previous, current)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "frozen" in findings[0].message and "class" in findings[0].message
    assert "17.0" in findings[0].message and "23.0" in findings[0].message


def test_check_ratchet_tamper_clean_when_boundary_lowered_or_unchanged() -> None:
    base = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    lowered = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 12.0, "iqr": 2.0, "boundary": 15.0, "population": 8}},
    }
    unchanged = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    assert check_ratchet_tamper(base, lowered) == []
    assert check_ratchet_tamper(base, unchanged) == []


def test_check_ratchet_tamper_clean_when_neither_document_has_kind_stats() -> None:
    # Pre-#1614 baselines: no kind_stats on either side -> no boundary finding.
    previous = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    current = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    assert check_ratchet_tamper(previous, current) == []
