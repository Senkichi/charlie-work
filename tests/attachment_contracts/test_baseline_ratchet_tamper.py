"""Tests for check_ratchet_tamper(): the diff-based tamper guard.

Two guards over consecutive committed baselines:

- member_count raise-to-match laundering (finding #1): an existing entry's
  member_count field may only ever hold or lower on the ratchet path, so any
  rise is a hand-edit.
- Frozen per-kind Tukey fence (issue #1614): a ratchet preserves ``kind_stats``
  verbatim; only a generation event (full ``baseline`` regen / ``--refreeze``,
  which stamp a fresh ``generated_at``) may recompute it. Any same-generation
  difference -- modified stats, a removed or added kind -- is tamper, and the
  key disappearing outright is tamper unconditionally.
"""

from __future__ import annotations

from charlie_work.attachment_contracts.baseline import (
    KIND_STATS_KEY,
    check_ratchet_tamper,
    compare,
    generate,
    kind_stats_of,
    with_kind_stats,
)
from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    BaselineEntry,
    Bump,
    SaturationVerdict,
)


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


# ---------------------------------------------------------------------------
# member_count: raise-to-match laundering (finding #1)
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
# kind_stats: frozen per-kind Tukey fence diff guard (issue #1614)
# ---------------------------------------------------------------------------


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


def test_check_ratchet_tamper_clean_when_frozen_boundary_unchanged() -> None:
    base = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    unchanged = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    assert check_ratchet_tamper(base, unchanged) == []


def test_check_ratchet_tamper_flags_lowered_frozen_boundary() -> None:
    # A ratchet preserves kind_stats VERBATIM -- there is no "ratchet down"
    # for the frozen fence the way there is for member_count, so a lowered
    # boundary on a same-generation transition is just as unsanctioned as a
    # raised one: only a generation event (regen / --refreeze, which re-stamp
    # generated_at) may move it.
    base = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    lowered = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 12.0, "iqr": 2.0, "boundary": 15.0, "population": 8}},
    }
    findings = check_ratchet_tamper(base, lowered)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "lowered" in findings[0].message


def test_check_ratchet_tamper_flags_same_boundary_stats_tweak() -> None:
    # The disable-via-metadata vector: a hand-edit that keeps the boundary but
    # zeroes iqr (or drops population below FLOOR) makes saturate_with_fence
    # report nothing saturated for the kind -- a silent exemption that a
    # boundary-only diff would never see.
    base = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    tweaked = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 0.0, "boundary": 17.0, "population": 8}},
    }
    findings = check_ratchet_tamper(base, tweaked)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "modified" in findings[0].message


def test_check_ratchet_tamper_clean_when_neither_document_has_kind_stats() -> None:
    # Pre-#1614 baselines: no kind_stats on either side -> no boundary finding.
    previous = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    current = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0)
    )
    assert check_ratchet_tamper(previous, current) == []


# ---------------------------------------------------------------------------
# Generation-event discriminator + key-removal
# (round-3 review: --refreeze false-positive / kind_stats-strip false-negative)
# ---------------------------------------------------------------------------


def test_check_ratchet_tamper_clean_on_refreeze_raised_boundary() -> None:
    # Regression (round-3 finding #1): the sanctioned --refreeze path
    # recomputes the frozen fence and MAY legitimately raise a kind's
    # boundary. It must not be flagged as tamper. Exercises the real CLI
    # sequence: compare() against live verdicts, then with_kind_stats().
    previous = generate(
        (_verdict_kind("big", 20, "class", boundary=17.0, saturated=True),),
        generated_by="x",
        generated_at="t1",
        floor=4,
    )
    live_verdicts = (_verdict_kind("big", 20, "class", boundary=23.0, saturated=True),)
    _findings, ratcheted = compare(live_verdicts, previous)
    refrozen = with_kind_stats(ratcheted, live_verdicts, generated_at="t2")

    assert kind_stats_of(refrozen)["class"].boundary == 23.0
    assert check_ratchet_tamper(previous, refrozen) == []


def test_check_ratchet_tamper_clean_on_full_regen() -> None:
    # A fresh full `baseline` run writes a new generated_at and may recompute
    # the fence upward; it is reviewed as a whole-document diff, not flagged.
    previous = generate(
        (_verdict_kind("big", 20, "class", boundary=17.0, saturated=True),),
        generated_by="x",
        generated_at="t1",
        floor=4,
    )
    regenerated = generate(
        (_verdict_kind("big", 20, "class", boundary=23.0, saturated=True),),
        generated_by="x",
        generated_at="t2",
        floor=4,
    )
    assert check_ratchet_tamper(previous, regenerated) == []


def test_check_ratchet_tamper_flags_stripped_kind_stats_key() -> None:
    # Regression (round-3 finding #2): deleting the kind_stats key entirely
    # silently reverts check_tree / --ratchet to the pre-#1614 live
    # recomputation fallback. No baseline writer ever drops the key, so its
    # absence is flagged as tamper.
    previous = generate(
        (_verdict_kind("big", 20, "class", boundary=17.0, saturated=True),),
        generated_by="x",
        generated_at="t1",
        floor=4,
    )
    stripped = {k: v for k, v in previous.items() if k != KIND_STATS_KEY}
    assert KIND_STATS_KEY in previous and KIND_STATS_KEY not in stripped

    findings = check_ratchet_tamper(previous, stripped)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].identity == KIND_STATS_KEY
    assert "tamper" in findings[0].message


def test_check_ratchet_tamper_flags_stripped_key_even_with_fresh_generated_at() -> None:
    # A hand-deletion that also bumps generated_at to impersonate a regen
    # still cannot explain a missing key: generate()/with_kind_stats() always
    # emit it. Key removal is therefore flagged unconditionally.
    previous = generate(
        (_verdict_kind("big", 20, "class", boundary=17.0, saturated=True),),
        generated_by="x",
        generated_at="t1",
        floor=4,
    )
    stripped = {
        **{k: v for k, v in previous.items() if k != KIND_STATS_KEY},
        "generated_at": "t2",
    }
    findings = check_ratchet_tamper(previous, stripped)
    assert len(findings) == 1
    assert findings[0].identity == KIND_STATS_KEY


def test_check_ratchet_tamper_flags_removed_kind() -> None:
    # Removing one kind from kind_stats silently exempts it:
    # saturate_all_with_fences produces no verdicts for a kind with no frozen
    # fence, so the whole kind escapes saturation.
    previous = generate(
        (
            _verdict_kind("big", 20, "class", boundary=17.0, saturated=True),
            _verdict_kind("bigmod", 50, "test_module", boundary=40.0, saturated=True),
        ),
        generated_by="x",
        generated_at="t1",
        floor=4,
    )
    shrunk = {
        **previous,
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 4}},
    }
    findings = check_ratchet_tamper(previous, shrunk)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].identity == "kind_stats:test_module"
    assert "removed" in findings[0].message


def test_check_ratchet_tamper_flags_added_kind() -> None:
    # A kind may only enter kind_stats through a generation event; a hand-add
    # on a ratchet transition is unsanctioned (e.g. turning a live-recompute
    # baseline into a selectively-frozen one that exempts unlisted kinds).
    previous = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {"class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8}},
    }
    added = {
        **_doc_with_entries(),
        KIND_STATS_KEY: {
            "class": {"q3": 14.0, "iqr": 2.0, "boundary": 17.0, "population": 8},
            "test_module": {"q3": 30.0, "iqr": 5.0, "boundary": 1000.0, "population": 8},
        },
    }
    findings = check_ratchet_tamper(previous, added)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert findings[0].identity == "kind_stats:test_module"
    assert "appeared" in findings[0].message
