"""Tests for pinned baseline entries (issue #1620).

A ``"pinned": true`` baseline row is an operator-authored sub-saturation
contract: it records a ceiling for a point deliberately de-godded BELOW the
Tukey fence, so a class the campaign shrank cannot silently regrow -- before
this, a de-saturated class simply lost its baseline row and could regrow up
to the fence unchecked. Covered here: compare() semantics (enforce below the
fence, ratchet down, never drop, ceiling capped at the fence), serialization
round-trip, the check_tamper exemption, the check_ratchet_tamper removal
guard, and the check_tree end-to-end positive control from the issue's
acceptance criteria.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.attachment_contracts import baseline_dir
from charlie_work.attachment_contracts.__main__ import main
from charlie_work.attachment_contracts.baseline import (
    BASELINE_DIRNAME,
    TamperError,
    compare,
    dumps,
    entries_of,
    generate,
    loads,
)
from charlie_work.attachment_contracts.check import check_file, check_tree
from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.archetypes import scan_tree
from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    BaselineEntry,
    Bump,
    SaturationVerdict,
)
from charlie_work.attachment_contracts.outliers import saturate_all
from charlie_work.attachment_contracts.tamper import (
    check_ratchet_tamper,
    check_tamper,
)


def _point(identity: str, count: int, file: str | None = None) -> AttachmentPoint:
    return AttachmentPoint(
        kind="class",
        identity=identity,
        file=file or f"src/{identity}.py",
        members=tuple(f"m{i}x" for i in range(count)),
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


def _doc_with_entries(*entries: BaselineEntry) -> dict[str, object]:
    doc_entries: list[dict[str, object]] = []
    for e in entries:
        raw: dict[str, object] = {
            "kind": e.kind,
            "identity": e.identity,
            "file": e.file,
            "member_count": e.member_count,
            "boundary": e.boundary,
            "bumps": [
                {"to": b.to, "reason": b.reason, "actor": b.actor, "ack": b.ack} for b in e.bumps
            ],
        }
        if e.pinned:
            raw["pinned"] = True
        doc_entries.append(raw)
    return {
        "version": 1,
        "generated_by": "x",
        "generated_at": "t",
        "floor": 4,
        "entries": doc_entries,
    }


# ---------------------------------------------------------------------------
# compare(): pinned rows are enforced below the fence, ratchet down, never drop
# ---------------------------------------------------------------------------


def test_compare_pinned_unsaturated_point_over_ceiling_blocks() -> None:
    # The issue's core defect: a de-godded class below the fence grows past
    # its pinned ceiling -> block, exactly like over-fence saturation.
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    findings, ratcheted = compare(
        (_verdict("a", 6, boundary=6.0, saturated=False),),
        _doc_with_entries(pin),
    )
    assert len(findings) == 1
    assert findings[0].severity == "block"
    assert findings[0].identity == "a"
    assert "pinned ceiling" in findings[0].message
    # The row is kept unchanged while the violation is outstanding.
    ratcheted_entries = entries_of(ratcheted)
    assert ratcheted_entries == (pin,)


def test_compare_pinned_unsaturated_point_within_ceiling_clean() -> None:
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    findings, ratcheted = compare(
        (_verdict("a", 5, boundary=6.0, saturated=False),),
        _doc_with_entries(pin),
    )
    assert findings == []
    assert entries_of(ratcheted) == (pin,)


def test_compare_pinned_row_survives_desaturation() -> None:
    # The exact gap this issue closes: an ordinary row is dropped by omission
    # the moment its point de-saturates; a pinned row must persist.
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    findings, ratcheted = compare(
        (_verdict("a", 3, boundary=6.0, saturated=False),),
        _doc_with_entries(pin),
    )
    # 3 < member_count 5 -> the pin ratchets DOWN with the class, flag kept.
    assert findings == []
    ratcheted_entries = entries_of(ratcheted)
    assert len(ratcheted_entries) == 1
    assert ratcheted_entries[0].member_count == 3
    assert ratcheted_entries[0].pinned is True


def test_compare_pinned_row_carried_when_point_ineligible() -> None:
    # A pinned point that produced no verdict at all (deleted, renamed, or
    # became ledger/trivial/empty) keeps its row verbatim -- dropping it
    # would silently end the contract.
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    findings, ratcheted = compare(
        (_verdict("unrelated", 10, boundary=6.0, saturated=True),),
        _doc_with_entries(pin),
    )
    # `unrelated` is a new saturated point (finding #13 -> block + snapshot);
    # the pinned row for absent `a` must still be carried.
    assert len(findings) == 1
    assert findings[0].identity == "unrelated"
    keys = {(e.kind, e.file, e.identity) for e in entries_of(ratcheted)}
    assert ("class", "src/a.py", "a") in keys


def test_compare_pinned_bump_raises_ceiling_below_fence() -> None:
    # A validly-acked bump on a pin row is the sanctioned way to relax the
    # contract partway -- up to the fence, never past it.
    bump = Bump(to=5, reason="temporary regrowth", actor="interactive", ack="handle:senkichi")
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=3,
        boundary=6.0,
        bumps=(bump,),
        pinned=True,
    )
    findings, ratcheted = compare(
        (_verdict("a", 5, boundary=6.0, saturated=False),),
        _doc_with_entries(pin),
    )
    assert findings == []
    assert entries_of(ratcheted)[0].bumps == (bump,)


def test_compare_pinned_ceiling_capped_at_fence() -> None:
    # Anti-laundering: a pin row with member_count ABOVE the boundary must
    # not widen a class's budget past the fence -- the pin can only tighten.
    # A saturated point under an over-fence pin ceiling still blocks.
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=50,
        boundary=6.0,
        pinned=True,
    )
    findings, _ratcheted = compare(
        (_verdict("a", 15, boundary=6.0, saturated=True),),
        _doc_with_entries(pin),
    )
    assert len(findings) == 1
    assert findings[0].severity == "block"
    assert "pinned ceiling" in findings[0].message


def test_compare_ratchet_never_creates_pinned_rows() -> None:
    # Acceptance: `baseline --ratchet` adds no row by itself -- pins exist
    # only because an operator authored one.
    entry = BaselineEntry(
        kind="class", identity="a", file="src/a.py", member_count=10, boundary=6.0
    )
    doc = _doc_with_entries(entry)
    _findings, ratcheted = compare((_verdict("a", 10, boundary=6.0, saturated=True),), doc)
    assert all(not e.pinned for e in entries_of(ratcheted))


# ---------------------------------------------------------------------------
# Serialization: the flag round-trips; malformed values fail closed
# ---------------------------------------------------------------------------


def test_pinned_flag_round_trips_dumps_loads() -> None:
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    plain = BaselineEntry(
        kind="class", identity="b", file="src/b.py", member_count=10, boundary=6.0
    )
    reloaded = loads(dumps(_doc_with_entries(pin, plain)))
    by_identity = {e.identity: e for e in entries_of(reloaded)}
    assert by_identity["a"].pinned is True
    assert by_identity["b"].pinned is False
    # And the flag is emitted only when True -- no "pinned": false churn on
    # pre-#1620 entry files.
    parsed = json.loads(dumps(_doc_with_entries(plain)))
    assert "pinned" not in parsed["entries"][0]


def test_loads_rejects_non_bool_pinned() -> None:
    doc = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=5, boundary=6.0)
    )
    doc["entries"][0]["pinned"] = "yes"  # type: ignore[index]
    with pytest.raises(TamperError):
        loads(json.dumps(doc))


# ---------------------------------------------------------------------------
# check_tamper: pin ceilings legitimately sit above the live count
# ---------------------------------------------------------------------------


def test_tamper_clean_for_pinned_ceiling_above_actual() -> None:
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    # Live count 3 < pin ceiling 5 -- the headroom IS the contract, not tamper.
    findings = check_tamper(
        (_verdict("a", 3, boundary=6.0, saturated=False),),
        _doc_with_entries(pin),
    )
    assert findings == []


def test_tamper_still_flags_invalid_bump_on_pinned_entry() -> None:
    bump = Bump(to=8, reason="spike", actor="worker", ack="")  # G4 violation
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        bumps=(bump,),
        pinned=True,
    )
    findings = check_tamper(
        (_verdict("a", 3, boundary=6.0, saturated=False),),
        _doc_with_entries(pin),
    )
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "G4" in findings[0].message


# ---------------------------------------------------------------------------
# check_ratchet_tamper: a pin that vanished or lost its flag is a hand-edit
# ---------------------------------------------------------------------------


def test_ratchet_tamper_flags_removed_pinned_entry() -> None:
    # Unlike removing an ordinary entry (self-healing: finding #13 re-blocks
    # the point the moment it re-saturates), deleting a pin row re-opens a
    # silent regrowth window -- the point stays below the fence and no other
    # guard ever looks at it again.
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    previous = _doc_with_entries(pin)
    current = _doc_with_entries()

    findings = check_ratchet_tamper(previous, current, live_verdicts=None)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "pinned" in findings[0].message
    assert findings[0].identity == "a"


def test_ratchet_tamper_flags_unpinned_entry() -> None:
    # Laundering the removal across commits: drop the flag in one commit and
    # let a later ratchet drop the (now-ordinary, un-saturated) row. The flag
    # flip alone is flagged.
    previous = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="a",
            file="src/a.py",
            member_count=5,
            boundary=6.0,
            pinned=True,
        )
    )
    current = _doc_with_entries(
        BaselineEntry(kind="class", identity="a", file="src/a.py", member_count=5, boundary=6.0)
    )

    findings = check_ratchet_tamper(previous, current, live_verdicts=None)

    assert len(findings) == 1
    assert "pinned" in findings[0].message


def test_ratchet_tamper_flags_member_count_rise_on_pinned_entry() -> None:
    # member_count on a pin row is the contract's ceiling -- raising it is
    # governed by the same unconditional rise guard as any other entry
    # (legitimate growth goes through an acked bump, never a rewrite).
    previous = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="a",
            file="src/a.py",
            member_count=5,
            boundary=6.0,
            pinned=True,
        )
    )
    current = _doc_with_entries(
        BaselineEntry(
            kind="class",
            identity="a",
            file="src/a.py",
            member_count=8,
            boundary=6.0,
            pinned=True,
        )
    )
    findings = check_ratchet_tamper(previous, current, live_verdicts=None)
    assert len(findings) == 1
    assert "rose" in findings[0].message


def test_ratchet_tamper_clean_when_pinned_entry_unchanged() -> None:
    pin = BaselineEntry(
        kind="class",
        identity="a",
        file="src/a.py",
        member_count=5,
        boundary=6.0,
        pinned=True,
    )
    doc = _doc_with_entries(pin)
    assert check_ratchet_tamper(doc, doc, live_verdicts=None) == []


def test_ratchet_tamper_pin_removal_sanctioned_on_regen() -> None:
    # A full `baseline` regen re-derives entries from saturated verdicts
    # alone -- it is the sanctioned way to retire a pin (the only write path
    # that may legitimately drop one). The generation event is verified the
    # same way as for kind_stats: fresh generated_at AND stats equal to a
    # live recompute.
    live_verdicts = (_verdict("big", 20, boundary=17.0, saturated=True),)
    previous = generate(live_verdicts, generated_by="x", generated_at="t1", floor=4)
    previous["entries"].append(  # type: ignore[union-attr]
        {
            "kind": "class",
            "identity": "a",
            "file": "src/a.py",
            "member_count": 5,
            "boundary": 6.0,
            "bumps": [],
            "pinned": True,
        }
    )
    regenerated = generate(live_verdicts, generated_by="x", generated_at="t2", floor=4)
    # Sanity: the regen genuinely dropped the pin.
    assert entries_of(regenerated) != entries_of(previous)

    assert check_ratchet_tamper(previous, regenerated, live_verdicts=live_verdicts) == []


# ---------------------------------------------------------------------------
# check_tree end-to-end: the issue's acceptance positive control
# ---------------------------------------------------------------------------


def _class_source(name: str, count: int) -> str:
    # Method names deliberately do NOT end in a bare digit (`mNx`, not `mN`) --
    # a `<prefix><int>` sequence would be structurally reclassified as a
    # linear-ledger migration_runner by ledger.py and become exempt.
    methods = "\n".join(f"    def m{i}x(self): pass" for i in range(count))
    return f"\nclass {name}:\n{methods}\n"


def _build_repo(root: Path) -> None:
    """Four class APs with counts [2,3,4,20]: population hits the outlier
    FLOOR with a non-degenerate fence (q1=2, q3=4, iqr=2, boundary=7), so
    ``Big`` saturates and ``A``/``B``/``C`` are ordinary unsaturated points."""
    (root / "src" / "pkg").mkdir(parents=True)
    for name, count in (("A", 2), ("B", 3), ("C", 4), ("Big", 20)):
        (root / "src" / "pkg" / f"{name.lower()}.py").write_text(
            _class_source(name, count), encoding="utf-8"
        )


def _freeze_baseline(root: Path) -> None:
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes)
    kinds = sorted({p.kind for p in scan.points})
    verdicts = saturate_all(scan.points, kinds)
    document = generate(verdicts, generated_by="test", generated_at="t", floor=4)
    baseline_dir.dump(document, root / BASELINE_DIRNAME)


def _pin_entry(root: Path, *, identity: str, file: str, member_count: int) -> None:
    """Author a pinned entry row into the committed baseline -- the operator
    opt-in path (an entry file carrying ``"pinned": true``)."""
    document = baseline_dir.load(root / BASELINE_DIRNAME)
    document["entries"].append(  # type: ignore[union-attr]
        {
            "kind": "class",
            "identity": identity,
            "file": file,
            "member_count": member_count,
            "boundary": 7.0,
            "bumps": [],
            "pinned": True,
        }
    )
    baseline_dir.dump(document, root / BASELINE_DIRNAME)


def test_check_tree_blocks_pinned_class_growth_below_fence(tmp_path: Path) -> None:
    # Issue #1620 acceptance, positive control: a fixture class below the
    # fence with a pin row grows by one member -> check-tree reports `block`.
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)
    # A is unsaturated (2 members, fence 7); the operator pins it at its last
    # measured count -- the de-godded-state contract.
    _pin_entry(tmp_path, identity="A", file="src/pkg/a.py", member_count=2)

    # Grow A by one member (2 -> 3): still far below the fence at 7.
    (tmp_path / "src" / "pkg" / "a.py").write_text(_class_source("A", 3), encoding="utf-8")

    findings = check_tree(tmp_path)

    block_findings = [f for f in findings if f.severity == "block"]
    assert len(block_findings) == 1
    assert block_findings[0].identity == "A"
    assert block_findings[0].file == "src/pkg/a.py"
    assert "pinned ceiling 2" in block_findings[0].message
    # The hook path (check_file) surfaces the same finding for the host file.
    assert check_file("src/pkg/a.py", tmp_path)[0].severity == "block"


def test_check_tree_same_growth_without_pin_reports_nothing(tmp_path: Path) -> None:
    # Positive-control contrast (today's behavior): the identical growth on
    # the identical repo with NO pin row produces no finding -- the class can
    # silently regrow up to the fence.
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)

    (tmp_path / "src" / "pkg" / "a.py").write_text(_class_source("A", 3), encoding="utf-8")

    assert check_tree(tmp_path) == []


def test_check_tree_pin_with_headroom_allows_growth_up_to_ceiling(tmp_path: Path) -> None:
    # The pin records last-measured-plus-delta, not necessarily the exact
    # count: ceiling 4 allows growth to 4 but blocks at 5 -- still below the
    # fence either way.
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)
    _pin_entry(tmp_path, identity="A", file="src/pkg/a.py", member_count=4)

    (tmp_path / "src" / "pkg" / "a.py").write_text(_class_source("A", 4), encoding="utf-8")
    assert check_tree(tmp_path) == []

    (tmp_path / "src" / "pkg" / "a.py").write_text(_class_source("A", 5), encoding="utf-8")
    findings = check_tree(tmp_path)
    assert [f.identity for f in findings if f.severity == "block"] == ["A"]


def test_baseline_ratchet_cli_lowers_pinned_row_and_never_deletes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # End-to-end through the real `baseline --ratchet` path: a pinned row
    # whose class shrank ratchets down with the flag preserved, and a pinned
    # row whose class is unchanged survives a ratchet that drops nothing
    # else -- while the command adds no rows by itself.
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)
    _pin_entry(tmp_path, identity="A", file="src/pkg/a.py", member_count=2)

    # Shrink A 2 -> 1 member and B 3 -> 2 members (B has no baseline row at
    # all -- the ratchet must not invent one, pinned or otherwise).
    (tmp_path / "src" / "pkg" / "a.py").write_text(_class_source("A", 1), encoding="utf-8")
    (tmp_path / "src" / "pkg" / "b.py").write_text(_class_source("B", 2), encoding="utf-8")

    rc = main(["baseline", "--ratchet", "--root", str(tmp_path)])

    assert rc == 0
    document = baseline_dir.load(tmp_path / BASELINE_DIRNAME)
    entries = {(e.identity): e for e in entries_of(document)}
    assert entries["A"].pinned is True
    assert entries["A"].member_count == 1  # ratcheted down with the class
    assert "B" not in entries  # no row added for an unpinned, unsaturated point
    assert "Big" in entries  # the saturated row is untouched
