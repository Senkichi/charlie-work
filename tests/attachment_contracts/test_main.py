"""Tests for the ``python -m charlie_work.attachment_contracts`` CLI (``__main__.py``).

Round-5 review: the ``baseline --refreeze`` branch of ``_cmd_baseline`` is
the only ratchet-shaped path that may legitimately raise the frozen per-kind
Tukey fence (issue #1614). These tests pin its contract end-to-end through
``check_tree``: the statistics it writes must equal a LIVE recompute of the
tree -- the same ``saturate_all`` verdict stream ``check_tree`` recomputes
for the ratchet-tamper guard -- or the guard reads the result as a forged
regen claim and flags it. A regression in the branch (e.g. saturating
against the frozen fence instead of the live one, or skipping the
``generated_at`` re-stamp) makes the end-to-end check fail.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.attachment_contracts.__main__ import main
from charlie_work.attachment_contracts.baseline import (
    BASELINE_FILENAME,
    dump,
    generate,
    load,
)
from charlie_work.attachment_contracts.check import check_tree
from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.archetypes import scan_tree
from charlie_work.attachment_contracts.outliers import saturate_all


def _class_source(name: str, count: int) -> str:
    # Method names deliberately do NOT end in a bare digit (`mNx`, not `mN`) --
    # a `<prefix><int>` sequence would be structurally reclassified as a
    # linear-ledger migration_runner by ledger.py and become exempt.
    methods = "\n".join(f"    def m{i}x(self): pass" for i in range(count))
    return f"\nclass {name}:\n{methods}\n"


def _build_freeze_repo(root: Path) -> None:
    """Eight class APs with member counts [1,2,3,4,6,8,16,20].

    Freeze-time fence (n=8, nearest-rank quartiles): q1=2, q3=8, iqr=6,
    boundary = 8 + 1.5*6 = 17. ``Big`` (20) saturates; nothing else does.
    """
    (root / "src" / "pkg").mkdir(parents=True)
    for name, count in (
        ("A", 1),
        ("B", 2),
        ("C", 3),
        ("D", 4),
        ("E", 6),
        ("F", 8),
        ("P", 16),
        ("Big", 20),
    ):
        (root / "src" / "pkg" / f"{name.lower()}.py").write_text(
            _class_source(name, count), encoding="utf-8"
        )


def _freeze_baseline(root: Path, generated_at: str = "t1") -> None:
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes)
    kinds = sorted({p.kind for p in scan.points})
    verdicts = saturate_all(scan.points, kinds)
    document = generate(verdicts, generated_by="test", generated_at=generated_at, floor=4)
    dump(document, root / BASELINE_FILENAME)


def test_baseline_refreeze_recomputes_kind_stats_and_passes_check_tree(
    tmp_path: Path, capsys
) -> None:
    # The sanctioned --refreeze scenario the round-1 reviewer verified by
    # hand: the tree grows so the LIVE fence legitimately rises (17 -> 23),
    # and --refreeze is the ratchet-shaped path that may record the raise.
    _build_freeze_repo(tmp_path)
    _freeze_baseline(tmp_path)

    # The committed baseline at the base ref -- what CI hands check_tree via
    # ``check-tree --base-ref``.
    previous = load(tmp_path / BASELINE_FILENAME)
    assert previous["kind_stats"]["class"]["boundary"] == 17.0

    # Grow the tree so the live fence rises while every frozen entry stays
    # put. New population [1,2,3,4,6,8,9,10,11,16,20,50] (n=12):
    # q1 rank = ceil(3) = 3 -> 3, q3 rank = ceil(9) = 9 -> 11,
    # iqr = 8, boundary = 11 + 1.5*8 = 23.
    for name, count in (("M9", 9), ("M10", 10), ("M11", 11), ("Huge", 50)):
        (tmp_path / "src" / "pkg" / f"{name.lower()}.py").write_text(
            _class_source(name, count), encoding="utf-8"
        )

    rc = main(["baseline", "--refreeze", "--root", str(tmp_path)])

    assert rc == 0
    assert "refrozen baseline written" in capsys.readouterr().out

    refrozen = load(tmp_path / BASELINE_FILENAME)
    # A generation event re-stamps generated_at and recomputes the frozen
    # fence from the LIVE population.
    assert refrozen["generated_at"] != "t1"
    assert refrozen["kind_stats"]["class"] == {
        "q3": 11.0,
        "iqr": 8.0,
        "boundary": 23.0,
        "population": 12,
    }
    # Entries ratcheted against the live fence: Huge entered, Big (now below
    # the raised fence) dropped.
    assert [e["identity"] for e in refrozen["entries"]] == ["Huge"]

    # Through the real check_tree call path the refrozen document verifies
    # as a sanctioned generation event: generated_at differs AND kind_stats
    # equals check_tree's own live recompute of this tree -- so the raised
    # frozen boundary is NOT flagged as tamper and nothing else fires either.
    findings = check_tree(tmp_path, previous_baseline_document=previous)
    assert findings == []


def test_baseline_refreeze_without_existing_baseline_fails(tmp_path: Path, capsys) -> None:
    # --refreeze is a ratchet-shaped path: with no committed baseline to
    # ratchet there is nothing to refreeze -- exit 1, no file written.
    _build_freeze_repo(tmp_path)

    rc = main(["baseline", "--refreeze", "--root", str(tmp_path)])

    assert rc == 1
    assert "no baseline" in capsys.readouterr().err
    assert not (tmp_path / BASELINE_FILENAME).exists()
