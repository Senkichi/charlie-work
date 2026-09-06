"""Tests for check.py: check_file / check_tree orchestration.

Covers baseline block findings (with an attached redirect), the tamper guard,
and G6 (parse failure -> Finding severity error, never silently dropped).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.attachment_contracts.baseline import (
    BASELINE_FILENAME,
    dump,
    generate,
    load,
)
from charlie_work.attachment_contracts.check import check_file, check_tree
from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.archetypes import scan_tree
from charlie_work.attachment_contracts.outliers import saturate_all

_SMALL_CLASS_TEMPLATE = """
class {name}:
{methods}
"""

_BIG_CLASS_TEMPLATE = """
class Big:
{methods}
"""


def _big_class_source(count: int) -> str:
    # Method names deliberately do NOT end in a bare digit (`mNx`, not `mN`) --
    # a `<prefix><int>` sequence would be structurally reclassified as a
    # linear-ledger migration_runner by ledger.py and become exempt from the
    # ratchet, which would silently break every test in this module.
    methods = "\n".join(f"    def m{i}x(self): pass" for i in range(count))
    return _BIG_CLASS_TEMPLATE.format(methods=methods)


def _small_class_source(name: str, count: int) -> str:
    methods = "\n".join(f"    def s{i}x(self): pass" for i in range(count))
    return _SMALL_CLASS_TEMPLATE.format(name=name, methods=methods)


def _build_repo(root: Path, big_member_count: int = 20) -> None:
    """Four class attachment points: three small (2/3/4 members -- distinct
    counts so Q1 != Q3 and the IQR==0 degenerate-fence guard (finding #9)
    doesn't collapse the population to "nothing ever saturates"), one big
    (default 20) -- population 4 hits the outlier FLOOR with a non-degenerate
    Tukey fence (boundary == 7), so `big` saturates and the others don't.
    """
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text(_small_class_source("A", 2), encoding="utf-8")
    (root / "src" / "pkg" / "b.py").write_text(_small_class_source("B", 3), encoding="utf-8")
    (root / "src" / "pkg" / "c.py").write_text(_small_class_source("C", 4), encoding="utf-8")
    (root / "src" / "pkg" / "big.py").write_text(
        _big_class_source(big_member_count), encoding="utf-8"
    )


def _freeze_baseline(root: Path) -> None:
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes)
    kinds = sorted({p.kind for p in scan.points})
    verdicts = saturate_all(scan.points, kinds)
    document = generate(verdicts, generated_by="test", generated_at="t", floor=4)
    dump(document, root / BASELINE_FILENAME)


def test_check_tree_clean_when_matching_baseline(tmp_path: Path) -> None:
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)

    findings = check_tree(tmp_path)

    assert findings == []


def test_check_tree_blocks_when_saturated_point_grows_past_baseline(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=20)
    _freeze_baseline(tmp_path)

    # Grow Big past its frozen ceiling (20 -> 25); everything else unchanged.
    (tmp_path / "src" / "pkg" / "big.py").write_text(_big_class_source(25), encoding="utf-8")

    findings = check_tree(tmp_path)

    block_findings = [f for f in findings if f.severity == "block"]
    assert len(block_findings) == 1
    assert block_findings[0].identity == "Big"
    assert block_findings[0].file == "src/pkg/big.py"
    # G2: a redirect must be attached, pointing at a non-saturated sibling.
    assert block_findings[0].redirect in (
        "src/pkg/a.py",
        "src/pkg/b.py",
        "src/pkg/c.py",
    )


def test_check_file_filters_to_requested_path(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=20)
    _freeze_baseline(tmp_path)
    (tmp_path / "src" / "pkg" / "big.py").write_text(_big_class_source(25), encoding="utf-8")

    findings = check_file("src/pkg/big.py", tmp_path)
    other_findings = check_file("src/pkg/a.py", tmp_path)

    assert len(findings) == 1
    assert findings[0].file == "src/pkg/big.py"
    assert other_findings == []


def test_check_tree_g6_parse_failure_is_error_finding_never_silent(tmp_path: Path) -> None:
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)
    (tmp_path / "src" / "pkg" / "broken.py").write_text(
        "def broken(:\n    pass\n", encoding="utf-8"
    )

    findings = check_tree(tmp_path)

    parse_findings = [f for f in findings if f.file == "src/pkg/broken.py"]
    assert len(parse_findings) == 1
    assert parse_findings[0].severity == "error"
    assert "G6" in parse_findings[0].message


def test_check_file_g6_parse_failure_surfaces_for_single_file(tmp_path: Path) -> None:
    _build_repo(tmp_path)
    _freeze_baseline(tmp_path)
    (tmp_path / "src" / "pkg" / "broken.py").write_text(
        "def broken(:\n    pass\n", encoding="utf-8"
    )

    findings = check_file("src/pkg/broken.py", tmp_path)

    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "G6" in findings[0].message


def test_check_tree_no_baseline_yields_no_block_findings(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=20)
    # No baseline committed at all -- freeze-on-adopt hasn't happened yet.

    findings = check_tree(tmp_path)

    assert all(f.severity != "block" for f in findings)


def test_check_tree_new_saturated_ap_with_existing_baseline_blocks(tmp_path: Path) -> None:
    # Round-2 review finding #13: with a baseline already committed, a
    # brand-new file containing an already-saturated AP must not be frozen
    # silently -- it needs an explicit bump/redirect, same as growth past an
    # existing ceiling.
    _build_repo(tmp_path, big_member_count=8)
    _freeze_baseline(tmp_path)

    # A far larger outlier than `big` so it is still saturated after it
    # joins the population and shifts Q3/IQR upward.
    (tmp_path / "src" / "pkg" / "new_big.py").write_text(
        _big_class_source(50).replace("class Big:", "class NewBig:"), encoding="utf-8"
    )

    findings = check_tree(tmp_path)

    block_findings = [f for f in findings if f.severity == "block"]
    assert len(block_findings) == 1
    assert block_findings[0].identity == "NewBig"
    assert block_findings[0].file == "src/pkg/new_big.py"


def test_check_tree_tamper_detects_hand_raised_baseline(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=20)
    _freeze_baseline(tmp_path)

    document = load(tmp_path / BASELINE_FILENAME)
    document["entries"][0]["member_count"] = 999  # hand-edit the JSON
    dump(document, tmp_path / BASELINE_FILENAME)

    findings = check_tree(tmp_path)

    tamper_findings = [f for f in findings if f.severity == "error" and "tamper" in f.message]
    assert len(tamper_findings) == 1


# ---------------------------------------------------------------------------
# Issue #1614: freeze the per-kind Tukey fence per baseline
# ---------------------------------------------------------------------------


def _class_source(name: str, count: int) -> str:
    # Method names deliberately do NOT end in a bare digit (`mNx`, not `mN`) --
    # a `<prefix><int>` sequence would be structurally reclassified as a
    # linear-ledger migration_runner by ledger.py and become exempt.
    methods = "\n".join(f"    def m{i}x(self): pass" for i in range(count))
    return f"\nclass {name}:\n{methods}\n"


def _build_freeze_repo(root: Path) -> None:
    """Eight class APs with member counts [1,2,3,4,6,8,16,20].

    Hand-computed freeze-time fence (n=8, nearest-rank quartiles):
      q1 rank = ceil(0.25*8) = 2 -> sorted[1] = 2
      q3 rank = ceil(0.75*8) = 6 -> sorted[5] = 8
      iqr = 6; boundary = 8 + 1.5*6 = 17
    So ``Big`` (20) saturates and is baselined; ``P`` (16) does NOT saturate
    (16 > 17 is False) and has no baseline entry.
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


def test_check_tree_frozen_fence_keeps_findings_stable_when_median_module_added(
    tmp_path: Path,
) -> None:
    # Issue #1614: adding one median-sized module must NOT churn baseline
    # entries for files the PR never touched. With the fence FROZEN at the
    # baseline value (17), ``P`` (16, unchanged) stays not-saturated and
    # produces no finding. The positive control below shows the LIVE
    # recomputation WOULD move the fence and surface a spurious block.
    _build_freeze_repo(tmp_path)
    _freeze_baseline(tmp_path)

    # Sanity: the frozen baseline carries kind_stats and Big is the only entry.
    document = load(tmp_path / BASELINE_FILENAME)
    assert "kind_stats" in document
    assert document["kind_stats"]["class"]["boundary"] == 17.0
    assert [e["identity"] for e in document["entries"]] == ["Big"]

    # Add one median-sized module (count=8, at the old q3). The LIVE fence
    # drops to 15.5 (n=9: q1=3, q3=8, iqr=5, boundary=15.5), which would
    # newly saturate the UNCHANGED ``P`` (16). The FROZEN fence must not.
    (tmp_path / "src" / "pkg" / "median.py").write_text(
        _class_source("Median", 8), encoding="utf-8"
    )

    findings = check_tree(tmp_path)

    block_findings = [f for f in findings if f.severity == "block"]
    assert block_findings == [], (
        "frozen fence must not surface a block finding for the unchanged P "
        f"(got {[f.identity for f in block_findings]})"
    )


def test_check_tree_live_fence_positive_control_when_kind_stats_absent(
    tmp_path: Path,
) -> None:
    # Positive control for the bug (issue #1614): the SAME augmented repo,
    # but with a pre-#1614 baseline that carries NO kind_stats, falls back to
    # LIVE recomputation. The live fence drops to 15.5 and the UNCHANGED ``P``
    # (16) is newly saturated with no baseline entry -> a spurious block
    # finding. This is exactly the churn the freeze eliminates, and it proves
    # the frozen-fence test above is exercising the fix (not a vacuous pass).
    _build_freeze_repo(tmp_path)
    _freeze_baseline(tmp_path)

    # Strip kind_stats to simulate a pre-#1614 baseline (live fallback path).
    document = load(tmp_path / BASELINE_FILENAME)
    document.pop("kind_stats", None)
    dump(document, tmp_path / BASELINE_FILENAME)

    # Add the same median module as the frozen-fence test.
    (tmp_path / "src" / "pkg" / "median.py").write_text(
        _class_source("Median", 8), encoding="utf-8"
    )

    findings = check_tree(tmp_path)

    block_findings = [f for f in findings if f.severity == "block"]
    p_findings = [f for f in block_findings if f.identity == "P"]
    assert len(p_findings) == 1, (
        "positive control: live recomputation must surface a spurious block "
        "for the unchanged P when the fence is not frozen"
    )
    assert p_findings[0].file == "src/pkg/p.py"
