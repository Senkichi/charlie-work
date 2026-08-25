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

_SMALL_CLASS = """
class {name}:
    def m1(self): pass
    def m2(self): pass
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


def _build_repo(root: Path, big_member_count: int = 20) -> None:
    """Four class attachment points: three small (2 members), one big
    (default 20) -- population 4 hits the outlier FLOOR with a clean Tukey
    fence (boundary == 2), so `big` saturates and the others don't.
    """
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text(_SMALL_CLASS.format(name="A"), encoding="utf-8")
    (root / "src" / "pkg" / "b.py").write_text(_SMALL_CLASS.format(name="B"), encoding="utf-8")
    (root / "src" / "pkg" / "c.py").write_text(_SMALL_CLASS.format(name="C"), encoding="utf-8")
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


def test_check_tree_tamper_detects_hand_raised_baseline(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=20)
    _freeze_baseline(tmp_path)

    document = load(tmp_path / BASELINE_FILENAME)
    document["entries"][0]["member_count"] = 999  # hand-edit the JSON
    dump(document, tmp_path / BASELINE_FILENAME)

    findings = check_tree(tmp_path)

    tamper_findings = [f for f in findings if f.severity == "error" and "tamper" in f.message]
    assert len(tamper_findings) == 1
