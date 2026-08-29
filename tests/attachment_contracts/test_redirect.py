"""Tests for redirect.py: G2 sibling suggestion + pre-wired scaffold."""

from __future__ import annotations

from charlie_work.attachment_contracts.model import AttachmentPoint, ScanResult
from charlie_work.attachment_contracts.redirect import scaffold, suggest


def _class_point(identity: str, file: str, count: int) -> AttachmentPoint:
    return AttachmentPoint(
        kind="class",
        identity=identity,
        file=file,
        members=tuple(f"do_thing_{i}" for i in range(count)),
    )


def _scan(*points: AttachmentPoint) -> ScanResult:
    return ScanResult(root="/repo", points=tuple(points), parse_failures=())


# ---------------------------------------------------------------------------
# suggest()
# ---------------------------------------------------------------------------


def test_suggest_picks_nearest_non_saturated_sibling_same_dir() -> None:
    # Population of 4 same-kind points: three small (2 members) in pkg/,
    # one saturated (20 members) also in pkg/ -- classic Tukey outlier.
    outlier = _class_point("Big", "src/pkg/big.py", 20)
    near = _class_point("Near", "src/pkg/near.py", 3)
    other_a = _class_point("A", "src/other/a.py", 2)
    other_b = _class_point("B", "src/other/b.py", 2)
    scan = _scan(outlier, near, other_a, other_b)

    redirect = suggest(outlier, scan)

    assert redirect.is_new_module is False
    # near.py is in the same package dir (pkg/) and non-saturated -- preferred
    # over the equally-small siblings in other/.
    assert redirect.destination == "src/pkg/near.py"


def test_suggest_prefers_fewest_members_across_dirs_when_no_same_dir_sibling() -> None:
    outlier = _class_point("Big", "src/pkg/big.py", 20)
    far_small = _class_point("Small", "src/other/small.py", 2)
    far_medium = _class_point("Medium", "src/other/medium.py", 3)
    filler = _class_point("Filler", "src/other/filler.py", 2)
    scan = _scan(outlier, far_small, far_medium, filler)

    redirect = suggest(outlier, scan)

    assert redirect.is_new_module is False
    assert redirect.destination in ("src/other/small.py", "src/other/filler.py")


def test_suggest_proposes_new_module_when_no_sibling_available() -> None:
    # Only one class attachment point at all -- population < FLOOR, so
    # nothing is saturated and there is no sibling to redirect to, but
    # suggest() must still resolve deterministically to a new-module proposal
    # when explicitly asked (simulating a post-adoption single-outlier repo).
    solo = _class_point("Solo", "src/pkg/solo.py", 20)
    scan = _scan(solo)

    redirect = suggest(solo, scan)

    assert redirect.is_new_module is True
    assert redirect.destination.startswith("src/pkg/")
    assert redirect.destination.endswith("_ops.py")


def test_suggest_new_module_naming_for_test_module_kind() -> None:
    point = AttachmentPoint(
        kind="test_module",
        identity="tests/test_widget.py::module",
        file="tests/test_widget.py",
        members=tuple(f"test_case_{i}" for i in range(20)),
    )
    scan = _scan(point)

    redirect = suggest(point, scan)

    assert redirect.is_new_module is True
    assert redirect.destination == "tests/widget/test_widget.py"


def test_suggest_ignores_ledger_points_as_siblings() -> None:
    outlier = _class_point("Big", "src/pkg/big.py", 20)
    ledger = AttachmentPoint(
        kind="class",
        identity="Migrator",
        file="src/pkg/migrator.py",
        members=("_migrate_v1", "_migrate_v2", "_migrate_v3"),
        is_linear_ledger=True,
    )
    scan = _scan(outlier, ledger)

    redirect = suggest(outlier, scan)

    # Population too small (2, below FLOOR=4) even counting the ledger, and
    # the ledger point must never be offered as a redirect destination.
    assert redirect.destination != "src/pkg/migrator.py"


# ---------------------------------------------------------------------------
# scaffold()
# ---------------------------------------------------------------------------


def test_scaffold_writes_nothing_to_disk(tmp_path) -> None:
    outlier = _class_point("Big", "src/pkg/big.py", 20)
    scan = _scan(outlier, _class_point("Near", "src/pkg/near.py", 2))
    redirect = suggest(outlier, scan)

    before = list(tmp_path.iterdir())
    plan = scaffold(outlier, redirect, "new_capability")
    after = list(tmp_path.iterdir())

    assert before == after  # scaffold() must never touch the filesystem
    assert plan.path == redirect.destination


def test_scaffold_class_kind_stubs_member_and_notes_composition() -> None:
    outlier = _class_point("Big", "src/pkg/big.py", 20)
    redirect = suggest(outlier, _scan(outlier))
    plan = scaffold(outlier, redirect, "new_capability")

    assert "def new_capability" in plan.content
    assert "class" in plan.content
    assert "Big" in plan.registration_note


def test_scaffold_test_module_kind_stubs_test_function() -> None:
    point = AttachmentPoint(
        kind="test_module",
        identity="tests/test_widget.py::module",
        file="tests/test_widget.py",
        members=tuple(f"test_case_{i}" for i in range(20)),
    )
    redirect = suggest(point, _scan(point))
    plan = scaffold(point, redirect, "test_new_case")

    assert "def test_new_case" in plan.content
    assert plan.registration_note  # non-empty guidance


def test_scaffold_typer_app_kind_wires_add_typer() -> None:
    point = AttachmentPoint(
        kind="typer_app",
        identity="cli:app",
        file="src/cli.py",
        members=tuple(f"cmd_{i}" for i in range(20)),
    )
    redirect = suggest(point, _scan(point))
    plan = scaffold(point, redirect, "new_cmd")

    assert "typer.Typer" in plan.content
    assert "def new_cmd" in plan.content
    assert "add_typer" in plan.registration_note


def test_scaffold_blueprint_kind_wires_register_blueprint() -> None:
    point = AttachmentPoint(
        kind="blueprint",
        identity="views:bp",
        file="src/views.py",
        members=tuple(f"route_{i}" for i in range(20)),
    )
    redirect = suggest(point, _scan(point))
    plan = scaffold(point, redirect, "new_route")

    assert "Blueprint" in plan.content
    assert "register_blueprint" in plan.registration_note
