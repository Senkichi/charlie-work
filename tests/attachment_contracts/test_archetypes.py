"""Hand-computed expectations for archetype detection (scan_source, scan_tree)."""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.attachment_contracts.archetypes import iter_source_files, scan_source, scan_tree
from charlie_work.attachment_contracts.excludes import Excludes

TYPER_SRC = """
import typer

app = typer.Typer()

@app.command()
def hello():
    ...

@app.command()
def goodbye():
    ...

def helper():
    ...
"""


def test_typer_app_detected_with_command_members() -> None:
    points = scan_source(TYPER_SRC, "src/pkg/cli_typer.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "typer_app"
    assert point.identity == "cli_typer:app"
    assert point.file == "src/pkg/cli_typer.py"
    assert point.members == ("hello", "goodbye")
    assert point.member_count == 2
    assert point.is_linear_ledger is False


CLICK_GROUP_DECORATOR_SRC = """
import click

@click.group()
def cli():
    ...

@cli.command()
def run():
    ...

@cli.command()
def stop():
    ...

def not_a_command():
    ...
"""


def test_click_group_via_decorator_detected() -> None:
    points = scan_source(CLICK_GROUP_DECORATOR_SRC, "src/pkg/cli_click.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "click_group"
    assert point.identity == "cli_click:cli"
    assert point.members == ("run", "stop")


CLICK_GROUP_ASSIGN_SRC = """
import click

grp = click.Group()

@grp.command()
def a():
    ...

@grp.command
def b():
    ...
"""


def test_click_group_via_assign_and_bare_decorator() -> None:
    points = scan_source(CLICK_GROUP_ASSIGN_SRC, "src/pkg/cli_click2.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "click_group"
    assert point.identity == "cli_click2:grp"
    assert point.members == ("a", "b")


BLUEPRINT_SRC = """
from flask import Blueprint

bp = Blueprint("users", __name__)

@bp.route("/users")
def list_users():
    ...

@bp.get("/users/<id>")
def get_user(id):
    ...

@bp.post("/users")
def create_user():
    ...

def helper():
    ...
"""


def test_blueprint_detected_with_route_and_verb_members() -> None:
    points = scan_source(BLUEPRINT_SRC, "src/pkg/blueprint_mod.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "blueprint"
    assert point.identity == "blueprint_mod:bp"
    assert point.members == ("list_users", "get_user", "create_user")


CLASS_WITH_ASYNC_SRC = """
class Widget:
    def __init__(self):
        ...

    def render(self):
        ...

    async def fetch(self):
        ...
"""


def test_plain_class_includes_dunders_and_async_methods() -> None:
    points = scan_source(CLASS_WITH_ASYNC_SRC, "src/pkg/widget.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "class"
    assert point.identity == "Widget"
    assert point.members == ("__init__", "render", "fetch")
    assert point.is_linear_ledger is False


MIGRATION_CONTIGUOUS_SRC = """
class Migrator:
    def _migrate_v1(self):
        ...
    def _migrate_v2(self):
        ...
    def _migrate_v3(self):
        ...
    def _migrate_v4(self):
        ...
    def _migrate_v5(self):
        ...
"""


def test_migration_runner_contiguous_is_linear_ledger() -> None:
    points = scan_source(MIGRATION_CONTIGUOUS_SRC, "src/pkg/migrator.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "migration_runner"
    assert point.identity == "Migrator"
    assert point.is_linear_ledger is True
    assert point.members == (
        "_migrate_v1",
        "_migrate_v2",
        "_migrate_v3",
        "_migrate_v4",
        "_migrate_v5",
    )


MIGRATION_GAPPED_SRC = """
class Migrator:
    def _migrate_v1(self):
        ...
    def _migrate_v2(self):
        ...
    def _migrate_v4(self):
        ...
    def _migrate_v5(self):
        ...
"""


def test_migration_runner_single_gap_still_linear_ledger() -> None:
    points = scan_source(MIGRATION_GAPPED_SRC, "src/pkg/migrator_gapped.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "migration_runner"
    assert point.is_linear_ledger is True
    assert point.members == ("_migrate_v1", "_migrate_v2", "_migrate_v4", "_migrate_v5")


MIGRATION_INTRUDER_SRC = """
class Migrator:
    def _migrate_v1(self):
        ...
    def _migrate_v2(self):
        ...
    def _migrate_v3(self):
        ...
    def helper(self):
        ...
    def other_helper(self):
        ...
"""


def test_migration_runner_intruder_loses_ledger_exemption() -> None:
    # 3/5 matching members = 60% dominance < 80% -> falls back to plain "class",
    # not exempt from the ratchet.
    points = scan_source(MIGRATION_INTRUDER_SRC, "src/pkg/migrator_intruder.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "class"
    assert point.is_linear_ledger is False
    assert point.members == (
        "_migrate_v1",
        "_migrate_v2",
        "_migrate_v3",
        "helper",
        "other_helper",
    )


MODULE_LEDGER_SRC = """
def _apply_m1():
    ...

def _apply_m2():
    ...

def _apply_m3():
    ...
"""


def test_module_level_ledger_detected_without_class() -> None:
    points = scan_source(MODULE_LEDGER_SRC, "src/pkg/migrations.py")
    assert len(points) == 1
    point = points[0]
    assert point.kind == "migration_runner"
    assert point.identity == "module:migrations"
    assert point.is_linear_ledger is True
    assert point.members == ("_apply_m1", "_apply_m2", "_apply_m3")


TEST_MODULE_SRC = """
import pytest

def test_one():
    ...

def test_two():
    ...

class TestGroup:
    def test_a(self):
        ...
    def test_b(self):
        ...

def helper():
    ...
"""


def test_test_module_counts_functions_and_test_classes_as_one_member_each() -> None:
    points = scan_source(TEST_MODULE_SRC, "tests/test_sample.py")
    by_kind = {p.kind: p for p in points}
    assert set(by_kind) == {"class", "test_module"}

    class_point = by_kind["class"]
    assert class_point.identity == "TestGroup"
    assert class_point.members == ("test_a", "test_b")

    module_point = by_kind["test_module"]
    assert module_point.identity == "tests/test_sample.py::module"
    assert module_point.members == ("test_one", "test_two", "TestGroup")


MULTI_AP_SRC = """
import typer
from flask import Blueprint

app = typer.Typer()

@app.command()
def sync():
    ...

bp = Blueprint("api", __name__)

@bp.get("/health")
def health():
    ...

class Service:
    def start(self):
        ...
    def stop(self):
        ...
"""


def test_single_file_emits_multiple_attachment_points() -> None:
    points = scan_source(MULTI_AP_SRC, "src/pkg/multi.py")
    kinds = [p.kind for p in points]
    assert kinds == ["typer_app", "blueprint", "class"]

    typer_point, blueprint_point, class_point = points
    assert typer_point.identity == "multi:app"
    assert typer_point.members == ("sync",)
    assert blueprint_point.identity == "multi:bp"
    assert blueprint_point.members == ("health",)
    assert class_point.identity == "Service"
    assert class_point.members == ("start", "stop")


BROKEN_SRC = "def broken(:\n    pass\n"


def test_scan_source_raises_syntax_error_on_broken_source() -> None:
    with pytest.raises(SyntaxError):
        scan_source(BROKEN_SRC, "src/pkg/broken.py")


def test_scan_tree_routes_parse_failures_and_still_scans_valid_files(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "broken.py").write_text(BROKEN_SRC, encoding="utf-8")
    (src_dir / "widget.py").write_text(CLASS_WITH_ASYNC_SRC, encoding="utf-8")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sample.py").write_text(TEST_MODULE_SRC, encoding="utf-8")

    # Excluded directory: must never contribute points or parse_failures.
    cache_dir = src_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "stale.py").write_text(BROKEN_SRC, encoding="utf-8")

    result = scan_tree(tmp_path, Excludes())

    assert result.parse_failures == ("src/pkg/broken.py",)
    assert "src/pkg/__pycache__/stale.py" not in result.parse_failures

    files_seen = {p.file for p in result.points}
    assert files_seen == {"src/pkg/widget.py", "tests/test_sample.py"}

    # Deterministic ordering: sorted by (file, identity).
    ordering = [(p.file, p.identity) for p in result.points]
    assert ordering == sorted(ordering)


def test_scan_tree_honors_configured_exclude_globs(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "src" / "legacy"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "old.py").write_text(CLASS_WITH_ASYNC_SRC, encoding="utf-8")

    kept_dir = tmp_path / "src" / "current"
    kept_dir.mkdir(parents=True)
    (kept_dir / "new.py").write_text(CLASS_WITH_ASYNC_SRC, encoding="utf-8")

    excludes = Excludes(exclude_globs=("src/legacy/*",))
    result = scan_tree(tmp_path, excludes)

    files_seen = {p.file for p in result.points}
    assert files_seen == {"src/current/new.py"}


def test_scan_tree_missing_src_and_tests_dirs_returns_empty(tmp_path: Path) -> None:
    result = scan_tree(tmp_path, Excludes())
    assert result.points == ()
    assert result.parse_failures == ()


# ---------------------------------------------------------------------------
# finding #7's root cause: same-named nested local classes get disambiguated
# identities so they never collide in baseline.py's (kind, file, identity) key
# ---------------------------------------------------------------------------

_DUPLICATE_NESTED_CLASS_SRC = """
def test_one():
    class FakeGitHub:
        def a(self): pass

def test_two():
    class FakeGitHub:
        def b(self): pass
        def c(self): pass
"""


def test_same_named_nested_classes_get_distinct_qualified_identities() -> None:
    points = scan_source(_DUPLICATE_NESTED_CLASS_SRC, "tests/test_worker.py")
    class_points = [p for p in points if p.kind == "class"]

    identities = [p.identity for p in class_points]
    assert len(identities) == len(set(identities)), f"collided: {identities}"
    assert identities == ["test_one.FakeGitHub", "test_two.FakeGitHub"]


def test_top_level_class_identity_stays_bare_for_baseline_stability() -> None:
    points = scan_source("class Widget:\n    def render(self): pass\n", "src/w.py")
    assert points[0].identity == "Widget"


# ---------------------------------------------------------------------------
# finding #2: a decode/OS failure on the read must not crash the tree scan
# ---------------------------------------------------------------------------


def test_scan_tree_routes_undecodable_file_to_parse_failures(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "good.py").write_text(CLASS_WITH_ASYNC_SRC, encoding="utf-8")
    # Invalid UTF-8 byte sequence -- read_text(encoding="utf-8") raises
    # UnicodeDecodeError, previously uncaught (outside the SyntaxError-only
    # try/except), crashing the whole scan.
    (src_dir / "bad_encoding.py").write_bytes(b"\xff\xfe garbage not utf-8")

    result = scan_tree(tmp_path, Excludes())

    assert "src/pkg/bad_encoding.py" in result.parse_failures
    assert any(p.file == "src/pkg/good.py" for p in result.points)


# ---------------------------------------------------------------------------
# iter_source_files: the full scanned-file universe (finding #3b's dependency)
# ---------------------------------------------------------------------------


def test_iter_source_files_includes_files_with_no_archetype_match(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "bare.py").write_text("def helper():\n    pass\n", encoding="utf-8")

    files = iter_source_files(tmp_path, Excludes())

    assert files == ("src/pkg/bare.py",)
    # bare.py has no class/typer/blueprint archetype -- confirms scan_tree's
    # points would NOT have seen this file at all without iter_source_files.
    scanned = scan_tree(tmp_path, Excludes())
    assert scanned.points == ()


def test_scan_tree_honors_content_overrides(tmp_path: Path) -> None:
    src_dir = tmp_path / "src" / "pkg"
    src_dir.mkdir(parents=True)
    (src_dir / "a.py").write_text("class OnDisk:\n    def m(self): pass\n", encoding="utf-8")

    result = scan_tree(
        tmp_path,
        Excludes(),
        content_overrides={"src/pkg/a.py": "class Overridden:\n    def m(self): pass\n"},
    )

    assert [p.identity for p in result.points] == ["Overridden"]
