"""Tests for issue #1444: derived module map in worker dispatch prompts.

The module map is derived from the live source tree at packet build time --
never a hand-maintained list. These tests cover:

* ``build_module_map`` derives module names, docstring first lines, and
  public-surface sizes from a tree on disk with zero hardcoded module names.
* A newly added module appears in the next built map with no config change.
* Fail-soft: an unparseable file yields an omitted section (empty string) plus
  a ``worker_module_map_failed`` warning event, never a dispatch failure.
* The ``worker_module_map_failed`` event kind is registered at warning level
  so ``heartbeat_check.check_warning_events`` (which reads every
  ``level='warning'`` row from events.db, derived from the level column) is
  its consumer.
* ``module_map`` is a member of ``WORKER_PROMPT_KEYS`` and the real writer
  supplies it, so the drift check stays honest.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _LEVEL_BY_KIND
from charlie_work.module_map import build_module_map
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp, WORKER_PROMPT_KEYS


def _make_tree(src_root: Path) -> Path:
    """Create a minimal ``src/charlie_work`` tree and return the package dir."""
    pkg = src_root / "charlie_work"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""charlie_work package."""\n', encoding="utf-8")
    (pkg / "workflow.py").write_text(
        '"""The god file.\n\nLots of things live here.\n"""\n'
        "import os\n\n"
        "def run():\n    pass\n\n"
        "class App:\n    pass\n\n"
        "_private = 1\n"
        "PUBLIC = 2\n",
        encoding="utf-8",
    )
    (pkg / "config.py").write_text(
        '"""Config value objects."""\n'
        '__all__ = ["LabelConfig", "DispatchConfig"]\n'
        "class LabelConfig:\n    pass\n\n"
        "class DispatchConfig:\n    pass\n",
        encoding="utf-8",
    )
    return pkg


# ---------------------------------------------------------------------------
# build_module_map: derivation from the tree
# ---------------------------------------------------------------------------


def test_build_module_map_lists_modules_with_docstring_and_public_surface(
    tmp_path: Path,
) -> None:
    """The map lists every module's dotted name, docstring first line, and
    public-surface size, derived from the tree -- zero hardcoded names."""
    src = tmp_path / "src"
    _make_tree(src)

    section = build_module_map(src / "charlie_work", src)

    assert section.startswith("## Module map")
    # workflow.py: 3 public top-level names (run, App, PUBLIC); _private excluded.
    assert "`charlie_work.workflow`" in section
    assert "The god file." in section  # first docstring line
    assert "| 3 |" in section  # public surface size
    # config.py: __all__ length is 2 (LabelConfig, DispatchConfig).
    assert "`charlie_work.config`" in section
    assert "Config value objects." in section
    assert "| 2 |" in section
    # __init__.py: dotted name collapses to charlie_work.
    assert "`charlie_work`" in section


def test_build_module_map_uses_all_length_when_defined(tmp_path: Path) -> None:
    """When ``__all__`` is defined, its length is the public surface -- not the
    raw top-level name count (which may include non-exported helpers)."""
    src = tmp_path / "src"
    pkg = src / "charlie_work"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "shapes.py").write_text(
        '"""Shapes module."""\n'
        '__all__ = ["Circle", "Square"]\n'
        "class Circle:\n    pass\n\n"
        "class Square:\n    pass\n\n"
        "class _Internal:\n    pass\n\n"
        "def helper():\n    pass\n",
        encoding="utf-8",
    )

    section = build_module_map(pkg, src)

    # __all__ has 2 members, even though there are 3 public top-level names
    # (Circle, Square, helper) -- __all__ wins.
    assert "`charlie_work.shapes`" in section
    assert "| 2 |" in section


def test_build_module_map_counts_public_names_when_no_all(tmp_path: Path) -> None:
    """Without ``__all__``, the public surface is the count of top-level names
    not starting with an underscore (functions, classes, assignments)."""
    src = tmp_path / "src"
    pkg = src / "charlie_work"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "utils.py").write_text(
        '"""Utils."""\n'
        "def public_func():\n    pass\n\n"
        "async def async_func():\n    pass\n\n"
        "class PublicClass:\n    pass\n\n"
        "CONSTANT = 1\n\n"
        "_private = 2\n\n"
        "def _helper():\n    pass\n",
        encoding="utf-8",
    )

    section = build_module_map(pkg, src)

    # 4 public: public_func, async_func, PublicClass, CONSTANT
    assert "`charlie_work.utils`" in section
    assert "| 4 |" in section


def test_newly_added_module_appears_with_no_config_change(tmp_path: Path) -> None:
    """AC2: a newly added module appears in the next built map with no config
    change -- the map is derived from the tree, not a hardcoded list."""
    src = tmp_path / "src"
    _make_tree(src)

    first = build_module_map(src / "charlie_work", src)
    assert "charlie_work.new_module" not in first

    (src / "charlie_work" / "new_module.py").write_text(
        '"""A brand new module."""\ndef thing():\n    pass\n',
        encoding="utf-8",
    )

    second = build_module_map(src / "charlie_work", src)
    assert "`charlie_work.new_module`" in second
    assert "A brand new module." in second


def test_build_module_map_empty_when_no_package_dir(tmp_path: Path) -> None:
    """A missing package directory yields an empty string (omitted section),
    not a crash."""
    assert build_module_map(tmp_path / "nonexistent", tmp_path) == ""


def test_build_module_map_empty_when_no_py_files(tmp_path: Path) -> None:
    """A package directory with no .py files yields an empty string."""
    pkg = tmp_path / "src" / "charlie_work"
    pkg.mkdir(parents=True)
    assert build_module_map(pkg, tmp_path / "src") == ""


def test_build_module_map_sorted_deterministically(tmp_path: Path) -> None:
    """Rows are sorted by dotted module name for deterministic output."""
    src = tmp_path / "src"
    pkg = src / "charlie_work"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "zeta.py").write_text('"""z."""\n', encoding="utf-8")
    (pkg / "alpha.py").write_text('"""a."""\n', encoding="utf-8")
    (pkg / "mid.py").write_text('"""m."""\n', encoding="utf-8")

    section = build_module_map(pkg, src)

    # charlie_work (the __init__) sorts first; then alpha, mid, zeta.
    lines = [ln for ln in section.splitlines() if ln.startswith("| `")]
    names = [ln.split("`")[1] for ln in lines]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# Fail-soft through the real writer (_build_module_map_value)
# ---------------------------------------------------------------------------


def test_unparseable_file_omits_section_and_logs_warning_event(
    tmp_path: Path,
) -> None:
    """AC4: an unparseable file yields an omitted section (empty module_map)
    plus a ``worker_module_map_failed`` warning event, never a dispatch
    failure."""
    src = tmp_path / "src"
    pkg = src / "charlie_work"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "broken.py").write_text("def (\n", encoding="utf-8")  # SyntaxError

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    value = app._build_module_map_value(issue_number=99)

    assert value == "", "a parse failure must yield an empty module_map (omitted section)"

    # The warning event must be in events.db.
    db_path = paths.state_file.parent / "events.db"
    assert db_path.exists(), "events.db must exist after a map failure"
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT kind, level, payload FROM events WHERE kind = 'worker_module_map_failed'"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, f"expected one worker_module_map_failed event; got {rows}"
    kind, level, _payload = rows[0]
    assert kind == "worker_module_map_failed"
    assert level == "warning"


def test_missing_package_dir_omits_section_without_event(tmp_path: Path) -> None:
    """A repo with no ``src/charlie_work`` directory yields an empty module_map
    with no failure event -- the map is simply absent, not broken."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    value = app._build_module_map_value(issue_number=1)

    assert value == ""
    db_path = paths.state_file.parent / "events.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT 1 FROM events WHERE kind = 'worker_module_map_failed'"
            ).fetchall()
        finally:
            conn.close()
        assert rows == [], "a missing package dir is not a failure"


def test_write_worker_prompt_includes_module_map_from_live_tree(
    tmp_path: Path,
) -> None:
    """The real writer derives the map from the live tree and embeds it in the
    rendered prompt."""
    src = tmp_path / "src"
    _make_tree(src)

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh=None)

    prompt_path = app._write_worker_prompt({"number": 1, "title": "T", "url": "u", "body": "b"})
    text = prompt_path.read_text(encoding="utf-8")

    assert "## Module map" in text
    assert "`charlie_work.workflow`" in text


# ---------------------------------------------------------------------------
# Event kind registration + consumer (AC4)
# ---------------------------------------------------------------------------


def test_worker_module_map_failed_registered_as_warning() -> None:
    """The event kind is registered at ``warning`` level so
    ``heartbeat_check.check_warning_events`` -- which reads every
    ``level='warning'`` row from events.db (derived from the persisted level
    column, never a hardcoded kind list) -- is its consumer. An unregistered
    or non-warning kind would silently vanish from the heartbeat."""
    assert "worker_module_map_failed" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["worker_module_map_failed"] == "warning"


def test_module_map_is_a_worker_prompt_key() -> None:
    """``module_map`` must be a member of ``WORKER_PROMPT_KEYS`` so the drift
    check (``check_prompt_template_drift``) stays honest: the templates
    reference ``$module_map`` and the writer supplies it."""
    assert "module_map" in WORKER_PROMPT_KEYS
