"""Per-kind event-level registry tests (issue #1838).

``instrumentation._LEVEL_BY_KIND`` is built once at import from the per-kind
entry files in ``src/charlie_work/event_levels/`` -- one ``<kind>.level``
file per kind, first line the level (``info``/``warning``/``error``), every
remaining line blank or a ``#`` rationale comment. The directory replaces
the monolithic dict that made every kind-adding PR collide on the same
lines of ``instrumentation.py`` (the shared-append-point defect class of
#1802/#1837).

These tests pin the loader's fail-closed contract -- a missing directory, a
stray file, a subdirectory, an invalid kind name, or a malformed body all
raise ``LevelRegistryError`` rather than silently skipping -- and verify
the shipped registry is what the module snapshot actually loaded. The
emit-site coverage direction (every literal kind passed to
``log_event``/``append_event``/``_record_event`` must be registered) is
enforced by ``test_event_kind_registry_exhaustive`` in
``test_instrumentation_event_kind_registry.py``, unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.instrumentation import (
    _EVENT_LEVELS_DIR,
    _LEVEL_BY_KIND,
    _LEVEL_VALUES,
    LevelRegistryError,
    _load_level_registry,
    _parse_level_entry,
)


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_registry_loads_strictly_and_matches_snapshot() -> None:
    """The committed ``event_levels/`` dir is exactly what ``_LEVEL_BY_KIND`` holds.

    The strict load raising here would mean a malformed entry reached the
    tree; the equality check pins that the module-level snapshot is built
    from this directory (not a stale in-module dict or a second source).
    """
    loaded = _load_level_registry(_EVENT_LEVELS_DIR)
    assert loaded, "the shipped event-level registry must not be empty"
    assert dict(_LEVEL_BY_KIND) == loaded
    assert set(loaded.values()) <= _LEVEL_VALUES


def test_shipped_registry_has_one_file_per_kind() -> None:
    """Every kind is its own file -- the property that removes the append point."""
    files = [p for p in _EVENT_LEVELS_DIR.iterdir() if p.is_file()]
    stems = {p.name[: -len(".level")] for p in files}
    assert len(files) == len(stems), "two files resolved to the same kind"
    assert stems == set(_LEVEL_BY_KIND)


def test_parse_level_entry_accepts_level_then_comments() -> None:
    text = "warning\n# Issue #1234: rationale line one\n# rationale line two\n\n"
    assert _parse_level_entry(text, name="foo.level") == "warning"


def test_parse_level_entry_rejects_bad_first_line() -> None:
    with pytest.raises(LevelRegistryError, match="first line"):
        _parse_level_entry("bogus\n", name="foo.level")


def test_parse_level_entry_rejects_empty_body() -> None:
    with pytest.raises(LevelRegistryError, match="first line"):
        _parse_level_entry("", name="foo.level")


def test_parse_level_entry_rejects_non_comment_body_line() -> None:
    with pytest.raises(LevelRegistryError, match="neither blank"):
        _parse_level_entry("info\nthis is not a comment\n", name="foo.level")


def test_load_reads_levels_and_ignores_comments(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.level", "error\n# rationale\n")
    _write(tmp_path, "beta.level", "info\n")
    _write(tmp_path, "gamma.level", "warning\n\n# spaced\n")
    assert _load_level_registry(tmp_path) == {
        "alpha": "error",
        "beta": "info",
        "gamma": "warning",
    }


def test_load_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(LevelRegistryError, match="not found"):
        _load_level_registry(tmp_path / "does_not_exist")


def test_load_rejects_stray_file(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.level", "info\n")
    _write(tmp_path, "notes.txt", "not an entry\n")
    with pytest.raises(LevelRegistryError, match="unexpected entry"):
        _load_level_registry(tmp_path)


def test_load_rejects_subdirectory(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.level", "info\n")
    (tmp_path / "nested").mkdir()
    with pytest.raises(LevelRegistryError, match="unexpected entry"):
        _load_level_registry(tmp_path)


def test_load_rejects_invalid_kind_name(tmp_path: Path) -> None:
    _write(tmp_path, "Not-A-Kind.level", "info\n")
    with pytest.raises(LevelRegistryError, match="invalid event-kind name"):
        _load_level_registry(tmp_path)


def test_load_rejects_empty_stem(tmp_path: Path) -> None:
    _write(tmp_path, ".level", "info\n")
    with pytest.raises(LevelRegistryError, match="invalid event-kind name"):
        _load_level_registry(tmp_path)


def test_load_rejects_malformed_entry_body(tmp_path: Path) -> None:
    _write(tmp_path, "alpha.level", "catastrophic\n")
    with pytest.raises(LevelRegistryError, match="first line"):
        _load_level_registry(tmp_path)
