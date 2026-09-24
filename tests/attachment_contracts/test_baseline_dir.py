"""Tests for issue #1839: the per-entry ``.attachment-budgets/`` directory.

The single ``.attachment-budgets.json`` registry was a shared append point:
two PRs touching DIFFERENT attachment points still edited the same JSON
array and conflicted on merge (and a conflicting PR gets no pull_request CI
run at all -- the #1801 silent-stall class). The baseline now lives as one
file per ``(kind, file, identity)`` entry under ``.attachment-budgets/entries/``
mirroring the guarded file's repo-relative path, plus a ``meta.json`` for the
top-level keys. These tests pin the layout, the fail-closed load contract,
stale-entry pruning, and the legacy single-file fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work.attachment_contracts.baseline import (
    BASELINE_DIRNAME,
    BASELINE_FILENAME,
    TamperError,
    entries_of,
)
from charlie_work.attachment_contracts.baseline_dir import (
    ENTRIES_DIRNAME,
    META_FILENAME,
    document_files,
    dump,
    entry_relpath,
    find_baseline,
    load,
    load_dir,
    load_files,
    read_files,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _entry_dict(
    identity: str,
    file: str,
    member_count: int = 10,
    kind: str = "class",
    boundary: float = 4.0,
    bumps: list[dict] | None = None,
) -> dict:
    return {
        "kind": kind,
        "identity": identity,
        "file": file,
        "member_count": member_count,
        "boundary": boundary,
        "bumps": bumps if bumps is not None else [],
    }


def _doc(entries: list[dict], **meta_overrides) -> dict:
    document = {
        "version": 1,
        "generated_by": "test",
        "generated_at": "2026-09-05T00:00:00Z",
        "floor": 4,
        "entries": entries,
    }
    document.update(meta_overrides)
    return document


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _entry(raw: dict):
    (entry,) = entries_of({"entries": [raw]})
    return entry


# ---------------------------------------------------------------------------
# Layout + round-trip
# ---------------------------------------------------------------------------


def test_dump_writes_meta_and_per_entry_files(tmp_path: Path) -> None:
    document = _doc(
        [
            _entry_dict("Big", "src/pkg/big.py"),
            _entry_dict("tests/test_x.py::module", "tests/test_x.py", kind="test_module"),
        ]
    )

    dump(document, tmp_path / BASELINE_DIRNAME)

    meta_path = tmp_path / BASELINE_DIRNAME / META_FILENAME
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    # Top-level keys live in meta.json; the entries array does NOT -- the
    # whole point of the layout is that no single file aggregates them.
    assert meta["generated_by"] == "test"
    assert meta["floor"] == 4
    assert "entries" not in meta

    assert (tmp_path / BASELINE_DIRNAME / "entries/src/pkg/big.py/class--Big.json").is_file()
    assert (
        tmp_path / BASELINE_DIRNAME / "entries/tests/test_x.py/test_module--module.json"
    ).is_file()


def test_entry_path_mirrors_host_file_and_strips_identity_prefix(tmp_path: Path) -> None:
    """The file layout derives from the entry's own key so diffs read
    naturally: ``entries/<file>/<kind>--<identity-minus-'file::'>.json``."""
    entry = _entry_dict("src/pkg/big.py::Outer.Inner", "src/pkg/big.py")
    rel = entry_relpath(_entry(entry))
    assert rel == "entries/src/pkg/big.py/class--Outer.Inner.json"


def test_entry_path_escapes_non_file_prefix_identities(tmp_path: Path) -> None:
    """An identity that does not start with ``file::`` is percent-escaped
    wholesale (Windows-unsafe and ``/``-bearing characters cannot appear in
    the leaf name)."""
    entry = _entry_dict("cli:app", "src/pkg/cli.py", kind="typer_app")
    rel = entry_relpath(_entry(entry))
    leaf = rel.split("/")[-1]
    assert leaf == "typer_app--cli%3Aapp.json"
    assert rel == f"entries/src/pkg/cli.py/{leaf}"


def test_distinct_entries_live_in_distinct_files(tmp_path: Path) -> None:
    """The #1839 property: two entries -> two files, so PRs touching
    different attachment points edit different paths and merge cleanly."""
    document = _doc(
        [
            _entry_dict("A", "src/pkg/a.py"),
            _entry_dict("B", "src/pkg/b.py"),
        ]
    )
    files = document_files(document)
    entry_paths = [rel for rel in files if rel != META_FILENAME]
    assert len(entry_paths) == 2
    assert len(set(entry_paths)) == 2


def test_round_trip_preserves_bumps_kind_stats_and_mode(tmp_path: Path) -> None:
    bump = {"to": 20, "reason": "reviewed growth", "actor": "worker", "ack": "#123"}
    document = _doc(
        [_entry_dict("Big", "src/pkg/big.py", bumps=[bump])],
        kind_stats={"class": {"q3": 4.0, "iqr": 2.0, "boundary": 8.0, "population": 6}},
        mode="enforce",
    )
    dump(document, tmp_path / BASELINE_DIRNAME)

    loaded = load(tmp_path / BASELINE_DIRNAME)

    assert loaded["kind_stats"] == document["kind_stats"]
    assert loaded["mode"] == "enforce"
    (entry,) = entries_of(loaded)
    assert entry.identity == "Big"
    (loaded_bump,) = entry.bumps
    assert loaded_bump.to == 20
    assert loaded_bump.actor == "worker"
    assert loaded_bump.ack == "#123"


def test_load_files_matches_load_dir(tmp_path: Path) -> None:
    """``load_files`` is the pure-map form used by git-ref reconstruction;
    it must agree byte-for-byte with the on-disk reader."""
    document = _doc([_entry_dict("Big", "src/pkg/big.py")])
    dump(document, tmp_path / BASELINE_DIRNAME)

    from_disk = load_dir(tmp_path / BASELINE_DIRNAME)
    from_map = load_files(read_files(tmp_path / BASELINE_DIRNAME))

    assert from_map == from_disk


def test_empty_baseline_directory_is_valid(tmp_path: Path) -> None:
    """A zero-entry baseline is a directory containing only meta.json --
    git cannot track an empty ``entries/`` dir, so its absence is normal."""
    document = _doc([])
    dump(document, tmp_path / BASELINE_DIRNAME)

    loaded = load(tmp_path / BASELINE_DIRNAME)
    assert loaded["entries"] == []


def test_top_level_non_entry_files_are_tolerated(tmp_path: Path) -> None:
    """A README.md (or future sibling) at the directory root is not part of
    the entry set and must not trip the fail-closed loader."""
    document = _doc([_entry_dict("Big", "src/pkg/big.py")])
    dump(document, tmp_path / BASELINE_DIRNAME)
    _write(tmp_path / BASELINE_DIRNAME / "README.md", "# attachment budgets\n")

    loaded = load(tmp_path / BASELINE_DIRNAME)
    assert len(entries_of(loaded)) == 1


# ---------------------------------------------------------------------------
# Fail-closed loading (TamperError, never a silent pass)
# ---------------------------------------------------------------------------


def test_missing_meta_fails_closed(tmp_path: Path) -> None:
    directory = tmp_path / BASELINE_DIRNAME
    _write(
        directory / "entries/src/pkg/big.py/class--Big.json",
        json.dumps(_entry_dict("Big", "src/pkg/big.py")),
    )

    with pytest.raises(TamperError):
        load(directory)


def test_stray_non_entry_file_under_entries_fails_closed(tmp_path: Path) -> None:
    document = _doc([_entry_dict("Big", "src/pkg/big.py")])
    dump(document, tmp_path / BASELINE_DIRNAME)
    _write(tmp_path / BASELINE_DIRNAME / "entries" / "notes.txt", "hello\n")

    with pytest.raises(TamperError):
        load(tmp_path / BASELINE_DIRNAME)


def test_non_object_entry_file_fails_closed(tmp_path: Path) -> None:
    meta = {k: v for k, v in _doc([]).items() if k != "entries"}
    _write(tmp_path / BASELINE_DIRNAME / META_FILENAME, json.dumps(meta))
    _write(
        tmp_path / BASELINE_DIRNAME / "entries/src/pkg/big.py/class--Big.json",
        "[1, 2, 3]",
    )

    with pytest.raises(TamperError):
        load(tmp_path / BASELINE_DIRNAME)


def test_malformed_entry_json_fails_closed(tmp_path: Path) -> None:
    meta = {k: v for k, v in _doc([]).items() if k != "entries"}
    _write(tmp_path / BASELINE_DIRNAME / META_FILENAME, json.dumps(meta))
    _write(
        tmp_path / BASELINE_DIRNAME / "entries/src/pkg/big.py/class--Big.json",
        "not json {{{",
    )

    with pytest.raises(TamperError):
        load(tmp_path / BASELINE_DIRNAME)


def test_malformed_meta_json_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path / BASELINE_DIRNAME / META_FILENAME, "not json {{{")

    with pytest.raises(TamperError):
        load(tmp_path / BASELINE_DIRNAME)


def test_meta_must_not_carry_entries_key(tmp_path: Path) -> None:
    """A meta.json smuggling an ``entries`` array would re-create the shared
    registry inside the directory -- reject it outright."""
    _write(
        tmp_path / BASELINE_DIRNAME / META_FILENAME,
        json.dumps(_doc([_entry_dict("Big", "src/pkg/big.py")])),
    )

    with pytest.raises(TamperError):
        load(tmp_path / BASELINE_DIRNAME)


def test_filename_content_mismatch_fails_closed(tmp_path: Path) -> None:
    """The filename is derived from the entry's key: a file named for one
    entry but carrying another's content is inconsistent (host rename without
    the rename, or hand-forgery) and fails closed."""
    meta = {k: v for k, v in _doc([]).items() if k != "entries"}
    _write(tmp_path / BASELINE_DIRNAME / META_FILENAME, json.dumps(meta))
    _write(
        tmp_path / BASELINE_DIRNAME / "entries/src/pkg/big.py/class--Big.json",
        json.dumps(_entry_dict("Other", "src/pkg/other.py")),
    )

    with pytest.raises(TamperError):
        load(tmp_path / BASELINE_DIRNAME)


def test_unsafe_entry_file_field_rejected() -> None:
    """A ``file`` that escapes the baseline directory (``..``, absolute,
    backslash) must never reach the filesystem -- reject at payload-build
    time, before any write."""
    document = _doc([_entry_dict("Evil", "../../outside.py")])

    with pytest.raises(TamperError):
        document_files(document)


def test_duplicate_entry_key_fails_closed() -> None:
    document = _doc(
        [
            _entry_dict("Big", "src/pkg/big.py"),
            _entry_dict("Big", "src/pkg/big.py"),
        ]
    )

    with pytest.raises(TamperError):
        document_files(document)


def test_load_dir_on_missing_directory_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TamperError):
        load_dir(tmp_path / BASELINE_DIRNAME)


# ---------------------------------------------------------------------------
# Stale-entry pruning on write
# ---------------------------------------------------------------------------


def test_dump_prunes_stale_entries_and_empty_dirs(tmp_path: Path) -> None:
    directory = tmp_path / BASELINE_DIRNAME
    document = _doc(
        [
            _entry_dict("A", "src/pkg/a.py"),
            _entry_dict("B", "src/pkg/sub/b.py"),
        ]
    )
    dump(document, directory)
    assert (directory / "entries/src/pkg/sub/b.py/class--B.json").is_file()

    # Re-dump without B: its file is deleted and the emptied subdir pruned
    # (git cannot track empty dirs, so leaving the husk litters the tree).
    dump(_doc([_entry_dict("A", "src/pkg/a.py")]), directory)

    assert (directory / "entries/src/pkg/a.py/class--A.json").is_file()
    assert not (directory / "entries/src/pkg/sub").exists()


def test_dump_leaves_top_level_extras_alone(tmp_path: Path) -> None:
    directory = tmp_path / BASELINE_DIRNAME
    dump(_doc([]), directory)
    _write(directory / "README.md", "# attachment budgets\n")

    dump(_doc([_entry_dict("A", "src/a.py")]), directory)

    assert (directory / "README.md").is_file()


# ---------------------------------------------------------------------------
# find_baseline + layout dispatch
# ---------------------------------------------------------------------------


def test_find_baseline_prefers_directory(tmp_path: Path) -> None:
    _write(tmp_path / BASELINE_FILENAME, json.dumps(_doc([])))
    dump(_doc([]), tmp_path / BASELINE_DIRNAME)

    assert find_baseline(tmp_path) == tmp_path / BASELINE_DIRNAME


def test_find_baseline_falls_back_to_legacy_file(tmp_path: Path) -> None:
    _write(tmp_path / BASELINE_FILENAME, json.dumps(_doc([])))

    assert find_baseline(tmp_path) == tmp_path / BASELINE_FILENAME


def test_find_baseline_returns_none_when_absent(tmp_path: Path) -> None:
    assert find_baseline(tmp_path) is None


def test_legacy_file_round_trips_through_dispatch(tmp_path: Path) -> None:
    """Pre-#1839 checkouts keep working: the same load/dump entry points read
    and write the single-file layout when that is what exists on disk."""
    path = tmp_path / BASELINE_FILENAME
    document = _doc([_entry_dict("Big", "src/pkg/big.py")])

    dump(document, path)
    loaded = load(path)

    assert loaded["entries"] == document["entries"]


# ---------------------------------------------------------------------------
# Committed store
# ---------------------------------------------------------------------------


def test_committed_baseline_file_loads() -> None:
    """The committed baseline's FILES load through the file-map path.

    Relocated verbatim-by-name from ``test_committed_baseline.py``
    (collect-only gate leaf continuity, issue #1538 -- the same pattern as
    #1828's ``test_cw_1518_...`` move): the pre-#1839 assertion "the single
    ``.attachment-budgets.json`` document loads" is gone by design -- the
    shared append point no longer exists. What remains true, and what this
    pins, is the property the name describes literally: every file of the
    committed ``.attachment-budgets/`` store loads through ``read_files`` +
    ``load_files``, the same reconstruction path the review packet feeds
    with PR-head content rebuilt from a diff.
    """
    files = read_files(_REPO_ROOT / BASELINE_DIRNAME)
    assert META_FILENAME in files
    assert any(rel.startswith(ENTRIES_DIRNAME + "/") for rel in files)
    document = load_files(files)
    assert document["entries"], "committed baseline must carry at least one entry"
