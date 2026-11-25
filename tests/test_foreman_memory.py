"""``memory/*.md`` bounds and escapes (issue #3117)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from charlie_work.foreman import paths
from charlie_work.foreman.memory import (
    MAX_MEMORY_BYTES,
    MAX_MEMORY_FILES,
    delete_memory,
    list_memory,
    read_memory,
    write_memory,
)


def _foreman(tmp_path: Path) -> Path:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    return fdir


def test_write_list_read_delete_round_trip(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    assert write_memory(fdir, "decomp", "the plan\n").ok
    assert write_memory(fdir, "facts.md", "known\n").ok
    assert list_memory(fdir) == ("decomp", "facts")
    assert read_memory(fdir, "decomp").value == "the plan\n"
    assert delete_memory(fdir, "decomp").ok
    assert list_memory(fdir) == ("facts",)
    # Deleting an absent file is a clean no-op.
    assert delete_memory(fdir, "decomp").ok
    assert not read_memory(fdir, "decomp").ok


def test_size_bound_refused(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    big = "x" * (MAX_MEMORY_BYTES + 1)
    result = write_memory(fdir, "huge", big)
    assert not result.ok
    assert str(MAX_MEMORY_BYTES) in result.error
    exact = "x" * MAX_MEMORY_BYTES
    assert write_memory(fdir, "exact", exact).ok


def test_count_bound_refused_but_rewrite_ok(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    for i in range(MAX_MEMORY_FILES):
        assert write_memory(fdir, f"m{i:02d}", "x").ok
    result = write_memory(fdir, "one-too-many", "x")
    assert not result.ok
    assert str(MAX_MEMORY_FILES) in result.error
    # Rewriting an existing file still fits.
    assert write_memory(fdir, "m00", "y").ok


def test_name_escapes_refused(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    for bad in ("../outside", "sub/dir", "..", ""):
        assert not write_memory(fdir, bad, "x").ok
        assert not read_memory(fdir, bad).ok


def test_symlink_inside_memory_escaping_is_refused(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    mem = fdir / "memory"
    mem.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    try:
        (mem / "planted.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    result = write_memory(fdir, "planted", "overwritten\n")
    assert not result.ok
    assert outside.read_text(encoding="utf-8") == "secret"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a Windows reparse point")
def test_memory_junction_escape_is_refused(tmp_path: Path) -> None:
    """A ``memory/`` junction resolving outside the foreman dir refuses every name."""
    fdir = _foreman(tmp_path)
    outside = tmp_path / "outside-mem"
    outside.mkdir()
    import subprocess

    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(fdir / "memory"), str(outside)],
        check=True,
        capture_output=True,
    )
    result = write_memory(fdir, "note", "x")
    assert not result.ok
    assert "escapes" in result.error
    assert list_memory(fdir) == ()
