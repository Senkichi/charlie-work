"""The foreman path layer: name validation, containment, discovery (issue #3117)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from charlie_work.foreman import paths


def test_foreman_name_regex() -> None:
    assert paths.is_foreman_name("the-gang-migrates-the-transport")
    assert paths.is_foreman_name("a")
    assert paths.is_foreman_name("a1-b2")
    for bad in ("", "-lead", "Upper", "has_underscore", "_shared", "..", "a/b", "a b"):
        assert not paths.is_foreman_name(bad), bad


def test_foreman_dir_resolves_inside_foremen(tmp_path: Path) -> None:
    root = paths.foreman_dir(tmp_path, "g1")
    assert root.name == "g1"
    assert root.parent.name == paths.FOREMEN_DIRNAME


def test_foreman_dir_rejects_bad_names(tmp_path: Path) -> None:
    for bad in ("..", "_shared", "has_underscore", "a/b", ""):
        with pytest.raises(ValueError, match="invalid foreman name"):
            paths.foreman_dir(tmp_path, bad)


def test_foreman_dir_rejects_junction_escape(tmp_path: Path) -> None:
    """A ``foremen/<name>`` that is a reparse point escaping the root is refused."""
    root = paths.foremen_root(tmp_path)
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "g-escape"
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
                check=True,
                capture_output=True,
            )
        else:
            link.symlink_to(outside, target_is_directory=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("reparse points unavailable")
    with pytest.raises(ValueError, match="escapes"):
        paths.foreman_dir(tmp_path, "g-escape")
    # And discovery never surfaces it as a foreman.
    assert paths.list_foreman_dirs(tmp_path) == []


def test_memory_file_refuses_escape(tmp_path: Path) -> None:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    for bad in ("../escape", "..", "/abs/path", "sub/dir"):
        with pytest.raises(ValueError):
            paths.memory_file(fdir, bad)


def test_memory_file_accepts_stem_and_md(tmp_path: Path) -> None:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    a = paths.memory_file(fdir, "decomp")
    b = paths.memory_file(fdir, "decomp.md")
    assert a == b
    assert a.name == "decomp.md"


def test_list_foreman_dirs_skips_non_foreman_entries(tmp_path: Path) -> None:
    root = paths.foremen_root(tmp_path)
    (root / "g-b").mkdir(parents=True)
    (root / "_shared").mkdir()
    (root / "stray.txt").write_text("x")
    (root / "Bad Name").mkdir()
    (root / "g-a").mkdir()
    found = paths.list_foreman_dirs(tmp_path)
    assert [p.name for p in found] == ["g-a", "g-b"]


def test_list_foreman_dirs_empty_without_root(tmp_path: Path) -> None:
    assert paths.list_foreman_dirs(tmp_path) == []
