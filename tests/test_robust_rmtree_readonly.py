"""``_robust_rmtree`` removes trees holding read-only files.

Every reclaim pass on 2026-10-06 reported "7 orphan removal(s) failed": the
orphan directories held git clones (read-only object files) and copied
``.venv`` trees, and a plain ``shutil.rmtree`` raises ``PermissionError`` on a
read-only entry on Windows.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from charlie_work import worktree
from charlie_work.worktree import _clear_readonly_and_retry, _robust_rmtree


def _make_readonly_tree(root: Path) -> Path:
    obj = root / "clone" / ".git" / "objects" / "44"
    obj.mkdir(parents=True)
    blob = obj / "4a8fa98e219b9ee8585973bba9425676aba452"
    blob.write_bytes(b"blob")
    os.chmod(blob, stat.S_IREAD)
    return blob


def test_robust_rmtree_removes_tree_with_readonly_file(tmp_path: Path) -> None:
    target = tmp_path / "orphan"
    _make_readonly_tree(target)

    assert _robust_rmtree(target) is True
    assert not target.exists()


def test_clear_readonly_and_retry_reraises_non_permission_errors(tmp_path: Path) -> None:
    calls: list[str] = []
    err = FileNotFoundError("gone")

    with pytest.raises(FileNotFoundError):
        _clear_readonly_and_retry(calls.append, str(tmp_path / "x"), err)
    assert calls == []


def test_clear_readonly_and_retry_never_chmods_a_reparse_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chmods: list[str] = []
    monkeypatch.setattr(worktree, "is_junction", lambda p: True)
    monkeypatch.setattr(worktree.os, "chmod", lambda p, mode: chmods.append(p))
    err = PermissionError(13, "denied")

    with pytest.raises(PermissionError):
        _clear_readonly_and_retry(lambda p: None, str(tmp_path / "link"), err)
    assert chmods == []
