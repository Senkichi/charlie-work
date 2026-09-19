"""``_calculate_patch_id`` tests for the janitor no-op-rework gate.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): diff-content
hashing stability, metadata / offset immunity, and empty-diff / git-failure
behavior.
"""

from __future__ import annotations

import pytest

from charlie_work.janitor import _calculate_patch_id


def test_calculate_patch_id_stable_for_same_diff() -> None:
    """Patch-id calculation should be stable for the same diff content."""
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    patch_id1 = _calculate_patch_id(diff)
    patch_id2 = _calculate_patch_id(diff)
    assert patch_id1 == patch_id2
    assert len(patch_id1) == 40  # SHA-1 hex string from git patch-id --stable


def test_calculate_patch_id_different_for_different_diffs() -> None:
    """Patch-id calculation should differ for different diff content."""
    diff1 = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    diff2 = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 different
"""
    patch_id1 = _calculate_patch_id(diff1)
    patch_id2 = _calculate_patch_id(diff2)
    assert patch_id1 != patch_id2


def test_calculate_patch_id_ignores_metadata() -> None:
    """Patch-id calculation should ignore diff metadata (hashes, timestamps)."""
    diff1 = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    # Same content but different metadata (different index hashes)
    diff2 = """diff --git a/test.txt b/test.txt
index 9999999..8888888 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    patch_id1 = _calculate_patch_id(diff1)
    patch_id2 = _calculate_patch_id(diff2)
    assert patch_id1 == patch_id2


def test_calculate_patch_id_empty_diff() -> None:
    """Patch-id calculation should return empty string for empty diff."""
    assert _calculate_patch_id("") == ""
    assert _calculate_patch_id("   \n  ") == ""


def test_calculate_patch_id_offset_immune() -> None:
    """Patch-id is identical for diffs with the same content but shifted hunk offsets.

    A base-update merge that adds lines to files shared with an open PR shifts
    hunk-header line numbers (@@ -N,M +N,M @@) without touching content lines.
    git patch-id strips hunk headers for this reason; _calculate_patch_id must
    do the same so that offset-only shifts do not change the patch-id.

    MUTATION CHECK: this test MUST FAIL if the @@ skip is removed from
    _calculate_patch_id (verified during development — see PR #229 rework notes).
    """
    # Two diffs with identical content lines but hunk offsets shifted by 4 lines
    diff_original = """\
diff --git a/src/foo.py b/src/foo.py
index aaaaaaa..bbbbbbb 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -10,5 +10,6 @@
 context line
-old line
+new line
 another context
"""
    diff_shifted = """\
diff --git a/src/foo.py b/src/foo.py
index aaaaaaa..bbbbbbb 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -14,5 +14,6 @@
 context line
-old line
+new line
 another context
"""
    id_original = _calculate_patch_id(diff_original)
    id_shifted = _calculate_patch_id(diff_shifted)
    assert id_original == id_shifted, (
        f"Hunk-offset shift changed patch-id: {id_original!r} != {id_shifted!r}. "
        "Did git patch-id --stable stop ignoring hunk headers?"
    )
    assert len(id_original) == 40  # SHA-1 hex string from git patch-id --stable


def test_calculate_patch_id_returns_empty_for_diff_without_hunks() -> None:
    """A diff with no hunk header is not a real patch and cannot be compared."""
    assert _calculate_patch_id("diff --git a/file b/file\n") == ""


def test_calculate_patch_id_returns_empty_when_git_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git failures during patch-id computation must fail closed (empty string)."""
    from charlie_work import janitor as janitor_module

    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""

    def _fake_run_captured(*_args, **_kwargs):
        from charlie_work.subprocess_runner import RunResult

        return RunResult(returncode=1, stdout="", stderr="git failed", error="git failed")

    monkeypatch.setattr(janitor_module, "run_captured", _fake_run_captured)
    assert _calculate_patch_id(diff) == ""
