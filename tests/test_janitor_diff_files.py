"""``iter_diff_files`` parser tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): unified-diff
file iteration — single/multi-file hunks, new-file flags, renames, and the
no-newline marker.
"""

from __future__ import annotations

from charlie_work.janitor import iter_diff_files


def test_iter_diff_files_single_file_single_hunk() -> None:
    """A one-file diff yields exactly one (filename, False, hunk_lines) tuple with the expected hunk-header + body lines."""
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 1
    filename, is_new_file, hunk_lines = results[0]
    assert filename == "test.txt"
    assert is_new_file is False
    assert len(hunk_lines) == 4
    assert hunk_lines[0] == "@@ -1,2 +1,2 @@"
    assert hunk_lines[1] == " line 1"
    assert hunk_lines[2] == "-line 2"
    assert hunk_lines[3] == "+line 2 modified"


def test_iter_diff_files_multi_file() -> None:
    """A diff touching 2+ files yields one tuple per file, in source order, each with its own hunk lines (not cross-contaminated)."""
    diff = """diff --git a/a_module.py b/a_module.py
index 1234567..abcdef0 100644
--- a/a_module.py
+++ b/a_module.py
@@ -1,2 +1,2 @@
 def a_func():
-    return 'a'
+    return 'a_modified'
diff --git a/b_module.py b/b_module.py
index 1234567..abcdef0 100644
--- a/b_module.py
+++ b/b_module.py
@@ -1,2 +1,2 @@
 def b_func():
-    return 'b'
+    return 'b_modified'
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 2
    filename_a, is_new_file_a, hunk_lines_a = results[0]
    filename_b, is_new_file_b, hunk_lines_b = results[1]
    assert filename_a == "a_module.py"
    assert is_new_file_a is False
    assert len(hunk_lines_a) == 4
    assert "@@ -1,2 +1,2 @@" in hunk_lines_a[0]
    assert "def a_func():" in hunk_lines_a[1]
    assert filename_b == "b_module.py"
    assert is_new_file_b is False
    assert len(hunk_lines_b) == 4
    assert "@@ -1,2 +1,2 @@" in hunk_lines_b[0]
    assert "def b_func():" in hunk_lines_b[1]
    # Verify no cross-contamination
    assert "a_func" not in hunk_lines_b
    assert "b_func" not in hunk_lines_a


def test_iter_diff_files_new_file_flag() -> None:
    """A diff containing `new file mode` for a path sets is_new_file=True; an existing-file modification sets it False."""
    diff = """diff --git a/new_file.py b/new_file.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/new_file.py
@@ -0,0 +1,2 @@
+def new_func():
+    return 'new'
diff --git a/existing.py b/existing.py
index 1234567..abcdef0 100644
--- a/existing.py
+++ b/existing.py
@@ -1,1 +1,2 @@
 old line
+new line
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 2
    filename_new, is_new_file_new, _ = results[0]
    filename_existing, is_new_file_existing, _ = results[1]
    assert filename_new == "new_file.py"
    assert is_new_file_new is True
    assert filename_existing == "existing.py"
    assert is_new_file_existing is False


def test_iter_diff_files_rename_no_hunk_body() -> None:
    """A pure rename (100% similarity, no @@ hunks) is skipped since it has no +++ b/ line (and thus no hunk body)."""
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py
"""
    results = list(iter_diff_files(diff))
    # The function skips sections without a +++ b/ line
    assert len(results) == 0


def test_iter_diff_files_strips_no_newline_marker() -> None:
    r"""A hunk containing a `\ No newline at end of file` metadata line does not include that line in hunk_lines."""
    diff = """diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
\\ No newline at end of file
+line 2 modified
"""
    results = list(iter_diff_files(diff))
    assert len(results) == 1
    _, _, hunk_lines = results[0]
    # The metadata line should be stripped
    assert not any(line.startswith("\\") for line in hunk_lines)
    assert len(hunk_lines) == 4


def test_iter_diff_files_empty_diff() -> None:
    """Empty diff yields nothing (empty iterator)."""
    results = list(iter_diff_files(""))
    assert results == []
