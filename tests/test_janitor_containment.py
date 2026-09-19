"""``check_operator_containment`` tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): working-tree
containment leak detection against a real temp git repo — clean trees,
unrelated dirty files, untracked files, partial and multi-file leaks, and
git-failure degradation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _janitor_fixtures import _init_repo

from charlie_work.janitor import check_operator_containment


def test_containment_clean_tree_no_warnings(tmp_path: Path) -> None:
    """Clean operator checkout produces no containment warnings."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Empty diff (no PR changes)
    diff = ""
    warnings = check_operator_containment(tmp_path, diff, 123)

    assert warnings == ()


def test_containment_leak_detection(tmp_path: Path) -> None:
    """Detect leaked worker edits: working-tree file byte-identical to PR post-image."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify the file to match PR post-image
    leaked_content = "leaked content from PR"
    (tmp_path / "test.txt").write_text(leaked_content)

    # Get the actual working-tree diff to use as the PR diff
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", "test.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    diff = result.stdout

    warnings = check_operator_containment(tmp_path, diff, 123)

    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "test.txt" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_containment_unrelated_dirty_file(tmp_path: Path) -> None:
    """Unrelated dirty files produce generic warning, not leak warning."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Modify file to content NOT in the PR diff
    (tmp_path / "test.txt").write_text("unrelated local work")

    # Create a diff that doesn't match the dirty file
    diff = """diff --git a/other.txt b/other.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/other.txt
@@ -0,0 +1 @@
+other file content
"""

    warnings = check_operator_containment(tmp_path, diff, 123)

    assert len(warnings) == 1
    assert "not a leak" in warnings[0]
    assert "test.txt" in warnings[0]
    assert "Containment leak" not in warnings[0]


def test_containment_git_failure_graceful_degradation(tmp_path: Path) -> None:
    """Git failures produce no warnings rather than blocking."""
    # Not a git repo, so git status will fail
    diff = ""
    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should return empty tuple on git failure (graceful degradation)
    assert warnings == ()


def test_containment_partial_file_leak_detection(tmp_path: Path) -> None:
    """Detect leaked worker edits in multi-line file with mid-file edit (partial-file leak scenario)."""
    _init_repo(tmp_path)
    # Create a multi-line file
    initial_content = """line 1
line 2
line 3
line 4
line 5
"""
    (tmp_path / "multi.txt").write_text(initial_content)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify a middle line to match PR post-image
    leaked_content = """line 1
line 2
MODIFIED LINE
line 4
line 5
"""
    (tmp_path / "multi.txt").write_text(leaked_content)

    # Get the actual working-tree diff to use as the PR diff
    result = subprocess.run(
        ["git", "diff", "HEAD", "--", "multi.txt"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    diff = result.stdout

    warnings = check_operator_containment(tmp_path, diff, 123)

    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "multi.txt" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_containment_untracked_file_no_warning(tmp_path: Path) -> None:
    """Untracked files unrelated to the PR diff produce zero warnings."""
    _init_repo(tmp_path)
    # Create an initial commit
    (tmp_path / "test.txt").write_text("initial content")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Create an untracked file (simulating orchestrator.config.yaml)
    (tmp_path / "orchestrator.config.yaml").write_text("config: value")

    # Create a diff that doesn't include the untracked file
    diff = """diff --git a/other.txt b/other.txt
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/other.txt
@@ -0,0 +1 @@
+other file content
"""

    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should produce zero warnings - untracked files not in the diff are ignored
    assert warnings == ()


def test_containment_multi_file_leak_first_file(tmp_path: Path) -> None:
    """Detect leaked worker edits when PR diff spans multiple files and first file is leaked."""
    _init_repo(tmp_path)
    # Create two files with initial content
    (tmp_path / "a_module.py").write_text("def a_func():\n    return 'a'\n")
    (tmp_path / "b_module.py").write_text("def b_func():\n    return 'b'\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify the FIRST file (a_module.py) to match PR post-image
    leaked_content = "def a_func():\n    return 'a_modified'\n"
    (tmp_path / "a_module.py").write_text(leaked_content)

    # Create a multi-file PR diff (both a_module.py and b_module.py)
    # The PR modifies both files, but only a_module.py is leaked in the working tree
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

    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should detect the leak in a_module.py (first file in diff)
    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "a_module.py" in warnings[0]
    assert "git checkout --" in warnings[0]


def test_containment_multi_file_leak_last_file(tmp_path: Path) -> None:
    """Detect leaked worker edits when PR diff spans multiple files and last file is leaked."""
    _init_repo(tmp_path)
    # Create two files with initial content
    (tmp_path / "a_module.py").write_text("def a_func():\n    return 'a'\n")
    (tmp_path / "b_module.py").write_text("def b_func():\n    return 'b'\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    # Simulate a leak: modify the LAST file (b_module.py) to match PR post-image
    leaked_content = "def b_func():\n    return 'b_modified'\n"
    (tmp_path / "b_module.py").write_text(leaked_content)

    # Create a multi-file PR diff (both a_module.py and b_module.py)
    # The PR modifies both files, but only b_module.py is leaked in the working tree
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

    warnings = check_operator_containment(tmp_path, diff, 123)

    # Should detect the leak in b_module.py (last file in diff)
    assert len(warnings) == 1
    assert "Containment leak detected" in warnings[0]
    assert "PR #123" in warnings[0]
    assert "b_module.py" in warnings[0]
    assert "git checkout --" in warnings[0]
