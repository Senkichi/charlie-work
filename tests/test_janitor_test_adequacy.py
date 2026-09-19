"""``check_test_adequacy`` tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the
test-adequacy gate — feature-with-tests pass, pure-skip / zero-assertion
handling, docs / examples / workflow / rename exemptions, binary and
malformed diffs, exemption markers, and product-line thresholds.
"""

from __future__ import annotations

from _janitor_fixtures import (
    _test_adequacy_config,
    _test_pr,
)

from charlie_work.janitor import check_test_adequacy


def test_check_test_adequacy_feature_with_tests_passes() -> None:
    """Feature diff + test file with real recognized assertions → ok=True, no warnings."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
 def feature():
     pass
+def new_feature():
+    pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+    assert new_feature() is not None
+    assert new_feature() == "expected"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 2
    assert verdict.facts.added_test_loc == 2
    assert verdict.facts.assertion_count == 2
    assert verdict.facts.test_files_changed == 1


def test_check_test_adequacy_pure_skip_hard_fails() -> None:
    """Feature diff, zero test files changed (pure skip), added_product_loc >= min_product_lines → ok=False."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is False
    assert len(verdict.failures) == 1
    assert "Product code changed" in verdict.failures[0]
    assert "no test files changed" in verdict.failures[0]
    assert "src/feature.py" in verdict.failures[0]
    assert verdict.facts.added_product_loc == 8
    assert verdict.facts.test_files_changed == 0


def test_check_test_adequacy_zero_assertions_warns_by_default() -> None:
    """Feature diff + test file present but zero recognized assertion markers → default config: ok=True with warning."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+def test_new_feature():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert len(verdict.warnings) == 1
    assert "zero recognized assertions" in verdict.warnings[0]
    assert verdict.facts.assertion_count == 0


def test_check_test_adequacy_zero_assertions_hard_fails_when_required() -> None:
    """Feature diff + test file present but zero recognized assertion markers → require_assertions=True: ok=False."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+def test_new_feature():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=5, require_assertions=True)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is False
    assert len(verdict.failures) == 1
    assert "zero recognized assertions" in verdict.failures[0]


def test_check_test_adequacy_docs_only_passes() -> None:
    """Docs-only / config-only diff (all files match exempt_path_globs) → ok=True, facts.added_product_loc == 0."""
    diff = """diff --git a/README.md b/README.md
index 123..456 100644
--- a/README.md
+++ b/README.md
@@ -1,3 +1,5 @@
 # README
-Old text
+New text
diff --git a/pyproject.toml b/pyproject.toml
index 123..456 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -1,3 +1,5 @@
 [tool.pytest]
- old_setting = "value"
+ new_setting = "value"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 0


def test_check_test_adequacy_examples_only_passes() -> None:
    """Examples-only diff (files under examples/** match exempt_path_globs) → ok=True, facts.added_product_loc == 0.

    The examples/ directory holds portable templates and config samples (XML,
    YAML, cron), not executable product code — same category as docs/**. This
    guards against the false positive that flagged
    examples/schedule/charlie-fleet-task.xml as untested product code (PR #690).
    """
    diff = """diff --git a/examples/schedule/charlie-fleet-task.xml b/examples/schedule/charlie-fleet-task.xml
index 123..456 100644
--- a/examples/schedule/charlie-fleet-task.xml
+++ b/examples/schedule/charlie-fleet-task.xml
@@ -1,3 +1,5 @@
 <?xml version="1.0" encoding="UTF-8"?>
 <Task>
+  <Triggers>
+    <TimeTrigger/>
+  </Triggers>
 </Task>
diff --git a/examples/orchestrator.config.devin.yaml b/examples/orchestrator.config.devin.yaml
index 123..456 100644
--- a/examples/orchestrator.config.devin.yaml
+++ b/examples/orchestrator.config.devin.yaml
@@ -1,3 +1,5 @@
 fleet:
-  old: value
+  new: value
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 0
    assert verdict.facts.untested_product_files == ()


def test_check_test_adequacy_workflow_files_exempt() -> None:
    """GitHub Actions workflow YAML files (.github/workflows/**) are CI
    infrastructure, not product code — exempt from the test-adequacy gate.

    Guards against the false positive that flagged .github/workflows/ci.yml
    as untested product code (PR #1127 / issue #1115). Workflow files are
    validated by CI itself: a broken workflow cannot start, so a workflow-only
    PR is self-validating in the same way a docs-only PR is.
    """
    diff = """diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
index 123..456 100644
--- a/.github/workflows/ci.yml
+++ b/.github/workflows/ci.yml
@@ -1,5 +1,8 @@
 jobs:
   test:
     steps:
-      - uses: actions/checkout@v5
+      - uses: actions/checkout@v5
+      - name: Add uv to PATH
+        run: |
+          echo "$RUNNER_TOOL_CACHE/uv" >> "$GITHUB_PATH"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 0
    assert verdict.facts.untested_product_files == ()


def test_check_test_adequacy_rename_only_passes() -> None:
    """Rename-only diff (100% similarity, no hunk body) → ok=True, facts.added_product_loc == 0."""
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0


def test_check_test_adequacy_rename_with_modify_counts_added_lines() -> None:
    """Rename+modify diff (small number of added lines on the renamed file) → counts added lines."""
    diff = """diff --git a/old_name.py b/new_name.py
similarity index 95%
rename from old_name.py
rename to new_name.py
index 123..456 100644
--- a/old_name.py
+++ b/new_name.py
@@ -1,3 +1,5 @@
 def old_func():
     pass
+def new_func():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.facts.added_product_loc == 2
    assert verdict.facts.test_files_changed == 0


def test_check_test_adequacy_binary_diff_warns() -> None:
    """Binary-file diff → ok=True with a warning, never raises."""
    diff = """Binary files a/image.png and b/image.png differ
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert len(verdict.warnings) == 1
    assert "diff unparseable" in verdict.warnings[0]


def test_check_test_adequacy_malformed_diff_warns() -> None:
    """Malformed/garbage diff string → ok=True, never raises."""
    diff = """this is not a valid diff
at all
just random text
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert len(verdict.warnings) == 1
    assert "diff unparseable" in verdict.warnings[0]


def test_check_test_adequacy_valid_exemption_passes() -> None:
    """Valid Test-exempt: <reason> in PR body → ok=True, facts.exempt is True, facts.exempt_reason == "<reason>"."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
 def feature():
     pass
+def new_feature():
+    pass
"""
    pr = _test_pr(body="Closes #123.\n\nTest-exempt: this is a documentation-only change")
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.exempt is True
    assert verdict.facts.exempt_reason == "this is a documentation-only change"


def test_check_test_adequacy_exemption_without_reason_not_exempt() -> None:
    """Test-exempt: with no trailing reason text → NOT exempt (regex requires non-empty reason)."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,10 @@
 def feature():
     pass
+def new_feature():
+    pass
+def another():
+    pass
+def third():
+    pass
+def fourth():
+    pass
"""
    pr = _test_pr(body="Closes #123.\n\nTest-exempt:")
    config = _test_adequacy_config(min_product_lines=5)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is False
    assert verdict.facts.exempt is False
    assert verdict.facts.exempt_reason == ""


def test_check_test_adequacy_custom_exempt_marker_honored() -> None:
    """Custom exempt_marker config override is honored; the default Test-exempt: does NOT match when overridden."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,5 @@
 def feature():
     pass
+def new_feature():
+    pass
"""
    pr = _test_pr(body="Closes #123.\n\nNo-Test-Needed: this is a config-only change")
    config = _test_adequacy_config(min_product_lines=5, exempt_marker="No-Test-Needed:")

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.facts.exempt is True
    assert verdict.facts.exempt_reason == "this is a config-only change"


def test_check_test_adequacy_below_min_product_lines_passes() -> None:
    """Product diff below min_product_lines → ok=True regardless of test presence."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,3 +1,4 @@
 def feature():
     pass
+def new_feature():
+    pass
"""
    pr = _test_pr()
    config = _test_adequacy_config(min_product_lines=10)

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()


def test_check_test_adequacy_test_only_diff_passes() -> None:
    """Test-only diff (no product files changed) → ok=True."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    pass
+    assert new_feature() is not None
+    assert new_feature() == "expected"
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 2
    assert verdict.facts.test_files_changed == 1


def test_check_test_adequacy_bugfix_test_only_passes() -> None:
    """Bugfix diff that only modifies existing test files (with assertions present) and touches no product files → ok=True."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,5 @@
 def test_feature():
-    assert old_behavior() == "old"
+    assert new_behavior() == "new"
+    assert edge_case() is not None
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.assertion_count == 2


def test_check_test_adequacy_conftest_limitation_accepted() -> None:
    """conftest.py carrying non-trivial added logic, classified as test via default glob → those added lines do NOT count toward added_product_loc (locks in the documented accepted evasion)."""
    diff = """diff --git a/conftest.py b/conftest.py
index 123..456 100644
--- a/conftest.py
+++ b/conftest.py
@@ -1,3 +1,10 @@
 import pytest
-
+def new_fixture():
+    return "value"
+
+@pytest.fixture
+def custom_config():
+    return {"key": "value"}
"""
    pr = _test_pr()
    config = _test_adequacy_config()

    verdict = check_test_adequacy(diff, pr, config)

    assert verdict.ok is True
    assert verdict.failures == ()
    assert verdict.warnings == ()
    # conftest.py is classified as test (matches default test_path_globs)
    # so its added lines count toward added_test_loc, not added_product_loc
    # Note: blank lines (like the one with just "-") are not counted as added lines
    assert verdict.facts.added_product_loc == 0
    assert verdict.facts.added_test_loc == 5
    assert verdict.facts.test_files_changed == 1
