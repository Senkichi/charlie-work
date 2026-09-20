"""``check_stub_tests`` tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): stub-test
detection (pass/ellipsis bodies, constant assertions, seam-name mismatch,
gutted existing bodies, deletion-only gutting) and the ``run_janitor``
wiring that appends stub warnings from the PR diff.
"""

from __future__ import annotations

from pathlib import Path

from _janitor_fixtures import (
    _config,
    _green_checks,
    _green_pr,
    _test_adequacy_config,
)

from charlie_work.janitor import (
    check_stub_tests,
    run_janitor,
)


def test_check_stub_tests_pass_body_marker() -> None:
    """Only-pass/.../docstring bodies are flagged as stub tests."""
    diff = '''diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,6 @@
+def test_pass_stub():
+    pass  # placeholder
+def test_ellipsis_stub():
+    ...
+def test_docstring_stub():
+    """docstring"""
'''
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert len(warnings) == 3
    assert any("test_pass_stub" in w for w in warnings)
    assert any("test_ellipsis_stub" in w for w in warnings)
    assert any("test_docstring_stub" in w for w in warnings)


def test_check_stub_tests_assert_constant_ignores_product_references() -> None:
    """Assertions referencing product modules are fine; unrelated module constants are flagged."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,2 @@
 def feature():
-    return 1
+    return 2
diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,9 @@
+from feature import do_thing
+from other import UNRELATED
+def test_do_thing_works():
+    assert do_thing() is not None
+def test_constant_stub():
+    assert UNRELATED > 0
+def test_local_only():
+    result = do_thing()
+    assert result == 1
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_constant_stub" in w and "assert-constant" in w for w in warnings)
    assert not any("test_do_thing_works" in w for w in warnings)
    assert not any("test_local_only" in w for w in warnings)


def test_check_stub_tests_seam_name_mismatch() -> None:
    """Test names containing a seam keyword require the body to call/mention that seam."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,5 @@
+def test_call_model_smoke():
+    assert True
+def test_call_model_real():
+    call_model()
+def test_route_smoke():
+    assert True
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_call_model_smoke" in w and "seam-name" in w for w in warnings)
    assert any("test_route_smoke" in w and "seam-name" in w for w in warnings)
    assert not any("test_call_model_real" in w and "seam-name" in w for w in warnings)


def test_check_stub_tests_async_body_is_flagged() -> None:
    """async def test_... functions are inspected the same as sync functions."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,2 @@
+async def test_async_pass_stub():
+    pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_async_pass_stub" in w for w in warnings)


def test_check_stub_tests_gutted_existing_body_is_flagged() -> None:
    """A pre-existing test whose body is gutted to pass is flagged even if the def line is context."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,2 @@
 def test_existing():
-    result = feature()
-    assert result
+    pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_existing" in w for w in warnings)


def test_check_stub_tests_unmodified_test_not_flagged_by_added_blank_line() -> None:
    """An unmodified test is not flagged when only a blank line elsewhere is added."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,5 @@
 def test_one():
     pass
+
 def test_two():
     pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_one" in w or "test_two" in w for w in warnings)


def test_check_stub_tests_added_decorator_does_not_flag_unrelated_test() -> None:
    """A decorator added to another function must not flag an unmodified test."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,5 @@
+@pytest.mark.slow
 def test_one():
     pass
 def test_two():
     pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_one" in w or "test_two" in w for w in warnings)


def test_check_stub_tests_async_helper_reference_not_assert_constant() -> None:
    """A test asserting on a top-level async helper's name is not assert-constant.

    _collect_test_defined_names must record AsyncFunctionDef names; otherwise
    the helper name reads as an unknown external and the assertion is
    misclassified as constant.
    """
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,7 @@
+async def _drain_queue():
+    return 1
+
+
+def test_helper_is_exported():
+    handler = _drain_queue
+    assert handler is _drain_queue
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_helper_is_exported" in w for w in warnings)


def test_check_stub_tests_gutted_existing_async_body_is_flagged() -> None:
    """An existing async test gutted to pass is flagged even if the def line is context."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,3 +1,2 @@
 async def test_existing_async():
-    result = await feature()
-    assert result
+    pass
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_existing_async" in w for w in warnings)


def test_check_stub_tests_deletion_only_gutted_body_is_flagged() -> None:
    """Gutting by pure deletion (no added lines in the function) is still caught."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,2 @@
 def test_existing():
     '''checks the feature'''
-    result = feature()
-    assert result
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert any("test_existing" in w for w in warnings)


def test_check_stub_tests_deletion_near_healthy_test_not_flagged() -> None:
    """A deletion adjacent to a substantive test does not flag that test."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,4 +1,3 @@
 def test_existing():
     result = feature()
     assert result
-# stale comment
"""
    warnings = check_stub_tests(diff, _test_adequacy_config())

    assert not any("test_existing" in w for w in warnings)


def test_run_janitor_appends_stub_warnings_from_pr_diff() -> None:
    """run_janitor calls check_stub_tests and adds its warnings to the verdict."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,2 @@
+def test_pass_stub():
+    pass  # placeholder
"""
    verdict = run_janitor(
        _green_pr(), _green_checks(), _config(), repo_root=Path.cwd(), pr_diff=diff
    )

    assert verdict.ok is True
    assert any("test_pass_stub" in w for w in verdict.warnings)
