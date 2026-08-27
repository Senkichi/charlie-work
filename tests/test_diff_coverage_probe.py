"""Tests for diff_coverage_probe.py (issues #1260/#1261).

W3 (branch-token-vs-test-add heuristic) and W20 item 1 (unwired-symbol AST
probe). Both halves are pure/read-only and are exercised directly here;
review()-level wiring/gating tests live in test_charlie_work.py alongside
the sibling test_adequacy integration tests.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import CoverageProbeConfig
from charlie_work.diff_coverage_probe import (
    BranchCoverageFinding,
    UnwiredSymbolFinding,
    check_branch_coverage,
    check_unwired_symbols,
    run_static_probe,
)

# ---------------------------------------------------------------------------
# W3: check_branch_coverage
# ---------------------------------------------------------------------------


def test_branch_adds_with_test_adds_is_silent() -> None:
    """Branch-adds present + test-adds present, ratio under threshold -> silent."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def feature():
     pass
+def new_feature(x):
+    if x:
+        return 1
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,4 @@
 def test_feature():
-    pass
+    assert new_feature(True) == 1
+    assert new_feature(False) is None
"""
    findings = check_branch_coverage(diff, CoverageProbeConfig())
    assert findings == ()


def test_branch_adds_with_zero_test_adds_is_flagged() -> None:
    """Branch-adds present, zero test-adds anywhere in the diff -> flagged."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def feature():
     pass
+def new_feature(x):
+    if x:
+        return 1
"""
    findings = check_branch_coverage(diff, CoverageProbeConfig())
    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, BranchCoverageFinding)
    assert finding.filename == "src/feature.py"
    assert finding.branch_adds == 1
    assert finding.test_adds == 0
    assert finding.reason == "no_test_adds"


def test_test_only_diff_is_silent() -> None:
    """No non-test file touched at all -> silent regardless of test content."""
    diff = """diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,5 @@
 def test_feature():
-    pass
+    assert True
+    if True:
+        assert True
"""
    findings = check_branch_coverage(diff, CoverageProbeConfig())
    assert findings == ()


def test_rename_only_diff_is_silent() -> None:
    """A pure rename (no hunk body) must not be misread as branch-adds."""
    diff = """diff --git a/src/old_name.py b/src/new_name.py
similarity index 100%
rename from src/old_name.py
rename to src/new_name.py
"""
    findings = check_branch_coverage(diff, CoverageProbeConfig())
    assert findings == ()


def test_ratio_exceeded_flags_even_with_some_test_adds() -> None:
    """branch:test-add ratio over threshold flags even when test_adds > 0."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,10 @@
 def feature():
     pass
+def new_feature(a, b, c):
+    if a:
+        pass
+    elif b:
+        pass
+    elif c:
+        pass
+    if a and b:
+        pass
+    if b or c:
+        pass
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,3 @@
 def test_feature():
-    pass
+    assert True
"""
    config = CoverageProbeConfig()
    findings = check_branch_coverage(diff, config)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.reason == "ratio_exceeded"
    assert finding.branch_adds / finding.test_adds > config.branch_to_assert_ratio_threshold


# ---------------------------------------------------------------------------
# W20 item 1: check_unwired_symbols
# ---------------------------------------------------------------------------


def test_new_symbol_with_src_caller_is_silent(tmp_path: Path) -> None:
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def existing():
     pass
+def helper():
+    return 42
diff --git a/src/caller.py b/src/caller.py
index 123..456 100644
--- a/src/caller.py
+++ b/src/caller.py
@@ -1,2 +1,4 @@
 def existing_caller():
     pass
+def uses_helper():
+    return helper()
"""
    findings, warnings = check_unwired_symbols(diff, tmp_path, CoverageProbeConfig())
    assert findings == ()
    assert warnings == ()


def test_symbol_referenced_only_from_tests_is_flagged(tmp_path: Path) -> None:
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def existing():
     pass
+def helper():
+    return 42
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,4 @@
 def test_existing():
     pass
+def test_helper():
+    assert helper() == 42
"""
    findings, warnings = check_unwired_symbols(diff, tmp_path, CoverageProbeConfig())
    assert warnings == ()
    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, UnwiredSymbolFinding)
    assert finding.symbol == "helper"
    assert finding.filename == "src/feature.py"
    assert finding.kind == "function"


def test_registration_only_reference_is_silent(tmp_path: Path) -> None:
    """A non-call reference (e.g. a registry/dict entry) still counts as wired."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def existing():
     pass
+def helper():
+    return 42
diff --git a/src/registry.py b/src/registry.py
index 123..456 100644
--- a/src/registry.py
+++ b/src/registry.py
@@ -1,3 +1,4 @@
 REGISTRY = {
     "existing": existing,
+    "helper": helper,
 }
"""
    findings, warnings = check_unwired_symbols(diff, tmp_path, CoverageProbeConfig())
    assert findings == ()
    assert warnings == ()


def test_same_diff_definition_and_call_is_silent(tmp_path: Path) -> None:
    """Symbol defined and called in the same file's added lines -> no tree lookup needed."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,7 @@
 def existing():
     pass
+def helper():
+    return 42
+
+def caller():
+    return helper()
"""
    findings, warnings = check_unwired_symbols(diff, tmp_path, CoverageProbeConfig())
    assert findings == ()
    assert warnings == ()


def test_check_unwired_symbols_ignores_private_names(tmp_path: Path) -> None:
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def existing():
     pass
+def _helper():
+    return 42
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,4 @@
 def test_existing():
     pass
+def test_helper():
+    assert _helper() == 42
"""
    findings, warnings = check_unwired_symbols(diff, tmp_path, CoverageProbeConfig())
    assert findings == ()
    assert warnings == ()


def test_check_unwired_symbols_disabled_via_config(tmp_path: Path) -> None:
    """check_unwired_symbols is a pure function; the enable/disable knob is
    ``run_static_probe``'s responsibility (config.check_unwired_symbols),
    exercised at that boundary instead of here."""
    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def existing():
     pass
+def helper():
+    return 42
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,4 @@
 def test_existing():
     pass
+def test_helper():
+    assert helper() == 42
"""
    config = CoverageProbeConfig(check_unwired_symbols=False)
    verdict = run_static_probe(diff, tmp_path, config)
    assert verdict.unwired_findings == ()


# ---------------------------------------------------------------------------
# run_static_probe: the never-raises boundary
# ---------------------------------------------------------------------------


def test_run_static_probe_never_raises_and_reports_degradation(
    monkeypatch, tmp_path: Path
) -> None:
    """An internal exception in either half must become a rendered warning,
    never propagate, and never silently render nothing (design item 7)."""
    import charlie_work.diff_coverage_probe as probe_mod

    def _boom(diff, config):
        raise ValueError("boom")

    monkeypatch.setattr(probe_mod, "check_branch_coverage", _boom)

    verdict = run_static_probe("diff --git a/x b/x", tmp_path, CoverageProbeConfig())

    assert verdict.branch_findings == ()
    assert any("static probe degraded" in w for w in verdict.warnings)
    assert any("branch-coverage heuristic failed" in w for w in verdict.warnings)


def test_run_static_probe_reports_unwired_symbol_degradation(monkeypatch, tmp_path: Path) -> None:
    import charlie_work.diff_coverage_probe as probe_mod

    def _boom(diff, repo_root, config):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(probe_mod, "check_unwired_symbols", _boom)

    verdict = run_static_probe("diff --git a/x b/x", tmp_path, CoverageProbeConfig())

    assert verdict.unwired_findings == ()
    assert any("unwired-symbol probe failed" in w for w in verdict.warnings)


def test_run_static_probe_empty_diff_is_clean() -> None:
    verdict = run_static_probe("", Path("."), CoverageProbeConfig())
    assert verdict.branch_findings == ()
    assert verdict.unwired_findings == ()
    assert verdict.warnings == ()


def test_check_unwired_symbols_reports_unparseable_file_as_warning(tmp_path: Path) -> None:
    """A file under repo_root/src that fails to parse must surface as a
    visible warning from the consumer search, not silently narrow it."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "broken.py").write_text("def not valid python(:\n", encoding="utf-8")

    diff = """diff --git a/src/feature.py b/src/feature.py
index 123..456 100644
--- a/src/feature.py
+++ b/src/feature.py
@@ -1,2 +1,5 @@
 def existing():
     pass
+def helper():
+    return 42
diff --git a/tests/test_feature.py b/tests/test_feature.py
index 123..456 100644
--- a/tests/test_feature.py
+++ b/tests/test_feature.py
@@ -1,2 +1,4 @@
 def test_existing():
     pass
+def test_helper():
+    assert helper() == 42
"""
    findings, warnings = check_unwired_symbols(diff, tmp_path, CoverageProbeConfig())
    assert len(findings) == 1
    assert any("broken.py" in w for w in warnings)
