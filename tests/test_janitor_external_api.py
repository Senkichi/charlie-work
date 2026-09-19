"""External-API-call diff-scan tests for the janitor.

Split out of ``tests/test_janitor.py`` (issue #1558, Track 1): the
``_check_external_api_fixtures`` warning surface — ``gh api`` / ``requests``
call detection and the fixture / body-evidence / test-file / exempt-file
suppressions.
"""

from __future__ import annotations

from _janitor_fixtures import (
    _config,
    _green_checks,
    _green_pr,
)

from charlie_work.janitor import run_janitor


def test_external_api_call_warns_for_gh_run_api() -> None:
    """A product file adding gh.run(['api', ...]) with no fixture or body evidence warns."""
    diff = """diff --git a/src/runners.py b/src/runners.py
index 123..456 100644
--- a/src/runners.py
+++ b/src/runners.py
@@ -1,3 +1,5 @@
 def observe(gh):
+    data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
+    runners = data.get("runners", [])
     return runners
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)
    assert any("gh.run" in w for w in verdict.warnings)


def test_external_api_call_warns_for_self_run_api_multiline() -> None:
    """Multi-line self.run(['api', ...]) in a product file warns."""
    diff = """diff --git a/src/github.py b/src/github.py
index 123..456 100644
--- a/src/github.py
+++ b/src/github.py
@@ -1,3 +1,10 @@
 class GitHub:
     def actions_job(self, job_id):
+        result = self.run(
+            [
+                "api",
+                f"repos/{{owner}}/{{repo}}/actions/jobs/{job_id}",
+            ],
+            json_output=True,
+        )
+        return result
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_warns_for_requests_get() -> None:
    """A product file adding requests.get() with no fixture or body evidence warns."""
    diff = """diff --git a/src/foo.py b/src/foo.py
index 123..456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def fetch():
+    import requests
+    return requests.get("https://api.github.com/repos/owner/repo")
     pass
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)
    assert any("requests.get" in w for w in verdict.warnings)


def test_external_api_call_warns_for_subprocess_gh_api() -> None:
    """subprocess.run(['gh', 'api', ...]) in a product file warns."""
    diff = """diff --git a/src/foo.py b/src/foo.py
index 123..456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def fetch():
+    import subprocess
+    subprocess.run(["gh", "api", "repos/owner/repo"], check=True)
     pass
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_with_fixture() -> None:
    """A product file adding gh.run(['api', ...]) plus a tests/fixtures/ file is clean."""
    diff = """diff --git a/src/runners.py b/src/runners.py
index 123..456 100644
--- a/src/runners.py
+++ b/src/runners.py
@@ -1,3 +1,5 @@
 def observe(gh):
+    data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
+    runners = data.get("runners", [])
     return runners

diff --git a/tests/fixtures/runners.json b/tests/fixtures/runners.json
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/tests/fixtures/runners.json
@@ -0,0 +1,3 @@
+{
+    "runners": [{"id": 1, "name": "jc-1"}]
+}
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_with_body_evidence() -> None:
    """A product file adding an external call is clean when the PR body has live evidence."""
    diff = """diff --git a/src/runners.py b/src/runners.py
index 123..456 100644
--- a/src/runners.py
+++ b/src/runners.py
@@ -1,3 +1,5 @@
 def observe(gh):
+    data = gh.run(["api", "repos/{owner}/{repo}/actions/runners"], json_output=True)
+    runners = data.get("runners", [])
     return runners
"""
    pr = _green_pr(
        body="Closes #123.\n\nVerified with live payload: gh api repos/{owner}/{repo}/actions/runners"
    )
    verdict = run_janitor(pr, _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_for_test_files() -> None:
    """Test files are excluded from external API call detection."""
    diff = """diff --git a/tests/test_runners.py b/tests/test_runners.py
index 123..456 100644
--- a/tests/test_runners.py
+++ b/tests/test_runners.py
@@ -1,3 +1,5 @@
 def test_observe():
+    gh = MagicMock()
+    gh.run = MagicMock(side_effect=lambda args: _mock(args))
     assert observe(gh)
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_for_exempt_files() -> None:
    """Markdown/docs files are excluded from external API call detection."""
    diff = """diff --git a/docs/api.md b/docs/api.md
index 123..456 100644
--- a/docs/api.md
+++ b/docs/api.md
@@ -1,3 +1,5 @@
 # Notes
+
+    gh api repos/{owner}/{repo}/actions/runners
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)


def test_external_api_call_no_warning_without_any_call() -> None:
    """A product diff without external API calls does not warn."""
    diff = """diff --git a/src/foo.py b/src/foo.py
index 123..456 100644
--- a/src/foo.py
+++ b/src/foo.py
@@ -1,3 +1,5 @@
 def compute():
+    return 1 + 1
     pass
"""
    verdict = run_janitor(_green_pr(), _green_checks(), _config(), pr_diff=diff)

    assert verdict.ok is True
    assert not any("external API/library call" in w for w in verdict.warnings)
