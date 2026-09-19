"""Review CI status: required/non-required check sections, annotation fallback.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the CI-status-section and CI-failure half of the review-prompt seam -- required/non-required check reporting and annotation fallback. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub, FakeGitHubWithChecks, FakeGitHubWithChecksAndAnnotations
from _review_fixtures import _required_checks_config
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_review_ci_status_section_reports_checks_unavailable(tmp_path: Path) -> None:
    """No required checks configured + gh pr checks failure: the CI status section
    must warn that CI could not be fetched, never claim green (issue: reviewer
    token efficiency)."""
    config = OrchestratorConfig()  # default: required_checks == ()

    class FakeGitHubWithChecksUnavailable(FakeGitHub):
        def pr_checks(self, number: int):
            return None

    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecksUnavailable()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "## CI status" in packet
    assert "could not be fetched" in packet
    assert "checks.json" in packet
    # Never claim CI is green when it was unfetchable.
    assert "verified deterministically" not in packet


def test_review_ci_status_section_reports_no_required_checks_configured(
    tmp_path: Path,
) -> None:
    """When required_checks is empty, the janitor never verifies CI at all — the
    section must say so rather than implying a deterministic pass."""
    config = OrchestratorConfig()  # default: required_checks == ()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # default pr_checks() returns 3 passing checks
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "No required checks are configured" in packet
    assert "verified deterministically" not in packet


def test_review_ci_status_section_reports_all_required_passing(tmp_path: Path) -> None:
    """All required checks green: the reviewer should be told not to re-inspect,
    without needing to open checks.json."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # default pr_checks() returns the 3 required checks, all passing
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "verified deterministically by the orchestrator before dispatch" in packet
    assert "Tests passed" in packet
    assert "do not spend turns re-inspecting" in packet.lower()


def test_review_ci_status_section_lists_failing_non_required_checks(tmp_path: Path) -> None:
    """A failing check that is NOT in the required set must not block review
    (janitor only gates required checks), but should be surfaced by name so the
    reviewer can weigh it without reading checks.json."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
            {"name": "Codecov", "state": "FAILURE"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "Non-required/informational check(s) currently failing" in packet
    assert "Codecov" in packet


def test_review_ci_status_section_skipped_non_required_check_not_listed_as_failing(
    tmp_path: Path,
) -> None:
    """A SKIPPED non-required check (path-filtered/matrix-conditional job that
    legitimately did not run) must never be reported as 'currently failing'."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
            {"name": "Optional Job", "state": "SKIPPED"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "Non-required/informational check(s) currently failing" not in packet
    assert "Non-required/informational check(s) cancelled" not in packet
    assert "Optional Job" not in packet


def test_review_ci_status_section_neutral_non_required_check_not_listed_as_failing(
    tmp_path: Path,
) -> None:
    """A NEUTRAL non-required check conclusion is neither pass nor fail and
    must never be reported as 'currently failing'."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
            {"name": "Advisory Job", "state": "NEUTRAL"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "Non-required/informational check(s) currently failing" not in packet
    assert "Non-required/informational check(s) cancelled" not in packet
    assert "Advisory Job" not in packet


def test_review_ci_status_section_cancelled_non_required_check_worded_distinctly(
    tmp_path: Path,
) -> None:
    """A CANCELLED non-required check (often an infra hiccup, not a code
    failure) must be surfaced with distinct wording, never called 'failing'."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecks(
        checks=[
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
            {"name": "Flaky Job", "state": "CANCELLED"},
        ]
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = (paths.prs / "pr-456" / "review-prompt.md").read_text(encoding="utf-8")
    assert "Non-required/informational check(s) currently failing" not in packet
    assert "Non-required/informational check(s) cancelled" in packet
    assert "Flaky Job" in packet


def test_review_ci_failure_with_annotations_populates_required_changes(tmp_path: Path) -> None:
    """Issue #771: a required-check failure whose check run carries GitHub
    annotations yields required_changes naming the real file/line, and the
    rendered rework brief contains them -- not just the bare check-name
    summary the route previously emitted."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecksAndAnnotations(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9001},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        annotations_by_check_run_id={
            9001: [
                {
                    "path": "src/charlie_work/workflow.py",
                    "start_line": 7296,
                    "message": "F821 undefined name 'checks'",
                    "annotation_level": "failure",
                },
            ],
        },
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["required_changes"] == [
        "Tests passed: src/charlie_work/workflow.py:7296 — F821 undefined name 'checks'",
    ]

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    assert "## Required changes" in prompt_text
    assert "src/charlie_work/workflow.py:7296" in prompt_text
    assert "F821 undefined name 'checks'" in prompt_text
    # The failing check's own name reaches the brief via $dispatch_note (the
    # rework_summary passed to _write_rework_prompt, kept as a separate
    # template slot from $required_changes_section) regardless of which
    # required_changes tier rendered -- a worker sees which check was red
    # even when tier 1 (the enumerated list) is what fired.
    assert "Tests passed" in prompt_text


def test_review_ci_failure_without_annotations_degrades_without_fabricating(
    tmp_path: Path,
) -> None:
    """Issue #771: a failing required check whose run carries zero GitHub
    annotations AND no link (e.g. a process-level crash with nothing GitHub
    can point at) must not fabricate a file/line -- no bogus per-file bullet
    is synthesized. Issue #792: record_review now derives a single
    required_changes entry from the non-vacuous summary text (marking
    findings_channel="derived"), and _render_required_changes_section renders
    that marker with the same summary-only fallback shape (tier 2) as before
    -- the rendered prompt is unchanged even though required_changes is no
    longer empty."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecksAndAnnotations(
        checks=[
            {"name": "Tests passed", "state": "FAILURE", "databaseId": 9002},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        annotations_by_check_run_id={},  # 9002 resolves to zero annotations
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["required_changes"] == ["CI failed on Tests passed; push a fix"]
    assert decision["findings_channel"] == "derived"
    assert decision["summary"] == "CI failed on Tests passed; push a fix"

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    # Tier-2 fallback renders the real summary verbatim -- not a fabricated
    # file/line, and still strictly more informative than an empty section.
    assert "CI failed on Tests passed; push a fix" in prompt_text
    assert "## Required changes" in prompt_text


def test_review_ci_failure_without_annotations_but_with_link_uses_link_fallback(
    tmp_path: Path,
) -> None:
    """Issue #771: a failing required check with zero GitHub annotations but a
    real ``link`` (the common case: GitHub Actions always assigns a run URL,
    but only some failure modes emit per-line annotations) falls back to
    pointing the worker at the failing run instead of degrading all the way
    to the bare check-name summary."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHubWithChecksAndAnnotations(
        checks=[
            {
                "name": "Tests passed",
                "state": "FAILURE",
                "databaseId": 9003,
                "link": "https://github.com/o/r/actions/runs/1/jobs/9003",
            },
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ],
        annotations_by_check_run_id={},  # 9003 resolves to zero annotations
    )
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["required_changes"] == [
        "Tests passed: no per-line annotations available from GitHub; "
        "inspect the failing run at https://github.com/o/r/actions/runs/1/jobs/9003",
    ]

    rework_prompt = paths.prs / "pr-456" / "rework-prompt.md"
    prompt_text = rework_prompt.read_text(encoding="utf-8")
    assert "## Required changes" in prompt_text
    assert "https://github.com/o/r/actions/runs/1/jobs/9003" in prompt_text
