"""Tests for OrchestratorApp.review() cross-family injection/opt-out/cache-reuse wiring, direct run_cross_family_review spawn/report-writing, and env-sanitization-at-spawn, carved out of test_charlie_work.py (#1284)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from _helpers import VALID_CROSS_FAMILY_REPORT, _cross_family_app
from charlie_work.rescue_review import (
    _CAVEAT,
    CrossFamilyResult,
    CrossFamilyVerdict,
    run_cross_family_review,
)


def _fake_completed(
    returncode: int = 0, stdout: str = "**MAJOR**\nx\n\nVerdict: safe", stderr: str = ""
):
    def _runner(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return _runner


def test_run_cross_family_writes_report_with_caveat(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="attack this",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_fake_completed(0, "**BLOCKER**\nboom\n\nVerdict: safe"),
    )

    assert result.ok is True
    assert result.returncode == 0
    assert prompt.read_text(encoding="utf-8") == "attack this"
    body = report.read_text(encoding="utf-8")
    assert "leads, not verdicts" in body
    assert "**BLOCKER**" in body
    assert "Verdict: safe" in body
    assert "codex" in body


def test_run_cross_family_timeout_is_captured_not_raised(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    def _runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=tmp_path / "p.md",
        report_path=report,
        timeout_seconds=3,
        runner=_runner,
    )

    assert result.ok is False
    assert "timed out" in (result.error or "")
    assert "UNAVAILABLE" in report.read_text(encoding="utf-8")


def test_run_cross_family_nonzero_exit_is_captured(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=tmp_path / "p.md",
        report_path=report,
        timeout_seconds=5,
        runner=_fake_completed(2, "partial output", "stderr boom"),
    )

    assert result.ok is False
    assert result.returncode == 2
    text = report.read_text(encoding="utf-8")
    assert "exited 2" in text
    assert "partial output" in text


def test_run_cross_family_missing_binary_is_captured(tmp_path: Path) -> None:
    report = tmp_path / "report.md"

    def _runner(command, **kwargs):
        raise FileNotFoundError("devin not on PATH")

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=tmp_path / "p.md",
        report_path=report,
        timeout_seconds=5,
        runner=_runner,
    )

    assert result.ok is False
    assert "failed to start" in (result.error or "")


def test_review_injects_cross_family_section_when_enabled(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    VALID_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    result = app.review(456)

    assert calls["n"] == 1
    assert result.data["cross_family_ok"] is True
    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    prompt_text = (prs_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "Cross-family adversarial pass" in prompt_text
    assert "leads, not verdicts" in prompt_text
    assert (prs_dir / "cross-family-review.md").exists()


def test_review_reuses_existing_cross_family_report(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}
    VALID_REPORT = "**MAJOR**\nissue\n\nVerdict: safe"

    # Set a consistent head SHA for the PR
    app.gh.pr_head_shas[456] = "sha-abc123"

    def _fake_run(**kwargs):
        calls["n"] += 1
        # Write a report with the current head SHA so it can be reused
        head_sha = kwargs.get("head_ref_oid", "sha-abc123")
        report_content = (
            f"# Cross-family adversarial review — `codex`\n\n"
            f"<!-- PR head SHA: {head_sha} -->\n\n"
            f"> Findings below are **leads, not verdicts**\n\n"
            f"---\n\n"
            f"{VALID_REPORT}\n"
        )
        Path(kwargs["report_path"]).write_text(report_content, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    app.review(456)
    app.review(456)

    assert calls["n"] == 1  # the second pass reused the report; codex did not re-run


def test_review_no_cross_family_override_skips(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)

    def _boom(**kwargs):
        raise AssertionError("cross-family must not run when disabled per call")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _boom)

    result = app.review(456, cross_family=False)

    assert result.data["cross_family_ok"] is None
    prompt_text = (
        tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    ).read_text(encoding="utf-8")
    assert "Cross-family adversarial pass" not in prompt_text


def test_review_skips_cross_family_for_draft_pr(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    app.gh.prs[0] = {**app.gh.prs[0], "isDraft": True}

    def _boom(**kwargs):
        raise AssertionError("cross-family must not run for a draft PR")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _boom)

    result = app.review(456)

    # Issue #818: the default fixture PR is otherwise fully janitor-green
    # (linked issue, tests mentioned, CLEAN merge state), so draft is the
    # ONLY failure -- the janitor gate now auto-readies it via `gh pr ready`
    # instead of parking it, deferring the actual review to the next poll
    # pass (the pass that never comes in this test, since _boom would raise
    # if cross-family review ran). The pre-fix pin (janitor_ok False,
    # "draft" in janitor_failures) no longer holds for a pure-draft block.
    assert result.ok is False
    assert result.data["draft_readied"] is True
    assert app.gh.pr_ready_calls == [456]


# --- Issue #38 regression: transient retry + empty/blocked report guard --------


def test_run_cross_family_retries_once_on_transient_rate_limit_then_success(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"
    calls: list[str] = []
    rate_msg = (
        "Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 1 minute."
    )

    def _runner(command, **kwargs):
        if not calls:
            calls.append("fail")
            return subprocess.CompletedProcess(command, 1, stdout="", stderr=rate_msg)
        calls.append("success")
        return subprocess.CompletedProcess(command, 0, stdout=VALID_CROSS_FAMILY_REPORT, stderr="")

    sleep_calls: list[float] = []
    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="attack this",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_runner,
        sleep=lambda s: sleep_calls.append(s),
    )

    assert result.ok is True
    assert result.returncode == 0
    assert calls == ["fail", "success"]
    assert sleep_calls == [90.0]
    assert "**MAJOR**" in report.read_text(encoding="utf-8")


def test_run_cross_family_rate_limit_retry_exhausted_then_fails(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"
    rate_msg = "Rate limit exceeded. Try again later."
    calls: list[str] = []

    def _runner(command, **kwargs):
        calls.append("fail")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=rate_msg)

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_runner,
        sleep=lambda s: None,
    )

    assert result.ok is False
    assert result.returncode == 1
    assert calls == ["fail", "fail"]
    assert "UNAVAILABLE" in report.read_text(encoding="utf-8")


def test_run_cross_family_exit_zero_blocked_output_is_stubbed(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    prompt = tmp_path / "prompt.md"
    blocked = (
        "I'm blocked from performing the review. All tool calls are being rejected. Please re-run."
    )

    result = run_cross_family_review(
        model="codex",
        command=("devin",),
        repo_root=tmp_path,
        prompt_text="x",
        prompt_path=prompt,
        report_path=report,
        timeout_seconds=5,
        runner=_fake_completed(0, blocked),
    )

    assert result.ok is False
    assert result.returncode == 0
    assert "UNAVAILABLE" in report.read_text(encoding="utf-8")
    assert "empty or blocked report" in (result.error or "")


def test_review_does_not_reuse_semantically_empty_cross_family_report(
    tmp_path: Path, monkeypatch
) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_CROSS_FAMILY_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    report_path = prs_dir / "cross-family-review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "I'm blocked from performing the review. Tool calls rejected. Please re-run.",
        encoding="utf-8",
    )

    app.review(456)

    assert calls["n"] == 1
    assert report_path.read_text(encoding="utf-8") == VALID_CROSS_FAMILY_REPORT


def test_run_cross_family_sanitizes_environment_at_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_cross_family_review must pass sanitized env to the actual subprocess runner."""
    import subprocess
    from charlie_work.rescue_review import run_cross_family_review

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    report_path = tmp_path / "report.md"
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("test prompt", encoding="utf-8")

    # Set parent env variables (simulating orchestrator leak)
    monkeypatch.setenv("VIRTUAL_ENV", "/orchestrator/.venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/orchestrator/.venv")

    captured_env: dict[str, str] | None = None

    def _fake_runner(command, **kwargs):
        nonlocal captured_env
        captured_env = kwargs.get("env")
        # Return a valid report
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="**MINOR**\nissue\n\nVerdict: safe",
            stderr="",
        )

    result = run_cross_family_review(
        model="codex",
        command=("echo", "test"),
        repo_root=repo_root,
        prompt_text="test prompt",
        prompt_path=prompt_path,
        report_path=report_path,
        timeout_seconds=30,
        runner=_fake_runner,
    )

    assert result.ok is True
    assert captured_env is not None, "Runner should have received env kwarg"
    assert "VIRTUAL_ENV" not in captured_env, (
        "VIRTUAL_ENV must be sanitized in the actual subprocess env"
    )
    assert "UV_PROJECT_ENVIRONMENT" not in captured_env, (
        "UV_PROJECT_ENVIRONMENT must be sanitized in the actual subprocess env"
    )


def test_review_does_not_reuse_legacy_wrapped_blocked_report(tmp_path: Path, monkeypatch) -> None:
    """Regression for issue #38: a legacy wrapped report whose body is a blocked
    refusal must not be reused as a success report on subsequent passes.
    """
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text(VALID_CROSS_FAMILY_REPORT, encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    report_path = prs_dir / "cross-family-review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    blocked = "I'm blocked from performing the review. Tool calls rejected. Please re-run."
    report_path.write_text(
        f"# Cross-family adversarial review — `codex`\n\n{_CAVEAT}\n\n---\n\n{blocked}\n",
        encoding="utf-8",
    )

    app.review(456)

    assert calls["n"] == 1
    assert report_path.read_text(encoding="utf-8") == VALID_CROSS_FAMILY_REPORT


def test_cross_family_failure_stub_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "cross-family-review.md").write_text(
        "# Cross-family adversarial review — `codex` (UNAVAILABLE)\n\n> timed out\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text("# real findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    result = app.review(456)

    assert calls["n"] == 1  # the stub did NOT satisfy the reuse check
    assert result.data["cross_family_ok"] is True


def test_cross_family_report_invalidated_on_head_sha_change(tmp_path: Path, monkeypatch) -> None:
    """Regression test for issue #156: cross-family reports must be invalidated
    when the PR head SHA changes to prevent reviewing stale code."""
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)

    # Create a report with an old head SHA
    old_head_sha = "abc123def456"
    report_content = (
        f"# Cross-family adversarial review — `codex`\n\n"
        f"<!-- PR head SHA: {old_head_sha} -->\n\n"
        f"> Findings below are **leads, not verdicts**\n\n"
        f"---\n\n"
        f"**MAJOR**\nfile:line: old bug\n\nVerdict: needs work\n"
    )
    (pr_dir / "cross-family-review.md").write_text(report_content, encoding="utf-8")

    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        # Verify that the new head SHA is passed to run_cross_family_review
        assert kwargs.get("head_ref_oid") == "newheadsha789"
        Path(kwargs["report_path"]).write_text("# new findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    # Update the PR to have a new head SHA
    app.gh.pr_head_shas[456] = "newheadsha789"

    result = app.review(456)

    # The old report should NOT be reused because the head SHA changed
    assert calls["n"] == 1
    assert result.data["cross_family_ok"] is True


def test_cross_family_report_reused_when_head_sha_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Regression test for issue #156: cross-family reports should be reused
    when the PR head SHA has not changed."""
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)

    # Create a report with a head SHA
    head_sha = "abc123def456"
    report_content = (
        f"# Cross-family adversarial review — `codex`\n\n"
        f"<!-- PR head SHA: {head_sha} -->\n\n"
        f"> Findings below are **leads, not verdicts**\n\n"
        f"---\n\n"
        f"**MAJOR**\nfile:line: bug\n\nVerdict: needs work\n"
    )
    (pr_dir / "cross-family-review.md").write_text(report_content, encoding="utf-8")

    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text("# new findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", _fake_run)

    # The PR still has the same head SHA
    app.gh.pr_head_shas[456] = head_sha

    result = app.review(456)

    # The report should be reused because the head SHA has not changed
    assert calls["n"] == 0
    assert result.data["cross_family_ok"] is True
    assert result.data["cross_family_reused"] is True


def test_cross_family_verdict_post_init_rejects_content_free_request_changes() -> None:
    """Issue #784 AC-6: the invalid state -- request_changes with neither
    itemized required_changes nor a real summary -- must be unrepresentable
    at construction, not just avoided by callers that remember to check."""
    with pytest.raises(ValueError, match="content-free"):
        CrossFamilyVerdict(decision="request_changes", summary="", required_changes=())


def test_cross_family_verdict_post_init_rejects_whitespace_only_summary() -> None:
    """Whitespace-only is not a real summary either -- ``.strip()`` is
    applied before the emptiness check, so padding cannot smuggle a
    content-free verdict past the guard."""
    with pytest.raises(ValueError, match="content-free"):
        CrossFamilyVerdict(decision="request_changes", summary="   \n  ", required_changes=())


def test_cross_family_verdict_post_init_allows_request_changes_with_only_summary() -> None:
    """Narrower than "always require required_changes": the legacy Markdown
    parse path never itemizes findings, so a request_changes verdict with a
    real extracted summary and empty required_changes remains legitimate
    and constructible -- this is exactly what the legacy-path tests above
    rely on."""
    verdict = CrossFamilyVerdict(
        decision="request_changes", summary="a real extracted summary", required_changes=()
    )
    assert verdict.summary == "a real extracted summary"


def test_cross_family_verdict_post_init_allows_request_changes_with_only_required_changes() -> (
    None
):
    """A JSON-block verdict with itemized required_changes but an empty
    summary is also legitimate -- required_changes alone is something a
    rework brief can act on."""
    verdict = CrossFamilyVerdict(
        decision="request_changes", summary="", required_changes=("fix the null check",)
    )
    assert verdict.required_changes == ("fix the null check",)


def test_cross_family_verdict_post_init_allows_approved_with_empty_summary() -> None:
    """The guard is scoped to ``request_changes`` only -- an approved
    verdict never needs anything for a rework brief to act on, so an empty
    summary there is unaffected."""
    verdict = CrossFamilyVerdict(decision="approved", summary="")
    assert verdict.decision == "approved"


def test_dry_run_skips_cross_family_review(monkeypatch, tmp_path: Path) -> None:
    """Test that --dry-run prevents cross-family model subprocess execution."""
    subprocess_calls: list[list[str]] = []

    def fake_run(*args, **kwargs):
        subprocess_calls.append(args[0])
        raise AssertionError("subprocess.run should not be called in dry-run mode")

    monkeypatch.setattr("charlie_work.rescue_review.subprocess.run", fake_run)

    result = run_cross_family_review(
        model="test-model",
        command=["echo", "test"],
        repo_root=tmp_path,
        prompt_text="test prompt",
        prompt_path=tmp_path / "prompt.md",
        report_path=tmp_path / "report.md",
        timeout_seconds=30,
        dry_run=True,
    )

    assert result.ok is False
    assert result.error == "DRY-RUN: cross-family review not executed"
    assert len(subprocess_calls) == 0  # No subprocess should be invoked
