from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from devin_orchestrator import cli
from devin_orchestrator import github as github_module
from devin_orchestrator.checks import summarize_checks
from devin_orchestrator.config import (
    CrossFamilyConfig,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    RuntimeConfig,
    find_config_path,
    load_config,
)
from devin_orchestrator.cross_family import (
    CrossFamilyResult,
    render_command,
    run_cross_family_review,
)
from devin_orchestrator.github import label_names, linked_issue_number
from devin_orchestrator.paths import runtime_paths
from devin_orchestrator.prompts import render_prompt
from devin_orchestrator.state import load_state, save_state
from devin_orchestrator.workflow import OrchestratorApp, slugify

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_default_config_enables_auto_merge() -> None:
    config = load_config()

    assert config.auto_merge.enabled is True
    # A shared package cannot know a consumer's CI check names; unconfigured
    # means empty, and `doctor` flags it.
    assert config.auto_merge.required_checks == ()
    assert config.labels.ready == "automated-ready"


def test_runtime_paths_are_repo_relative(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path, ".var/devin-orchestrator")

    assert paths.root == tmp_path / ".var" / "devin-orchestrator"
    assert paths.state_file == paths.root / "state.json"


def test_state_round_trip(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state = load_state(state_path)
    state["issues"]["123"] = {"title": "Example"}

    save_state(state_path, state)
    loaded = load_state(state_path)

    assert loaded["issues"]["123"]["title"] == "Example"
    assert loaded["version"] == 1


def test_worker_prompt_renders_issue_values() -> None:
    prompt = render_prompt(
        "worker.md",
        {
            "issue_number": 123,
            "issue_title": "Fix search",
            "issue_url": "https://example.test/issues/123",
            "issue_body": "Body text",
            "branch_name": "agent/issue-123-fix-search",
            "worker_model_tier": "capable",
        },
    )

    assert "Issue #123" in prompt
    assert "agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_claude_code_worker_prompt_renders_issue_values() -> None:
    prompt = render_prompt(
        "worker_claude_code.md",
        {
            "issue_number": 123,
            "issue_title": "Fix search",
            "issue_url": "https://example.test/issues/123",
            "issue_body": "Body text",
            "branch_name": "agent/issue-123-fix-search",
            "worker_model_tier": "capable",
        },
    )

    assert "Issue #123" in prompt
    assert "git switch -c agent/issue-123-fix-search" in prompt
    assert "Closes #123" in prompt


def test_repo_local_prompt_dir_overrides_package_template(tmp_path: Path) -> None:
    override_dir = tmp_path / "my-prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text(
        "CUSTOM for #$issue_number on $branch_name", encoding="utf-8"
    )

    prompt = render_prompt(
        "worker.md",
        {"issue_number": 5, "branch_name": "agent/issue-5-x"},
        search_dirs=(override_dir,),
    )

    assert prompt == "CUSTOM for #5 on agent/issue-5-x"


def test_missing_repo_local_template_falls_back_to_package(tmp_path: Path) -> None:
    override_dir = tmp_path / "my-prompts"
    override_dir.mkdir()

    prompt = render_prompt(
        "rework.md",
        {"pr_number": 9, "pr_title": "t", "pr_url": "u", "issue_number": 1, "review_summary": "s"},
        search_dirs=(override_dir,),
    )

    assert "Rework Task: PR #9" in prompt


def test_slugify_makes_branch_safe_slug() -> None:
    assert slugify("Fix: Search / Windows path!!!") == "fix-search-windows-path"


def test_label_names_accepts_gh_shape() -> None:
    issue = {"labels": [{"name": "automated-ready"}, {"name": "agent:in-progress"}]}

    assert label_names(issue) == {"automated-ready", "agent:in-progress"}


def test_linked_issue_number_from_branch_body_or_title() -> None:
    assert linked_issue_number({"headRefName": "agent/issue-456-fix"}) == 456
    assert linked_issue_number({"body": "Closes #789"}) == 789
    assert linked_issue_number({"title": "Fix #321: thing"}) == 321


def test_linked_issue_number_ignores_unqualified_body_references() -> None:
    body = "Bumps actions/checkout. See dependabot/dependabot-core#2454 for details."

    assert linked_issue_number({"body": body}) is None


def test_summarize_checks_requires_all_configured_checks() -> None:
    checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "conclusion": "FAILURE"},
    ]

    summary = summarize_checks(checks, ("Tests passed", "Lint & Format", "Pre-commit"))

    assert summary.ready is False
    assert summary.passed == ("Tests passed", "Lint & Format")
    assert summary.failed == ("Pre-commit",)


def test_state_json_is_valid_after_save(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    save_state(state_path, {"version": 1, "issues": {}, "prs": {}, "events": []})

    payload = json.loads(state_path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert payload["generated_at"].endswith("Z")


def test_find_config_path_prefers_explicit_then_repo_root(tmp_path: Path) -> None:
    explicit = tmp_path / "elsewhere.yaml"
    assert find_config_path(tmp_path, explicit) == explicit

    assert find_config_path(tmp_path) is None

    repo_config = tmp_path / "orchestrator.config.yaml"
    repo_config.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")
    assert find_config_path(tmp_path) == repo_config


def test_cli_accepts_json_after_subcommand(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_app", lambda args: object())
    monkeypatch.setattr(
        cli,
        "run_command",
        lambda app, args: cli.CommandResult(True, "ok", {"json_output": args.json_output}),
    )

    assert cli.main(["status", "--json"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["data"]["json_output"] is True


def test_github_run_parses_allow_failure_json_stdout(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout='[{"name": "Tests passed", "state": "FAILURE"}]',
            stderr="checks failed",
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    result = github_module.GitHub(tmp_path).run(
        ["pr", "checks", "123"], json_output=True, allow_failure=True
    )

    assert result == [{"name": "Tests passed", "state": "FAILURE"}]


def _required_checks_config(**kwargs) -> OrchestratorConfig:
    from devin_orchestrator.config import AutoMergeConfig

    auto_merge = AutoMergeConfig(
        required_checks=("Tests passed", "Lint & Format", "Pre-commit"), **kwargs
    )
    return OrchestratorConfig(auto_merge=auto_merge)


class FakeGitHub:
    def __init__(self) -> None:
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
            }
        ]
        self.pr = {
            "number": 456,
            "title": "Fix #123: search",
            "url": "https://example.test/pull/456",
            "headRefName": "agent/issue-123-fix-search",
            "body": "Closes #123",
            "labels": [],
        }
        self.labels_added: list[tuple[int, str]] = []
        self.labels_removed: list[tuple[int, str]] = []
        self.merged: list[tuple[int, str]] = []
        self.deleted_branches: list[str] = []
        self.delete_branch_ok = True

    def issue_list(self, ready_label: str):
        return self.issues

    def issue_view(self, number: int):
        return self.issues[0]

    def pr_list(self):
        return [self.pr]

    def pr_view(self, number: int):
        return self.pr

    def pr_checks(self, number: int):
        return [
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": "Lint & Format", "bucket": "pass"},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]

    def pr_diff(self, number: int):
        return "diff --git a/file b/file"

    def add_issue_label(self, number: int, label: str) -> None:
        self.labels_added.append((number, label))

    def remove_issue_label(self, number: int, label: str) -> None:
        self.labels_removed.append((number, label))

    def merge_pr(self, number: int, strategy: str) -> str:
        self.merged.append((number, strategy))
        return "merged"

    def delete_branch(self, branch: str) -> bool:
        self.deleted_branches.append(branch)
        return self.delete_branch_ok

    def label_create(self, label: str, color: str, description: str) -> None:
        pass

    def pr_comment(self, number: int, body_file: Path) -> None:
        pass


def test_dispatch_writes_worker_prompt_and_session_manifest(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    prompt_path = (
        tmp_path / ".var" / "devin-orchestrator" / "issues" / "issue-123" / "worker-prompt.md"
    )
    manifest_path = (
        tmp_path / ".var" / "devin-orchestrator" / "dispatches" / "session-manifest.json"
    )
    assert prompt_path.exists()
    assert manifest_path.exists()
    assert "Closes #123" in prompt_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sessions"][0]["branch_name"] == "agent/issue-123-fix-search"
    # Manual adapter honesty: a written manifest means QUEUED — no worker has
    # been independently confirmed, so in-progress must not be applied.
    assert (123, "agent:queued") in fake_gh.labels_added
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_dispatch_only_issues_selects_explicit_subset(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Numbers not among the dispatchable candidates are skipped; only the
    # explicit, dispatchable match is selected (dependency-ordered waves).
    result = app.dispatch(only_issues="999, 123")

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert (123, "agent:queued") in fake_gh.labels_added


def test_dispatch_worker_template_selects_claude_code_variant(tmp_path: Path) -> None:
    config = OrchestratorConfig(dispatch=DispatchConfig(worker_template="worker_claude_code.md"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.dispatch(limit=1)

    prompt_path = (
        tmp_path / ".var" / "devin-orchestrator" / "issues" / "issue-123" / "worker-prompt.md"
    )
    text = prompt_path.read_text(encoding="utf-8")
    assert "git switch -c agent/issue-123-fix-search" in text  # Claude Code loop
    assert "/create-branch" not in text  # not the Devin skills loop


def test_app_prompts_dir_override_wins_for_worker_prompt(tmp_path: Path) -> None:
    override_dir = tmp_path / "orchestrator-prompts"
    override_dir.mkdir()
    (override_dir / "worker.md").write_text("REPO-LOCAL #$issue_number", encoding="utf-8")
    config = OrchestratorConfig(runtime=RuntimeConfig(prompts_dir="orchestrator-prompts"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.dispatch(limit=1)

    prompt_path = (
        tmp_path / ".var" / "devin-orchestrator" / "issues" / "issue-123" / "worker-prompt.md"
    )
    assert prompt_path.read_text(encoding="utf-8") == "REPO-LOCAL #123"


def test_command_dispatch_labels_only_successful_launches(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            ),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data["failed_count"] == 0
    assert result.data["dispatch_results"][0]["stdout"].strip() == "123"
    assert (123, "agent:in-progress") in fake_gh.labels_added
    results_path = tmp_path / ".var" / "devin-orchestrator" / "dispatches" / "session-results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["results"][0]["ok"] is True


def test_command_dispatch_failure_does_not_label_in_progress(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(7)"),
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["selected_count"] == 0
    assert result.data["failed_count"] == 1
    assert result.data["dispatch_results"][0]["returncode"] == 7
    assert (123, "agent:in-progress") not in fake_gh.labels_added
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"


def test_merge_ready_requires_approved_decision_then_merges(tmp_path: Path) -> None:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    not_ready = app.merge_ready(456)
    assert not_ready.data["can_merge"] is False
    assert fake_gh.merged == []

    decision_dir = tmp_path / ".var" / "devin-orchestrator" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved"}), encoding="utf-8"
    )

    ready = app.merge_ready(456)

    assert ready.data["can_merge"] is True
    assert ready.data["merged"] is True
    assert fake_gh.merged == [(456, "squash")]
    assert (123, "agent:done") in fake_gh.labels_added
    assert fake_gh.deleted_branches == ["agent/issue-123-fix-search"]
    assert ready.data["branch_deleted"] is True


def test_merge_ready_branch_delete_failure_never_blocks_labels(tmp_path: Path) -> None:
    """The empericus failure mode: a branch checked out in a local worktree made
    `gh pr merge --delete-branch` abort the post-merge label update. Deletion is
    now decoupled and best-effort — labels always land."""
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.delete_branch_ok = False
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "devin-orchestrator" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved"}), encoding="utf-8"
    )

    ready = app.merge_ready(456)

    assert ready.data["merged"] is True
    assert ready.data["branch_deleted"] is False
    assert (123, "agent:done") in fake_gh.labels_added


def test_merge_ready_honors_delete_branch_false(tmp_path: Path) -> None:
    config = _required_checks_config(delete_branch=False)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    decision_dir = tmp_path / ".var" / "devin-orchestrator" / "prs" / "pr-456"
    decision_dir.mkdir(parents=True)
    (decision_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved"}), encoding="utf-8"
    )

    ready = app.merge_ready(456)

    assert ready.data["merged"] is True
    assert fake_gh.deleted_branches == []
    assert ready.data["branch_deleted"] is None


def test_github_delete_branch_failure_returns_false(monkeypatch, tmp_path: Path) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args, output="", stderr="Reference does not exist")

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    assert github_module.GitHub(tmp_path).delete_branch("agent/issue-1-x") is False


# --- Cross-family adversarial review ------------------------------------------


def _fake_completed(returncode: int = 0, stdout: str = "## MAJOR\nx", stderr: str = ""):
    def _runner(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    return _runner


def _cross_family_app(tmp_path: Path, *, enabled: bool) -> OrchestratorApp:
    config = OrchestratorConfig(cross_family=CrossFamilyConfig(enabled=enabled))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, FakeGitHub())


def test_render_command_templates_list_and_string() -> None:
    values = {"model": "codex", "prompt_path": "/tmp/p.md"}
    assert render_command(
        ("devin", "--model", "{model}", "-p", "--prompt-file", "{prompt_path}"), values
    ) == ["devin", "--model", "codex", "-p", "--prompt-file", "/tmp/p.md"]
    assert render_command("devin --model {model}", values) == "devin --model codex"


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
        runner=_fake_completed(0, "## BLOCKER\nboom"),
    )

    assert result.ok is True
    assert result.returncode == 0
    assert prompt.read_text(encoding="utf-8") == "attack this"
    body = report.read_text(encoding="utf-8")
    assert "leads, not verdicts" in body
    assert "## BLOCKER" in body
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


def test_devin_example_config_enables_cross_family() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.devin.yaml")

    assert config.cross_family.enabled is True
    assert config.cross_family.model == "codex"
    assert config.cross_family.command[0] == "devin"
    assert config.dispatch.worker_template == "worker.md"


def test_claude_code_example_config_selects_claude_worker() -> None:
    config = load_config(EXAMPLES_DIR / "orchestrator.config.claude-code.yaml")

    assert config.dispatch.worker_template == "worker_claude_code.md"
    assert config.cross_family.enabled is False


def test_config_absent_cross_family_block_defaults_disabled(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("labels:\n  ready: automated-ready\n", encoding="utf-8")

    config = load_config(path)

    assert config.cross_family.enabled is False


def test_config_parses_cross_family_command_list_to_tuple(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "cross_family:\n  enabled: true\n  model: codex\n"
        "  command: [devin, --model, '{model}']\n  timeout_seconds: 120\n",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.cross_family.command == ("devin", "--model", "{model}")
    assert config.cross_family.timeout_seconds == 120


def test_review_injects_cross_family_section_when_enabled(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text("codex findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("devin_orchestrator.workflow.run_cross_family_review", _fake_run)

    result = app.review(456)

    assert calls["n"] == 1
    assert result.data["cross_family_ok"] is True
    prs_dir = tmp_path / ".var" / "devin-orchestrator" / "prs" / "pr-456"
    prompt_text = (prs_dir / "review-prompt.md").read_text(encoding="utf-8")
    assert "Cross-family adversarial pass" in prompt_text
    assert "leads, not verdicts" in prompt_text
    assert (prs_dir / "cross-family-review.md").exists()


def test_review_reuses_existing_cross_family_report(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    calls = {"n": 0}

    def _fake_run(**kwargs):
        calls["n"] += 1
        Path(kwargs["report_path"]).write_text("codex findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("devin_orchestrator.workflow.run_cross_family_review", _fake_run)

    app.review(456)
    app.review(456)

    assert calls["n"] == 1  # the second pass reused the report; codex did not re-run


def test_review_no_cross_family_override_skips(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)

    def _boom(**kwargs):
        raise AssertionError("cross-family must not run when disabled per call")

    monkeypatch.setattr("devin_orchestrator.workflow.run_cross_family_review", _boom)

    result = app.review(456, cross_family=False)

    assert result.data["cross_family_ok"] is None
    prompt_text = (
        tmp_path / ".var" / "devin-orchestrator" / "prs" / "pr-456" / "review-prompt.md"
    ).read_text(encoding="utf-8")
    assert "Cross-family adversarial pass" not in prompt_text


def test_review_skips_cross_family_for_draft_pr(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    app.gh.pr = {**app.gh.pr, "isDraft": True}

    def _boom(**kwargs):
        raise AssertionError("cross-family must not run for a draft PR")

    monkeypatch.setattr("devin_orchestrator.workflow.run_cross_family_review", _boom)

    result = app.review(456)

    assert result.data["cross_family_ok"] is None


def test_spec_review_runs_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    spec = tmp_path / "SPEC.md"
    spec.write_text("# My spec\nclaims", encoding="utf-8")
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    def _fake_run(**kwargs):
        assert "My spec" in kwargs["prompt_text"]  # artifact text inlined into the prompt
        Path(kwargs["report_path"]).write_text("spec findings", encoding="utf-8")
        return CrossFamilyResult(ok=True, report_path=str(kwargs["report_path"]), model="codex")

    monkeypatch.setattr("devin_orchestrator.workflow.run_cross_family_review", _fake_run)

    result = app.spec_review(spec)

    assert result.ok is True
    assert Path(result.data["report_path"]).read_text(encoding="utf-8") == "spec findings"


def test_spec_review_missing_file_returns_error(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.spec_review(tmp_path / "nope.md")

    assert result.ok is False


# --- P0 fixes: state safety, label honesty, rework cap, loop isolation --------


def test_load_state_quarantines_corrupt_file(tmp_path: Path) -> None:
    from devin_orchestrator.state import load_state as _load

    state_path = tmp_path / "state.json"
    state_path.write_text("{truncated garbage", encoding="utf-8")

    state = _load(state_path)

    assert state["issues"] == {}
    assert not state_path.exists()
    quarantined = list(tmp_path.glob("state.json.corrupt-*"))
    assert len(quarantined) == 1
    assert "truncated garbage" in quarantined[0].read_text(encoding="utf-8")


def test_review_preserves_recorded_decision_in_state(tmp_path: Path) -> None:
    from devin_orchestrator.state import load_state as _load
    from devin_orchestrator.state import save_state as _save

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    state = _load(paths.state_file)
    state["prs"]["456"] = {"decision": "approved", "custom": "kept"}
    _save(paths.state_file, state)

    app.review(456)

    after = _load(paths.state_file)
    assert after["prs"]["456"]["decision"] == "approved"  # was clobbered pre-fix
    assert after["prs"]["456"]["custom"] == "kept"
    assert after["prs"]["456"]["status"] == "reviewing"


def test_string_dispatch_command_rejects_issue_title(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        devin=DevinConfig(adapter="command", dispatch_command="echo {issue_title}")
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)

    assert result.ok is False
    assert result.data["failed_count"] == 1
    assert "list form" in result.data["dispatch_results"][0]["error"]
    assert (123, "agent:in-progress") not in fake_gh.labels_added


def test_rework_cap_escalates_to_human(tmp_path: Path) -> None:
    config = OrchestratorConfig()  # max_rework_cycles = 2
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    first = app.record_review(456, "request_changes", summary="fix A")
    second = app.record_review(456, "request_changes", summary="fix B")
    third = app.record_review(456, "request_changes", summary="fix C")

    assert first.data["escalated"] is False and first.data["rework_path"]
    assert second.data["escalated"] is False and second.data["rework_path"]
    assert third.data["escalated"] is True
    assert third.data["rework_path"] is None  # no third rework prompt
    assert fake_gh.labels_added.count((123, "agent:needs-rework")) == 2
    assert (123, "agent:human-needed") in fake_gh.labels_added


def test_cross_family_failure_stub_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    app = _cross_family_app(tmp_path, enabled=True)
    pr_dir = tmp_path / ".var" / "devin-orchestrator" / "prs" / "pr-456"
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

    monkeypatch.setattr("devin_orchestrator.workflow.run_cross_family_review", _fake_run)

    result = app.review(456)

    assert calls["n"] == 1  # the stub did NOT satisfy the reuse check
    assert result.data["cross_family_ok"] is True


def test_loop_isolates_per_pr_errors(tmp_path: Path) -> None:
    from devin_orchestrator.github import GitHubError as _GitHubError

    class ExplodingGitHub(FakeGitHub):
        def pr_view(self, number: int):
            raise _GitHubError("merge conflict boom")

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, ExplodingGitHub())

    result = app.loop(limit=0)

    assert result.data["errors"] == [{"pr": 456, "error": "merge conflict boom"}]
    assert result.ok is False


def test_run_captured_decodes_bytes_safely(tmp_path: Path) -> None:
    from devin_orchestrator.subprocess_runner import run_captured

    result = run_captured(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'caf' + bytes([0xE9]))"],
        cwd=tmp_path,
        timeout_seconds=30,
    )

    assert result.ok is True
    assert result.stdout == "caf�"  # invalid UTF-8 replaced, never raises
