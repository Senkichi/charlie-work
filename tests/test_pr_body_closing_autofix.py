"""Issue #2108: a PR-body-only closing-keyword gate failure is orchestrator-fixable.

The CI-red short-circuit in ``review()`` must repair a Lint failure whose only
failing step is the closing-keyword gate ``via pr body`` (body edit + Lint
re-run, no rework), while ``via commit message`` findings and mixed failures
keep routing to worker rework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from _fakes_github_rerun import FakeGitHubWithRerunCapture
from _review_fixtures import _required_checks_config
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.pr_body_closing_autofix import rewrite_unexpected_body_references
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

_LINT = "Lint & Format"
_GATE_STEP = "Closing-keyword gate (issue 790)"
_LINK = "https://github.com/o/r/actions/runs/900/job/901"
_BAD_BODY = "Closes #123\n\nTests: added.\n\nThis closed #60 earlier."


def _step(name: str, conclusion: str) -> dict[str, Any]:
    return {"name": name, "conclusion": conclusion}


class _GateFake(FakeGitHubWithRerunCapture):
    def __init__(
        self,
        *,
        failed_steps: list[str],
        commit_messages: list[str] | None = None,
        body: str = _BAD_BODY,
        extra_failing_check: bool = False,
        **kwargs: Any,
    ) -> None:
        checks = [
            {"name": "Tests passed", "state": "SUCCESS"},
            {"name": _LINT, "state": "FAILURE", "link": _LINK},
            {"name": "Pre-commit", "state": "SUCCESS"},
        ]
        if extra_failing_check:
            checks[0] = {"name": "Tests passed", "state": "FAILURE"}
        super().__init__(checks=checks, **kwargs)
        self.prs[0]["body"] = body
        self.failed_steps = failed_steps
        self.commit_messages = commit_messages or []

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        steps = [_step("Set up job", "success")]
        steps += [_step(name, "failure") for name in self.failed_steps]
        return {"conclusion": "failure", "steps": steps}

    def pr_commits(self, number: int) -> list[dict[str, Any]] | None:
        return [
            {"sha": f"c{i}", "parents": [], "commit": {"message": m}}
            for i, m in enumerate(self.commit_messages)
        ]


def _review(tmp_path: Path, fake: _GateFake, *, passes: int = 1) -> tuple[Any, Any, Any]:
    config = _required_checks_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake.diffs[456] = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b"
    app = OrchestratorApp(tmp_path, paths, config, fake)
    # passes=2 lets the pre-existing one-shot flake rerun (#391) fire first, so the
    # second pass reaches the rework routing under test.
    for _ in range(passes):
        result = app.review(456)
    return result, paths, config


def _kinds(paths: Any) -> list[str]:
    return [e["kind"] for e in query_events(paths.state_file)]


def test_body_only_closing_keyword_failure_edits_body_and_reruns_without_rework(
    tmp_path: Path,
) -> None:
    fake = _GateFake(failed_steps=[_GATE_STEP])

    result, paths, config = _review(tmp_path, fake)

    assert result.ok is False
    assert fake.pr_edits == [
        (456, "Closes #123\n\nTests: added.\n\nThis closed issue 60 earlier.")
    ]
    assert fake.rerun_calls == [["run", "rerun", "900", "--failed"]]
    assert "pr_body_closing_keyword_autofixed" in _kinds(paths)
    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("decision") != "request_changes"
    assert (123, config.labels.needs_rework) not in fake.labels_added
    assert not (paths.prs / "pr-456" / "rework-prompt.md").exists()


def test_commit_message_closing_keyword_failure_still_routes_to_rework(tmp_path: Path) -> None:
    fake = _GateFake(
        failed_steps=[_GATE_STEP],
        body="Closes #123\n\nTests: added.",
        commit_messages=["Fixes #60"],
    )

    result, paths, config = _review(tmp_path, fake, passes=2)

    assert result.ok is True
    assert fake.pr_edits == []
    assert len(fake.rerun_calls) == 1  # only the pre-existing flake debounce
    assert "pr_body_closing_keyword_autofixed" not in _kinds(paths)
    assert load_state(paths.state_file)["prs"]["456"]["decision"] == "request_changes"
    assert (123, config.labels.needs_rework) in fake.labels_added


def test_mixed_closing_keyword_and_real_failure_still_routes_to_rework(tmp_path: Path) -> None:
    fake = _GateFake(failed_steps=[_GATE_STEP, "Run tests"])

    result, paths, config = _review(tmp_path, fake, passes=2)

    assert result.ok is True
    assert fake.pr_edits == []
    assert len(fake.rerun_calls) == 1  # only the pre-existing flake debounce
    assert "pr_body_closing_keyword_autofixed" not in _kinds(paths)
    assert (123, config.labels.needs_rework) in fake.labels_added


def test_body_edit_failure_escalates_instead_of_rework(tmp_path: Path) -> None:
    fake = _GateFake(failed_steps=[_GATE_STEP])

    def _boom(number: int, body_file: Path) -> None:
        from charlie_work.github import GitHubError

        raise GitHubError("edit refused")

    fake.pr_edit = _boom  # type: ignore[method-assign]

    result, paths, config = _review(tmp_path, fake)

    assert result.ok is False
    assert fake.rerun_calls == []
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert "pr_body_closing_keyword_autofix_failed" in _kinds(paths)
    assert (123, config.labels.needs_rework) not in fake.labels_added


def test_rerun_refused_while_running_holds_without_escalation(tmp_path: Path) -> None:
    fake = _GateFake(
        failed_steps=[_GATE_STEP], rerun_ok=False, rerun_error="run 900 is already running"
    )

    result, paths, _config = _review(tmp_path, fake)

    assert result.ok is False
    assert result.data.get("already_running") is True
    assert load_state(paths.state_file)["issues"].get("123", {}).get("status") != "escalated"


def test_rewrite_keeps_declared_target_and_negated_references() -> None:
    body = "Closes #123. Also fixes #60 and resolved #61, but does not close #62."

    out = rewrite_unexpected_body_references(body, intended=123)

    assert out == "Closes #123. Also fixes issue 60 and resolved issue 61, but does not close #62."
