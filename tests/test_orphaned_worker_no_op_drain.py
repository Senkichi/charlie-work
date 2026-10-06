"""App-level coverage for the #2034 no-op rework disposition.

A rework worker that exits without changing the PR used to leave the issue
``dispatched`` on ``agent:in-progress`` forever (one fingerprinted drift event,
no consumer). ``orphaned_worker_no_op_drain`` gives every such finding a
disposition; these tests drive the real sweep with the real
``OrchestratorApp.record_review`` / ``review`` and pin one test per branch:

- live CI red -> CI-failure rework whose brief names the failing step (the #2005
  shape: exit 0, only a merge commit pushed, Lint red);
- CI green + the worker rebutted the verdict -> ``review()`` once per head;
- otherwise, or a repeat no-op on the same head -> ``rework_no_op`` escalation;
- CI not settled -> deliberate deferral (retry, #654-bounded): the checks fetch
  returning ``None``, or a required check still pending, missing, infra-failed,
  or infra-blocked -- on both the clean-exit route and the head-change route
  (the janitor-flagged ``is_no_op_rework`` refusal), including the head-change
  retry itself: a deferred route must be re-collected on later passes (the
  review drain's fingerprint stamp would otherwise wedge it), and the retried
  pass disposes once CI settles red or green.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
from _fakes_github import FakeGitHubWithChecksAndAnnotations
from _orphan_sweep_fixtures import _dead_worker_rework_bed, _write_outcome
from _rework_dispatch_fixtures import _wg
from _host_fixtures import host_probe
from charlie_work.config import (
    AutoMergeConfig,
    DevinConfig,
    OrchestratorConfig,
    WatchdogConfig,
    WorkerRoleConfig,
)
from charlie_work.janitor import _calculate_patch_id
from charlie_work.process_utils import write_worker_terminal_status
from charlie_work.state import load_state, save_state
from charlie_work.workflow import CommandResult, OrchestratorApp

_LINT = "Lint & Format"
_LINT_LINK = "https://github.com/o/r/actions/runs/5/job/77"
_LOG = "\n".join(f"noise line {i}" for i in range(80)) + "\nsrc/foo.py:3:1: E501 line too long\n"
# A real unified diff: _calculate_patch_id shells out to `git patch-id --stable`,
# which yields "" for a hunk-less stub -- the janitor's patch-id no-op gate only
# fires (and is_no_op_rework propagates out of review()) when this matches the
# verdict's recorded reviewed_patch_id.
_DIFF = """\
diff --git a/file b/file
index 0000000..1111111 100644
--- a/file
+++ b/file
@@ -1,1 +1,1 @@
-old
+new
"""


class _CiGh(FakeGitHubWithChecksAndAnnotations):
    """FakeGitHub with a failing Lint job: step list and log tail available."""

    def __init__(self, checks: list[dict[str, Any]] | None) -> None:
        super().__init__(checks=checks)
        self.checks_unavailable = False

    def pr_checks(self, number: int):
        return None if self.checks_unavailable else super().pr_checks(number)

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        return {
            "conclusion": "failure",
            "steps": [
                {"name": "Set up job", "conclusion": "success"},
                {"name": "Run ruff check", "conclusion": "failure"},
            ],
        }

    def run(self, args, *, json_output: bool = False, allow_failure: bool = False):
        if any(str(a).endswith("/logs") for a in args):
            return _LOG
        return super().run(args, json_output=json_output, allow_failure=allow_failure)


def _bed(
    tmp_path: Path,
    *,
    checks: list[dict[str, Any]],
    live_head: str = "abc123",
    decision: str = "request_changes",
    pr_state_status: str | None = None,
):
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        watchdog=WatchdogConfig(enabled=True, stall_minutes=20),
        auto_merge=AutoMergeConfig(required_checks=(_LINT,)),
    )
    config, paths, bed_gh, _ = _dead_worker_rework_bed(
        tmp_path,
        decision=decision,
        pr_state_status=pr_state_status,
        config=config,
        janitor_green=True,
    )
    gh = _CiGh(checks)
    gh.repo_root = tmp_path
    gh.issues = bed_gh.issues
    gh.prs = bed_gh.prs
    gh.prs[0]["headRefOid"] = live_head
    gh.prs[0]["body"] = "Closes #207\n\n## Tests\nuv run --extra dev pytest -q"
    # The reviewed verdict pinned this diff, so an unchanged diff is a no-op.
    gh.diffs[100] = _DIFF
    state = load_state(paths.state_file)
    state["prs"]["100"]["reviewed_patch_id"] = _calculate_patch_id(_DIFF)
    save_state(paths.state_file, state)
    return config, paths, gh


def _bed_lint(tmp_path: Path, *, lint_red: bool, live_head: str = "abc123"):
    return _bed(
        tmp_path,
        checks=[
            {"name": _LINT, "state": "FAILURE" if lint_red else "SUCCESS", "link": _LINT_LINK}
        ],
        live_head=live_head,
    )


def _terminal_exit_zero(tmp_path: Path) -> None:
    sessions = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    # The record must post-date this dispatch (a stale one is ignored).
    now = datetime.now(UTC)
    write_worker_terminal_status(
        sessions / "issue-207.claude-code.terminal.json",
        pid=99999,
        exit_code=0,
        started_at=(now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ended_at=(now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_seconds=300.0,
    )


_UNSET: Any = object()


def _sweep(
    tmp_path: Path,
    paths,
    config,
    gh,
    app: OrchestratorApp,
    review: Any = _UNSET,
    record_review: Any = _UNSET,
) -> None:

    from charlie_work.workflow import _detect_and_handle_orphaned_workers

    sessions = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    with host_probe(alive=False):
        _detect_and_handle_orphaned_workers(
            sessions,
            paths.state_file,
            config,
            gh,
            review_callback=app.review if review is _UNSET else review,
            record_review_callback=app.record_review if record_review is _UNSET else record_review,
            enrich_checks_callback=app._enrich_checks_infra_blocked,
            write_gate=_wg(paths.state_file),
        )


def _redispatch(paths) -> None:
    """Put issue 207 back to a dead ``dispatched`` worker, as a second no-op would."""
    state = load_state(paths.state_file)
    state["issues"]["207"] = {
        **state["issues"]["207"],
        "status": "dispatched",
        "dispatched_at": "2024-01-01T00:00:00Z",
    }
    save_state(paths.state_file, state)


def _decision(paths) -> dict[str, Any]:
    return json.loads((paths.prs / "pr-100" / "review-decision.json").read_text(encoding="utf-8"))


def _assert_ci_rework_brief(paths, tmp_path: Path, app: OrchestratorApp, gh) -> None:
    changes = _decision(paths)["required_changes"]
    assert any("failing step(s): Run ruff check" in c for c in changes)
    assert any("E501 line too long" in c for c in changes)
    brief = app._write_rework_prompt(gh.prs[0], 207, "note").read_text(encoding="utf-8")
    assert "Run ruff check" in brief
    assert "E501 line too long" in brief


def test_2005_shape_merge_only_push_lint_red_reaches_rework_naming_failing_step(
    tmp_path: Path,
) -> None:
    """Regression: exit 0, only a merge commit pushed (head moved, diff unchanged),
    Lint red. The janitor refuses the head-change review; the issue must still reach
    a rework whose prompt contains the failing step."""
    config, paths, gh = _bed_lint(tmp_path, lint_red=True, live_head="def456")
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "rework_requested"
    assert state["issues"]["207"]["no_op_handled_head"] == "def456"
    assert _decision(paths)["decision"] == "request_changes"
    _assert_ci_rework_brief(paths, tmp_path, app, gh)
    kinds = [e["kind"] for e in state["events"]]
    assert "rework_no_op_ci_rework_requested" in kinds


def test_clean_exit_red_ci_records_ci_rework(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=True)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    state = load_state(paths.state_file)
    assert state["issues"]["207"]["status"] == "rework_requested"
    _assert_ci_rework_brief(paths, tmp_path, app, gh)


def test_repeat_no_op_on_same_head_escalates_after_ci_rework(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=True)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)
    assert load_state(paths.state_file)["issues"]["207"]["status"] == "rework_requested"

    _redispatch(paths)  # the CI-rework worker no-oped on the same head
    _sweep(tmp_path, paths, config, gh, app)

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert (207, config.labels.human_needed) in gh.labels_added


def test_ci_green_with_rebuttal_returns_to_review_once_per_head(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": False, "head_sha": "abc123", "pr_comment": "Head already fixes it."},
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    calls: list[int] = []

    def review(pr_number: int) -> CommandResult:
        calls.append(pr_number)
        return CommandResult(True, "packet", {"pr": pr_number})

    _sweep(tmp_path, paths, config, gh, app, review=review)
    entry = load_state(paths.state_file)["issues"]["207"]
    assert calls == [100]
    assert entry["status"] == "reviewing"

    _redispatch(paths)  # second no-op on the same head: no second rebuttal review
    _sweep(tmp_path, paths, config, gh, app, review=review)
    entry = load_state(paths.state_file)["issues"]["207"]
    assert calls == [100]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert entry["no_op_worker_comment"] == "Head already fixes it."


def test_ci_green_rebuttal_review_refused_escalates_with_worker_comment(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": False, "head_sha": "abc123", "pr_comment": "Nothing left to fix."},
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(
        tmp_path,
        paths,
        config,
        gh,
        app,
        review=lambda n: CommandResult(False, "janitor refused", {"pr": n}),
    )
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    escalated = [e for e in state["events"] if e["kind"] == "rework_no_op_escalated"]
    assert escalated[0]["payload"]["worker_comment"] == "Nothing left to fix."


def test_stale_prior_round_outcome_is_not_quoted_in_rebuttal_or_escalation(
    tmp_path: Path,
) -> None:
    """#2102: an outcome written before this dispatch is an earlier round's."""
    config, paths, gh = _bed_lint(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": True, "head_sha": "abc123", "pr_comment": "Round-2 rework pushed."},
        mtime=datetime.now(UTC) - timedelta(hours=3),  # dispatched_at is 1h ago
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    calls: list[int] = []

    def review(pr_number: int) -> CommandResult:
        calls.append(pr_number)
        return CommandResult(True, "packet", {"pr": pr_number})

    _sweep(tmp_path, paths, config, gh, app, review=review)
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert calls == []  # no comment -> no rebuttal review
    assert entry["status"] == "escalated"
    assert entry["no_op_worker_comment"] is None
    escalated = [e for e in state["events"] if e["kind"] == "rework_no_op_escalated"]
    assert escalated[0]["payload"]["worker_comment"] is None
    assert "Round-2" not in json.dumps(state["events"])
    # Reported (deduped per source/written_at) rather than silently dropped.
    assert any(k.startswith("worktree:") for k in entry["stale_evidence_reported"])


def test_fresh_terminal_record_embedding_leftover_outcome_is_not_quoted(
    tmp_path: Path,
) -> None:
    """#2102: a fresh ``ended_at`` must not launder a reused worktree's leftover.

    The watcher copies the outcome file into the terminal record at exit; with a
    reused worktree that file is a prior round's, and only the record's
    ``worker_outcome_written_at`` says so.
    """
    config, paths, gh = _bed_lint(tmp_path, lint_red=False)
    sessions = tmp_path / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    write_worker_terminal_status(
        sessions / "issue-207.claude-code.terminal.json",
        pid=99999,
        exit_code=0,
        started_at=(now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ended_at=(now - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        duration_seconds=300.0,
        worker_outcome={
            "push_succeeded": True,
            "head_sha": "abc123",
            "pr_comment": "Round-2 rework pushed.",
        },
        worker_outcome_written_at=(now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    calls: list[int] = []

    def review(pr_number: int) -> CommandResult:
        calls.append(pr_number)
        return CommandResult(True, "packet", {"pr": pr_number})

    _sweep(tmp_path, paths, config, gh, app, review=review)
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert calls == []
    assert entry["no_op_worker_comment"] is None
    assert "Round-2" not in json.dumps(state["events"])
    assert any(k.startswith("terminal:") for k in entry["stale_evidence_reported"])


def test_fresh_outcome_after_dispatch_is_still_quoted(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    _write_outcome(
        paths,
        tmp_path,
        {"push_succeeded": False, "head_sha": "abc123", "pr_comment": "Fresh rebuttal."},
        mtime=datetime.now(UTC) - timedelta(minutes=2),
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(
        tmp_path,
        paths,
        config,
        gh,
        app,
        review=lambda n: CommandResult(False, "janitor refused", {"pr": n}),
    )
    state = load_state(paths.state_file)
    escalated = [e for e in state["events"] if e["kind"] == "rework_no_op_escalated"]
    assert escalated[0]["payload"]["worker_comment"] == "Fresh rebuttal."


def test_ci_green_without_rebuttal_escalates_human_needed(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=False)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert (207, config.labels.human_needed) in gh.labels_added


def test_unknown_ci_state_defers_instead_of_escalating(tmp_path: Path) -> None:
    config, paths, gh = _bed_lint(tmp_path, lint_red=True)
    gh.checks_unavailable = True
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    _assert_deferred(load_state(paths.state_file)["issues"]["207"], head="abc123")


_UNSETTLED_CHECKS = (
    # Pending: a required check is still running -- red vs green undecided.
    [{"name": _LINT, "state": "PENDING", "link": _LINT_LINK}],
    # Missing: the required check never reported at all.
    [],
    # Infra-blocked: a fleet-wide condition (#1383), never a code failure.
    [{"name": _LINT, "state": "INFRA_BLOCKED", "link": _LINT_LINK}],
    # Infra-failed: per-PR CANCELLED/TIMED_OUT (#841) -- also not a code verdict.
    [{"name": _LINT, "state": "CANCELLED", "link": _LINT_LINK}],
)


def _assert_deferred(entry: dict[str, Any], head: str) -> None:
    assert entry["status"] == "dispatched"
    assert entry["no_op_deferred_head"] == head
    assert entry["orphan_drift_at"]  # the #654 backstop is armed and bounds the deferral


@pytest.mark.parametrize(
    "checks",
    _UNSETTLED_CHECKS,
    ids=["pending", "missing", "infra_blocked", "infra_failed"],
)
def test_clean_exit_unsettled_ci_defers_then_disposes_when_settled(
    tmp_path: Path, checks: list[dict[str, Any]]
) -> None:
    """Clean-exit no-op + unsettled required CI -> defer, not escalate.

    The drain must not guess red-vs-green: it leaves the issue ``dispatched``
    under the #654 backstop and retries on a later pass -- proven here by
    flipping CI to settled-green and re-sweeping to the ordinary escalation.
    """
    config, paths, gh = _bed(tmp_path, checks=checks)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    state = load_state(paths.state_file)
    _assert_deferred(state["issues"]["207"], head="abc123")
    assert "rework_no_op_escalated" not in [e["kind"] for e in state["events"]]
    assert (207, config.labels.human_needed) not in gh.labels_added
    assert any(e["kind"] == "rework_no_op_deferred" for e in state["events"])

    # Second pass while still unsettled: deferred again, event dedup'd per head.
    _sweep(tmp_path, paths, config, gh, app)
    state = load_state(paths.state_file)
    _assert_deferred(state["issues"]["207"], head="abc123")
    assert len([e for e in state["events"] if e["kind"] == "rework_no_op_deferred"]) == 1

    # CI settles green: the deferred finding still reaches a disposition.
    gh.checks = [{"name": _LINT, "state": "SUCCESS", "link": _LINT_LINK}]
    _sweep(tmp_path, paths, config, gh, app)
    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"


@pytest.mark.parametrize(
    "checks",
    _UNSETTLED_CHECKS,
    ids=["pending", "missing", "infra_blocked", "infra_failed"],
)
def test_head_change_unsettled_ci_defers_instead_of_escalating(
    tmp_path: Path, checks: list[dict[str, Any]]
) -> None:
    """Head-change no-op route + unsettled required CI -> defer.

    The route arrives via the review drain's ``is_no_op_rework``-flagged
    refusal (the flag is stubbed here; real review() propagation is pinned by
    the #2005-shape test). ``no_op_deferred_head`` distinguishes "drain ran
    and deferred" from "route never collected".
    """
    config, paths, gh = _bed(tmp_path, checks=checks, live_head="def456")
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(
        tmp_path,
        paths,
        config,
        gh,
        app,
        review=lambda n: CommandResult(
            False, "janitor gate blocked PR #100", {"is_no_op_rework": True}
        ),
    )

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    _assert_deferred(entry, head="def456")
    assert entry["orphan_drift_fingerprint"]  # the head-change drift fingerprint armed it
    assert "rework_no_op_escalated" not in [e["kind"] for e in state["events"]]
    assert (207, config.labels.human_needed) not in gh.labels_added


def test_head_change_deferred_no_op_retries_to_ci_rework_when_ci_settles_red(
    tmp_path: Path,
) -> None:
    """A deferred head-change no-op must be retried, not fingerprint-wedged.

    The review drain stamps ``orphan_drift_fingerprint`` when it hands the
    janitor-refused head change to the no-op drain, so without the sweep's
    ``no_op_deferred_head`` re-collection the next pass short-circuits on the
    fingerprint and the deferral is never retried -- a merge-only push
    detected while CI is still pending (the #2005 shape) falls to the
    generic #654 reap instead of the CI-rework lane. The flag is stubbed
    (real ``review()`` propagation is pinned by the #2005-shape test); the
    call count pins that review() is not re-run while the deferral stands.
    """
    config, paths, gh = _bed(
        tmp_path,
        checks=[{"name": _LINT, "state": "PENDING", "link": _LINT_LINK}],
        live_head="def456",
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    review_calls: list[int] = []

    def counted_review(pr_number: int) -> CommandResult:
        review_calls.append(pr_number)
        return CommandResult(False, "janitor gate blocked PR #100", {"is_no_op_rework": True})

    _sweep(tmp_path, paths, config, gh, app, review=counted_review)
    state = load_state(paths.state_file)
    _assert_deferred(state["issues"]["207"], head="def456")
    assert review_calls == [100]

    # Still pending: the route is re-collected and re-deferred, review() is
    # not re-run, and the deferred event stays once-per-head.
    _sweep(tmp_path, paths, config, gh, app, review=counted_review)
    state = load_state(paths.state_file)
    _assert_deferred(state["issues"]["207"], head="def456")
    assert review_calls == [100]
    assert len([e for e in state["events"] if e["kind"] == "rework_no_op_deferred"]) == 1

    # CI settles red: the re-collected route reaches the CI-rework lane.
    gh.checks = [{"name": _LINT, "state": "FAILURE", "link": _LINT_LINK}]
    _sweep(tmp_path, paths, config, gh, app, review=counted_review)
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "rework_requested"
    assert entry["no_op_handled_head"] == "def456"
    assert entry["no_op_deferred_head"] is None
    assert review_calls == [100]
    _assert_ci_rework_brief(paths, tmp_path, app, gh)
    assert "rework_no_op_ci_rework_requested" in [e["kind"] for e in state["events"]]


def test_head_change_deferred_no_op_retries_to_escalation_when_ci_settles_green(
    tmp_path: Path,
) -> None:
    """Green variant: the retried head-change deferral escalates rework_no_op.

    Same deferral as the red variant, but CI settles green and the worker
    left no rebuttal -- the re-collected route must reach the ordinary
    ``rework_no_op`` escalation rather than resting dispatched.
    """
    config, paths, gh = _bed(
        tmp_path,
        checks=[{"name": _LINT, "state": "PENDING", "link": _LINT_LINK}],
        live_head="def456",
    )
    app = OrchestratorApp(tmp_path, paths, config, gh)
    review_calls: list[int] = []

    def counted_review(pr_number: int) -> CommandResult:
        review_calls.append(pr_number)
        return CommandResult(False, "janitor gate blocked PR #100", {"is_no_op_rework": True})

    _sweep(tmp_path, paths, config, gh, app, review=counted_review)
    _assert_deferred(load_state(paths.state_file)["issues"]["207"], head="def456")

    # Still pending: re-collected and re-deferred without a second review().
    _sweep(tmp_path, paths, config, gh, app, review=counted_review)
    state = load_state(paths.state_file)
    _assert_deferred(state["issues"]["207"], head="def456")
    assert review_calls == [100]
    assert len([e for e in state["events"] if e["kind"] == "rework_no_op_deferred"]) == 1

    # CI settles green with no rebuttal: the deferred finding escalates.
    gh.checks = [{"name": _LINT, "state": "SUCCESS", "link": _LINT_LINK}]
    _sweep(tmp_path, paths, config, gh, app, review=counted_review)
    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert (207, config.labels.human_needed) in gh.labels_added
    assert review_calls == [100]
    assert len([e for e in state["events"] if e["kind"] == "rework_no_op_deferred"]) == 1


def test_clean_exit_ci_red_record_review_failure_escalates(tmp_path: Path) -> None:
    """CI-red tier degraded: record_review returning not-ok falls through to
    the rework_no_op escalation -- the issue must not rest dispatched."""
    config, paths, gh = _bed_lint(tmp_path, lint_red=True)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(
        tmp_path,
        paths,
        config,
        gh,
        app,
        record_review=lambda *a, **kw: CommandResult(False, "record refused", {}),
    )

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"
    assert (207, config.labels.human_needed) in gh.labels_added


def test_clean_exit_ci_red_without_record_review_callback_escalates(tmp_path: Path) -> None:
    """CI-red tier unavailable: a None record_review_callback (a caller that
    never wired it) falls through to escalation rather than stranding."""
    config, paths, gh = _bed_lint(tmp_path, lint_red=True)
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app, record_review=None)

    entry = load_state(paths.state_file)["issues"]["207"]
    assert entry["status"] == "escalated"
    assert entry["escalation_reason"] == "rework_no_op"


def test_approved_rework_clean_exit_no_op_reaches_drain(tmp_path: Path) -> None:
    """approved + PR-status rework_requested + head unchanged + exit 0.

    The post-approval rework lane's clean-exit no-op collects the same route
    as the request_changes branch; with live CI red the drain issues the
    CI-failure rework verdict -- it must not wedge dispatched nor auto-reset.
    """
    config, paths, gh = _bed(
        tmp_path,
        checks=[{"name": _LINT, "state": "FAILURE", "link": _LINT_LINK}],
        decision="approved",
        pr_state_status="rework_requested",
    )
    _terminal_exit_zero(tmp_path)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    _sweep(tmp_path, paths, config, gh, app)

    state = load_state(paths.state_file)
    entry = state["issues"]["207"]
    assert entry["status"] == "rework_requested"
    assert entry["no_op_handled_head"] == "abc123"
    assert "rework_no_op_ci_rework_requested" in [e["kind"] for e in state["events"]]


def test_maintenance_lane_wires_no_op_drain_callbacks(tmp_path: Path) -> None:
    """Wiring: run_deadline_guarded_maintenance must forward
    ``record_review_callback`` and ``enrich_checks_callback`` into the orphan
    sweep -- without them the no-op drain's CI-red tier is dead code."""
    from unittest.mock import patch

    from charlie_work.pass_deadline import PassDeadline, run_deadline_guarded_maintenance

    config, paths, gh = _bed_lint(tmp_path, lint_red=True)
    app = OrchestratorApp(tmp_path, paths, config, gh)
    captured: dict[str, Any] = {}

    def fake_detect(*args: Any, **kwargs: Any) -> None:
        captured.update(kwargs)

    with (
        patch(
            "charlie_work.workflow._detect_and_handle_orphaned_workers",
            side_effect=fake_detect,
        ),
        host_probe(alive=True),
    ):
        run_deadline_guarded_maintenance(PassDeadline(None, CommandResult), app)

    assert captured.get("review_callback") == app.review
    assert captured.get("record_review_callback") == app.record_review
    assert captured.get("enrich_checks_callback") == app._enrich_checks_infra_blocked
