"""Tests for PR #550's three escalation call sites in workflow.py:

* ``dispatch_reviews()``: a PR whose ``review_dispatch_attempt_count`` has
  reached ``max_review_dispatch_attempts`` must escalate through the same
  ``transition()`` helper every other escalation site uses (pr-lifecycle.md
  Finding 3: this call site used to skip the label edge entirely, leaving
  PRs escalated in state.json but invisible on GitHub -- the same class of
  bug fixed for janitor-rework escalation in test_fix_janitor_routing.py).
* ``record_review()``: escalation must be terminal for verdict recording
  too, mirroring ``review()``'s guard -- a late-arriving verdict for an
  already-escalated PR/issue must not silently re-enter the pipeline.
* ``dispatch()``/``_dispatch_impl()``: a fresh dispatch failure whose
  ``SessionDispatchResult.failure_kind`` is a confirmed-deterministic member
  of ``DETERMINISTIC_ESCALATION_FAILURE_KINDS`` must escalate on the FIRST
  occurrence, instead of burning through ``max_auto_redispatch`` retries
  that cannot possibly fix a deterministic failure.

Reuses ``FakeGitHub`` from test_charlie_work.py (PR #456 <-> issue #123).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from charlie_work.adapters import SessionDispatchResult
from charlie_work.config import DevinConfig, OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from test_charlie_work import FakeGitHub


def _events(state, kind: str) -> list[dict]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


# --- dispatch_reviews(): attempt-cap escalation ---


def _write_review_packet(paths, pr_number: int, head_sha: str) -> None:
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        f'{{"number": {pr_number}, "headRefOid": "{head_sha}"}}', encoding="utf-8"
    )
    (pr_dir / "review-prompt.md").write_text("review prompt", encoding="utf-8")


def _seed_attempt_count(paths, pr_number: int, issue_number: int, count: int) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "review_dispatch_attempt_count": count,
        }
        save_state(paths.state_file, state)


def test_dispatch_reviews_attempt_cap_escalates_with_label(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")  # matches FakeGitHub's PR #456 headRefOid
    _seed_attempt_count(paths, 456, 123, 2)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert state["issues"]["123"]["status"] == "escalated"
    assert (123, config.labels.human_needed) in fake_gh.labels_added

    escalated_events = _events(state, "review_dispatch_escalated")
    assert len(escalated_events) == 1
    assert escalated_events[0]["payload"]["attempt_count"] == 2
    assert escalated_events[0]["payload"]["reason"] == "max_review_dispatch_attempts_exceeded"


def _seed_dispatched_at_cap(paths, pr_number: int, issue_number: int, count: int) -> None:
    from datetime import UTC, datetime

    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "review_dispatch_attempt_count": count,
            "review_dispatch_status": "review_dispatch_dispatched",
            # Fresh claim: the reviewer launched moments ago and is mid-review.
            "review_dispatched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "reviewer_pid": 424242,
            "reviewer_process_start_time": 1.0,
        }
        save_state(paths.state_file, state)


def test_attempt_cap_never_escalates_over_live_reviewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #573: a PR at the attempt cap whose dispatched reviewer is ALIVE
    must not be escalated out from under it — the in-flight verdict would be
    orphaned (the reaper only records verdicts for dispatched claims).
    Observed live on PR #540 (2026-07-25 02:21Z): the guard nulled the pid
    and marked the claim failed while the reviewer's log was still growing."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_dispatched_at_cap(paths, 456, 123, 2)
    monkeypatch.setattr("charlie_work.workflow._reviewer_pid_alive", lambda *_: True)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"].get("status") != "escalated"
    assert state["prs"]["456"]["review_dispatch_status"] == "review_dispatch_dispatched"
    assert state["prs"]["456"]["reviewer_pid"] == 424242
    assert _events(state, "review_dispatch_escalated") == []
    assert (123, config.labels.human_needed) not in fake_gh.labels_added


def test_attempt_cap_still_escalates_dead_dispatched_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cap keeps escalating when the dispatched reviewer is dead — the
    liveness guard must not shield corpses."""
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_dispatched_at_cap(paths, 456, 123, 2)
    monkeypatch.setattr("charlie_work.workflow._reviewer_pid_alive", lambda *_: False)

    result = app.dispatch_reviews()

    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    assert len(_events(state, "review_dispatch_escalated")) == 1


def test_dispatch_reviews_attempt_cap_escalation_label_failure_records_error(
    tmp_path: Path,
) -> None:
    class LabelFailingGitHub(FakeGitHub):
        def add_issue_label(self, number: int, label: str) -> bool:
            self.labels_added.append((number, label))
            return False

    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = LabelFailingGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_attempt_count(paths, 456, 123, 2)

    app.dispatch_reviews()

    state = load_state(paths.state_file)
    assert state["prs"]["456"]["status"] == "escalated"
    label_error = state["issues"]["123"].get("label_error")
    assert label_error is not None
    assert label_error["edge"] == "escalated"
    assert label_error["outcome"] == "partial_failure"
    # state.json round-trips through JSON, so tuples become lists.
    assert label_error["add_failures"] == [[123, config.labels.human_needed]]


class _ABSENT:
    """Sentinel for "key absent" in _seed_escalated_pr."""


def _seed_escalated_pr(
    paths, pr_number: int, issue_number: int, *, label_error: object = _ABSENT
) -> None:
    """Seed an already-escalated PR (with a review packet) in state.

    ``label_error`` controls the issue entry's label_error field:
    - ``_ABSENT`` (default): the key is absent (pre-#556 escalation that
      never attempted the label edge).
    - a ``dict``: a prior transition() failure recorded on the issue.
    - ``None``: the edge was verified OK on a prior pass.
    """
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": "escalated",
            "review_dispatch_attempt_count": 3,
            "review_dispatch_status": "review_dispatch_failed",
        }
        issue_entry: dict[str, Any] = {
            "number": issue_number,
            "status": "escalated",
        }
        if label_error is not _ABSENT:
            issue_entry["label_error"] = label_error
        state["issues"][str(issue_number)] = issue_entry
        save_state(paths.state_file, state)


def test_dispatch_reviews_self_heals_escalated_label_never_attempted(tmp_path: Path) -> None:
    """Issue #586: a PR escalated by a path that predated the label edge
    (label_error key absent, human_needed missing on GitHub) must get the
    label re-applied on the next dispatch_reviews pass -- not sit invisibly
    escalated forever. This is the "21 jc issues" backfill case.
    """
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_escalated_pr(paths, 456, 123)  # label_error absent

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["escalated_skipped"] == [456]
    # The human-needed label was re-applied.
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    state = load_state(paths.state_file)
    # label_error is now None (verified/applied), not absent.
    assert state["issues"]["123"]["label_error"] is None
    # A repair event was recorded.
    repair_events = _events(state, "escalated_label_repaired")
    assert len(repair_events) == 1
    assert 123 in repair_events[0]["payload"]["issue_numbers"]


def test_dispatch_reviews_self_heals_escalated_label_error_retry(tmp_path: Path) -> None:
    """Issue #586: a PR whose escalation label edge failed on the first
    attempt (label_error is a dict) must be retried every pass until
    transition() succeeds -- the edge is no longer fire-and-forget.
    """
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_escalated_pr(
        paths,
        456,
        123,
        label_error={"edge": "escalated", "outcome": "partial_failure"},
    )

    result = app.dispatch_reviews()

    assert result.ok is True
    assert (123, config.labels.human_needed) in fake_gh.labels_added
    state = load_state(paths.state_file)
    # The prior label_error was cleared (set to None) on successful repair.
    assert state["issues"]["123"]["label_error"] is None


def test_dispatch_reviews_skips_repair_when_label_already_verified(
    tmp_path: Path,
) -> None:
    """Issue #586: once the escalated label edge has been verified (label_error
    is None), subsequent passes must skip the GitHub fetch entirely -- no
    issue_view, no transition. Steady-state cost is zero.
    """
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class SpyGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issue_view_calls: list[int] = []

        def issue_view(self, number: int):
            self.issue_view_calls.append(number)
            return super().issue_view(number)

    fake_gh = SpyGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_escalated_pr(paths, 456, 123, label_error=None)

    result = app.dispatch_reviews()

    assert result.ok is True
    # No GitHub fetch, no label mutation.
    assert fake_gh.issue_view_calls == []
    assert (123, config.labels.human_needed) not in fake_gh.labels_added


def test_dispatch_reviews_repair_skips_when_status_no_longer_escalated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (review finding on PR #670): if a concurrent unescalate()
    frees the issue between subject collection and the GitHub call, the repair
    must NOT re-apply the agent:human-needed label -- it would silently undo the
    unescalate. ``_repair_escalated_labels`` re-checks each subject's current
    status in its own state_lock read immediately before touching GitHub, and
    skips when neither the PR nor the issue is still "escalated".

    How the race is simulated, and why this way (issue #1088 rewrite): the race
    is injected as a side effect of ``_collect_escalated_label_subjects`` -- the
    unescalate lands on disk the instant after the batch read produced the
    subject list. That is exactly the window the per-item re-check exists to
    close, and it is expressed with no coupling to *how many* times
    ``load_state`` happens to be called.

    The previous version of this test counted ``load_state`` calls and poisoned
    "the 2nd read that sees escalated", returning stale-escalated state after
    having written unescalated state to the file. That was calibrated to the old
    call sequence, where the repair set was collected by ``dispatch_reviews``'
    escalation gate. It also modelled a torn read -- a read returning a value
    already false when it returned -- which ``state_lock`` plus atomic writes
    make impossible. Both reasons are why the mechanism changed here while the
    asserted property did not.
    """
    import json as _json

    from charlie_work import workflow as wf
    from charlie_work.state import PASSIVE_OPEN_STATUS

    # review_dispatch DISABLED -- the configuration both deployed fleets run, and
    # the one #1088 is about. It also isolates what this test asserts: with
    # dispatch off, `_repair_escalated_labels` is the ONLY thing in this call
    # that can add a label, so "no label was added" is a statement about the
    # repair rather than about the pass as a whole. With dispatch on, freeing the
    # PR hands it to candidate selection and the attempt-cap branch escalates it
    # FRESH -- real behaviour, but a different mechanism than this test asserts.
    # The positive control that the sweep DOES act when it should lives in
    # test_escalated_label_repair_runs_with_review_dispatch_disabled.
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=False, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class SpyGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issue_view_calls: list[int] = []

        def issue_view(self, number: int):
            self.issue_view_calls.append(number)
            return super().issue_view(number)

    fake_gh = SpyGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_escalated_pr(paths, 456, 123)  # label_error absent, status escalated

    original_collect = wf._collect_escalated_label_subjects
    raced = [0]

    def racing_collect(state):
        subjects = original_collect(state)
        # The instant after collection, a concurrent unescalate() wins the lock
        # and frees the issue. Use the literals unescalate actually writes: with
        # the PR live and open both records land on PASSIVE_OPEN_STATUS, and
        # label_error is POPPED (it is a member of
        # _UNESCALATE_ISSUE_RESET_FIELDS) -- which is what puts the freed issue
        # in the absent-key "never attempted" arm that the status re-check has
        # to override.
        if subjects and not raced[0]:
            raced[0] = 1
            on_disk = _json.loads(paths.state_file.read_text())
            if "456" in on_disk.get("prs", {}):
                on_disk["prs"]["456"]["status"] = PASSIVE_OPEN_STATUS
                # unescalate() also clears the dispatch counters --
                # "review_dispatch_attempt_count" is the FIRST member of
                # _UNESCALATE_PR_RESET_FIELDS. Popping them here is fidelity,
                # not convenience: _seed_escalated_pr seeds a count of 3 against
                # a cap of 2, so a freed-but-uncleared PR immediately re-hits the
                # attempt-cap branch and is escalated FRESH. That is a different
                # mechanism from the repair undoing an unescalate, and leaving it
                # in would have this test fail for a reason it does not assert.
                for field in ("review_dispatch_attempt_count", "review_dispatch_status"):
                    on_disk["prs"]["456"].pop(field, None)
            if "123" in on_disk.get("issues", {}):
                on_disk["issues"]["123"]["status"] = PASSIVE_OPEN_STATUS
                on_disk["issues"]["123"].pop("label_error", None)
            paths.state_file.write_text(_json.dumps(on_disk))
        return subjects

    monkeypatch.setattr(wf, "_collect_escalated_label_subjects", racing_collect)

    result = app.dispatch_reviews()

    # Control: the race actually fired. Without this the assertions below would
    # pass just as well if `racing_collect` never ran -- "no label applied" is
    # equally consistent with "the guard worked" and "nothing was ever a
    # subject", and only one of those is the property under test.
    assert raced[0] == 1

    assert result.ok is True
    # The human-needed label was NOT re-applied -- the race guard skipped
    # repair because neither the PR nor the issue was still "escalated".
    assert (123, config.labels.human_needed) not in fake_gh.labels_added
    assert fake_gh.issue_view_calls == []
    state = load_state(paths.state_file)
    # No repair event was recorded.
    assert _events(state, "escalated_label_repaired") == []


def test_dispatch_reviews_dry_run_no_mutation_for_escalated_pr(tmp_path: Path) -> None:
    """Regression (review finding on PR #670): ``dispatch_reviews(dry_run=True)``
    must not perform live GitHub label mutations or state.json writes for an
    already-escalated PR with an unverified label (label_error absent). The
    self-heal repair loop and the fresh-escalation label loop are both gated
    behind ``if not self.dry_run``.
    """
    config = OrchestratorConfig(
        review_dispatch=ReviewDispatchConfig(enabled=True, max_review_dispatch_attempts=2)
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class SpyGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issue_view_calls: list[int] = []

        def issue_view(self, number: int):
            self.issue_view_calls.append(number)
            return super().issue_view(number)

    fake_gh = SpyGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)
    _write_review_packet(paths, 456, "sha-abc123")
    _seed_escalated_pr(paths, 456, 123)  # label_error absent, status escalated

    state_before = load_state(paths.state_file)

    result = app.dispatch_reviews()

    assert result.ok is True
    assert result.data["escalated_skipped"] == [456]
    # No GitHub label mutation.
    assert (123, config.labels.human_needed) not in fake_gh.labels_added
    # No GitHub fetch (issue_view) for the repair loop.
    assert fake_gh.issue_view_calls == []
    state_after = load_state(paths.state_file)
    # No state mutation: the issue entry is unchanged (no label_error written,
    # no repair event appended).
    assert state_after["issues"]["123"] == state_before["issues"]["123"]
    assert _events(state_after, "escalated_label_repaired") == []


# --- record_review(): escalated guard ---


def test_record_review_escalated_pr_blocks_and_writes_no_decision(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {"number": 456, "issue_number": 123, "status": "escalated"}
        save_state(app.paths.state_file, state)
    before = load_state(app.paths.state_file)

    result = app.record_review(456, "approved", summary="lgtm")

    assert result.ok is False
    assert "unescalate" in result.message
    assert result.data["escalated"] is True
    decision_path = app.paths.prs / "pr-456" / "review-decision.json"
    assert not decision_path.exists()
    after = load_state(app.paths.state_file)
    assert after["prs"] == before["prs"]
    assert after["issues"] == before["issues"]
    assert after["events"] == before["events"]


def test_record_review_escalated_linked_issue_blocks_even_if_pr_state_is_clean(
    tmp_path: Path,
) -> None:
    """Escalation can be recorded on the ISSUE side only (e.g. dispatch-side
    escalation of the issue while the PR record itself carries no status).
    record_review must still refuse -- the guard checks both sides."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"]["123"] = {"number": 123, "status": "escalated"}
        save_state(app.paths.state_file, state)

    result = app.record_review(456, "approved", summary="lgtm")

    assert result.ok is False
    assert "unescalate" in result.message
    decision_path = app.paths.prs / "pr-456" / "review-decision.json"
    assert not decision_path.exists()


# --- dispatch(): deterministic failure_kind escalates immediately ---


def _closed_pr_app(tmp_path: Path) -> tuple[OrchestratorApp, FakeGitHub]:
    """A dispatchable ready issue #123, with the default fixture PR #456
    closed so it doesn't trip the open-PR exclusion (mirrors
    test_dispatch_failed_retries_are_capped_and_escalate in test_charlie_work.py)."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            adapter="command",
            dispatch_command=(sys.executable, "-c", "import sys; sys.exit(1)"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs[0]["state"] = "CLOSED"
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def _fake_dispatch_sessions_factory(failure_kind: str | None):
    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=False,
                error="launch failed",
                failure_kind=failure_kind,
            )
            for request in requests
        ]

    return fake_dispatch_sessions


def test_dispatch_deterministic_failure_kind_escalates_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, fake_gh = _closed_pr_app(tmp_path)
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        _fake_dispatch_sessions_factory("worktree_unsafe"),
    )

    result = app.dispatch(limit=1)

    assert result.ok is False
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["escalation_reason"] == "worktree_unsafe"
    # Escalated on the FIRST failure -- not after burning max_auto_redispatch.
    assert len(state["issues"]["123"]["dispatch_failed_at"]) == 1
    assert (123, app.config.labels.human_needed) in fake_gh.labels_added


def test_dispatch_non_deterministic_failure_kind_still_uses_redispatch_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, fake_gh = _closed_pr_app(tmp_path)
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        _fake_dispatch_sessions_factory(None),
    )

    result = app.dispatch(limit=1)

    assert result.ok is False
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == "dispatch_failed"
    assert "escalation_reason" not in state["issues"]["123"]
    assert (123, app.config.labels.human_needed) not in fake_gh.labels_added


# --- Issues #837 / #779: collapsed dispatch-outcome/escalation branch pin ---


def _seed_dispatch_failed_at(paths, issue_number: int, attempts: list[str]) -> None:
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"][str(issue_number)] = {
            **state["issues"].get(str(issue_number), {}),
            "dispatch_failed_at": attempts,
            # Seed a stale escalation too (issues #837/#779): every non-
            # terminal outcome arm calls clear_escalation(entry), including
            # the transient-failure-under-cap arm. Without a prior value here
            # "escalation_reason not in entry" is vacuously true for a fresh
            # prev_entry, exactly like the dispatch_failed_at case above.
            "escalation_reason": "stale_prior_reason",
            "reason_class": "mechanical",
        }
        save_state(paths.state_file, state)


def _fake_dispatch_result_factory(
    *,
    ok: bool,
    failure_kind: str | None,
    pid: int | None,
    process_start_time: float | None,
):
    def fake_dispatch_sessions(_repo_root, _manifest, _results, _settings, requests):
        return [
            SessionDispatchResult(
                issue_number=request.issue_number,
                issue_title=request.issue_title,
                prompt_path=str(request.prompt_path),
                branch_name=request.branch_name,
                adapter="command",
                ok=ok,
                error=None if ok else "launch failed",
                failure_kind=failure_kind,
                pid=pid,
                process_start_time=process_start_time,
            )
            for request in requests
        ]

    return fake_dispatch_sessions


@pytest.mark.parametrize(
    (
        "scenario",
        "ok",
        "failure_kind",
        "pid_alive",
        "expect_status",
        "expect_dispatch_failed_at_len",
        "expect_escalation_reason",
    ),
    [
        ("ok", True, None, None, "dispatched", None, None),
        (
            "live_worker",
            False,
            "live_worker_redispatch_averted",
            True,
            "dispatched",
            None,
            None,
        ),
        (
            "phantom_live_worker",
            False,
            "live_worker_redispatch_averted",
            False,
            "dispatch_failed",
            None,
            None,
        ),
        ("transient_failure_under_cap", False, None, None, "dispatch_failed", 2, None),
        (
            "deterministic_failure_escalates_first_try",
            False,
            "worktree_unsafe",
            None,
            "escalated",
            2,
            "worktree_unsafe",
        ),
    ],
)
def test_dispatch_outcome_field_sets_pin_the_collapsed_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    ok: bool,
    failure_kind: str | None,
    pid_alive: bool | None,
    expect_status: str,
    expect_dispatch_failed_at_len: int | None,
    expect_escalation_reason: str | None,
) -> None:
    """Issues #837 / #779 regression pin.

    ``_dispatch_impl`` used to decide the dispatch outcome (status /
    dispatched_at) in one if/elif chain and its escalation bookkeeping
    (dispatch_failed_at / escalation_reason / reason_class) in a second,
    textually separate chain keyed on the same four predicates (ok /
    is_live_worker / is_phantom_live_worker / else) -- safe only because the
    two enumerations happened to agree, which is exactly what let
    all_attempts / failed_result / terminal_failure be "possibly unbound" per
    pyright. They are now decided together, one branch per outcome. This
    test pins the *complete* field set each outcome produces so a future
    re-split (or a new outcome added to one enumeration but not mirrored in
    consumers of dispatch_failed_at) shows up here as a wrong field set,
    not a silent NameError on a real dispatch-failure incident.

    ``phantom_live_worker`` and ``transient_failure_under_cap`` both produce
    ``status == "dispatch_failed"`` yet must NOT share a field set (a phantom
    slot is freed without burning a redispatch attempt, so it never touches
    dispatch_failed_at). A structure that branched on the resulting
    ``status`` string instead of the originating predicate could not tell
    these apart; asserting both here pins that the branch, not the status
    value, is what determines the field set.

    Every scenario seeds one prior ``dispatch_failed_at`` entry *and* a stale
    ``escalation_reason``/``reason_class`` pair before dispatching, so the
    assertions are non-vacuous in both directions: the ``ok`` /
    ``live_worker`` / ``phantom_live_worker`` / ``transient_failure_under_cap``
    arms must actively *clear* both fields (an entry field that was never
    populated would pass an "absent" assertion for free -- this applies to
    the escalation fields too, since every non-terminal arm calls
    ``clear_escalation(entry)``), the two failure arms must *append to*
    rather than replace prior ``dispatch_failed_at`` history (``len == 2``,
    not ``len == 1`` from a fresh list, pinning that ``all_attempts`` is
    seeded from ``prev_entry`` and not reconstructed from scratch), and the
    deterministic-escalation arm must *overwrite* the stale escalation
    reason with the new one rather than leaving it in place.
    """
    app, fake_gh = _closed_pr_app(tmp_path)
    _seed_dispatch_failed_at(app.paths, 123, ["2020-01-01T00:00:00+00:00"])
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        _fake_dispatch_result_factory(
            ok=ok,
            failure_kind=failure_kind,
            pid=12345 if pid_alive is not None else None,
            process_start_time=1_234_567.0 if pid_alive is not None else None,
        ),
    )
    if pid_alive is not None:
        monkeypatch.setattr("charlie_work.workflow.is_pid_alive", lambda pid, start: pid_alive)

    app.dispatch(limit=1)

    state = load_state(app.paths.state_file)
    entry = state["issues"]["123"]

    assert entry["status"] == expect_status, scenario
    if expect_dispatch_failed_at_len is None:
        assert "dispatch_failed_at" not in entry, scenario
    else:
        assert len(entry["dispatch_failed_at"]) == expect_dispatch_failed_at_len, scenario
    if expect_escalation_reason is None:
        assert "escalation_reason" not in entry, scenario
        assert "reason_class" not in entry, scenario
    else:
        assert entry["escalation_reason"] == expect_escalation_reason, scenario
        assert entry["reason_class"] == "mechanical", scenario
