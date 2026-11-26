"""Characterisation of the decision rules inside ``record_review`` and
``record_local_review`` (issue #3103).

Written before the rules moved into ``verdict_rules`` and passing on the unmoved source.
Every case runs through both public entry points (a seed the Local path cannot take, such
as a PR with no linked issue, runs on the PR path only). Each outcome is flattened to what
the decision rules decide: the result, the PR record, the issue entry, every Event (kind,
level, order, payload, indexed columns), the label edges, the write routes taken
(``workflow.save_state`` / ``workflow.transition`` versus the ``WriteGate`` methods), and how
many times ``utc_now`` was read. The outcomes are pinned in
``fixtures/verdict_rules_golden.json``; a handful of readable assertions below pin the
rules themselves, so a regenerated golden file cannot hide a change.

Regenerate the golden file only for an intended change:
``CW_REGEN_VERDICT_RULES_GOLDEN=1 uv run pytest tests/test_verdict_rules_characterisation.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import _git_templates
import charlie_work.selection_wrapper as selection_wrapper
import charlie_work.workflow as workflow
from _verdict_parity_driver import (
    ISSUE,
    Scenario,
    _call_args,
    _fresh_repo,
    _label_names,
    _local_app,
    _normalise,
    _pin_worker_selection,
    _remote_app,
    _seed,
    _write_packet,
)
from charlie_work.instrumentation import query_events
from charlie_work.labels import TransitionOutcome, TransitionResult
from charlie_work.rescue_review import LEGACY_VACUOUS_SUMMARY
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.write_gate import WriteGate

GOLDEN = Path(__file__).parent / "fixtures" / "verdict_rules_golden.json"
HUMAN_CALL = "This needs a human decision before it can ship."

_STALE_ESCALATION = {
    "escalation_reason": "stale_reason",
    "reason_class": "judgment",
    "escalation_reasons_seen": ["stale_reason"],
}
_WORKER = {"worker_pid": 4242, "worker_process_start_time": 17.5}
_STREAKS = {
    "review_dispatch_attempt_count": 3,
    "review_log_unreadable_streak": 2,
    "review_api_error_streak": 1,
    "review_turn_limit_miss_streak": 4,
    "consecutive_failed_merge_attempts": 2,
    "reviewer_pid": 99,
    "reviewer_process_start_time": 5.0,
    "review_dispatch_status": "review_dispatched",
    "review_session_metrics": {"cost_usd": 1.5},
}


@dataclass(frozen=True)
class Case:
    name: str
    scenario: Scenario
    pr_seed: dict[str, Any] = field(default_factory=dict)
    issue_seed: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    issue_view_raises: bool = False
    fail_labels: bool = False
    no_linked_issue: bool = False  # the PR path resolves no issue number
    pr_only: bool = False


def _rc(name: str, decision: str = "request_changes", **kwargs: Any) -> Scenario:
    return Scenario(name, decision, **kwargs)


def _build_cases() -> tuple[Case, ...]:
    cases: list[Case] = []

    def add(name: str, scenario: Scenario | None = None, /, **kwargs: Any) -> None:
        scenario = scenario or _rc(name)
        cases.append(Case(name, scenario, **kwargs))

    # The rework counter and its head_advanced gate (``max_rework_cycles`` is 2).
    add("counter_first_verdict")
    add("counter_head_advanced", _rc("a", prior_count=1, prior_head="older"))
    add("counter_head_unchanged", _rc("a", prior_count=1, prior_head="same"))
    add("counter_zero_head_unchanged", _rc("a", prior_count=0, prior_head="same"))
    # The cap at, below and above, with the head advanced and unchanged.
    add("cap_below", _rc("a", prior_count=1, prior_head="older"))
    add("cap_at", _rc("a", prior_count=2, prior_head="older"))
    add("cap_above", _rc("a", prior_count=3, prior_head="older"))
    add("cap_at_head_unchanged", _rc("a", prior_count=2, prior_head="same"))
    add("cap_above_head_unchanged", _rc("a", prior_count=5, prior_head="same"))
    add("cap_at_first_verdict", _rc("a", prior_count=2))
    # The Rescue tier on, off and already spent.
    add("rescue_on_at_cap", _rc("a", prior_count=2, prior_head="older", rescue_enabled=True))
    add(
        "rescue_on_spent",
        _rc("a", prior_count=2, prior_head="older", rescue_enabled=True, rescue_attempted=True),
    )
    add("rescue_off_at_cap", _rc("a", prior_count=2, prior_head="older", rescue_enabled=False))
    add("rescue_on_below_cap", _rc("a", prior_count=1, prior_head="older", rescue_enabled=True))
    add(
        "rescue_on_at_cap_head_unchanged",
        _rc("a", prior_count=2, prior_head="same", rescue_enabled=True),
    )
    add(
        "rescue_on_above_cap",
        _rc("a", prior_count=4, prior_head="older", rescue_enabled=True),
    )
    add(
        "rescue_on_at_cap_issue_closed",
        _rc("a", prior_count=2, prior_head="older", rescue_enabled=True, issue_closed=True),
    )
    # Mechanical versus judgment escalation.
    add("blocked", Scenario("a", "blocked", summary="Needs a security call."))
    add(
        "blocked_stale_escalation",
        Scenario("a", "blocked"),
        pr_seed=_STALE_ESCALATION,
        issue_seed=_STALE_ESCALATION,
    )
    add(
        "reclassified_human_call",
        _rc("a", required_changes=(HUMAN_CALL,), summary="For the operator."),
    )
    add(
        "cap_mechanical_with_worker_pid",
        _rc("a", prior_count=2, prior_head="older"),
        issue_seed=_WORKER,
    )
    # Status transitions and label edges, approve / request changes / blocked.
    add("approve", Scenario("a", "approved"))
    add(
        "approve_clears_escalation_and_worker",
        Scenario("a", "approved"),
        pr_seed={**_STALE_ESCALATION, "status": "blocked"},
        issue_seed={**_STALE_ESCALATION, **_WORKER, "status": "blocked", "merge_alert": "BAD"},
    )
    add(
        "approve_after_escalated_status_record",
        Scenario("a", "approved", pr_status="reviewing"),
        issue_seed={"status": "escalated", **_STALE_ESCALATION},
    )
    add(
        "request_changes_clears_escalation_keeps_worker",
        _rc("a", prior_count=0),
        pr_seed={**_STALE_ESCALATION, "status": "blocked"},
        issue_seed={**_STALE_ESCALATION, **_WORKER, "status": "blocked", "merge_alert": "BAD"},
    )
    add("blocked_pops_worker", Scenario("a", "blocked"), issue_seed=_WORKER)
    add(
        "blocked_after_rework_status",
        Scenario("a", "blocked", issue_status="rework_requested"),
        pr_seed={"escalation_reason": "old_pr_reason"},
    )
    # content_advanced and the no-op checkpoint.
    add("noop_patch_id_new", _rc("a"), pr_seed={"reviewed_patch_id": "different"})
    add("noop_patch_id_same", _rc("a"), pr_seed={"reviewed_patch_id": "SAME"})
    add("noop_patch_id_first", _rc("a"), pr_seed={})
    add(
        "noop_excluded_provenance",
        _rc("a", provenance="ci_gate_auto_reject"),
        pr_seed={"reviewed_patch_id": "different"},
    )
    add(
        "noop_operator_provenance",
        _rc("a", provenance="operator_manual"),
        pr_seed={"reviewed_patch_id": "different"},
    )
    add(
        "noop_no_packet_live_diff",
        _rc("a", packet="none"),
        pr_seed={"reviewed_patch_id": "different"},
    )
    add(
        "noop_checkpoint_not_set_when_escalated",
        _rc("a", prior_count=2, prior_head="older"),
        pr_seed={"reviewed_patch_id": "different"},
    )
    # The record carries forward and resets its fields.
    add("streaks_reset_request_changes", _rc("a"), pr_seed=_STREAKS)
    add("streaks_reset_approved", Scenario("a", "approved"), pr_seed=_STREAKS)
    add(
        "session_metrics_given",
        Scenario("a", "approved", session_metrics=(("cost_usd", 2.5), ("verdict_source", "log"))),
        pr_seed=_STREAKS,
    )
    add("verdict_source_log", Scenario("a", "blocked", verdict_source="log"))
    # Events.
    add("vacuous_summary", _rc("a", summary=LEGACY_VACUOUS_SUMMARY))
    add("derived_required_changes", _rc("a", summary="Add the missing test for b."))
    add("itemized_required_changes", _rc("a", required_changes=("one", "two")))
    add("external_finding", _rc("a", external_comment=True), pr_only=True)
    add(
        "external_finding_vacuous",
        _rc("a", summary=LEGACY_VACUOUS_SUMMARY, external_comment=True),
        pr_only=True,
    )
    # Issue closed, label failures, issue_view failing.
    add("issue_closed_request_changes", _rc("a", issue_closed=True))
    add("issue_closed_at_cap", _rc("a", prior_count=2, prior_head="older", issue_closed=True))
    add("issue_closed_approved", Scenario("a", "approved", issue_closed=True))
    add("issue_closed_blocked", Scenario("a", "blocked", issue_closed=True))
    add("issue_view_raises", _rc("a"), issue_view_raises=True)
    add("label_error_request_changes", _rc("a"), fail_labels=True)
    add("label_error_escalated", _rc("a", prior_count=2, prior_head="older"), fail_labels=True)
    add("label_error_approved", Scenario("a", "approved"), fail_labels=True)
    add("label_error_blocked", Scenario("a", "blocked"), fail_labels=True)
    add(
        "label_error_with_comment_error", Scenario("a", "approved", comment=True), fail_labels=True
    )
    # No linked issue (PR path only): no issue rules, no label edges.
    add(
        "no_issue_request_changes",
        _rc("a"),
        no_linked_issue=True,
        pr_seed={"escalation_reason": "keep"},
        pr_only=True,
    )
    add(
        "no_issue_approved",
        Scenario("a", "approved"),
        no_linked_issue=True,
        pr_seed={"escalation_reason": "keep"},
        pr_only=True,
    )
    add(
        "no_issue_blocked",
        Scenario("a", "blocked"),
        no_linked_issue=True,
        pr_seed={"escalation_reason": "keep"},
        pr_only=True,
    )
    add(
        "no_issue_cap",
        _rc("a", prior_count=2, prior_head="older"),
        no_linked_issue=True,
        pr_only=True,
    )
    # Dry run: the two paths write differently today.
    add("dry_run_approved", Scenario("a", "approved"), dry_run=True)
    add("dry_run_request_changes", _rc("a"), dry_run=True)
    add("dry_run_blocked", Scenario("a", "blocked"), dry_run=True)
    add("dry_run_cap", _rc("a", prior_count=2, prior_head="older"), dry_run=True)
    return tuple(cases)


CASES = _build_cases()
_FAILED = TransitionResult(TransitionOutcome.PARTIAL_FAILURE, [(ISSUE, "x")], [])


@dataclass
class _Run:
    calls: list[str] = field(default_factory=list)
    utc_now: int = 0


def _install_spies(monkeypatch: pytest.MonkeyPatch, run: _Run, case: Case) -> None:
    real_utc_now = workflow.utc_now
    real_wf_save, real_wf_transition = workflow.save_state, workflow.transition
    real_gate_save, real_gate_transition = WriteGate.save_state, WriteGate.transition
    real_gate_record = WriteGate.record_event

    def utc_now(*args: Any, **kwargs: Any) -> Any:
        run.utc_now += 1
        return real_utc_now(*args, **kwargs)

    def wf_save(path: Path, data: Any, *args: Any, **kwargs: Any) -> Any:
        run.calls.append("workflow.save_state")
        return real_wf_save(path, data, *args, **kwargs)

    def wf_transition(gh: Any, labels: Any, number: int, edge: str, **kwargs: Any) -> Any:
        keys = sorted(k for k, v in kwargs.items() if v is not None)
        run.calls.append(
            f"workflow.transition:{edge}:{','.join(keys)}:pr={kwargs.get('pr_number')}"
        )
        if case.fail_labels:
            return _FAILED
        return real_wf_transition(gh, labels, number, edge, **kwargs)

    def gate_save(self: WriteGate, data: Any) -> Any:
        run.calls.append("gate.save_state")
        return real_gate_save(self, data)

    def gate_transition(
        self: WriteGate, gh: Any, labels: Any, number: int, edge: str, **kw: Any
    ) -> Any:
        run.calls.append(f"gate.transition:{edge}:pr={kw.get('pr_number')}")
        if case.fail_labels:
            return _FAILED
        return real_gate_transition(self, gh, labels, number, edge, **kw)

    def gate_record(self: WriteGate, state: Any, kind: str, payload: Any, **kw: Any) -> Any:
        run.calls.append(f"gate.record_event:{kind}")
        return real_gate_record(self, state, kind, payload, **kw)

    monkeypatch.setattr(workflow, "utc_now", utc_now)
    monkeypatch.setattr(workflow, "save_state", wf_save)
    monkeypatch.setattr(workflow, "transition", wf_transition)
    monkeypatch.setattr(WriteGate, "save_state", gate_save)
    monkeypatch.setattr(WriteGate, "transition", gate_transition)
    monkeypatch.setattr(WriteGate, "record_event", gate_record)


def _extend_seed(app: Any, case: Case) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        pr = state["prs"][str(ISSUE)]
        for key, value in case.pr_seed.items():
            pr[key] = value
        state["issues"][str(ISSUE)].update(case.issue_seed)
        save_state(app.paths.state_file, state)


def _snapshot(case: Case, side: str, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    scenario = case.scenario
    repo = _fresh_repo()
    run = _Run()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(selection_wrapper, "ci_fleet_executable", lambda: Path("ci-fleet"))
        if side == "local":
            app = _local_app(scenario, repo)
            roots = (repo.root, app.paths.state_file.parent)
        else:
            root = Path(tempfile.mkdtemp(prefix="cw-verdict-pr-"))
            app, gh = _remote_app(scenario, repo, root)
            roots = (root,)
            if case.no_linked_issue:
                gh.prs[0].update(headRefName="feature/x", title="Thing", body="No link.")
        _pin_worker_selection(*roots[:1])
        _seed(app, scenario, repo, local=side == "local")
        _write_packet(app, scenario, repo)
        _extend_seed(app, case)
        if case.pr_seed.get("reviewed_patch_id") == "SAME":
            _same_patch_id(app, scenario, repo)
        if case.dry_run:
            app.dry_run = True
            app.write_gate = WriteGate(True, app.paths.state_file, app.repo_root.name)
        labels_before = _labels(app, side)
        if case.issue_view_raises:
            _raise_once(app.gh)
        events_before = len(query_events(app.paths.state_file))
        _install_spies(patch, run, case)
        args = _call_args(app, scenario, repo)
        result = (app.record_local_review if side == "local" else app.record_review)(**args)
    flat = _flatten(app, side, case, result, run, roots, labels_before, events_before)
    text = json.dumps(flat, sort_keys=True)
    for name, sha in (("HEAD", repo.head), ("OLDER", repo.older_head)):
        text = text.replace(sha, name)
    return json.loads(text)


def _same_patch_id(app: Any, scenario: Scenario, repo: Any) -> None:
    patch_id = workflow._calculate_patch_id(repo.diff_text)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(ISSUE)]["reviewed_patch_id"] = patch_id
        save_state(app.paths.state_file, state)


def _raise_once(gh: Any) -> None:
    real = gh.issue_view
    calls: list[int] = []

    def issue_view(number: int) -> Any:
        calls.append(number)
        if len(calls) == 1:
            raise RuntimeError("issue_view failed")
        return real(number)

    object.__setattr__(gh, "issue_view", issue_view)


def _labels(app: Any, side: str) -> frozenset[str]:
    issue = app.gh.issue_view(ISSUE)
    return _label_names(issue) if issue else frozenset()


def _flatten(
    app: Any,
    side: str,
    case: Case,
    result: Any,
    run: _Run,
    roots: tuple[Path, ...],
    labels_before: frozenset[str],
    events_before: int,
) -> dict[str, Any]:
    def norm(value: Any) -> Any:
        return _normalise(value, root=roots[0], extra_roots=roots[1:])

    state = load_state(app.paths.state_file)
    pr = dict(state["prs"].get(str(ISSUE)) or {})
    for transport in ("branch", "baseRefName", "decision_path"):
        pr.pop(transport, None)
    data = dict(result.data or {})
    data.pop("decision_path", None)
    data["rework_path"] = bool(data.get("rework_path"))
    events = [
        {
            "kind": event["kind"],
            "level": event.get("level"),
            "pr_number": event.get("pr_number"),
            "issue_number": event.get("issue_number"),
            "repo_is_repo_name": event.get("repo") == app.repo_root.name,
            "payload": norm(event.get("payload") or {}),
        }
        for event in query_events(app.paths.state_file)[events_before:]
    ]
    pr_dir = app.paths.prs / f"pr-{ISSUE}"
    decision_file = pr_dir / "review-decision.json"
    decision = json.loads(decision_file.read_text("utf-8")) if decision_file.is_file() else {}
    return {
        "result": {"ok": result.ok, "message": norm(result.message), "data": norm(data)},
        "pr_record": norm(pr),
        "issue_entry": norm(dict(state["issues"].get(str(ISSUE)) or {})),
        "events": events,
        "labels_after": sorted(_labels(app, side)),
        "labels_before": sorted(labels_before),
        "calls": run.calls,
        "utc_now_reads": run.utc_now,
        "decision_file": {
            key: decision.get(key)
            for key in ("decision", "escalated", "findings_channel", "reviewed_head_sha")
        },
        "rework_prompt": (pr_dir / "rework-prompt.md").is_file(),
        "rounds": sorted(p.name for p in (pr_dir / "rounds").glob("round-*")),
    }


def _all_snapshots(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case in CASES:
        for side in ("pr", "local"):
            if side == "local" and (case.pr_only or case.no_linked_issue):
                continue
            out[f"{case.name}|{side}"] = _snapshot(case, side, monkeypatch)
    return out


def _dump(snapshots: dict[str, Any]) -> str:
    lines = [
        f"{json.dumps(key)}: {json.dumps(value, sort_keys=True)}"
        for key, value in snapshots.items()
    ]
    return "{\n" + ",\n".join(lines) + "\n}\n"


@pytest.fixture(scope="module")
def snapshots() -> dict[str, Any]:
    with pytest.MonkeyPatch.context() as patch:
        # ``_isolate_git_env``'s GIT_CONFIG_* trio is function-scoped; apply it
        # here too or ``_repo_template`` builds under a fingerprint the call
        # phase never looks up (the issue-#2565 rule ``_git_templates``
        # documents) and the verdict suites keep a second, cold template.
        for key, value in _git_templates.GIT_CONFIG_ENV.items():
            patch.setenv(key, value)
        return _all_snapshots(patch)


def test_outcomes_match_the_pinned_golden_file(snapshots: dict[str, Any]) -> None:
    if os.environ.get("CW_REGEN_VERDICT_RULES_GOLDEN") == "1":
        GOLDEN.write_text(_dump(snapshots), encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(golden) == sorted(snapshots)
    for key in golden:
        assert snapshots[key] == golden[key], key


# --- the rules themselves, readable ---------------------------------------------------


def _both(snapshots: dict[str, Any], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return snapshots[f"{name}|pr"], snapshots[f"{name}|local"]


@pytest.mark.parametrize(
    ("name", "count", "escalated"),
    [
        ("counter_first_verdict", 1, False),
        ("counter_head_advanced", 2, False),
        ("counter_head_unchanged", 1, False),
        ("cap_below", 2, False),
        ("cap_at", 2, True),
        ("cap_above", 3, True),
        ("cap_at_head_unchanged", 2, False),
        ("cap_above_head_unchanged", 5, False),
    ],
)
def test_rework_counter_and_cap(
    snapshots: dict[str, Any], name: str, count: int, escalated: bool
) -> None:
    for outcome in _both(snapshots, name):
        data = outcome["result"]["data"]
        assert (data["request_changes_count"], data["escalated"]) == (count, escalated)
        assert outcome["pr_record"]["status"] == ("escalated" if escalated else "request_changes")
        assert outcome["issue_entry"]["status"] == (
            "escalated" if escalated else "rework_requested"
        )
        assert outcome["rework_prompt"] is (not escalated)


def test_rescue_tier_pr_path_configured_and_local_path_off(snapshots: dict[str, Any]) -> None:
    pr, local = _both(snapshots, "rescue_on_at_cap")
    assert pr["result"]["data"]["rescue_dispatched"] is True
    assert pr["result"]["data"]["escalated"] is False
    assert pr["pr_record"]["status"] == "request_changes"
    assert [e["kind"] for e in pr["events"]][-2:] == ["rescue_dispatched", "lifecycle_transition"]
    # The Local path never consults ``rescue.enabled``: it escalates at the cap.
    assert local["result"]["data"]["escalated"] is True
    assert "rescue_dispatched" not in local["result"]["data"]
    assert "rescue_dispatched" not in [e["kind"] for e in local["events"]]
    spent, _ = _both(snapshots, "rescue_on_spent")
    assert spent["result"]["data"]["escalated"] is True
    assert spent["result"]["data"]["rescue_dispatched"] is False
    off, _ = _both(snapshots, "rescue_off_at_cap")
    assert off["result"]["data"]["escalated"] is True


def test_mechanical_and_judgment_escalation_classes(snapshots: dict[str, Any]) -> None:
    for outcome in _both(snapshots, "cap_at"):
        assert outcome["issue_entry"]["escalation_reason"] == "max_rework_cycles_exceeded"
        assert outcome["issue_entry"]["reason_class"] == "mechanical"
        assert outcome["pr_record"]["escalation_reason"] == "max_rework_cycles_exceeded"
    for outcome in _both(snapshots, "blocked"):
        assert outcome["issue_entry"]["escalation_reason"] == "review_blocked"
        assert outcome["issue_entry"]["reason_class"] == "judgment"
        assert outcome["issue_entry"]["status"] == "blocked"
        assert outcome["pr_record"]["escalation_reason"] == "review_blocked"


def test_approve_clears_escalation_and_the_worker_pid(snapshots: dict[str, Any]) -> None:
    for outcome in _both(snapshots, "approve_clears_escalation_and_worker"):
        issue = outcome["issue_entry"]
        assert issue["status"] == "approved"
        assert issue["merge_alert"] == "OK"
        for key in ("escalation_reason", "reason_class", "escalation_reasons_seen"):
            assert key not in issue
        assert "worker_pid" not in issue
        assert "worker_process_start_time" not in issue
        assert "escalation_reason" not in outcome["pr_record"]
    for outcome in _both(snapshots, "blocked_pops_worker"):
        assert "worker_pid" not in outcome["issue_entry"]
    for outcome in _both(snapshots, "request_changes_clears_escalation_keeps_worker"):
        assert outcome["issue_entry"]["worker_pid"] == 4242
        assert "escalation_reason" not in outcome["issue_entry"]
        assert "escalation_reason" not in outcome["pr_record"]


@pytest.mark.parametrize(
    ("name", "checkpoint"),
    [
        ("noop_patch_id_new", True),
        ("noop_patch_id_same", False),
        ("noop_patch_id_first", True),
        ("noop_excluded_provenance", False),
        ("noop_operator_provenance", True),
        ("noop_no_packet_live_diff", True),
        ("noop_checkpoint_not_set_when_escalated", False),
    ],
)
def test_no_op_checkpoint_follows_content_advance(
    snapshots: dict[str, Any], name: str, checkpoint: bool
) -> None:
    for outcome in _both(snapshots, name):
        assert ("no_op_checkpoint_at" in outcome["issue_entry"]) is checkpoint


@pytest.mark.parametrize(
    "name", ["no_issue_request_changes", "no_issue_approved", "no_issue_blocked", "no_issue_cap"]
)
def test_pr_without_a_linked_issue_takes_no_issue_rules_and_no_label_edge(
    snapshots: dict[str, Any], name: str
) -> None:
    outcome = snapshots[f"{name}|pr"]
    assert outcome["pr_record"]["issue_number"] is None
    assert outcome["result"]["data"]["label_error"] is None
    assert not [c for c in outcome["calls"] if ".transition" in c]
    # The seeded issue entry is untouched: no status write, no escalation, no merge_alert.
    assert outcome["issue_entry"]["status"] == "dispatched"
    assert "merge_alert" not in outcome["issue_entry"]
    assert "escalation_reason" not in outcome["issue_entry"]
    assert not [e for e in outcome["events"] if e["kind"] == "lifecycle_transition"]
    # A stale escalation_reason on the PR record is not cleared without an issue.
    if name != "no_issue_cap":
        assert outcome["pr_record"]["escalation_reason"] in {"keep", "review_blocked"}


def test_label_edges_per_decision(snapshots: dict[str, Any]) -> None:
    def edges(outcome: dict[str, Any]) -> list[str]:
        return [c.split(":")[1] for c in outcome["calls"] if ".transition:" in c]

    assert [edges(o) for o in _both(snapshots, "approve")] == [["review_approved"]] * 2
    assert [edges(o) for o in _both(snapshots, "blocked")] == [["blocked"]] * 2
    assert [edges(o) for o in _both(snapshots, "counter_first_verdict")] == [
        ["rework_requested"]
    ] * 2
    assert [edges(o) for o in _both(snapshots, "cap_at")] == [["operator_queued"]] * 2
    for outcome in _both(snapshots, "issue_closed_request_changes"):
        assert edges(outcome) == []
        assert outcome["events"][-1]["kind"] == "rework_label_skipped_issue_closed"
        assert outcome["events"][-1]["level"] == "warning"
    for outcome in _both(snapshots, "issue_closed_approved"):
        assert edges(outcome) == ["review_approved"]


def test_label_transition_failures_are_reported_as_label_error(snapshots: dict[str, Any]) -> None:
    for name, edge in (
        ("label_error_request_changes", "rework_requested"),
        ("label_error_approved", "review_approved"),
        ("label_error_blocked", "blocked"),
    ):
        for outcome in _both(snapshots, name):
            error = outcome["result"]["data"]["label_error"]
            assert error["edge"] == edge
            assert error["outcome"] == "partial_failure"
            assert "(label update failed: partial_failure)" in outcome["result"]["message"]
    for outcome in _both(snapshots, "label_error_with_comment_error"):
        assert outcome["result"]["data"]["label_error"]["edge"] == "review_approved"


def test_write_routes_per_path(snapshots: dict[str, Any]) -> None:
    pr, local = _both(snapshots, "approve")
    assert pr["calls"].count("workflow.save_state") >= 1
    assert any(c.startswith("workflow.transition:review_approved") for c in pr["calls"])
    assert not any(
        c.startswith("gate.save_state") or c.startswith("gate.transition") for c in pr["calls"]
    )
    assert any(c == "gate.save_state" for c in local["calls"])
    assert any(c.startswith("gate.transition:review_approved") for c in local["calls"])
    assert not any(c.startswith("workflow.save_state") for c in local["calls"])


def test_dry_run_today_differs_between_the_paths(snapshots: dict[str, Any]) -> None:
    pr, local = _both(snapshots, "dry_run_approved")
    # PR path: still saves state and calls the label write, reports no label_error.
    assert pr["issue_entry"]["status"] == "approved"
    assert pr["result"]["data"]["label_error"] is None
    # Local path: gate suppresses the save and the label write.
    assert local["issue_entry"]["status"] != "approved"
    assert local["result"]["data"]["label_error"]["outcome"] == "nothing_changed"
    assert "(label update failed: nothing_changed)" in local["result"]["message"]
    for outcome in (pr, local):
        # Neither path emits a Verdict Event (the PR path's raw transition still emits
        # its lifecycle_transition because the test gh is not a dry-run transport).
        assert "record_review" not in [e["kind"] for e in outcome["events"]]
