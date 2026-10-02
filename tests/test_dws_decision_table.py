"""Decision-table tests for the dead-worker sweep, through the public interface.

``decide(facts, observed)`` is pure, so each row builds facts, answers the
requests the plan asks for, and asserts the commits. The apply shell is covered
by a few gate tests with fake ports. (Named ``test_dws_*`` so the dormant-module
guard does not mistake this for a one-module-one-test file.)
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path

import pytest
from _dws_facts import (
    NOW,
    STAMP,
    dispatched,
    drive,
    drive_all_phases,
    for_phase,
    make_facts,
    state_with,
    sweep_config,
)
from charlie_work.dead_worker_sweep import PhaseOrderError, decide
from charlie_work.dead_worker_sweep.decide_common import drift_fingerprint
from charlie_work.dead_worker_sweep.model import (
    AdvanceToPrOpen,
    ApplyOutcomes,
    BackstopResult,
    CollectLiveHandoff,
    CreditDeadWorker,
    CreditResult,
    DrainNoOp,
    Emit,
    Escalate,
    FateResult,
    FetchOpenIssues,
    GuardedUpdate,
    FetchOpenPrs,
    FetchPrView,
    IssuesResult,
    LabelWrite,
    LiveHandoffFound,
    OpenPrForBranch,
    PrOpenResult,
    ProbeRemoteBranch,
    ProbeWorktreeHead,
    ReadBlockedOutcome,
    ReadClock,
    ReadReviewDecision,
    ReadTerminal,
    RemoteProbe,
    RepoFacts,
    ReportStaleEvidence,
    ResolveFate,
    Review,
    ReviewDecisionFacts,
    ReviewResult,
    SalvagePush,
    SalvagePushResult,
    StripAndFlag,
    TerminalFacts,
    TransitionLabel,
    UpdateIssue,
)

CFG = sweep_config()
ACTIVE = sorted(CFG.labels.active)[0]
BRANCH = "agent/7-x"
PR = {"number": 70, "headRefName": BRANCH, "headRefOid": "live1"}
NO_PR_ISSUE = {"number": 7, "title": "t", "labels": [{"name": ACTIVE}]}
BARE_ISSUE = {"number": 7, "title": "t", "labels": []}


def _fate(**kw) -> FateResult:
    return FateResult(
        **{"branch": BRANCH, "kind": "Failed", "blocked_escalatable": False, **kw}  # type: ignore[arg-type]
    )


def _answers(extra=None):
    from charlie_work.dead_worker_sweep.model import (
        ParkBackstop,
        ParkOrReclaim,
        ProbeCrossRepoScope,
        ProbeZeroArtifact,
        ScopeResult,
    )

    base = {
        CollectLiveHandoff: LiveHandoffFound({}),
        FetchOpenPrs: {},
        FetchOpenIssues: IssuesResult({7: NO_PR_ISSUE}),
        ResolveFate: lambda r: _fate(),
        ProbeZeroArtifact: False,
        ProbeCrossRepoScope: ScopeResult(True),
        ParkOrReclaim: False,
        ParkBackstop: BackstopResult({}, {}),
        ReadClock: NOW,
    }
    return base | (extra or {})


# ------------------------------------------------------------------ pre: early exit


def test_no_dispatched_issues_exits_after_the_handoff_probe() -> None:
    facts = make_facts(state_with({}))
    trace = drive(facts, {CollectLiveHandoff: LiveHandoffFound({})})
    assert trace.request_types() == ["CollectLiveHandoff"]
    assert FetchOpenPrs() not in trace.observed
    assert trace.commits == (ReportStaleEvidence("live_handoff"),)


def test_live_worker_pid_is_not_an_orphan() -> None:
    facts = make_facts(state_with({7: dispatched()}), alive={7: True})
    trace = drive(facts, {CollectLiveHandoff: LiveHandoffFound({})})
    assert trace.request_types() == ["CollectLiveHandoff"]
    assert FetchOpenPrs() not in trace.observed


def test_stale_live_handoff_alone_keeps_the_sweep_going() -> None:
    facts = make_facts(state_with({7: dispatched()}), alive={7: True})
    candidate = {"branch": BRANCH, "outcome_at": STAMP, "worker_pid": 1, "outcome_age_minutes": 9}
    trace = drive(
        facts,
        {
            CollectLiveHandoff: LiveHandoffFound({7: candidate}),
            FetchOpenPrs: {},
            FetchOpenIssues: IssuesResult({7: BARE_ISSUE}),
            ReadClock: NOW,
        },
    )
    assert FetchOpenPrs() in trace.observed
    assert "FetchOpenIssues" in trace.request_types()


# ------------------------------------------------------------------ no open PR


def test_blocked_escalatable_no_pr_strips_labels_then_escalates_and_relabels() -> None:
    answers = _answers(
        {
            ResolveFate(7, "no_pr"): _fate(
                kind="Blocked",
                blocked_escalatable=True,
                blocked_reason_kind="needs_human",
                blocked_detail="stuck",
            ),
            StripAndFlag: LabelWrite(True, (ACTIVE,)),
        }
    )
    pre, lock, post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert "StripAndFlag" in pre.request_types()
    assert StripAndFlag(7, "worker_declared_blocked") in pre.requests
    (escalate,) = lock.commits_of(Escalate)
    assert (escalate.issue, escalate.reason, escalate.reason_class) == (
        7,
        "worker_declared_blocked",
        "mechanical",
    )
    assert "worker_declared_blocked" in lock.emitted()
    assert post.commits_of(TransitionLabel) == [TransitionLabel(7, "escalated")]


def test_zero_artifact_dispatch_loop_escalates_with_session_failed_event() -> None:
    from charlie_work.dead_worker_sweep.model import ProbeZeroArtifact

    answers = _answers({ProbeZeroArtifact: True, StripAndFlag: LabelWrite(True, (ACTIVE,))})
    pre, lock, _post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert StripAndFlag(7, "zero_artifact") in pre.requests
    assert [c.reason for c in lock.commits_of(Escalate)] == ["zero_artifact_dispatch_loop"]
    assert "session_failed_escalated" in lock.emitted()


def test_throttled_death_skips_the_zero_artifact_probe() -> None:
    answers = _answers({ResolveFate: lambda r: _fate(throttled=True)})
    pre, _lock, _post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert "ProbeZeroArtifact" not in pre.request_types()


def test_issue_without_active_labels_is_never_escalated_or_parked() -> None:
    answers = _answers({FetchOpenIssues: IssuesResult({7: BARE_ISSUE})})
    pre, lock, _post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert not {"StripAndFlag", "ParkOrReclaim", "ProbeZeroArtifact"} & set(pre.request_types())
    assert not lock.commits_of(Escalate)


def test_pushed_branch_without_pr_opens_one_in_the_lock() -> None:
    repo = RepoFacts(True, True, True)
    answers = _answers(
        {
            FetchOpenIssues: IssuesResult({7: BARE_ISSUE}),
            SalvagePush: SalvagePushResult(pushed=False),
            ProbeRemoteBranch: RemoteProbe(2, None, "remote1"),
            ResolveFate: lambda r: _fate(pushed_without_pr=r.stage == "pushed"),
            OpenPrForBranch: PrOpenResult(pr_number=71, error=None),
        }
    )
    answers[ProbeWorktreeHead] = "local1"
    pre, lock, _post = drive_all_phases(
        make_facts(state_with({7: dispatched()}), repo=repo), answers
    )
    assert SalvagePush(7, BRANCH) in pre.requests
    assert OpenPrForBranch(7, BRANCH, "pushed_orphan") in lock.requests
    assert "orphaned_worker_opened_pr" in lock.emitted()
    (update,) = lock.commits_of(UpdateIssue)
    assert update.set_fields["pr_number"] == 71


def test_pr_create_failure_emits_once_per_fingerprint() -> None:
    repo = RepoFacts(True, True, True)
    answers = _answers(
        {
            FetchOpenIssues: IssuesResult({7: BARE_ISSUE}),
            SalvagePush: SalvagePushResult(pushed=False),
            ProbeRemoteBranch: RemoteProbe(2, None, "remote1"),
            ProbeWorktreeHead: "local1",
            ResolveFate: lambda r: _fate(pushed_without_pr=r.stage == "pushed"),
            OpenPrForBranch: PrOpenResult(pr_number=None, error="boom"),
        }
    )
    fresh = state_with({7: dispatched()})
    _pre, lock, _post = drive_all_phases(make_facts(fresh, repo=repo), answers)
    assert "pr_create_failed_branch_stranded" in lock.emitted()

    seen = drift_fingerprint(
        reason="dead_worker_branch_pushed_pr_create_failed", branch_name=BRANCH, error="boom"
    )
    again = state_with({7: dispatched(orphan_drift_fingerprint=seen)})
    _pre, lock, _post = drive_all_phases(make_facts(again, repo=repo), answers)
    assert "pr_create_failed_branch_stranded" not in lock.emitted()


@pytest.mark.parametrize(
    ("max_redispatch", "escalated"),
    [(1, True), (10, False)],
    ids=["over-cap", "under-cap"],
)
def test_redispatch_cap(max_redispatch: int, escalated: bool) -> None:
    from charlie_work.dead_worker_sweep.decide_common import orphan_head_fingerprint

    fingerprint = orphan_head_fingerprint(None, None)
    entry = dispatched(
        orphan_redispatch_head_sha=fingerprint,
        orphan_redispatch_at=["2026-09-30T11:50:00Z", "2026-09-30T11:55:00Z"],
        orphan_redispatch_counted_dispatch="older-dispatch",
    )
    answers = _answers({FetchOpenIssues: IssuesResult({7: BARE_ISSUE})})
    facts = make_facts(
        state_with({7: entry}), config=sweep_config(max_auto_redispatch=max_redispatch)
    )
    _pre, lock, post = drive_all_phases(facts, answers)
    reasons = [c.reason for c in lock.commits_of(Escalate)]
    if escalated:
        assert reasons == ["orphan_sweep_redispatch_cap_exceeded"]
        assert "orphan_sweep_redispatch_escalated" in lock.emitted()
        assert post.commits_of(TransitionLabel) == [TransitionLabel(7, "escalated")]
    else:
        assert reasons == []
        assert not post.commits_of(TransitionLabel)


# ------------------------------------------------------------------ with an open PR


def _pr_answers(*, decision="request_changes", reviewed="old", exit_code=1, extra=None):
    def decision_for(req: ReadReviewDecision) -> ReviewDecisionFacts:
        return ReviewDecisionFacts(decision, False, False, reviewed)

    return _answers(
        {
            FetchOpenPrs: {7: PR},
            ReadReviewDecision: decision_for,
            ReadTerminal: TerminalFacts(exit_code, 5.0),
            ReadBlockedOutcome: None,
            FetchPrView: None,
            DrainNoOp: True,
            ApplyOutcomes: True,
            GuardedUpdate: True,
            CreditDeadWorker: CreditResult(failure_kind=None, throttled_until=None),
        }
        | (extra or {})
    )


def _review_ok(**kw) -> ReviewResult:
    base = dict(
        ok=True,
        routed_to_rework=False,
        closed_unmerged_converged=False,
        escalation_deferred_live_worker=False,
        is_no_op_rework=False,
        raised_error=None,
        entry_status="dispatched",
        pr_reviewed_head_sha="old",
        entry_branch=BRANCH,
    )
    return ReviewResult(**(base | kw))


def _guarded(post) -> list[GuardedUpdate]:
    return [r for r in post.requests if isinstance(r, GuardedUpdate)]


def _head_changed_run(review: ReviewResult, *, review_available: bool = True):
    facts = make_facts(state_with({7: dispatched()}), review_available=review_available)
    return drive_all_phases(facts, _pr_answers(extra={Review: review}))


def test_head_changed_routes_to_review_and_flips_status_under_guards() -> None:
    _pre, lock, post = _head_changed_run(_review_ok())
    assert Review(7, 70, "dead_worker_with_head_change") in post.requests
    (update,) = _guarded(post)
    assert dict(update.set_items) == {"status": "reviewing"}
    assert update.require_status == "dispatched"
    assert update.require_pr_reviewed_head == "old"
    assert update.event is not None and update.event[0] == "orphaned_worker_routed_to_review"
    assert dict(update.event[1])["routed"] is True
    assert "orphaned_worker_routed_to_review" not in post.emitted()  # rides the write
    assert not lock.commits_of(UpdateIssue) or all(
        u.set_fields.get("status") != "reviewing" for u in lock.commits_of(UpdateIssue)
    )


def test_review_callback_error_emits_a_warning_and_changes_nothing() -> None:
    _pre, _lock, post = _head_changed_run(_review_ok(ok=False, raised_error="RuntimeError: boom"))
    assert not _guarded(post)
    (warn,) = [c for c in post.commits_of(Emit) if c.kind == "orphaned_worker_review_route_failed"]
    assert warn.level == "warning"
    assert warn.audit_only  # events.db only, as the original ``log_event``
    assert warn.payload["error"] == "RuntimeError: boom"


def test_review_that_routed_to_rework_is_not_marked_reviewing() -> None:
    _pre, _lock, post = _head_changed_run(_review_ok(routed_to_rework=True))
    assert all(dict(u.set_items).get("status") != "reviewing" for u in _guarded(post))


def test_failed_review_records_drift_under_a_status_guard() -> None:
    _pre, _lock, post = _head_changed_run(_review_ok(ok=False))
    (update,) = _guarded(post)
    assert {k for k, _ in update.set_items} == {"orphan_drift_fingerprint"}
    assert update.stamp_fields == ("orphan_drift_at",)  # stamped at write time
    assert update.require_status == "dispatched"
    assert update.require_pr_reviewed_head is None
    assert update.event is not None and update.event[0] == "orphaned_worker_drift"
    assert "orphaned_worker_drift" not in post.emitted()  # rides the write


def test_issue_that_left_dispatched_during_review_is_left_alone() -> None:
    _pre, _lock, post = _head_changed_run(_review_ok(entry_status="reviewing"))
    assert not _guarded(post)


def test_reviewed_head_that_moved_during_review_is_not_marked_reviewing() -> None:
    _pre, _lock, post = _head_changed_run(_review_ok(pr_reviewed_head_sha="newer"))
    assert all(dict(u.set_items).get("status") != "reviewing" for u in _guarded(post))


def test_refused_guarded_write_suppresses_the_dependent_event() -> None:
    """A write the shell refused (concurrent writer won) records no routed event."""
    facts = make_facts(state_with({7: dispatched()}))
    answers = _pr_answers(extra={Review: _review_ok(), GuardedUpdate: False})
    _pre, _lock, post = drive_all_phases(facts, answers)
    (event,) = [c for c in post.commits_of(Emit) if c.kind == "orphaned_worker_routed_to_review"]
    assert event.payload["routed"] is False


def test_refused_reviewing_flip_falls_through_like_the_elif_chain() -> None:
    """Refused ok-route flip: the not-ok arms need ``not ok``, so nothing else writes."""
    facts = make_facts(state_with({7: dispatched()}))
    answers = _pr_answers(extra={Review: _review_ok(), GuardedUpdate: False})
    _pre, _lock, post = drive_all_phases(facts, answers)
    assert len(_guarded(post)) == 1


def test_without_a_review_callback_head_change_is_recorded_as_drift() -> None:
    _pre, lock, post = _head_changed_run(_review_ok(), review_available=False)
    assert "Review" not in post.request_types()
    assert "orphaned_worker_drift" in lock.emitted()


def test_same_head_nonzero_exit_resets_to_rework_and_credits_the_death() -> None:
    answers = _pr_answers(reviewed="live1", exit_code=1)
    pre, lock, post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert CreditDeadWorker(7) in lock.requests
    statuses = [u.set_fields.get("status") for u in lock.commits_of(UpdateIssue)]
    assert "rework_requested" in statuses
    assert "orphaned_worker_recovered" in lock.emitted()
    assert "DrainNoOp" not in post.request_types()


def test_same_head_clean_exit_queues_the_no_op_drain() -> None:
    answers = _pr_answers(reviewed="live1", exit_code=0)
    _pre, lock, post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert "CreditDeadWorker" not in lock.request_types()
    (drain,) = [r for r in post.requests if isinstance(r, DrainNoOp)]
    assert [r.reason for r in drain.routes] == ["dead_worker_no_op"]


def test_same_head_declared_blocked_escalates_the_pr_issue() -> None:
    answers = _pr_answers(
        reviewed="live1", extra={ReadBlockedOutcome: {"reason_kind": "k", "detail": "d"}}
    )
    _pre, lock, post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    (escalate,) = lock.commits_of(Escalate)
    assert (escalate.reason, escalate.pr_number) == ("worker_declared_blocked", 70)
    assert post.commits_of(TransitionLabel) == [TransitionLabel(7, "escalated")]


def test_unverdicted_pr_advances_to_pr_open() -> None:
    answers = _pr_answers(decision="pending", reviewed=None) | {
        FetchOpenIssues: IssuesResult({7: NO_PR_ISSUE}),
        AdvanceToPrOpen: True,
    }
    pre, lock, _post = drive_all_phases(make_facts(state_with({7: dispatched()})), answers)
    assert AdvanceToPrOpen(7) in lock.requests
    assert "orphaned_worker_advanced_to_pr_open" in lock.emitted()
    assert CreditDeadWorker(7, classify_log=False) in lock.requests


# ------------------------------------------------------------------ phase contract


def test_lock_phase_before_pre_completes_is_a_phase_order_error() -> None:
    facts = for_phase(make_facts(state_with({7: dispatched()})), "lock")
    with pytest.raises(PhaseOrderError):
        decide(facts, {})


def test_post_phase_before_lock_completes_is_a_phase_order_error() -> None:
    facts = make_facts(state_with({7: dispatched()}))
    observed: dict = {}
    drive(facts, _pr_answers(), observed=observed)
    with pytest.raises(PhaseOrderError):
        decide(for_phase(facts, "post"), observed)


def test_decide_is_deterministic_and_never_mutates_its_inputs() -> None:
    facts = make_facts(state_with({7: dispatched()}))
    answers = _answers()
    observed: dict = {}
    drive(facts, answers, observed=observed)
    lock_facts = for_phase(facts, "lock")
    before = (copy.deepcopy(dict(lock_facts.snapshot)), copy.deepcopy(dict(lock_facts.locked)))
    observed_before = dict(observed)
    first = decide(lock_facts, observed)
    second = decide(lock_facts, observed)
    assert first == second
    assert (dict(lock_facts.snapshot), dict(lock_facts.locked)) == before
    assert observed == observed_before


def test_commits_only_grow_as_results_arrive() -> None:
    facts = make_facts(state_with({7: dispatched()}))
    answers = _answers()
    observed: dict = {}
    prefix: tuple = ()

    def check(plan) -> None:
        nonlocal prefix
        assert plan.commits[: len(prefix)] == prefix
        prefix = plan.commits

    drive(facts, answers, observed=observed, on_round=check)
    prefix = ()  # a new phase starts a new commit sequence
    drive(for_phase(facts, "lock"), answers, observed=observed, on_round=check)


def test_a_plan_asks_for_at_most_one_request() -> None:
    facts = make_facts(state_with({7: dispatched()}))
    plan = decide(facts, {})
    assert len(plan.requests) == 1


def test_facts_are_frozen() -> None:
    facts = make_facts(state_with({}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        facts.phase = "lock"  # type: ignore[misc]


# ------------------------------------------------------------------ purity guard

_PKG = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "dead_worker_sweep"
_FORBIDDEN_MODULES = {
    "charlie_work.workflow",
    "charlie_work.github",
    "charlie_work.state",
    "charlie_work.write_gate",
    "charlie_work.instrumentation",
    "subprocess",
    "time",
    "random",
}
_FORBIDDEN_CALLS = {"utc_now", "now", "utcnow", "save_state", "load_state", "append_event"}


def _imports(tree: ast.AST) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            found.add(base)
            found |= {f"{base}.{a.name}".lstrip(".") for a in node.names}
    return found


@pytest.mark.parametrize("path", sorted(_PKG.glob("decide*.py")), ids=lambda p: p.name)
def test_decide_modules_are_pure(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad_imports = {
        name
        for name in _imports(tree)
        if name.lstrip(".") in _FORBIDDEN_MODULES
        or name.lstrip(".").split(".")[0] in {"subprocess", "time", "random"}
        or name.lstrip(".") in {"workflow", "github", "state", "write_gate", "instrumentation"}
    }
    assert not bad_imports, f"{path.name} imports side-effect modules: {sorted(bad_imports)}"
    bad_calls = {
        (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        in _FORBIDDEN_CALLS
    }
    assert not bad_calls, f"{path.name} reads the clock or writes state: {sorted(bad_calls)}"


def test_the_purity_guard_detects_a_violation() -> None:
    """Control: the guard's import/call scan flags a module that breaks the rule."""
    tree = ast.parse("import subprocess\nfrom datetime import datetime\ndatetime.now()\n")
    assert "subprocess" in _imports(tree)
    assert any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", "") in _FORBIDDEN_CALLS
        for n in ast.walk(tree)
    )


# ------------------------------------------------------------------ shell gates


def _shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plans, **port_overrides):
    """Run ``run_orphan_sweep`` with ``decide`` replaced by ``plans(facts, observed)``."""
    from _rework_dispatch_fixtures import _wg
    from charlie_work import state as state_mod
    from charlie_work.dead_worker_sweep import SweepPorts, apply, ports_from_workflow

    state_file = tmp_path / "state.json"
    state_mod.save_state(state_file, state_with({7: dispatched()}))
    real = ports_from_workflow()
    ports = SweepPorts(
        **{
            **{f.name: getattr(real, f.name) for f in dataclasses.fields(SweepPorts)},
            "worker_pid_alive": lambda entry: False,
            "utc_now": lambda: STAMP,
            **port_overrides,
        }
    )
    monkeypatch.setattr(apply, "decide", plans)

    class _Gh:
        repo_root = None

        def pr_list(self):
            return []

    apply.run_orphan_sweep(
        tmp_path, state_file, CFG, _Gh(), write_gate=_wg(state_file), ports=ports
    )
    return state_file


def _events(state_file: Path, kind: str) -> list[dict]:
    from charlie_work.instrumentation import query_events

    return query_events(state_file, kind=kind)


def test_shell_aborts_when_the_commit_prefix_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work.dead_worker_sweep.model import SweepPlan

    calls = {"n": 0}

    def flapping(facts, observed):
        calls["n"] += 1
        if calls["n"] == 1:
            return SweepPlan((), (Emit("a", {}),))
        return SweepPlan((), (Emit("b", {}),))

    # Two rounds are needed: the first plan must carry a request so the loop continues.
    def plans(facts, observed):
        plan = flapping(facts, observed)
        return SweepPlan((CollectLiveHandoff(),), plan.commits) if calls["n"] == 1 else plan

    state_file = _shell(tmp_path, monkeypatch, plans)
    (event,) = _events(state_file, "dead_worker_sweep_plan_violation")
    assert event["level"] == "error"
    assert event["payload"]["phase"] == "pre"


def test_shell_aborts_loudly_on_a_phase_order_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work.dead_worker_sweep import PhaseOrderError

    def unfinished(facts, observed):
        raise PhaseOrderError("pre flow is unfinished: no observed result")

    state_file = _shell(tmp_path, monkeypatch, unfinished)  # must not raise out of the sweep
    (event,) = _events(state_file, "dead_worker_sweep_plan_violation")
    assert event["level"] == "error"
    assert event["payload"]["phase"] == "pre"
    assert "PhaseOrderError" in event["payload"]["detail"]


def test_shell_aborts_on_a_request_that_is_illegal_in_the_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work.dead_worker_sweep.model import SweepPlan

    state_file = _shell(
        tmp_path, monkeypatch, lambda facts, observed: SweepPlan((Review(7, 70, "x"),), ())
    )
    (event,) = _events(state_file, "dead_worker_sweep_plan_violation")
    assert "Review illegal in pre" in event["payload"]["detail"]


def test_shell_aborts_when_a_request_comes_back_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work.dead_worker_sweep.model import SweepPlan

    state_file = _shell(
        tmp_path,
        monkeypatch,
        lambda facts, observed: SweepPlan((ReadClock(),), ()),
    )
    (event,) = _events(state_file, "dead_worker_sweep_no_progress")
    assert event["level"] == "error"


def test_ports_cover_exactly_the_workflow_attributes() -> None:
    import charlie_work.workflow as wf
    from charlie_work.dead_worker_sweep.ports import _WORKFLOW_ATTRS, SweepPorts

    assert {f.name for f in dataclasses.fields(SweepPorts)} == set(_WORKFLOW_ATTRS)
    missing = [attr for attr in _WORKFLOW_ATTRS.values() if not hasattr(wf, attr)]
    assert missing == []


def test_ports_resolve_workflow_attributes_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    import charlie_work.workflow as wf
    from charlie_work.dead_worker_sweep import ports_from_workflow

    ports = ports_from_workflow()
    monkeypatch.setattr(wf, "utc_now", lambda: "patched-after-construction")
    assert ports.utc_now() == "patched-after-construction"


def test_lock_abort_after_a_network_effect_still_saves_the_committed_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from charlie_work import state as state_mod
    from charlie_work.dead_worker_sweep.model import FetchOpenPrs, SweepPlan, UpdateIssue

    opened = OpenPrForBranch(7, BRANCH, "pushed_orphan")
    second = OpenPrForBranch(7, BRANCH, "live_handoff")
    commit = UpdateIssue(7, {"status": "pr_open", "pr_number": 71})

    def plans(facts, observed):
        if facts.phase == "pre":
            return (
                SweepPlan((), ())
                if FetchOpenPrs() in observed
                else SweepPlan((FetchOpenPrs(),), ())
            )
        if opened not in observed:
            return SweepPlan((opened,), ())
        if second not in observed:
            return SweepPlan((second,), (commit,))
        return SweepPlan((), ())  # commit prefix vanished: plan violation after the effect

    state_file = _shell(
        tmp_path,
        monkeypatch,
        plans,
        open_pr_for_orphaned_branch=lambda **kw: (71, None, None),
    )

    assert len(_events(state_file, "dead_worker_sweep_plan_violation")) == 1
    entry = state_mod.load_state(state_file)["issues"]["7"]
    assert entry["pr_number"] == 71
    assert entry["status"] == "pr_open"


def test_draft_take_rejects_a_deleted_base_key() -> None:
    from charlie_work.dead_worker_sweep.decide_common import Draft

    draft = Draft(7, {"status": "dispatched", "pid": 1})
    draft.work.pop("pid")
    with pytest.raises(ValueError, match="pid"):
        draft.take()


def test_strip_and_flag_writes_through_gate_with_repo_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #2226: the sweep's strip lane writes through the WriteGate —
    the ``lifecycle_transition`` row lands in events.db bound to the sweep's
    repo (``_wg`` binds ``charlie-work``), not repo=NULL."""
    from _fakes_github import FakeGitHub
    from _rework_dispatch_fixtures import _wg
    from charlie_work import state as state_mod
    from charlie_work.dead_worker_sweep import SweepPorts, apply, ports_from_workflow
    from charlie_work.dead_worker_sweep.model import SweepPlan

    state_file = tmp_path / "state.json"
    state_mod.save_state(state_file, state_with({7: dispatched()}))
    real = ports_from_workflow()
    ports = SweepPorts(
        **{
            **{f.name: getattr(real, f.name) for f in dataclasses.fields(SweepPorts)},
            "worker_pid_alive": lambda entry: False,
            "utc_now": lambda: STAMP,
        }
    )
    gh = FakeGitHub()
    gh.issues = [NO_PR_ISSUE]

    def plans(facts, observed):
        if facts.phase != "pre":
            return SweepPlan((), ())
        for req in (FetchOpenIssues(), StripAndFlag(7, "worker_declared_blocked")):
            if req not in observed:
                return SweepPlan((req,), ())
        return SweepPlan((), ())

    monkeypatch.setattr(apply, "decide", plans)
    apply.run_orphan_sweep(tmp_path, state_file, CFG, gh, write_gate=_wg(state_file), ports=ports)

    assert (7, CFG.labels.human_needed) in gh.labels_added
    assert (7, ACTIVE) in gh.labels_removed
    (event,) = _events(state_file, "lifecycle_transition")
    assert event["payload"]["to_state"] == "human_needed"
    assert event["payload"]["cause"] == "escalated"
    assert event["repo"] == "charlie-work"
