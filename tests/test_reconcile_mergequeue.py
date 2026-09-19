"""Mergequeue-lane tests for ``reconcile``.

Split out of ``tests/test_reconcile.py`` (issue #1559, Track-1):
``detect_mergequeue_wedged`` (issue #1401),
``detect_mergequeue_not_approved`` (issue #819), and the
``apply_fixes`` lanes for the mergequeue_revoked / mergequeue_wedged
drift kinds.
"""

from __future__ import annotations

import pytest
from dataclasses import replace
from datetime import (
    UTC,
    datetime,
    timedelta,
)
from pathlib import Path
from typing import Any
from _reconcile_fixtures import (
    FakeGitHub,
    _AVIATOR_FAILURE_OUTPUT,
    _aviator_check_run,
    _passing_check_run,
    _pr,
    _write_review_decision,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import read_event_log
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import (
    DriftItem,
    apply_fixes,
    detect_mergequeue_not_approved,
    detect_mergequeue_wedged,
)
from charlie_work.state import empty_state


# ---------------------------------------------------------------------------
# detect_mergequeue_wedged (issue #1401: mergequeue PR wedged with Aviator
# FAILURE for 28h+ and no re-alert -- one-shot failed-attempt alarm has no
# time-in-queue watchdog)
# ---------------------------------------------------------------------------


def _wedged_config(
    *, mergequeue_label: str = "mergequeue", wedge_hours: float = 24.0
) -> OrchestratorConfig:
    return replace(
        OrchestratorConfig(),
        auto_merge=replace(
            OrchestratorConfig().auto_merge,
            mergequeue_label=mergequeue_label,
            mergequeue_wedge_hours=wedge_hours,
        ),
    )


def _failing_check_run(name: str, *, run_id: int) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": name,
        "status": "completed",
        "conclusion": "failure",
        "output": {},
    }


def test_detect_mergequeue_wedged_aviator_failure_escalates() -> None:
    """Condition 2 (live #1751 shape): PR carries mergequeue + Aviator blocked
    with aviator/checks completed FAILURE and a genuinely failing non-aviator
    check -> escalate (strip mergequeue, add human_needed)."""
    config = _wedged_config()
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-1751",
        "labels": [{"name": "mergequeue"}, {"name": "blocked"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1751"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _failing_check_run("Tests passed", run_id=2),
    ]

    drift = detect_mergequeue_wedged(gh, config, empty_state())

    assert len(drift) == 1
    item = drift[0]
    assert item.kind == "mergequeue_wedged"
    assert item.pr_number == 1751
    assert item.issue_number == 1751
    assert item.remove_labels == ("mergequeue",)
    assert item.add_labels == (config.labels.human_needed,)
    assert gh.commit_check_runs_calls == ["sha-1751"]


def test_detect_mergequeue_wedged_aviator_failure_all_green_does_not_fire() -> None:
    """When every non-aviator check is green the blocked label is STALE, not a
    genuine Aviator failure -- that is detect_aviator_stale_blocked's recovery
    case (remove blocked, re-queue), not an escalation. Condition 2 must stay
    out of its way to avoid re-queuing AND escalating the same PR in one pass."""
    config = _wedged_config()
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-1751",
        "labels": [{"name": "mergequeue"}, {"name": "blocked"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1751"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _passing_check_run("Tests passed", run_id=2),
    ]

    assert detect_mergequeue_wedged(gh, config, empty_state()) == []


def test_detect_mergequeue_wedged_aviator_pending_does_not_fire_condition2() -> None:
    """aviator/checks still running (no conclusion) is not a definitive
    failure -- condition 2 must not escalate a PR mid-evaluation."""
    config = _wedged_config()
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-1751",
        "labels": [{"name": "mergequeue"}, {"name": "blocked"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1751"] = [
        _aviator_check_run(None, run_id=1),
        _failing_check_run("Tests passed", run_id=2),
    ]

    assert detect_mergequeue_wedged(gh, config, empty_state()) == []


def test_detect_mergequeue_wedged_time_in_queue_escalates() -> None:
    """Condition 1: PR has carried mergequeue for > wedge_hours with no head
    movement (state mergequeue_head_sha == live head) -> escalate."""
    config = _wedged_config(wedge_hours=12.0)
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-frozen",
        "labels": [{"name": "mergequeue"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    state = empty_state()
    state["prs"]["1751"] = {
        "mergequeue_since": (datetime.now(UTC) - timedelta(hours=28)).isoformat(),
        "mergequeue_head_sha": "sha-frozen",
        "status": "mergequeue",
    }

    drift = detect_mergequeue_wedged(gh, config, state)

    assert len(drift) == 1
    assert drift[0].kind == "mergequeue_wedged"
    assert drift[0].pr_number == 1751
    assert drift[0].remove_labels == ("mergequeue",)
    # No check-run walk when the PR is not blocked.
    assert gh.commit_check_runs_calls == []


def test_detect_mergequeue_wedged_time_in_queue_head_moved_does_not_fire() -> None:
    """Head movement (Aviator rebased) resets the dwell timer: the recorded
    mergequeue_head_sha no longer matches the live head, so the PR is making
    progress and must not be escalated."""
    config = _wedged_config(wedge_hours=12.0)
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-new",
        "labels": [{"name": "mergequeue"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    state = empty_state()
    state["prs"]["1751"] = {
        "mergequeue_since": (datetime.now(UTC) - timedelta(hours=28)).isoformat(),
        "mergequeue_head_sha": "sha-old",
        "status": "mergequeue",
    }

    assert detect_mergequeue_wedged(gh, config, state) == []


def test_detect_mergequeue_wedged_time_in_queue_under_threshold_does_not_fire() -> None:
    """Dwell under the configured threshold is normal queue progress, not a wedge."""
    config = _wedged_config(wedge_hours=24.0)
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-frozen",
        "labels": [{"name": "mergequeue"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    state = empty_state()
    state["prs"]["1751"] = {
        "mergequeue_since": (datetime.now(UTC) - timedelta(hours=2)).isoformat(),
        "mergequeue_head_sha": "sha-frozen",
        "status": "mergequeue",
    }

    assert detect_mergequeue_wedged(gh, config, state) == []


def test_detect_mergequeue_wedged_time_disabled_still_allows_aviator_failure() -> None:
    """mergequeue_wedge_hours=0 disables condition 1 only; condition 2 (the
    definitive Aviator-failure signal) stays armed when mergequeue_label is set."""
    config = _wedged_config(wedge_hours=0.0)
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-1751",
        "labels": [{"name": "mergequeue"}, {"name": "blocked"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1751"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _failing_check_run("Tests passed", run_id=2),
    ]

    drift = detect_mergequeue_wedged(gh, config, empty_state())
    assert len(drift) == 1


def test_detect_mergequeue_wedged_no_mergequeue_label_config_returns_empty() -> None:
    """Without a mergequeue_label configured there is no Aviator handoff to watchdog."""
    config = replace(
        OrchestratorConfig(),
        auto_merge=replace(OrchestratorConfig().auto_merge, mergequeue_label=None),
    )
    pr = {
        **_pr(1751, "OPEN"),
        "headRefOid": "sha-1751",
        "labels": [{"name": "mergequeue"}, {"name": "blocked"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])

    assert detect_mergequeue_wedged(gh, config, empty_state()) == []


def test_detect_mergequeue_wedged_skips_pr_without_mergequeue_label() -> None:
    """Cost gate: a PR not carrying the mergequeue label is not in Aviator's queue."""
    config = _wedged_config()
    pr = {**_pr(1751, "OPEN"), "headRefOid": "sha-1751", "labels": [{"name": "blocked"}]}
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1751"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _failing_check_run("Tests passed", run_id=2),
    ]

    assert detect_mergequeue_wedged(gh, config, empty_state()) == []
    assert gh.commit_check_runs_calls == []


def test_detect_mergequeue_wedged_no_linked_issue_strips_mergequeue_only() -> None:
    """A cross-repo PR with no resolvable linked issue can still be pulled out
    of the queue (strip mergequeue); the human_needed escalation is skipped
    because there is no issue to label."""
    config = _wedged_config()
    pr = {
        **_pr(1751, "OPEN", is_cross_repository=True),
        "headRefOid": "sha-1751",
        "labels": [{"name": "mergequeue"}, {"name": "blocked"}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    gh.check_runs_by_sha["sha-1751"] = [
        _aviator_check_run("failure", _AVIATOR_FAILURE_OUTPUT, run_id=1),
        _failing_check_run("Tests passed", run_id=2),
    ]

    drift = detect_mergequeue_wedged(gh, config, empty_state())
    assert len(drift) == 1
    assert drift[0].issue_number is None
    assert drift[0].add_labels == ()
    assert drift[0].remove_labels == ("mergequeue",)


def test_apply_fixes_mergequeue_wedged_strips_mergequeue_and_escalates_issue() -> None:
    config = _wedged_config()
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["1751"] = {
        "mergequeue_since": (datetime.now(UTC) - timedelta(hours=28)).isoformat(),
        "mergequeue_head_sha": "sha-frozen",
        "status": "mergequeue",
    }
    drift = [
        DriftItem(
            kind="mergequeue_wedged",
            issue_number=1751,
            pr_number=1751,
            detail="PR #1751 wedged in mergequeue",
            fix_actions=(
                "remove label 'mergequeue' from PR #1751",
                "escalate issue #1751 to 'agent:human-needed'",
            ),
            remove_labels=("mergequeue",),
            add_labels=(config.labels.human_needed,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert gh.pr_labels_removed == [(1751, "mergequeue")]
    # transition("escalated") adds human_needed and removes active labels via
    # gh.add_issue_label / gh.remove_issue_label.
    assert (1751, config.labels.human_needed) in gh.labels_added
    # Issue state status converges to "escalated".
    assert new_state["issues"]["1751"]["status"] == "escalated"
    # The mergequeue dwell-tracking fields are cleared so the post-fix
    # re-detect does not re-fire condition 1 for the same window.
    assert "mergequeue_since" not in new_state["prs"]["1751"]
    assert "mergequeue_head_sha" not in new_state["prs"]["1751"]


def test_apply_fixes_mergequeue_wedged_records_label_write_failure() -> None:
    config = _wedged_config()
    gh = FakeGitHub(prs=[], issues=[])
    gh._fail_remove_pr_labels = {(1751, "mergequeue")}
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_wedged",
            issue_number=1751,
            pr_number=1751,
            detail="PR #1751 wedged in mergequeue",
            fix_actions=("remove label 'mergequeue' from PR #1751",),
            remove_labels=("mergequeue",),
            add_labels=(config.labels.human_needed,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    events = [e for e in new_state.get("events", []) if e.get("kind") == "reconcile"]
    assert any(
        "label_write_failed: true" in e.get("payload", {}).get("fix_actions", []) for e in events
    )


# ---------------------------------------------------------------------------
# detect_mergequeue_not_approved (issue #819 -- irrevocable mergequeue label)
# ---------------------------------------------------------------------------


def _mergequeue_config(mergequeue_label: str | None = "mergequeue") -> OrchestratorConfig:
    config = OrchestratorConfig()
    return replace(
        config, auto_merge=replace(config.auto_merge, mergequeue_label=mergequeue_label)
    )


def test_detect_mergequeue_not_approved_regression_pr_695(tmp_path: Path) -> None:
    """Reproduces PR #695's exact sequence (issue #819): ``mergequeue`` was
    applied by ship_it, the recorded verdict later flipped to
    ``request_changes`` at the PR's still-current head, and nothing in the
    orchestrator ever called ``remove_pr_label`` for ``mergequeue`` --
    Aviator (``number_of_approvals: 0``) merged the PR anyway once CI went
    green, over a standing request-changes verdict. This is the single most
    important test in this change: it must fail red on main (label never
    removed) and pass green with the fix."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(695, "OPEN"),
        "headRefOid": "sha-695-live",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path,
        config,
        695,
        {"decision": "request_changes", "reviewed_head_sha": "sha-695-live"},
    )

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    item = drift[0]
    assert item.kind == "mergequeue_revoked"
    assert item.pr_number == 695
    assert item.remove_labels == (mergequeue_label,)
    assert item.add_labels == ()

    # And the fix actually strips the label on the next reconcile pass --
    # this is the mechanical step that would have saved #695.
    state_path = tmp_path / "state.json"
    new_state = apply_fixes(gh, empty_state(), drift, config, state_path=state_path)
    assert gh.pr_labels_removed == [(695, mergequeue_label)]

    events = read_event_log(state_path)
    reconcile_events = [e for e in events if e["kind"] == "reconcile"]
    assert len(reconcile_events) == 1
    assert reconcile_events[0]["payload"]["kind"] == "mergequeue_revoked"
    assert reconcile_events[0]["payload"]["pr_number"] == 695
    # Issue #1402: apply_fixes now records the revocation reason in state so
    # merge_ready can distinguish self-revocation from #823 rejection.
    assert new_state["prs"] == {"695": {"mergequeue_revoked_reason": "not_approved"}}


def test_detect_mergequeue_not_approved_leaves_approved_at_head_alone(tmp_path: Path) -> None:
    """Negative test, equally important: a PR genuinely approved at its
    current head must never be revoked -- a false-positive revocation here
    would kick every legitimately-queued PR in the fleet out of Aviator."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(700, "OPEN"),
        "headRefOid": "sha-700",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path, config, 700, {"decision": "approved", "reviewed_head_sha": "sha-700"}
    )

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []


def test_detect_mergequeue_not_approved_fails_closed_when_decision_missing(
    tmp_path: Path,
) -> None:
    """No review-decision.json at all (never reviewed) must revoke, not
    leave a merge-authorizing label in place by default."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(701, "OPEN"),
        "headRefOid": "sha-701",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    # Deliberately no _write_review_decision call -- the file is absent.

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].remove_labels == (mergequeue_label,)
    assert "no readable review-decision.json" in drift[0].detail


def test_detect_mergequeue_not_approved_fails_closed_when_decision_malformed(
    tmp_path: Path,
) -> None:
    """Corrupt/unreadable JSON must revoke, not be silently ignored --
    mirrors ``_pr_review_approved_at_head``'s own fail-closed contract."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(702, "OPEN"),
        "headRefOid": "sha-702",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    pr_dir = paths.prs / "pr-702"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text("{not valid json", encoding="utf-8")

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].remove_labels == (mergequeue_label,)


def test_detect_mergequeue_not_approved_stale_head_still_revoked(tmp_path: Path) -> None:
    """Issue #819 item 4 -- carry-forward interaction: a PR approved at an
    OLDER sha whose head then moved (a rebase in flight) is revoked too,
    not deferred. reconcile.py cannot cheaply re-validate a rebase itself
    (that needs merge_ready's per-PR gh.pr_diff carry-forward check, the
    issue-#361 cost class this module stays out of), and merge_ready's own
    carry-forward-failure path never strips mergequeue either -- so leaving
    a stale-head approval alone reopens the exact #695 hole through a second
    door. This is deliberately revoked rather than escalated to a human:
    revoke-then-cooperative-reapply (a clean rebase gets mergequeue back via
    carry-forward + the idempotent add_pr_label on its next merge_ready
    pass) is safe, and the emitted detail distinguishes this case from a
    genuine request_changes revoke instead of collapsing both into one
    indistinguishable string."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(703, "OPEN"),
        "headRefOid": "sha-703-new",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path, config, 703, {"decision": "approved", "reviewed_head_sha": "sha-703-old"}
    )

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    item = drift[0]
    assert item.remove_labels == (mergequeue_label,)
    assert "approved at stale head" in item.detail
    assert "sha-703-old" in item.detail
    # Distinguishable from the genuine not-approved case in the same field.
    assert "recorded decision is" not in item.detail


def test_detect_mergequeue_not_approved_skips_fs_reads_when_no_pr_labeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cost gate: the per-PR review-decision.json read must only happen for
    PRs that are OPEN and already carry mergequeue -- cost must not scale
    with open-PR count, matching detect_aviator_stale_blocked's discipline."""
    config = _mergequeue_config()
    prs = [{**_pr(n, "OPEN"), "headRefOid": f"sha-{n}"} for n in range(1, 6)]
    gh = FakeGitHub(prs=prs, issues=[])

    calls: list[int] = []
    import charlie_work.reconcile as reconcile_module

    original_predicate = reconcile_module._pr_review_approved_at_head

    def _spy(cfg: Any, root: Any, pr_number: int, head_sha: str) -> bool:
        calls.append(pr_number)
        return original_predicate(cfg, root, pr_number, head_sha)

    monkeypatch.setattr(reconcile_module, "_pr_review_approved_at_head", _spy)

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []
    assert calls == []

    # Sanity: labeling exactly one PR triggers exactly one predicate call --
    # proves the spy would have caught a scaling regression above.
    mergequeue_label = config.auto_merge.mergequeue_label
    prs[2] = {**prs[2], "labels": [{"name": mergequeue_label}]}
    gh2 = FakeGitHub(prs=prs, issues=[])
    calls.clear()
    detect_mergequeue_not_approved(gh2, config, repo_root=tmp_path)
    assert calls == [3]


def test_detect_mergequeue_not_approved_blind_without_repo_root_does_not_revoke_fleet(
    tmp_path: Path,
) -> None:
    """``repo_root is None`` means the detector cannot read ANY decision
    file -- it must return [] rather than revoke mergequeue from every
    labeled PR in the fleet. A blanket revocation triggered by the
    detector's own blindness would be a false-positive catastrophe, not
    fail-closed behavior."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    prs = [
        {**_pr(n, "OPEN"), "headRefOid": f"sha-{n}", "labels": [{"name": mergequeue_label}]}
        for n in (710, 711, 712)
    ]
    gh = FakeGitHub(prs=prs, issues=[])

    assert detect_mergequeue_not_approved(gh, config, repo_root=None) == []


def test_detect_mergequeue_not_approved_noop_when_label_unconfigured(tmp_path: Path) -> None:
    """No mergequeue_label configured means Aviator handoff isn't in use at
    all -- nothing to revoke, and no decision-file reads should happen."""
    config = _mergequeue_config(mergequeue_label=None)
    pr = {**_pr(713, "OPEN"), "headRefOid": "sha-713", "labels": [{"name": "mergequeue"}]}
    gh = FakeGitHub(prs=[pr], issues=[])

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []


def test_detect_mergequeue_not_approved_ignores_merged_pr_carrying_label(
    tmp_path: Path,
) -> None:
    """Issue #819 notes PR #695 still carries mergequeue today, post-merge --
    the repo has merged PRs wearing the label right now. Only OPEN PRs are
    eligible for revocation; a merged PR's label is cosmetic history, not a
    live merge authorization."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(695, "MERGED"),
        "headRefOid": "sha-695-final",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])

    assert detect_mergequeue_not_approved(gh, config, repo_root=tmp_path) == []


def test_apply_fixes_mergequeue_revoked_removes_label(tmp_path: Path) -> None:
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_revoked",
            issue_number=None,
            pr_number=695,
            detail="PR #695 carries mergequeue but is not approved at its current head",
            fix_actions=(f"remove label {mergequeue_label!r} from PR #695",),
            remove_labels=(mergequeue_label,),
        )
    ]

    apply_fixes(gh, state, drift, config)

    assert gh.pr_labels_removed == [(695, mergequeue_label)]
    assert gh.pr_labels_added == []


def test_apply_fixes_mergequeue_revoked_records_label_write_failure() -> None:
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    gh = FakeGitHub(prs=[], issues=[])
    gh._fail_remove_pr_labels = {(695, mergequeue_label)}
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_revoked",
            issue_number=None,
            pr_number=695,
            detail="PR #695 carries mergequeue but is not approved at its current head",
            fix_actions=(f"remove label {mergequeue_label!r} from PR #695",),
            remove_labels=(mergequeue_label,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    events = [e for e in new_state.get("events", []) if e.get("kind") == "reconcile"]
    assert any(
        "label_write_failed: true" in e.get("payload", {}).get("fix_actions", []) for e in events
    )


def test_detect_mergequeue_not_approved_stale_head_records_revocation_reason(
    tmp_path: Path,
) -> None:
    """Issue #1402: a stale-head revocation (approved at an older head, the
    #819 cooperative self-revocation case) must carry
    ``mergequeue_revoked_reason='stale_head_pending_carry_forward'`` on the
    DriftItem so ``merge_ready`` can distinguish it from Aviator's #823 silent
    rejection and avoid the false handoff-failure alarm."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(703, "OPEN"),
        "headRefOid": "sha-703-new",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path, config, 703, {"decision": "approved", "reviewed_head_sha": "sha-703-old"}
    )

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].mergequeue_revoked_reason == "stale_head_pending_carry_forward"


def test_detect_mergequeue_not_approved_not_approved_records_revocation_reason(
    tmp_path: Path,
) -> None:
    """Issue #1402: a genuine not-approved revocation (request_changes verdict,
    the PR #695 case) must carry ``mergequeue_revoked_reason='not_approved'``
    so it is distinguishable from the stale-head self-revocation."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(695, "OPEN"),
        "headRefOid": "sha-695-live",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])
    _write_review_decision(
        tmp_path,
        config,
        695,
        {"decision": "request_changes", "reviewed_head_sha": "sha-695-live"},
    )

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].mergequeue_revoked_reason == "not_approved"


def test_detect_mergequeue_not_approved_missing_decision_records_not_approved(
    tmp_path: Path,
) -> None:
    """Issue #1402: a missing decision file (never reviewed) must also carry
    ``mergequeue_revoked_reason='not_approved'``."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    pr = {
        **_pr(701, "OPEN"),
        "headRefOid": "sha-701",
        "labels": [{"name": mergequeue_label}],
    }
    gh = FakeGitHub(prs=[pr], issues=[])

    drift = detect_mergequeue_not_approved(gh, config, repo_root=tmp_path)

    assert len(drift) == 1
    assert drift[0].mergequeue_revoked_reason == "not_approved"


def test_apply_fixes_mergequeue_revoked_writes_reason_to_state(tmp_path: Path) -> None:
    """Issue #1402: ``apply_fixes`` must persist
    ``mergequeue_revoked_reason`` to ``state["prs"][n]`` so ``merge_ready``
    can read it cross-pass. Verifies the stale-head reason is written; the
    not-approved reason is covered by the symmetric structure."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_revoked",
            issue_number=None,
            pr_number=695,
            detail="PR #695 carries mergequeue but is not approved at its current head",
            fix_actions=(f"remove label {mergequeue_label!r} from PR #695",),
            remove_labels=(mergequeue_label,),
            mergequeue_revoked_reason="stale_head_pending_carry_forward",
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert (
        new_state["prs"]["695"]["mergequeue_revoked_reason"] == "stale_head_pending_carry_forward"
    )
    # The reconcile event also carries the reason for observability.
    events = [e for e in new_state.get("events", []) if e.get("kind") == "reconcile"]
    assert any(
        e.get("payload", {}).get("mergequeue_revoked_reason") == "stale_head_pending_carry_forward"
        for e in events
    )


def test_apply_fixes_mergequeue_revoked_without_reason_does_not_write_field(
    tmp_path: Path,
) -> None:
    """Issue #1402: a ``mergequeue_revoked`` DriftItem without a
    ``mergequeue_revoked_reason`` (e.g. from a pre-#1402 reconcile or a
    hand-constructed drift item) must not synthesize a reason -- the field
    stays absent so ``merge_ready`` treats the revocation as a #823 case
    (the safe default)."""
    config = _mergequeue_config()
    mergequeue_label = config.auto_merge.mergequeue_label
    gh = FakeGitHub(prs=[], issues=[])
    state = empty_state()
    drift = [
        DriftItem(
            kind="mergequeue_revoked",
            issue_number=None,
            pr_number=695,
            detail="PR #695 carries mergequeue but is not approved at its current head",
            fix_actions=(f"remove label {mergequeue_label!r} from PR #695",),
            remove_labels=(mergequeue_label,),
        )
    ]

    new_state = apply_fixes(gh, state, drift, config)

    assert "mergequeue_revoked_reason" not in new_state["prs"].get("695", {})
