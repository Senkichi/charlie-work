"""Host-independent tests for the live Merge path shell's small pure seams.

The end-to-end behaviour is pinned by ``test_merge_path_characterization*.py``;
these cover the predicates the live driver and the decision module share, the
skip fast path, and the Convention-B write-gate guard on every apply entry point.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import _merge_path_facts as mf
import pytest
from charlie_work.labels import TransitionOutcome
from charlie_work.merge_path.apply import run_merge_ready
from charlie_work.merge_path.apply_accounting import settle_accounting
from charlie_work.merge_path.apply_merge import apply_merge_plan, label_error_of
from charlie_work.merge_path.mergequeue import mergequeue_stamp_needs_now
from charlie_work.merge_path.readiness_gates import deescalation_read_needed
from charlie_work.merge_path.gather import gather_skip
from charlie_work.merge_path.model import EffectResults, MergePlan, PlanKind
from charlie_work.merge_path.ports import ports_from_workflow
from charlie_work.write_gate import WriteGate


@pytest.mark.parametrize(
    ("labels", "issue", "hold", "unavailable", "expected"),
    [
        (("human-merge",), 7, False, False, True),
        ((), 7, False, False, False),
        (("human-merge",), None, False, False, False),
        (("human-merge",), 7, True, False, False),
        (("human-merge",), 7, False, True, False),
    ],
    ids=["armed", "no-labels", "unbound", "label-present", "check-unavailable"],
)
def test_deescalation_read_needed(labels, issue, hold, unavailable, expected) -> None:
    config = mf.cfg(human_merge_labels=labels)
    assert deescalation_read_needed(config, issue, hold, unavailable) is expected


@pytest.mark.parametrize(
    ("status", "since", "stamped", "merged", "live", "expected"),
    [
        ("mergequeue", None, None, False, "abc", True),
        ("mergequeue", "t0", "abc", False, "abc", False),
        ("mergequeue", "t0", "old", False, "abc", True),
        ("mergequeue", None, None, False, None, False),
        ("approved", None, None, False, "abc", False),
        ("approved", None, None, True, "abc", False),
    ],
    ids=[
        "first-stamp",
        "already-stamped",
        "head-moved",
        "no-live-head",
        "not-queued",
        "merged-now",
    ],
)
def test_mergequeue_stamp_needs_now(status, since, stamped, merged, live, expected) -> None:
    locked = mf.persisted(status=status, mergequeue_since=since, mergequeue_head_sha=stamped)
    assert mergequeue_stamp_needs_now(locked, merged, live) is expected


def test_gather_skip_only_for_a_persisted_merged_pr() -> None:
    config = mf.cfg()
    skipped = gather_skip(5, config, {"status": "merged", "issue_number": 9})
    assert skipped is not None
    assert skipped.issue_number == 9
    assert skipped.admission.kind.name == "SKIP"
    assert gather_skip(5, config, {"status": "approved"}) is None
    assert gather_skip(5, config, {}) is None


def test_label_error_of_reports_only_failed_transitions() -> None:
    applied = SimpleNamespace(
        outcome=TransitionOutcome.APPLIED, add_failures=(), remove_failures=()
    )
    assert label_error_of("merged", applied) is None
    failed = SimpleNamespace(
        outcome=next(o for o in TransitionOutcome if o is not TransitionOutcome.APPLIED),
        add_failures=("x",),
        remove_failures=(),
    )
    err = label_error_of("merged", failed)
    assert err is not None
    assert err["edge"] == "merged"
    assert err["add_failures"] == ("x",)


@pytest.mark.parametrize(
    "entry_point",
    [run_merge_ready, settle_accounting, apply_merge_plan],
    ids=["run_merge_ready", "settle_accounting", "apply_merge_plan"],
)
def test_apply_entry_points_refuse_a_non_write_gate(entry_point) -> None:
    """Convention B: the gate is validated before anything else happens."""
    kwargs = {"write_gate": object()}
    if entry_point is run_merge_ready:
        args = (SimpleNamespace(), 1)
    elif entry_point is settle_accounting:
        args = (SimpleNamespace(), None)
        kwargs.update(pr_number=1, issue_number=None, cfg=mf.cfg(), plan=None, results=None, pr={})
    else:
        args = (SimpleNamespace(), None)
        kwargs.update(
            pr_number=1,
            pr={},
            decision={},
            issue_number=None,
            cfg=mf.cfg(),
            plan=None,
            results=None,
        )
    gate = kwargs.pop("write_gate")
    if entry_point is run_merge_ready:
        with pytest.raises(TypeError):
            entry_point(*args, write_gate=gate)
    else:
        with pytest.raises(TypeError):
            entry_point(args[0], args[1], gate, **kwargs)


def test_dry_run_write_gate_is_a_no_op_for_a_settled_accounting(tmp_path: Path) -> None:
    """Under a dry-run gate ``settle_accounting`` decides but must not touch state.json."""
    state_file = tmp_path / "state.json"
    app = SimpleNamespace(paths=SimpleNamespace(state_file=state_file), gh=SimpleNamespace())
    gate = WriteGate(dry_run=True, state_path=state_file, repo="r")

    accounting = settle_accounting(
        app,
        ports_from_workflow(),
        gate,
        pr_number=1,
        issue_number=None,
        cfg=mf.cfg(),
        plan=MergePlan(
            kind=PlanKind.NONE, readiness=mf.readiness(gate=mf.gate(summary_ready=False))
        ),
        results=EffectResults(),
        pr={"headRefOid": "abc"},
    )

    assert accounting.failed_attempts == 1  # the settle really ran and counted
    assert not state_file.exists()
