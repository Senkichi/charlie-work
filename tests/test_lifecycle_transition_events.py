"""Lifecycle-transition instrumentation tests (issue #2226).

The issue: events.db had no record of lifecycle transitions — nothing marked
an issue becoming Ready or moving between Queued / In progress / PR open /
Reviewing / Needs rework / Done, so lead time and stage time had to be
reconstructed in the dashboard from unrelated events.

These tests pin the contract:

- ``labels.transition``/``labels.apply_issue_labels`` is the single
  issue-label write seam and emits ``lifecycle_transition`` (ordered
  ``from_state``/``to_state`` chain read back out of events.db).
- Reapplying a transition to the already-recorded state emits no duplicate.
- Intake emits ``ready_observed`` once per issue per Ready episode.
- Both kinds are registered at ``info`` level.
- No ``add_issue_label``/``remove_issue_label`` call may live outside
  ``labels.py`` — the AST-derived seam-coverage invariant.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _src_ast import parsed, source_files, source_text
from _fakes_github import FakeGitHub
from _reconcile_fixtures import FakeGitHub as ReconcileFakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _LEVEL_BY_KIND, close_db, query_events
from charlie_work.labels import (
    _edges,
    _EDGE_STATE_OVERRIDES,
    _label_state_map,
    _state_for_labels,
    apply_issue_labels,
    transition,
)
from charlie_work.labels import TransitionOutcome
from charlie_work.paths import runtime_paths
from charlie_work.reconcile import DriftItem, apply_fixes
from charlie_work.state import empty_state
from charlie_work.workflow import OrchestratorApp
from charlie_work.write_gate import WriteGate


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    yield
    close_db(tmp_path / ".var" / "charlie-work" / "state.json")
    close_db(tmp_path / "state.json")


def _state_path(tmp_path: Path, config: OrchestratorConfig) -> Path:
    return runtime_paths(tmp_path, config.runtime.state_dir).state_file


def _transitions(state_path: Path, issue_number: int) -> list[tuple[str | None, str]]:
    events = query_events(state_path, kind="lifecycle_transition", issue_number=issue_number)
    return [(e["payload"]["from_state"], e["payload"]["to_state"]) for e in events]


# ---------------------------------------------------------------------------
# Ordered lifecycle sequence — the canonical path end to end.
# ---------------------------------------------------------------------------


def test_lifecycle_sequence_records_ordered_transitions(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    # Intake observes the ready label — the episode-opening event.
    result = app.intake()
    assert result.ok

    labels = config.labels
    sp = paths.state_file
    # ready -> in_progress -> pr_open -> reviewing -> needs_rework
    #       -> in_progress -> pr_open -> reviewing -> done
    transition(gh, labels, 123, "dispatched", state_path=sp)
    transition(gh, labels, 123, "unescalated_pr_open", state_path=sp, pr_number=456)
    transition(gh, labels, 123, "review_started", state_path=sp, pr_number=456)
    transition(gh, labels, 123, "rework_requested", state_path=sp, pr_number=456)
    transition(gh, labels, 123, "rework_dispatched", state_path=sp, pr_number=456)
    transition(gh, labels, 123, "unescalated_pr_open", state_path=sp, pr_number=456)
    transition(gh, labels, 123, "review_started", state_path=sp, pr_number=456)
    transition(
        gh,
        labels,
        123,
        "merged",
        state_path=sp,
        pr_number=456,
        cause="merge_finalize",
    )

    assert _transitions(sp, 123) == [
        ("ready", "in_progress"),
        ("in_progress", "pr_open"),
        ("pr_open", "reviewing"),
        ("reviewing", "needs_rework"),
        ("needs_rework", "in_progress"),
        ("in_progress", "pr_open"),
        ("pr_open", "reviewing"),
        ("reviewing", "done"),
    ]

    ready_events = query_events(sp, kind="ready_observed", issue_number=123)
    assert len(ready_events) == 1
    assert ready_events[0]["payload"]["issue_number"] == 123

    # The merged transition carries the full payload contract.
    done_event = query_events(sp, kind="lifecycle_transition", issue_number=123)[-1]
    assert done_event["payload"]["pr_number"] == 456
    assert done_event["payload"]["cause"] == "merge_finalize"
    assert done_event["level"] == "info"


# ---------------------------------------------------------------------------
# Duplicate suppression.
# ---------------------------------------------------------------------------


def test_reapplied_edge_emits_no_duplicate(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()

    transition(gh, config.labels, 123, "dispatched", state_path=sp)
    transition(gh, config.labels, 123, "dispatched", state_path=sp)
    transition(gh, config.labels, 123, "dispatched", state_path=sp)

    assert _transitions(sp, 123) == [(None, "in_progress")]


def test_first_transition_records_none_from_state(tmp_path: Path) -> None:
    """No prior event chain -> ``from_state`` is None (unknown, not assumed)."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()

    transition(gh, config.labels, 123, "dispatched", state_path=sp)

    events = query_events(sp, kind="lifecycle_transition", issue_number=123)
    assert len(events) == 1
    assert events[0]["payload"]["from_state"] is None
    assert events[0]["payload"]["to_state"] == "in_progress"
    # The edge name is the default cause.
    assert events[0]["payload"]["cause"] == "dispatched"


# ---------------------------------------------------------------------------
# apply_issue_labels — the explicit-set repair seam.
# ---------------------------------------------------------------------------


def test_apply_issue_labels_emits_for_explicit_repair_sets(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    labels = config.labels

    # A reconcile-style repair converging on done via explicit sets.
    result = apply_issue_labels(
        gh,
        labels,
        123,
        add=(labels.done,),
        remove=tuple(sorted(labels.workflow_labels - {labels.done})),
        state_path=sp,
        pr_number=456,
        cause="merged_outside_orchestrator",
    )

    assert result.outcome is TransitionOutcome.APPLIED
    assert _transitions(sp, 123) == [(None, "done")]
    event = query_events(sp, kind="lifecycle_transition", issue_number=123)[0]
    assert event["payload"]["cause"] == "merged_outside_orchestrator"
    assert event["payload"]["pr_number"] == 456


def test_apply_issue_labels_non_lifecycle_labels_emit_nothing(tmp_path: Path) -> None:
    """Unrelated labels (e.g. prose_only_deps) must not fake a lifecycle move."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()

    result = apply_issue_labels(
        gh,
        config.labels,
        123,
        add=(config.labels.prose_only_deps,),
        state_path=sp,
        cause="intake_prose_only_deps",
    )

    assert result.outcome is TransitionOutcome.APPLIED
    assert _transitions(sp, 123) == []


def test_nothing_changed_repair_observes_state(tmp_path: Path) -> None:
    """An idempotent repair (empty add/remove) still records the state it
    converged on — e.g. a closed issue whose active labels were already gone."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()

    result = apply_issue_labels(
        gh,
        config.labels,
        123,
        to_state="closed",
        state_path=sp,
        cause="state_active_status_issue_closed",
    )

    assert result.outcome is TransitionOutcome.NOTHING_CHANGED
    assert _transitions(sp, 123) == [(None, "closed")]

    # A second identical repair is a genuine no-op now.
    result2 = apply_issue_labels(
        gh,
        config.labels,
        123,
        to_state="closed",
        state_path=sp,
        cause="state_active_status_issue_closed",
    )
    assert result2.outcome is TransitionOutcome.NOTHING_CHANGED
    assert _transitions(sp, 123) == [(None, "closed")]


def test_partial_failure_emits_nothing(tmp_path: Path) -> None:
    """A transition whose writes did not all land records no lifecycle move —
    the event chain must not claim a state the labels may not actually hold."""

    class FlakyRemoveGitHub(FakeGitHub):
        def remove_issue_label(self, number: int, label: str) -> bool:
            super().remove_issue_label(number, label)
            return False

    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FlakyRemoveGitHub()

    result = transition(gh, config.labels, 123, "dispatched", state_path=sp)

    assert result.outcome is TransitionOutcome.PARTIAL_FAILURE
    assert _transitions(sp, 123) == []


def test_transition_without_state_path_still_writes_labels(tmp_path: Path) -> None:
    """``state_path=None`` keeps the label writes — only the event is skipped.
    This is the escape hatch for call sites with no events.db in scope."""
    config = OrchestratorConfig()
    gh = FakeGitHub()

    result = transition(gh, config.labels, 123, "dispatched", state_path=None)

    assert result.outcome is TransitionOutcome.APPLIED
    assert (123, config.labels.in_progress) in gh.labels_added


def test_dry_run_write_gate_transition_emits_nothing(tmp_path: Path) -> None:
    """WriteGate dry-run: no label writes, no event — the no-op invariant."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    gate = WriteGate(dry_run=True, state_path=sp, repo="test-repo")

    result = gate.transition(gh, config.labels, 123, "dispatched")

    assert result.outcome is TransitionOutcome.NOTHING_CHANGED
    assert gh.labels_added == []
    assert gh.labels_removed == []
    assert query_events(sp, kind="lifecycle_transition") == []


def test_raw_transition_under_dry_run_gh_emits_nothing(tmp_path: Path) -> None:
    """Callers that bypass ``WriteGate`` still get no event under dry-run:
    ``gh.dry_run`` suppresses the emit at the seam — the label writes are
    transport no-ops, so no transition actually occurred."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub(dry_run=True)

    result = transition(gh, config.labels, 123, "dispatched", state_path=sp)

    assert result.outcome is TransitionOutcome.APPLIED  # transport-level no-op
    assert query_events(sp, kind="lifecycle_transition") == []


def test_write_gate_transition_binds_state_path_repo_and_payload(
    tmp_path: Path,
) -> None:
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    gate = WriteGate(dry_run=False, state_path=sp, repo="test-repo")

    gate.transition(gh, config.labels, 123, "unescalated_pr_open", pr_number=456, cause="salvage")

    events = query_events(sp, kind="lifecycle_transition", issue_number=123)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["from_state"] is None
    assert payload["to_state"] == "pr_open"
    assert payload["pr_number"] == 456
    assert payload["cause"] == "salvage"
    assert events[0]["repo"] == "test-repo"


# ---------------------------------------------------------------------------
# WriteGate.apply_issue_labels — repo-bound seam for computed repair sets.
# ---------------------------------------------------------------------------


def test_write_gate_apply_issue_labels_binds_repo_and_payload(
    tmp_path: Path,
) -> None:
    """``gate.apply_issue_labels`` emits ``lifecycle_transition`` into the
    bound state.db with the gate's ``repo`` on the row, forwarding
    ``pr_number``/``cause`` verbatim."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    gate = WriteGate(dry_run=False, state_path=sp, repo="test-repo")
    labels = config.labels

    result = gate.apply_issue_labels(
        gh,
        labels,
        123,
        add=(labels.pr_open,),
        remove=(labels.in_progress,),
        pr_number=456,
        cause="pr_salvage",
    )

    assert result.ok
    assert (123, labels.pr_open) in gh.labels_added
    events = query_events(sp, kind="lifecycle_transition", issue_number=123)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["to_state"] == "pr_open"
    assert payload["pr_number"] == 456
    assert payload["cause"] == "pr_salvage"
    assert events[0]["repo"] == "test-repo"


def test_write_gate_apply_issue_labels_dry_run_is_noop(tmp_path: Path) -> None:
    """Dry-run gate: no label writes and no event — the no-op invariant holds
    for the repair-set seam exactly as it does for ``gate.transition``."""
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    gate = WriteGate(dry_run=True, state_path=sp, repo="test-repo")

    result = gate.apply_issue_labels(
        gh,
        config.labels,
        123,
        add=(config.labels.pr_open,),
        remove=(config.labels.in_progress,),
        cause="pr_salvage",
    )

    assert result.outcome is TransitionOutcome.NOTHING_CHANGED
    assert gh.labels_added == []
    assert gh.labels_removed == []
    assert query_events(sp, kind="lifecycle_transition") == []


# ---------------------------------------------------------------------------
# repo binding on the non-gate paths — events.db rows must not be repo=NULL.
# ---------------------------------------------------------------------------


def test_reconcile_relabel_reclaim_binds_repo(tmp_path: Path) -> None:
    """``session_failed_relabeled`` reclaim: ``apply_fixes`` is a non-gate
    (write-gate-exempt) caller — the emitted row still carries ``repo`` so the
    dashboard can correlate it to the owning repository."""
    config = OrchestratorConfig()
    labels = config.labels
    sp = _state_path(tmp_path, config)
    repo_root = tmp_path / "owner-repo"
    repo_root.mkdir()
    gh = ReconcileFakeGitHub(prs=[], issues=[])
    drift = [
        DriftItem(
            kind="session_failed_relabeled",
            issue_number=10,
            pr_number=None,
            detail="dead worker with no open PR",
            fix_actions=("relabel",),
            remove_labels=(labels.in_progress,),
            add_labels=(labels.ready,),
        )
    ]

    apply_fixes(gh, empty_state(), drift, config, repo_root=repo_root, state_path=sp)

    events = query_events(sp, kind="lifecycle_transition", issue_number=10)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["to_state"] == "ready"
    assert payload["cause"] == "session_failed_relabeled"
    assert events[0]["repo"] == "owner-repo"


def test_salvage_pr_open_binds_repo_and_pr_number(tmp_path: Path) -> None:
    """``_open_salvage_pr`` (the sweep's salvage lane) is non-gate exempt —
    the ``pr_salvage`` event still records the repo name and the created
    PR number."""
    from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels

    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    repo_root = tmp_path / "salvage-repo"
    repo_root.mkdir()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=repo_root, pr_create_return=101)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=sp,
    )

    assert (pr_number, error) == (101, None)
    events = query_events(sp, kind="lifecycle_transition", issue_number=42)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["to_state"] == "pr_open"
    assert payload["pr_number"] == 101
    assert payload["cause"] == "pr_salvage"
    assert events[0]["repo"] == "salvage-repo"


def test_salvage_pr_open_falsy_pr_number_sentinel_emits_nothing(
    tmp_path: Path,
) -> None:
    """Dry-run sentinel: ``gh.pr_create`` returns a falsy PR number (0) under
    dry-run — ``state_path`` degrades to ``None`` so no lifecycle event claims
    a transition for a PR that was never opened."""
    from _salvage_fixtures import _SalvageTestGitHub, _salvage_labels

    from charlie_work.workflow import _open_salvage_pr

    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    repo_root = tmp_path / "salvage-repo"
    repo_root.mkdir()
    active_labels, issue_labels = _salvage_labels(config)
    gh = _SalvageTestGitHub(repo_root=repo_root, pr_create_return=0)

    pr_number, error, _closing_ref = _open_salvage_pr(
        gh=gh,
        config=config,
        repo_root=repo_root,
        branch="agent/issue-42",
        base_ref="main",
        issue_number=42,
        active_labels=active_labels,
        issue_labels=issue_labels,
        state_file=sp,
    )

    assert (pr_number, error) == (0, None)
    assert query_events(sp, kind="lifecycle_transition") == []


def test_sweep_strip_lane_emits_human_needed_with_repo(tmp_path: Path) -> None:
    """The orphan sweep's strip-and-flag lane writes through the WriteGate:
    active labels removed, ``human_needed`` applied, and the event row bound
    to the sweep's repo."""
    from charlie_work.escalation import _strip_active_and_flag_human_needed

    config = OrchestratorConfig()
    labels = config.labels
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    gate = WriteGate(dry_run=False, state_path=sp, repo="sweep-repo")

    ok = _strip_active_and_flag_human_needed(
        gh,
        config,
        42,
        active_labels={labels.in_progress},
        issue_labels={labels.in_progress, labels.ready},
        write_gate=gate,
    )

    assert ok
    assert (42, labels.in_progress) in gh.labels_removed
    assert (42, labels.human_needed) in gh.labels_added
    events = query_events(sp, kind="lifecycle_transition", issue_number=42)
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["to_state"] == "human_needed"
    assert payload["cause"] == "escalated"
    assert events[0]["repo"] == "sweep-repo"


# ---------------------------------------------------------------------------
# ready_observed — once per issue per Ready episode.
# ---------------------------------------------------------------------------


def test_intake_emits_ready_observed_once_per_episode(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    assert app.intake().ok
    assert app.intake().ok  # label unchanged across passes: no duplicate

    ready_events = query_events(paths.state_file, kind="ready_observed", issue_number=123)
    assert len(ready_events) == 1
    assert ready_events[0]["payload"]["issue_number"] == 123
    assert ready_events[0]["level"] == "info"


def test_ready_removed_and_readded_opens_new_episode(tmp_path: Path) -> None:
    """Merged/closed transitions strip ``ready``; the next sighting of the
    label is a new episode and emits ``ready_observed`` again."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    assert app.intake().ok
    # Episode 1 ends: the merged edge strips the ready label (and the fake
    # records the write) — the event chain moves to ``done``.
    transition(gh, config.labels, 123, "merged", state_path=paths.state_file)
    # The fake does not mutate its issue label list, so intake re-observes
    # the still-present ready label — a fresh episode in event-chain terms.
    assert app.intake().ok

    ready_events = query_events(paths.state_file, kind="ready_observed", issue_number=123)
    assert len(ready_events) == 2


def test_ready_episode_stays_open_through_dispatch(tmp_path: Path) -> None:
    """``dispatched`` keeps the ``ready`` label on the issue — intake must not
    re-emit ``ready_observed`` for an in-flight issue every pass."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    assert app.intake().ok
    transition(gh, config.labels, 123, "dispatched", state_path=paths.state_file)
    assert app.intake().ok
    transition(gh, config.labels, 123, "unescalated_pr_open", state_path=paths.state_file)
    assert app.intake().ok

    ready_events = query_events(paths.state_file, kind="ready_observed", issue_number=123)
    assert len(ready_events) == 1


# ---------------------------------------------------------------------------
# Done via reconcile — merge observed outside the orchestrator.
# ---------------------------------------------------------------------------


def test_reconcile_merged_outside_orchestrator_records_done(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    sp = _state_path(tmp_path, config)
    gh = ReconcileFakeGitHub(prs=[], issues=[])
    state = empty_state()
    state["prs"]["1"] = {"status": "reviewing"}
    drift = [
        DriftItem(
            kind="merged_outside_orchestrator",
            issue_number=10,
            pr_number=1,
            detail="PR #1 merged outside orchestrator",
            fix_actions=("mark state prs[1].status = 'merged'", "transition issue #10"),
        )
    ]

    apply_fixes(gh, state, drift, config, state_path=sp)

    transitions = query_events(sp, kind="lifecycle_transition", issue_number=10)
    assert len(transitions) == 1
    payload = transitions[0]["payload"]
    assert payload["to_state"] == "done"
    assert payload["pr_number"] == 1
    assert payload["cause"] == "merged_outside_orchestrator"


# ---------------------------------------------------------------------------
# Registry + seam coverage.
# ---------------------------------------------------------------------------


def test_lifecycle_event_kinds_registered_as_info() -> None:
    assert _LEVEL_BY_KIND["lifecycle_transition"] == "info"
    assert _LEVEL_BY_KIND["ready_observed"] == "info"


def test_issue_label_writes_go_through_the_seam() -> None:
    """AST-derived: ``gh.add_issue_label``/``gh.remove_issue_label`` calls are
    forbidden outside ``labels.py``, the module that owns every issue-label
    write and emits ``lifecycle_transition``."""
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    offenders: list[str] = []
    for path in source_files(src_root):
        if path.name == "labels.py":
            continue
        text = source_text(path)
        if "add_issue_label" not in text and "remove_issue_label" not in text:
            continue
        tree = parsed(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("add_issue_label", "remove_issue_label")
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


def test_every_named_edge_emits_its_lifecycle_state(tmp_path: Path) -> None:
    """Seam coverage: every named edge in ``_edges()`` produces exactly one
    ``lifecycle_transition`` event with the edge's resolved ``to_state`` —
    either derived from its add-set via ``LabelConfig`` or pinned by
    ``_EDGE_STATE_OVERRIDES`` for add-less dispositions."""
    config = OrchestratorConfig()
    labels = config.labels
    sp = _state_path(tmp_path, config)
    gh = FakeGitHub()
    state_names = set(_label_state_map(labels).values()) | {"closed"}

    for edge, (add, _remove) in sorted(_edges(labels).items()):
        expected = _EDGE_STATE_OVERRIDES.get(edge) or _state_for_labels(labels, add)
        assert expected is not None, f"edge {edge!r} resolves no lifecycle state"
        assert expected in state_names, f"edge {edge!r} resolves to unknown state {expected!r}"
        issue = 10_000 + list(sorted(_edges(labels))).index(edge)
        result = transition(gh, labels, issue, edge, state_path=sp, cause=edge)
        assert result.outcome is not TransitionOutcome.PARTIAL_FAILURE
        assert _transitions(sp, issue) == [(None, expected)], (
            f"edge {edge!r}: expected [(None, {expected!r})]"
        )


# ``issue["status"]``/``pr["status"]`` cache writes that legitimately assign a
# lifecycle-state name -- each is a state.json mirror paired with the label
# transition that owns the lifecycle move (in the same function or in the
# review/record-review callback it delegates to). A NEW write of a lifecycle
# state name outside this set fails the build: it would move the cache without
# the seam's ``lifecycle_transition`` event.
_STATUS_CACHE_MIRROR_ALLOWLIST = {
    ("apply_stages.py", "re_review"),
    ("local_lanes.py", "_local_build_packet"),
    ("state_rework_review.py", "_route_rework_candidate_to_review"),
    ("orphaned_worker_no_op_drain.py", "_drain_one"),
    ("workflow.py", "review"),
}


def test_status_writes_of_lifecycle_states_stay_paired_with_the_seam() -> None:
    """AST-derived: a ``["status"] = <lifecycle state>`` write (subscript or
    dict literal) may only appear in a function that calls the label seam
    (``transition``/``apply_issue_labels``, on any receiver) or in the
    allow-listed cache-mirror functions above."""
    lifecycle_states = set(_label_state_map(OrchestratorConfig().labels).values())
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    offenders: list[str] = []
    for path in source_files(src_root):
        tree = parsed(path)
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls_seam = False
            writes_lifecycle_status = False
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    func_expr = node.func
                    call_name = (
                        func_expr.id
                        if isinstance(func_expr, ast.Name)
                        else func_expr.attr
                        if isinstance(func_expr, ast.Attribute)
                        else ""
                    )
                    if call_name in ("transition", "apply_issue_labels"):
                        calls_seam = True
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Subscript)
                            and isinstance(target.slice, ast.Constant)
                            and target.slice.value == "status"
                            and isinstance(node.value, ast.Constant)
                            and node.value.value in lifecycle_states
                        ):
                            writes_lifecycle_status = True
                elif isinstance(node, ast.Dict):
                    for key, value in zip(node.keys, node.values):
                        if (
                            isinstance(key, ast.Constant)
                            and key.value == "status"
                            and isinstance(value, ast.Constant)
                            and value.value in lifecycle_states
                        ):
                            writes_lifecycle_status = True
            if (
                writes_lifecycle_status
                and not calls_seam
                and (path.name, func.name) not in _STATUS_CACHE_MIRROR_ALLOWLIST
            ):
                offenders.append(f"{path.name}:{func.name}")
    assert offenders == []


def test_production_label_seam_calls_pass_state_path() -> None:
    """AST-derived: every ``transition``/``apply_issue_labels`` call in src/
    passes ``state_path=`` explicitly — the keyword is required (no default),
    so a missing ``state_path=`` means the call would fail open as a
    ``TypeError`` the moment it ran. Calls on a ``WriteGate`` receiver
    (``self.write_gate``, ``ctx.write_gate``, ``write_gate``, ``gate``) are
    exempt — the gate binds its own state path."""
    src_root = Path(__file__).parents[1] / "src" / "charlie_work"
    gate_receivers = {"write_gate", "gate", "wg"}
    offenders: list[str] = []
    for path in source_files(src_root):
        tree = parsed(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_expr = node.func
            call_name = (
                func_expr.attr
                if isinstance(func_expr, ast.Attribute)
                else getattr(func_expr, "id", "")
            )
            if call_name not in ("transition", "apply_issue_labels"):
                continue
            if isinstance(func_expr, ast.Attribute):
                receiver = func_expr.value
                if isinstance(receiver, ast.Attribute) and receiver.attr == "write_gate":
                    continue
                if isinstance(receiver, ast.Name) and receiver.id in gate_receivers:
                    continue
            if "state_path" not in {kw.arg for kw in node.keywords}:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []
