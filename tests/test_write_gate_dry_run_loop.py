"""Keystone dry-run integration test for a full OrchestratorApp loop pass
(issue #1264, W6 PR4).

``WriteGate``'s own per-call invariant (binding decision C1.2, quoted in
``write_gate.py``'s module docstring) is that a single migrated call site
under ``dry_run=True`` must produce a byte-identical ``state.json`` and
``events.db`` footprint to a call that never ran at all -- "no event at
all under dry-run." This module raises that invariant one level: a FULL
``_loop_body`` pass -- intake, dispatch, review-dispatch, rework-dispatch,
the dead-worker cluster (W6 PR3), the stalled-review cluster (W6 PR2), and
every cadence-gated ``_maybe_*`` lane -- under ``dry_run=True`` against a
fixture with real, reachable work must be exactly as inert as a pass that
never ran, not merely that an empty queue makes every lane trivially
skip its own body.

Entry point: ``OrchestratorApp._loop_body()``, deliberately NOT the public
``.loop()`` / ``_loop_impl()`` wrapper. ``_loop_impl`` unconditionally
calls ``instrumentation.log_event(..., "loop_started"/"loop_completed",
...)`` and ``record_loop_pass(...)`` on every pass, dry-run or not
(workflow.py ``_loop_impl``, ~17846-17936) -- this is deliberate,
orthogonal pass-level telemetry ("a preview pass happened, and here is how
long it took") that predates the WriteGate migration and was never one of
the six primitives WriteGate wraps (``save_state``, ``append_event``,
``_record_event``, ``log_event``, ``transition``, ``kill_process`` -- see
``write_gate.py``). ``_loop_impl`` calls the *raw* ``log_event`` /
``record_loop_pass`` primitives directly, by design, and none of the
wave's clusters (dead-worker four, stalled-review sweep, kill-gating) ever
targeted this wrapper -- it is not a "caller migrated onto WriteGate" in
C1.2's sense. Routing this test through ``.loop()`` would force it to
launder a real, fixed 2-row ``events`` delta into a "zero events"
assertion, which would be dishonest; calling ``_loop_body()`` directly is
the honest choice that still exercises every lane the wave's clusters
converted. ``_loop_body`` is fully self-contained (it takes no
``correlation_context()``/``cid`` argument and derives nothing from
``_loop_impl``'s wrapper state), so calling it directly changes nothing
about how its callees observe the pass. The companion test below,
``test_loop_wrapper_telemetry_is_the_only_delta_under_dry_run_true``,
exercises the real ``.loop()`` entry point and pins that exact, bounded,
already-understood delta -- so a future reader does not mistake the choice
above for an oversight.

Four config knobs isolate leaks the wave's clusters never targeted, so
this test's zero-delta assertion stays honestly scoped to the WriteGate
mutator layer instead of silently depending on lanes with their own,
separately tracked defects:

  - ``deescalation.enabled=False`` -- ``_maybe_deescalate_mechanical``
    writes unconditionally (a raw ``save_state`` inside its own "is this
    due" check, before any ``dry_run`` consideration) once its cadence is
    due; issue #1327's territory, not this wave's.
  - ``worktree_reclamation.enabled=False`` -- ``_maybe_reclaim_worktrees``
    always emits a ``worktrees_reclaimed`` summary event on every
    cadence-due pass by deliberate design (documented in its own
    docstring -- not a defect, just out of this wave's scope).
  - ``main_ci_reclaim.enabled=False`` -- ``_maybe_reclaim_superseded_main_ci``
    writes ``main_ci_reclaim_failed``/``main_ci_reclaim_cancelled`` on its
    failure/success paths with no ``dry_run`` gate at all (#815 deliberately
    has no cadence gate either); traced directly (workflow.py
    ``_maybe_reclaim_superseded_main_ci``, ~18332-18440) rather than
    reused from a precedent's claim -- it *does* carry a
    ``main_ci_reclaim.enabled`` config gate, even though
    ``tests/test_workflow_dead_worker_write_gate.py``'s inline comment
    (written for a narrower #1311 regression scope) describes it as having
    "no config toggle." That comment is stale in this one respect; it does
    not affect that test's own correctness (it isolates the same lane via
    ``patch.object`` instead, which works regardless), but this module
    isolates it the more direct way its own source actually supports.
  - ``reconcile_pass.enabled=False`` -- ``_maybe_reconcile_drift`` emits
    exactly one summary event on every cadence-due pass by deliberate
    design (documented in its own docstring -- not a defect). Also
    directly traced to carry a real ``reconcile_pass.enabled`` gate,
    for the same reason as above.

``quota_probe`` is deliberately left at its default (``enabled=True``):
traced directly (workflow.py ``_maybe_probe_quota_recovery``,
~17938-18010), it gates on ``is_quota_probe_actionable(state)`` before
doing anything else, and a fresh state with no active throttle is not
actionable -- so it returns having written nothing. Leaving it enabled and
proving it stays silent against this fixture is stronger evidence than
disabling it outright.

The fixture seeds ONE real, provably-mutating scenario reused verbatim
from ``tests/test_workflow_dead_worker_write_gate.py``'s own loop-level
#1311 regression test: a dead worker session (issue #42) whose log tail
matches the rate-limit signature, which
``_classify_dead_sessions_and_update_throttle_state`` turns into a real
``state["throttled_until"]`` write and a real event under
``dry_run=False``. An all-idle fixture would let every assertion below
pass without the pass ever reaching a real write -- this repo's own stated
discipline (see that file's docstring: "NOT an early-return/empty-queue
fixture, which would let the assertions below pass without ever reaching
the converted code"). The default ``FakeGitHub`` seed content (one open
issue, one open PR) is explicitly cleared to keep this fixture's only
non-trivial lane the one deliberately seeded, rather than also depending
on ``dispatch()``/``dispatch_reviews()`` independently being no-ops
against arbitrary default content (already confirmed true, but keeping
the fixture minimal removes the need to lean on that fact here too).

Positive control: identical seed, ``dry_run=False``, a separate directory,
run BEFORE any assertion on the ``dry_run=True`` side -- proves this
harness can actually detect both a state mutation (``throttled_until``)
and an event emission (events.db row-count increase), not merely that
neither happened to fire ("control must be positive by evidence").
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from charlie_work.config import (
    DeescalationConfig,
    DevinConfig,
    MainCiReclaimConfig,
    OrchestratorConfig,
    ReconcilePassConfig,
    WorkerRoleConfig,
    WorktreeReclamationConfig,
)
from charlie_work.devin_shell import SessionRecord
from charlie_work.instrumentation import event_counts_by_kind
from charlie_work.paths import runtime_paths
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.workflow import OrchestratorApp


def _build_app(root: Path, *, dry_run: bool) -> tuple[OrchestratorApp, Any]:
    config = OrchestratorConfig(
        devin=DevinConfig(dispatch_command=(sys.executable, "-c", "import sys; print('ok')")),
        deescalation=DeescalationConfig(enabled=False),
        worktree_reclamation=WorktreeReclamationConfig(enabled=False),
        main_ci_reclaim=MainCiReclaimConfig(enabled=False),
        reconcile_pass=ReconcilePassConfig(enabled=False),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(root, config.runtime.state_dir)
    paths.state_file.parent.mkdir(parents=True, exist_ok=True)
    save_state(paths.state_file, empty_state())

    fake_gh = FakeGitHub()
    # Keep the fixture's only non-trivial lane the one deliberately seeded
    # below -- see the module docstring.
    fake_gh.issues = []
    fake_gh.prs = []

    app = OrchestratorApp(root, paths, config, fake_gh, dry_run=dry_run)

    sessions_dir = root / ".var" / "charlie-work" / "dispatches" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / "issue-42.log"
    log_path.write_text(
        "Some work done...\n"
        "Error: Reached overall message rate limit. Please try again later. "
        "Your limit will reset in 10 minutes.\n",
        encoding="utf-8",
    )
    record = SessionRecord(
        issue_number=42,
        branch="agent/issue-42-x",
        worktree_path=str(root / "worktree-42"),
        prompt_path=str(root / "prompt-42.md"),
        command=("devin", "--prompt-file", "prompt-42.md"),
        pid=None,
        started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
    )
    (sessions_dir / "issue-42.json").write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return app, paths


def test_loop_body_dry_run_true_produces_zero_state_mutations_and_zero_events(
    tmp_path: Path,
) -> None:
    """C1.2, raised to the full-loop-pass level: a real, reachable write in
    the dead-worker cluster must not land -- neither in state.json's
    bytes nor as any row in events.db -- when the pass runs under
    ``dry_run=True``, through the real ``_loop_body`` entry point rather
    than an isolated per-function call."""
    frozen_now = datetime.now(UTC) + timedelta(hours=1)

    # Positive control: dry_run=False, separate directory, before any
    # assertion on the dry_run=True side.
    control_app, control_paths = _build_app(tmp_path / "control", dry_run=False)
    control_events_before = sum(event_counts_by_kind(control_paths.state_file).values())
    control_app._loop_body(limit=0, merge=False, now=frozen_now)
    control_state = load_state(control_paths.state_file)
    assert control_state.get("throttled_until") is not None, (
        "positive control: dry_run=False must really set throttled_until"
    )
    control_events_after = sum(event_counts_by_kind(control_paths.state_file).values())
    assert control_events_after > control_events_before, (
        "positive control: dry_run=False must really add events.db rows -- "
        "the events.db assertion below must be decisive, not vacuously "
        "true regardless of what the pass does"
    )

    # dry_run=True: identical scenario through the real _loop_body entry
    # point.
    app, paths = _build_app(tmp_path / "dry", dry_run=True)
    before_bytes = paths.state_file.read_bytes()
    events_before = sum(event_counts_by_kind(paths.state_file).values())

    app._loop_body(limit=0, merge=False, now=frozen_now)

    assert paths.state_file.read_bytes() == before_bytes, (
        "issue #1264 (W6 PR4): a full dry_run=True loop pass must leave "
        "state.json byte-identical to a pass that never ran"
    )
    state = load_state(paths.state_file)
    assert state.get("throttled_until") is None, (
        "dry_run=True must not leak a real throttled_until write through "
        "the full loop-body entry point"
    )
    events_after = sum(event_counts_by_kind(paths.state_file).values())
    assert events_after == events_before, (
        "issue #1264 (W6 PR4), binding decision C1.2: a full dry_run=True "
        "loop pass must emit zero events -- 'no event at all under "
        f"dry-run' -- but events.db row count moved from {events_before} "
        f"to {events_after}"
    )


def test_loop_wrapper_telemetry_is_the_only_delta_under_dry_run_true(
    tmp_path: Path,
) -> None:
    """Documents, rather than merely asserts by omission, why the keystone
    test above calls ``_loop_body()`` and not ``.loop()``: driving the
    same fixture through the real public ``.loop()`` entry point under
    ``dry_run=True`` produces EXACTLY the wrapper's own pass-level
    telemetry (``loop_started`` and ``loop_completed``, one row each) and
    nothing else -- proving that delta is bounded and understood, not an
    unexamined leak. ``state.json`` stays byte-identical either way:
    ``log_event``/``record_loop_pass`` write only to events.db's
    ``events``/``loop_passes`` tables, never to state.json's content."""
    frozen_now = datetime.now(UTC) + timedelta(hours=1)
    app, paths = _build_app(tmp_path / "dry", dry_run=True)
    before_bytes = paths.state_file.read_bytes()
    events_before = event_counts_by_kind(paths.state_file)

    app.loop(limit=0, now=frozen_now)

    assert paths.state_file.read_bytes() == before_bytes, (
        "the loop_started/loop_completed telemetry writes only to "
        "events.db, never to state.json -- state.json must stay "
        "byte-identical even through the full .loop() wrapper"
    )
    events_after = event_counts_by_kind(paths.state_file)
    all_kinds = set(events_before) | set(events_after)
    delta = {
        kind: events_after.get(kind, 0) - events_before.get(kind, 0)
        for kind in all_kinds
        if events_after.get(kind, 0) != events_before.get(kind, 0)
    }
    assert delta == {"loop_started": 1, "loop_completed": 1}, (
        "the ONLY events.db delta a dry_run=True .loop() pass may produce "
        "is the wrapper's own pass-level telemetry -- any other kind "
        "appearing here is a leak the keystone test above did not catch "
        "because it calls _loop_body() directly"
    )
