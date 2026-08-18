"""New WriteGate-conversion coverage for the stalled-review cluster (issue
#1264, W6 PR2).

Four tests prove each of the four converted functions' ``write_gate``
parameter actually gates the write primitive it wraps under ``dry_run=True``
-- not merely that the function's *return value* looks unchanged (a "no PR
needed recovery" fixture would satisfy that trivially), but that the
underlying ``charlie_work.write_gate.append_event`` / ``.save_state``
primitives the gate wraps are never invoked, and that on-disk ``state.json``
is byte-identical before and after the call (C1.2).

A fifth test is the R4 inverse: it proves the ``log_event`` observability
call in ``_detect_and_handle_stalled_reviews`` (the "prompt_path missing"
stale-claim-recovery skip) is NOT accidentally gated -- dry_run must not
silence it, since none of the four ``log_event`` sites in that function were
in scope for W6 PR2 (issue #1264 comment 1, item R4).

That same test also asserts on ``events.db`` directly, via
``event_counts_by_kind()``, rather than relying only on a monkeypatched-spy
call count. A spy on ``charlie_work.write_gate.append_event`` is silent
under ``dry_run=True`` *by ``WriteGate``'s own construction* -- it never
reaches the module-level primitive regardless of whether the call site under
test is actually threaded through the gate or still calls the raw
``append_event`` directly (a different, unspied module attribute). Mutation-
tested: reverting stalled_review_reap.py's unclaimed-packet site to a raw
``append_event`` call left the spy-based assertion passing. The
``event_counts_by_kind`` assertion is paired with a ``dry_run=False``
positive control run against an identical seed in a separate state
directory, so "zero rows" is evidenced as a real zero, not an unreachable
query (per the "control must be positive by evidence" discipline).

A sixth test originally documented the R5 boundary deliberately left open by
PR2: ``_append_sweep_events`` (the batched "dispatched claim went stale"
path) called the RAW, unconverted ``append_event`` regardless of
``write_gate.dry_run``, because that batcher's conversion was PR3's
territory, not PR2's. Issue #1264 (W6 PR3, R5 completion) closes that
boundary: ``_append_sweep_events`` now requires a real ``WriteGate`` and
routes both its ``append_event`` calls through it, so the same PR-100
dead-reviewer-claim scenario that used to prove the raw call fired now
proves the opposite -- the sweep path is gated exactly like the direct
call sites the fifth test covers, and events.db gets zero rows for it under
dry_run=True (with a dry_run=False positive control, per this file's
established mutation-hardening discipline, proving the query itself is
decisive).

A seventh, cluster-level test drives ``OrchestratorApp.dispatch_reviews``
(Convention A) through its three ``write_gate``-threaded call sites with the
method's own ``if not self.dry_run`` short-circuits deliberately defeated
(``app.write_gate`` swapped to a ``dry_run=True`` gate on an otherwise
normal, non-dry-run app) -- proving the write_gate conversion protects each
site on its own, independent of the surrounding guards, which is the whole
point of the R7 conversion (workflow.py's distant-early-return site is
normally unreachable under ``self.dry_run=True`` in the first place, so a
plain ``dispatch_reviews(dry_run=True)`` call would never actually exercise
the converted code).

This test drives an actual reviewer-launch quota-hit (a launch error
matching the usage-limit signature, mirroring
test_charlie_work.py's ``test_dispatch_reviews_probe_failure_sets_reviewer_
quota_and_rolls_back``) rather than an empty candidate queue, and asserts
``result.data["quota_hit"] is True`` before checking the write-gate spies.
That assertion is load-bearing: workflow.py's ``dispatch_reviews`` early-
returns at the "no candidates in the review queue" check (~line 10981)
*before* the R7 block (~11464-11565) is ever reached, so an empty-queue
fixture (``prs=[]``) would let the spy-based assertions below pass
vacuously without ever exercising the code this test is supposed to cover.
Mutation-tested: with an empty-queue fixture and the R7 ``save_state`` site
reverted to a raw call, the test still passed -- the quota-hit fixture with
the ``quota_hit`` assertion is what makes this test actually load-bearing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import event_counts_by_kind
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.stalled_review_reap import (
    _detect_and_handle_stalled_reviews,
    _merge_on_write_save,
    _reap_orphaned_review_checkouts,
    _remove_review_checkout_with_warning,
)
from charlie_work.write_gate import WriteGate

from _fakes_github import FakeGitHub
from _review_fixtures import _dispatch_reviews_app, _write_review_packet


# Issue #1264 (W6 PR2): every WriteGate constructed below must carry THAT
# test's own state_file as state_path -- WriteGate.save_state() writes to
# self.state_path, not to whatever path a converted function was also given,
# so a gate built with a different path would silently write to the wrong
# file while assertions below keep reading the real state_file.
def _wg(state_file: Path, *, dry_run: bool = False) -> WriteGate:
    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def test_remove_review_checkout_with_warning_dry_run_true_appends_no_event(
    monkeypatch, tmp_path: Path
) -> None:
    """C1.2: a failed checkout removal under dry_run=True must not call the
    underlying append_event primitive (which dual-writes to events.db
    independent of whether the in-memory state it returns is ever persisted
    -- see write_gate.py's module docstring), and the returned state must not
    carry the event either."""
    repo_root = tmp_path / "repo"
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state = empty_state()

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: False
    )
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.append_event",
        lambda *a, **k: calls.append((a, k)) or {"BUG": "raw append_event was called"},
    )

    new_state, removed = _remove_review_checkout_with_warning(
        state,
        repo_root,
        reviews_dir,
        100,
        write_gate=_wg(tmp_path / "state.json", dry_run=True),
    )

    assert removed is False
    assert calls == []
    # The in-memory warning marker is set unconditionally (it happens before
    # the gated call, at stalled_review_reap.py:112-116) -- only the event
    # emission itself is gated.
    assert new_state["prs"]["100"]["review_checkout_removal_warned"] is True
    assert not any(
        e.get("kind") == "review_checkout_removal_failed" for e in new_state.get("events", [])
    )


def test_merge_on_write_save_dry_run_true_writes_nothing_to_disk(
    monkeypatch, tmp_path: Path
) -> None:
    """C1.2: a merge that DOES have real changes to persist (status flipped
    open -> merged) must not touch state.json on disk, and must not call the
    underlying save_state primitive, when write_gate.dry_run is True."""
    state_file = tmp_path / "state.json"
    initial = empty_state()
    initial["prs"]["100"] = {"number": 100, "status": "open"}
    save_state(state_file, initial)
    before_bytes = state_file.read_bytes()

    snapshot_prs = {"100": {"number": 100, "status": "open"}}
    changed_state = {**initial, "prs": {"100": {"number": 100, "status": "merged"}}}

    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.save_state",
        lambda *a, **k: calls.append((a, k)) or {"BUG": "raw save_state was called"},
    )

    _merge_on_write_save(
        state_file,
        changed_state,
        write_gate=_wg(state_file, dry_run=True),
        snapshot_prs=snapshot_prs,
        snapshot_reviewer_quota=initial.get("reviewer_quota"),
        snapshot_events=initial.get("events", []),
        event_ring_cap=2000,
    )

    assert calls == []
    assert state_file.read_bytes() == before_bytes


def test_detect_and_handle_stalled_reviews_dry_run_true_gates_direct_writes_but_not_log_events(
    monkeypatch, tmp_path: Path
) -> None:
    """C1.2 + R4 inverse: an "unclaimed packet" (PR 300) goes through one of
    the six DIRECT ``write_gate.append_event`` call sites this PR converted
    -- under dry_run=True that call must never reach the raw primitive, and
    state.json must be byte-identical before/after. A second PR (200, no
    ``prompt_path`` at all) hits the R4-exempt, deliberately-raw
    ``log_event("review_stale_claim_recovery_skipped", ...)`` observability
    site in the SAME pass -- that one must still fire under dry_run=True,
    proving PR2 did not accidentally gate it."""
    repo_root = tmp_path / "repo"
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state_file = tmp_path / "state.json"

    # PR 300: an unclaimed review packet, past the stale-claim timeout --
    # stalled_review_reap.py:1016-1052, the direct write_gate.append_event
    # ("review_dispatch_stalled", level="warning") site.
    prompt_path = tmp_path / "issue-300-review-prompt.md"
    prompt_path.write_text("review prompt", encoding="utf-8")

    state = empty_state()
    state["prs"]["300"] = {
        "number": 300,
        "status": "reviewing",
        "prompt_path": str(prompt_path),
    }
    # PR 200: reviewing, but with no prompt_path at all --
    # stalled_review_reap.py:911-936, the raw (unconverted) log_event site.
    state["prs"]["200"] = {"number": 200, "status": "reviewing"}
    save_state(state_file, state)
    before_bytes = state_file.read_bytes()

    # Far enough past the prompt's real (just-written) mtime to trip the
    # 5-minute _REVIEW_STALE_CLAIM_TIMEOUT_MINUTES.
    now = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=1)

    # Mutation-hardening positive control (see module docstring), run BEFORE
    # any monkeypatch touches charlie_work.write_gate.append_event: an
    # identical PR-300 seed in a separate state directory, detected with
    # dry_run=False, must produce a real review_dispatch_stalled row in
    # events.db. This proves the event_counts_by_kind query below is
    # decisive, not silently broken -- running it after the spy patches
    # would make it silent for the SAME reason a spy-based assertion is
    # (WriteGate.append_event's raw call is what the patch intercepts,
    # dry_run or not).
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / "reviews").mkdir()
    control_prompt_path = control_dir / "issue-300-review-prompt.md"
    control_prompt_path.write_text("review prompt", encoding="utf-8")
    control_state_file = control_dir / "state.json"
    control_state = empty_state()
    control_state["prs"]["300"] = {
        "number": 300,
        "status": "reviewing",
        "prompt_path": str(control_prompt_path),
    }
    save_state(control_state_file, control_state)
    _detect_and_handle_stalled_reviews(
        control_dir / "reviews",
        control_state_file,
        OrchestratorConfig(),
        control_dir / "repo",
        write_gate=_wg(control_state_file, dry_run=False),
        now=now,
    )
    control_counts = event_counts_by_kind(control_state_file)
    assert control_counts.get("review_dispatch_stalled", 0) >= 1, (
        "positive control: the same detection with dry_run=False must produce "
        "a real review_dispatch_stalled row, proving the query itself works"
    )

    append_calls: list[tuple[tuple, dict]] = []
    save_calls: list[tuple[tuple, dict]] = []
    log_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.append_event",
        lambda *a, **k: append_calls.append((a, k)) or {"BUG": "raw append_event was called"},
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.save_state",
        lambda *a, **k: save_calls.append((a, k)) or {"BUG": "raw save_state was called"},
    )
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.log_event",
        lambda *a, **k: log_calls.append((a, k)),
    )

    _detect_and_handle_stalled_reviews(
        reviews_dir,
        state_file,
        OrchestratorConfig(),
        repo_root,
        write_gate=_wg(state_file, dry_run=True),
        now=now,
    )

    assert append_calls == [], "PR 300's direct write_gate.append_event site must be gated"
    assert save_calls == [], "the merge-on-write save must be gated"
    assert state_file.read_bytes() == before_bytes

    assert log_calls, "PR 200's raw log_event site must still fire under dry_run=True (R4)"
    assert any(
        call_args[1] == "review_stale_claim_recovery_skipped" for call_args, _kwargs in log_calls
    )
    assert any(
        kwargs.get("level") == "warning" or "warning" in call_args
        for call_args, kwargs in log_calls
    )

    # Mutation-hardening (see module docstring): the spy assertions above are
    # silent under dry_run=True regardless of whether PR 300's site is
    # actually gated, because WriteGate.append_event() itself short-circuits
    # before ever calling the spied module attribute. Assert directly on
    # events.db instead -- if the site were still a raw, unconverted
    # append_event call, this row would exist here too.
    dry_run_counts = event_counts_by_kind(state_file)
    assert dry_run_counts.get("review_dispatch_stalled", 0) == 0, (
        "PR 300's direct write site must not reach events.db under dry_run=True"
    )


def test_detect_and_handle_stalled_reviews_dry_run_true_now_gates_the_sweep_events_path(
    monkeypatch, tmp_path: Path
) -> None:
    """R5 completion regression (issue #1264, W6 PR3): a stale
    ``review_dispatch_dispatched`` claim (a dead reviewer PID past the
    stale-claim timeout) is queued into the sweep's batched
    ``sweep_events`` list and flushed through ``_append_sweep_events``.
    Before PR3, that batcher called the RAW ``append_event`` primitive
    regardless of ``write_gate.dry_run`` (see the module docstring's sixth
    test note) -- PR3 threads a required ``write_gate`` through it, so this
    is now the same C1.2 shape as every other converted call site: the
    underlying primitive must never be reached, and events.db must show
    zero rows for it, under dry_run=True.

    A dry_run=False positive control (a byte-identical PR-100 seed in a
    separate state directory, run BEFORE any monkeypatch touches
    ``charlie_work.write_gate.append_event``) proves the events.db query
    below is decisive rather than silently broken -- the same discipline
    the fifth test above already applies to the direct write sites.
    """
    repo_root = tmp_path / "repo"
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    state_file = tmp_path / "state.json"

    def _seed(dead_pid_state_file: Path) -> None:
        seed_state = empty_state()
        seed_state["prs"]["100"] = {
            "number": 100,
            "review_dispatch_status": "review_dispatch_dispatched",
            "review_dispatched_at": "2026-08-01T00:00:00Z",
            "reviewer_pid": 99999,
            "reviewer_process_start_time": 1.0,
        }
        save_state(dead_pid_state_file, seed_state)

    _seed(state_file)
    before_bytes = state_file.read_bytes()

    now = datetime(2026, 8, 1, 0, 30, tzinfo=UTC)  # 30 min after dispatch: past the 5-min timeout

    monkeypatch.setattr("charlie_work.stalled_review_reap.is_pid_alive", lambda *a, **k: False)
    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: True
    )

    # Positive control -- run to completion with dry_run=False, in a separate
    # directory, BEFORE any monkeypatch touches write_gate.append_event.
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    (control_dir / "reviews").mkdir()
    control_state_file = control_dir / "state.json"
    _seed(control_state_file)
    _detect_and_handle_stalled_reviews(
        control_dir / "reviews",
        control_state_file,
        OrchestratorConfig(),
        control_dir / "repo",
        write_gate=_wg(control_state_file, dry_run=False),
        now=now,
    )
    control_counts = event_counts_by_kind(control_state_file)
    assert control_counts.get("review_dispatch_stalled", 0) >= 1, (
        "positive control: the same sweep with dry_run=False must produce a "
        "real review_dispatch_stalled row, proving the query itself works"
    )

    append_calls: list[tuple[tuple, dict]] = []
    save_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.append_event",
        lambda *a, **k: append_calls.append((a, k)) or {"BUG": "raw append_event was called"},
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.save_state",
        lambda *a, **k: save_calls.append((a, k)) or {"BUG": "raw save_state was called"},
    )

    _detect_and_handle_stalled_reviews(
        reviews_dir,
        state_file,
        OrchestratorConfig(),
        repo_root,
        write_gate=_wg(state_file, dry_run=True),
        now=now,
    )

    # R5 completion: the sweep's batched append_event call is now gated too.
    assert append_calls == [], (
        "the _append_sweep_events path must now be gated under dry_run=True "
        "-- issue #1264 W6 PR3 closes the R5 boundary PR2 left open"
    )
    assert save_calls == []
    assert state_file.read_bytes() == before_bytes

    dry_run_counts = event_counts_by_kind(state_file)
    assert dry_run_counts.get("review_dispatch_stalled", 0) == 0, (
        "the sweep's direct write site must not reach events.db under dry_run=True"
    )


def test_reap_orphaned_review_checkouts_dry_run_true_writes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    """C1.2: the same checkout-removal-failure scenario proven at #526
    (test_review_checkout_reap.py's warns-once-and-retries test) must,
    under dry_run=True, call neither the underlying append_event nor
    save_state primitive, and must leave state.json untouched on disk."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()
    checkout_dir = reviews_dir / "pr-100"
    checkout_dir.mkdir()
    state_file = tmp_path / "state.json"

    config = OrchestratorConfig()
    state = empty_state()
    state["prs"]["100"] = {
        "number": 100,
        "review_dispatch_status": "review_dispatch_dispatched",
        "review_dispatched_at": "2026-07-20T00:00:00Z",
        "reviewer_pid": 12345,
        "review_process_start_time": 1.0,
    }
    save_state(state_file, state)
    before_bytes = state_file.read_bytes()

    fake_gh = FakeGitHub()
    fake_gh.prs = [
        {
            "number": 100,
            "title": "Fix #1",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-100-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "body": "Closes #1",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]

    monkeypatch.setattr(
        "charlie_work.stalled_review_reap.remove_review_checkout", lambda *a, **k: False
    )
    monkeypatch.setattr("charlie_work.worker.WorkerView.is_alive", lambda self: False)

    append_calls: list[tuple[tuple, dict]] = []
    save_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.append_event",
        lambda *a, **k: append_calls.append((a, k)) or {"BUG": "raw append_event was called"},
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.save_state",
        lambda *a, **k: save_calls.append((a, k)) or {"BUG": "raw save_state was called"},
    )

    _reap_orphaned_review_checkouts(
        fake_gh,
        repo_root,
        reviews_dir,
        state_file,
        config,
        write_gate=_wg(state_file, dry_run=True),
    )

    assert append_calls == []
    assert save_calls == []
    assert state_file.read_bytes() == before_bytes


def test_dispatch_reviews_write_gate_dry_run_true_prevents_writes_even_past_the_dry_run_short_circuit(
    monkeypatch, tmp_path: Path
) -> None:
    """C2.d, cluster-level: dispatch_reviews's own ``if not self.dry_run``
    guards (workflow.py ~10699 and ~10786) already prevent every write when
    ``self.dry_run`` is True in the normal case -- a plain
    ``dispatch_reviews(dry_run=True)`` call never even reaches the
    write_gate-threaded call sites downstream. R7's point is that those call
    sites are no longer safe ONLY because of those guards. This test proves
    it directly: build a normal (non-dry-run) app so control flow reaches
    the sweep calls, but swap ``app.write_gate`` for a dry_run=True gate
    right before calling ``dispatch_reviews`` -- simulating "the outer guard
    didn't fire" without needing to actually break it. If write_gate
    threading had a gap (a raw primitive slipped back in, or a site never
    got converted), this is the test that would catch it; the
    guard-decorator flags alone cannot, since they never let control flow
    reach that far under dry_run in the first place.

    Issue #1264/#1329 (W6 PR4): the pre-launch claim write
    (``review_dispatch_claim`` append_event + the save_state that follows,
    workflow.py ~11363/11373) used to be a separate, raw, out-of-scope write
    that landed on disk regardless of ``write_gate`` -- PR2 converted only
    the R7 block below it. PR4 converts the claim write too, closing that
    exact gap, so this test's assertions now cover the claim write as well:
    under ``write_gate.dry_run=True`` the PR is never claimed at all, not
    merely rolled back by the R7 tail. ``quota_hit`` in the result payload
    (computed by the launch loop from ``fake_launch``'s error, independent
    of any write) is the reachability proof that survives this change."""
    # A real, selectable candidate PR (open, linked issue, matching review
    # packet) so review_queue() returns a non-empty candidate list --
    # otherwise dispatch_reviews early-returns at the "no candidates" check
    # (workflow.py ~10981) well before the R7 block, and this test's spy
    # assertions would pass without ever exercising R7 (see module
    # docstring's mutation-testing note).
    prs = [
        {
            "number": 100,
            "title": "Fix #10",
            "url": "https://example.test/pull/100",
            "headRefName": "agent/issue-10-fix",
            "baseRefName": "main",
            "headRefOid": "sha-100",
            "mergeStateStatus": "CLEAN",
            "body": "Closes #10",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _dispatch_reviews_app(tmp_path, prs=prs, dry_run=False)
    _write_review_packet(tmp_path, 100, "sha-100")

    # A launch error matching the usage-limit signature drives quota_hit=True
    # (workflow.py's launch loop breaks on it), which is what actually
    # reaches the R7 block: the quota_hit-specific append_event (~11542),
    # the unconditional review_dispatch append_event (~11556), and the final
    # save_state (~11565). Mirrors test_charlie_work.py's
    # test_dispatch_reviews_probe_failure_sets_reviewer_quota_and_rolls_back.
    def fake_launch(*args: object, **kwargs: object) -> ClaudeWorkerRecord:
        return ClaudeWorkerRecord(
            issue_number=kwargs.get("issue_number") or args[0],
            branch=kwargs.get("branch") or args[1],
            worktree_path="/fake/worktree",
            prompt_path="/fake/prompt.md",
            command=("claude", "-p", "--permission-mode", "plan"),
            pid=None,
            started_at="2026-07-20T12:00:00Z",
            log_path="/fake/log.log",
            error="usage limit exceeded",
            process_start_time=1.0,
        )

    monkeypatch.setattr("charlie_work.workflow.launch_claude_worker", fake_launch)

    append_calls: list[tuple[tuple, dict]] = []
    save_calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "charlie_work.write_gate.append_event",
        lambda *a, **k: append_calls.append((a, k)) or {"BUG": "raw append_event was called"},
    )
    monkeypatch.setattr(
        "charlie_work.write_gate.save_state",
        lambda *a, **k: save_calls.append((a, k)) or {"BUG": "raw save_state was called"},
    )

    # Defeat the outer dry_run short-circuits: app.dry_run stays False so
    # control flow reaches the write_gate-threaded call sites, but the gate
    # itself now says dry_run=True.
    app.write_gate = WriteGate(
        dry_run=True, state_path=app.paths.state_file, repo=app.repo_root.name
    )

    result = app.dispatch_reviews()

    # Load-bearing reachability proof (see module docstring): without this,
    # the spy assertions below would pass even if the R7 block were never
    # reached at all.
    assert result.data.get("quota_hit") is True, (
        "fixture must actually drive dispatch_reviews into the R7 block"
    )
    assert append_calls == []
    assert save_calls == []

    # Byte-identity now holds where it did not before PR4: the pre-launch
    # claim write (~11362-11373) is write_gate-threaded as of issue #1329,
    # so under write_gate.dry_run=True the claim itself never lands on
    # disk -- not merely the R7 block's downstream rollback. "100" must be
    # entirely absent from state["prs"], not present-with-a-stale-status.
    final_state = load_state(app.paths.state_file)
    assert "100" not in final_state["prs"], (
        "the pre-launch claim must not reach disk under write_gate.dry_run=True "
        "now that it is write_gate-threaded (issue #1329)"
    )

    # events.db corroboration: every event kind this call would emit on a
    # real (non-gated) run -- the claim, the quota-hit-specific kind, and
    # the unconditional review_dispatch kind -- is now write_gate-threaded,
    # so all three must be absent. quota_hit=True above is the sole
    # reachability proof (computed from fake_launch's error, independent of
    # any write), replacing the claim-event count this test used before the
    # claim write was converted.
    counts = event_counts_by_kind(app.paths.state_file)
    assert counts.get("review_dispatch_claim", 0) == 0
    assert counts.get("review_quota_exhausted", 0) == 0
    assert counts.get("review_dispatch", 0) == 0
