"""Seam tests for ``worker_fate``: liveness, stale evidence, persisted failures,
and the adapter fate profiles (architecture-deepening candidate 1)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from charlie_work.worker_fate import (
    Crashed,
    EvidenceSource,
    FailureEvidence,
    Live,
    OutcomeEvidence,
    StaleReason,
    Stranded,
    Throttled,
    is_alive,
    persist_failure,
    persisted_failure,
    resolve_fate,
    stale_evidence_events,
    stale_evidence_key,
    throttle_failure,
)
from charlie_work.worker import WorkerHealth
from _worker_fate_fixtures import (
    AFTER,
    BEFORE,
    DISPATCHED,
    NOW,
    _branch,
    _evidence,
    _outcome,
    _terminal,
)


# --------------------------------------------------------------------------
# is_alive: process_utils.is_pid_alive stays the primitive; the only
# additions are the ``pid is None`` and ``pid <= 0`` cases (wf-r2-s2 folded
# the deleted per-adapter wrappers' ``<= 0`` guard in here).
# --------------------------------------------------------------------------


def test_is_alive_none_pid_is_never_alive() -> None:
    assert is_alive(None, None) is False


def test_is_alive_nonpositive_pid_is_never_alive_and_never_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(pid: int, expected_start_time: float | None = None) -> bool:
        raise AssertionError("a non-positive pid must not reach the process probe")

    monkeypatch.setattr("charlie_work.process_utils.is_pid_alive", _boom)
    assert is_alive(0, None) is False
    assert is_alive(-1, 123.0) is False


def test_is_alive_delegates_to_process_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, float | None]] = []

    def _fake(pid: int, expected_start_time: float | None = None) -> bool:
        calls.append((pid, expected_start_time))
        return True

    monkeypatch.setattr("charlie_work.process_utils.is_pid_alive", _fake)
    assert is_alive(1234, 5678.0) is True
    assert calls == [(1234, 5678.0)]


def test_n6_build_profiles_completeness_check_survives_dash_o(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """N6 (wf-review-opus.md): the WORKER_HARNESSES/profile-registry
    completeness guard must be an explicit ``raise``, not a bare
    ``assert`` -- ``python -O`` strips ``assert`` statements, which would
    silently disable this check exactly when a harness/profile drift
    needs to be caught. Force a mismatch (a bogus extra harness) and
    confirm ``_build_profiles`` still raises regardless.
    """
    from charlie_work import adapter_fate_profile

    monkeypatch.setattr(adapter_fate_profile, "_PROFILES", None)
    monkeypatch.setattr(
        "charlie_work.harnesses.WORKER_HARNESSES",
        frozenset({"devin-shell", "claude-code", "api", "command", "manual", "bogus-harness"}),
    )
    with pytest.raises(AssertionError, match="WORKER_HARNESSES"):
        adapter_fate_profile._build_profiles()


def test_n7_outcome_evidence_raw_is_immutable_against_caller_mutation() -> None:
    """N7 (wf-review-opus.md): `OutcomeEvidence.raw` must not be the
    caller's own live dict -- freezing the dataclass does not freeze
    `raw`'s contents, so a caller mutating its own dict after
    construction would otherwise be visible through the "frozen"
    evidence. `raw` must also reject direct mutation itself.
    """
    caller_dict = {"push_succeeded": True, "pr_created": False, "head_sha": "abc123"}
    evidence = OutcomeEvidence(
        source=EvidenceSource.WORKTREE,
        written_at=None,
        outcome=None,
        push_succeeded=True,
        pr_created=False,
        head_sha="abc123",
        raw=caller_dict,
    )

    # Mutating the caller's own dict afterward must not leak through.
    caller_dict["head_sha"] = "mutated"
    assert evidence.raw["head_sha"] == "abc123"

    # `raw` itself must refuse direct mutation.
    with pytest.raises(TypeError):
        evidence.raw["head_sha"] = "mutated"  # type: ignore[index]


# --------------------------------------------------------------------------
# B6 (wf-review-opus.md) / design doc §5: the stale-evidence event.
# --------------------------------------------------------------------------


def test_stale_evidence_events_builds_one_event_per_stale_candidate() -> None:
    stale_terminal_outcome = _outcome(
        source=EvidenceSource.TERMINAL, written_at=BEFORE, outcome="blocked"
    )
    terminal = _terminal(ended_at=AFTER, exit_code=None, outcome=stale_terminal_outcome)
    fresh_worktree_outcome = _outcome(
        source=EvidenceSource.WORKTREE, written_at=AFTER, outcome="blocked"
    )
    fate = resolve_fate(
        _evidence(terminal=terminal, worktree_outcome=fresh_worktree_outcome), now=NOW
    )
    assert fate.basis.stale, "fixture must actually produce stale evidence"

    events = stale_evidence_events({}, fate)

    assert len(events) == len(fate.basis.stale)
    kind, payload = events[0]
    assert kind == "worker_evidence_stale"
    assert payload["issue_number"] == fate.basis.issue_number
    assert payload["source"] == "terminal"
    assert payload["reason"] == "older_than_dispatch"
    assert payload["written_at"] == BEFORE.isoformat()
    assert payload["evidence_head"] is None
    assert payload["live_head"] is None


def test_stale_evidence_events_reads_dispatched_at_and_adapter_from_entry() -> None:
    stale_outcome = _outcome(written_at=BEFORE, outcome="blocked")
    fate = resolve_fate(_evidence(worktree_outcome=stale_outcome), now=NOW)
    assert fate.basis.stale

    entry = {"dispatched_at": "2026-01-01T12:00:00Z", "adapter": "devin"}
    events = stale_evidence_events(entry, fate)

    assert len(events) == 1
    _, payload = events[0]
    assert payload["dispatched_at"] == "2026-01-01T12:00:00Z"
    assert payload["adapter"] == "devin"


def test_stale_evidence_events_empty_when_nothing_is_stale() -> None:
    fate = resolve_fate(_evidence(pid_alive=True, health=WorkerHealth.HEALTHY), now=NOW)
    assert fate.basis.stale == ()
    assert stale_evidence_events({}, fate) == []


def test_stale_evidence_events_dedups_against_already_reported_keys() -> None:
    stale_outcome = _outcome(written_at=BEFORE, outcome="blocked")
    fate = resolve_fate(_evidence(worktree_outcome=stale_outcome), now=NOW)
    assert len(fate.basis.stale) == 1
    key = stale_evidence_key(fate.basis.stale[0])

    # Not yet reported: one event.
    assert len(stale_evidence_events({}, fate)) == 1

    # Already reported: deduped to nothing, same fate, same stale evidence.
    entry = {"stale_evidence_reported": [key]}
    assert stale_evidence_events(entry, fate) == []


def test_stale_evidence_key_distinguishes_source_and_written_at() -> None:
    a = _outcome(source=EvidenceSource.TERMINAL, written_at=BEFORE, outcome="blocked")
    b = _outcome(source=EvidenceSource.WORKTREE, written_at=BEFORE, outcome="blocked")
    fate_a = resolve_fate(
        _evidence(terminal=_terminal(ended_at=AFTER, exit_code=None, outcome=a)), now=NOW
    )
    fate_b = resolve_fate(_evidence(worktree_outcome=b), now=NOW)
    assert fate_a.basis.stale and fate_b.basis.stale
    key_a = stale_evidence_key(fate_a.basis.stale[0])
    key_b = stale_evidence_key(fate_b.basis.stale[0])
    assert key_a != key_b


# --------------------------------------------------------------------------
# Rule 6, write side: ``persist_failure`` is the single primitive; read side:
# the persisted stamp is fed back as evidence so ``Throttled`` is reachable.
# --------------------------------------------------------------------------

_STATE = {
    "throttled_until": None,
    "issues": {"7": {"status": "dispatched", "branch_name": "b"}, "8": {"status": "queued"}},
}


def test_failure_evidence_from_classification_parses_classifier_iso() -> None:
    failure = FailureEvidence.from_classification(
        "rate_limited", "2026-01-01T12:15:00Z", fresh=True
    )
    assert failure.kind == "rate_limited"
    assert failure.throttled_until == datetime(2026, 1, 1, 12, 15, tzinfo=UTC)
    assert failure.fresh is True
    assert (
        FailureEvidence.from_classification("stalled", None, fresh=False).throttled_until is None
    )


def test_persist_failure_writes_cooldown_and_kind_together_without_mutating() -> None:
    before = {"throttled_until": None, "issues": {"7": {"status": "dispatched"}}}
    failure = FailureEvidence.from_classification(
        "rate_limited", "2026-01-01T12:15:00Z", fresh=True
    )

    new = persist_failure(before, 7, failure, adapter_kind="devin", now=NOW, source="test")

    assert new["throttled_until"] == "2026-01-01T12:15:00Z"  # round-trips, never recomputed
    assert new["throttle_reason"] == "rate_limited"
    assert new["throttle_adapter_kind"] == "devin"
    entry = new["issues"]["7"]
    assert entry["status"] == "dispatched"
    assert entry["dead_worker_failure_kind"] == "rate_limited"
    assert entry["dead_worker_failure_classified_at"] == "2026-01-01T12:10:00Z"
    assert before == {"throttled_until": None, "issues": {"7": {"status": "dispatched"}}}


def test_persist_failure_without_cooldown_leaves_throttled_until_alone() -> None:
    state = {"throttled_until": "2030-01-01T00:00:00Z", "issues": {"7": {"status": "dispatched"}}}
    failure = FailureEvidence(kind="stalled", throttled_until=None, fresh=True)

    new = persist_failure(state, 7, failure, adapter_kind="claude-code", now=NOW, source="test")

    assert new["throttled_until"] == "2030-01-01T00:00:00Z"
    assert new["issues"]["7"]["dead_worker_failure_kind"] == "stalled"


def test_persist_failure_never_invents_an_issue_entry() -> None:
    failure = FailureEvidence(kind="rate_limited", throttled_until=None, fresh=True)
    new = persist_failure(_STATE, 999, failure, adapter_kind=None, now=NOW, source="test")
    assert "999" not in new["issues"]


def test_persist_failure_with_no_kind_stamps_nothing() -> None:
    failure = FailureEvidence(kind=None, throttled_until=None, fresh=True)
    assert persist_failure(_STATE, 7, failure, adapter_kind=None, now=NOW, source="test") == _STATE


def test_persisted_failure_round_trips_the_stamp_and_feeds_evidence() -> None:
    failure = FailureEvidence(kind="rate_limited", throttled_until=None, fresh=True)
    entry = persist_failure(_STATE, 7, failure, adapter_kind=None, now=NOW, source="test")[
        "issues"
    ]["7"]

    persisted = persisted_failure(entry)

    assert persisted.kind == "rate_limited"
    assert persisted.is_throttle is True
    assert persisted.classified_at == NOW
    assert persisted.as_evidence() == FailureEvidence(
        kind="rate_limited", throttled_until=None, fresh=False
    )
    assert persisted_failure({}).as_evidence() is None
    assert persisted_failure({}).classified_at is None


def test_persisted_throttle_makes_throttled_reachable_for_a_dead_worker() -> None:
    entry = {"dead_worker_failure_kind": "rate_limited"}
    fate = resolve_fate(_evidence(failure=persisted_failure(entry).as_evidence()), now=NOW)
    assert isinstance(fate, Throttled)
    assert fate.basis.rule == "R6"
    assert throttle_failure(fate) is fate.failure


def test_throttle_failure_reads_throttle_kinds_off_carrying_variants_only() -> None:
    throttle = FailureEvidence(kind="rate_limited", throttled_until=None, fresh=False)
    other = FailureEvidence(kind="stalled", throttled_until=None, fresh=False)

    crashed = resolve_fate(
        _evidence(
            failure=throttle,
            worktree_outcome=_outcome(outcome="completed", push_succeeded=False),
        ),
        now=NOW,
    )
    assert isinstance(crashed, Crashed)
    assert throttle_failure(crashed) is throttle

    stranded = resolve_fate(_evidence(failure=throttle, branch=_branch(unpushed=2)), now=NOW)
    assert isinstance(stranded, Stranded)
    assert throttle_failure(stranded) is throttle

    assert throttle_failure(resolve_fate(_evidence(failure=other), now=NOW)) is None
    assert throttle_failure(resolve_fate(_evidence(), now=NOW)) is None
    live = resolve_fate(_evidence(pid_alive=True, failure=throttle), now=NOW)
    assert isinstance(live, Live)
    assert throttle_failure(live) is None
    assert throttle_failure(None) is None


def _gate(state_file, *, dry_run: bool = False):
    from charlie_work.write_gate import WriteGate

    return WriteGate(dry_run=dry_run, state_path=state_file, repo="charlie-work")


def test_report_stale_evidence_without_a_state_entry_emits_and_persists_nothing(
    tmp_path,
) -> None:
    """B6 (wf-r2-s6): an issue with no state entry still emits (nothing to
    dedup against) and does not invent an entry; an empty mapping is a no-op."""
    import json

    from charlie_work.instrumentation import query_events
    from charlie_work.state import load_state
    from charlie_work.worker_fate import report_stale_evidence

    state_file = tmp_path / "state.json"
    fate = resolve_fate(
        _evidence(worktree_outcome=_outcome(written_at=BEFORE, outcome="blocked")), now=NOW
    )
    assert fate.basis.stale

    report_stale_evidence(state_file, {}, write_gate=_gate(state_file))
    assert query_events(state_file, kind="worker_evidence_stale") == []

    report_stale_evidence(
        state_file, {fate.basis.issue_number: [fate]}, write_gate=_gate(state_file)
    )

    events = query_events(state_file, kind="worker_evidence_stale")
    assert len(events) == 1
    raw = events[0]["payload"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    # No entry to read ``dispatched_at`` from: falls back to the fate basis.
    assert payload["dispatched_at"] == DISPATCHED.isoformat()
    assert str(fate.basis.issue_number) not in load_state(state_file)["issues"]


def test_report_stale_evidence_dry_run_writes_nothing(tmp_path) -> None:
    """Review wf-r2-1 #2: ``write_gate`` is a required keyword and, under dry-run, short-circuits
    every local write (events.db and the dedup marker)."""
    from charlie_work.instrumentation import query_events
    from charlie_work.state import load_state, save_state
    from charlie_work.worker_fate import report_stale_evidence

    fate = resolve_fate(
        _evidence(worktree_outcome=_outcome(written_at=BEFORE, outcome="blocked")), now=NOW
    )
    state_file = tmp_path / "state.json"
    state = load_state(state_file)
    state["issues"][str(fate.basis.issue_number)] = {"status": "dispatched"}
    save_state(state_file, state)

    report_stale_evidence(
        state_file, {fate.basis.issue_number: [fate]}, write_gate=_gate(state_file, dry_run=True)
    )

    assert query_events(state_file, kind="worker_evidence_stale") == []
    entry = load_state(state_file)["issues"][str(fate.basis.issue_number)]
    assert "stale_evidence_reported" not in entry


def test_report_stale_evidence_merges_every_fate_for_an_issue(tmp_path) -> None:
    """Review wf-r2-1 #1: the with-PR sweep resolves the same outcome twice --
    once against the live head (``HEAD_MISMATCH``), once without it (no
    stale). Last-write-wins collection dropped the mismatch; collecting every
    fate reports it exactly once."""
    import json

    from charlie_work.instrumentation import query_events
    from charlie_work.worker_fate import collect_fate, report_stale_evidence

    outcome = _outcome(written_at=AFTER, push_succeeded=True, head_sha="a" * 40)
    with_live_head = resolve_fate(
        _evidence(
            worktree_outcome=outcome,
            branch=_branch(remote_head_sha="b" * 40),
        ),
        now=NOW,
    )
    without_live_head = resolve_fate(_evidence(worktree_outcome=outcome), now=NOW)
    assert [s.reason for s in with_live_head.basis.stale] == [StaleReason.HEAD_MISMATCH]
    assert without_live_head.basis.stale == ()

    collected: dict[int, list] = {}
    collect_fate(collected, with_live_head)
    collect_fate(collected, without_live_head)
    assert len(collected[with_live_head.basis.issue_number]) == 2

    state_file = tmp_path / "state.json"
    report_stale_evidence(state_file, collected, write_gate=_gate(state_file))

    events = query_events(state_file, kind="worker_evidence_stale")
    assert len(events) == 1
    raw = events[0]["payload"]
    payload = json.loads(raw) if isinstance(raw, str) else raw
    assert payload["reason"] == "head_mismatch"


def test_stale_evidence_events_emit_a_shared_key_once() -> None:
    """N-d: a legacy terminal record yields two ``StaleEvidence`` with one
    key; the events list must not carry the same key twice."""
    from charlie_work.worker_fate import StaleEvidence, stale_evidence_events, stale_evidence_key

    base = resolve_fate(_evidence(), now=NOW)
    first = StaleEvidence(
        source=EvidenceSource.TERMINAL,
        reason=StaleReason.OLDER_THAN_DISPATCH,
        written_at=BEFORE,
        evidence_head=None,
        live_head=None,
    )
    twin = StaleEvidence(
        source=EvidenceSource.TERMINAL,
        reason=StaleReason.OLDER_THAN_DISPATCH,
        written_at=BEFORE,
        evidence_head=None,
        live_head=None,
    )
    assert stale_evidence_key(first) == stale_evidence_key(twin)
    from dataclasses import replace

    fate = replace(base, basis=replace(base.basis, stale=(first, twin)))

    assert len(stale_evidence_events({}, fate)) == 1


@pytest.mark.parametrize(
    ("view_kind", "module", "attr", "extra_kwargs"),
    [
        (
            "devin",
            "charlie_work.devin_shell",
            "update_session_record_with_failure_classification",
            {},
        ),
        (
            "claude-code",
            "charlie_work.claude_code",
            "update_worker_record_with_failure_classification",
            {"adapter_kind": "claude-code"},
        ),
        (
            "api",
            "charlie_work.claude_code",
            "update_worker_record_with_failure_classification",
            {"adapter_kind": "api"},
        ),
    ],
)
@pytest.mark.parametrize("registry_prebuilt", [False, True])
def test_profile_record_failure_is_late_bound(
    monkeypatch, view_kind, module, attr, extra_kwargs, registry_prebuilt
) -> None:
    """Review wf-r2-1 #3: the profile registry is cached per process, so a
    ``record_failure`` bound at build time made a test's ``patch`` of the
    adapter writer either vacuous (registry already built) or -- when the patch
    was live during the first ``profile_for`` -- leaked a Mock into every later
    test. The writer is resolved at call time: the patch reaches the call in
    both orders, and nothing outlives it."""
    from unittest.mock import MagicMock, patch

    import charlie_work.adapter_fate_profile as worker_fate

    monkeypatch.setattr(worker_fate, "_PROFILES", None)
    if registry_prebuilt:
        worker_fate.profile_for(view_kind)

    with patch(f"{module}.{attr}", return_value=("patched", None)) as mock_writer:
        profile = worker_fate.profile_for(view_kind)
        assert profile is not None and profile.record_failure is not None
        result = profile.record_failure("sessions", 7, fallback_kind="stalled")

    assert result == ("patched", None)
    mock_writer.assert_called_once_with("sessions", 7, fallback_kind="stalled", **extra_kwargs)
    survivor = worker_fate.profile_for(view_kind)
    assert survivor is not None
    assert not isinstance(survivor.record_failure, MagicMock)
