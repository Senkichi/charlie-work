"""``_build_fleet_attention_digest`` / ``_filter_fleet_health_transitions`` stateful digest tests.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
    _repair_payload,
)
from charlie_work.fleet_dispatch import (
    _build_fleet_attention_digest,
    _extract_attention_events,
)
from charlie_work.workflow import CommandResult


def test_build_fleet_attention_digest_maps_escalated_label_repair_error() -> None:
    """Issue #1088: the event survives the full digest-rendering pipeline as ERROR.

    This is the property that matters most: it is not enough for the collector
    to emit the right dict, since ``_build_fleet_attention_digest`` has an
    explicit branch per event type and a generic fallback for anything else.
    Issue #590 made that fallback render (instead of silently dropping)
    unbranched types, keyed off an ``_error``-suffixed type name mapping to
    ``health="ERROR"``. This test pins that ``escalated_label_repair_error``
    actually rides that fallback through to a real ``AttentionEntry`` end to
    end, rather than trusting the type-name convention by inspection.
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[901])},
    )
    events = _extract_attention_events("owner/repo", result)

    digest = _build_fleet_attention_digest(events)

    repair_entries = [e for e in digest.transitions if e.issue_number == 901]
    assert len(repair_entries) == 1
    entry = repair_entries[0]
    assert entry.health == "ERROR"
    assert entry.adapter_kind == "owner/repo"
    assert "901" in (entry.last_log_line or "")


def test_build_fleet_attention_digest_maps_review_verdict_events() -> None:
    """Issue #507: review verdict events map to OK/ERROR attention entries."""
    events = [
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_recorded",
            "issue_number": 10,
            "pr": 100,
            "decision": "approved",
        },
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_missed",
            "issue_number": 11,
            "pr": 101,
            "reason": "no parseable verdict",
        },
    ]

    digest = _build_fleet_attention_digest(events)

    by_health = {e.health: e for e in digest.transitions}
    assert by_health["OK"].last_log_line == "approved recorded for PR 100"
    assert by_health["OK"].issue_number == 10
    assert by_health["ERROR"].last_log_line == "no parseable verdict"
    assert by_health["ERROR"].issue_number == 11


def test_build_fleet_attention_digest_observed_repo_keys_reconciles_stale_error(
    tmp_path: Path,
) -> None:
    """Issue #817 item 2: ``_build_fleet_attention_digest`` forwards
    ``observed_repo_keys`` through to ``_filter_fleet_health_transitions``,
    so ``fleet_loop``'s per-pass reconciliation actually reaches the
    baseline sidecar rather than being silently dropped somewhere in
    between.
    """
    from charlie_work.fleet_dispatch import (
        _build_fleet_attention_digest,
        _fleet_health_state_path,
        _load_fleet_health_state,
    )

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo",
            "type": "error",
            "issue_number": 100,
            "error": "PR #100 review failed: timeout",
        }
    ]
    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert _load_fleet_health_state(state_file) == {"owner/repo:100": "ERROR"}

    # Next pass: issue #100 is healthy again (no error event for it), but
    # the repo's lane ran to completion -- observed_repo_keys reconciles the
    # stale key away.
    digest2 = _build_fleet_attention_digest(
        [], state_file=state_file, observed_repo_keys=frozenset({"owner/repo"})
    )
    assert digest2.transitions == ()
    assert _load_fleet_health_state(state_file) == {}


def test_build_fleet_attention_digest_stateful_keeps_review_verdict_heartbeat(
    tmp_path: Path,
) -> None:
    """PR #669 review: occurrence-style events must not be deduped by the
    stateful filter. ``review_verdict_recorded`` carries a constant ``OK``
    health string by construction; if the cross-pass dedup applied to it, the
    recorded-verdict heartbeat would collapse after the first occurrence per
    issue/repo and a silent 0%-recording-rate regression would go invisible.
    The fleet path in production goes through ``_build_fleet_attention_digest``
    with ``state_file`` set, so this locks in the filtered path (not just the
    unfiltered one covered by ``test_build_fleet_attention_digest_maps_review_verdict_events``).
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_recorded",
            "issue_number": 10,
            "pr": 100,
            "decision": "approved",
        }
    ]

    # Two passes, identical recorded-verdict event — both must emit. The
    # heartbeat stays visible; the baseline sidecar is not consulted for
    # occurrence-style entries.
    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert digest1.transitions[0].health == "OK"
    assert digest1.transitions[0].last_log_line == "approved recorded for PR 100"

    digest2 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest2.transitions) == 1
    assert digest2.transitions[0].health == "OK"


def test_build_fleet_attention_digest_stateful_keeps_review_verdict_missed_every_pass(
    tmp_path: Path,
) -> None:
    """PR #669 review: ``review_verdict_missed`` has health ``ERROR`` (the same
    string as a persistent worker error) but is an occurrence-style event. The
    dedup must be scoped by entry type, not by health string, or repeat
    missed-verdict signals would be silently dropped after the first one per
    issue/repo. Locks in the exception for the ERROR-health occurrence case.
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo1",
            "type": "review_verdict_missed",
            "issue_number": 11,
            "pr": 101,
            "reason": "no parseable verdict",
        }
    ]

    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert digest1.transitions[0].health == "ERROR"
    assert digest1.transitions[0].last_log_line == "no parseable verdict"

    # Same missed verdict next pass — still emits despite health == "ERROR",
    # because the event type is occurrence-style and bypasses the baseline.
    digest2 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest2.transitions) == 1
    assert digest2.transitions[0].health == "ERROR"


def test_build_fleet_attention_digest_stateful_mixed_persistent_and_occurrence(
    tmp_path: Path,
) -> None:
    """PR #669 review: a persistent ``error`` and an occurrence
    ``review_verdict_recorded`` for the same issue/repo share the dedup key
    ``adapter_kind:issue_number``. The persistent entry must still be deduped
    across passes while the occurrence entry keeps emitting, and the occurrence
    entry must not poison the persistent baseline (so a later persistent
    transition is not masked).
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    persistent_event = {
        "repo_key": "owner/repo",
        "type": "error",
        "issue_number": 42,
        "error": "launch failed: OSError",
    }
    occurrence_event = {
        "repo_key": "owner/repo",
        "type": "review_verdict_recorded",
        "issue_number": 42,
        "pr": 420,
        "decision": "approved",
    }

    # Pass 1: persistent null->ERROR emits; occurrence OK emits alongside.
    digest1 = _build_fleet_attention_digest(
        [persistent_event, occurrence_event], state_file=state_file
    )
    healths = [t.health for t in digest1.transitions]
    assert healths == ["ERROR", "OK"]

    # Pass 2: persistent ERROR->ERROR is deduped away; occurrence OK still emits.
    digest2 = _build_fleet_attention_digest(
        [persistent_event, occurrence_event], state_file=state_file
    )
    assert [t.health for t in digest2.transitions] == ["OK"]

    # Pass 3: persistent ERROR->STALLED is a real transition and must emit even
    # though the occurrence event sat between them every pass — the occurrence
    # entry never wrote the shared baseline key.
    stalled_event = {
        "repo_key": "owner/repo",
        "type": "stalled",
        "issue_number": 42,
        "reason": "no progress for 30m",
    }
    digest3 = _build_fleet_attention_digest(
        [stalled_event, occurrence_event], state_file=state_file
    )
    healths3 = [t.health for t in digest3.transitions]
    assert healths3 == ["STALLED", "OK"]
    assert digest3.transitions[0].previous_health == "ERROR"


def test_build_fleet_attention_digest_stateful_skips_repeats(tmp_path: Path) -> None:
    """Issue #554: _build_fleet_attention_digest with state_file emits only on real transitions."""
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest, _fleet_health_state_path

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    events = [
        {
            "repo_key": "owner/repo",
            "type": "error",
            "issue_number": 100,
            "error": "PR #100 review failed: timeout",
        }
    ]

    # First pass: null -> ERROR transition emits.
    digest1 = _build_fleet_attention_digest(events, state_file=state_file)
    assert len(digest1.transitions) == 1
    assert digest1.transitions[0].health == "ERROR"
    assert digest1.transitions[0].previous_health is None

    # Second pass: same ERROR, no transition — digest is empty.
    digest2 = _build_fleet_attention_digest(events, state_file=state_file)
    assert digest2.transitions == ()


def test_filter_fleet_health_transitions_dedups_repeated_error(tmp_path: Path) -> None:
    """Issue #554: a persistent ERROR must not re-fire with previous_health:null every pass."""
    from dataclasses import replace

    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    entry = AttentionEntry(
        issue_number=42,
        adapter_kind="owner/repo",
        health="ERROR",
        previous_health=None,
        last_log_line="failed to launch claude: OSError",
        pid=None,
    )

    # Pass 1: null -> ERROR is a real transition; emits with previous_health=None.
    first = _filter_fleet_health_transitions([entry], state_file)
    assert len(first) == 1
    assert first[0].health == "ERROR"
    assert first[0].previous_health is None

    # Pass 2: ERROR -> ERROR is not a transition; nothing emits.
    second = _filter_fleet_health_transitions([entry], state_file)
    assert second == []

    # Pass 3: ERROR -> STALLED is a real transition; emits with previous_health=ERROR.
    stalled = replace(entry, health="STALLED")
    third = _filter_fleet_health_transitions([stalled], state_file)
    assert len(third) == 1
    assert third[0].health == "STALLED"
    assert third[0].previous_health == "ERROR"


def test_filter_fleet_health_transitions_leaves_unobserved_repo_keys_untouched(
    tmp_path: Path,
) -> None:
    """A repo whose lane did NOT run this pass (missing repo_root, lock held,
    unhandled exception) must not have its stale keys reconciled away --
    absence of a check is not evidence of health. Only keys under repos
    present in ``observed_repo_keys`` are eligible for clearing.
    """
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
        _load_fleet_health_state,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    entry = AttentionEntry(
        issue_number=7,
        adapter_kind="owner/skipped-repo",
        health="ERROR",
        previous_health=None,
        last_log_line="stalled",
        pid=None,
    )
    first = _filter_fleet_health_transitions([entry], state_file)
    assert len(first) == 1

    # A different repo's lane ran this pass; "owner/skipped-repo" did not.
    second = _filter_fleet_health_transitions(
        [], state_file, observed_repo_keys=frozenset({"owner/other-repo"})
    )
    assert second == []
    assert _load_fleet_health_state(state_file) == {"owner/skipped-repo:7": "ERROR"}


def test_filter_fleet_health_transitions_reconciles_stale_key_when_repo_observed(
    tmp_path: Path,
) -> None:
    """Issue #817 item 2/AC5: a stale ERROR baseline for an issue that is
    healthy again (produces no unhealthy event this pass) is cleared once
    its repo's lane is confirmed observed, instead of latching forever.

    This is also the drain mechanism for the pre-existing 34 latched keys
    (issue #817's diagnosis): each key's repo needs exactly one observed
    pass with no matching unhealthy entry to clear it, after which the next
    real failure emits with ``previous_health: null`` again instead of
    staying permanently suppressed by a baseline that could never move
    except deeper into an unhealthy value.
    """
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
        _load_fleet_health_state,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    entry = AttentionEntry(
        issue_number=42,
        adapter_kind="owner/repo",
        health="ERROR",
        previous_health=None,
        last_log_line="failed to launch claude: OSError",
        pid=None,
    )

    # Pass 1: latch ERROR.
    first = _filter_fleet_health_transitions([entry], state_file)
    assert len(first) == 1
    assert _load_fleet_health_state(state_file) == {"owner/repo:42": "ERROR"}

    # Pass 2: issue #42 recovered -- no unhealthy entry for it this pass, but
    # its repo's lane still ran to completion (observed_repo_keys includes
    # "owner/repo"). The stale key must be cleared, not re-emitted as a
    # synthetic recovery entry.
    second = _filter_fleet_health_transitions(
        [], state_file, observed_repo_keys=frozenset({"owner/repo"})
    )
    assert second == []
    assert _load_fleet_health_state(state_file) == {}

    # Pass 3: the same issue fails again. Because the baseline was cleared,
    # this is a fresh null -> ERROR transition, not a suppressed repeat.
    third = _filter_fleet_health_transitions([entry], state_file)
    assert len(third) == 1
    assert third[0].previous_health is None


def test_filter_fleet_health_transitions_self_deploy_key_survives_repo_reconciliation(
    tmp_path: Path,
) -> None:
    """Item 1's ``self-deploy:-1`` baseline key and item 2's per-repo
    reconciliation share the same sidecar file within one supervisor pass
    (self_deploy emits first, then fleet_loop's digest reconciles). The
    ``self-deploy`` adapter_kind is a fixed literal, never a real repo's
    ``name_with_owner``, so it can never appear in ``observed_repo_keys`` and
    must never be cleared by issue-health reconciliation -- confirmed here by
    a test rather than left as an unverified by-construction claim.
    """
    from charlie_work.fleet_dispatch import (
        _fleet_health_state_path,
        _filter_fleet_health_transitions,
        _load_fleet_health_state,
    )
    from charlie_work.notify import AttentionEntry

    state_file = _fleet_health_state_path(str(tmp_path / "fleet"))
    self_deploy_entry = AttentionEntry(
        issue_number=-1,
        adapter_kind="self-deploy",
        health="ERROR",
        previous_health=None,
        last_log_line="pull failed",
        pid=None,
    )
    _filter_fleet_health_transitions([self_deploy_entry], state_file)
    assert _load_fleet_health_state(state_file) == {"self-deploy:-1": "ERROR"}

    # A real repo's lane runs and reconciles this pass; self-deploy's key
    # must not be touched even though no self-deploy entry was emitted.
    result = _filter_fleet_health_transitions(
        [], state_file, observed_repo_keys=frozenset({"owner/repo"})
    )
    assert result == []
    assert _load_fleet_health_state(state_file) == {"self-deploy:-1": "ERROR"}


def test_digest_renders_event_types_that_have_no_explicit_branch() -> None:
    """An unmapped event type must not vanish.

    The mapper was an if/elif chain with no else, so the prologue's own
    ``runner_allocation_error`` was emitted correctly and then dropped on the
    floor — the digest could not show the one signal that mattered for #590.
    """
    events = [
        {"repo_key": "fleet", "type": "runner_allocation_error", "error": "managed_root missing"},
        {
            "repo_key": "fleet",
            "type": "runner_allocation_skipped",
            "reason": "no usable repo root",
        },
    ]

    digest = _build_fleet_attention_digest(events)

    assert len(digest.transitions) == 2
    rendered = {entry.health: entry.last_log_line for entry in digest.transitions}
    assert rendered["ERROR"] == "managed_root missing"
    assert rendered["INFO"] == "no usable repo root"


def test_digest_stays_quiet_on_a_converged_allocation_pass() -> None:
    """A healthy rebalance must not put an entry in every 5-minute digest.

    The prologue emits `runner_allocation` when anything moved OR any note was
    produced, and the notes include standing advisory conditions that persist as long
    as the condition does. Verified against this host's events.db: every recorded pass
    carried such a note while moving zero slots. So routing this event through the
    generic fallback would render a near-identical attention entry on every pass.

    Precise about what this does and does not assert. It pins the *entry*: none is
    rendered for a converged pass. The emission-level guarantee -- that `emit_digest`
    is not called at all on such a pass -- is covered by
    ``test_fleet_loop_converged_pass_does_not_emit_digest`` (issue #610), which drives
    the full ``fleet_loop`` notify gate. This test stays at the builder level so a
    regression in the routing (re-introducing a fallback for ``runner_allocation``)
    is caught here even if the gate test's mocks mask it.
    """
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest

    converged = [
        {
            "repo_key": "fleet",
            "type": "runner_allocation",
            "started": 0,
            "parked": 0,
            "budget": 8,
            "notes": ["Senkichi/job-cannon: holding 4 surplus slot(s) - slack for 0/3 pass(es)"],
            "dry_run": False,
        }
    ]
    digest = _build_fleet_attention_digest(converged)
    assert digest.transitions == ()


def test_digest_still_surfaces_allocation_failures() -> None:
    """Dropping the success event must not also drop the errors and skips."""
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest

    failures = [
        {"repo_key": "fleet", "type": "runner_allocation_error", "error": "managed_root missing"},
        {
            "repo_key": "fleet",
            "type": "runner_allocation_slot_error",
            "runner": "cw-2",
            "action": "park",
            "message": "still busy",
        },
        {"repo_key": "fleet", "type": "runner_allocation_skipped", "reason": "no registry anchor"},
    ]
    digest = _build_fleet_attention_digest(failures)
    assert len(digest.transitions) == 3
    healths = {e.health for e in digest.transitions}
    # Both error types must reach the desktop-eligible severity set, not just the log.
    assert "ERROR" in healths
    reasons = " ".join(str(e.last_log_line) for e in digest.transitions)
    assert "managed_root missing" in reasons
    assert "no registry anchor" in reasons
