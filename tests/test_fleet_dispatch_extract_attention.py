"""``_extract_attention_events``: event-to-attention-entry mapping.

Split out of ``tests/test_fleet_dispatch.py`` (issue #1557, Track 1) --
bodies are verbatim relocations; shared helpers and the autouse hermeticity
fixtures live in ``tests/_fleet_dispatch_fixtures.py``.
"""

from __future__ import annotations

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


def test_extract_attention_events_deferred_by_concurrency_truncation_desync() -> None:
    """Issue #1005 review: ``deferred_by_concurrency`` is truncated to
    ``_MAX_DEFERRED_CONCURRENCY_EXAMPLES`` (5) in the persisted payload, but
    the ``failures`` map it feeds is not. A set-membership exclusion check
    against the truncated list would silently re-report the 6th+ deferred
    issue as a genuine launch failure -- a diagnostic regression in exactly
    the dimension #1005 is about. Exclusion is matched by reason-string
    prefix instead (``DEFERRED_BY_CONCURRENCY_REASON_PREFIX``), which is
    immune to truncation of the list.
    """
    deferred_issue_numbers = [101, 102, 103, 104, 105, 106, 107]
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "intake": {"failed": []},
            "dispatch": {
                "selected_count": 0,
                # Truncated to 5, mirroring the real persisted payload shape.
                "deferred_by_concurrency": deferred_issue_numbers[:5],
                "deferred_by_concurrency_count": len(deferred_issue_numbers),
                "failures": {
                    n: "deferred by concurrency cap (limit: 0)" for n in deferred_issue_numbers
                },
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    error_events = [e for e in events if e["type"] == "error"]
    assert error_events == []


def test_extract_attention_events_empty() -> None:
    """_extract_attention_events returns empty list for result with no events."""
    result = CommandResult(True, "loop complete", {"stalled": [], "errors": []})

    events = _extract_attention_events("owner/repo1", result)

    assert events == []


def test_extract_attention_events_errors() -> None:
    """_extract_attention_events extracts PR errors."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [
                {"pr": 789, "error": "merge conflict"},
                {"pr": 101, "error": "network error"},
            ],
        },
    )

    events = _extract_attention_events("owner/repo2", result)

    assert len(events) == 2
    assert events[0]["repo_key"] == "owner/repo2"
    assert events[0]["type"] == "error"
    assert events[0]["pr"] == 789
    assert events[0]["error"] == "merge conflict"
    assert events[1]["repo_key"] == "owner/repo2"
    assert events[1]["type"] == "error"
    assert events[1]["pr"] == 101


def test_extract_attention_events_errors_prefer_issue_number() -> None:
    """_extract_attention_events surfaces the linked issue number for errors that carry one (issue #502)."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [
                {"pr": 501, "issue": 494, "error": "possible worker self-merge"},
            ],
        },
    )

    events = _extract_attention_events("owner/repo2", result)

    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["issue_number"] == 494
    assert events[0]["pr"] == 501


def test_extract_attention_events_escalated_label_repair_deferred_silent() -> None:
    """Issue #1088: subjects held back by the per-pass cap must NOT produce an event.

    ``deferred`` counts subjects beyond ``escalated_label_repair_max_per_pass``;
    the sweep converges over subsequent passes by design, so this is normal
    steady-state progress, not a fault worth an operator's attention.
    """
    # Positive control -- same shape, `errored` populated, must fire.
    control_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[801])},
    )
    control_events = _extract_attention_events("owner/repo", control_result)
    assert len([e for e in control_events if e["type"] == "escalated_label_repair_error"]) == 1

    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(deferred=3)},
    )
    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert repair_events == []


def test_extract_attention_events_escalated_label_repair_errored() -> None:
    """Issue #1088: a subject whose GitHub call raised must surface in the digest.

    ``errored`` is the only durable record that an escalated-label repair was
    attempted and failed to reach GitHub -- state.json gets nothing written for
    it, so events.db and this digest are the sole places an operator could ever
    learn about it. This also serves as the positive control for the
    success/deferred/steady-state tests below: it proves the collector CAN
    produce an event from this payload shape before those tests assert it does
    not.
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[501, 502])},
    )

    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 501
    assert repair_events[0]["repo_key"] == "owner/repo"


def test_extract_attention_events_escalated_label_repair_failures() -> None:
    """Issue #1088: a subject whose transition() ran but did not fully apply.

    ``failures`` (label add/remove partially rejected by GitHub) is a distinct
    operational state from ``errored`` (nothing written) -- both are things a
    human eventually has to look at, so both must produce an entry. ``errored``
    is empty here to isolate that ``failures`` alone is sufficient.
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(issue_numbers=[601], failures=[601])},
    )

    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 601


def test_extract_attention_events_escalated_label_repair_malformed_input() -> None:
    """Issue #1088: malformed ``escalated_labels_repaired`` shapes must not raise.

    This payload comes from a per-repo ``CommandResult.data`` that ultimately
    traces back to another process's JSON. A shape drift there (e.g. a future
    refactor that changes ``errored`` from a list to a dict, or the whole key
    to a bool) must degrade to "no event" in the fleet digest builder, not
    crash the whole attention-extraction pass for every other repo in the
    fleet loop.
    """
    not_a_dict_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": "not-a-dict"},
    )
    events_not_dict = _extract_attention_events("owner/repo", not_a_dict_result)
    assert [e for e in events_not_dict if e["type"] == "escalated_label_repair_error"] == []

    errored_not_list_result = CommandResult(
        True,
        "review dispatch disabled",
        {
            "escalated_labels_repaired": {
                "issue_numbers": [],
                "failures": [],
                "errored": "901",
                "deferred": 0,
            }
        },
    )
    events_bad_errored = _extract_attention_events("owner/repo", errored_not_list_result)
    assert [e for e in events_bad_errored if e["type"] == "escalated_label_repair_error"] == []


def test_extract_attention_events_escalated_label_repair_mixed_prefers_errored_anchor() -> None:
    """Issue #1088: with both `errored` and `failures` populated, `errored` anchors.

    A single real pass can produce both: one subject's ``issue_view``/``transition()``
    call raises (-> ``errored``) while a *different* subject's ``transition()``
    completes but doesn't fully apply (-> ``failures``). Every other test in this
    file exercises a payload where one of the two lists is empty, so the collector's
    ``(errored or failures)[0]`` choice is never actually exercised elsewhere --
    it degrades to "return the only non-empty list" and would pass just as well
    under a reversed `(failures or errored)[0]`. This pins the real tie-break: the
    unreachable subject (nothing durable in state.json, so this digest is its only
    record) anchors the entry over the diagnosable one (already recorded via
    `label_error` in state.json).
    """
    result = CommandResult(
        True,
        "review dispatch disabled",
        {
            "escalated_labels_repaired": _repair_payload(
                issue_numbers=[602], failures=[602], errored=[501]
            )
        },
    )

    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 501
    assert "1 unreachable" in repair_events[0]["error"]
    assert "1 not applied" in repair_events[0]["error"]


def test_extract_attention_events_escalated_label_repair_steady_state_silent() -> None:
    """Issue #1088: the idle steady state (nothing to repair at all) is silent.

    This is the ``empty`` sentinel ``_repair_escalated_labels`` returns when
    there were no escalated subjects needing repair -- the common case on a
    healthy fleet. It must not manufacture a digest entry every single pass.
    """
    # Positive control -- same shape, `errored` populated, must fire.
    control_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[901])},
    )
    control_events = _extract_attention_events("owner/repo", control_result)
    assert len([e for e in control_events if e["type"] == "escalated_label_repair_error"]) == 1

    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload()},
    )
    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert repair_events == []


def test_extract_attention_events_escalated_label_repair_success_silent() -> None:
    """Issue #1088: a fully successful repair must NOT produce an attention event.

    Deliberate, per the collector's docstring: a self-healed success is not
    something needing attention, and it is already durable in events.db via
    the ``escalated_label_repaired`` state event. Flooding the digest with a
    healthy sweep's output would bury the ``errored``/``failures`` signal this
    whole feature exists to surface.
    """
    # Positive control: the same call shape, but with `errored` populated,
    # must produce an event -- otherwise the "no event" assertion below is
    # equally consistent with a broken test harness as with correct behavior.
    control_result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(errored=[701])},
    )
    control_events = _extract_attention_events("owner/repo", control_result)
    assert len([e for e in control_events if e["type"] == "escalated_label_repair_error"]) == 1

    result = CommandResult(
        True,
        "review dispatch disabled",
        {"escalated_labels_repaired": _repair_payload(issue_numbers=[701])},
    )
    events = _extract_attention_events("owner/repo", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert repair_events == []


def test_extract_attention_events_health_transitions() -> None:
    """_extract_attention_events extracts health transitions."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "health_transitions": [
                {"session_id": "sess1", "from_state": "running", "to_state": "stalled"},
                {"session_id": "sess2", "from_state": "stalled", "to_state": "running"},
            ],
        },
    )

    events = _extract_attention_events("owner/repo3", result)

    assert len(events) == 2
    assert events[0]["repo_key"] == "owner/repo3"
    assert events[0]["type"] == "health_transition"
    assert events[0]["session_id"] == "sess1"
    assert events[0]["from_state"] == "running"
    assert events[0]["to_state"] == "stalled"
    assert events[1]["repo_key"] == "owner/repo3"
    assert events[1]["type"] == "health_transition"
    assert events[1]["session_id"] == "sess2"


def test_extract_attention_events_includes_live_worker_redispatch_averted() -> None:
    """Issue #506: fleet attention digest surfaces live-worker redispatch averted outcomes."""
    from charlie_work.fleet_dispatch import _build_fleet_attention_digest

    result = CommandResult(
        True,
        "dispatch complete",
        {
            "dispatch": {
                "live_worker_redispatch_averted": [
                    {
                        "issue_number": 1317,
                        "branch_name": "agent/issue-1317-fix-search",
                        "pid": 12345,
                        "probe_result": "pid_alive",
                        "adapter_kind": "devin-shell",
                    }
                ]
            }
        },
    )

    events = _extract_attention_events("owner/repo", result)
    assert len(events) == 1
    assert events[0]["type"] == "live_worker_redispatch_averted"
    assert events[0]["issue_number"] == 1317
    assert events[0]["reason"] == "pid_alive"
    assert events[0]["adapter_kind"] == "devin-shell"

    digest = _build_fleet_attention_digest(events)
    assert len(digest.transitions) == 1
    entry = digest.transitions[0]
    assert entry.issue_number == 1317
    assert entry.health == "DISPATCH_AVERTED"
    assert entry.last_log_line == "pid_alive"
    assert entry.adapter_kind == "devin-shell"


def test_extract_attention_events_nested_dispatch_failures() -> None:
    """Issue #497: worker/rework/reviewer launch failures in nested dispatch
    sub-results are surfaced as attention events with the actual error text
    and a non-sentinel issue/PR identifier. Concurrency deferrals are not
    treated as launch failures.
    """
    review_error = (
        "failed to launch claude: [WinError 2] The system cannot find the file specified"
    )
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "intake": {"failed": []},
            "dispatch": {
                "selected_count": 0,
                "failures": {11: "failed to launch claude: OSError"},
            },
            "dispatch_rework": {
                "selected_count": 0,
                "deferred_by_concurrency": [13],
                "failures": {
                    12: "failed to launch claude: timeout",
                    13: "deferred by concurrency cap (limit: 1)",
                },
            },
            "dispatch_reviews": {
                "selected_count": 1,
                "failed_count": 1,
                "failed": [{"pr": 100, "error": review_error}],
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 3
    by_issue = {e.get("issue_number", e.get("pr")): e for e in error_events}
    assert by_issue[11]["error"] == "failed to launch claude: OSError"
    assert by_issue[12]["error"] == "failed to launch claude: timeout"
    assert by_issue[100]["error"] == review_error
    assert 13 not in by_issue

    digest = _build_fleet_attention_digest(events)
    entries = [e for e in digest.transitions if e.health == "ERROR"]
    assert len(entries) == 3
    assert all(e.issue_number != -1 for e in entries)
    assert any(review_error in (e.last_log_line or "") for e in entries)


def test_extract_attention_events_nested_escalated_label_repair() -> None:
    """Issue #1088: the key nested under ``dispatch_reviews`` (as loop() nests it).

    ``dispatch_reviews()``'s own CommandResult carries ``escalated_labels_repaired``
    at its top level; ``loop()`` nests that whole dict under a ``dispatch_reviews``
    key in its own result. Mirrors
    ``test_extract_attention_events_nested_review_verdicts`` -- if only the
    top-level form were checked, every deployed fleet pass (which goes through
    ``loop()``) would never surface this event at all.
    """
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "dispatch_reviews": {
                "escalated_labels_repaired": _repair_payload(errored=[801]),
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    repair_events = [e for e in events if e["type"] == "escalated_label_repair_error"]
    assert len(repair_events) == 1
    assert repair_events[0]["issue_number"] == 801


def test_extract_attention_events_nested_loop_skip() -> None:
    """Nested loop() sub-results (intake/dispatch/rework/reviews) carry their
    own skip signals, so extraction must recurse one level and surface them.
    """
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "intake": {"pass_skipped": True, "reason": "state_lock_busy"},
            "dispatch": {"state_lock_busy": True, "reason": "state_lock_busy"},
            "dispatch_rework": {"deferred_reason": "graphql_rate_limit"},
            "dispatch_reviews": {"pass_skipped": True, "reason": "state_lock_busy"},
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    assert all(event["repo_key"] == "owner/repo1" for event in events)
    assert all(event["type"] == "skipped" for event in events)
    assert {event["reason"] for event in events} == {"state_lock_busy", "graphql_rate_limit"}
    assert len(events) == 2


def test_extract_attention_events_nested_review_verdicts() -> None:
    """Issue #507: review verdict events in nested dispatch_reviews sub-results are extracted."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [],
            "errors": [],
            "dispatch_reviews": {
                "recorded_verdicts": [{"pr": 200, "issue": 20, "decision": "request_changes"}],
                "missed_verdicts": [],
            },
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    assert len(events) == 1
    assert events[0]["type"] == "review_verdict_recorded"
    assert events[0]["pr"] == 200
    assert events[0]["issue_number"] == 20


def test_extract_attention_events_review_verdicts() -> None:
    """Issue #507: recorded/missed review verdicts surface in fleet attention events."""
    result = CommandResult(
        True,
        "review dispatch: 0 launched, 0 failed; 1 verdict(s) recorded, 1 missed",
        {
            "stalled": [],
            "errors": [],
            "recorded_verdicts": [{"pr": 100, "issue": 10, "decision": "approved"}],
            "missed_verdicts": [{"pr": 101, "issue": 11, "reason": "no parseable verdict"}],
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    recorded = [e for e in events if e["type"] == "review_verdict_recorded"]
    missed = [e for e in events if e["type"] == "review_verdict_missed"]
    assert len(recorded) == 1
    assert recorded[0]["pr"] == 100
    assert recorded[0]["issue_number"] == 10
    assert recorded[0]["decision"] == "approved"
    assert len(missed) == 1
    assert missed[0]["pr"] == 101
    assert missed[0]["reason"] == "no parseable verdict"


def test_extract_attention_events_stalled() -> None:
    """_extract_attention_events extracts stalled sessions."""
    result = CommandResult(
        True,
        "loop complete",
        {
            "stalled": [
                {"session_id": "sess1", "issue_number": 123, "reason": "timeout"},
                {"session_id": "sess2", "issue_number": 456, "reason": "crash"},
            ],
            "errors": [],
        },
    )

    events = _extract_attention_events("owner/repo1", result)

    assert len(events) == 2
    assert events[0]["repo_key"] == "owner/repo1"
    assert events[0]["type"] == "stalled"
    assert events[0]["session_id"] == "sess1"
    assert events[0]["issue_number"] == 123
    assert events[0]["reason"] == "timeout"
    assert events[1]["repo_key"] == "owner/repo1"
    assert events[1]["type"] == "stalled"
    assert events[1]["session_id"] == "sess2"
