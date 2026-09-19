"""Review test-adequacy gate: hard fails, label edges, rework-cap escalation.

Split out of ``tests/test_charlie_work.py`` (issue #1549, Track-1 wave 3/8):
the review-time test-adequacy gate seam (issue #179) -- hard-fail recording, label edges, escalation at the rework cap, and no-op paths. Shared fakes and helpers in ``tests/_review_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from _review_fixtures import _test_adequacy_app
from charlie_work.state import load_state, save_state
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


# --- Test-adequacy gate (issue #179) ------------------------------------------


def test_review_test_adequacy_disabled_is_noop(tmp_path: Path, monkeypatch) -> None:
    """When test_adequacy.enabled=False (default), check_test_adequacy is never called."""
    app = _test_adequacy_app(tmp_path, enabled=False)
    calls = {"n": 0}

    def _fake_check(diff, pr, config):
        calls["n"] += 1
        raise AssertionError("check_test_adequacy should not be called when disabled")

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)

    result = app.review(456)

    assert calls["n"] == 0
    assert result.ok is True
    assert "prompt_path" in result.data


def test_review_test_adequacy_hard_fail_records_request_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """When test_adequacy hard-fails, review() calls record_review with request_changes."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True)
    calls = {"check_test_adequacy": 0, "record_review": 0, "transition": 0}

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py", "src/bar.py"),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        calls["check_test_adequacy"] += 1
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        calls["record_review"] += 1
        assert decision == "request_changes"
        summary = kwargs.get("summary", "")
        assert "Test adequacy check failed" in summary
        assert "src/foo.py" in summary
        assert "src/bar.py" in summary
        assert "Test-exempt:" in summary
        return CommandResult(True, "record_review called", {})

    def _fake_transition(gh, labels, issue_number, edge):
        calls["transition"] += 1
        assert edge == "review_started"
        from charlie_work.labels import TransitionResult, TransitionOutcome

        return TransitionResult(
            outcome=TransitionOutcome.APPLIED,
            add_failures=[],
            remove_failures=[],
        )

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    result = app.review(456)

    assert calls["check_test_adequacy"] == 1
    assert calls["transition"] == 1
    assert calls["record_review"] == 1
    assert result.ok is True
    assert result.data == {}


def test_review_test_adequacy_hard_fail_label_set(tmp_path: Path, monkeypatch) -> None:
    """After hard-fail, label transitions compose to {in_progress, pr_open, needs_rework}."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.labels import TransitionResult, TransitionOutcome
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True)
    transition_calls = []

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        # Simulate the rework_requested transition that record_review would call
        transition_calls.append("rework_requested")
        return CommandResult(True, "record_review called", {})

    def _fake_transition(gh, labels, issue_number, edge):
        transition_calls.append(edge)
        if edge == "review_started":
            return TransitionResult(
                outcome=TransitionOutcome.APPLIED,
                add_failures=[],
                remove_failures=[],
            )
        elif edge == "rework_requested":
            return TransitionResult(
                outcome=TransitionOutcome.APPLIED,
                add_failures=[],
                remove_failures=[],
            )
        raise AssertionError(f"Unexpected transition: {edge}")

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    app.review(456)

    assert transition_calls == ["review_started", "rework_requested"]


def test_review_test_adequacy_unchanged_head_not_rerecorded(tmp_path: Path, monkeypatch) -> None:
    """A second review() on the same unchanged head is blocked by janitor gate (no-op rework check)."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True)
    check_calls = {"n": 0}

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        check_calls["n"] += 1
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        return CommandResult(True, "record_review called", {})

    def _fake_transition(gh, labels, issue_number, edge):
        from charlie_work.labels import TransitionResult, TransitionOutcome

        return TransitionResult(
            outcome=TransitionOutcome.APPLIED,
            add_failures=[],
            remove_failures=[],
        )

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    # First review: hard-fail records request_changes
    app.gh.pr_head_shas[456] = "abc123abcdefabcdefabcdefabcdefabcdefabcd"
    result1 = app.review(456)
    assert check_calls["n"] == 1
    assert result1.ok is True

    # Simulate state after first review: decision=request_changes,
    # and the rework_requested issue status the real record_review sets (the janitor gate's
    # pending-rework guard reads it).
    state = load_state(app.paths.state_file)
    if "456" not in state["prs"]:
        state["prs"]["456"] = {}
    state["prs"]["456"]["decision"] = "request_changes"
    state["prs"]["456"]["reviewed_head_sha"] = "abc123abcdefabcdefabcdefabcdefabcdefabcd"
    state["issues"]["123"] = {
        **state["issues"].get("123", {}),
        "number": 123,
        "status": "rework_requested",
    }
    save_state(app.paths.state_file, state)

    # Issue #1362 Stage 1: _check_no_op_rework now reads the last verdict
    # from the file-first review_decision reader, not pr_state["decision"],
    # so the live request_changes decision must exist on disk too.
    pr_decision_dir = app.paths.prs / "pr-456"
    pr_decision_dir.mkdir(parents=True, exist_ok=True)
    (pr_decision_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "reviewed_head_sha": "abc123abcdefabcdefabcdefabcdefabcdefabcd",
            }
        ),
        encoding="utf-8",
    )

    # Second review on same head: janitor gate blocks (no-op rework check)
    # check_test_adequacy should NOT be called again
    result2 = app.review(456)
    assert check_calls["n"] == 1  # Still 1, not 2
    assert result2.ok is False
    assert "janitor gate blocked" in result2.message
    assert "unchanged since request_changes" in result2.message


def test_review_test_adequacy_escalates_at_max_rework_cycles(tmp_path: Path, monkeypatch) -> None:
    """After max_rework_cycles hard-fails, escalate to agent:human-needed."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict
    from charlie_work.workflow import CommandResult

    app = _test_adequacy_app(tmp_path, enabled=True, max_rework_cycles=2)
    check_calls = {"n": 0}

    hard_fail_verdict = TestAdequacyVerdict(
        ok=False,
        failures=("Product code changed (15 LOC added) but no test files changed.",),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=0,
            assertion_count=0,
            test_files_changed=0,
            untested_product_files=("src/foo.py",),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        check_calls["n"] += 1
        return hard_fail_verdict

    def _fake_record_review(pr_number, decision, **kwargs):
        state = load_state(app.paths.state_file)
        pr_state = state["prs"].get(str(pr_number), {})
        request_changes_count = int(pr_state.get("request_changes_count", 0))
        # Simulate the increment that record_review would do
        if decision == "request_changes":
            escalated = request_changes_count >= 2  # Check before increment
            # Simulate head_advanced check (issue #208)
            reviewed_head_sha = app.gh.pr_head_shas.get(pr_number)
            head_advanced = reviewed_head_sha != pr_state.get("reviewed_head_sha")
            if not escalated and head_advanced:
                request_changes_count += 1
            pr_state["request_changes_count"] = request_changes_count
            state["prs"][str(pr_number)] = pr_state
            save_state(app.paths.state_file, state)
        else:
            escalated = False
        return CommandResult(True, "record_review called", {"escalated": escalated})

    def _fake_transition(gh, labels, issue_number, edge):
        from charlie_work.labels import TransitionResult, TransitionOutcome

        return TransitionResult(
            outcome=TransitionOutcome.APPLIED,
            add_failures=[],
            remove_failures=[],
        )

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)
    monkeypatch.setattr("charlie_work.workflow.transition", _fake_transition)
    monkeypatch.setattr(app, "record_review", _fake_record_review)

    # Cycle 1: request_changes_count = 0 -> 1
    app.gh.pr_head_shas[456] = "sha-1"
    result1 = app.review(456)
    assert result1.data["escalated"] is False

    # Cycle 2: request_changes_count = 1 -> 2
    app.gh.pr_head_shas[456] = "sha-2"
    result2 = app.review(456)
    assert result2.data["escalated"] is False

    # Cycle 3: request_changes_count = 2 -> escalate
    app.gh.pr_head_shas[456] = "sha-3"
    result3 = app.review(456)
    assert result3.data["escalated"] is True


def test_review_test_adequacy_pass_proceeds_to_packet(tmp_path: Path, monkeypatch) -> None:
    """When test_adequacy passes, review() proceeds to normal packet path."""
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict

    app = _test_adequacy_app(tmp_path, enabled=True)
    check_calls = {"n": 0}

    pass_verdict = TestAdequacyVerdict(
        ok=True,
        failures=(),
        warnings=(),
        facts=TestAdequacyFacts(
            added_product_loc=15,
            added_test_loc=20,
            assertion_count=5,
            test_files_changed=2,
            untested_product_files=(),
            exempt=False,
            exempt_reason="",
        ),
    )

    def _fake_check(diff, pr, config):
        check_calls["n"] += 1
        return pass_verdict

    monkeypatch.setattr("charlie_work.workflow.check_test_adequacy", _fake_check)

    result = app.review(456)

    assert check_calls["n"] == 1
    assert result.ok is True
