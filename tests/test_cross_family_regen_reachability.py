"""Tests for issue #1081: an unusable cross-family report must be regenerated,
not skipped forever.

Follow-up to #1079, which made this failure *visible* (one error-level
``cross_family_verdict_head_indeterminate`` event) but deliberately left the PR
waiting on a human indefinitely.

The defect was NOT missing regeneration logic. ``_cross_family_for_pr`` already
refuses to reuse a ``(UNAVAILABLE)`` failure stub or a semantically empty report
and re-runs the model instead. That code was **unreachable**: its only caller is
``review()``, and ``loop()``'s same-head packet skip bypassed ``review()``
whenever the packet head and prompt-template digest were both unchanged. A PR
whose head never moves therefore kept an unusable report permanently, while
``_record_cross_family_verdicts`` skipped it on every pass.

The fix adds the report's usability as a *third* staleness input to that skip,
evaluated with ``report_is_reusable`` -- the same predicate the regenerator
uses, so the two cannot disagree about what "reusable" means.

Measured against production before writing this (charlie-work, 2026-08-06):
exactly one open PR was in this state (#1073), and its report was a 109-byte
``(UNAVAILABLE)`` glm-5.2 timeout stub -- *not* the "empty headRefOid at
generation time" shape the issue hypothesised. Its live and packet heads were
both ``f8c21ed2``, which is precisely why the skip fired. The other three open
PRs with unusable reports all had a packet-vs-live head mismatch, so the
pre-existing head check already re-ran ``review()`` for them; they were never
stuck. Hence the fixtures here pin the *static-head* case specifically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import (
    AutoMergeConfig,
    CrossFamilyConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.cross_family import _CAVEAT, CrossFamilyResult
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
from charlie_work.workflow import OrchestratorApp

# Reuse the shared FakeGitHub whose default PR #456 is janitor-green.
from test_charlie_work import FakeGitHub

HEAD = "sha-static"

# A body that parses as a real review: severity marker + non-refusal verdict.
_GOOD_BODY = "**MINOR**\nsmall issue\n\nVerdict: No BLOCKERs or MAJORs — fix is correct"

# The exact production failure-stub shape written by cross_family._fail() when
# the model times out. This is what PR #1073 actually carried.
_UNAVAILABLE_STUB = (
    "# Cross-family adversarial review — `glm-5.2` (UNAVAILABLE)\n\n"
    "> cross-family review timed out after 600s\n"
)

# Parses cleanly, but carries no "<!-- PR head SHA: ... -->" comment, so
# extract_head_ref_oid() returns None and the head guard can never adjudicate.
_NO_HEAD_SHA_REPORT = (
    f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{_GOOD_BODY}\n"
)


def _good_report(head_sha: str) -> str:
    """A reusable report: real body, and generated against ``head_sha``."""
    return (
        f"# Cross-family adversarial review — `glm-5.2`\n\n"
        f"<!-- PR head SHA: {head_sha} -->\n\n"
        f"{_CAVEAT}\n\n---\n\n{_GOOD_BODY}\n"
    )


def _pr456(head_sha: str, *, draft: bool = False) -> dict[str, Any]:
    return {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": head_sha,
        # The body must satisfy the janitor's require_tests_or_rationale gate.
        # Without it review() returns early ("janitor gate blocked") and never
        # reaches the cross-family pass, so every assertion about regeneration
        # here would pass or fail for entirely the wrong reason.
        "body": "Closes #123\n\nTests: added unit tests covering the change.",
        "labels": [],
        "isCrossRepository": False,
        "isDraft": draft,
        "state": "OPEN",
    }


def _make_app(
    tmp_path: Path,
    *,
    prs: list[dict[str, Any]],
    cross_family_enabled: bool = True,
    max_regen_attempts: int = 2,
    auto_verdict: bool = True,
    dry_run: bool = False,
    review_dispatch_enabled: bool = False,
) -> OrchestratorApp:
    """Mirror of test_charlie_work._make_loop_app, but with cross-family ON.

    The shared helper hardcodes ``CrossFamilyConfig(enabled=False)``, which
    would make every assertion here vacuous -- with the pass disabled the new
    staleness input deliberately reports "current" and never fires.
    """
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(
            enabled=cross_family_enabled,
            auto_verdict=auto_verdict,
            max_regen_attempts=max_regen_attempts,
        ),
        auto_merge=AutoMergeConfig(required_checks=(), require_approved_review=True),
        # Defaults to False, matching ReviewDispatchConfig's own default and
        # both live fleets. The `review_started` label edge at the tail of
        # review() is gated on this, so it is off in every other test here --
        # which is precisely why the #384 clobber below stayed invisible.
        review_dispatch=ReviewDispatchConfig(enabled=review_dispatch_enabled),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.prs = prs
    return OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=dry_run)


def _plant_packet(
    app: OrchestratorApp,
    tmp_path: Path,
    pr_number: int,
    *,
    head_sha: str,
    report_text: str | None,
) -> Path:
    """Plant a review packet that is current on head AND template.

    The template digest is taken from ``_review_template_sha()`` rather than
    hardcoded, so these tests keep testing the cross-family input specifically
    and do not silently start passing because the template happened to look
    stale.
    """
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps(
            {
                "number": pr_number,
                "headRefOid": head_sha,
                "prompt_template_sha": app._review_template_sha(),
            }
        ),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}", encoding="utf-8"
    )
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "pending", "reviewed_head_sha": None}), encoding="utf-8"
    )
    if report_text is not None:
        (pr_dir / "cross-family-review.md").write_text(report_text, encoding="utf-8")
    return pr_dir


def _track_review(app: OrchestratorApp) -> list[int]:
    calls: list[int] = []
    original = app.review

    def tracking(pr_number: int) -> object:
        calls.append(pr_number)
        return original(pr_number)

    app.review = tracking  # type: ignore[method-assign]
    return calls


def _stub_model(monkeypatch: pytest.MonkeyPatch, *, writes: str | None) -> list[dict[str, Any]]:
    """Replace the cross-family subprocess; record every invocation.

    ``writes`` is the report text the fake model leaves behind (None = it
    leaves the existing unusable report in place, i.e. the model is still
    down). Invocation of THIS is the end-to-end signal that regeneration was
    actually reached -- ``review()`` being called is necessary but not
    sufficient.

    Issue #1078: the code now calls ``launch_cross_family_review`` (async,
    Popen-based) instead of ``run_cross_family_review`` (synchronous). The
    fake simulates immediate completion — it writes the report and returns a
    non-pending result — so these regen-reachability tests can still exercise
    the full ``loop() → review() → _cross_family_for_pr`` path in a single
    pass. The async-specific behaviour (pending, reap on next pass) is covered
    by ``test_issue_1078_async_cross_family.py``.
    """
    calls: list[dict[str, Any]] = []

    def fake_launch(**kwargs: Any) -> CrossFamilyResult:
        calls.append(kwargs)
        report_path = Path(kwargs["report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if writes is not None:
            report_path.write_text(writes, encoding="utf-8")
        else:
            report_path.write_text(_UNAVAILABLE_STUB, encoding="utf-8")
        return CrossFamilyResult(
            ok=writes is not None, report_path=str(report_path), model="glm-5.2"
        )

    monkeypatch.setattr("charlie_work.workflow.launch_cross_family_review", fake_launch)
    return calls


def _labels_added(app: OrchestratorApp) -> list[tuple[int, str]]:
    """The FakeGitHub's recorded ``(issue, label)`` additions.

    ``app.gh`` is declared as the ``GitHubLike`` protocol, which has no
    ``labels_added`` -- that is test-double bookkeeping, not part of the
    interface. Narrowed here once rather than per-assertion.
    """
    gh: Any = app.gh
    return gh.labels_added


def _events(app: OrchestratorApp, kind: str) -> list[dict[str, Any]]:
    return query_events(app.paths.state_file, kind=kind)


# ---------------------------------------------------------------------------
# The core defect: a static-head PR with an unusable report
# ---------------------------------------------------------------------------


def test_loop_regenerates_when_report_is_unavailable_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production shape from PR #1073: packet head == live head, template
    current, report is a ``(UNAVAILABLE)`` timeout stub.

    Before this fix the skip fired and the stub survived forever.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)])
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_UNAVAILABLE_STUB)
    model_calls = _stub_model(monkeypatch, writes=_good_report(HEAD))
    review_calls = _track_review(app)

    result = app.loop(limit=0)

    assert 456 in review_calls, "review() must re-run so regeneration is reachable"
    assert result.data["skipped_reviews"] == 0
    # The end-to-end assertion: the model was actually re-invoked. review()
    # alone would not prove the stub was replaced.
    assert len(model_calls) == 1
    regen = _events(app, "cross_family_report_regen_forced")
    assert [e["payload"]["attempt"] for e in regen if e.get("pr_number") == 456] == [1]


def test_loop_regenerates_when_report_has_no_head_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1081's literal AC-3 shape: a report that parses cleanly but
    carries no head SHA, so the head guard can never adjudicate it.

    Distinct from the stub case: this one reaches ``parse_cross_family_verdict``
    successfully and dies at the head guard, so a fix that only special-cased
    ``(UNAVAILABLE)`` would leave it stuck.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)])
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_NO_HEAD_SHA_REPORT)
    model_calls = _stub_model(monkeypatch, writes=_good_report(HEAD))
    review_calls = _track_review(app)

    result = app.loop(limit=0)

    assert 456 in review_calls
    assert result.data["skipped_reviews"] == 0
    assert len(model_calls) == 1
    # And the regenerated report now carries the head SHA, so the guard the
    # issue was filed against can finally adjudicate it.
    report = (
        tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "cross-family-review.md"
    ).read_text(encoding="utf-8")
    assert f"<!-- PR head SHA: {HEAD} -->" in report


# ---------------------------------------------------------------------------
# Negative controls -- the new input must not fire on healthy packets
# ---------------------------------------------------------------------------


def test_loop_skips_when_report_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid report generated against the live head keeps the deliberate
    idempotence guard intact.

    This is the control that stops the new staleness input from degenerating
    into an always-fire: without it, a check that returned "stale"
    unconditionally would pass every positive test above.

    ``auto_verdict`` is off here deliberately. With it on, a *reusable* report
    is immediately converted into an approved verdict and the PR proceeds to
    merge via the ``already_approved`` branch -- so it never reaches the
    same-head skip at all and ``skipped_reviews`` would be 0 for a reason that
    has nothing to do with this change.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)], auto_verdict=False)
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_good_report(HEAD))
    model_calls = _stub_model(monkeypatch, writes=_good_report(HEAD))
    review_calls = _track_review(app)

    result = app.loop(limit=0)

    assert 456 not in review_calls
    assert result.data["skipped_reviews"] == 1
    assert model_calls == []
    assert _events(app, "cross_family_report_regen_forced") == []


def test_disabled_cross_family_never_forces_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the pass disabled no report is ever written, so an absent report is
    the steady state -- not staleness.

    Treating it as stale would force ``review()`` on every single pass forever
    for every PR in the fleet. This pins the early return that prevents it.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)], cross_family_enabled=False)
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=None)
    model_calls = _stub_model(monkeypatch, writes=_good_report(HEAD))
    review_calls = _track_review(app)

    result = app.loop(limit=0)

    assert 456 not in review_calls
    assert result.data["skipped_reviews"] == 1
    assert model_calls == []


def test_draft_pr_never_forces_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_cross_family_for_pr`` returns early on a draft and never writes a
    report, so "no report" is permanent for a draft. Without mirroring that
    early return here, every draft PR would re-enter review() on every pass."""
    app = _make_app(tmp_path, prs=[_pr456(HEAD, draft=True)])
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=None)
    model_calls = _stub_model(monkeypatch, writes=_good_report(HEAD))
    review_calls = _track_review(app)

    app.loop(limit=0)

    assert 456 not in review_calls
    assert model_calls == []


# ---------------------------------------------------------------------------
# The bound, and what it terminates in
# ---------------------------------------------------------------------------


def test_regeneration_is_bounded_and_escalates_to_a_human(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permanently-failing model must not re-burn its timeout every pass.

    Regeneration runs synchronously for up to ``timeout_seconds`` (600s in this
    fleet); unbounded, one down model would starve the other repo in the shared
    sequential loop (#1078). After the budget is spent the PR escalates to a
    human and stops regenerating.

    Critically it is NOT recorded as approved. ``max_parse_failures`` ends in a
    caveated ``approved``; doing that here would approve against a head that was
    never positively confirmed -- the fail-open #1079 closed.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)], max_regen_attempts=2)
    pr_dir = _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_UNAVAILABLE_STUB)
    model_calls = _stub_model(monkeypatch, writes=None)  # model stays down
    review_calls = _track_review(app)

    for _ in range(4):
        app.loop(limit=0)

    # Exactly max_regen_attempts regenerations, then it stops.
    assert len(model_calls) == 2, "the bound must cap repeated 600s regenerations"
    assert review_calls.count(456) == 2

    exhausted = _events(app, "cross_family_report_regen_exhausted")
    assert [e.get("pr_number") for e in exhausted] == [456], "exactly one, not one per pass"

    state = load_state(app.paths.state_file)
    issue = state["issues"].get("123", {})
    assert issue.get("status") == "escalated"
    assert issue.get("escalation_reason") == "cross_family_report_unusable"
    # judgment, NOT mechanical: a mechanical escalation is eligible for the
    # automatic de-escalation sweep, which would drop the PR straight back into
    # the indefinite silent wait this fix exists to make unreachable.
    assert issue.get("reason_class") == "judgment"

    # AC-4: the escalation must reach a CONSUMER, not just state.json.
    # _escalate_issue writes state only -- it applies no label itself. If
    # nothing downstream turned status="escalated" into the human_needed label,
    # this would be a signal nobody receives: invisible on GitHub, and the
    # RUNBOOK's "lands on agent:human-needed" would be false.
    assert (123, app.config.labels.human_needed) in _labels_added(app)

    # AC-2: no verdict was recorded against the unconfirmed head.
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] != "approved"


def test_the_escalating_pass_does_not_strip_the_human_needed_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #384: `review_started` must not fire on the pass that escalates.

    Since #1099 the cross-family exhaustion escalates from *inside* review()
    (`_cross_family_for_pr` -> `_escalate_cross_family_regen_exhausted`), below
    the escalated-guard at the head of the method. review() then continues to
    its label side-effects, and the `review_started` edge adds
    `(pr_open, reviewing)` while removing every other workflow label --
    `human_needed` among them. Left unguarded, the escalating pass applies the
    label and strips it moments later: escalated in state, invisible on GitHub.

    ``_labels_added`` cannot see this. It is append-only bookkeeping, so the
    assertion in the sibling test above stays true even if the label is removed
    on the next line. The discriminating signal is *how many* `review_started`
    edges fire, which is ordering-independent: the model runs on passes 1 and 2,
    and the escalation lands at the end of pass 2, so the edge must fire once.
    Without the guard it fires twice -- the second one being the clobber.

    This is the only test here that enables `review_dispatch`; with it at its
    default the edge is skipped entirely and the whole path is unreachable.
    """
    app = _make_app(
        tmp_path,
        prs=[_pr456(HEAD)],
        max_regen_attempts=2,
        review_dispatch_enabled=True,
    )
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_UNAVAILABLE_STUB)
    model_calls = _stub_model(monkeypatch, writes=None)  # model stays down

    for _ in range(4):
        app.loop(limit=0)

    reviewing = app.config.labels.reviewing
    human_needed = app.config.labels.human_needed

    # Control: the edge is genuinely live in this configuration. If this is 0
    # the test proves nothing -- it would pass just as happily with the label
    # machinery switched off, which is the state every other test here runs in.
    assert (123, reviewing) in _labels_added(app), "review_started never fired at all"

    assert len(model_calls) == 2, "precondition: escalation lands on the second pass"
    assert (123, human_needed) in _labels_added(app), "escalation must apply the label"
    assert _labels_added(app).count((123, reviewing)) == 1, (
        "review_started fired on the escalating pass and stripped human_needed"
    )


def test_pr_blocked_upstream_of_cross_family_terminates_by_parking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work must also terminate when regeneration can never even run.

    This is the shape PR #1073 actually has in production: it is janitor-blocked,
    so ``review()`` returns on the janitor gate and never reaches the
    cross-family pass at all. The model is therefore never invoked and the report
    can never improve, no matter how many passes run. Unbounded, this PR would
    re-enter ``review()`` on literally every pass forever.

    **The termination changed in #1099, deliberately, and this test changed with
    it.** It used to end in the same ``cross_family_report_unusable`` escalation
    as the previous test, on the reasoning that "the report is unusable" is
    observed rather than inferred. That reasoning was wrong in a way only
    production showed: "unusable" is observed, but the escalation asserts
    unusable *and unfixable*, and regeneration was never tried even once. In
    job-cannon 36 of 54 escalated issues carried that reason for a report whose
    model had never been invoked, and 26 of 27 re-escalated within hours of
    being re-armed. The sink refilled as fast as it was drained.

    So this path now terminates by PARKING -- the counter is recorded, no
    escalation, no label. The asymmetry is the point: the record is keyed by
    head SHA, so a park self-heals on the next push, whereas
    ``reason_class="judgment"`` is excluded from the automatic de-escalation
    sweep and needs a human. A wrong park costs a few cheap passes; a wrong
    escalation costs a person. The PR is not abandoned either -- the janitor
    gate reports its actual problem on its own channel every pass, which is the
    consumer that was always supposed to own this.
    """
    pr = _pr456(HEAD)
    # Drop the tests/rationale mention -> janitor gate blocks review() early.
    pr["body"] = "Closes #123"
    app = _make_app(tmp_path, prs=[pr], max_regen_attempts=2)
    pr_dir = _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_UNAVAILABLE_STUB)
    model_calls = _stub_model(monkeypatch, writes=_good_report(HEAD))
    review_calls = _track_review(app)

    for _ in range(5):
        app.loop(limit=0)

    assert model_calls == [], "review() never reaches the cross-family pass here"
    # The bound still binds: two forced passes, then parked. Five would mean it
    # never terminated, which is the failure this test has always existed for.
    assert review_calls.count(456) == 2

    not_reached = _events(app, "cross_family_regen_not_reached")
    assert [e["payload"]["not_reached"] for e in not_reached] == [1, 2]

    # The regeneration budget is untouched -- nothing was regenerated, so
    # nothing may be charged for regenerating. This is the assertion that would
    # have caught the production defect.
    assert _events(app, "cross_family_report_regen_forced") == []
    assert _events(app, "cross_family_report_regen_exhausted") == []

    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["cross_family_regen"]["not_reached"] == 2
    assert state["prs"]["456"]["cross_family_regen"]["attempts"] == 0
    assert state["issues"].get("123", {}).get("status") != "escalated"
    assert (123, app.config.labels.human_needed) not in _labels_added(app)
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] != "approved"


def test_dry_run_claims_nothing_and_fires_no_label_edit(tmp_path: Path) -> None:
    """``--dry-run`` must not write ``state.json`` or edit a live label.

    Since #1099 the bound has two mutating halves, and the gate is exercised on
    BOTH: ``_claim_...`` writes ``state.json``, and
    ``_escalate_cross_family_regen_exhausted`` is the only one that reaches
    GitHub. Proving the gate on one half would leave the other -- for the
    escalation, the part that mutates something outside this process --
    untested. That is not hypothetical: the gate used to be a single check in a
    single method, and the split is exactly the kind of change that silently
    drops one of them.

    Each half carries its own positive control. "Nothing was written" is equally
    consistent with "the gate works" and with "the call never reached the
    mutating code", so the identical call is made on a live app first and
    asserted to mutate. Without those controls this test would still pass if
    either branch were deleted outright.
    """
    for name in ("live-claim", "dry-claim", "live-esc", "dry-esc"):
        (tmp_path / name).mkdir()

    # --- half 1: the claim, which writes state.json --------------------------
    # max_regen_attempts=1 so there IS a budget to claim; at 0 the claim
    # short-circuits before the write and the control would prove nothing.
    live_claim = _make_app(tmp_path / "live-claim", prs=[_pr456(HEAD)], max_regen_attempts=1)
    assert _claim(live_claim) is True
    assert live_claim.paths.state_file.exists(), "control: the claim really does write"
    assert _events(live_claim, "cross_family_report_regen_forced") != []

    dry_claim = _make_app(
        tmp_path / "dry-claim", prs=[_pr456(HEAD)], max_regen_attempts=1, dry_run=True
    )
    assert _claim(dry_claim) is False
    assert not dry_claim.paths.state_file.exists()

    # --- half 2: the escalation, which reaches GitHub ------------------------
    live_esc = _make_app(tmp_path / "live-esc", prs=[_pr456(HEAD)], max_regen_attempts=0)
    _adjudicate(live_esc)
    assert live_esc.paths.state_file.exists()
    assert (123, live_esc.config.labels.human_needed) in _labels_added(live_esc)

    dry_esc = _make_app(
        tmp_path / "dry-esc", prs=[_pr456(HEAD)], max_regen_attempts=0, dry_run=True
    )
    _adjudicate(dry_esc)
    assert not dry_esc.paths.state_file.exists()
    assert _labels_added(dry_esc) == []


def test_regen_budget_resets_when_the_head_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attempts are counted per head SHA.

    A new push is a genuinely new review, and its report has never failed. If
    the counter carried across heads, a long-lived PR would exhaust its budget
    on one bad head and then escalate on an unrelated later push that had never
    been tried even once.
    """
    pr = _pr456(HEAD)
    app = _make_app(tmp_path, prs=[pr], max_regen_attempts=2)
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_UNAVAILABLE_STUB)
    model_calls = _stub_model(monkeypatch, writes=None)
    _track_review(app)

    # Exactly one pass of a budget of 2, so the budget is spent-but-not-
    # exhausted. Running it to exhaustion here would escalate the issue, and an
    # escalated PR is skipped earlier in loop(), so the head-move reset below
    # would never be reached and this test would pass for the wrong reason.
    app.loop(limit=0)
    assert len(model_calls) == 1, "one attempt spent at the first head"

    # New push: live head and packet head advance together, staying "same-head"
    # so the pre-existing head check still skips and only the reset is under test.
    new_head = "sha-moved"
    pr["headRefOid"] = new_head
    _plant_packet(app, tmp_path, 456, head_sha=new_head, report_text=_UNAVAILABLE_STUB)

    app.loop(limit=0)
    assert len(model_calls) == 2, "the new head gets its own budget"

    state = load_state(app.paths.state_file)
    record = state["prs"]["456"]["cross_family_regen"]
    assert record["head_sha"] == new_head
    assert record["attempts"] == 1


def test_the_operator_manual_rerun_is_exempt_from_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`charlie why-charlie-hate --pr <n>` must regenerate even once parked.

    This is RUNBOOK.md's documented recovery for an exhausted budget, and it is
    only safe to exempt because a human typing a command is not the loop the
    bound defends against (#1078). #1099 moved the claim INTO the regenerator,
    which is exactly the change that would have silently killed this path --
    hence the explicit ``enforce_regen_budget=False`` and this test.

    The negative control runs first: the identical call WITH the budget enforced
    must be refused. Without it this test would pass just as well if the budget
    had never been spent, or if the bound had been deleted outright.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)], max_regen_attempts=2)
    _plant_packet(app, tmp_path, 456, head_sha=HEAD, report_text=_UNAVAILABLE_STUB)
    model_calls = _stub_model(monkeypatch, writes=None)  # model stays down

    # A spent budget on a PR that is NOT escalated. Planted rather than driven
    # through loop(), because loop() escalates in the same pass that spends the
    # budget and review() hard-stops on an escalated issue -- which would make
    # both assertions below pass for a reason that has nothing to do with the
    # budget.
    state = load_state(app.paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "cross_family_regen": {"head_sha": HEAD, "attempts": 2, "not_reached": 0},
    }
    save_state(app.paths.state_file, state)

    # Control: the budgeted path is genuinely refused at this point.
    app.review(456)
    assert model_calls == [], "control: an enforced re-run must not regenerate"

    app.review(456, enforce_regen_budget=False)
    assert len(model_calls) == 1, "the operator's manual re-run gets a fresh attempt"
    # ...and it charges nothing, so repeated manual runs during diagnosis cannot
    # consume the loop's budget, and recovery needs no hand-edit of state.json.
    record = load_state(app.paths.state_file)["prs"]["456"]["cross_family_regen"]
    assert record["attempts"] == 2


def test_a_head_move_resets_both_budgets_together(tmp_path: Path) -> None:
    """``attempts`` and ``not_reached`` share one head-keyed record.

    That shared key is the whole reason the two budgets can share
    ``max_regen_attempts`` as their bound instead of taking a second config
    knob. If they could reset independently, a PR that burned its not-reached
    budget while janitor-blocked could push the fix that unblocks it and STILL
    be parked -- a silent park, with no escalation and no label, that nobody
    would think to look for.
    """
    app = _make_app(tmp_path, prs=[_pr456(HEAD)], max_regen_attempts=2)
    state = load_state(app.paths.state_file)
    state["prs"]["456"] = {
        "number": 456,
        "cross_family_regen": {"head_sha": HEAD, "attempts": 2, "not_reached": 2},
    }
    save_state(app.paths.state_file, state)

    # Control: at the head that spent them, both budgets read as spent and the
    # PR really is parked. Without this the assertions below would also pass if
    # the record were simply never read.
    spent = app._cross_family_regen_record(pr_number=456, head_sha=HEAD)
    assert (spent["attempts"], spent["not_reached"]) == (2, 2)
    assert app._cross_family_report_current(pr=_pr456(HEAD), pr_number=456) is True

    moved = _pr456("sha-moved")
    fresh = app._cross_family_regen_record(pr_number=456, head_sha="sha-moved")
    assert (fresh["attempts"], fresh["not_reached"]) == (0, 0)
    assert app._cross_family_report_current(pr=moved, pr_number=456) is False


# ---------------------------------------------------------------------------
# The label edge has to survive a transition() failure (AC-4)
#
# Escalating writes state only; the agent:human-needed label is what a human
# actually sees. #586 built a self-heal sweep for exactly this, but it lives in
# dispatch_reviews BELOW the review_dispatch.enabled early return, and its input
# set is built by the dispatch-selection loop that same return skips -- so with
# review dispatch off (this fleet's configuration) it is unreachable, not merely
# narrow. Without the retry below, one transient GitHub failure at escalation
# time would leave the PR escalated-in-state and invisible on GitHub forever,
# because the `record.get("escalated")` early return means the edge is never
# revisited. That is the precise defect #586 existed to kill, reintroduced.
# ---------------------------------------------------------------------------


class _FlakyLabelGitHub(FakeGitHub):
    """``add_issue_label`` fails the first ``fail_adds`` times, then succeeds."""

    def __init__(self, fail_adds: int) -> None:
        super().__init__()
        self.remaining_failures = fail_adds

    def add_issue_label(self, number: int, label: str) -> bool:
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            return False
        return super().add_issue_label(number, label)


def _escalating_app(tmp_path: Path, gh: FakeGitHub) -> OrchestratorApp:
    """An app whose regen budget is already spent, so the call escalates."""
    config = OrchestratorConfig(
        cross_family=CrossFamilyConfig(enabled=True, auto_verdict=True, max_regen_attempts=0),
        auto_merge=AutoMergeConfig(required_checks=(), require_approved_review=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh.prs = [_pr456(HEAD)]
    return OrchestratorApp(tmp_path, paths, config, gh)


def _claim(app: OrchestratorApp) -> bool:
    return app._claim_cross_family_regen_attempt(
        pr_number=456, issue_number=123, head_sha=HEAD, report_head=None
    )


def _adjudicate(app: OrchestratorApp) -> None:
    """Run the exhaustion adjudication for PR #456.

    Since #1099 the bound has two halves: ``_claim_...`` counts, and this one
    decides. Escalation and the label edge moved here because only a caller
    positioned *after* the model call can observe that the report is still
    unusable -- which is what ``cross_family_report_unusable`` asserts. No
    report file is planted, so ``report_is_reusable("")`` is False and these
    apps land on the escalating branch exactly as they did before the split.
    """
    app._escalate_cross_family_regen_exhausted(
        pr_number=456,
        issue_number=123,
        head_sha=HEAD,
        report_path=app.paths.prs / "pr-456" / "cross-family-review.md",
    )


def _label_error(app: OrchestratorApp) -> Any:
    """``label_error`` as persisted, distinguishing absent from None."""
    entry = load_state(app.paths.state_file)["issues"].get("123", {})
    return entry["label_error"] if "label_error" in entry else "ABSENT"


def test_a_failed_label_edge_is_retried_on_the_next_pass(tmp_path: Path) -> None:
    """A transition() failure must not make the escalation permanently invisible.

    The first pass escalates and the label add fails. The second pass must
    re-apply it -- even though there is nothing left to claim and the
    ``escalated`` flag short-circuits the rest of the method.
    """
    app = _escalating_app(tmp_path, _FlakyLabelGitHub(fail_adds=1))

    _adjudicate(app)
    human_needed = app.config.labels.human_needed
    assert (123, human_needed) not in _labels_added(app)
    assert isinstance(_label_error(app), dict), "a failed edge must be recorded as a dict"

    # Second pass: already escalated, but the edge is still owed.
    _adjudicate(app)
    assert (123, human_needed) in _labels_added(app)
    assert _label_error(app) is None, "a verified edge must be recorded as None"


def test_a_verified_label_edge_is_not_re_applied(tmp_path: Path) -> None:
    """Steady state costs nothing: once verified, later passes make no label call.

    This is the negative control for the retry above. Without it the retry
    could be satisfied by re-applying the edge unconditionally on every pass,
    which would hammer the GitHub API for every escalated PR forever.
    """
    app = _escalating_app(tmp_path, _FlakyLabelGitHub(fail_adds=0))
    human_needed = app.config.labels.human_needed

    _adjudicate(app)
    assert _labels_added(app).count((123, human_needed)) == 1
    assert _label_error(app) is None

    for _ in range(3):
        _adjudicate(app)
    assert _labels_added(app).count((123, human_needed)) == 1, "re-applied a verified edge"


def test_a_concurrent_unescalate_is_not_undone(tmp_path: Path) -> None:
    """The retry must not resurrect an escalation a concurrent unescalate freed.

    unescalate() clears ``label_error`` along with the status, so the
    absent-key arm ("never attempted") would otherwise read that cleared state
    as licence to re-apply agent:human-needed and silently undo the release.
    """
    app = _escalating_app(tmp_path, _FlakyLabelGitHub(fail_adds=1))
    human_needed = app.config.labels.human_needed

    _adjudicate(app)
    assert (123, human_needed) not in _labels_added(app)

    # Simulate the concurrent unescalate, using the literals it actually writes
    # rather than merely analogous ones: with the PR live and open both records
    # land on PASSIVE_OPEN_STATUS (workflow.py `_apply_pr_reset` /
    # `_apply_issue_reset`), and `label_error` is dropped because it is a member
    # of `_UNESCALATE_ISSUE_RESET_FIELDS` -- which is what puts this in the
    # absent-key ("never attempted") arm the status check has to override.
    state = load_state(app.paths.state_file)
    state["prs"]["456"] = {**state["prs"]["456"], "status": PASSIVE_OPEN_STATUS}
    issue_entry = {**state["issues"].get("123", {}), "status": PASSIVE_OPEN_STATUS}
    issue_entry.pop("label_error", None)
    state["issues"]["123"] = issue_entry
    save_state(app.paths.state_file, state)

    _adjudicate(app)
    assert (123, human_needed) not in _labels_added(app), "re-escalated a released issue"
