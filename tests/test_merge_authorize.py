"""Tests for issue #934: record an authorized override at merge time.

The unauthorized-merge tripwire (#673) and the ``merge-check`` preflight (#894)
both infer authorization from ``decision == "approved"`` and
``reviewed_head_sha == live_head_sha``. An operator who legitimately adjudicates
a worker PR whose recorded decision is stale, absent, or pending — and merges
it — has no way to record that adjudication, so every legitimate operator merge
becomes a tripwire finding.

``merge_authorize`` writes an ``authorized_override`` into the PR's
``review-decision.json``, bound to the SHA and carrying a mandatory reason. The
tripwire and ``merge-check`` treat a matching override as explicit
authorization (via ``_authorized_override_matches``), so the control reads a
**recorded** authorization rather than inferring one.

Three properties from the issue are tested here:

1. **Does not weaken the control** — an unrecorded merge is still a finding;
   a malformed or SHA-mismatched override does not authorize.
2. **The reason stays mandatory** — matching ``tripwire ack``.
3. **Bind to the SHA** — a rebase after authorization invalidates the override.

Reuses the tripwire fixtures from ``test_charlie_work.py``
(``_arm_unauthorized_merge_tripwire``, ``_merged_worker_pr``, ``FakeGitHub``)
and the merge-check fixtures (``_merge_check_app``, ``_write_decision``) per the
existing convention in this test suite.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work import cli
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _LEVEL_BY_KIND, close_db, query_events
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import (
    _arm_unauthorized_merge_tripwire,
    _merge_check_app,
    _merged_worker_pr,
    _write_decision,
)


# ---------------------------------------------------------------------------
# merge_authorize command
# ---------------------------------------------------------------------------


def test_merge_authorize_requires_reason(tmp_path: Path) -> None:
    """A tripwire that can be silenced silently is no control — before the
    merge as much as after it (issue #934, property 2)."""
    app, _, _ = _merge_check_app(tmp_path)

    result = app.merge_authorize(456, "", by="operator")

    assert result.ok is False
    assert "reason" in result.message.lower()


def test_authorize_requires_reason_whitespace(tmp_path: Path) -> None:
    """A tripwire that can be silenced silently is no control — before the
    merge as much as after it (issue #934, property 2)."""
    app, _, _ = _merge_check_app(tmp_path)

    result = app.merge_authorize(456, "   ", by="operator")

    assert result.ok is False


def test_merge_authorize_records_override_bound_to_live_head(tmp_path: Path) -> None:
    """The override is written into review-decision.json with the SHA and
    reason (issue #934, properties 2 and 3)."""
    app, paths, _ = _merge_check_app(tmp_path)

    result = app.merge_authorize(456, "CI green, stale decision overridden", by="senkichi")

    assert result.ok is True
    assert result.data["authorized"] is True
    assert result.data["authorized_sha"] == "sha-abc123"
    assert result.data["authorized_by"] == "senkichi"

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    assert decision_path.exists()
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    override = decision["authorized_override"]
    assert override["authorized_sha"] == "sha-abc123"
    assert override["reason"] == "CI green, stale decision overridden"
    assert override["by"] == "senkichi"
    assert "authorized_at" in override


def test_merge_authorize_preserves_existing_review_verdict(tmp_path: Path) -> None:
    """The override is merge-updated into the existing decision record, so the
    reviewer's original verdict is preserved alongside the operator's
    authorization (issue #934)."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(tmp_path, 456, {"decision": "request_changes", "reviewed_head_sha": "sha-old"})

    result = app.merge_authorize(456, "overridden after rebase", by="senkichi")

    assert result.ok is True
    decision_path = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-decision.json"
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    # Original verdict preserved
    assert decision["decision"] == "request_changes"
    assert decision["reviewed_head_sha"] == "sha-old"
    # Override added
    assert decision["authorized_override"]["authorized_sha"] == "sha-abc123"


def test_authorized_override_survives_subsequent_record_review(tmp_path: Path) -> None:
    """Regression test for issue #934 review finding: ``record_review`` builds a
    fresh ``decision_payload`` and overwrites ``review-decision.json`` in full,
    so an ``authorized_override`` written by ``merge_authorize`` would be
    silently discarded — resurrecting the false-positive tripwire finding this
    PR exists to eliminate. The override must survive a later ``record_review``
    call for the same PR (a plausible sequential scenario, not just a race:
    issue #934's own 'pending review' use case is a reviewer re-recording a
    verdict after the operator already authorized)."""
    app, paths, _ = _merge_check_app(tmp_path)

    # 1. Operator records an authorization at the live head.
    auth_result = app.merge_authorize(456, "CI green, stale decision overridden", by="senkichi")
    assert auth_result.ok is True

    decision_path = paths.prs / "pr-456" / "review-decision.json"
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    override_before = decision["authorized_override"]
    assert override_before["authorized_sha"] == "sha-abc123"
    assert override_before["reason"] == "CI green, stale decision overridden"

    # 2. A reviewer subsequently records a verdict for the same PR. This is the
    #    sequential (not racy) scenario: the operator authorized, then a review
    #    round completes and record_review overwrites the decision file.
    review_result = app.record_review(
        456, "approved", summary="lgtm after rebase", verdict_provenance="fresh_llm_review"
    )
    assert review_result.ok is True

    # 3. The override must survive the full-file overwrite.
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    assert decision["decision"] == "approved"
    assert "authorized_override" in decision, (
        "record_review's full-file overwrite discarded the authorized_override "
        "written by merge_authorize — the tripwire finding this PR exists to "
        "eliminate would resurrect"
    )
    override_after = decision["authorized_override"]
    assert override_after["authorized_sha"] == "sha-abc123"
    assert override_after["reason"] == "CI green, stale decision overridden"
    assert override_after["by"] == "senkichi"

    # 4. The surviving override must still authorize via merge_check — proving
    #    the preserved override is structurally valid, not just present as a
    #    stale key.
    check_result = app.merge_check(456)
    assert check_result.ok is True
    assert check_result.data["reason"] == "authorized_override"


def test_merge_authorize_creates_decision_file_when_absent(tmp_path: Path) -> None:
    """If no review-decision.json exists, one is created with just the
    override — the reviewer verdict is absent, and the override is the
    authorization (issue #934)."""
    app, paths, _ = _merge_check_app(tmp_path)
    assert not (paths.prs / "pr-456" / "review-decision.json").exists()

    result = app.merge_authorize(456, "no review needed, operator adjudicated", by="op")

    assert result.ok is True
    decision_path = paths.prs / "pr-456" / "review-decision.json"
    assert decision_path.exists()
    with decision_path.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    assert "authorized_override" in decision
    assert decision["authorized_override"]["authorized_sha"] == "sha-abc123"


def test_merge_authorize_explicit_sha_binds_to_specified_sha(tmp_path: Path) -> None:
    """--sha overrides the live head, binding the authorization to a specific
    SHA (issue #934, property 3)."""
    app, _, _ = _merge_check_app(tmp_path)

    result = app.merge_authorize(456, "authorized specific commit", by="op", sha="sha-explicit")

    assert result.ok is True
    assert result.data["authorized_sha"] == "sha-explicit"


def test_merge_authorize_reachable_through_cli(tmp_path: Path) -> None:
    """Wiring check (L3): a hook or operator shelling out to
    `charlie merge-authorize` reaches it through this path only."""
    app, _, _ = _merge_check_app(tmp_path)

    args = cli.build_parser().parse_args(
        ["merge-authorize", "456", "--reason", "operator adjudicated", "--by", "senkichi"]
    )
    assert args.command == "merge-authorize"
    assert args.pr == 456

    result = cli.run_command(app, args)
    assert result.ok is True
    assert result.data["authorized"] is True


def test_merge_authorize_records_info_level_event(tmp_path: Path) -> None:
    """Issue #934 rework finding: ``merge_authorize`` emits a ``merge_authorized``
    event via ``self._record_event`` (dual-written to ``events.db``). That kind
    must be registered in ``_LEVEL_BY_KIND`` -- ``test_event_kind_registry_exhaustive``
    fails the build for any unregistered emit-site kind. An operator
    authorization is an audit fact, not a fault, so it must classify as
    ``info`` (sibling to ``unauthorized_merge_acknowledged``), never ``error``
    or ``warning``.

    The explicit ``_LEVEL_BY_KIND`` membership assertion is load-bearing:
    ``_classify_level`` defaults unknown kinds to ``"info"``, so the
    recorded-event level alone would pass even with the registry entry
    missing. Asserting the kind is a registered key with an ``info`` value
    makes this test fail against the unfixed (unregistered) code, not just
    against a releveling.
    """
    # The kind must be explicitly registered, not just falling through the
    # info default.
    assert "merge_authorized" in _LEVEL_BY_KIND, (
        "merge_authorized must be registered in _LEVEL_BY_KIND -- an "
        "unregistered emit-site kind fails test_event_kind_registry_exhaustive"
    )
    assert _LEVEL_BY_KIND["merge_authorized"] == "info", (
        "merge_authorized is an operator audit fact, not a fault -- it "
        "must classify as info, sibling to unauthorized_merge_acknowledged"
    )

    app, paths, _ = _merge_check_app(tmp_path)
    try:
        result = app.merge_authorize(456, "CI green, stale decision overridden", by="senkichi")
        assert result.ok is True

        events = query_events(paths.state_file, kind="merge_authorized")
        assert len(events) == 1
        event = events[0]
        assert event["kind"] == "merge_authorized"
        assert event["level"] == "info"
        assert event["pr_number"] == 456
    finally:
        close_db(paths.state_file)


# ---------------------------------------------------------------------------
# merge_check honors the override
# ---------------------------------------------------------------------------


def test_merge_check_authorizes_via_override_at_current_head(tmp_path: Path) -> None:
    """A PR with a stale request_changes decision but a valid override at the
    live head is authorized — this is the Class B case from issue #934."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(
        tmp_path,
        456,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "senkichi",
                "reason": "CI green, stale decision overridden",
                "authorized_sha": "sha-abc123",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    result = app.merge_check(456)

    assert result.ok is True
    assert result.data["authorized"] is True
    assert result.data["reason"] == "authorized_override"
    assert result.data["authorized_by"] == "senkichi"


def test_merge_check_override_with_wrong_sha_does_not_authorize(tmp_path: Path) -> None:
    """Property 3 (bind to the SHA): an override for a different SHA than the
    live head does not authorize — the head moved, re-authorize."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(
        tmp_path,
        456,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "senkichi",
                "reason": "authorized old head",
                "authorized_sha": "sha-different",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    result = app.merge_check(456)

    assert result.ok is False
    # Falls through to the existing not_approved check
    assert result.data["reason"] == "not_approved"


def test_merge_check_override_with_empty_reason_does_not_authorize(tmp_path: Path) -> None:
    """Property 2 (reason mandatory): an override with an empty reason is not a
    valid override — a control that can be silenced silently is no control."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(
        tmp_path,
        456,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "someone",
                "reason": "   ",
                "authorized_sha": "sha-abc123",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "not_approved"


def test_merge_check_override_missing_sha_does_not_authorize(tmp_path: Path) -> None:
    """A malformed override without a SHA is treated as absent — the control
    never reads a malformed record as authorization (issue #934, property 1)."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(
        tmp_path,
        456,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "someone",
                "reason": "valid reason",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    result = app.merge_check(456)

    assert result.ok is False
    assert result.data["reason"] == "not_approved"


def test_merge_check_approved_at_head_still_works_alongside_override(
    tmp_path: Path,
) -> None:
    """The existing approved-at-head path is not displaced by the override
    check — a PR with a genuine approval and no override still authorizes."""
    app, _, _ = _merge_check_app(tmp_path)
    _write_decision(tmp_path, 456, {"decision": "approved", "reviewed_head_sha": "sha-abc123"})

    result = app.merge_check(456)

    assert result.ok is True
    assert result.data["reason"] == "approved_at_head"


# ---------------------------------------------------------------------------
# Tripwire (_detect_unauthorized_merges) honors the override
# ---------------------------------------------------------------------------


def _tripwire_app(tmp_path: Path, merged_prs: list[dict]) -> tuple[OrchestratorApp, object]:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class _FakeGH(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.prs = list(merged_prs)

        def merged_pr_list(self):
            return list(merged_prs)

    app = OrchestratorApp(tmp_path, paths, config, _FakeGH())
    return app, paths


def test_tripwire_does_not_flag_pr_with_valid_override(tmp_path: Path) -> None:
    """The Class B case: a dispatched worker PR merged by the operator after
    recording an override. The tripwire must not flag it (issue #934)."""
    pr = _merged_worker_pr(601, 494, "sha-601")
    app, paths = _tripwire_app(tmp_path, [pr])
    # Write a stale decision with a valid override at the merged head
    _write_decision(
        tmp_path,
        601,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "senkichi",
                "reason": "CI green, content reviewed, stale decision overridden",
                "authorized_sha": "sha-601",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    detected = app._detect_unauthorized_merges()

    assert detected == []


def test_tripwire_flags_pr_with_override_at_wrong_sha(tmp_path: Path) -> None:
    """Property 3 (bind to the SHA): an override for a different SHA than the
    merged head is still a finding — the head moved after authorization."""
    pr = _merged_worker_pr(602, 495, "sha-602")
    app, paths = _tripwire_app(tmp_path, [pr])
    _write_decision(
        tmp_path,
        602,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "senkichi",
                "reason": "authorized old head",
                "authorized_sha": "sha-different",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 602


def test_tripwire_flags_pr_with_override_empty_reason(tmp_path: Path) -> None:
    """Property 2 (reason mandatory): an override with an empty reason does not
    suppress the finding."""
    pr = _merged_worker_pr(603, 496, "sha-603")
    app, paths = _tripwire_app(tmp_path, [pr])
    _write_decision(
        tmp_path,
        603,
        {
            "decision": "request_changes",
            "reviewed_head_sha": "sha-old",
            "authorized_override": {
                "by": "someone",
                "reason": "",
                "authorized_sha": "sha-603",
                "authorized_at": "2026-08-13T00:00:00Z",
            },
        },
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 603


def test_tripwire_flags_pr_with_no_override(tmp_path: Path) -> None:
    """Property 1 (do not weaken the control): an unrecorded merge is still a
    finding. The override mechanism adds a way to record, not a way to skip."""
    pr = _merged_worker_pr(604, 497, "sha-604")
    app, paths = _tripwire_app(tmp_path, [pr])
    _write_decision(
        tmp_path,
        604,
        {"decision": "request_changes", "reviewed_head_sha": "sha-old"},
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 604


def test_tripwire_override_suppresses_finding_for_missing_decision(tmp_path: Path) -> None:
    """An override in a decision file with no reviewer verdict (decision
    absent) still authorizes — the override is the authorization."""
    pr = _merged_worker_pr(605, 498, "sha-605")
    app, paths = _tripwire_app(tmp_path, [pr])
    # Only the override, no decision field
    _write_decision(
        tmp_path,
        605,
        {
            "authorized_override": {
                "by": "senkichi",
                "reason": "operator adjudicated, no review on record",
                "authorized_sha": "sha-605",
                "authorized_at": "2026-08-13T00:00:00Z",
            }
        },
    )

    detected = app._detect_unauthorized_merges()

    assert detected == []


# ---------------------------------------------------------------------------
# _authorized_override_matches unit tests
# ---------------------------------------------------------------------------


def test_authorized_override_matches_valid_override() -> None:
    from charlie_work.workflow import _authorized_override_matches

    decision = {
        "authorized_override": {
            "by": "op",
            "reason": "valid",
            "authorized_sha": "sha-1",
            "authorized_at": "2026-08-13T00:00:00Z",
        }
    }
    assert _authorized_override_matches(decision, "sha-1") is True


def test_authorized_override_does_not_match_wrong_sha() -> None:
    from charlie_work.workflow import _authorized_override_matches

    decision = {
        "authorized_override": {
            "by": "op",
            "reason": "valid",
            "authorized_sha": "sha-1",
            "authorized_at": "2026-08-13T00:00:00Z",
        }
    }
    assert _authorized_override_matches(decision, "sha-2") is False


def test_authorized_override_does_not_match_empty_reason() -> None:
    from charlie_work.workflow import _authorized_override_matches

    decision = {
        "authorized_override": {
            "by": "op",
            "reason": "",
            "authorized_sha": "sha-1",
            "authorized_at": "2026-08-13T00:00:00Z",
        }
    }
    assert _authorized_override_matches(decision, "sha-1") is False


def test_authorized_override_does_not_match_none_head() -> None:
    from charlie_work.workflow import _authorized_override_matches

    decision = {
        "authorized_override": {
            "by": "op",
            "reason": "valid",
            "authorized_sha": "sha-1",
            "authorized_at": "2026-08-13T00:00:00Z",
        }
    }
    assert _authorized_override_matches(decision, None) is False


def test_authorized_override_does_not_match_no_override() -> None:
    from charlie_work.workflow import _authorized_override_matches

    assert _authorized_override_matches({"decision": "approved"}, "sha-1") is False


def test_authorized_override_does_not_match_non_dict_override() -> None:
    from charlie_work.workflow import _authorized_override_matches

    decision = {"authorized_override": "not a dict"}
    assert _authorized_override_matches(decision, "sha-1") is False
