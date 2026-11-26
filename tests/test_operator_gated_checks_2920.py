"""Issue #2920: a PR held by an operator-only required check is not sent to worker rework.

``dispatch.operator_gated_checks`` names required checks that only the operator can
clear. When every failed required check is in that list, the failure is the PR's sole
janitor failure, and the bound issue carries a ``dispatch.human_merge_labels`` label,
``review()`` parks the issue on the operator queue instead of recording
``request_changes`` and sending a worker to fix what no push can fix. Every other shape
(another check also failed, no human-merge label, an empty list, a co-occurring janitor
failure) keeps today's rework routing.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHubWithChecks
from _fakes_github_rerun import FakeGitHubWithRerunCapture
from _helpers import EXAMPLES_DIR
from _review_fixtures import _required_checks_config
from charlie_work.config import ConfigError, DispatchConfig, OrchestratorConfig, load_config
from charlie_work.instrumentation import query_events
from charlie_work.janitor import JanitorVerdict
from charlie_work.operator_gated_route import (
    OPERATOR_GATED_ESCALATION_REASON,
    is_operator_gated_failure,
    park_operator_gated_review,
    route_operator_gated_pr,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

_DIFF = "diff --git a/file b/file\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new"
_RUN_LINK = "https://github.com/owner/repo/actions/runs/12345/job/67890"
_GATED = "Tests passed"
_HUMAN_MERGE = "needs-design"

_ONLY_GATED_FAILS = [
    {"name": "Tests passed", "state": "FAILURE"},
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "SUCCESS"},
]
_GATED_AND_OTHER_FAIL = [
    {"name": "Tests passed", "state": "FAILURE"},
    {"name": "Lint & Format", "bucket": "pass"},
    {"name": "Pre-commit", "state": "FAILURE"},
]


def _config(
    *,
    gated: tuple[str, ...] = (_GATED,),
    human_merge: tuple[str, ...] = (_HUMAN_MERGE,),
) -> OrchestratorConfig:
    base = _required_checks_config()
    dispatch = dataclasses.replace(
        base.dispatch, operator_gated_checks=gated, human_merge_labels=human_merge
    )
    return dataclasses.replace(base, dispatch=dispatch)


def _app(
    tmp_path: Path,
    config: OrchestratorConfig,
    *,
    checks: list[dict[str, Any]] | None = None,
    issue_labels: tuple[str, ...] = (_HUMAN_MERGE,),
    gh_cls: type[FakeGitHubWithChecks] = FakeGitHubWithChecks,
) -> tuple[OrchestratorApp, FakeGitHubWithChecks, Any]:
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = gh_cls(checks=list(_ONLY_GATED_FAILS if checks is None else checks))
    fake_gh.issues[0]["labels"] = [{"name": name} for name in issue_labels]
    fake_gh.diffs[456] = _DIFF
    return OrchestratorApp(tmp_path, paths, config, fake_gh), fake_gh, paths


def _assert_reworked_as_today(app, fake_gh, paths, config, result) -> None:
    assert result.ok is True
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "rework_requested"
    assert state["prs"]["456"]["decision"] == "request_changes"
    assert (123, config.labels.needs_rework) in fake_gh.labels_added
    assert (123, config.labels.operator_queue) not in fake_gh.labels_added
    assert (paths.prs / "pr-456" / "rework-prompt.md").exists()
    assert query_events(paths.state_file, kind="review_dispatch_escalated") == []


# --------------------------------------------------------------------------- #
# Acceptance 1: only failed check is gated + human-merge label -> operator queue
# --------------------------------------------------------------------------- #


def test_only_gated_check_failed_with_human_merge_label_parks_on_operator_queue(
    tmp_path: Path,
) -> None:
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)

    result = app.review(456)

    assert result.ok is False
    assert result.data["operator_gated"] is True
    assert result.data["escalated"] is True
    assert result.data["label_error"] is None
    assert result.data["checks_unavailable"] is False
    state = load_state(paths.state_file)
    # No request_changes verdict: nothing recorded, no rework brief written.
    assert state["prs"]["456"].get("decision") != "request_changes"
    assert state["prs"]["456"].get("request_changes_count", 0) == 0
    assert not (paths.prs / "pr-456" / "rework-prompt.md").exists()
    assert not (paths.prs / "pr-456" / "review-decision.json").exists()
    # No rework dispatch: dispatch_rework selects only status == rework_requested.
    issue = state["issues"]["123"]
    assert issue["status"] == "escalated"
    assert issue["reason_class"] == "mechanical"
    assert issue["escalation_reason"] == OPERATOR_GATED_ESCALATION_REASON
    assert state["prs"]["456"]["status"] == "escalated"
    # The issue goes to the operator queue, not human-needed and not needs-rework.
    assert (123, config.labels.operator_queue) in fake_gh.labels_added
    assert (123, config.labels.needs_rework) not in fake_gh.labels_added
    assert (123, config.labels.human_needed) not in fake_gh.labels_added
    # The CI-red short-circuit's provenance event is for rework; this is not that.
    assert query_events(paths.state_file, kind="review_dispatch_skipped_ci_red") == []
    (event,) = query_events(paths.state_file, kind="review_dispatch_escalated")
    assert event["payload"]["reason"] == OPERATOR_GATED_ESCALATION_REASON
    assert event["payload"]["pr_number"] == 456
    assert event["payload"]["issue_number"] == 123
    assert event["payload"]["failed_required_checks"] == [_GATED]
    assert event["payload"]["head_sha"] == "sha-abc123"


def test_operator_gated_check_is_not_flake_rerun(tmp_path: Path) -> None:
    """A rerun cannot clear a check that waits for an operator, so the route runs first.

    The failed check carries a workflow-run link, which is exactly what makes the
    flake-aware debounce (#391) trigger ``gh run rerun --failed`` on the first
    failure of a head.
    """
    checks = [{**_ONLY_GATED_FAILS[0], "link": _RUN_LINK}, *_ONLY_GATED_FAILS[1:]]
    app, fake_gh, paths = _app(
        tmp_path, _config(), checks=checks, gh_cls=FakeGitHubWithRerunCapture
    )

    result = app.review(456)

    assert result.data["operator_gated"] is True
    assert fake_gh.rerun_calls == []


def test_operator_gated_escalation_survives_the_escalated_label_repair_sweep(
    tmp_path: Path,
) -> None:
    """The route is ``mechanical`` so the repair sweep re-derives ``operator-queue``.

    A ``policy`` escalation sitting on the same label is re-derived to
    ``agent:human-needed`` by that sweep (it maps every non-mechanical class to the
    ``escalated`` edge). Running the real sweep after the route is what pins the
    class choice.
    """
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)
    app.review(456)
    assert load_state(paths.state_file)["issues"]["123"]["reason_class"] == "mechanical"

    app.dispatch_reviews()

    assert (123, config.labels.human_needed) not in fake_gh.labels_added
    assert (123, config.labels.operator_queue) in fake_gh.labels_added


def test_operator_gated_escalation_clears_once_the_operator_clears_the_check(
    tmp_path: Path,
) -> None:
    """The lifecycle after the route: stays parked while red, resumes when green.

    The mechanical de-escalation sweep is the route's only way back into review, so
    its ``_NO_COUNTER_ALLOWLIST`` justification (test_issue_1683) is pinned here:
    the red check is a janitor blocker the sweep will not clear through, and the
    same check going green clears the escalation.
    """
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)
    app.review(456)
    assert load_state(paths.state_file)["issues"]["123"]["status"] == "escalated"

    app._maybe_deescalate_mechanical()
    red_state = load_state(paths.state_file)
    still_red = red_state["issues"]["123"]
    assert still_red["status"] == "escalated"
    (red_pass,) = [e for e in red_state["events"] if e["kind"] == "deescalation_pass_completed"]
    assert red_pass["payload"]["skipped"] == {"janitor_blocked": 1}
    assert still_red["escalation_reason"] == OPERATOR_GATED_ESCALATION_REASON

    fake_gh.checks = [
        {"name": "Tests passed", "state": "SUCCESS"},
        {"name": "Lint & Format", "bucket": "pass"},
        {"name": "Pre-commit", "state": "SUCCESS"},
    ]
    # The sweep is interval-throttled; an absent schedule is "due immediately".
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state.pop("deescalation_pass", None)
        save_state(paths.state_file, state)
    app._maybe_deescalate_mechanical()

    cleared = load_state(paths.state_file)["issues"]["123"]
    assert cleared["status"] != "escalated"
    assert "escalation_reason" not in cleared


# --------------------------------------------------------------------------- #
# Acceptance 2-4: every other shape keeps today's rework routing
# --------------------------------------------------------------------------- #


def test_another_required_check_also_failed_reworks_as_today(tmp_path: Path) -> None:
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config, checks=_GATED_AND_OTHER_FAIL)

    result = app.review(456)

    _assert_reworked_as_today(app, fake_gh, paths, config, result)
    prompt = (paths.prs / "pr-456" / "rework-prompt.md").read_text(encoding="utf-8")
    assert "Tests passed" in prompt and "Pre-commit" in prompt


def test_bound_issue_without_human_merge_label_reworks_as_today(tmp_path: Path) -> None:
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config, issue_labels=("unrelated",))

    result = app.review(456)

    _assert_reworked_as_today(app, fake_gh, paths, config, result)


def test_empty_operator_gated_checks_reworks_as_today(tmp_path: Path) -> None:
    """Default config: the human-merge label alone changes nothing."""
    config = _config(gated=())
    assert config.dispatch.operator_gated_checks == ()
    app, fake_gh, paths = _app(tmp_path, config)

    result = app.review(456)

    _assert_reworked_as_today(app, fake_gh, paths, config, result)


def test_no_human_merge_labels_configured_reworks_as_today(tmp_path: Path) -> None:
    config = _config(human_merge=())
    app, fake_gh, paths = _app(tmp_path, config)

    result = app.review(456)

    _assert_reworked_as_today(app, fake_gh, paths, config, result)


def test_co_occurring_janitor_failure_reworks_as_today(tmp_path: Path) -> None:
    """A body the worker can fix beside a gated check is not an operator-only hold."""
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)
    fake_gh.prs[0]["body"] = ""

    result = app.review(456)

    _assert_reworked_as_today(app, fake_gh, paths, config, result)
    prompt = (paths.prs / "pr-456" / "rework-prompt.md").read_text(encoding="utf-8")
    assert "PR body is empty" in prompt


# --------------------------------------------------------------------------- #
# route_operator_gated_pr: silent and failure exits
# --------------------------------------------------------------------------- #


def _route(app, paths, fake_gh, config):
    return route_operator_gated_pr(
        state_file=paths.state_file,
        write_gate=app.write_gate,
        gh=fake_gh,
        labels=config.labels,
        pr_number=456,
        issue_number=123,
        failed_checks=(_GATED,),
        head_sha="sha-abc123",
    )


def test_route_stays_silent_when_another_lane_already_escalated(tmp_path: Path) -> None:
    """A sink status set since the caller's read wins; its reason is not clobbered (#1461)."""
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "status": "escalated",
            "escalation_reason": "other_lane",
            "reason_class": "judgment",
        }
        save_state(paths.state_file, state)

    outcome = _route(app, paths, fake_gh, config)

    assert outcome.routed is False
    issue = load_state(paths.state_file)["issues"]["123"]
    assert issue["escalation_reason"] == "other_lane"
    assert issue["reason_class"] == "judgment"
    assert fake_gh.labels_added == []
    assert query_events(paths.state_file, kind="review_dispatch_escalated") == []


class _FailingAddGitHub(FakeGitHubWithChecks):
    def add_issue_label(self, number: int, label: str) -> bool:
        self.labels_added.append((number, label))
        return False


def test_route_records_a_failed_label_edge_for_the_repair_sweep(tmp_path: Path) -> None:
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config, gh_cls=_FailingAddGitHub)

    result = app.review(456)

    assert result.data["escalated"] is True
    assert result.data["label_error"]["edge"] == "operator_queued"
    state = load_state(paths.state_file)
    assert state["issues"]["123"]["status"] == "escalated"
    assert state["issues"]["123"]["label_error"]["edge"] == "operator_queued"


# --------------------------------------------------------------------------- #
# park_operator_gated_review: the glue's non-routing exits
# --------------------------------------------------------------------------- #

_GATED_VERDICT = JanitorVerdict(
    ok=False,
    failures=(),
    warnings=(),
    failed_required_checks=(_GATED,),
    is_check_failure_block=True,
)


def _park(app, paths, fake_gh, config, *, verdict=_GATED_VERDICT, issue=None, issue_number=123):
    if issue is None:
        issue = {"number": 123, "labels": [{"name": _HUMAN_MERGE}]}
    return park_operator_gated_review(
        config=config,
        state_file=paths.state_file,
        write_gate=app.write_gate,
        gh=fake_gh,
        pr={"number": 456, "headRefOid": "sha-abc123"},
        pr_number=456,
        issue=issue,
        issue_number=issue_number,
        verdict=verdict,
    )


def test_park_routes_when_every_condition_holds(tmp_path: Path) -> None:
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)

    parked = _park(app, paths, fake_gh, config)

    assert parked is not None
    assert parked.data["operator_gated"] is True
    assert parked.data["escalated"] is True
    assert "held by operator-gated check(s) Tests passed" in parked.message
    assert load_state(paths.state_file)["issues"]["123"]["status"] == "escalated"


@pytest.mark.parametrize(
    "case",
    [
        "no_issue_number",
        "janitor_failure_beside_the_check",
        "issue_payload_without_labels_key",
        "issue_payload_not_a_dict",
    ],
)
def test_park_returns_none_and_writes_nothing(tmp_path: Path, case: str) -> None:
    config = _config()
    app, fake_gh, paths = _app(tmp_path, config)
    kwargs: dict[str, Any] = {}
    if case == "no_issue_number":
        kwargs["issue_number"] = None
    elif case == "janitor_failure_beside_the_check":
        kwargs["verdict"] = dataclasses.replace(
            _GATED_VERDICT, failures=("PR body is empty",), is_check_failure_block=False
        )
    elif case == "issue_payload_without_labels_key":
        kwargs["issue"] = {"number": 123}
    else:
        kwargs["issue"] = ["not", "a", "dict"]

    assert _park(app, paths, fake_gh, config, **kwargs) is None

    assert fake_gh.labels_added == []
    assert query_events(paths.state_file, kind="review_dispatch_escalated") == []


# --------------------------------------------------------------------------- #
# Predicate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("gated", "human_merge", "failed", "labels", "expected"),
    [
        ((_GATED,), (_HUMAN_MERGE,), (_GATED,), {_HUMAN_MERGE}, True),
        ((_GATED, "B"), (_HUMAN_MERGE,), (_GATED,), {_HUMAN_MERGE}, True),
        ((_GATED, "B"), (_HUMAN_MERGE,), (_GATED, "B"), {_HUMAN_MERGE, "x"}, True),
        ((), (_HUMAN_MERGE,), (_GATED,), {_HUMAN_MERGE}, False),
        ((_GATED,), (_HUMAN_MERGE,), (_GATED, "other"), {_HUMAN_MERGE}, False),
        ((_GATED,), (_HUMAN_MERGE,), ("other",), {_HUMAN_MERGE}, False),
        ((_GATED,), (_HUMAN_MERGE,), (), {_HUMAN_MERGE}, False),
        ((_GATED,), (_HUMAN_MERGE,), (_GATED,), {"x"}, False),
        ((_GATED,), (_HUMAN_MERGE,), (_GATED,), set(), False),
        ((_GATED,), (), (_GATED,), {_HUMAN_MERGE}, False),
        # An unreadable label set must never become a skipped rework.
        ((_GATED,), (_HUMAN_MERGE,), (_GATED,), None, False),
    ],
)
def test_is_operator_gated_failure(gated, human_merge, failed, labels, expected) -> None:
    dispatch = DispatchConfig(operator_gated_checks=gated, human_merge_labels=human_merge)
    assert is_operator_gated_failure(dispatch, failed, labels) is expected


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_dispatch_config_operator_gated_checks_default_empty() -> None:
    assert DispatchConfig().operator_gated_checks == ()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Tests passed", ("Tests passed",)), (["a", "b"], ("a", "b")), (("a",), ("a",))],
)
def test_dispatch_config_operator_gated_checks_normalizes_to_tuple(raw, expected) -> None:
    assert DispatchConfig(operator_gated_checks=raw).operator_gated_checks == expected


def test_load_config_operator_gated_checks_yaml_list(tmp_path: Path) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "dispatch:\n  operator_gated_checks:\n    - Tests passed\n    - Sign-off\n",
        encoding="utf-8",
    )
    assert load_config(cfg_path).dispatch.operator_gated_checks == ("Tests passed", "Sign-off")


@pytest.mark.parametrize("body", ["42", "\n    - 42", "null"])
def test_load_config_operator_gated_checks_rejects_bad_values(tmp_path: Path, body: str) -> None:
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(f"dispatch:\n  operator_gated_checks: {body}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


@pytest.mark.parametrize(
    "name", ["orchestrator.config.devin.yaml", "orchestrator.config.claude-code.yaml"]
)
def test_example_configs_document_operator_gated_checks(name: str) -> None:
    """Both shipped examples document the knob (commented out) and load with it empty."""
    assert "operator_gated_checks" in (EXAMPLES_DIR / name).read_text(encoding="utf-8")
    assert load_config(EXAMPLES_DIR / name).dispatch.operator_gated_checks == ()
