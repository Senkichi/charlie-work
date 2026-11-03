"""Issue #2763 part (a): the fleet only drives a PR it OWNS.

``issue_linking.linked_issue_number`` binds ANY same-repo PR that names an issue
with a closing keyword. Before this fix that binding was licence enough for the
merge lane to repair labels, run a fleet review and hand the PR to the merge
queue -- on 2026-10-19 it did so for a session-authored PR over an outstanding
human review. ``fleet_owns_pr`` is the single act-site gate; this module covers

1. the pure predicate, including the self-legitimising control (the labels and
   state entry adoption itself writes must NOT confer ownership);
2. the shared shell ``fleet_ownership.pr_fleet_owned`` (label source, fail-closed);
3. the ``reason`` carried by the ``pr_unlinked_skipped`` edge detector;
4. the act-sites: ``_loop_body``, ``review_queue`` and ``review()``.

Every negative has a positive control on the byte-identical fixture, so an
empty result cannot be a fixture that never armed the rule. ``detect_drift`` is
covered in ``test_reconcile_fleet_ownership_2763.py``.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub
from _review_fixtures import _review_queue_app, _write_review_packet

from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_ownership import NOT_FLEET_OWNED_REASON, pr_fleet_owned
from charlie_work.instrumentation import _LEVEL_BY_KIND, query_events
from charlie_work.issue_linking import fleet_owns_pr, linked_issue_number
from charlie_work.paths import runtime_paths
from charlie_work.pr_unlinked_visibility import (
    NO_LINKED_ISSUE_REASON,
    UNLINKED_PR_NOTICE_KEY,
    UnlinkedPrFingerprint,
    compute_unlinked_pr_transition,
)
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

CONFIG = OrchestratorConfig()
READY = CONFIG.labels.ready
ISSUE = 123
SESSION_BRANCH = "session/own-work"


def _pr(
    number: int = 456,
    *,
    branch: str = SESSION_BRANCH,
    issue: int = ISSUE,
    cross_repo: bool | None = False,
    head_sha: str | None = None,
) -> dict[str, Any]:
    """A same-repo PR that names ``issue`` with a closing keyword."""
    return {
        "number": number,
        "title": f"Fix #{issue}",
        "url": f"https://example.test/pull/{number}",
        "headRefName": branch,
        "baseRefName": "main",
        "headRefOid": head_sha or f"sha-{number}",
        "mergeStateStatus": "CLEAN",
        "mergeable": "MERGEABLE",
        "body": f"Closes #{issue}",
        "labels": [],
        "isCrossRepository": cross_repo,
        "state": "OPEN",
        "author": {"login": "a-human", "type": "User"},
    }


def _issue(labels: list[str] | None = None, number: int = ISSUE) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Fix search",
        "state": "OPEN",
        "labels": [{"name": name} for name in (labels or [])],
    }


def _make_app(
    tmp_path: Path,
    gh: FakeGitHub,
    config: OrchestratorConfig | None = None,
    *,
    dry_run: bool = False,
) -> OrchestratorApp:
    config = config or CONFIG
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    return OrchestratorApp(tmp_path, paths, config, gh, dry_run=dry_run)


def _without_gate(config: OrchestratorConfig = CONFIG) -> OrchestratorConfig:
    return dataclasses.replace(
        config, review=dataclasses.replace(config.review, require_fleet_ownership=False)
    )


# ---------------------------------------------------------------------------
# 1. The pure predicate
# ---------------------------------------------------------------------------


def test_keyword_only_pr_is_not_owned() -> None:
    """The headline shape: same-repo, off-prefix, issue not ready, no provenance."""
    assert (
        linked_issue_number(_pr(), is_cross_repository=False, branch_prefix="agent/issue-")
        == ISSUE
    )
    assert fleet_owns_pr(_pr(), (), None, CONFIG) is False


@pytest.mark.parametrize(
    ("pr", "labels", "entry"),
    [
        pytest.param(_pr(branch="agent/issue-123-x"), (), None, id="dispatch-branch-prefix"),
        pytest.param(_pr(), (READY,), None, id="ready-label"),
        pytest.param(_pr(), (), {"dispatched_at": "x"}, id="provenance-dispatched_at"),
        pytest.param(_pr(), (), {"branch_name": "agent/issue-123-x"}, id="provenance-branch_name"),
    ],
)
def test_each_ownership_signal_alone_is_enough(
    pr: dict[str, Any], labels: tuple[str, ...], entry: dict[str, Any] | None
) -> None:
    """Positive control for ``test_keyword_only_pr_is_not_owned``."""
    assert fleet_owns_pr(pr, labels, entry, CONFIG) is True


def test_branch_prefix_comes_from_config_not_a_literal() -> None:
    renamed = dataclasses.replace(
        CONFIG, dispatch=dataclasses.replace(CONFIG.dispatch, branch_prefix="bot/")
    )
    assert fleet_owns_pr(_pr(branch="bot/123-x"), (), None, renamed) is True
    assert fleet_owns_pr(_pr(branch="agent/issue-123-x"), (), None, renamed) is False


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({"number": ISSUE, "status": "open_passive"}, id="bare-open_passive"),
        pytest.param({"number": ISSUE, "status": "reviewing", "pr_number": 456}, id="reviewing"),
        pytest.param(
            {"status": "open_passive", "dispatched_at": None, "branch_name": ""},
            id="falsy-provenance",
        ),
        pytest.param({}, id="empty-entry"),
    ],
)
def test_adoption_artifacts_do_not_make_a_pr_owned(entry: dict[str, Any]) -> None:
    """Self-legitimising control: reconcile writes ``agent:pr-open`` and an
    ``open_passive`` state entry for whatever PR it adopts. If either counted,
    the first adoption would legitimise every later one."""
    labels = (CONFIG.labels.pr_open, CONFIG.labels.reviewing, CONFIG.labels.needs_rework)
    assert fleet_owns_pr(_pr(), labels, entry, CONFIG) is False


@pytest.mark.parametrize("cross_repo", [True, None])
def test_fork_or_unknown_provenance_is_never_owned(cross_repo: bool | None) -> None:
    """Even with every other signal present: a fork PR (or one whose provenance
    is unknown) is never the fleet's -- the same fail-closed rule as
    ``linked_issue_number``."""
    pr = _pr(branch="agent/issue-123-x", cross_repo=cross_repo)
    assert fleet_owns_pr(pr, (READY,), {"dispatched_at": "x"}, CONFIG) is False
    # Control: the same PR as a same-repo PR is owned.
    assert fleet_owns_pr(_pr(branch="agent/issue-123-x"), (READY,), {"dispatched_at": "x"}, CONFIG)


def test_kill_switch_treats_every_pr_as_owned() -> None:
    assert fleet_owns_pr(_pr(), (), None, CONFIG) is False
    assert fleet_owns_pr(_pr(), (), None, _without_gate()) is True


def test_require_fleet_ownership_defaults_on() -> None:
    assert CONFIG.review.require_fleet_ownership is True


# ---------------------------------------------------------------------------
# 2. The shell: labels from the open-issue list, state from the snapshot
# ---------------------------------------------------------------------------


def _gh(issues: list[dict[str, Any]]) -> FakeGitHub:
    gh = FakeGitHub()
    gh.issues = issues
    return gh


def test_shell_reads_the_ready_label_from_the_open_issue_list() -> None:
    state: dict[str, Any] = {"issues": {}}
    assert pr_fleet_owned(_gh([_issue([])]), CONFIG, _pr(), ISSUE, state) is False
    assert pr_fleet_owned(_gh([_issue([READY])]), CONFIG, _pr(), ISSUE, state) is True


def test_shell_issue_missing_from_the_open_list_contributes_no_labels() -> None:
    """Fail closed: an issue the list does not show (closed, past the limit, an
    unreadable list) can never ADOPT a PR through a label it was not seen with."""
    assert pr_fleet_owned(_gh([]), CONFIG, _pr(), ISSUE, {"issues": {}}) is False


def test_shell_reads_dispatch_provenance_from_the_state_snapshot() -> None:
    gh = _gh([_issue([])])
    bare = {"issues": {str(ISSUE): {"number": ISSUE, "status": "open_passive"}}}
    dispatched = {"issues": {str(ISSUE): {"status": "reviewing", "dispatched_at": "2000-01-01"}}}
    assert pr_fleet_owned(gh, CONFIG, _pr(), ISSUE, bare) is False
    assert pr_fleet_owned(gh, CONFIG, _pr(), ISSUE, dispatched) is True


def test_shell_makes_no_issue_list_call_when_the_branch_already_decides() -> None:
    """Labels are consulted last: a dispatch-branch PR costs no extra read."""

    class ExplodingGh:
        def issue_list(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("issue_list must not be called")

    assert pr_fleet_owned(
        ExplodingGh(), CONFIG, _pr(branch="agent/issue-123-x"), ISSUE, {"issues": {}}
    )


# ---------------------------------------------------------------------------
# 3. The edge detector carries the reason
# ---------------------------------------------------------------------------

_FP = UnlinkedPrFingerprint(mergeable="MERGEABLE", merge_state_status="CLEAN", head_sha="sha-1")
_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_transition_reason_change_re_emits_on_an_unchanged_fingerprint() -> None:
    marker, event = compute_unlinked_pr_transition(None, _FP, now=_NOW)
    assert event is not None
    # Same fingerprint, same (absent) reason: no event -- the pre-#2763 contract.
    assert compute_unlinked_pr_transition(marker, _FP, now=_NOW)[1] is None
    # Same fingerprint, NEW reason: announced once, then quiet again.
    marker, event = compute_unlinked_pr_transition(
        marker, _FP, now=_NOW, reason=NOT_FLEET_OWNED_REASON
    )
    assert event is not None
    assert marker["last_reason"] == NOT_FLEET_OWNED_REASON
    assert (
        compute_unlinked_pr_transition(marker, _FP, now=_NOW, reason=NOT_FLEET_OWNED_REASON)[1]
        is None
    )


def test_transition_legacy_marker_without_a_reason_stays_quiet_for_the_old_case() -> None:
    marker, _ = compute_unlinked_pr_transition(None, _FP, now=_NOW)
    assert "last_reason" not in marker
    assert compute_unlinked_pr_transition(marker, _FP, now=_NOW, reason=None)[1] is None


def test_not_fleet_owned_event_level_is_registered_as_warning() -> None:
    assert _LEVEL_BY_KIND["review_refused_not_fleet_owned"] == "warning"


# ---------------------------------------------------------------------------
# 4a. _loop_body: an un-owned keyword-linked PR goes to the unlinked path
# ---------------------------------------------------------------------------


def _loop_fixture(
    tmp_path: Path, *, issue_labels: list[str], **app_kwargs: Any
) -> tuple[OrchestratorApp, FakeGitHub, list[int]]:
    gh = FakeGitHub()
    gh.issues = [_issue(issue_labels)]
    gh.prs = [_pr()]
    app = _make_app(tmp_path, gh, **app_kwargs)
    reviewed: list[int] = []

    def spy_review(pr_number: int, **_kwargs: Any) -> Any:
        reviewed.append(pr_number)
        from charlie_work.command_result import CommandResult

        return CommandResult(False, "stub review", {"pr": pr_number})

    app.review = spy_review  # type: ignore[method-assign]
    return app, gh, reviewed


def test_loop_routes_an_unowned_keyword_pr_to_pr_unlinked_skipped(tmp_path: Path) -> None:
    app, gh, reviewed = _loop_fixture(tmp_path, issue_labels=[])

    assert app.loop(limit=0).ok is True

    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert [e["payload"]["pr_number"] for e in events] == [456]
    assert events[0]["payload"]["reason"] == NOT_FLEET_OWNED_REASON
    assert events[0]["level"] == "warning"
    # Nothing downstream of the gate ran for it.
    assert reviewed == []
    assert gh.labels_added == []
    assert gh.pr_labels_added == []
    assert query_events(app.paths.state_file, kind="merge_ready") == []
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"][UNLINKED_PR_NOTICE_KEY]["last_reason"] == NOT_FLEET_OWNED_REASON
    assert state["issues"].get(str(ISSUE), {}).get("pr_number") is None


def test_loop_unowned_pr_is_announced_once_not_every_pass(tmp_path: Path) -> None:
    app, _gh, _reviewed = _loop_fixture(tmp_path, issue_labels=[])

    for _ in range(3):
        assert app.loop(limit=0).ok is True

    assert len(query_events(app.paths.state_file, kind="pr_unlinked_skipped")) == 1


def test_loop_owned_pr_is_unchanged(tmp_path: Path) -> None:
    """Positive control on the byte-identical fixture: with the ready label the
    PR is driven as before -- it reaches ``review()`` and gets no skip event."""
    app, _gh, reviewed = _loop_fixture(tmp_path, issue_labels=[READY])

    assert app.loop(limit=0).ok is True

    assert reviewed == [456]
    assert query_events(app.paths.state_file, kind="pr_unlinked_skipped") == []


def test_loop_dispatch_branch_pr_is_unchanged(tmp_path: Path) -> None:
    gh = FakeGitHub()
    gh.issues = [_issue([])]
    gh.prs = [_pr(branch="agent/issue-123-fix-search")]
    app = _make_app(tmp_path, gh)
    reviewed: list[int] = []

    def spy_review(pr_number: int, **_kwargs: Any) -> Any:
        reviewed.append(pr_number)
        from charlie_work.command_result import CommandResult

        return CommandResult(False, "stub review", {"pr": pr_number})

    app.review = spy_review  # type: ignore[method-assign]

    assert app.loop(limit=0).ok is True
    assert reviewed == [456]
    assert query_events(app.paths.state_file, kind="pr_unlinked_skipped") == []


def test_loop_kill_switch_restores_adoption(tmp_path: Path) -> None:
    app, _gh, reviewed = _loop_fixture(tmp_path, issue_labels=[], config=_without_gate())

    assert app.loop(limit=0).ok is True

    assert reviewed == [456]
    assert query_events(app.paths.state_file, kind="pr_unlinked_skipped") == []


def test_loop_dry_run_persists_no_ownership_marker(tmp_path: Path) -> None:
    app, _gh, reviewed = _loop_fixture(tmp_path, issue_labels=[], dry_run=True)

    assert app.loop(limit=0).ok is True

    assert reviewed == []
    assert query_events(app.paths.state_file, kind="pr_unlinked_skipped") == []


def test_ownership_resolving_later_evicts_the_not_fleet_owned_marker(tmp_path: Path) -> None:
    """Falling edge (#1781) still works for the new reason: once the issue
    becomes ready the PR is linked AND owned, so its marker is evicted."""
    app, gh, _reviewed = _loop_fixture(tmp_path, issue_labels=[])
    app.loop(limit=0)
    assert UNLINKED_PR_NOTICE_KEY in load_state(app.paths.state_file)["prs"]["456"]

    gh.issues = [_issue([READY])]
    app.loop(limit=0)

    assert UNLINKED_PR_NOTICE_KEY not in (load_state(app.paths.state_file)["prs"].get("456") or {})
    assert len(query_events(app.paths.state_file, kind="pr_unlinked_resolved")) == 1


def test_loop_issueless_pr_event_still_says_no_linked_issue(tmp_path: Path) -> None:
    """The pre-existing case keeps its meaning under the new ``reason`` field."""
    gh = FakeGitHub()
    gh.issues = []
    pr = _pr()
    # The title ("Fix #123") also binds an issue, so clear both.
    pr["title"], pr["body"] = "Dependency bump", "no closing keyword"
    gh.prs = [pr]
    app = _make_app(tmp_path, gh)

    app.loop(limit=0)

    events = query_events(app.paths.state_file, kind="pr_unlinked_skipped")
    assert [e["payload"]["reason"] for e in events] == [NO_LINKED_ISSUE_REASON]


# ---------------------------------------------------------------------------
# 4b. review_queue
# ---------------------------------------------------------------------------


def _queue_app(tmp_path: Path, *, issue_labels: list[str], **kwargs: Any) -> OrchestratorApp:
    app = _review_queue_app(tmp_path, prs=[_pr(100, issue=10), _pr(200, issue=20)], **kwargs)
    app.gh.issues = [_issue(issue_labels, 10), _issue([READY], 20)]
    _write_review_packet(tmp_path, 100, "sha-100")
    _write_review_packet(tmp_path, 200, "sha-200")
    return app


def test_review_queue_skips_an_unowned_pr_and_keeps_the_owned_one(tmp_path: Path) -> None:
    app = _queue_app(tmp_path, issue_labels=[])

    queue = app.review_queue().data["queue"]

    assert [entry["pr"] for entry in queue] == [200]


def test_review_queue_lists_the_pr_once_the_issue_is_ready(tmp_path: Path) -> None:
    """Positive control: the same two PRs, both owned."""
    app = _queue_app(tmp_path, issue_labels=[READY])

    assert sorted(entry["pr"] for entry in app.review_queue().data["queue"]) == [100, 200]


def test_review_queue_provenance_makes_a_pr_owned(tmp_path: Path) -> None:
    app = _queue_app(tmp_path, issue_labels=[])
    state = load_state(app.paths.state_file)
    state["issues"]["10"] = {"number": 10, "status": "reviewing", "dispatched_at": "2000-01-01"}
    from charlie_work.state import save_state

    save_state(app.paths.state_file, state)

    assert sorted(entry["pr"] for entry in app.review_queue().data["queue"]) == [100, 200]


def test_review_queue_kill_switch(tmp_path: Path) -> None:
    app = _review_queue_app(tmp_path, prs=[_pr(100, issue=10)])
    app.config = _without_gate(app.config)
    app.gh.issues = [_issue([], 10)]
    _write_review_packet(tmp_path, 100, "sha-100")

    assert [entry["pr"] for entry in app.review_queue().data["queue"]] == [100]


# ---------------------------------------------------------------------------
# 4c. review(): the defence-in-depth refusal
# ---------------------------------------------------------------------------


def _review_app(
    tmp_path: Path, *, issue_labels: list[str], dry_run: bool = False
) -> OrchestratorApp:
    app = _review_queue_app(tmp_path, prs=[_pr(7, issue=ISSUE)], dry_run=dry_run)
    app.gh.issues = [_issue(issue_labels)]
    return app


def test_review_refuses_an_unowned_pr_before_any_write(tmp_path: Path) -> None:
    app = _review_app(tmp_path, issue_labels=[])

    result = app.review(7)

    assert result.ok is False
    assert result.data == {"pr": 7, "issue": ISSUE, "not_fleet_owned": True}
    events = query_events(app.paths.state_file, kind="review_refused_not_fleet_owned")
    assert [e["payload"] for e in events] == [{"pr_number": 7, "issue_number": ISSUE}]
    assert events[0]["level"] == "warning"
    assert not (app.paths.prs / "pr-7").exists()
    assert app.gh.labels_added == []


def test_review_proceeds_for_an_owned_pr(tmp_path: Path) -> None:
    """Positive control: same PR, ready issue -- not refused (it may still fail
    later for unrelated fixture reasons, which is not what this pins)."""
    app = _review_app(tmp_path, issue_labels=[READY])

    result = app.review(7)

    assert "not_fleet_owned" not in result.data
    assert query_events(app.paths.state_file, kind="review_refused_not_fleet_owned") == []


def test_review_dry_run_refuses_without_writing_an_event(tmp_path: Path) -> None:
    app = _review_app(tmp_path, issue_labels=[], dry_run=True)

    result = app.review(7)

    assert result.ok is False
    assert result.data["not_fleet_owned"] is True
    assert query_events(app.paths.state_file, kind="review_refused_not_fleet_owned") == []


def test_review_kill_switch_restores_the_old_behaviour(tmp_path: Path) -> None:
    app = _review_app(tmp_path, issue_labels=[])
    app.config = _without_gate(app.config)

    result = app.review(7)

    assert "not_fleet_owned" not in result.data


def test_review_of_a_pr_with_no_linked_issue_is_out_of_scope(tmp_path: Path) -> None:
    """Ownership needs an issue to own it; an issue-less PR keeps its behaviour."""
    pr = _pr(7)
    pr["title"], pr["body"] = "Dependency bump", "no keyword"
    app = _review_queue_app(tmp_path, prs=[pr], dry_run=False)
    app.gh.issues = []

    result = app.review(7)

    assert "not_fleet_owned" not in result.data
