"""Tests for issue #2314: the no-ready arm of
``classify_backlog_reachability`` splits in two.

An open issue carrying a parked/triage label
(``heartbeat.stale_mention_parked_labels`` -- ``blocked``, ``needs-design``,
``human-action``, ``question``, ``wontfix``, ``duplicate``, ``invalid``,
``tracker``, ``umbrella``, ``epic`` by default) was deliberately parked by a
human and bins as ``parked_unready``. ``missing_ready`` is left to name only
the genuinely un-triaged pool, so a "N issues need triage" reading is true
when it fires instead of being inflated by parked issues.

In its own module (not ``test_backlog_reachability.py``) because that file
sits exactly at its file-size ratchet mark; ``FakeGh``/``_issue`` are
duplicated from it (the tests/ cross-import guard forbids importing from a
sibling test module). The classifier is imported from
``charlie_work.backlog_reachability`` directly, matching
``tests/test_dependency_root_blocker_idle.py``.
"""

from __future__ import annotations

from typing import Any

from charlie_work import cli
from charlie_work.backlog_reachability import classify_backlog_reachability
from charlie_work.config import HeartbeatConfig, OrchestratorConfig


class FakeGh:
    """Minimal stub for the GitHubLike methods classify_backlog_reachability
    calls: issue_list(state=...) and the blocker-check surface
    (are_issues_open + the dependency cache that
    get_github_issue_dependencies reads). Duplicated from
    tests/test_backlog_reachability.py -- never touches the network."""

    def __init__(
        self,
        issues: list[dict[str, Any]],
        *,
        open_blockers: set[int] | None = None,
        dependencies: dict[int, list[int]] | None = None,
    ) -> None:
        self._issues = issues
        self.calls: list[dict[str, Any]] = []
        self._list_cache: dict[tuple[str, Any], Any] = {}
        for number, deps in (dependencies or {}).items():
            self._list_cache[("issue_dependencies", number)] = deps
        self._open_blockers = open_blockers or set()

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        self.calls.append({"labels": labels, "state": state})
        return list(self._issues)

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        return {n for n in issue_numbers if n in self._open_blockers}

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        return None


def _issue(number: int, names: set[str], body: str = "") -> dict[str, Any]:
    return {"number": number, "labels": [{"name": n} for n in names], "body": body}


def test_parked_label_without_ready_bins_as_parked_unready() -> None:
    # The observed fleet state the issue reports: a backlog full of
    # human-triaged, deliberately parked issues (``blocked``/``human-action``/
    # ``needs-design``), none carrying the ready label.
    config = OrchestratorConfig()
    issues = [
        _issue(1, {"blocked"}),
        _issue(2, {"human-action"}),
        _issue(3, {"needs-design"}),
    ]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)

    assert result["parked_unready"] == 3
    assert result["missing_ready"] == 0
    assert result["dispatchable"] == 0
    assert result["unreachable_examples"]["parked_unready"] == [1, 2, 3]
    assert "missing_ready" not in result["unreachable_examples"]


def test_unlabeled_issue_still_bins_as_missing_ready() -> None:
    # The genuinely un-triaged pool: no ready label and no parked label.
    config = OrchestratorConfig()
    gh = FakeGh([_issue(1, set())])

    result = classify_backlog_reachability(gh, config)

    assert result["missing_ready"] == 1
    assert result["parked_unready"] == 0
    assert result["unreachable_examples"]["missing_ready"] == [1]


def test_parked_unready_bins_still_partition_open_total() -> None:
    config = OrchestratorConfig()
    labels = config.labels
    issues = [
        _issue(1, {"blocked"}),  # parked_unready
        _issue(2, set()),  # missing_ready
        _issue(3, {labels.ready}),  # dispatchable
    ]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)

    assert result["parked_unready"] == 1
    assert result["missing_ready"] == 1
    assert result["dispatchable"] == 1
    # Every open issue lands in exactly one bin -- the counts partition
    # open_total.
    assert (
        result["missing_ready"]
        + result["parked_unready"]
        + result["terminal_label"]
        + result["active_label"]
        + result["operator_claimed"]
        + result["blocked_by_open_dependency"]
        + result["mention_covered_awaiting_operator"]
        + result["unidentified"]
        + result["dispatchable"]
        == result["open_total"]
    )


def test_parked_taxonomy_comes_from_config_not_a_hardcoded_list() -> None:
    # Spec compliance: the parked set is ``heartbeat.stale_mention_parked_labels``.
    # A label the operator added to the configured taxonomy bins as
    # parked_unready, and a default-taxonomy label dropped from the configured
    # set falls back to missing_ready.
    config = OrchestratorConfig(
        heartbeat=HeartbeatConfig(stale_mention_parked_labels=("custom-park",))
    )
    issues = [
        _issue(1, {"custom-park"}),  # configured parked label
        _issue(2, {"blocked"}),  # default taxonomy, absent from the configured set
    ]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)

    assert result["parked_unready"] == 1
    assert result["missing_ready"] == 1
    assert result["unreachable_examples"]["parked_unready"] == [1]
    assert result["unreachable_examples"]["missing_ready"] == [2]


def test_parked_check_only_fires_when_ready_label_is_absent() -> None:
    # A ready-labelled issue carrying a parked label is still gated by the
    # ready arm's own checks (terminal/active/claimed/dispatchable) -- the
    # parked split lives only inside the no-ready arm.
    config = OrchestratorConfig()
    labels = config.labels
    gh = FakeGh([_issue(1, {labels.ready, "blocked"})])

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 1
    assert result["parked_unready"] == 0
    assert result["missing_ready"] == 0


def test_renderer_names_parked_unready_separately_from_missing_ready() -> None:
    # When the whole backlog is un-dispatchable, the renderer names both halves
    # of the no-ready split so "N parked" does not read as "N need triage".
    config = OrchestratorConfig()
    gh = FakeGh([_issue(1, {"blocked"}), _issue(2, set())])

    result = classify_backlog_reachability(gh, config)
    rendered = cli._render_backlog_reachability(result)

    assert "0 dispatchable" in rendered
    assert "missing_ready=1" in rendered
    assert "parked_unready=1" in rendered
    assert rendered.isascii()
