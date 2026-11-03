"""Host routing (spec C2, interface I-7): ``dispatch.host``, ``dispatch.default_host``
and the ``host:<name>`` issue label.

An issue labelled ``host:H`` belongs to the orchestrator whose ``dispatch.host`` is
``H``; an issue with no host label belongs to the default host; with
``dispatch.host`` unset every issue is local (the kill switch).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub

from charlie_work import cli
from charlie_work.config import (
    ConfigError,
    DispatchConfig,
    OrchestratorConfig,
    build_config_from_data,
)
from charlie_work.host_routing import issue_hosts, owned_here
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, record_non_empty_dispatch
from charlie_work.workflow import (
    OrchestratorApp,
    check_dispatch_staleness,
    classify_backlog_reachability,
)


def _config(host: str | None, *, default_host: bool = True) -> OrchestratorConfig:
    base = OrchestratorConfig()
    return replace(base, dispatch=replace(base.dispatch, host=host, default_host=default_host))


def _issue(number: int, names: set[str]) -> dict[str, Any]:
    return {"number": number, "labels": [{"name": n} for n in names], "state": "OPEN"}


def test_defaults_route_every_issue_locally() -> None:
    config = OrchestratorConfig()
    assert config.dispatch.host is None
    assert config.dispatch.default_host is True
    assert config.labels.host_prefix == "host:"
    assert owned_here({"host:macbook"}, config) is True


def test_yaml_sets_host_and_default_host() -> None:
    config = build_config_from_data({"dispatch": {"host": "macbook", "default_host": False}})
    assert config.dispatch.host == "macbook"
    assert config.dispatch.default_host is False


@pytest.mark.parametrize("bad", ["", 7, ["macbook"]])
def test_yaml_rejects_a_malformed_host(bad: object) -> None:
    with pytest.raises(ConfigError, match=r"dispatch\.host"):
        build_config_from_data({"dispatch": {"host": bad}})


@pytest.mark.parametrize("bad", ["yes", None])
def test_yaml_rejects_a_non_bool_default_host(bad: object) -> None:
    with pytest.raises(ConfigError, match=r"dispatch\.default_host"):
        build_config_from_data({"dispatch": {"default_host": bad}})


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (set(), ()),
        ({"automated-ready"}, ()),
        ({"host:macbook"}, ("macbook",)),
        ({"Host:MacBook "}, ("macbook",)),
        ({"host:macbook", "host:workstation"}, ("macbook", "workstation")),
        ({"host:"}, ()),
    ],
)
def test_issue_hosts(labels: set[str], expected: tuple[str, ...]) -> None:
    assert issue_hosts(labels, "host:") == expected


def test_issue_hosts_with_an_empty_prefix_is_routing_off() -> None:
    assert issue_hosts({"host:macbook"}, "") == ()


@pytest.mark.parametrize(
    ("host", "default_host", "labels", "expected"),
    [
        ("macbook", False, {"host:macbook"}, True),
        ("macbook", False, {"HOST:MacBook"}, True),
        ("macbook", False, set(), False),
        ("macbook", False, {"host:workstation"}, False),
        ("workstation", True, set(), True),
        ("workstation", True, {"host:macbook"}, False),
        ("workstation", True, {"host:workstation"}, True),
        ("workstation", True, {"host:macbook", "host:workstation"}, False),
        ("macbook", False, {"host:macbook", "host:workstation"}, False),
    ],
)
def test_owned_here(host: str, default_host: bool, labels: set[str], expected: bool) -> None:
    assert owned_here(labels, _config(host, default_host=default_host)) is expected


def test_empty_prefix_turns_routing_off() -> None:
    base = _config("macbook", default_host=False)
    config = replace(base, labels=replace(base.labels, host_prefix=""))
    assert owned_here(set(), config) is True


def test_dispatchable_skips_an_issue_owned_by_another_host(tmp_path: Path) -> None:
    config = _config("workstation", default_host=True)
    ready = config.labels.ready
    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, None
    )
    # Positive control: the same ready issue without the host label dispatches.
    assert app._is_dispatchable(_issue(1, {ready}), set()) is True
    assert app._is_dispatchable(_issue(2, {ready, "host:macbook"}), set()) is False


def test_dispatchable_on_the_mac_takes_only_its_own_label(tmp_path: Path) -> None:
    config = _config("macbook", default_host=False)
    ready = config.labels.ready
    app = OrchestratorApp(
        tmp_path, runtime_paths(tmp_path, config.runtime.state_dir), config, None
    )
    assert app._is_dispatchable(_issue(1, {ready, "host:macbook"}), set()) is True
    assert app._is_dispatchable(_issue(2, {ready}), set()) is False


class _FakeGh:
    """The GitHubLike surface classify_backlog_reachability reads; no network."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self._issues = issues
        self._list_cache: dict[tuple[str, Any], Any] = {}

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        return list(self._issues)

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        return set()

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        return None


def test_backlog_bins_a_foreign_ready_issue_as_other_host() -> None:
    config = _config("workstation", default_host=True)
    ready = config.labels.ready
    gh = _FakeGh([_issue(1, {ready}), _issue(2, {ready, "host:macbook"})])

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 1
    assert result["other_host"] == 1
    assert result["unreachable_examples"]["other_host"] == [2]


def test_status_line_names_other_host_when_nothing_is_local() -> None:
    config = _config("macbook", default_host=False)
    ready = config.labels.ready
    gh = _FakeGh([_issue(1, {ready}), _issue(2, {ready, "host:workstation"})])

    rendered = cli._render_backlog_reachability(classify_backlog_reachability(gh, config))

    assert "other_host=2" in rendered


def test_backlog_other_host_is_zero_with_routing_off() -> None:
    config = OrchestratorConfig()
    ready = config.labels.ready
    gh = _FakeGh([_issue(1, {ready}), _issue(2, {ready, "host:macbook"})])

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 2
    assert result["other_host"] == 0


def _stale_inputs(now: datetime) -> tuple[dict[str, Any], dict[str, object]]:
    old = (now - timedelta(minutes=120)).isoformat().replace("+00:00", "Z")
    state = record_non_empty_dispatch({}, old, [1])
    return state, {"observed": True, "open_total": 3, "dispatchable": 0, "other_host": 3}


def test_non_default_host_with_nothing_local_is_idle_not_stale() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    state, backlog = _stale_inputs(now)
    config = DispatchConfig(dispatch_staleness_minutes=60, host="macbook", default_host=False)

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is False
    assert result["reason"] == "not_default_host"


def test_default_host_keeps_the_repo_wide_stall_alarm() -> None:
    # Positive control: the same inputs on the default host still alarm.
    now = datetime.now(UTC).replace(microsecond=0)
    state, backlog = _stale_inputs(now)
    config = DispatchConfig(dispatch_staleness_minutes=60, host="workstation")

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True


def test_host_unset_with_default_host_false_keeps_the_stall_alarm() -> None:
    """The ``config.host is not None`` kill-switch clause in
    ``check_dispatch_staleness`` (issue #2821 review). With no host
    configured, routing is off entirely -- every issue is local -- so
    ``default_host`` alone must not trigger the not_default_host exemption:
    the repo-wide stall alarm applies exactly as before."""
    now = datetime.now(UTC).replace(microsecond=0)
    state, backlog = _stale_inputs(now)
    config = DispatchConfig(dispatch_staleness_minutes=60, host=None, default_host=False)

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dispatch_stale"


def test_non_default_host_wedged_dependency_chain_still_alarms() -> None:
    """Issue #2821 review: the not_default_host exemption must not swallow the
    #1110/#1682 dependency arm. A non-default host whose ready issues are all
    dependency-blocked on an idle root is wedged, not idle-by-design -- the
    dependency_root_blocker_idle alarm still raises for its own backlog."""
    now = datetime.now(UTC).replace(microsecond=0)
    old = (now - timedelta(minutes=180)).isoformat().replace("+00:00", "Z")
    state = record_non_empty_dispatch({}, old, [1])
    backlog: dict[str, Any] = {
        "observed": True,
        "open_total": 2,
        "dispatchable": 0,
        "other_host": 0,
        "blocked_by_open_dependency": 2,
        "dependency_root_blockers": [{"number": 2, "updated_at": old}],
    }
    config = DispatchConfig(
        dispatch_staleness_minutes=60,
        dependency_stall_minutes=60,
        host="macbook",
        default_host=False,
    )

    result = check_dispatch_staleness(state, config, backlog, now=now)

    assert result["stale"] is True
    assert result["reason"] == "dependency_root_blocker_idle"
    assert [r["issue"] for r in result["dependency_root_blockers"]] == [2]


# ---------------------------------------------------------------------------
# Intake (issue #2906): another host intakes the issues it owns.
# ---------------------------------------------------------------------------


class _ViewRecordingGitHub(FakeGitHub):
    """FakeGitHub over a given issue list that records every ``issue_view``."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        super().__init__()
        self.issues = issues
        self.viewed: list[int] = []

    def issue_view(self, number: int) -> dict[str, Any]:
        self.viewed.append(number)
        return super().issue_view(number)


def _intake_issue(number: int, names: set[str]) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Issue {number}",
        "url": f"https://example.test/issues/{number}",
        "body": "A plain body with no dependencies.",
        "labels": [{"name": name} for name in sorted(names)],
        "state": "OPEN",
    }


def _intake_ready_observed(paths: Any, number: int) -> list[dict[str, Any]]:
    return query_events(paths.state_file, kind="ready_observed", issue_number=number)


@pytest.mark.parametrize(
    ("host", "default_host", "local_extra", "foreign_extra"),
    [
        # A labelled issue for another host, seen from the default host.
        ("workstation", True, set(), {"host:macbook"}),
        # An unlabelled issue belongs to the default host, so a non-default host skips it.
        ("macbook", False, {"host:macbook"}, set()),
    ],
)
def test_intake_skips_a_ready_issue_another_host_owns(
    tmp_path: Path,
    host: str,
    default_host: bool,
    local_extra: set[str],
    foreign_extra: set[str],
) -> None:
    config = _config(host, default_host=default_host)
    ready = config.labels.ready
    local, foreign = 1, 2
    gh = _ViewRecordingGitHub(
        [
            _intake_issue(local, {ready, *local_extra}),
            _intake_issue(foreign, {ready, *foreign_extra}),
        ]
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.intake()

    assert result.ok is True
    # Positive control: the issue this host owns is taken in end to end...
    assert gh.viewed == [local]
    assert [entry["issue"] for entry in result.data["issues"]] == [local]
    assert (paths.issues / f"issue-{local}" / "issue.json").is_file()
    assert (paths.issues / f"issue-{local}" / "worker-prompt.md").is_file()
    assert len(_intake_ready_observed(paths, local)) == 1
    # ...while the foreign one costs no read, no folder, no prompt, no episode.
    assert not (paths.issues / f"issue-{foreign}").exists()
    assert _intake_ready_observed(paths, foreign) == []
    state = load_state(paths.state_file)
    assert str(local) in state["issues"]
    assert str(foreign) not in state["issues"]
    intake_events = [e for e in state["events"] if e.get("kind") == "intake"]
    assert [e["payload"]["issue_count"] for e in intake_events] == [1]


def test_intake_skips_every_issue_when_this_host_owns_none(tmp_path: Path) -> None:
    config = _config("macbook", default_host=False)
    gh = _ViewRecordingGitHub([_intake_issue(7, {config.labels.ready})])
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.intake()

    assert result.ok is True
    assert gh.viewed == []
    assert result.data["issues"] == []
    assert not (paths.issues / "issue-7").exists()
    assert _intake_ready_observed(paths, 7) == []
    assert "7" not in load_state(paths.state_file)["issues"]


def test_intake_takes_in_the_same_issue_when_dispatch_host_is_unset(tmp_path: Path) -> None:
    """The kill switch: with ``dispatch.host`` unset a ``host:<name>`` label is inert."""
    config = _config(None, default_host=False)
    ready = config.labels.ready
    gh = _ViewRecordingGitHub([_intake_issue(2, {ready, "host:macbook"})])
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, gh)

    result = app.intake()

    assert result.ok is True
    assert gh.viewed == [2]
    assert [entry["issue"] for entry in result.data["issues"]] == [2]
    assert (paths.issues / "issue-2" / "issue.json").is_file()
    assert (paths.issues / "issue-2" / "worker-prompt.md").is_file()
    assert len(_intake_ready_observed(paths, 2)) == 1
    assert "2" in load_state(paths.state_file)["issues"]
