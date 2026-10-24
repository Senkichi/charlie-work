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

from charlie_work import cli
from charlie_work.config import (
    ConfigError,
    DispatchConfig,
    OrchestratorConfig,
    build_config_from_data,
)
from charlie_work.host_routing import issue_hosts, owned_here
from charlie_work.paths import runtime_paths
from charlie_work.state import record_non_empty_dispatch
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
