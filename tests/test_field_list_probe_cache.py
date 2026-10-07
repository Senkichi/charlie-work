"""Field-list validation is cached per process + on disk (issue #2438)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _fake_transport import (
    FakeAdapter,
    connection_page,
    graphql_failure,
    graphql_variables,
    make_github,
    ok,
)

from ci_fleet.github import GitHubError

from charlie_work.config import ConfigError
from charlie_work.github_capabilities import _field_list_stamp as stamp
from charlie_work.github_capabilities.circuit_breaker_transport import (
    circuit_breaker_state_path,
)
from charlie_work.github_transport.outcome import GraphQLError, Response
from charlie_work.github_transport.request import RestRequest

_REJECT = Response(
    200,
    (),
    '{"data": null}',
    "http",
    graphql_errors=(
        GraphQLError("Field 'bogus' doesn't exist on type 'Issue'", "undefinedField"),
    ),
)


def _schema_accepting_reply(request):
    """Every probed selection is valid: empty connection, NOT_FOUND for views, empty REST."""
    if isinstance(request, RestRequest):
        return ok({"workflow_runs": []} if "actions/runs" in request.route else [])
    if "number" in graphql_variables(request):
        return graphql_failure("Could not resolve to a node with the number of 0.")
    connection = "issues" if "issues(" in request.document else "pullRequests"
    return connection_page(connection, [])


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(stamp.KILL_SWITCH_ENV, raising=False)
    stamp._VALIDATED.clear()
    yield
    stamp._VALIDATED.clear()


def _construct_and_validate(tmp_path: Path):
    """What each OrchestratorApp construction does: a fresh GitHub + validate."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=_schema_accepting_reply))
    gh.validate_field_lists()
    return gh, http


def test_probes_sent_once_across_repeated_constructions(tmp_path: Path) -> None:
    counts = [len(_construct_and_validate(tmp_path)[1].api_requests) for _ in range(4)]

    assert counts == [10, 0, 0, 0]


def test_disk_stamp_survives_a_new_process(tmp_path: Path) -> None:
    _construct_and_validate(tmp_path)
    stamp._VALIDATED.clear()  # a fresh process has only the disk stamp

    _gh, http = _construct_and_validate(tmp_path)

    assert len(http.api_requests) == 0


def test_stamp_expires_after_24h(tmp_path: Path) -> None:
    gh, _ = _construct_and_validate(tmp_path)
    path = circuit_breaker_state_path(gh.runtime, tmp_path).parent / stamp.STAMP_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    data["validated_at"] -= stamp.STAMP_TTL_SECONDS + 1
    path.write_text(json.dumps(data), encoding="utf-8")
    stamp._VALIDATED.clear()

    _gh, http = _construct_and_validate(tmp_path)

    assert len(http.api_requests) == 10


def test_stamp_invalidated_when_field_lists_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _construct_and_validate(tmp_path)
    stamp._VALIDATED.clear()
    monkeypatch.setattr(
        "charlie_work.github_capabilities.transport.RECONCILE_PR_FIELDS",
        "number,title",
    )

    _gh, http = _construct_and_validate(tmp_path)

    assert len(http.api_requests) == 10


def test_stamp_is_per_repo(tmp_path: Path) -> None:
    gh, _ = _construct_and_validate(tmp_path)
    stamp._VALIDATED.clear()
    gh2, http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=_schema_accepting_reply))
    gh2._transport._repo_owner_name = lambda: ("octo", "other")  # type: ignore[method-assign]

    gh2.validate_field_lists()

    assert len(http.api_requests) == 10


def test_unclean_validation_is_not_stamped(tmp_path: Path) -> None:
    bad = Response(422, (), '{"message":"Validation Failed"}', "http")
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=lambda _r: bad))
    gh.validate_field_lists()  # inconclusive, warns

    assert not stamp._VALIDATED
    assert not (
        circuit_breaker_state_path(gh.runtime, tmp_path).parent / stamp.STAMP_FILENAME
    ).exists()


def test_kill_switch_always_probes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(stamp.KILL_SWITCH_ENV, "off")

    counts = [len(_construct_and_validate(tmp_path)[1].api_requests) for _ in range(2)]

    assert counts == [10, 10]


def test_real_call_schema_rejection_clears_stamp_and_raises(tmp_path: Path) -> None:
    gh, http = _construct_and_validate(tmp_path)
    stamp_path = circuit_breaker_state_path(gh.runtime, tmp_path).parent / stamp.STAMP_FILENAME
    assert stamp_path.exists()
    http.handler = lambda _r: _REJECT

    with pytest.raises(ConfigError, match="bogus"):
        gh.pr_list()

    assert not stamp_path.exists()
    assert not stamp._VALIDATED
    # The next construction re-probes.
    assert len(_construct_and_validate(tmp_path)[1].api_requests) == 10


def test_real_call_rejection_without_cached_validation_keeps_github_error(tmp_path: Path) -> None:
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", handler=lambda _r: _REJECT))

    with pytest.raises(GitHubError):  # plain read failure, not ConfigError
        gh.pr_list()
