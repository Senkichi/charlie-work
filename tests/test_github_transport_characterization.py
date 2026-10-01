"""Observable contracts the ADR-0006 migration had to keep (gt-design M1).

Each case drives a real ``GitHub`` over scripted adapters (``make_github``):
the dry-run returns, the error text downstream matchers read, and the
pending / no-checks / absent-label outcomes. Nothing touches the network,
``gh`` or the host clock.
"""

from __future__ import annotations

from pathlib import Path

from _fake_transport import FakeAdapter, check_run, checks_reply, graphql_failure, make_github, ok
from charlie_work.workflow import _is_rerun_already_running_error


def test_rerun_already_running_error_is_seen_by_the_matcher(tmp_path: Path) -> None:
    reply = ok({"message": "Cannot rerun a workflow that is already running"}, status=403)
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    result = gh.run(["run", "rerun", "123"], allow_failure=True)

    assert result.ok is False
    assert _is_rerun_already_running_error(result.error)


def test_remove_label_of_an_absent_label_succeeds(tmp_path: Path) -> None:
    reply = ok({"message": "Label does not exist"}, status=404)
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    assert gh.remove_issue_label(7, "agent:queued") is True


def test_pr_checks_pending_is_a_list_not_a_failure(tmp_path: Path) -> None:
    gh, _http, _ = make_github(
        tmp_path, http=FakeAdapter("http", [checks_reply(check_run("Tests", "IN_PROGRESS"))])
    )

    checks = gh.pr_checks(5)

    assert [(c["name"], c["bucket"]) for c in checks] == [("Tests", "pending")]


def test_pr_checks_with_no_checks_is_an_empty_list(tmp_path: Path) -> None:
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [checks_reply()]))

    assert gh.pr_checks(5) == []


def test_dry_run_mutation_through_run_keeps_the_legacy_return(tmp_path: Path) -> None:
    gh, http, gh_adapter = make_github(tmp_path, dry_run=True)

    assert gh.run(["pr", "close", "7"]) == "DRY-RUN: gh pr close 7"
    assert gh.run(["pr", "close", "7"], json_output=True) == []
    assert http.api_requests == []
    assert gh_adapter.api_requests == []


def test_dry_run_capability_mutation_is_a_typed_success_without_a_request(
    tmp_path: Path,
) -> None:
    gh, http, _ = make_github(tmp_path, dry_run=True)

    result = gh.pr_close(7)

    assert result.ok is True
    assert http.api_requests == []


def test_dry_run_still_sends_reads(tmp_path: Path) -> None:
    gh, http, _ = make_github(
        tmp_path,
        http=FakeAdapter("http", [ok({"data": {"repository": {"issue": {"number": 7}}}})]),
        dry_run=True,
    )

    assert gh.issue_view(7)["number"] == 7
    assert len(http.api_requests) == 1


def test_a_graphql_error_renders_as_the_api_message(tmp_path: Path) -> None:
    reply = graphql_failure("Could not resolve to a PullRequest with the number of 9.")
    gh, _http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))

    result = gh.run(["pr", "view", "9", "--json", "number"], allow_failure=True)

    assert result.ok is False
    assert "Could not resolve to a PullRequest" in result.error
