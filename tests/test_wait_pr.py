"""``charlie wait-pr`` (#2444): REST-only polling, exit codes, zero GraphQL.

Driven at the ``Adapter`` seam with the shared scripted fakes, so the real
``GuardedTransport`` sits between the poll loop and the counting fake.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any

import pytest

from charlie_work import cli, wait_pr
from charlie_work.github_transport import GraphQLRequest, Request, RestRequest

from _fake_transport import FakeAdapter, Sleeps, build_guard, ok

SHA = "a" * 40


def _run(name: str, status: str = "completed", conclusion: str | None = "success") -> dict:
    return {"name": name, "status": status, "conclusion": conclusion}


class FakePR:
    """Serves the REST routes wait-pr reads; ``runs``/``statuses`` can change per poll."""

    def __init__(self, *rounds: tuple[list[dict], list[dict]], headers: dict | None = None):
        self.rounds = list(rounds)
        self.headers = headers or {}
        self.poll = 0
        self.required: dict | None = None

    def handler(self, request: Request) -> Any:
        assert isinstance(request, RestRequest), f"non-REST request: {request!r}"
        route = request.route
        runs, statuses = self.rounds[min(self.poll, len(self.rounds) - 1)]
        if route.endswith("/pulls/7"):
            return ok({"head": {"sha": SHA}, "base": {"ref": "main"}}, headers=self.headers)
        if route.endswith("/check-runs"):
            return ok({"total_count": len(runs), "check_runs": runs})
        if route.endswith("/status"):
            self.poll += 1  # status is the last read of a poll
            return ok({"statuses": statuses}, headers=self.headers)
        if route.endswith("/required_status_checks"):
            if self.required is None:
                return ok({"message": "Not Found"}, status=404)
            return ok(self.required)
        raise AssertionError(f"unexpected route {route}")


def _wait(fake: FakePR, **kwargs: Any) -> tuple[wait_pr.WaitResult, FakeAdapter, Sleeps]:
    http = FakeAdapter("http", handler=fake.handler)
    guard, http, _gh, _ = build_guard(http=http)
    sleeps = Sleeps()
    clock = {"t": 0.0}

    def sleep(seconds: float) -> None:
        sleeps(seconds)
        clock["t"] += seconds

    result = wait_pr.wait_for_pr(
        guard.send,
        "octo",
        "hello",
        7,
        sleep=sleep,
        monotonic=lambda: clock["t"],
        out=io.StringIO(),
        **kwargs,
    )
    return result, http, sleeps


def test_all_passed_exits_0_with_zero_graphql_calls() -> None:
    fake = FakePR(
        ([_run("lint", "in_progress", None)], []),
        ([_run("lint"), _run("test")], [{"context": "ci/x", "state": "success"}]),
    )
    result, http, _ = _wait(fake)
    assert result.exit_code == wait_pr.EXIT_PASSED == 0
    assert result.polls == 2
    assert http.requests, "control: the fake must have seen traffic"
    assert not any(isinstance(r, GraphQLRequest) for r in http.requests)
    assert all(isinstance(r, RestRequest) and r.method == "GET" for r in http.requests)
    # Positive control for the counter: a GraphQL request would have been recorded.
    probe = FakeAdapter("http", handler=lambda r: ok({}))
    probe.send(GraphQLRequest.of("query { viewer { login } }"), token=None, timeout=1)
    assert any(isinstance(r, GraphQLRequest) for r in probe.requests)


def test_failed_check_exits_1_without_waiting_for_the_rest() -> None:
    fake = FakePR(([_run("lint", conclusion="failure"), _run("slow", "queued", None)], []))
    result, _, sleeps = _wait(fake)
    assert result.exit_code == wait_pr.EXIT_FAILED == 1
    assert "lint (failure)" in result.reason
    assert sleeps.delays == []


def test_failing_commit_status_exits_1() -> None:
    fake = FakePR(([_run("lint")], [{"context": "ci/x", "state": "error"}]))
    result, _, _ = _wait(fake)
    assert result.exit_code == 1


def test_timeout_exits_2_and_backs_off() -> None:
    fake = FakePR(([_run("lint", "in_progress", None)], []))
    result, _, sleeps = _wait(fake, timeout=100.0)
    assert result.exit_code == wait_pr.EXIT_TIMEOUT == 2
    assert sleeps.delays[0] == wait_pr.MIN_POLL_SECONDS
    assert sleeps.delays[1] > sleeps.delays[0]
    assert max(sleeps.delays) <= wait_pr.MAX_POLL_SECONDS
    assert sum(sleeps.delays) <= 100.0


def test_no_checks_yet_is_pending_not_passed() -> None:
    result, _, _ = _wait(FakePR(([], [])), timeout=30.0)
    assert result.exit_code == 2


def test_poll_interval_hint_is_a_floor_on_the_delay() -> None:
    fake = FakePR(([_run("lint", "queued", None)], []), headers={"x-poll-interval": "45"})
    _, _, sleeps = _wait(fake, timeout=100.0)
    assert min(sleeps.delays[:-1] or sleeps.delays) >= 45.0


def test_missing_pr_is_fatal_exit_2_immediately() -> None:
    http = FakeAdapter("http", handler=lambda r: ok({"message": "Not Found"}, status=404))
    guard, *_ = build_guard(http=http)
    result = wait_pr.wait_for_pr(guard.send, "octo", "hello", 7, sleep=Sleeps(), out=io.StringIO())
    assert result.exit_code == 2 and result.polls == 1


def test_required_only_ignores_unrequired_failures() -> None:
    fake = FakePR(([_run("lint"), _run("optional", conclusion="failure")], []))
    fake.required = {"contexts": [], "checks": [{"context": "lint"}]}
    result, _, _ = _wait(fake, required_only=True)
    assert result.exit_code == 0
    assert [c.name for c in result.snapshot.checks] == ["lint"]


def test_required_only_falls_back_to_all_checks_when_unreadable() -> None:
    fake = FakePR(([_run("lint"), _run("optional", conclusion="failure")], []))
    result, _, _ = _wait(fake, required_only=True)  # protection route 404s
    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("rounds", "timeout", "code"),
    [
        ((([_run("a")], []),), "30", 0),
        ((([_run("a", conclusion="failure")], []),), "30", 1),
        ((([_run("a", "queued", None)], []),), "0", 2),
    ],
)
def test_cli_main_maps_outcomes_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, rounds: tuple, timeout: str, code: int
) -> None:
    fake = FakePR(*rounds)
    guard, *_ = build_guard(http=FakeAdapter("http", handler=fake.handler))
    gh = SimpleNamespace(_transport_v2=guard, _repo_owner_name=lambda: ("octo", "hello"))
    monkeypatch.setattr(cli, "bootstrap_command", lambda args: SimpleNamespace(gh=gh))
    monkeypatch.setattr(wait_pr.time, "sleep", lambda s: None)
    assert cli.main(["wait-pr", "7", "--timeout", timeout]) == code


def test_cli_rejects_malformed_repo_slug() -> None:
    assert cli.main(["wait-pr", "7", "--repo", "nonsense"]) == 2
