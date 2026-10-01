"""GuardedTransport through its public interface (ADR-0006).

Everything is driven at the ``Adapter`` seam with scripted fakes
(``tests/_fake_transport.py``); sleep, jitter and the clock are injected, so
nothing here touches the network, a subprocess or real time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.github_capabilities.circuit_breaker import CircuitBreakerState
from charlie_work.github_capabilities.circuit_breaker_transport import (
    circuit_breaker_open_message,
)
from charlie_work.github_transport import (
    CliCommand,
    CliRequest,
    FailureKind,
    GraphQLRequest,
    Response,
    RestRequest,
    TransportFailure,
)
from charlie_work.instrumentation import query_events
from charlie_work.pass_deadline import PassDeadlineExceeded

from _fake_transport import FakeAdapter, Runtime, build_guard, failure, ok

GET_PR = RestRequest.of("GET", "repos/{owner}/{repo}/pulls/7")
POST_COMMENT = RestRequest.of(
    "POST", "repos/{owner}/{repo}/issues/7/comments", body={"body": "hi"}
)
QUERY = GraphQLRequest.of("query Q($n: Int!) { viewer { login } }", {"n": 1})
MUTATION = GraphQLRequest.of("mutation M { addStar(input: {}) { clientMutationId } }")


def _gh(*script: object, token: str | None = "tok-1") -> FakeAdapter:
    return FakeAdapter("gh", list(script), token=token)  # type: ignore[arg-type]


def _state(tmp_path: Path) -> Path:
    return tmp_path / "state.json"


def _kinds(state: Path, kind: str) -> list[dict]:
    return [e for e in query_events(state, kind=kind)]


# -- routing and placeholders -------------------------------------------------


def test_http_is_the_default_and_fills_placeholders_with_the_token() -> None:
    guard, http, gh, _ = build_guard(http=FakeAdapter("http", [ok({"n": 7})]))
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.json() == {"n": 7}
    (call,) = http.calls
    assert call.request == RestRequest.of("GET", "repos/octo/hello/pulls/7")
    assert call.token == "tok-1"
    assert gh.api_requests == []  # gh only ever answered `auth token`


def test_unresolvable_owner_repo_is_a_value_not_a_raise() -> None:
    guard, http, _, _ = build_guard(owner_repo=None)
    out = guard.send(GET_PR)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.ADAPTER_DEFECT
    assert http.calls == []


def test_cli_requests_always_go_to_gh() -> None:
    gh = _gh(Response(200, (), "Logged in", "gh", returncode=0))
    guard, http, _, _ = build_guard(gh=gh)
    out = guard.send(CliRequest(CliCommand.AUTH_STATUS))
    assert isinstance(out, Response) and out.body == "Logged in"
    assert http.calls == []
    assert gh.requests[-1] == CliRequest(CliCommand.AUTH_STATUS)


# -- kill switch --------------------------------------------------------------


def test_kill_switch_forces_gh_and_emits_no_fallback_event(tmp_path: Path) -> None:
    gh = _gh(ok({"via": "gh"}))
    guard, http, _, _ = build_guard(
        gh=gh, runtime=Runtime(gh_transport="gh"), state_path=_state(tmp_path)
    )
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.json() == {"via": "gh"}
    assert http.calls == []
    assert gh.api_requests == [RestRequest.of("GET", "repos/octo/hello/pulls/7")]
    assert _kinds(_state(tmp_path), "github_transport_fallback") == []


def test_kill_switch_is_read_live_per_call() -> None:
    class LiveRuntime:
        gh_max_retries = 0
        gh_retry_base_seconds = 1.0
        gh_timeout_seconds = 30.0
        gh_long_call_timeout_seconds = 120.0
        gh_transport = "http"

    runtime = LiveRuntime()
    guard, _, _, _ = build_guard(
        http=FakeAdapter("http", [ok({"via": "http"})]),
        gh=_gh(ok({"via": "gh"})),
        runtime=runtime,  # type: ignore[arg-type]
    )
    assert guard.send(GET_PR).json() == {"via": "http"}  # type: ignore[union-attr]
    runtime.gh_transport = "gh"
    assert guard.send(GET_PR).json() == {"via": "gh"}  # type: ignore[union-attr]


# -- per-call fallback --------------------------------------------------------


@pytest.mark.parametrize(
    "kind", [FailureKind.TOKEN_UNAVAILABLE, FailureKind.ADAPTER_DEFECT, FailureKind.CONNECT]
)
def test_fallback_kinds_replay_through_gh_and_emit_one_event(
    tmp_path: Path, kind: FailureKind
) -> None:
    http = FakeAdapter("http", [failure(kind, "tls handshake timeout")])
    gh = _gh(ok({"via": "gh"}))
    guard, _, _, sleeps = build_guard(
        http=http, gh=gh, state_path=_state(tmp_path), runtime=Runtime(gh_max_retries=0)
    )
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.json() == {"via": "gh"}
    events = _kinds(_state(tmp_path), "github_transport_fallback")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["reason"] == kind.value
    assert payload["mutation"] is False
    assert payload["request"] == "GET repos/octo/hello/pulls/7"
    assert "gh api" in payload["command"]
    assert sleeps.delays == []


def test_fallback_replays_a_mutation_because_the_request_was_not_applied(
    tmp_path: Path,
) -> None:
    http = FakeAdapter("http", [failure(FailureKind.CONNECT)])
    gh = _gh(ok({"id": 1}, status=201))
    guard, _, _, _ = build_guard(
        http=http, gh=gh, state_path=_state(tmp_path), runtime=Runtime(gh_max_retries=0)
    )
    out = guard.send(POST_COMMENT)
    assert isinstance(out, Response) and out.status == 201
    assert len(gh.api_requests) == 1
    events = _kinds(_state(tmp_path), "github_transport_fallback")
    payload = events[0]["payload"]
    assert payload["mutation"] is True


@pytest.mark.parametrize("kind", [FailureKind.SENT_NO_RESPONSE, FailureKind.TIMEOUT])
def test_ambiguous_failures_never_fall_back(tmp_path: Path, kind: FailureKind) -> None:
    http = FakeAdapter("http", [failure(kind)])
    gh = _gh(ok({"via": "gh"}))
    guard, _, _, _ = build_guard(
        http=http, gh=gh, state_path=_state(tmp_path), runtime=Runtime(gh_max_retries=0)
    )
    out = guard.send(POST_COMMENT)
    assert isinstance(out, TransportFailure) and out.kind is kind
    assert gh.api_requests == []
    assert _kinds(_state(tmp_path), "github_transport_fallback") == []


@pytest.mark.parametrize("status", [400, 404, 422, 500])
def test_http_error_statuses_never_fall_back(status: int) -> None:
    http = FakeAdapter("http", [ok({"message": "nope"}, status=status)])
    gh = _gh(ok({"via": "gh"}))
    guard, _, _, _ = build_guard(http=http, gh=gh, runtime=Runtime(gh_max_retries=0))
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.status == status
    assert gh.api_requests == []


def test_no_token_anywhere_falls_back_to_gh() -> None:
    gh = FakeAdapter("gh", [ok({"via": "gh"})], token=None)
    # `gh auth token` hits the script (consuming entry 0, which is not a token);
    # the resolved token is empty, so the guard reports TOKEN_UNAVAILABLE and
    # falls back for the API request itself.
    gh.script = [Response(1, (), "not logged in", "gh", returncode=1), ok({"via": "gh"})]
    guard, http, _, _ = build_guard(gh=gh, runtime=Runtime(gh_max_retries=0))
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.json() == {"via": "gh"}
    assert http.calls == []


# -- dry run ------------------------------------------------------------------


@pytest.mark.parametrize(
    "request_",
    [
        POST_COMMENT,
        RestRequest.of("PUT", "repos/{owner}/{repo}/pulls/1/merge"),
        RestRequest.of("PATCH", "repos/{owner}/{repo}/issues/1", body={"state": "closed"}),
        RestRequest.of("DELETE", "repos/{owner}/{repo}/git/refs/heads/x"),
        MUTATION,
    ],
)
def test_dry_run_blocks_every_mutation_derived_from_method_or_operation(request_) -> None:
    guard, http, gh, _ = build_guard(dry_run=True)
    out = guard.send(request_)
    assert isinstance(out, Response) and out.adapter == "dry_run" and out.ok
    assert http.calls == [] and gh.calls == []


@pytest.mark.parametrize("request_", [GET_PR, QUERY])
def test_dry_run_lets_reads_through(request_) -> None:
    guard, http, _, _ = build_guard(dry_run=True, http=FakeAdapter("http", [ok({"data": {}})]))
    out = guard.send(request_)
    assert isinstance(out, Response) and out.adapter == "http"
    assert len(http.calls) == 1


def test_dry_run_does_not_touch_the_breaker_or_the_budget() -> None:
    breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=60)
    guard, _, _, _ = build_guard(dry_run=True, breaker=breaker)
    guard.send(POST_COMMENT)
    assert breaker.consecutive_failures == 0
    assert guard.budget.value.windows == ()


# -- GraphQL errors inside a 200 ---------------------------------------------


def test_graphql_errors_in_a_200_are_a_response_that_is_not_ok() -> None:
    body = '{"data": null, "errors": [{"message": "Could not resolve", "type": "NOT_FOUND"}]}'
    from charlie_work.github_transport import GraphQLError

    errs = (GraphQLError("Could not resolve", "NOT_FOUND"),)
    http = FakeAdapter("http", [Response(200, (), body, "http", errs)])
    guard, _, gh, _ = build_guard(http=http, runtime=Runtime(gh_max_retries=2))
    out = guard.send(QUERY)
    assert isinstance(out, Response)
    assert out.status == 200 and not out.ok
    assert out.graphql_errors[0].type == "NOT_FOUND"
    assert len(http.calls) == 1  # not retried, not fallen back
    assert gh.api_requests == []


def test_graphql_rate_limited_error_in_a_200_is_retried_for_queries() -> None:
    from charlie_work.github_transport import GraphQLError

    limited = Response(200, (), "{}", "http", (GraphQLError("slow down", "RATE_LIMITED"),))
    http = FakeAdapter("http", [limited, ok({"data": {"ok": 1}})])
    guard, _, _, sleeps = build_guard(http=http)
    out = guard.send(QUERY)
    assert isinstance(out, Response) and out.ok
    assert len(http.calls) == 2 and sleeps.delays == [1.0]


# -- retry --------------------------------------------------------------------


def test_reads_retry_transient_statuses_with_exponential_backoff() -> None:
    http = FakeAdapter("http", [ok("", status=503), ok("", status=502), ok({"n": 1})])
    guard, _, _, sleeps = build_guard(http=http)
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.ok
    assert sleeps.delays == [1.0, 2.0]


def test_retries_are_bounded_by_runtime_max_retries() -> None:
    http = FakeAdapter("http", [ok("", status=503)])
    guard, _, _, sleeps = build_guard(http=http, runtime=Runtime(gh_max_retries=2))
    out = guard.send(GET_PR)
    assert isinstance(out, Response) and out.status == 503
    assert len(http.calls) == 3 and len(sleeps.delays) == 2


def test_rate_limit_403_is_retried_but_a_plain_403_is_not() -> None:
    limited = ok(
        {"message": "API rate limit exceeded"}, headers={"X-RateLimit-Remaining": "0"}, status=403
    )
    http = FakeAdapter("http", [limited, ok({"n": 1})])
    guard, _, _, _ = build_guard(http=http)
    assert guard.send(GET_PR).ok  # type: ignore[union-attr]
    plain = FakeAdapter("http", [ok({"message": "forbidden"}, status=403)])
    guard2, _, _, _ = build_guard(http=plain)
    out = guard2.send(GET_PR)
    assert isinstance(out, Response) and out.status == 403 and len(plain.calls) == 1


def test_mutations_do_not_retry_a_5xx_or_an_ambiguous_failure() -> None:
    http = FakeAdapter("http", [ok("", status=503)])
    guard, _, _, sleeps = build_guard(http=http)
    guard.send(POST_COMMENT)
    assert len(http.calls) == 1 and sleeps.delays == []

    http2 = FakeAdapter("http", [failure(FailureKind.SENT_NO_RESPONSE)])
    guard2, _, _, sleeps2 = build_guard(http=http2)
    guard2.send(POST_COMMENT)
    assert len(http2.calls) == 1 and sleeps2.delays == []


def test_reads_retry_ambiguous_failures() -> None:
    http = FakeAdapter("http", [failure(FailureKind.TIMEOUT), ok({"n": 1})])
    guard, _, _, sleeps = build_guard(http=http)
    assert guard.send(GET_PR).ok  # type: ignore[union-attr]
    assert sleeps.delays == [1.0]


def test_long_call_uses_the_long_timeout() -> None:
    http = FakeAdapter("http", [ok("logs")])
    guard, _, _, _ = build_guard(http=http)
    guard.send(RestRequest.of("GET", "repos/{owner}/{repo}/actions/jobs/1/logs", long_call=True))
    guard.send(GET_PR)
    assert [c.timeout for c in http.calls] == [120.0, 30.0]


# -- 401 re-resolve -----------------------------------------------------------


def test_a_401_re_resolves_the_token_once_and_resends() -> None:
    http = FakeAdapter("http", [ok({"message": "Bad credentials"}, status=401), ok({"n": 1})])
    gh = _gh(token="tok-1")
    guard, _, _, _ = build_guard(http=http, gh=gh)
    out = guard.send(POST_COMMENT)  # safe for a mutation: a 401 was not applied
    assert isinstance(out, Response) and out.ok
    assert len(http.calls) == 2
    token_calls = [r for r in gh.requests if isinstance(r, CliRequest)]
    assert len(token_calls) == 2  # initial resolve + the re-resolve


def test_the_token_is_cached_across_calls() -> None:
    http = FakeAdapter("http", [ok({"n": 1})])
    gh = _gh(token="tok-1")
    guard, _, _, _ = build_guard(http=http, gh=gh)
    guard.send(GET_PR)
    guard.send(GET_PR)
    assert len([r for r in gh.requests if isinstance(r, CliRequest)]) == 1


# -- deadline -----------------------------------------------------------------


def test_a_spent_pass_deadline_aborts_before_any_adapter_call() -> None:
    guard, http, gh, _ = build_guard(exceeded=lambda: True)
    with pytest.raises(PassDeadlineExceeded):
        guard.send(GET_PR)
    assert http.calls == [] and gh.calls == []


def test_the_deadline_is_rechecked_before_a_retry_sleep() -> None:
    state = {"spent": False}
    http = FakeAdapter("http", [ok("", status=503)])
    guard, _, _, sleeps = build_guard(http=http, exceeded=lambda: state["spent"])

    original_send = http.send

    def send_then_spend(request, *, token, timeout):  # noqa: ANN001
        out = original_send(request, token=token, timeout=timeout)
        state["spent"] = True
        return out

    http.send = send_then_spend  # type: ignore[method-assign]
    with pytest.raises(PassDeadlineExceeded):
        guard.send(GET_PR)
    assert sleeps.delays == []


def test_set_pass_deadline_exceeded_installs_the_predicate_later() -> None:
    guard, _, _, _ = build_guard()
    guard.set_pass_deadline_exceeded(lambda: True)
    with pytest.raises(PassDeadlineExceeded):
        guard.send(GET_PR)


# -- rate-limit accounting ----------------------------------------------------


def test_every_response_feeds_the_rate_budget() -> None:
    headers = {
        "X-RateLimit-Limit": "5000",
        "X-RateLimit-Remaining": "4321",
        "X-RateLimit-Reset": "2000",
        "X-RateLimit-Resource": "core",
    }
    guard, _, _, _ = build_guard(http=FakeAdapter("http", [ok({"n": 1}, headers=headers)]))
    guard.send(GET_PR)
    (window,) = guard.budget.value.windows
    assert (window.resource, window.limit, window.remaining) == ("core", 5000, 4321)
    assert window.observed_epoch == 1000.0


def test_a_gh_adapter_response_is_accounted_the_same_way() -> None:
    headers = {
        "x-ratelimit-limit": "5000",
        "x-ratelimit-remaining": "10",
        "x-ratelimit-reset": "2000",
    }
    gh = _gh(ok({"n": 1}, headers=headers))
    guard, _, _, _ = build_guard(gh=gh, runtime=Runtime(gh_transport="gh"))
    guard.send(GET_PR)
    assert guard.budget.value.windows[0].remaining == 10


# -- circuit breaker ----------------------------------------------------------


def _breaker(threshold: int = 2) -> CircuitBreakerState:
    return CircuitBreakerState(failure_threshold=threshold, cooldown_seconds=60)


def test_breaker_counts_the_final_outcome_once_per_logical_call(tmp_path: Path) -> None:
    breaker = _breaker(threshold=3)
    http = FakeAdapter("http", [failure(FailureKind.SENT_NO_RESPONSE)])
    guard, _, _, _ = build_guard(
        http=http, breaker=breaker, runtime=Runtime(gh_max_retries=2), state_path=_state(tmp_path)
    )
    guard.send(GET_PR)  # three attempts, ONE breaker failure
    assert len(http.calls) == 3
    assert breaker.consecutive_failures == 1


def test_breaker_opens_emits_an_event_and_fails_fast_with_the_legacy_wording(
    tmp_path: Path,
) -> None:
    breaker = _breaker(threshold=2)
    http = FakeAdapter("http", [failure(FailureKind.SENT_NO_RESPONSE)])
    guard, _, _, _ = build_guard(
        http=http, breaker=breaker, runtime=Runtime(gh_max_retries=0), state_path=_state(tmp_path)
    )
    guard.send(GET_PR)
    guard.send(GET_PR)
    assert len(_kinds(_state(tmp_path), "github_circuit_opened")) == 1
    calls_before = len(http.calls)
    out = guard.send(GET_PR)
    assert isinstance(out, TransportFailure) and out.kind is FailureKind.CIRCUIT_OPEN
    assert len(http.calls) == calls_before
    assert out.detail == circuit_breaker_open_message(breaker, ["GET repos/octo/hello/pulls/7"])


def test_any_response_including_an_error_status_resets_the_streak() -> None:
    breaker = _breaker(threshold=2)
    http = FakeAdapter(
        "http",
        [failure(FailureKind.SENT_NO_RESPONSE), ok({"message": "nf"}, status=404)],
    )
    guard, _, _, _ = build_guard(http=http, breaker=breaker, runtime=Runtime(gh_max_retries=0))
    guard.send(GET_PR)
    assert breaker.consecutive_failures == 1
    guard.send(GET_PR)
    assert breaker.consecutive_failures == 0


def test_breaker_closes_after_a_successful_probe_and_emits_closed(tmp_path: Path) -> None:
    now = {"t": 0.0}
    breaker = CircuitBreakerState(failure_threshold=1, cooldown_seconds=10, clock=lambda: now["t"])
    http = FakeAdapter("http", [failure(FailureKind.SENT_NO_RESPONSE), ok({"n": 1})])
    guard, _, _, _ = build_guard(
        http=http, breaker=breaker, runtime=Runtime(gh_max_retries=0), state_path=_state(tmp_path)
    )
    guard.send(GET_PR)
    assert len(_kinds(_state(tmp_path), "github_circuit_opened")) == 1
    now["t"] = 11.0  # cooldown elapsed: one probe is admitted
    assert guard.send(GET_PR).ok  # type: ignore[union-attr]
    assert len(_kinds(_state(tmp_path), "github_circuit_closed")) == 1


def test_a_fallback_that_succeeds_counts_as_a_success_for_the_breaker() -> None:
    breaker = _breaker(threshold=1)
    http = FakeAdapter("http", [failure(FailureKind.CONNECT)])
    guard, _, _, _ = build_guard(
        http=http, gh=_gh(ok({"via": "gh"})), breaker=breaker, runtime=Runtime(gh_max_retries=0)
    )
    assert guard.send(GET_PR).ok  # type: ignore[union-attr]
    assert breaker.consecutive_failures == 0


def test_reset_circuit_breaker_clears_an_open_breaker() -> None:
    breaker = _breaker(threshold=1)
    http = FakeAdapter("http", [failure(FailureKind.SENT_NO_RESPONSE), ok({"n": 1})])
    guard, _, _, _ = build_guard(http=http, breaker=breaker, runtime=Runtime(gh_max_retries=0))
    guard.send(GET_PR)
    assert isinstance(guard.send(GET_PR), TransportFailure)
    guard.reset_circuit_breaker()
    assert guard.send(GET_PR).ok  # type: ignore[union-attr]


def test_dry_run_suppresses_a_templated_mutation_even_when_origin_is_unreadable() -> None:
    guard, http, gh, _ = build_guard(dry_run=True, owner_repo=None)
    out = guard.send(RestRequest.of("PUT", "repos/{owner}/{repo}/pulls/1/merge"))
    assert isinstance(out, Response) and out.adapter == "dry_run" and out.ok
    assert http.calls == [] and gh.calls == []
