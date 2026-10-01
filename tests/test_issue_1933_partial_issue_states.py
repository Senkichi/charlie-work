"""Issue #1933: one unresolvable issue number in the batched GraphQL
``are_issues_open`` query must not demote the whole batch to the per-issue
``issue_view`` fallback.

GitHub answers ``s_<n>: issue(number: <n>)`` aliases that cannot resolve with
a ``null`` data entry plus a per-node ``errors`` entry -- the HTTP status is
still 200 and every other alias resolves, but ``gh`` exits non-zero (and this
repo's pooled HTTP transport mirrors that by translating the error). Before
this fix the non-zero exit raised through ``_graphql_query`` and discarded
the partial ``data`` entirely, so ``are_issues_open`` re-fetched EVERY
uncached number through ``issue_view`` -- slow enough to blow the
``fleet status --json`` timeout whenever one stale issue number was in play.

The fix has three seams, each covered below:

* ``http_adapter`` preserves the response body on stdout
  for erroring GraphQL responses (real ``gh api`` does the same -- in
  cli/cli's ``pkg/cmd/api/api.go`` ``processResponse`` copies the body to the
  output writer before emitting the error), so the partial ``data`` reaches
  callers as ``GitHubRunResult.value``.
* ``github_capabilities.graphql_issue_states`` consumes that partial body:
  resolved aliases map normally; aliases with a null node are absent from
  the result.
* ``Issues.are_issues_open`` runs its per-issue ``issue_view`` fallback only
  over numbers absent from the batched result, and emits
  ``github_issue_state_partial_fallback`` once per newly-unresolved number
  per GitHub-instance lifetime: a number leaves the tracked set only when a
  later batch positively resolves it, so an unrelated clean batch cannot
  reset the baseline and whole-batch failures neither emit nor disturb it.
  Instance lifetime is caller-dependent -- the single-repo supervisor keeps
  one GitHub for the whole process, while ``fleet_loop`` builds a fresh
  GitHub per repo per pass -- so the kind is registered ``info``, not
  ``warning`` (the last test below covers the fleet lifecycle end-to-end
  through the real ``log_event``/events.db path).

No live network anywhere: both adapters are scripted fakes
(``tests/_fake_transport.py``) and no ``gh`` subprocess is ever spawned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _fake_transport import (
    FakeAdapter,
    FakeTransport,
    graphql_ok,
    graphql_variables,
    make_github,
    ok,
)

from charlie_work.github import GitHub, GitHubError
from charlie_work.github_capabilities._outcome import to_run_result
from charlie_work.github_capabilities.issues import Issues
from charlie_work.github_transport.outcome import Response, parse_graphql_errors


def _graphql_response(body: dict, *, status: int = 200) -> Response:
    """What the transport hands back for a GraphQL answer: the body text plus
    the typed per-node errors the adapters extract from it."""
    errors = parse_graphql_errors(body.get("errors"))
    return Response(status, (), json.dumps(body), "http", graphql_errors=errors)


def _serve_graphql(gh: GitHub, outcome) -> FakeTransport:
    """Answer every GraphQL request *gh* sends with *outcome* (the issue-view
    fallback still goes through ``GitHub.run``, patched per test)."""
    transport = FakeTransport(lambda request: outcome)
    object.__setattr__(gh, "_transport_v2", transport)
    return transport


def _route_issue_view_through(monkeypatch: pytest.MonkeyPatch, fake_run) -> None:
    """Serve the per-issue fallback (``Issues.issue_view``) from a ``fake_run``
    written in the ``["issue", "view", N]`` argv dialect these tests use."""

    def issue_view(self, number: int) -> dict:
        return fake_run(self, ["issue", "view", str(number)])

    monkeypatch.setattr(Issues, "issue_view", issue_view)


def _partial_body(*, resolved: dict[str, dict], unresolved: list[int]) -> dict:
    """A real-shaped GraphQL response: every resolvable alias carries data,
    the failing alias is null, and the ``errors`` array names it by path."""
    repo = dict(resolved)
    errors = []
    for number in unresolved:
        repo[f"s_{number}"] = None
        errors.append(
            {
                "message": f"Could not resolve to an Issue with the number of {number}.",
                "path": ["repository", f"s_{number}"],
            }
        )
    return {"data": {"repository": repo}, "errors": errors}


def test_graphql_error_response_body_reaches_stdout(monkeypatch, tmp_path: Path) -> None:
    """A 200 response with an ``errors`` array keeps its body in the run
    result -- that body is what carries the partial ``data`` the batched
    query parses. Real ``gh api graphql`` does the same (body copied to stdout
    before the error is emitted to stderr), so this is parity, not a new
    convention. (The leaf name predates the transport: it asserted the same
    through the pooled HTTP transport.)
    """
    body = _partial_body(resolved={"s_1": {"number": 1, "state": "OPEN"}}, unresolved=[361])
    outcome = _graphql_response(body)

    result = to_run_result(outcome, json_output=False, command="gh api graphql")

    assert result.ok is False
    assert result.stderr.startswith("GraphQL: Could not resolve to an Issue")
    parsed = json.loads(result.stdout)
    assert parsed["data"]["repository"]["s_1"] == {"number": 1, "state": "OPEN"}
    assert parsed["data"]["repository"]["s_361"] is None


def test_graphql_issue_states_omits_unresolved_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The batched function returns states for resolved aliases only; an
    unresolvable alias is absent from the mapping rather than forcing the
    whole result to fail."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    body = _partial_body(
        resolved={
            "s_1": {"number": 1, "state": "OPEN"},
            "s_2": {"number": 2, "state": "CLOSED"},
        },
        unresolved=[361],
    )

    _serve_graphql(gh, _graphql_response(body))
    assert gh._graphql_issue_states([1, 2, 361]) == {1: True, 2: False}


def test_graphql_issue_states_still_raises_without_usable_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failure with no partial ``data`` at all (transport error, or a
    body that never reached stdout) still raises -- the caller's whole-set
    fallback contract is unchanged for genuine batch failures."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")

    _serve_graphql(gh, Response(502, (), "Bad Gateway", "http"))
    with pytest.raises(GitHubError):
        gh._graphql_issue_states([1])


def test_are_issues_open_per_issue_fallback_covers_only_unresolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The incident shape: one unresolvable alias leaves the other numbers
    resolved by the single batched query, and ``issue_view`` runs only for
    the number the batch could not resolve."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    body = _partial_body(
        resolved={
            "s_1": {"number": 1, "state": "OPEN"},
            "s_2": {"number": 2, "state": "CLOSED"},
        },
        unresolved=[361],
    )
    issue_view_calls: list[int] = []

    _serve_graphql(gh, _graphql_response(body))

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        assert args[:2] == ["issue", "view"]
        number = int(args[2])
        issue_view_calls.append(number)
        # REST issue_view CAN resolve the number GraphQL choked on (issue
        # #1933's #361 evidence) -- and it turns out to be OPEN, so it must
        # land in the open set rather than being silently marked closed.
        return {"number": number, "state": "OPEN"}

    _route_issue_view_through(monkeypatch, fake_run)

    assert gh.are_issues_open([1, 2, 361]) == {1, 361}
    assert issue_view_calls == [361]
    assert gh._list_cache[("issue_open", 361)] is True
    assert gh._list_cache[("issue_open", 2)] is False


def test_are_issues_open_full_fallback_when_batch_yields_no_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Contract preservation: a batch failure with no usable partial data
    still per-issue-fetches every uncached number (the pre-#1933 path)."""
    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    issue_view_calls: list[int] = []

    _serve_graphql(gh, Response(502, (), "HTTP 502", "http"))

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        assert args[:2] == ["issue", "view"]
        number = int(args[2])
        issue_view_calls.append(number)
        return {"number": number, "state": "OPEN" if number != 2 else "CLOSED"}

    _route_issue_view_through(monkeypatch, fake_run)

    assert gh.are_issues_open([1, 2, 3]) == {1, 3}
    assert sorted(issue_view_calls) == [1, 2, 3]


def test_are_issues_open_emits_telemetry_for_partial_batch_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A partial-batch fallback must be observable: the degraded condition is
    exactly what the incident needed telemetry for -- which numbers were bad
    and how large the requested set was. Whole-batch failures (the pre-#1933
    path) keep their existing log-line-only behavior.

    Edge-triggered per number (rework): the event fires once per
    *newly-unresolved* issue number per GitHub-instance lifetime, not once
    per batch observation. The emitter tracks numbers already reported; a
    reported number leaves the tracked set only when a later batch
    positively resolves it -- an unrelated clean batch must NOT reset the
    baseline (the multi-call-per-pass flood the round-2 review caught). A
    newly-stale number joining an already-reported set is a new edge and
    still fires.

    Lifetime caveat (round-3 review): instance lifetime is
    caller-dependent. This test models the single-repo supervisor
    (``supervise.run_supervised``), which keeps one GitHub/Issues pair
    across passes -- there the set does suppress a persistent stale
    blocker's repeat. The fleet supervisor builds a fresh GitHub per repo
    per pass, so there the same blocker emits once per pass per repo;
    that expected repeat is why the kind is registered ``info``, not
    ``warning`` -- covered end-to-end by
    ``test_fleet_lifecycle_persistent_blocker_never_emits_warning``.
    Short-lived processes such as ``fleet status --json`` still emit once
    per invocation: their GitHub instance does not outlive the run, so
    there is nothing to dedupe against.
    """
    from charlie_work.github_capabilities import issues as issues_module

    gh = GitHub(tmp_path)
    unresolved_numbers = [361]
    events: list[tuple[str, dict]] = []

    def fake_log_event(state_path, kind, payload, **kwargs):
        events.append((kind, payload))

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        return {"number": int(args[2]), "state": "CLOSED"}

    def answer(request):
        return _graphql_response(
            _partial_body(
                resolved={"s_1": {"number": 1, "state": "OPEN"}},
                unresolved=unresolved_numbers,
            )
        )

    object.__setattr__(gh, "_transport_v2", FakeTransport(answer))

    def new_pass() -> None:
        # Simulate a pass boundary in the single-repo supervisor
        # (run_supervised): the pass-scoped list cache is cleared
        # (repo_meta.invalidate_list_cache) while the same GitHub/Issues
        # pair -- and its edge-trigger signature -- survives. fleet_loop
        # instead rebuilds GitHub per repo per pass; that lifecycle is
        # covered by the fresh-instance test at the bottom of this file.
        gh._list_cache.clear()
        gh._list_cache[("_repo_owner_name",)] = ("o", "r")

    _route_issue_view_through(monkeypatch, fake_run)
    monkeypatch.setattr(issues_module, "log_event", fake_log_event)

    new_pass()
    assert gh.are_issues_open([1, 361]) == {1}
    partial = [p for k, p in events if k == "github_issue_state_partial_fallback"]
    assert partial == [{"unresolved": [361], "requested": 2}]

    # The incident shape: the same stale blocker still unresolved on the
    # next pass must NOT refire -- that repeat carries no new information.
    new_pass()
    assert gh.are_issues_open([1, 361]) == {1}
    partial = [p for k, p in events if k == "github_issue_state_partial_fallback"]
    assert partial == [{"unresolved": [361], "requested": 2}]

    # A changed unresolved set is a new edge: it fires again.
    new_pass()
    unresolved_numbers.append(400)
    assert gh.are_issues_open([1, 361, 400]) == {1}
    partial = [p for k, p in events if k == "github_issue_state_partial_fallback"]
    assert partial == [
        {"unresolved": [361], "requested": 2},
        {"unresolved": [361, 400], "requested": 3},
    ]


def _telemetry_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Shared seam for the multi-call edge-trigger tests: a controllable
    ``_graphql_issue_states`` (``resolvable`` is the set of numbers the
    batch can resolve; ``batch_broken`` simulates a whole-query failure),
    a ``run`` that answers ``issue view`` with a state per number, and a
    captured ``log_event``. Returns (gh, knobs, issue_view_calls,
    events, new_pass)."""
    from charlie_work.github_capabilities import issues as issues_module

    gh = GitHub(tmp_path)
    knobs = {
        "resolvable": {1: True, 7: True, 8: True, 9: True},
        "batch_broken": False,
        "view_states": {1: "OPEN"},
    }
    issue_view_calls: list[int] = []
    events: list[tuple[str, dict]] = []

    def fake_log_event(state_path, kind, payload, **kwargs):
        events.append((kind, payload))

    def fake_states(self, issue_numbers):
        if knobs["batch_broken"]:
            raise GitHubError("batched state query failed")
        resolvable: dict[int, bool] = knobs["resolvable"]
        return {n: resolvable[n] for n in issue_numbers if n in resolvable}

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        assert args[:2] == ["issue", "view"]
        number = int(args[2])
        issue_view_calls.append(number)
        return {"number": number, "state": knobs["view_states"].get(number, "CLOSED")}

    def new_pass() -> None:
        # Single-repo supervisor (run_supervised) pass boundary: the
        # pass-scoped list cache is cleared
        # (repo_meta.invalidate_list_cache) while the same GitHub/Issues
        # pair -- and its reported-numbers set -- survives. fleet_loop's
        # rebuild-per-pass lifecycle is covered by the fresh-instance
        # test at the bottom of this file.
        gh._list_cache.clear()
        gh._list_cache[("_repo_owner_name",)] = ("o", "r")

    monkeypatch.setattr(GitHub, "_graphql_issue_states", fake_states)
    _route_issue_view_through(monkeypatch, fake_run)
    monkeypatch.setattr(issues_module, "log_event", fake_log_event)
    new_pass()
    return gh, knobs, issue_view_calls, events, new_pass


def _partial_events(events: list[tuple[str, dict]]) -> list[dict]:
    return [p for k, p in events if k == "github_issue_state_partial_fallback"]


def test_unrelated_clean_batch_cannot_clear_reported_numbers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The round-2 blocker shape: multiple are_issues_open calls per pass.
    A second call over OTHER uncached numbers that all resolve must not
    reset the baseline -- the retired single-frozenset signature cleared
    here and refired the same stale blocker every pass."""
    gh, knobs, _calls, events, new_pass = _telemetry_harness(monkeypatch, tmp_path)

    for _ in range(4):
        new_pass()
        assert gh.are_issues_open([1, 361]) == {1}
        assert gh.are_issues_open([7, 8, 9]) == {7, 8, 9}

    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]


def test_cached_stale_number_plus_new_number_keeps_single_edge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A same-pass second call re-including the stale number (now cached by
    the per-issue fallback) plus one new resolvable number produces a clean
    batch for the uncached remainder -- still not a baseline reset."""
    gh, knobs, _calls, events, new_pass = _telemetry_harness(monkeypatch, tmp_path)

    for _ in range(4):
        new_pass()
        assert gh.are_issues_open([1, 361]) == {1}
        # 361 is cached (closed, per the fallback's issue_view); 8 joins
        # the batch and resolves cleanly.
        assert gh.are_issues_open([361, 8]) == {8}

    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]


def test_resolved_then_restale_number_fires_fresh_edge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A reported number that a later batch positively resolves leaves the
    tracked set; if it goes stale again afterwards that is a new edge and
    the event refires."""
    gh, knobs, _calls, events, new_pass = _telemetry_harness(monkeypatch, tmp_path)
    resolvable: dict[int, bool] = knobs["resolvable"]

    assert gh.are_issues_open([1, 361]) == {1}
    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]

    # A later pass resolves 361 in the batch itself: it drops out of the
    # tracked set (and this batch emits nothing).
    resolvable[361] = True
    new_pass()
    assert gh.are_issues_open([1, 361]) == {1, 361}
    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]

    # 361 goes stale again: a fresh edge, so it fires a second time.
    del resolvable[361]
    new_pass()
    assert gh.are_issues_open([1, 361]) == {1}
    assert _partial_events(events) == [
        {"unresolved": [361], "requested": 2},
        {"unresolved": [361], "requested": 2},
    ]


def test_whole_batch_failure_neither_emits_nor_disturbs_tracked_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A batch failure with no usable data emits no telemetry event and
    leaves the reported-numbers set alone: afterwards the already-reported
    stale number is still deduped, not refired."""
    gh, knobs, issue_view_calls, events, new_pass = _telemetry_harness(monkeypatch, tmp_path)

    assert gh.are_issues_open([1, 361]) == {1}
    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]
    assert issue_view_calls == [361]

    # Whole-batch failure pass: no event, per-issue fallback covers every
    # uncached number (the pre-#1933 whole-set fallback contract).
    knobs["batch_broken"] = True
    new_pass()
    assert gh.are_issues_open([1, 361]) == {1}
    assert sorted(issue_view_calls[1:]) == [1, 361]
    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]

    # Recovery pass: the same stale number was already reported and the
    # failure did not disturb that, so still no new event.
    knobs["batch_broken"] = False
    new_pass()
    assert gh.are_issues_open([1, 361]) == {1}
    assert _partial_events(events) == [{"unresolved": [361], "requested": 2}]


def test_are_issues_open_end_to_end_over_http_transport(tmp_path: Path) -> None:
    """The production stack with only the adapters faked: the guarded transport
    keeps the erroring body -> the batch parses the partial states -> the
    per-issue read runs for the unresolved number only."""
    body = _partial_body(resolved={"s_1": {"number": 1, "state": "OPEN"}}, unresolved=[361])
    replies = [
        ok(body),
        graphql_ok({"repository": {"issue": {"number": 361, "state": "OPEN"}}}),
    ]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", replies))

    assert gh.are_issues_open([1, 361]) == {1, 361}
    batch, view = http.api_requests
    assert "s_361" in batch.document
    assert graphql_variables(view)["number"] == 361


def test_fleet_lifecycle_persistent_blocker_never_emits_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fleet supervisor lifecycle (round-3 review): ``fleet_loop``
    builds a NEW ``GitHub`` per repo on every pass
    (``fleet_dispatch.fleet_loop``), so the instance-scoped
    ``_partial_fallback_reported`` baseline resets each pass and a
    persistent stale blocker re-emits
    ``github_issue_state_partial_fallback`` every pass. The earlier tests
    model the other lifecycle (``run_supervised`` reusing one GitHub) via
    ``_list_cache.clear()``; this one constructs a fresh instance per
    pass exactly as fleet_loop does.

    Because the per-pass repeat is expected, the flood is closed by the
    kind's ``info`` registration, not by dedupe: at ``warning`` the same
    blocker would write one row per pass into ``check_warning_events``
    (the #1271/#1768 flood shape). The real ``log_event`` is left
    unpatched so the persisted ``level`` column itself is asserted --
    zero warning rows across all passes, while the info rows still land
    once per pass.
    """
    from charlie_work.github_capabilities.circuit_breaker_transport import (
        circuit_breaker_state_path,
    )
    from charlie_work.instrumentation import query_events

    def fake_states(self, issue_numbers):
        # 361 is persistently unresolvable; every other number resolves.
        return {n: True for n in issue_numbers if n != 361}

    def fake_run(self, args, *, json_output=False, allow_failure=False, long_call=False):
        assert args[:2] == ["issue", "view"]
        return {"number": int(args[2]), "state": "CLOSED"}

    monkeypatch.setattr(GitHub, "_graphql_issue_states", fake_states)
    _route_issue_view_through(monkeypatch, fake_run)

    # Same state path the emit site resolves:
    # circuit_breaker_state_path(self.runtime, self.repo_root) with
    # runtime=None for GitHub(tmp_path).
    state_path = circuit_breaker_state_path(None, tmp_path)
    passes = 4
    for _ in range(passes):
        # fleet_loop's lifecycle: a fresh GitHub per pass -- deliberately
        # NOT _list_cache.clear() on one long-lived instance.
        gh = GitHub(tmp_path)
        assert gh.are_issues_open([1, 361]) == {1}

    emitted = query_events(state_path, kind="github_issue_state_partial_fallback")
    # The telemetry still lands every pass (the degradation is not
    # hidden) ...
    assert len(emitted) == passes
    assert all(e["level"] == "info" for e in emitted)
    # ... but never at warning level, so check_warning_events sees nothing.
    assert not query_events(
        state_path, kind="github_issue_state_partial_fallback", level="warning"
    )
