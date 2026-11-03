"""Issue #2901: a PR number in the issue-state read fell back, and reported, on every pass.

Observed in a consumer repo: ``github_issue_state_partial_fallback`` fired on
every pass (2,081 rows in a week, one per ``loop_completed``), each payload
``{"requested": 50, "unresolved": [<n>]}`` with ``<n>`` a pull request number.

Two defects, one test section each:

* **Root cause** -- the batched query looked every number up with
  ``repository.issue(number:)``, which resolves issues only. A blocker declared
  as ``Blocked by #<PR>`` therefore came back ``null`` and was re-fetched
  through the per-issue fallback (which uses ``issueOrPullRequest``) every
  pass. The batch now uses the same union lookup.
* **Once-only reporting** -- a number that really is unresolved was reported
  once per pass, because the report-once baseline lived on an ``Issues``
  instance and the fleet supervisor builds a fresh ``GitHub`` per repo per
  pass. The baseline is now a per-repo sidecar
  (``github_capabilities.unresolved_issue_baseline``).

No live network: the HTTP adapter is a scripted fake and no ``gh`` subprocess
is spawned.
"""

from __future__ import annotations

import errno
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from _fake_transport import FakeAdapter, make_github, ok

from charlie_work import layout
from charlie_work.github import GitHub
from charlie_work.github_capabilities import graphql_issue_states as graphql_issue_states_module
from charlie_work.github_capabilities import issues as issues_module
from charlie_work.github_capabilities import unresolved_issue_baseline as baseline
from charlie_work.github_capabilities.circuit_breaker_transport import circuit_breaker_state_path
from charlie_work.github_capabilities.issues import Issues
from charlie_work.github_transport.outcome import Response, parse_graphql_errors
from charlie_work.state import StateLockBusy

KIND = "github_issue_state_partial_fallback"

# ``s_<n>: <lookup>(number: <n>)`` -- the alias grammar of the batched document.
_ALIAS = re.compile(r"s_(\d+): (issueOrPullRequest|issue)\(number: \d+\)")
# ``... on <Type> { <fields> }`` -- one inline fragment of a union selection.
_FRAGMENT = re.compile(r"\.\.\. on (Issue|PullRequest) \{([^{}]*)\}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _selects_state(selection: str, type_name: str) -> bool:
    """True when *selection* carries ``... on <type_name> { ... number ... state ... }``.

    ``issueOrPullRequest`` returns a union, which has no fields of its own: a
    node is populated only through the inline fragment naming its concrete
    type, and only with the fields that fragment selects.
    """
    return any(
        name == type_name and {"number", "state"} <= set(fields.split())
        for name, fields in _FRAGMENT.findall(selection)
    )


def _aliases(document: str) -> list[tuple[int, str, str]]:
    """``(number, lookup, selection)`` for every ``s_<n>`` alias in *document*.

    A selection runs from the end of its alias header to the start of the next
    alias, so one alias cannot satisfy another's fragments.
    """
    matches = list(_ALIAS.finditer(document))
    ends = [m.start() for m in matches[1:]] + [len(document)]
    return [
        (int(m.group(1)), m.group(2), document[m.end() : end])
        for m, end in zip(matches, ends, strict=True)
    ]


def _github_emulator(*, issues: dict[int, str], pulls: dict[int, str]) -> FakeAdapter:
    """An HTTP adapter that answers the batched issue-state document the way
    GitHub does: ``issue(number:)`` resolves issues only (a PR number is
    ``null`` plus a per-node ``errors`` entry, HTTP 200), while
    ``issueOrPullRequest(number:)`` resolves either -- but, being a union,
    populates the node only through the inline fragment for its concrete type.
    A PR whose document lacks ``... on PullRequest { number state }`` (or an
    issue whose document lacks ``... on Issue { number state }``) comes back as
    ``{}``: present, not ``null``, no ``errors`` entry, no ``state``.

    Anything that is not the batched document -- i.e. the per-issue fallback --
    fails the test: the point of the fix is that no fallback request is made.
    """

    def handler(request: object) -> Response:
        document = getattr(request, "document", "")
        assert "s_" in document, f"unexpected non-batch request (per-issue fallback?): {document}"
        repo: dict[str, dict | None] = {}
        errors: list[dict] = []
        for number, lookup, selection in _aliases(document):
            node: dict | None
            if lookup == "issue":
                node = {"number": number, "state": issues[number]} if number in issues else None
            elif number in issues:
                found = _selects_state(selection, "Issue")
                node = {"number": number, "state": issues[number]} if found else {}
            elif number in pulls:
                found = _selects_state(selection, "PullRequest")
                node = {"number": number, "state": pulls[number]} if found else {}
            else:
                node = None
            repo[f"s_{number}"] = node
            if node is None:
                errors.append(
                    {
                        "message": f"Could not resolve to an Issue with the number of {number}.",
                        "path": ["repository", f"s_{number}"],
                    }
                )
        body = {"data": {"repository": repo}, "errors": errors}
        return Response(
            200, (), json.dumps(body), "http", graphql_errors=parse_graphql_errors(errors)
        )

    return FakeAdapter("http", handler=handler)


def _capture_events(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []

    def fake_log_event(state_path, kind, payload, **kwargs):
        events.append((kind, payload))

    monkeypatch.setattr(issues_module, "log_event", fake_log_event)
    return events


def _sidecar(repo_root: Path) -> Path:
    state_path = circuit_breaker_state_path(None, repo_root)
    return layout.issue_state_unresolved_path(state_path.parent)


class _Fleet:
    """The fleet supervisor's lifecycle: a brand-new ``GitHub`` for every pass
    over one repo (``fleet_dispatch.fleet_loop``), batch results scripted per
    pass through *unresolvable*."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.unresolvable: set[int] = set()
        self.events = _capture_events(monkeypatch)
        unresolvable = self.unresolvable

        def fake_states(gh, issue_numbers):
            return {n: True for n in issue_numbers if n not in unresolvable}

        def fake_issue_view(gh, number):
            return {"number": number, "state": "CLOSED"}

        monkeypatch.setattr(GitHub, "_graphql_issue_states", fake_states)
        monkeypatch.setattr(Issues, "issue_view", fake_issue_view)

    def pass_(self, numbers: list[int]) -> set[int]:
        gh = GitHub(self.repo_root)
        gh._list_cache[("_repo_owner_name",)] = ("o", "r")
        return gh.are_issues_open(numbers)

    def reported(self) -> list[list[int]]:
        return [payload["unresolved"] for kind, payload in self.events if kind == KIND]


# ---------------------------------------------------------------------------
# root cause: a PR number resolves inside the batch
# ---------------------------------------------------------------------------


def test_pull_request_blockers_resolve_in_the_batch_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The incident shape: a blocker that is a PR number. One batched request
    answers every number -- issue, open PR, merged PR, closed PR -- so there is
    no per-issue fallback and no ``github_issue_state_partial_fallback``."""
    events = _capture_events(monkeypatch)
    http = _github_emulator(
        issues={10: "OPEN", 11: "CLOSED"},
        pulls={2930: "MERGED", 2931: "OPEN", 2932: "CLOSED"},
    )
    gh, http, _gh = make_github(tmp_path, http=http)

    # Open-ness is "state == OPEN", exactly as the per-issue fallback reads a
    # PR: an open PR still blocks; a merged or closed one no longer does.
    assert gh.are_issues_open([10, 11, 2930, 2931, 2932]) == {10, 2931}

    [batch] = http.api_requests
    for number in (10, 11, 2930, 2931, 2932):
        assert f"s_{number}: issueOrPullRequest(number: {number})" in batch.document
    # A union has no fields of its own: without BOTH inline fragments GitHub
    # answers `{}` for the node of the omitted type (the emulator models this),
    # which the parser would read as closed. Pin the document, not only the result.
    assert "... on Issue" in batch.document and "... on PullRequest" in batch.document
    assert [e for e in events if e[0] == KIND] == []
    # Nothing unresolved, so nothing is written to the baseline either.
    assert not _sidecar(tmp_path).exists()


def test_emulator_models_the_old_lookup_failing_on_a_pull_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Positive control for the test above: against the pre-fix ``issue(...)``
    document the same emulator DOES leave the PR number unresolved. Without
    this, the test above could pass for a reason other than the document."""
    http = _github_emulator(issues={10: "OPEN"}, pulls={2930: "OPEN"})
    gh, http, _gh = make_github(tmp_path, http=http)
    old_document = (
        "query($owner: String!, $name: String!) { repository(owner: $owner, name: $name) "
        "{ s_10: issue(number: 10) { number state } s_2930: issue(number: 2930) { number state } } }"
    )

    reply = http.handler(type("Req", (), {"document": old_document})())  # type: ignore[misc]

    body = json.loads(reply.body)
    assert body["data"]["repository"]["s_10"] == {"number": 10, "state": "OPEN"}
    assert body["data"]["repository"]["s_2930"] is None
    assert [e["path"] for e in body["errors"]] == [["repository", "s_2930"]]


@pytest.mark.parametrize(
    ("selection", "issue_node", "pull_node"),
    [
        ("... on Issue { number state } ... on PullRequest { number state }", True, True),
        ("... on Issue { number state }", True, False),  # PullRequest fragment dropped
        ("... on PullRequest { number state }", False, True),  # Issue fragment dropped
        ("... on Issue { number } ... on PullRequest { number }", False, False),  # no `state`
    ],
)
def test_emulator_populates_a_union_node_only_through_its_own_fragment(
    tmp_path: Path, selection: str, issue_node: bool, pull_node: bool
) -> None:
    """Positive control for the fragment assertions above: the emulator returns
    `{}` (present, not null, no errors) for a node whose concrete type has no
    fragment selecting ``number`` and ``state``. This is the response shape
    that makes a dropped fragment dangerous -- it is not an error to retry --
    and it is what lets the test above fail when a fragment is removed."""
    http = _github_emulator(issues={10: "OPEN"}, pulls={2930: "OPEN"})
    document = (
        f"query {{ repository {{ s_10: issueOrPullRequest(number: 10) {{ {selection} }} "
        f"s_2930: issueOrPullRequest(number: 2930) {{ {selection} }} }} }}"
    )

    reply = http.handler(type("Req", (), {"document": document})())  # type: ignore[misc]

    body = json.loads(reply.body)
    assert body["errors"] == []
    assert body["data"]["repository"]["s_10"] == (
        {"number": 10, "state": "OPEN"} if issue_node else {}
    )
    assert body["data"]["repository"]["s_2930"] == (
        {"number": 2930, "state": "OPEN"} if pull_node else {}
    )


@pytest.mark.parametrize("stateless", [{}, {"number": 12}, {"number": 12, "state": None}])
def test_a_node_without_a_state_is_unresolved_not_closed(
    tmp_path: Path, stateless: dict[str, object]
) -> None:
    """GitHub answers ``{}`` (present, no ``errors`` entry) for a union node
    whose fragment is missing. It must reach the caller as "unresolved" -- absent
    from the mapping, so the scoped per-issue fallback decides -- never as
    "not OPEN", which would silently unblock an issue whose blocker is open."""
    body = {"data": {"repository": {"s_11": {"number": 11, "state": "OPEN"}, "s_12": stateless}}}
    gh, _http, _gh = make_github(tmp_path, http=FakeAdapter("http", [ok(body)]))

    assert gh._graphql_issue_states([11, 12]) == {11: True}


def test_a_dropped_pull_request_fragment_degrades_to_the_fallback_not_to_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End to end, if the document ever regresses to selecting only
    ``... on Issue``, an open-PR blocker is re-fetched per issue and still
    blocks -- it is not read as closed. (The test that pins the document is
    ``test_pull_request_blockers_resolve_in_the_batch_without_fallback``; this
    one pins the safety net behind it.)"""
    events = _capture_events(monkeypatch)
    monkeypatch.setattr(
        graphql_issue_states_module,
        "_issue_state_fields",
        lambda chunk: " ".join(
            f"s_{n}: issueOrPullRequest(number: {n}) {{ ... on Issue {{ number state }} }}"
            for n in chunk
        ),
    )
    fetched: list[int] = []

    def issue_view(self: Issues, number: int) -> dict:
        fetched.append(number)
        return {"number": number, "state": "OPEN"}

    monkeypatch.setattr(Issues, "issue_view", issue_view)
    http = _github_emulator(issues={10: "OPEN"}, pulls={2931: "OPEN"})
    gh, _http, _gh = make_github(tmp_path, http=http)

    assert gh.are_issues_open([10, 2931]) == {10, 2931}
    assert fetched == [2931]
    assert [payload["unresolved"] for kind, payload in events if kind == KIND] == [[2931]]


# ---------------------------------------------------------------------------
# once-only reporting survives a fresh client per pass
# ---------------------------------------------------------------------------


def test_persistent_unresolved_number_reports_once_across_fresh_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fleet = _Fleet(monkeypatch, tmp_path)
    fleet.unresolvable.add(361)

    for _ in range(5):
        assert fleet.pass_([1, 361]) == {1}

    assert fleet.reported() == [[361]]
    assert json.loads(_sidecar(tmp_path).read_text(encoding="utf-8"))["unresolved"] == [361]


def test_resolved_then_stale_again_reports_afresh_across_fresh_clients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The edge semantics survive process/instance boundaries: a number that a
    later batch positively resolves leaves the baseline, so going stale again
    is a new edge."""
    fleet = _Fleet(monkeypatch, tmp_path)

    fleet.unresolvable.add(361)
    fleet.pass_([1, 361])  # stale: report
    fleet.pass_([1, 361])  # still stale: silent
    assert fleet.reported() == [[361]]

    fleet.unresolvable.clear()
    fleet.pass_([1, 361])  # resolved: leaves the baseline, no event
    assert fleet.reported() == [[361]]
    assert json.loads(_sidecar(tmp_path).read_text(encoding="utf-8"))["unresolved"] == []

    fleet.unresolvable.add(361)
    fleet.pass_([1, 361])  # stale again: a fresh edge
    assert fleet.reported() == [[361], [361]]


def test_unrelated_clean_batch_does_not_clear_the_durable_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fleet = _Fleet(monkeypatch, tmp_path)
    fleet.unresolvable.add(361)

    for _ in range(3):
        fleet.pass_([1, 361])
        fleet.pass_([7, 8, 9])  # never names 361

    assert fleet.reported() == [[361]]


def test_newly_unresolved_number_reports_beside_an_already_baselined_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fleet = _Fleet(monkeypatch, tmp_path)

    fleet.unresolvable.add(361)
    fleet.pass_([1, 361])
    fleet.unresolvable.add(400)
    fleet.pass_([1, 361, 400])  # 400 is a new edge
    fleet.pass_([1, 361, 400])  # nothing new

    assert fleet.reported() == [[361], [361, 400]]


def test_a_failing_whole_batch_neither_reports_nor_disturbs_the_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fleet = _Fleet(monkeypatch, tmp_path)
    fleet.unresolvable.add(361)
    fleet.pass_([1, 361])
    before = _sidecar(tmp_path).read_text(encoding="utf-8")

    def broken(gh, issue_numbers):
        raise OSError("transport down")

    monkeypatch.setattr(GitHub, "_graphql_issue_states", broken)
    fleet.pass_([1, 361])

    assert fleet.reported() == [[361]]
    assert _sidecar(tmp_path).read_text(encoding="utf-8") == before


def test_unusable_sidecar_falls_back_to_the_instance_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If the sidecar cannot be written the client degrades to the pre-#2901
    behaviour (instance-local baseline) instead of raising or going silent."""
    fleet = _Fleet(monkeypatch, tmp_path)
    fleet.unresolvable.add(361)

    def disk_full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(baseline, "write_json_atomic", disk_full)

    gh = GitHub(tmp_path)
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    assert gh.are_issues_open([1, 361]) == {1}
    gh._list_cache.clear()
    gh._list_cache[("_repo_owner_name",)] = ("o", "r")
    assert gh.are_issues_open([1, 361]) == {1}  # same instance: still deduped
    assert fleet.reported() == [[361]]

    assert fleet.pass_([1, 361]) == {1}  # fresh instance: degraded, reports again
    assert fleet.reported() == [[361], [361]]
    assert not _sidecar(tmp_path).exists()


# ---------------------------------------------------------------------------
# the sidecar itself
# ---------------------------------------------------------------------------


def test_settle_reported_edge_sequence(tmp_path: Path) -> None:
    path = tmp_path / "state" / "issue-state-unresolved.json"

    assert baseline.settle_reported(path, resolved=[], unresolved=[361]) == {361}
    assert baseline.settle_reported(path, resolved=[], unresolved=[361]) == frozenset()
    # Another number is fresh; the baselined one is not.
    assert baseline.settle_reported(path, resolved=[], unresolved=[361, 400]) == {400}
    # A batch that never names 361 leaves it in the baseline.
    assert baseline.settle_reported(path, resolved=[7], unresolved=[]) == frozenset()
    assert baseline.load_reported(path) == {361, 400}
    # Positively resolved: leaves the baseline, so re-stale is fresh.
    assert baseline.settle_reported(path, resolved=[361], unresolved=[]) == frozenset()
    assert baseline.load_reported(path) == {400}
    assert baseline.settle_reported(path, resolved=[], unresolved=[361]) == {361}


def test_steady_state_does_not_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "issue-state-unresolved.json"
    baseline.settle_reported(path, resolved=[], unresolved=[361])

    def forbidden(*args, **kwargs):
        raise AssertionError("steady state must be a lock-free read, not a write")

    monkeypatch.setattr(baseline, "write_json_atomic", forbidden)
    monkeypatch.setattr(baseline, "advisory_file_lock", forbidden)

    assert baseline.settle_reported(path, resolved=[1, 2], unresolved=[361]) == frozenset()
    assert baseline.settle_reported(path, resolved=[], unresolved=[]) == frozenset()


def test_concurrent_observers_of_one_new_number_report_it_exactly_once(tmp_path: Path) -> None:
    """The fresh/not-fresh decision is made inside the locked read-modify-write,
    so two processes (here: threads) that both just saw 361 go stale cannot
    both report it."""
    path = tmp_path / "issue-state-unresolved.json"
    workers = 8
    barrier = threading.Barrier(workers)

    def observe(_: int) -> frozenset[int] | None:
        barrier.wait()
        return baseline.settle_reported(path, resolved=[], unresolved=[361])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(observe, range(workers)))

    assert all(r is not None for r in results)
    assert sorted(len(r) for r in results if r is not None) == [0] * (workers - 1) + [1]


def test_corrupt_sidecar_reads_empty_and_is_healed_by_the_next_settle(tmp_path: Path) -> None:
    path = tmp_path / "issue-state-unresolved.json"
    path.write_text("{not json", encoding="utf-8")

    assert baseline.load_reported(path) == frozenset()
    # Fail-safe direction: telemetry is not lost to a bookkeeping failure.
    assert baseline.settle_reported(path, resolved=[], unresolved=[361]) == {361}
    assert json.loads(path.read_text(encoding="utf-8"))["unresolved"] == [361]


@pytest.mark.parametrize(
    "document",
    [
        [1, 2],  # not an object
        {"unresolved": "361"},  # not a list
        {"unresolved": {"361": True}},  # not a list
        {"other": [361]},  # key missing
    ],
)
def test_wrong_shaped_sidecar_reads_empty(tmp_path: Path, document: object) -> None:
    path = tmp_path / "issue-state-unresolved.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    assert baseline.load_reported(path) == frozenset()


def test_load_reported_keeps_only_integer_issue_numbers(tmp_path: Path) -> None:
    path = tmp_path / "issue-state-unresolved.json"
    path.write_text(json.dumps({"unresolved": [361, True, "7", 1.5, None, 400]}), encoding="utf-8")

    assert baseline.load_reported(path) == {361, 400}


@pytest.mark.parametrize("failure", [StateLockBusy("busy"), OSError("read-only state dir")])
def test_settle_reported_returns_none_when_the_sidecar_cannot_be_updated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
) -> None:
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(baseline, "advisory_file_lock", fail)

    assert baseline.settle_reported(tmp_path / "x.json", resolved=[], unresolved=[361]) is None


def test_settle_reported_returns_none_when_the_first_read_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``Path.exists`` raises for errnos it does not treat as "missing" (e.g.
    EACCES). The lock-free read that opens ``settle_reported`` sits inside the
    same guard as the locked write, so "never raises into ``are_issues_open``"
    holds for the first read too."""
    target = tmp_path / "issue-state-unresolved.json"
    real_exists = Path.exists

    def exists(self: Path, *args: object, **kwargs: object) -> bool:
        if self == target:
            raise PermissionError(errno.EACCES, "denied", str(self))
        return real_exists(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "exists", exists)

    assert baseline.settle_reported(target, resolved=[], unresolved=[361]) is None


def test_layout_places_the_sidecar_beside_state_json(tmp_path: Path) -> None:
    state_path = circuit_breaker_state_path(None, tmp_path)

    assert _sidecar(tmp_path) == state_path.parent / "issue-state-unresolved.json"
