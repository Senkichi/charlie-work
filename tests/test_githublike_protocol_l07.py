"""Track 2 L07 (issue #1591): Issues capability move regression tests.

New file, not appended to ``test_githublike_protocol.py`` (deliberate test-
layout change for this leaf) -- shares the AST/signature helpers via
``tests/_githublike_protocol_helpers.py`` (issue #1284 sanctions bare-name
``tests/_*.py`` modules for this kind of cross-file sharing).

Covers all five Cluster F members: ``close_issue``, ``issue_view``,
``issue_list``, ``issue_dependencies``, ``are_issues_open``. ``issue_view`` and
``are_issues_open`` move together (the must-not-split constraint): the latter's
thread-pool fallback closure calls ``self.issue_view``, which must resolve on
the same collaborator -- exercised directly by
``test_are_issues_open_fallback_closure_runs_through_collaborator``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import charlie_work.github as _github_module
from charlie_work.github import GitHub, _ROUTES
from charlie_work.github_capabilities import ISSUE_VIEW_FIELDS, IssuesLike

from _githublike_protocol_helpers import _compatible_signature, _lexical_github_defs

ISSUES_MOVED_MEMBERS = (
    "close_issue",
    "issue_view",
    "issue_list",
    "issue_dependencies",
    "are_issues_open",
)


def test_github_isinstance_issueslike(tmp_path: Path) -> None:
    """``GitHub`` must satisfy ``IssuesLike`` at runtime after the L07 move.

    A standalone ``isinstance`` assertion so this specific claim (survives the
    L07 move of all five issue members off ``GitHub``) has its own named test
    independent of the broader signature-conformance loop in
    ``test_githublike_protocol.py``.
    """
    assert isinstance(GitHub(tmp_path), IssuesLike)


def test_issues_members_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define any of the five moved members.

    Track 2 L07 (issue #1591) moved ``close_issue``, ``issue_view``,
    ``issue_list``, ``issue_dependencies`` and ``are_issues_open`` to
    ``Issues``; the names are now served on ``GitHub`` by L01 generated
    delegates (class-level assignments, not ``def``s).
    """
    lexical = _lexical_github_defs()
    for name in ISSUES_MOVED_MEMBERS:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    # And confirm the names *are* still resolvable, as delegates.
    for name in ISSUES_MOVED_MEMBERS:
        assert hasattr(getattr(GitHub, name), "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )


def test_issues_routes_point_at_the_issues_collaborator() -> None:
    """``_ROUTES`` must route all five moved names to the ``_issues`` collaborator.

    Forward-compatible: asserts only the five entries this leaf (L07) adds, not
    the full ``_ROUTES`` contents, so it keeps holding unmodified once later
    leaves populate more of the table.

    ``_graphql_issue_dependencies``/``_graphql_issue_states`` were NOT routed
    at L07 time -- they stayed lexical owner methods until Track 2 L09 (issue
    #1593), which moved them to the ``Transport`` collaborator. Updated here
    (rather than asserting the now-false "not in _ROUTES") because L09's move
    of these two names directly falsifies the original assertion; L09's own
    PR body (tests/test_githublike_protocol_l09.py) covers them as first-class
    members of that leaf.
    """
    for name in ISSUES_MOVED_MEMBERS:
        assert _ROUTES[name] == "_issues"
    assert _ROUTES["_graphql_issue_dependencies"] == "_transport"
    assert _ROUTES["_graphql_issue_states"] == "_transport"


def test_issues_members_signature_compatible() -> None:
    """Each moved member's concrete ``GitHub`` signature must match the
    ``IssuesLike`` protocol declaration (name/kind/return), through the
    installed delegate's ``__signature__``.
    """
    for name in ISSUES_MOVED_MEMBERS:
        proto_sig = inspect.signature(getattr(IssuesLike, name))
        concrete_sig = inspect.signature(getattr(GitHub, name))
        _compatible_signature(proto_sig, concrete_sig)


def test_close_issue_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh.close_issue(n)`` through the delegate must reach the patched
    class-level ``GitHub.run`` with the argv the moved body produces and
    return ``True``.

    The expected argv is transcribed by reading the moved ``Issues.close_issue``
    body directly, not derived by calling the code under test.
    """
    calls: list[list[str]] = []

    def fake_run(self: GitHub, args: list[str]) -> str:
        calls.append(args)
        return ""

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.close_issue(5)

    assert result is True
    assert calls == [["issue", "close", "5"]]


def test_close_issue_delegate_returns_false_on_githuberror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``close_issue`` swallows ``GitHubError`` and returns ``False`` -- the
    identity-sensitive ``ci_fleet.github.GitHubError`` imported into
    ``issues.py`` must be the *same* type ``run`` raises, or this ``except``
    would not catch it.
    """

    def fake_run(self: GitHub, args: list[str]) -> str:
        raise _github_module.GitHubError("boom")

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    assert gh.close_issue(5) is False


def test_issue_view_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh.issue_view(n)`` through the delegate must reach the patched
    class-level ``GitHub.run`` with the argv the moved body produces, using the
    ``ISSUE_VIEW_FIELDS`` bare global relocated to ``issues.py``.
    """
    calls: list[tuple[list[str], bool]] = []

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False) -> dict:
        calls.append((args, json_output))
        return {"number": 7, "state": "OPEN"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.issue_view(7)

    assert result == {"number": 7, "state": "OPEN"}
    assert calls == [(["issue", "view", "7", "--json", ISSUE_VIEW_FIELDS], True)]


def test_issue_list_delegate_forwards_through_list_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh.issue_list()`` through the delegate must reach the patched
    class-level ``GitHub._list_json`` with the argv the moved body produces
    (including ``_LIST_LIMIT``/``ISSUE_LIST_FIELDS``).
    """
    from charlie_work.github_capabilities import ISSUE_LIST_FIELDS, _LIST_LIMIT

    calls: list[tuple[list[str], int, str]] = []

    def fake_list_json(
        self: GitHub, args: list[str], *, limit: int, kind: str
    ) -> list[dict[str, str]]:
        calls.append((args, limit, kind))
        return [{"number": "1"}]

    monkeypatch.setattr(GitHub, "_list_json", fake_list_json)

    gh = GitHub(tmp_path)
    result = gh.issue_list()

    assert result == [{"number": "1"}]
    assert calls == [
        (
            [
                "issue",
                "list",
                "--limit",
                str(_LIST_LIMIT),
                "--state",
                "open",
                "--json",
                ISSUE_LIST_FIELDS,
            ],
            _LIST_LIMIT,
            "issues (labels=all, state=open)",
        ),
    ]


def test_issue_list_delegate_uses_owner_shared_list_cache(tmp_path: Path) -> None:
    """``issue_list``'s cache read/write must hit the OWNER's ``_list_cache``.

    The collaborator's ``_list_cache`` attribute must resolve (via
    ``__getattr__``) to the very same dict object the owner holds, and a
    pre-seeded cached value must be served without any ``_list_json`` call.
    """
    gh = GitHub(tmp_path)
    # The collaborator forwards _list_cache to the owner -- same object.
    assert gh._issues._list_cache is gh._list_cache

    # Pre-seed the cache the moved body keys on and confirm it is served.
    cache_key = ("issue_list", "open", ())
    gh._list_cache[cache_key] = [{"number": "cached"}]

    result = gh.issue_list()

    assert result == [{"number": "cached"}]
    # The collaborator must not have grown its own shadowing _list_cache.
    assert gh._issues.__dict__.get("_list_cache") is None


def test_issue_dependencies_delegate_forwards_through_graphql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh.issue_dependencies(nums)`` happy path delegates to the owner's
    ``_graphql_issue_dependencies`` (which stays on ``GitHub`` until L09,
    resolving through the collaborator's ``__getattr__``) and returns its result.
    """
    seen: list[list[int]] = []

    def fake_graphql_deps(self: GitHub, issue_numbers: list[int]) -> dict[int, list[int]]:
        seen.append(issue_numbers)
        return {1: [2, 3], 4: []}

    monkeypatch.setattr(GitHub, "_graphql_issue_dependencies", fake_graphql_deps)

    gh = GitHub(tmp_path)
    result = gh.issue_dependencies([1, 4])

    assert result == {1: [2, 3], 4: []}
    assert seen == [[1, 4]]


def test_issue_dependencies_fallback_runs_relocated_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``_graphql_issue_dependencies`` fails, ``issue_dependencies`` falls
    back to the module-level ``get_github_issue_dependencies`` relocated into
    ``issues.py`` in this leaf, which must resolve on the collaborator's owner
    and hit the REST ``blocked_by`` endpoint via ``run``.
    """

    def fake_graphql_deps(self: GitHub, issue_numbers: list[int]) -> dict[int, list[int]]:
        raise _github_module.GitHubError("batched query failed")

    rest_calls: list[list[str]] = []

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> list[dict[str, int]]:
        rest_calls.append(args)
        # blocked_by list for issue #9 -> blocked by #2
        return [{"number": 2}]

    monkeypatch.setattr(GitHub, "_graphql_issue_dependencies", fake_graphql_deps)
    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.issue_dependencies([9])

    assert result == {9: [2]}
    assert rest_calls == [
        ["api", "repos/{owner}/{repo}/issues/9/dependencies/blocked_by"],
    ]


def test_are_issues_open_happy_path_through_graphql(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh.are_issues_open(nums)`` batched path delegates to the owner's
    ``_graphql_issue_states`` and returns the set of open numbers.
    """

    def fake_states(self: GitHub, issue_numbers: list[int]) -> dict[int, bool]:
        return {1: True, 2: False, 3: True}

    monkeypatch.setattr(GitHub, "_graphql_issue_states", fake_states)

    gh = GitHub(tmp_path)
    assert gh.are_issues_open([1, 2, 3]) == {1, 3}


def test_are_issues_open_fallback_closure_runs_through_collaborator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The must-not-split constraint, exercised end to end.

    When ``_graphql_issue_states`` raises, ``are_issues_open`` falls back to a
    thread-pool closure that calls ``self.issue_view`` for each number. Because
    ``issue_view`` moved into ``Issues`` in this same leaf, that call resolves
    on the collaborator instance (``self`` inside the closure), not back across
    a thread boundary through the owner delegate. A fake class-level
    ``GitHub.run`` returns issue JSON for two numbers; the result must be the
    set of OPEN ones.
    """

    def fake_states(self: GitHub, issue_numbers: list[int]) -> dict[int, bool]:
        raise _github_module.GitHubError("batched state query failed")

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False) -> dict:
        # args == ["issue", "view", "<n>", "--json", ISSUE_VIEW_FIELDS]
        number = int(args[2])
        return {"number": number, "state": "OPEN" if number in (1, 2) else "CLOSED"}

    monkeypatch.setattr(GitHub, "_graphql_issue_states", fake_states)
    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.are_issues_open([1, 2, 99])

    assert result == {1, 2}
    # The fallback also warms the owner's shared per-number cache.
    assert gh._list_cache[("issue_open", 1)] is True
    assert gh._list_cache[("issue_open", 99)] is False
