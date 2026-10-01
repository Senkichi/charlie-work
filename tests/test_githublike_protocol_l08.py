"""Track 2 L08 (issue #1592): MergeBranch capability move regression tests.

New file, not appended to ``test_githublike_protocol.py`` (deliberate test-
layout change, consistent with L06/L07) -- shares the AST/signature helpers
via ``tests/_githublike_protocol_helpers.py`` (issue #1284 sanctions bare-name
``tests/_*.py`` modules for this kind of cross-file sharing).

Covers all seven Cluster G members: ``merge_pr``, ``delete_branch``,
``pr_update_branch``, ``pr_close``, ``pr_reopen``, ``push_empty_commit``,
``branch_protection``. Four of these (``branch_protection``, ``pr_close``,
``pr_reopen``, ``push_empty_commit``) are among the five members asserted
present in ``GitHubLike.__dict__`` (tests/test_githublike_protocol.py:33-84);
this file adds its own independent assertion of that fact so the L08 move has
leaf-scoped regression coverage, mirroring L07's precedent for the analogous
``are_issues_open``/``issue_view`` pair. Unlike L07, none of the seven moved
bodies calls a sibling moved member through ``self`` -- there is no
must-not-split constraint and no interception-path relocation to disclose
for this leaf.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from charlie_work.github import GitHub, GitHubLike, _ROUTES
from charlie_work.github_capabilities import MergeBranchLike
from charlie_work.github_capabilities._base import GitHubRunResult

from _githublike_protocol_helpers import _compatible_signature, _lexical_github_defs
from _fake_transport import FakeAdapter, make_github, ok, sent

MERGE_BRANCH_MOVED_MEMBERS = (
    "merge_pr",
    "delete_branch",
    "pr_update_branch",
    "pr_close",
    "pr_reopen",
    "push_empty_commit",
    "branch_protection",
)

# Subset also asserted present directly in GitHubLike.__dict__ (redeclared on
# the union body per design doc Section 4.1) -- test_githublike_protocol.py
# already asserts these five names individually (33-84; pr_ready is the fifth,
# owned by PullRequests/L06). This file adds independent, leaf-scoped
# coverage for the four that belong to this leaf, not a duplicate authority.
DICT_ASSERTED_MEMBERS = ("branch_protection", "pr_close", "pr_reopen", "push_empty_commit")


def test_github_isinstance_mergebranchlike(tmp_path: Path) -> None:
    """``GitHub`` must satisfy ``MergeBranchLike`` at runtime after the L08 move.

    A standalone ``isinstance`` assertion so this specific claim (survives the
    L08 move of all seven merge/branch members off ``GitHub``) has its own
    named test independent of the broader signature-conformance loop in
    ``test_githublike_protocol.py``.
    """
    assert isinstance(GitHub(tmp_path), MergeBranchLike)


def test_mergebranch_members_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define any of the seven moved members.

    Track 2 L08 (issue #1592) moved ``merge_pr``, ``delete_branch``,
    ``pr_update_branch``, ``pr_close``, ``pr_reopen``, ``push_empty_commit``,
    and ``branch_protection`` to ``MergeBranch``; the names are now served on
    ``GitHub`` by L01 generated delegates (class-level assignments, not
    ``def``s).
    """
    lexical = _lexical_github_defs()
    for name in MERGE_BRANCH_MOVED_MEMBERS:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    # And confirm the names *are* still resolvable, as delegates.
    for name in MERGE_BRANCH_MOVED_MEMBERS:
        assert hasattr(getattr(GitHub, name), "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )


def test_mergebranch_routes_point_at_the_merge_branch_collaborator() -> None:
    """``_ROUTES`` must route all seven moved names to the ``_merge_branch`` collaborator.

    Forward-compatible: asserts only the seven entries this leaf (L08) adds,
    not the full ``_ROUTES`` contents, so it keeps holding unmodified once
    later leaves populate more of the table.
    """
    for name in MERGE_BRANCH_MOVED_MEMBERS:
        assert _ROUTES[name] == "_merge_branch"


def test_mergebranch_members_signature_compatible() -> None:
    """Each moved member's concrete ``GitHub`` signature must match the
    ``MergeBranchLike`` protocol declaration (name/kind/return), through the
    installed delegate's ``__signature__``.
    """
    for name in MERGE_BRANCH_MOVED_MEMBERS:
        proto_sig = inspect.signature(getattr(MergeBranchLike, name))
        concrete_sig = inspect.signature(getattr(GitHub, name))
        _compatible_signature(proto_sig, concrete_sig)


def test_githublike_dict_still_has_the_four_redeclared_members() -> None:
    """The four ``__dict__``-asserted names must still be declared directly on
    ``GitHubLike``'s own body after L08 (design doc Section 4.1 redeclaration),
    independent of which sub-protocol/collaborator implements them and
    independent of ``test_githublike_protocol.py``'s own equivalent
    assertions (33-84).
    """
    for name in DICT_ASSERTED_MEMBERS:
        assert name in GitHubLike.__dict__, f"GitHubLike is missing {name} from its own __dict__"


def test_branch_protection_delegate_uses_owner_shared_list_cache(tmp_path: Path) -> None:
    """``branch_protection``'s cache read/write must hit the OWNER's ``_list_cache``.

    The collaborator's ``_list_cache`` attribute must resolve (via
    ``__getattr__``) to the very same dict object the owner holds.
    """
    gh = GitHub(tmp_path)
    assert gh._merge_branch._list_cache is gh._list_cache
    assert gh._merge_branch.__dict__.get("_list_cache") is None


# --- Behavioural tests, one per moved member, through a fake owner (a real
# ``GitHub`` instance with ``GitHub.run`` monkeypatched) --------------------


def test_merge_pr_delegate_forwards_through_run(tmp_path: Path) -> None:
    """``gh.merge_pr(...)`` PUTs the REST merge endpoint with the strategy as
    ``merge_method`` and returns the API's own message (B8).
    """
    reply = ok({"merged": True, "message": "Pull Request successfully merged"})
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    result = gh.merge_pr(42, "squash")

    assert result == "Pull Request successfully merged"
    assert sent(http) == [
        ("PUT", "repos/{owner}/{repo}/pulls/42/merge", {"merge_method": "squash"})
    ]


def test_merge_pr_delegate_admin_flag_and_merge_flags_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``merge_flags`` takes precedence over the legacy ``admin`` field, and
    ``run``'s literal output (when non-empty) is returned as-is.
    """
    calls: list[list[str]] = []

    def fake_run(self: GitHub, args: list[str]) -> str:
        calls.append(args)
        return "https://github.com/o/r/pull/42 merged"

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.merge_pr(42, "rebase", admin=True, merge_flags=("--auto",))

    assert result == "https://github.com/o/r/pull/42 merged"
    assert calls == [["pr", "merge", "42", "--auto", "--rebase"]]
    assert "--admin" not in calls[0]


def test_delete_branch_delegate_forwards_through_run(tmp_path: Path) -> None:
    """``gh.delete_branch(branch)`` DELETEs the git-refs endpoint and returns
    ``True`` on success.
    """
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok("", status=204)]))
    result = gh.delete_branch("agent/issue-1-slug")

    assert result is True
    assert sent(http) == [
        ("DELETE", "repos/{owner}/{repo}/git/refs/heads/agent/issue-1-slug", None)
    ]


def test_delete_branch_delegate_returns_false_on_githuberror(tmp_path: Path) -> None:
    """``delete_branch`` swallows ``GitHubError`` and returns ``False`` -- the
    identity-sensitive ``ci_fleet.github.GitHubError`` imported into
    ``merge_branch.py`` must be the *same* type the typed send raises, or this
    ``except`` would not catch it.
    """
    reply = ok({"message": "Reference does not exist"}, status=422)
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    assert gh.delete_branch("agent/issue-1-slug") is False


def test_pr_update_branch_delegate_forwards_through_run(tmp_path: Path) -> None:
    """``gh.pr_update_branch(n)`` PUTs ``pulls/{n}/update-branch`` and returns ``True``."""
    reply = ok({"message": "Updating"}, status=202)
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    result = gh.pr_update_branch(456)

    assert result is True
    assert sent(http) == [("PUT", "repos/{owner}/{repo}/pulls/456/update-branch", None)]


def test_pr_update_branch_delegate_returns_false_on_githuberror(tmp_path: Path) -> None:
    """``pr_update_branch`` swallows ``GitHubError`` and returns ``False``."""
    reply = ok({"message": "conflict"}, status=422)
    gh, _, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    assert gh.pr_update_branch(456) is False


def test_pr_close_delegate_dry_run_returns_synthetic_ok(tmp_path: Path) -> None:
    """Under ``dry_run``, ``pr_close`` returns an ``ok=True`` result and the
    transport sends nothing (the dry-run guard lives in the transport now, B2).
    """
    gh, http, _ = make_github(tmp_path, dry_run=True)
    result = gh.pr_close(7)

    assert isinstance(result, GitHubRunResult)
    assert result.ok is True
    assert http.calls == []


def test_pr_close_delegate_forwards_through_run(tmp_path: Path) -> None:
    """``gh.pr_close(n)`` PATCHes the pull with ``state=closed``."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"state": "closed"})]))
    result = gh.pr_close(7)

    assert isinstance(result, GitHubRunResult)
    assert result.ok is True
    assert sent(http) == [("PATCH", "repos/{owner}/{repo}/pulls/7", {"state": "closed"})]


def test_pr_reopen_delegate_dry_run_returns_synthetic_ok(tmp_path: Path) -> None:
    """Under ``dry_run``, ``pr_reopen`` returns an ``ok=True`` result and the
    transport sends nothing (the dry-run guard lives in the transport now, B2).
    """
    gh, http, _ = make_github(tmp_path, dry_run=True)
    result = gh.pr_reopen(7)

    assert isinstance(result, GitHubRunResult)
    assert result.ok is True
    assert http.calls == []


def test_pr_reopen_delegate_forwards_through_run(tmp_path: Path) -> None:
    """``gh.pr_reopen(n)`` PATCHes the pull with ``state=open``."""
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [ok({"state": "open"})]))
    result = gh.pr_reopen(7)

    assert isinstance(result, GitHubRunResult)
    assert result.ok is True
    assert sent(http) == [("PATCH", "repos/{owner}/{repo}/pulls/7", {"state": "open"})]


def test_push_empty_commit_delegate_dry_run_returns_synthetic_ok_without_calling_run(
    tmp_path: Path,
) -> None:
    """Under ``dry_run``, ``push_empty_commit`` must short-circuit before any
    request is sent -- it is unconditionally mutating end to end.
    """
    gh, http, _ = make_github(tmp_path, dry_run=True)
    result = gh.push_empty_commit("agent/issue-1-slug")

    assert isinstance(result, GitHubRunResult)
    assert result.ok is True
    assert http.calls == []


def test_push_empty_commit_delegate_forwards_the_full_four_step_sequence(tmp_path: Path) -> None:
    """``gh.push_empty_commit(branch)`` happy path issues the four requests in
    order (ref GET, commit GET, commit POST, ref PATCH) and returns the final
    ref-update ``GitHubRunResult``.
    """
    replies = [
        ok({"object": {"sha": "tip-sha"}}),
        ok({"tree": {"sha": "tree-sha"}}),
        ok({"sha": "new-sha"}),
        ok({"sha": "new-sha"}),
    ]
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", replies))
    result = gh.push_empty_commit("agent/issue-1-slug")

    assert result.ok is True
    assert result.value == {"sha": "new-sha"}
    calls = sent(http)
    assert len(calls) == 4
    assert calls[0] == ("GET", "repos/{owner}/{repo}/git/refs/heads/agent/issue-1-slug", None)
    assert calls[1] == ("GET", "repos/{owner}/{repo}/git/commits/tip-sha", None)
    assert calls[2][:2] == ("POST", "repos/{owner}/{repo}/git/commits")
    assert calls[2][2]["tree"] == "tree-sha"  # type: ignore[index]
    assert calls[2][2]["parents"] == ["tip-sha"]  # type: ignore[index]
    assert calls[3] == (
        "PATCH",
        "repos/{owner}/{repo}/git/refs/heads/agent/issue-1-slug",
        {"sha": "new-sha"},
    )


def test_branch_protection_delegate_forwards_through_run_and_caches(tmp_path: Path) -> None:
    """``gh.branch_protection(base)`` cache-miss path sends one GET and caches
    the parsed value in the owner's shared ``_list_cache``; a second call must
    not send again.
    """
    reply = ok({"required_status_checks": {"strict": True}})
    gh, http, _ = make_github(tmp_path, http=FakeAdapter("http", [reply]))
    first = gh.branch_protection("main")
    second = gh.branch_protection("main")

    assert first == {"required_status_checks": {"strict": True}}
    assert second == first
    assert sent(http) == [("GET", "repos/{owner}/{repo}/branches/main/protection", None)]
    assert gh._list_cache[("branch_protection", "main")] == first
