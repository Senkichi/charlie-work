"""Track 2 L06 (issue #1590): PullRequests capability move regression tests.

New file, not appended to ``test_githublike_protocol.py`` (deliberate test-
layout change for this leaf) -- shares the AST/signature helpers via
``tests/_githublike_protocol_helpers.py`` (issue #1284 sanctions bare-name
``tests/_*.py`` modules for this kind of cross-file sharing).

Covers seven of Cluster E's eight members: ``pr_create``, ``pr_list``,
``merged_pr_list``, ``pr_view``, ``pr_diff``, ``pr_commits``, ``pr_ready``.
``merged_prs_for_issue`` deliberately does NOT move in this leaf (see
``github_capabilities/pull_requests.py``'s module docstring for the full
``linked_issue_number`` circular-dependency rationale) and so is intentionally
absent from every list below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import charlie_work.github as _github_module
from charlie_work.github import GitHub, _ROUTES
from charlie_work.github_capabilities import PullRequestsLike

from _githublike_protocol_helpers import _lexical_github_defs

PULL_REQUESTS_MOVED_MEMBERS = (
    "pr_create",
    "pr_list",
    "merged_pr_list",
    "pr_view",
    "pr_diff",
    "pr_commits",
    "pr_ready",
)


def test_github_isinstance_pullrequestslike(tmp_path: Path) -> None:
    """``GitHub`` must satisfy ``PullRequestsLike`` at runtime after the L06 move.

    Duplicates part of ``test_github_satisfies_pullrequestslike`` (in
    ``test_githublike_protocol.py``) deliberately: a standalone ``isinstance``
    assertion so this specific claim (survives the L06 move of seven
    pull-request members off ``GitHub``) has its own named test independent
    of the broader signature-conformance loop.
    """
    assert isinstance(GitHub(tmp_path), PullRequestsLike)


def test_pullrequests_members_are_not_lexical_github_defs() -> None:
    """``GitHub`` must no longer lexically define any of the seven moved
    PullRequests members.

    Track 2 L06 (issue #1590) moved ``pr_create``, ``pr_list``,
    ``merged_pr_list``, ``pr_view``, ``pr_diff``, ``pr_commits``, and
    ``pr_ready`` to ``PullRequests``; the names are now served on ``GitHub``
    by L01 generated delegates (class-level assignments, not ``def``s).
    ``merged_prs_for_issue`` is deliberately excluded from this list -- it
    stays a lexical ``GitHub`` method in this leaf (see the module docstring
    of ``github_capabilities/pull_requests.py``).
    """
    lexical = _lexical_github_defs()
    for name in PULL_REQUESTS_MOVED_MEMBERS:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    # And confirm the names *are* still resolvable, as delegates.
    for name in PULL_REQUESTS_MOVED_MEMBERS:
        assert hasattr(getattr(GitHub, name), "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )
    # merged_prs_for_issue is the disclosed exception: it must remain a real
    # lexical def, not a delegate, in this leaf.
    assert "merged_prs_for_issue" in lexical, (
        "merged_prs_for_issue must remain a lexical GitHub method in L06"
    )


def test_pullrequests_routes_point_at_the_pull_requests_collaborator() -> None:
    """``_ROUTES`` must route all seven moved names to the ``_pull_requests`` collaborator.

    Forward-compatible: asserts only the seven entries this leaf (L06) adds,
    not the full ``_ROUTES`` contents, so it keeps holding unmodified once
    later leaves populate more of the table.
    """
    for name in PULL_REQUESTS_MOVED_MEMBERS:
        assert _ROUTES[name] == "_pull_requests"
    # merged_prs_for_issue must NOT be routed -- it stays lexical on GitHub.
    assert "merged_prs_for_issue" not in _ROUTES


def test_pr_create_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_create(...)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, and parse the PR number out of the returned URL.

    The expected argv is transcribed by reading the moved
    ``PullRequests.pr_create`` body directly, not derived by calling the
    same code under test.
    """
    calls: list[tuple[list[str], bool]] = []

    def fake_run(
        self: GitHub, args: list[str], *, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, allow_failure))
        return _github_module.GitHubRunResult(
            ok=True,
            returncode=0,
            stdout="",
            stderr="",
            value="https://github.com/OWNER/REPO/pull/42",
        )

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.pr_create("feature-branch", "main", "My title", "My body")

    assert result == 42
    assert calls == [
        (
            [
                "pr",
                "create",
                "--head",
                "feature-branch",
                "--base",
                "main",
                "--title",
                "My title",
                "--body",
                "My body",
            ],
            True,
        ),
    ]


def test_pr_create_delegate_returns_none_on_dry_run(tmp_path: Path) -> None:
    """``pr_create`` returns ``0`` under dry-run without calling ``run`` at all --
    transcribed directly from the moved body's ``if self.dry_run: return 0`` guard.
    """
    gh = GitHub(tmp_path, dry_run=True)
    result = gh.pr_create("feature-branch", "main", "title", "body")
    assert result == 0


def test_pr_list_delegate_forwards_through_list_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_list()`` through the delegate must reach the patched
    class-level ``GitHub._list_json`` with the same argv the moved body
    produces (including ``_LIST_LIMIT``/``PR_LIST_FIELDS``).

    The expected argv is transcribed by reading the moved
    ``PullRequests.pr_list`` body directly.
    """
    from charlie_work.github_capabilities import PR_LIST_FIELDS, _LIST_LIMIT

    calls: list[tuple[list[str], int, str]] = []

    def fake_list_json(
        self: GitHub, args: list[str], *, limit: int, kind: str
    ) -> list[dict[str, str]]:
        calls.append((args, limit, kind))
        return [{"number": "1"}]

    monkeypatch.setattr(GitHub, "_list_json", fake_list_json)

    gh = GitHub(tmp_path)
    result = gh.pr_list()

    assert result == [{"number": "1"}]
    assert calls == [
        (
            [
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                str(_LIST_LIMIT),
                "--json",
                PR_LIST_FIELDS,
            ],
            _LIST_LIMIT,
            "open PRs",
        ),
    ]


def test_pr_list_delegate_uses_owner_shared_list_cache(tmp_path: Path) -> None:
    """``pr_list``'s cache read/write must hit the OWNER's ``_list_cache``, the
    same shared-state pattern L05 established for ``invalidate_list_cache``.
    """
    gh = GitHub(tmp_path)
    gh._list_cache[("pr_list",)] = [{"number": "cached"}]

    result = gh.pr_list()

    assert result == [{"number": "cached"}]
    assert gh._pull_requests.__dict__.get("_list_cache") is None


def test_merged_pr_list_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.merged_pr_list()`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces for its first REST page, and filter to merged PRs only.

    The expected argv is transcribed by reading the moved
    ``PullRequests.merged_pr_list`` body directly.
    """
    calls: list[tuple[list[str], bool]] = []

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False) -> list[dict]:
        calls.append((args, json_output))
        if len(calls) == 1:
            return [
                {"number": 1, "merged_at": "2026-01-01T00:00:00Z"},
                {"number": 2, "merged_at": None},
            ]
        return []

    def fake_normalize_rest_pr(self: GitHub, pr: dict) -> dict:
        return pr

    monkeypatch.setattr(GitHub, "run", fake_run)
    monkeypatch.setattr(GitHub, "_normalize_rest_pr", fake_normalize_rest_pr)

    gh = GitHub(tmp_path)
    result = gh.merged_pr_list()

    assert result == [{"number": 1, "merged_at": "2026-01-01T00:00:00Z"}]
    assert calls[0] == (
        [
            "api",
            "repos/{owner}/{repo}/pulls?state=closed&sort=updated&direction=desc&per_page=100&page=1",
        ],
        True,
    )


def test_merged_pr_list_delegate_raises_githuberror_on_bad_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``merged_pr_list`` must still raise ``GitHubError`` (not some
    PullRequests-local, unrelated exception type) when ``gh api`` returns a
    non-list shape, exercising the identity-sensitive
    ``ci_fleet.github.GitHubError`` import this leaf adds to
    ``pull_requests.py`` (see that module's import block: a local
    re-declaration would be a structurally identical but unrelated type that
    no ``except GitHubError`` handler in the rest of the codebase would
    catch).
    """

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False) -> dict:
        return {"unexpected": "shape"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    with pytest.raises(_github_module.GitHubError):
        gh.merged_pr_list()


def test_pr_view_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_view(number)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, using the default ``PR_VIEW_FIELDS`` bound at *def* time.

    The expected argv is transcribed by reading the moved
    ``PullRequests.pr_view`` body directly.
    """
    from charlie_work.github_capabilities import PR_VIEW_FIELDS

    calls: list[tuple[list[str], bool]] = []

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False) -> dict:
        calls.append((args, json_output))
        return {"number": 7, "title": "t"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.pr_view(7)

    assert result == {"number": 7, "title": "t"}
    assert calls == [
        (["pr", "view", "7", "--json", PR_VIEW_FIELDS], True),
    ]


def test_pr_diff_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_diff(number)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, unwrapping a ``GitHubRunResult`` to its ``.value`` string, and
    must recognize a real ``GitHubRunResult`` returned across the
    collaborator boundary via ``isinstance`` (the ``github_capabilities/
    _base.py``-defined, ``github.py``-re-exported mechanism).

    The expected argv is transcribed by reading the moved
    ``PullRequests.pr_diff`` body directly.
    """
    calls: list[tuple[list[str], bool]] = []
    diff_text = "diff --git a/x b/x\n+added line\n"

    def fake_run(
        self: GitHub, args: list[str], *, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, allow_failure))
        return _github_module.GitHubRunResult(
            ok=True, returncode=0, stdout=diff_text, stderr="", value=diff_text
        )

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.pr_diff(9)

    assert result == diff_text
    assert calls == [
        (["pr", "diff", "9"], True),
    ]


def test_pr_commits_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_commits(number)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, unwrapping a ``GitHubRunResult`` to its ``.value`` list.

    The expected argv is transcribed by reading the moved
    ``PullRequests.pr_commits`` body directly.
    """
    calls: list[tuple[list[str], bool, bool]] = []
    commits = [{"commit": {"message": "subject\n\nbody"}}]

    def fake_run(
        self: GitHub, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, json_output, allow_failure))
        return _github_module.GitHubRunResult(
            ok=True, returncode=0, stdout="", stderr="", value=commits
        )

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.pr_commits(11)

    assert result == commits
    assert calls == [
        (["api", "repos/{owner}/{repo}/pulls/11/commits?per_page=100"], True, True),
    ]


def test_pr_ready_delegate_forwards_through_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Calling ``gh.pr_ready(number)`` through the delegate must reach the
    patched class-level ``GitHub.run`` with the same argv the moved body
    produces, and must recognize a real ``GitHubRunResult`` returned across
    the collaborator boundary via ``isinstance``.

    The expected argv is transcribed by reading the moved
    ``PullRequests.pr_ready`` body directly.
    """
    calls: list[tuple[list[str], bool]] = []

    def fake_run(
        self: GitHub, args: list[str], *, allow_failure: bool = False
    ) -> _github_module.GitHubRunResult:
        calls.append((args, allow_failure))
        return _github_module.GitHubRunResult(
            ok=True, returncode=0, stdout="", stderr="", value=None
        )

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh = GitHub(tmp_path)
    result = gh.pr_ready(13)

    assert isinstance(result, _github_module.GitHubRunResult)
    assert result.ok is True
    assert calls == [
        (["pr", "ready", "13"], True),
    ]


def test_pr_ready_delegate_returns_synthetic_result_on_dry_run(tmp_path: Path) -> None:
    """``pr_ready`` returns a synthetic ``ok=True`` result under dry-run
    without calling ``run`` at all -- transcribed directly from the moved
    body's ``if self.dry_run and _is_mutating(args): return GitHubRunResult(...)``
    guard. ``pr ready`` is a mutating subcommand, so ``_is_mutating`` (moved to
    ``_base.py`` in this same leaf) must correctly classify it as such for
    this guard to fire.
    """
    gh = GitHub(tmp_path, dry_run=True)
    result = gh.pr_ready(13)

    assert isinstance(result, _github_module.GitHubRunResult)
    assert result.ok is True
    assert result.value is None
