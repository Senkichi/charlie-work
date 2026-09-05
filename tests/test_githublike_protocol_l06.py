"""Track 2 L06 (issue #1590) + L06b (issue #1613): PullRequests capability
move regression tests.

New file, not appended to ``test_githublike_protocol.py`` (deliberate test-
layout change for this leaf) -- shares the AST/signature helpers via
``tests/_githublike_protocol_helpers.py`` (issue #1284 sanctions bare-name
``tests/_*.py`` modules for this kind of cross-file sharing).

L06 moved seven of Cluster E's eight members: ``pr_create``, ``pr_list``,
``merged_pr_list``, ``pr_view``, ``pr_diff``, ``pr_commits``, ``pr_ready``.
``merged_prs_for_issue`` deliberately did NOT move in that leaf -- its
verbatim body called ``linked_issue_number(...)`` as a bare global, and that
utility (plus its closing-keyword dependency chain) still lived in
``github.py`` with dozens of external call sites, so relocating it would have
been a second, much larger Mikado leaf of its own.

L06b (issue #1613; design doc Section 5, L06b) is that follow-up leaf:
``linked_issue_number`` and its dependency chain moved to a neutral
``issue_linking.py`` module with no ``charlie_work.github``/``gh`` coupling,
and ``merged_prs_for_issue`` moves here alongside the other seven Cluster E
members, importing ``linked_issue_number`` from ``issue_linking`` instead.
``PULL_REQUESTS_MOVED_MEMBERS`` below now covers all eight.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import charlie_work.github as _github_module
from charlie_work.github import GitHub, _ROUTES
from charlie_work.github_capabilities import PullRequestsLike
from charlie_work.github_capabilities.pull_requests import PullRequests

from _githublike_protocol_helpers import _lexical_github_defs

PULL_REQUESTS_MOVED_MEMBERS = (
    "pr_create",
    "pr_list",
    "merged_pr_list",
    "pr_view",
    "pr_diff",
    "pr_commits",
    "pr_ready",
    "merged_prs_for_issue",
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
    """``GitHub`` must no longer lexically define any of the eight moved
    PullRequests members.

    Track 2 L06 (issue #1590) moved ``pr_create``, ``pr_list``,
    ``merged_pr_list``, ``pr_view``, ``pr_diff``, ``pr_commits``, and
    ``pr_ready`` to ``PullRequests``; L06b (issue #1613) moved the eighth,
    ``merged_prs_for_issue``, once ``linked_issue_number`` had its own
    neutral home. All eight names are now served on ``GitHub`` by generated
    delegates (class-level assignments, not ``def``s) carrying
    ``__wrapped__`` pointers back to the exact ``PullRequests`` method object.
    """
    lexical = _lexical_github_defs()
    for name in PULL_REQUESTS_MOVED_MEMBERS:
        assert name not in lexical, f"{name} is still a lexical GitHub def"
    # And confirm the names *are* still resolvable, as delegates, and that
    # each delegate's __wrapped__ points at the exact PullRequests method
    # object (not merely "something with a __wrapped__ attribute").
    for name in PULL_REQUESTS_MOVED_MEMBERS:
        delegate = getattr(GitHub, name)
        assert hasattr(delegate, "__wrapped__"), (
            f"GitHub.{name} does not look like an installed delegate"
        )
        assert delegate.__wrapped__ is vars(PullRequests)[name], (
            f"GitHub.{name}.__wrapped__ does not point at PullRequests.{name}"
        )


def test_pullrequests_routes_point_at_the_pull_requests_collaborator() -> None:
    """``_ROUTES`` must route all eight moved names to the ``_pull_requests`` collaborator.

    Forward-compatible: asserts only the eight entries these two leaves (L06
    + L06b) add, not the full ``_ROUTES`` contents, so it keeps holding
    unmodified once later leaves populate more of the table.
    """
    for name in PULL_REQUESTS_MOVED_MEMBERS:
        assert _ROUTES[name] == "_pull_requests"


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


def test_pr_create_delegate_returns_zero_on_dry_run(tmp_path: Path) -> None:
    """``pr_create`` returns ``0`` under dry-run without calling ``run`` at all --
    transcribed directly from the moved body's ``if self.dry_run: return 0`` guard.

    Renamed from ``test_pr_create_delegate_returns_none_on_dry_run`` (Track 2,
    issue #1613, L06b) -- the old name said "none" but the assertion below has
    always checked ``== 0``, matching the body's actual ``return 0``.
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


# ---------------------------------------------------------------------------
# L06b (issue #1613): merged_prs_for_issue moves; linked_issue_number and its
# closing-keyword chain relocate to a neutral issue_linking.py module.
# ---------------------------------------------------------------------------


def test_l06b_reexports_have_cold_import_identity() -> None:
    """Every name relocated in this leaf must resolve to the SAME object
    whether reached through ``charlie_work.github`` or through its new home.

    27 external call sites (``workflow.py``, ``reconcile.py``, ``janitor.py``,
    ``cli.py``, ``dead_worker_reap.py``, ``backlog_reachability.py``,
    ``worktree.py``, plus tests) still do ``from .github import
    linked_issue_number``/``from charlie_work.github import
    linked_issue_number`` -- not repointed in this leaf (deferred follow-up)
    -- so this identity, not merely behavioral equivalence, is load-bearing.
    """
    import charlie_work.issue_linking as _issue_linking_module
    from charlie_work.github_capabilities import pull_requests as _pull_requests_module

    assert _github_module.linked_issue_number is _issue_linking_module.linked_issue_number
    assert (
        _github_module.iter_unnegated_closing_keyword_matches
        is _issue_linking_module.iter_unnegated_closing_keyword_matches
    )
    assert _github_module._CLOSING_KEYWORDS_ALT is _issue_linking_module._CLOSING_KEYWORDS_ALT
    assert _github_module._CLOSING_KEYWORD_REF is _issue_linking_module._CLOSING_KEYWORD_REF
    assert _github_module.MergedPRSearchResult is _pull_requests_module.MergedPRSearchResult
    assert _github_module._MergedPRSearchResult is _pull_requests_module.MergedPRSearchResult
    assert _github_module.MERGED_PR_LIST_FIELDS is _pull_requests_module.MERGED_PR_LIST_FIELDS


def test_merged_prs_for_issue_behavior_through_fake_owner() -> None:
    """``merged_prs_for_issue``, called directly on a bare ``PullRequests``
    collaborator constructed with a minimal fake owner (not a real ``GitHub``
    instance), proves the moved body's only sibling call (``self.run(...)``)
    forwards correctly through ``CapabilityCollaborator.__getattr__`` -- no
    subclass-override bypass hazard applies here (``run`` is never itself a
    routed/collaborator-side member, unlike L07's ``are_issues_open`` calling
    ``self.issue_view``) -- and that ``linked_issue_number`` (imported from
    ``issue_linking.py``, not ``charlie_work.github``) correctly filters
    search results down to PRs actually bound to the requested issue via
    both the branch-name and closing-keyword paths.
    """
    calls: list[list[str]] = []

    class _FakeOwner:
        def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
            calls.append(args)
            return [
                {
                    # No agent/issue-N branch match -> falls through to the
                    # closing-keyword path, which binds via the title.
                    "number": 10,
                    "title": "Fixes #42",
                    "body": "",
                    "headRefName": "worker-branch-no-issue-pattern",
                    "isCrossRepository": False,
                    "state": "MERGED",
                    "headRefOid": "abc123",
                },
                {
                    # Branch name binds this one to issue 7, not 42 -- must
                    # be excluded even though nothing in title/body mentions
                    # 42 or 7.
                    "number": 11,
                    "title": "unrelated change",
                    "body": "",
                    "headRefName": "agent/issue-7-something",
                    "isCrossRepository": False,
                    "state": "MERGED",
                    "headRefOid": "def456",
                },
            ]

    pull_requests = PullRequests(_FakeOwner())
    result = pull_requests.merged_prs_for_issue(42, branch_prefix="agent/issue")

    assert isinstance(result, _github_module.MergedPRSearchResult)
    assert result.ok is True
    assert [pr["number"] for pr in result] == [10]
    assert calls == [
        [
            "pr",
            "list",
            "--state",
            "merged",
            "--search",
            '"#42"',
            "--limit",
            "20",
            "--json",
            _github_module.MERGED_PR_LIST_FIELDS,
        ]
    ]


def test_merged_prs_for_issue_returns_not_ok_on_search_failure_through_fake_owner() -> None:
    """A failed search (``GitHubRunResult`` with ``ok=False``) must produce an
    empty, ``ok=False`` ``MergedPRSearchResult`` -- transcribed directly from
    the moved body's ``if not result.ok: return MergedPRSearchResult([], ok=False)``
    guard -- exercised through the same bare-collaborator construction as
    above rather than a full ``GitHub`` instance.
    """

    class _FakeOwner:
        def run(self, args: list[str], *, json_output: bool = False, allow_failure: bool = False):
            return _github_module.GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr="rate limited",
                value=None,
                error="rate limited",
            )

    pull_requests = PullRequests(_FakeOwner())
    result = pull_requests.merged_prs_for_issue(42, branch_prefix="agent/issue")

    assert isinstance(result, _github_module.MergedPRSearchResult)
    assert result.ok is False
    assert list(result) == []


def test_issue_linking_imports_without_charlie_work_github() -> None:
    """``issue_linking.py`` must have zero import-time coupling to
    ``charlie_work.github``/``charlie_work.github_capabilities``.

    This is the whole point of relocating ``linked_issue_number`` here (Track
    2, issue #1613): so ``pull_requests.py`` can import it without cycling
    back through ``github.py``, which imports ``github_capabilities`` at
    module load time. Run in a fresh subprocess so the check is not
    contaminated by whatever this test session's own import order already
    put into ``sys.modules``.
    """
    script = (
        "import sys\n"
        "import charlie_work.issue_linking\n"
        "assert 'charlie_work.github' not in sys.modules, sorted(sys.modules)\n"
        "assert 'charlie_work.github_capabilities' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_github_direct_def_count_is_two() -> None:
    """L06b drops ``GitHub`` from 3 direct defs (L09's final count) to
    exactly 2 (``__post_init__``, ``run``), by moving ``merged_prs_for_issue``
    off ``GitHub`` too, once ``linked_issue_number`` had its own neutral home.

    Uses a direct, unfiltered AST walk (matching the attachment-contracts
    ``member_count`` ratchet's own counting rule: every direct
    ``FunctionDef``/``AsyncFunctionDef`` child of ``ClassDef.body``, dunders
    included) rather than the shared ``_lexical_github_defs()`` helper, which
    deliberately excludes dunders for its own (protocol-conformance) purpose.
    """
    import ast

    source = Path(_github_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    (github_cls,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "GitHub"
    ]
    all_defs = {
        child.name
        for child in github_cls.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert all_defs == {"__post_init__", "run"}, all_defs
