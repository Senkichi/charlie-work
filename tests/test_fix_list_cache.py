"""Regression tests for per-pass GitHub list-cache invalidation.

A long-running supervisor (``charlie fleet supervise``) reuses one
``OrchestratorApp`` -- and therefore one ``GitHub`` instance -- across many
loop passes. The list cache exists to dedupe expensive list calls *within*
one pass; before this fix nothing ever cleared it, so a daemon's very first
pass froze the issue/PR list snapshot for the entire process lifetime:
issues filed or PRs opened after startup stayed invisible until the daemon
restarted (observed live 2026-07-24: intake frozen at a stale 10-issue set
while two freshly filed ``automated-ready`` issues sat unseen for an hour).

The invariant, enforced at the pass boundary (``_loop_body``): every
orchestrator pass begins with ``gh.invalidate_list_cache()`` and therefore
observes a fresh GitHub snapshot, no matter how many passes share one
process.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.github_transport import GraphQLRequest, Request, Response
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

from _fake_transport import FakeAdapter, connection_page, make_github, ok
from _fakes_github import FakeGitHub


def _counting_github(tmp_path: Path, counter: dict[str, int]):
    """A real ``GitHub`` over a fake transport that counts requests per kind.

    GraphQL list reads are keyed by their connection (``issue`` / ``pr``); REST
    reads are keyed ``api``. Every reply is an empty page / empty list.
    """

    def handler(request: Request) -> Response:
        if isinstance(request, GraphQLRequest):
            kind = "pr" if "pullRequests" in request.document else "issue"
            counter[kind] = counter.get(kind, 0) + 1
            return connection_page("issues" if kind == "issue" else "pullRequests", [])
        counter["api"] = counter.get("api", 0) + 1
        return ok([])

    gh, _http, _gh_adapter = make_github(tmp_path, http=FakeAdapter("http", handler=handler))
    return gh


def test_issue_list_caches_within_pass_and_refetches_after_invalidate(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    gh = _counting_github(tmp_path, counter)

    # Issue #2443: open issues are one REST read (counted under "api").
    gh.issue_list("automated-ready")
    gh.issue_list("automated-ready")
    gh.issue_list("some-other-label")
    assert counter["api"] == 1, "calls within a pass must share one cached read"
    assert counter.get("issue", 0) == 0, "no GraphQL issue list"

    gh.invalidate_list_cache()
    gh.issue_list("automated-ready")
    assert counter["api"] == 2, "post-invalidation call must refetch"


def test_pr_and_merged_pr_lists_refetch_after_invalidate(tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    gh = _counting_github(tmp_path, counter)

    gh.pr_list()
    gh.pr_list()
    assert counter["pr"] == 1

    gh.merged_pr_list()
    gh.merged_pr_list()
    api_calls_after_first = counter["api"]

    gh.invalidate_list_cache()
    gh.pr_list()
    gh.merged_pr_list()
    assert counter["pr"] == 2
    assert counter["api"] > api_calls_after_first


def test_loop_invalidates_list_cache_every_pass(tmp_path: Path) -> None:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    app.loop(limit=0)
    app.loop(limit=0)

    assert gh.list_cache_invalidations == 2


class CachingFakeGitHub(FakeGitHub):
    """A fake reproducing the real GitHub's cache semantics for issue_list:
    results freeze until invalidate_list_cache() is called. Lets the
    daemon-staleness regression run at the workflow level without real gh.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fake_list_cache: dict = {}

    def invalidate_list_cache(self) -> None:
        super().invalidate_list_cache()
        self._fake_list_cache.clear()

    def issue_list(self, labels=None, state=None):
        if isinstance(labels, str):
            label_key = (labels,)
        else:
            label_key = tuple(labels or ())
        key = ("issue_list", state or "open", label_key)
        if key not in self._fake_list_cache:
            self._fake_list_cache[key] = super().issue_list(labels=labels, state=state)
        return self._fake_list_cache[key]


def test_issue_filed_between_passes_is_intaken_by_next_pass(tmp_path: Path) -> None:
    """The live daemon-staleness shape: one app, many passes, an issue filed
    after pass N must be visible to pass N+1 without a process restart."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = CachingFakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    app.loop(limit=0)
    state = load_state(paths.state_file)
    assert "123" in state["issues"]
    assert "999" not in state["issues"]

    gh.issues.append(
        {
            "number": 999,
            "title": "Filed while the daemon was already running",
            "url": "https://example.test/issues/999",
            "body": "Fresh work",
            "labels": [{"name": "automated-ready"}],
            "state": "OPEN",
        }
    )

    app.loop(limit=0)
    state = load_state(paths.state_file)
    assert "999" in state["issues"], (
        "an issue filed between passes must be intaken by the next pass; "
        "a frozen list cache reproduces the 2026-07-24 daemon-staleness outage"
    )
