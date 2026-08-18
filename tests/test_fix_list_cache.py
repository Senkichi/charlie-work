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
from charlie_work.github import GitHub
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub


def _counting_run(counter: dict[str, int]):
    """A GitHub.run replacement that counts calls per gh subcommand."""

    def run(self, args, *, json_output=False, allow_failure=False):
        counter[args[0]] = counter.get(args[0], 0) + 1
        return [] if json_output else ""

    return run


def test_issue_list_caches_within_pass_and_refetches_after_invalidate(
    monkeypatch, tmp_path: Path
) -> None:
    counter: dict[str, int] = {}
    monkeypatch.setattr(GitHub, "run", _counting_run(counter))
    gh = GitHub(repo_root=tmp_path)

    gh.issue_list("automated-ready")
    gh.issue_list("automated-ready")
    assert counter["issue"] == 1, "second call within a pass must hit the cache"

    gh.invalidate_list_cache()
    gh.issue_list("automated-ready")
    assert counter["issue"] == 2, "post-invalidation call must refetch"


def test_pr_and_merged_pr_lists_refetch_after_invalidate(monkeypatch, tmp_path: Path) -> None:
    counter: dict[str, int] = {}
    monkeypatch.setattr(GitHub, "run", _counting_run(counter))
    gh = GitHub(repo_root=tmp_path)

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
