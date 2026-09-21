"""Tests for durable queue-sync coverage memoization (issue #1473).

``_detect_unauthorized_merges`` used to re-run the 3-``gh``-API-call
``_queue_sync_merge_covered`` predicate (two ``commit()`` lookups, one
``compare()``) for every merged, approval-mismatched PR on *every* loop
pass, forever, with no memoization of an already-determined
``covered=True`` verdict -- measured at 5,723-12,208
``unauthorized_merge_queue_sync_covered`` events/24h in a fleet repo, all
re-verifying the same handful of already-known-covered merges.
``queue_sync_coverage_cache.py`` durably memoizes only that ``covered=True``
outcome per ``(pr_number, reviewed_head_sha, live_head_sha)``, wired into
``OrchestratorApp._queue_sync_merge_covered``
(``orchestration/instrumentation_ops.py``).

SAFETY is the point of this suite, not just the happy path: a not-covered or
indeterminate verdict must never be memoized (an uncovered PR is
re-checked every pass, unchanged from before this fix), a brand-new merged
PR must always be independently verified regardless of what else is
cached, and a corrupt cache file must degrade to "re-verify everything"
rather than crash or silently assume coverage.

New file rather than additions to ``test_queue_sync_merge_retry.py`` or
``test_charlie_work_unauthorized_merge.py``: this module's fake-``gh``
call-counting helper is specific to memoization assertions (it counts calls
across *two* passes of the same app, which the existing retry-focused fakes
do not need to do) and ``tests/test_zero_cross_test_import_guard.py``
forbids one ``test_*.py`` module importing fixtures from another --
mirrors ``test_queue_sync_merge_retry.py``'s own stated rationale for
defining its own local ``FlakyFakeGitHub`` instead of sharing one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire
from charlie_work import layout, queue_sync_coverage_cache
from charlie_work.config import AutoMergeConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp, OrchestratorConfig


class CountingFakeGitHub(FakeGitHub):
    """``FakeGitHub`` subclass that counts ``commit()``/``compare()`` calls.

    Plain pass-through counters (no injected failures, unlike
    ``test_queue_sync_merge_retry.py``'s ``FlakyFakeGitHub``): this suite
    asserts on *how many times* each leg is called across successive
    passes, not on retry behavior.
    """

    def __init__(self) -> None:
        super().__init__()
        self.commit_calls: dict[str, int] = {}
        self.compare_calls: dict[tuple[str, str], int] = {}

    def commit(self, sha: str):
        self.commit_calls[sha] = self.commit_calls.get(sha, 0) + 1
        return super().commit(sha)

    def compare(self, base: str, head: str):
        key = (base, head)
        self.compare_calls[key] = self.compare_calls.get(key, 0) + 1
        return super().compare(base, head)


def _app(tmp_path: Path) -> tuple[OrchestratorApp, Any, CountingFakeGitHub]:
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    fake_gh = CountingFakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, paths, fake_gh


def _covered_pr(
    fake_gh: CountingFakeGitHub,
    paths: Any,
    *,
    pr_number: int = 701,
    issue_number: int = 701,
    reviewed_head_sha: str = "sha-approved",
    live_head_sha: str = "sha-syncmerge",
    merge_commit_sha: str = "sha-landing",
    pre_merge_base: str = "sha-premerge-base",
    other_parent: str = "sha-main-tip",
) -> dict[str, Any]:
    """Wire up ``fake_gh``/on-disk state for a covered Aviator queue
    sync-merge (issue #1194) and return its ``merged_pr_list()``-shaped
    dict. The caller assigns ``fake_gh.prs`` (a list, so multiple PRs can
    coexist in one fixture without one call clobbering another's).
    """
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": reviewed_head_sha}),
        encoding="utf-8",
    )
    fake_gh.commits[live_head_sha] = {
        "parents": [{"sha": reviewed_head_sha}, {"sha": other_parent}],
        "author": {"login": "aviator-app[bot]"},
        "committer": {"login": "web-flow"},
        "commit": {"committer": {"name": "GitHub"}},
    }
    fake_gh.commits[merge_commit_sha] = {"parents": [{"sha": pre_merge_base}]}
    fake_gh.compare_overrides[(pre_merge_base, other_parent)] = {"status": "behind"}
    return {
        "number": pr_number,
        "title": f"fix: queue sync merge #{pr_number}",
        "url": f"https://example.test/pull/{pr_number}",
        "headRefName": f"agent/issue-{issue_number}-fix",
        "baseRefName": "main",
        "headRefOid": live_head_sha,
        "mergeCommitOid": merge_commit_sha,
        "state": "MERGED",
        "isCrossRepository": False,
        "body": f"Closes #{issue_number}",
        "labels": [],
    }


def _arm_covered_queue_sync_fixture(
    fake_gh: CountingFakeGitHub, paths: Any, **kwargs: Any
) -> None:
    """Single-PR convenience wrapper over :func:`_covered_pr`."""
    fake_gh.prs = [_covered_pr(fake_gh, paths, **kwargs)]


def _arm_uncovered_queue_sync_fixture(
    fake_gh: CountingFakeGitHub,
    paths: Any,
    *,
    pr_number: int = 801,
    issue_number: int = 801,
    reviewed_head_sha: str = "sha-approved-uncovered",
    live_head_sha: str = "sha-badmerge",
) -> None:
    """An approved-but-mismatched PR whose live head is structurally NOT a
    queue sync-merge (3 parents instead of the required 2) -- a determined
    ``not_covered`` verdict that must never be memoized.
    """
    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": reviewed_head_sha}),
        encoding="utf-8",
    )
    fake_gh.commits[live_head_sha] = {
        "parents": [{"sha": reviewed_head_sha}, {"sha": "sha-x"}, {"sha": "sha-y"}],
    }
    fake_gh.prs = [
        {
            "number": pr_number,
            "title": f"fix: bad merge #{pr_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": live_head_sha,
            "mergeCommitOid": "sha-unused-landing",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": f"Closes #{issue_number}",
            "labels": [],
        },
    ]


# ---------------------------------------------------------------------------
# Integration: OrchestratorApp._detect_unauthorized_merges
# ---------------------------------------------------------------------------


def test_second_pass_makes_zero_verification_calls_for_covered_pr(tmp_path: Path) -> None:
    """The behavioral fix: a second pass over the same covered merge makes
    NO new ``commit()``/``compare()`` calls and emits NO second
    ``unauthorized_merge_queue_sync_covered`` event. Against the pre-fix
    code this assertion fails -- the counts double on the second pass.
    """
    app, paths, fake_gh = _app(tmp_path)
    _arm_covered_queue_sync_fixture(fake_gh, paths)

    first = app._detect_unauthorized_merges()
    assert first == []
    assert fake_gh.commit_calls == {"sha-syncmerge": 1, "sha-landing": 1}
    assert fake_gh.compare_calls == {("sha-premerge-base", "sha-main-tip"): 1}
    assert len(query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")) == 1

    second = app._detect_unauthorized_merges()
    assert second == []
    # No growth at all: the cache hit short-circuits before any gh call.
    assert fake_gh.commit_calls == {"sha-syncmerge": 1, "sha-landing": 1}
    assert fake_gh.compare_calls == {("sha-premerge-base", "sha-main-tip"): 1}
    assert len(query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")) == 1


def test_uncovered_pr_rechecked_every_pass(tmp_path: Path) -> None:
    """A determined not-covered verdict is never memoized: the live-head
    commit is re-fetched on every pass, unchanged from pre-fix behavior.
    """
    app, paths, fake_gh = _app(tmp_path)
    _arm_uncovered_queue_sync_fixture(fake_gh, paths)

    first = app._detect_unauthorized_merges()
    assert len(first) == 1
    assert first[0]["pr"] == 801
    assert fake_gh.commit_calls == {"sha-badmerge": 1}

    second = app._detect_unauthorized_merges()
    assert len(second) == 1
    assert fake_gh.commit_calls == {"sha-badmerge": 2}

    assert query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered") == []


def test_new_merged_pr_is_independently_verified(tmp_path: Path) -> None:
    """A cache already populated for one PR must not suppress -- or
    shortcut -- verification of a different, newly-merged PR.
    """
    app, paths, fake_gh = _app(tmp_path)
    pr_701 = _covered_pr(fake_gh, paths, pr_number=701, issue_number=701)
    fake_gh.prs = [pr_701]

    first = app._detect_unauthorized_merges()
    assert first == []
    assert fake_gh.commit_calls == {"sha-syncmerge": 1, "sha-landing": 1}

    pr_702 = _covered_pr(
        fake_gh,
        paths,
        pr_number=702,
        issue_number=702,
        reviewed_head_sha="sha-approved-2",
        live_head_sha="sha-syncmerge-2",
        merge_commit_sha="sha-landing-2",
        pre_merge_base="sha-premerge-base-2",
        other_parent="sha-main-tip-2",
    )
    fake_gh.prs = [pr_701, pr_702]

    second = app._detect_unauthorized_merges()
    assert second == []
    # #701 is a cache hit (no new calls); #702 is brand new and gets the
    # full 3-call verification exactly once.
    assert fake_gh.commit_calls == {
        "sha-syncmerge": 1,
        "sha-landing": 1,
        "sha-syncmerge-2": 1,
        "sha-landing-2": 1,
    }
    assert len(query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")) == 2


def test_corrupt_cache_file_degrades_to_reverify(tmp_path: Path) -> None:
    """A cache file corrupted on disk (e.g. a half-written or hand-edited
    file) must fail CLOSED: every previously-cached PR re-verifies from
    ``gh`` exactly as if nothing had ever been cached. It must never crash
    the tripwire and must never be mistaken for "still covered" without
    redoing the check.
    """
    app, paths, fake_gh = _app(tmp_path)
    _arm_covered_queue_sync_fixture(fake_gh, paths)

    app._detect_unauthorized_merges()
    assert fake_gh.commit_calls == {"sha-syncmerge": 1, "sha-landing": 1}

    cache_path = layout.queue_sync_coverage_cache_path(paths.root)
    assert cache_path.exists()
    cache_path.write_text("{ not valid json", encoding="utf-8")

    detected = app._detect_unauthorized_merges()
    assert detected == []
    assert fake_gh.commit_calls == {"sha-syncmerge": 2, "sha-landing": 2}
    assert len(query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")) == 2

    # The cache heals itself: valid JSON again, with the entry restored.
    restored = json.loads(cache_path.read_text(encoding="utf-8"))
    assert isinstance(restored.get("covered"), dict)
    assert len(restored["covered"]) == 1


# ---------------------------------------------------------------------------
# Unit: queue_sync_coverage_cache module itself
# ---------------------------------------------------------------------------


def test_is_covered_cached_round_trips_through_record_covered(tmp_path: Path) -> None:
    path = tmp_path / "queue-sync-coverage-cache.json"
    assert not queue_sync_coverage_cache.is_covered_cached(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="b"
    )

    assert queue_sync_coverage_cache.record_covered(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="b"
    )

    assert queue_sync_coverage_cache.is_covered_cached(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="b"
    )
    # A different live-head SHA (e.g. a later push) is an unrelated key.
    assert not queue_sync_coverage_cache.is_covered_cached(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="c"
    )


def test_record_covered_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    assert queue_sync_coverage_cache.record_covered(
        path, pr_number=5, reviewed_head_sha="x", live_head_sha="y"
    )
    assert queue_sync_coverage_cache.record_covered(
        path, pr_number=5, reviewed_head_sha="x", live_head_sha="y"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["covered"]) == 1


def test_is_covered_cached_false_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "does-not-exist.json"
    assert not queue_sync_coverage_cache.is_covered_cached(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="b"
    )


def test_is_covered_cached_false_for_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("not json at all", encoding="utf-8")
    assert not queue_sync_coverage_cache.is_covered_cached(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="b"
    )


def test_is_covered_cached_false_for_wrong_shaped_file(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"covered": "not-a-dict"}), encoding="utf-8")
    assert not queue_sync_coverage_cache.is_covered_cached(
        path, pr_number=1, reviewed_head_sha="a", live_head_sha="b"
    )
