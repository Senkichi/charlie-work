"""Tests for the retry-on-fetch-failure fix to ``_queue_sync_merge_covered``.

Issue: the #1194 queue-sync-merge coverage check (``OrchestratorApp.
_queue_sync_merge_covered``) makes three ``gh`` API calls (two ``commit()``
lookups, one ``compare()``) and fails the whole four-condition check closed
on ANY failure of any one of them -- including a purely transient failure
(rate limit, TLS blip, timeout) that says nothing about whether the shape is
actually covered. The caller (``_detect_unauthorized_merges``) cannot tell a
transient fetch failure apart from a genuinely uncovered shape, so a single
flaky API call on one leg misreports as a real ``unauthorized_merge_detected``
finding. Evidence: job-cannon PRs #1888, #1916, #1904, #1895 each logged
326-376 ``unauthorized_merge_queue_sync_covered`` events against exactly one
``unauthorized_merge_detected`` -- overwhelmingly a covered shape, with one
pass where some leg's ``gh api`` call failed transiently. Which leg failed on
those specific PRs is not recoverable from that evidence and is not claimed
here.

The fix retries only FETCH failures (``commit()`` not ok, or ``compare()``
returning ``None``) up to ``_QUEUE_SYNC_RETRY_ATTEMPTS`` times with short
backoff, and never retries a DETERMINED non-covered shape (wrong parent
count, identity mismatch, comparison status "ahead"/"diverged", ...) since
retrying a query that already answered would not change the answer. If
retries exhaust, the result still fails closed -- the caller still emits
``unauthorized_merge_detected`` -- but the event payload is marked
``coverage_check: "indeterminate"`` (with ``coverage_check_error`` naming the
leg and error) instead of ``coverage_check: "not_covered"`` (with
``coverage_reason``), so an operator triaging the finding can tell "the API
never answered" apart from "the shape was checked and rejected".

New file rather than additions to ``test_charlie_work.py``: a sibling PR
touches that file's queue-sync tests directly, and this file's fixtures are
self-contained (not imported from it) because
``tests/test_zero_cross_test_import_guard.py`` forbids one ``test_*.py``
module importing from another -- shared fixtures must live in a
``tests/_*.py`` module, and this file's fixture-builder is specific enough
(injectable per-key fetch-failure counts) that it does not belong in the
general-purpose ``_merge_tripwire_fixtures.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire
from charlie_work import github as github_module
from charlie_work.config import AutoMergeConfig
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp, OrchestratorConfig


class FlakyFakeGitHub(FakeGitHub):
    """``FakeGitHub`` subclass that can fail a configured number of times
    per lookup key before answering for real.

    ``commit_fail_counts`` / ``compare_fail_counts`` map a lookup key (a
    commit SHA, or a ``(base, head)`` tuple) to how many leading calls for
    that key should report a transient failure; the call after the budget is
    exhausted falls through to the real ``FakeGitHub`` behavior.
    ``commit_calls`` / ``compare_calls`` count every invocation per key
    (including failed ones), so tests can assert retry counts precisely --
    in particular that a determined non-covered shape is called exactly
    once, never retried.
    """

    def __init__(self) -> None:
        super().__init__()
        self.commit_fail_counts: dict[str, int] = {}
        self.compare_fail_counts: dict[tuple[str, str], int] = {}
        self.commit_calls: dict[str, int] = {}
        self.compare_calls: dict[tuple[str, str], int] = {}

    def commit(self, sha: str) -> github_module.GitHubRunResult:
        self.commit_calls[sha] = self.commit_calls.get(sha, 0) + 1
        remaining = self.commit_fail_counts.get(sha, 0)
        if remaining > 0:
            self.commit_fail_counts[sha] = remaining - 1
            return github_module.GitHubRunResult(
                ok=False,
                returncode=1,
                stdout="",
                stderr="simulated transient failure",
                value=None,
                error=f"simulated transient commit fetch failure for {sha}",
            )
        return super().commit(sha)

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        key = (base, head)
        self.compare_calls[key] = self.compare_calls.get(key, 0) + 1
        remaining = self.compare_fail_counts.get(key, 0)
        if remaining > 0:
            self.compare_fail_counts[key] = remaining - 1
            return None
        return super().compare(base, head)


def _arm_queue_sync_fixture(
    fake_gh: FlakyFakeGitHub,
    paths,
    *,
    pr_number: int = 701,
    issue_number: int = 701,
    reviewed_head_sha: str = "sha-approved",
    live_head_sha: str = "sha-syncmerge",
    merge_commit_sha: str = "sha-landing",
    pre_merge_base: str = "sha-premerge-base",
    other_parent: str = "sha-main-tip",
    live_head_parents: list[str] | None = None,
    author_login: str = "aviator-app[bot]",
    committer_login: str = "web-flow",
    committer_name: str = "GitHub",
    compare_status: str = "behind",
) -> None:
    """Wire up a merged worker PR shaped like a covered Aviator queue
    sync-merge (#1194) -- the happy-path shape every test here starts from,
    flipping only the one signal it is pinning (an injected fetch-failure
    count, or a mangled shape for the no-retry case).

    Deliberately parallel to ``test_charlie_work.py``'s
    ``_arm_queue_sync_fixture`` (same default SHAs and PR shape) but not
    imported from it -- see the module docstring for why.
    """
    fake_gh.prs = [
        {
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
        },
    ]

    pr_dir = paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": reviewed_head_sha}),
        encoding="utf-8",
    )

    actual_parents = (
        live_head_parents if live_head_parents is not None else [reviewed_head_sha, other_parent]
    )
    fake_gh.commits[live_head_sha] = {
        "parents": [{"sha": p} for p in actual_parents],
        "author": {"login": author_login},
        "committer": {"login": committer_login},
        "commit": {"committer": {"name": committer_name}},
    }
    fake_gh.commits[merge_commit_sha] = {"parents": [{"sha": pre_merge_base}]}
    fake_gh.compare_overrides[(pre_merge_base, other_parent)] = {"status": compare_status}


def _queue_sync_app(tmp_path: Path) -> tuple[OrchestratorApp, Any, FlakyFakeGitHub]:
    config = OrchestratorConfig(auto_merge=AutoMergeConfig(queue_bot_login="aviator-app[bot]"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    fake_gh = FlakyFakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, paths, fake_gh


def test_queue_sync_merge_covered_retries_transient_live_head_fetch_failure(
    tmp_path: Path,
) -> None:
    """Regression test for the misreport this PR fixes: a single transient
    ``gh.commit()`` failure on the live-head leg must not sink an otherwise
    covered queue sync-merge. Retrying just that leg recovers ``covered``
    instead of the caller emitting a false ``unauthorized_merge_detected``.
    """
    app, paths, fake_gh = _queue_sync_app(tmp_path)
    _arm_queue_sync_fixture(fake_gh, paths)
    fake_gh.commit_fail_counts["sha-syncmerge"] = 1  # fails once, succeeds on retry

    detected = app._detect_unauthorized_merges()

    assert detected == []
    covered_events = query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")
    assert len(covered_events) == 1
    assert covered_events[0]["pr_number"] == 701
    assert fake_gh.commit_calls["sha-syncmerge"] == 2
    assert query_events(paths.state_file, kind="unauthorized_merge_detected") == []


def test_queue_sync_merge_covered_exhausted_retries_indeterminate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All ``_QUEUE_SYNC_RETRY_ATTEMPTS`` attempts on the live-head fetch leg
    fail: the shape could never be evaluated, so the check still fails closed
    (a finding still fires -- ambiguity cannot suppress a security control),
    but the ``unauthorized_merge_detected`` payload is marked
    ``coverage_check: "indeterminate"`` (with a non-empty
    ``coverage_check_error``) rather than ``"not_covered"``, so triage can
    tell "the API never answered" apart from "the shape was rejected".

    Also proves the retry backoff is fully injectable: with
    ``_QUEUE_SYNC_RETRY_SLEEP`` patched, this test -- which exhausts a full
    retry budget -- completes in well under a second instead of the ~3s
    (1s + 2s) of real backoff it would otherwise cost, and the recorded sleep
    args (``[1, 2]``) double as a check on the backoff schedule itself.
    """
    # _QUEUE_SYNC_RETRY_SLEEP lives in queue_sync_coverage.py (issue #1442
    # extraction) -- workflow.py only re-exports the name via its facade
    # import block, and the retry loop that reads it resolves the bare name
    # against its OWN defining module's globals, not the re-exporting
    # facade. Patching charlie_work.workflow here would silently no-op.
    from charlie_work import queue_sync_coverage as queue_sync_coverage_module

    sleeps: list[float] = []
    monkeypatch.setattr(
        queue_sync_coverage_module, "_QUEUE_SYNC_RETRY_SLEEP", sleeps.append, raising=False
    )

    app, paths, fake_gh = _queue_sync_app(tmp_path)
    _arm_queue_sync_fixture(fake_gh, paths)
    fake_gh.commit_fail_counts["sha-syncmerge"] = 3  # exhausts the retry budget

    started = time.perf_counter()
    detected = app._detect_unauthorized_merges()
    elapsed = time.perf_counter() - started

    assert sleeps == [1, 2]
    assert elapsed < 0.5, f"retry backoff appears to have slept for real: {elapsed:.2f}s elapsed"
    assert fake_gh.commit_calls["sha-syncmerge"] == 3

    assert len(detected) == 1
    assert detected[0]["pr"] == 701

    detected_events = query_events(paths.state_file, kind="unauthorized_merge_detected")
    assert len(detected_events) == 1
    payload = detected_events[0]["payload"]
    assert payload["coverage_check"] == "indeterminate"
    assert payload["coverage_check_error"]
    assert payload.get("coverage_reason") is None


def test_queue_sync_merge_covered_determined_shape_no_retry(tmp_path: Path) -> None:
    """A structurally-determined non-covered shape (3 parents on the live
    head, instead of the required 2) must not retry -- retrying a query that
    already answered would not change the answer, and would waste seconds of
    backoff on every merged PR with a genuine unauthorized head. Exactly one
    ``gh.commit()`` call for the live-head leg, and the landing-commit /
    compare() legs are never reached at all. The
    ``unauthorized_merge_detected`` payload is marked
    ``coverage_check: "not_covered"`` with a non-empty ``coverage_reason``.
    """
    app, paths, fake_gh = _queue_sync_app(tmp_path)
    _arm_queue_sync_fixture(
        fake_gh,
        paths,
        live_head_parents=["sha-approved", "sha-main-tip", "sha-extra-parent"],
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 701
    assert fake_gh.commit_calls == {"sha-syncmerge": 1}
    assert fake_gh.compare_calls == {}

    detected_events = query_events(paths.state_file, kind="unauthorized_merge_detected")
    assert len(detected_events) == 1
    payload = detected_events[0]["payload"]
    assert payload["coverage_check"] == "not_covered"
    assert payload["coverage_reason"]
    assert payload.get("coverage_check_error") is None


def test_queue_sync_merge_covered_retries_transient_compare_failure(tmp_path: Path) -> None:
    """The ``compare()`` leg returning ``None`` (a transient ``gh api``
    failure, distinct from a real "diverged"/"ahead" status) twice before
    succeeding is the same fetch-failure class as the ``commit()`` legs:
    retried, and ``covered`` once ``compare()`` finally returns the real
    comparison.
    """
    app, paths, fake_gh = _queue_sync_app(tmp_path)
    _arm_queue_sync_fixture(fake_gh, paths)
    fake_gh.compare_fail_counts[("sha-premerge-base", "sha-main-tip")] = 2

    detected = app._detect_unauthorized_merges()

    assert detected == []
    covered_events = query_events(paths.state_file, kind="unauthorized_merge_queue_sync_covered")
    assert len(covered_events) == 1
    assert fake_gh.compare_calls[("sha-premerge-base", "sha-main-tip")] == 3
