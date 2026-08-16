"""Regression tests for cw#1273: bounded outer retry for ``gh pr create``.

Covers ``pr_create_retry.create_pr_with_retry`` in isolation (AC1-AC4, using a
lightweight fake that satisfies the ``PrCreator`` protocol) and its
composition with ``GitHub.run()``'s existing inner pre-connection-only retry
(AC5, using the real ``GitHub`` class with a monkeypatched ``subprocess.run``,
mirroring ``tests/test_github.py``'s own harness -- a hand-rolled fake could
assert "outer didn't retry" without ever proving the two layers actually
compose against real code).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from charlie_work import github as github_module
from charlie_work.config import RuntimeConfig
from charlie_work.instrumentation import _LEVEL_BY_KIND
from charlie_work.pr_create_retry import (
    DEFAULT_BASE_SECONDS,
    DEFAULT_MAX_RETRIES,
    PrCreateRetryResult,
    create_pr_with_retry,
)


class _FakePrCreator:
    """Minimal double satisfying the ``PrCreator`` protocol.

    ``pr_create_responses`` is consumed one value per live call (a
    ``StopIteration``/index error would be a test bug, not a case to guard
    against -- these tests always supply exactly as many responses as the
    expected number of live attempts). ``existing_prs`` backs ``pr_list()``
    for the duplicate-PR guard.
    """

    def __init__(
        self,
        pr_create_responses: list[int | None],
        *,
        existing_prs: list[dict[str, Any]] | None = None,
    ) -> None:
        self._responses = list(pr_create_responses)
        self._existing_prs = existing_prs if existing_prs is not None else []
        self.create_calls: list[dict[str, Any]] = []
        self.pr_list_calls = 0
        self.invalidate_calls = 0

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        self.create_calls.append({"head": head, "base": base, "title": title, "body": body})
        return self._responses[len(self.create_calls) - 1]

    def pr_list(self) -> list[dict[str, Any]]:
        self.pr_list_calls += 1
        return self._existing_prs

    def invalidate_list_cache(self) -> None:
        self.invalidate_calls += 1


def _sleep_recorder() -> tuple[list[float], Any]:
    sleeps: list[float] = []

    def _sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return sleeps, _sleep


# ---------------------------------------------------------------------------
# AC1: fault injection -- gh fails twice (terminal-for-mutation shape,
# collapsed to ``None`` by ``GitHub.pr_create`` either way) then succeeds.
# ---------------------------------------------------------------------------


def test_fault_injection_two_failures_then_success() -> None:
    gh = _FakePrCreator([None, None, 5001])
    sleeps, sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh,
        head="agent/issue-1273",
        base="main",
        title="Salvaged work",
        body="Closes #1273",
        sleep_fn=sleep_fn,
    )

    assert result == PrCreateRetryResult(
        ok=True, pr_number=5001, adopted_existing=False, attempts=3, error=None
    )
    # Exactly one live `pr_create` call reached the mock per attempt -- three
    # attempts total (two failures + the success), never more.
    assert len(gh.create_calls) == 3
    # Backoff before retry 1 and retry 2: base_seconds * 3**(n-1) with the
    # defaults (10.0 base, 3x multiplier) -> 10, 30.
    assert sleeps == [DEFAULT_BASE_SECONDS, DEFAULT_BASE_SECONDS * 3]


def test_fault_injection_never_raises_and_returns_result_object() -> None:
    """AC8: the wrapper is a pure result-returning function, even on total
    exhaustion -- callers check `.ok`, they never need a try/except."""
    gh = _FakePrCreator([None, None, None, None])
    _sleeps, sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh, head="agent/issue-x", base="main", title="t", body="b", sleep_fn=sleep_fn
    )

    assert isinstance(result, PrCreateRetryResult)
    assert result.ok is False
    assert result.pr_number is None


# ---------------------------------------------------------------------------
# AC2: exhausting every attempt (default: 1 + 3 retries = 4 live calls).
# The terminal event itself (`pr_create_failed_branch_stranded`, emitted from
# workflow.py's orphan-reap sweep, never from this module -- see its
# docstring) is covered by
# `test_charlie_work.py::test_orphaned_worker_reported_push_pr_create_failed_emits_distinct_drift`
# (updated for cw#1273). AC4 (repeated stranded terminal within one
# `_drift_fingerprint` window emits once) is covered by
# `test_charlie_work.py::test_orphaned_worker_pr_create_failed_stranded_drift_dedups_on_repeat_sweep`,
# which runs the orphan-reap sweep twice -- this module has no fingerprint
# state of its own to dedup against (see its docstring), so that coverage
# belongs one layer up, not here. This test covers the wrapper's own
# exhaustion contract plus the registry-guard half of AC2 ("kind registered").
# ---------------------------------------------------------------------------


def test_exhaustion_after_max_retries_reports_error_and_attempt_count() -> None:
    gh = _FakePrCreator([None, None, None, None])
    sleeps, sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh,
        head="agent/issue-1273",
        base="main",
        title="t",
        body="b",
        max_retries=DEFAULT_MAX_RETRIES,
        sleep_fn=sleep_fn,
    )

    assert result.ok is False
    assert result.pr_number is None
    assert result.adopted_existing is False
    assert result.attempts == DEFAULT_MAX_RETRIES + 1 == 4
    assert result.error is not None
    assert len(gh.create_calls) == 4
    # Backoff before retries 1, 2, 3: 10, 30, 90.
    assert sleeps == [10.0, 30.0, 90.0]


def test_pr_create_failed_branch_stranded_registered_as_warning() -> None:
    """AC2's registry half: `test_event_kind_registry_exhaustive`
    (test_instrumentation.py) already enforces this against every literal
    emitted in src/ -- this pins the specific level so a future edit that
    silently downgrades/upgrades it fails here with a direct name, not just
    a generic registry-exhaustiveness failure."""
    assert "pr_create_failed_branch_stranded" in _LEVEL_BY_KIND
    assert _LEVEL_BY_KIND["pr_create_failed_branch_stranded"] == "warning"


# ---------------------------------------------------------------------------
# AC3: duplicate-PR guard. The first attempt's failure is ambiguous (may have
# landed server-side); before retrying, a PR is found for the head branch and
# adopted -- no second `pr_create` call is ever issued.
# ---------------------------------------------------------------------------


def test_duplicate_pr_guard_adopts_existing_pr_without_a_second_create() -> None:
    gh = _FakePrCreator(
        [None],
        existing_prs=[{"headRefName": "agent/issue-1273", "number": 4242}],
    )
    sleeps, sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh,
        head="agent/issue-1273",
        base="main",
        title="t",
        body="b",
        sleep_fn=sleep_fn,
    )

    assert result == PrCreateRetryResult(
        ok=True, pr_number=4242, adopted_existing=True, attempts=1, error=None
    )
    # The whole point: only the one (failed) live create call ever reached
    # the mock. A second `pr_create` would silently open a duplicate PR.
    assert len(gh.create_calls) == 1
    # Adoption short-circuits before any backoff sleep.
    assert sleeps == []
    assert gh.pr_list_calls == 1
    assert gh.invalidate_calls == 1


def test_duplicate_pr_guard_ignores_prs_for_other_branches() -> None:
    """A PR open for a *different* head must never be adopted -- the guard
    matches on `headRefName`, not "any PR exists"."""
    gh = _FakePrCreator(
        [None, 9001],
        existing_prs=[{"headRefName": "agent/issue-9999", "number": 1}],
    )
    _sleeps, sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh, head="agent/issue-1273", base="main", title="t", body="b", sleep_fn=sleep_fn
    )

    assert result.ok is True
    assert result.pr_number == 9001
    assert result.adopted_existing is False
    assert len(gh.create_calls) == 2


def test_duplicate_pr_guard_survives_a_gh_without_the_optional_methods() -> None:
    """Not every `PrCreator` double in the wider suite implements
    `pr_list`/`invalidate_list_cache` (see test_issue_956.py's
    `_SalvageTestGitHub`). The guard must degrade to "just retry" rather than
    raise -- matching `_find_pr_for_head`'s own fail-open philosophy."""

    class _BareGitHub:
        def __init__(self) -> None:
            self.create_calls = 0

        def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
            self.create_calls += 1
            return None if self.create_calls == 1 else 7001

    gh = _BareGitHub()
    _sleeps, sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh, head="agent/issue-1273", base="main", title="t", body="b", sleep_fn=sleep_fn
    )

    assert result.ok is True
    assert result.pr_number == 7001
    assert gh.create_calls == 2


# ---------------------------------------------------------------------------
# AC5: composition with `GitHub.run()`'s existing inner pre-connection-only
# retry. Uses the real `GitHub` class with a monkeypatched `subprocess.run`,
# the same harness `tests/test_github.py` uses for the inner retry itself --
# a fake that merely "remembers" which class an error belongs to would prove
# nothing about whether the two layers actually compose.
# ---------------------------------------------------------------------------


def _fake_run_dispatcher(responses_by_command: dict[str, list[subprocess.CompletedProcess]]):
    """Route `subprocess.run(["gh", *args], ...)` calls by their `gh pr create`
    vs `gh pr list` shape, popping one canned response per call."""

    def _fake_run(cmd: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        key = "create" if "create" in cmd else "list"
        queue = responses_by_command[key]
        assert queue, f"no more canned responses for {key!r} (cmd={cmd!r})"
        return queue.pop(0)

    return _fake_run


def _cp(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_composition_inner_pre_connection_retry_to_success_never_triggers_outer_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-connection error is exactly the shape `GitHub.run()`'s inner
    retry already absorbs. By the time it reaches `create_pr_with_retry`,
    `gh.pr_create()` has already turned two subprocess calls (1 fail + 1
    success) into a single successful return -- the outer ladder must see
    one clean attempt and never call its own `sleep_fn`."""
    responses = {
        "create": [
            _cp(1, stderr="Post: net/http: TLS handshake timeout"),
            _cp(0, stdout="https://github.com/o/r/pull/9"),
        ],
        "list": [_cp(0, stdout="[]")],
    }
    monkeypatch.setattr(github_module.subprocess, "run", _fake_run_dispatcher(responses))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(
        tmp_path, runtime=RuntimeConfig(gh_max_retries=3, gh_retry_base_seconds=1.0)
    )
    outer_sleeps, outer_sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh, head="agent/issue-1273", base="main", title="t", body="b", sleep_fn=outer_sleep_fn
    )

    assert result.ok is True
    assert result.pr_number == 9
    assert result.attempts == 1
    assert outer_sleeps == []
    assert responses["create"] == []  # both canned inner responses were consumed


def test_composition_terminal_for_mutation_error_does_trigger_outer_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The other class: an error `_should_retry` refuses for a mutation
    (HTTP 422 -- not pre-connection) gets zero attempts from the inner layer,
    so `gh.pr_create()` returns `None` on the very first outer attempt. The
    outer ladder must be the thing that retries here."""
    responses = {
        "create": [
            _cp(1, stderr="HTTP 422: Validation Failed"),
            _cp(0, stdout="https://github.com/o/r/pull/11"),
        ],
        "list": [_cp(0, stdout="[]")],
    }
    monkeypatch.setattr(github_module.subprocess, "run", _fake_run_dispatcher(responses))
    monkeypatch.setattr(github_module.time, "sleep", lambda seconds: None)

    gh = github_module.GitHub(
        tmp_path, runtime=RuntimeConfig(gh_max_retries=3, gh_retry_base_seconds=1.0)
    )
    outer_sleeps, outer_sleep_fn = _sleep_recorder()

    result = create_pr_with_retry(
        gh, head="agent/issue-1273", base="main", title="t", body="b", sleep_fn=outer_sleep_fn
    )

    assert result.ok is True
    assert result.pr_number == 11
    # Two outer attempts were needed (inner never retried the 422 on its own).
    assert result.attempts == 2
    assert outer_sleeps == [DEFAULT_BASE_SECONDS]
