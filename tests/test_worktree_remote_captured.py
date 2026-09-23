"""Tests for ``worktree._run_remote_captured``'s ``run_git_with_retry`` wiring (issue #1778).

``_run_remote_captured`` is the repo's single chokepoint fronting ~15
read-only remote ``git fetch``/``git ls-remote`` call sites (including the
phantom-reap probes). Issue #1778 routed it through
``git_retry.run_git_with_retry`` so a transient TLS/connection blip is
retried in place instead of collapsing into the same failure value as
"branch genuinely missing". These tests pin that wiring: transient stderr
retries, terminal errors never retried, and the pre-existing
retry-once-on-timeout second chance preserved on top (a bare timeout has no
classifiable stderr, so ``run_git_with_retry`` itself never retries one).

Lives in its own module rather than ``tests/test_worktree.py`` because that
module's attachment point is saturated (243 members over a baselined
ceiling of 241) -- same carve-out convention as ``test_worktree_local_base.py``
et al.

Every test monkeypatches ``worktree.run_captured`` with a narrow
``(command, *, cwd, timeout_seconds)`` queue stub -- the seam the existing
worktree suite already relies on -- and no-ops ``git_retry.time.sleep`` so
no backoff is ever actually waited out (mirroring ``tests/test_git_retry.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import charlie_work.git_retry as git_retry_module
import charlie_work.worktree as worktree_module
from charlie_work.subprocess_runner import RunResult

# A real curl/schannel shape this host's own `git fetch`/`git ls-remote`
# produces against an unroutable remote (see transient_errors.py's module
# docstring) -- the same fixture text tests/test_git_retry.py uses.
_TRANSIENT_STDERR = (
    "fatal: unable to access 'https://github.com/x/y.git/': Failed to connect to "
    "github.com port 443 after 2093 ms: Couldn't connect to server"
)
# An auth refusal -- terminal for ls-remote, matches none of the transient
# allowlist's substrings.
_TERMINAL_STDERR = "fatal: Authentication failed for 'https://github.com/x/y.git/'"


def _ok_result() -> RunResult:
    return RunResult(returncode=0, stdout="", stderr="")


class _QueueRunner:
    """``worktree.run_captured`` stub that pops canned results in call order.

    Narrow ``(command, *, cwd, timeout_seconds)`` signature on purpose:
    ``_run_remote_captured``'s own docstring promises never to forward an
    ``extra_env=None`` a caller did not supply, and this stub would raise
    TypeError if it did.
    """

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        assert timeout_seconds == worktree_module._REMOTE_TIMEOUT_SECONDS
        self.calls.append(command)
        return self._results.pop(0)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No-op ``git_retry.time.sleep`` and return the list of requested delays."""
    sleeps: list[float] = []
    monkeypatch.setattr(git_retry_module.time, "sleep", sleeps.append)
    return sleeps


def test_transient_then_success_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """A transient network blip is retried in place; the second attempt's
    result is returned."""
    runner = _QueueRunner(
        [
            RunResult(
                returncode=128,
                stdout="",
                stderr=_TRANSIENT_STDERR,
                error="command exited 128",
            ),
            _ok_result(),
        ]
    )
    monkeypatch.setattr(worktree_module, "run_captured", runner)

    result = worktree_module._run_remote_captured(
        ["git", "ls-remote", "origin", "refs/heads/agent/x"], cwd=tmp_path
    )

    assert result.ok is True
    assert len(runner.calls) == 2
    # One backoff sleep happened between the two attempts.
    assert len(no_sleep) == 1


def test_terminal_error_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """A non-transient failure returns after exactly one attempt -- an auth
    refusal or "branch not found" shape must never burn retries."""
    runner = _QueueRunner(
        [
            RunResult(
                returncode=128,
                stdout="",
                stderr=_TERMINAL_STDERR,
                error="command exited 128",
            )
        ]
    )
    monkeypatch.setattr(worktree_module, "run_captured", runner)

    result = worktree_module._run_remote_captured(
        ["git", "ls-remote", "origin", "refs/heads/agent/x"], cwd=tmp_path
    )

    assert result.ok is False
    assert len(runner.calls) == 1
    assert no_sleep == []


def test_timeout_still_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """The pre-#1778 retry-once-on-timeout behavior is preserved: a bare
    timeout has no classifiable stderr, so ``run_git_with_retry`` does not
    retry it -- the chokepoint's own second attempt does."""
    timed_out = RunResult(
        returncode=None,
        stdout="",
        stderr="",
        timed_out=True,
        error="command timed out after 20s",
    )
    runner = _QueueRunner([timed_out, _ok_result()])
    monkeypatch.setattr(worktree_module, "run_captured", runner)

    result = worktree_module._run_remote_captured(["git", "fetch", "origin", "main"], cwd=tmp_path)

    assert result.ok is True
    assert len(runner.calls) == 2
    # The timeout second-chance is immediate -- no backoff sleep.
    assert no_sleep == []


def test_extra_env_forwarded_on_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    """``extra_env`` reaches ``run_captured`` on the first attempt and every
    retry -- a caller-supplied env override must not silently drop on the
    retried call."""
    seen_envs: list[dict[str, str] | None] = []
    results = [
        RunResult(returncode=128, stdout="", stderr=_TRANSIENT_STDERR),
        _ok_result(),
    ]

    def stub(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: int,
        extra_env: dict[str, str] | None = None,
    ) -> RunResult:
        seen_envs.append(extra_env)
        return results.pop(0)

    monkeypatch.setattr(worktree_module, "run_captured", stub)

    result = worktree_module._run_remote_captured(
        ["git", "fetch", "origin", "main"],
        cwd=tmp_path,
        extra_env={"GIT_TERMINAL_PROMPT": "0"},
    )

    assert result.ok is True
    assert seen_envs == [{"GIT_TERMINAL_PROMPT": "0"}, {"GIT_TERMINAL_PROMPT": "0"}]
