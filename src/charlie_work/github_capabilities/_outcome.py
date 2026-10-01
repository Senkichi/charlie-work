"""Capability-side conversion of a transport ``Outcome`` (ADR-0006, design 3.3).

The transport returns values; this module turns them into the two shapes the
capability layer has always exposed: a raised ``GitHubError`` /
``GitHubNotFoundError`` (``allow_failure=False``) or a ``GitHubRunResult``
value (``allow_failure=True``). It holds the one rendering of a failed
outcome into error text, so the raise-vs-return contract lives in one place.

``GitHubNotFoundError`` and its text classifier live here (and are re-exported
by ``github.py``) because the conversion must raise them and ``github.py``
imports this package, not the other way round.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ci_fleet.github import GitHubError

from ..github_transport.outcome import Outcome, Response, TransportFailure, render_legacy_error
from ._base import GitHubRunResult

# Conventional exit status for "killed by timeout" (GNU coreutils `timeout`).
TIMEOUT_RETURNCODE = 124
# Sentinel returncode for "the circuit breaker refused this call": no gh
# process ever ran, so there is no real exit status.
CIRCUIT_OPEN_RETURNCODE = 125


class GitHubNotFoundError(GitHubError):
    """The referenced GitHub object does not exist in this repository.

    Permanent (not retryable): raised when gh reports a GraphQL
    could-not-resolve or REST 404 for the requested object. Callers that
    derive object numbers from untrusted inputs (e.g. PR branch names) use
    this to distinguish "will never succeed" from transient gh failures.
    """


def is_not_found_gh_error(error: str) -> bool:
    """Classify a gh stderr/stdout string as an object-does-not-exist failure.

    Matches GitHub's GraphQL could-not-resolve shape and REST 404s -- the same
    signals `is_transient_network_error` already treats as terminal.
    """
    text = error.lower()
    if "could not resolve to a" in text or "not_found" in text:
        return True
    return bool(re.search(r"\bhttp 404\b", text))


def failure_text(outcome: Outcome) -> str:
    """The error text of a failed outcome, in today's ``final_error`` shape."""
    if isinstance(outcome, Response) and outcome.status == 0:
        # A verbatim gh run: stderr, else stdout, else the bare exit status.
        return outcome.stderr.strip() or outcome.body.strip() or str(outcome.returncode)
    return render_legacy_error(outcome)


def is_success(outcome: Outcome) -> bool:
    return isinstance(outcome, Response) and outcome.ok


def raise_for_failure(outcome: Outcome) -> None:
    """Raise the legacy exception for a failed outcome; no-op on success."""
    if is_success(outcome):
        return
    text = failure_text(outcome)
    if is_not_found_gh_error(text):
        raise GitHubNotFoundError(text)
    raise GitHubError(text)


def _failure_returncode(outcome: Outcome) -> int:
    if isinstance(outcome, TransportFailure):
        kind = outcome.kind.value
        if kind == "timeout":
            return TIMEOUT_RETURNCODE
        if kind == "circuit_open":
            return CIRCUIT_OPEN_RETURNCODE
        if kind == "cli_missing":
            return 0
        return 1
    return outcome.returncode if outcome.returncode is not None else 1


def to_run_result(outcome: Outcome, *, json_output: bool, command: str) -> GitHubRunResult:
    """The ``allow_failure=True`` shape for any outcome.

    Success: ``value`` is the parsed JSON (``json_output``) or the stripped
    stdout. A failure keeps whatever body the response carried, so a partial
    GraphQL answer still reaches the caller (#1933).
    """
    if isinstance(outcome, Response) and outcome.ok:
        output = outcome.body.strip()
        value: Any | None = None
        if not json_output:
            value = output
        elif output:
            try:
                value = json.loads(output)
            except json.JSONDecodeError:
                return GitHubRunResult(
                    ok=False,
                    returncode=_ok_returncode(outcome),
                    stdout=outcome.body,
                    stderr=outcome.stderr,
                    value=None,
                    error=f"Expected JSON from gh command: {command}",
                )
        return GitHubRunResult(
            ok=True,
            returncode=_ok_returncode(outcome),
            stdout=outcome.body,
            stderr=outcome.stderr,
            value=value,
            error=None,
        )
    error = failure_text(outcome)
    stdout = outcome.body if isinstance(outcome, Response) else ""
    if isinstance(outcome, Response) and outcome.status != 0 and not outcome.graphql_errors:
        stdout = ""  # a plain HTTP error carried no stdout in the gh-shaped result
    stderr = outcome.stderr if isinstance(outcome, Response) and outcome.status == 0 else error
    if isinstance(outcome, TransportFailure) and outcome.kind.value == "cli_missing":
        stderr = ""
    value = None
    output = stdout.strip()
    if json_output and output:
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            error = f"Expected JSON from gh command: {command}"
    return GitHubRunResult(
        ok=False,
        returncode=_failure_returncode(outcome),
        stdout=stdout,
        stderr=stderr,
        value=value,
        error=error,
    )


def _ok_returncode(response: Response) -> int:
    return response.returncode if response.returncode is not None else 0


def expect_ok(outcome: Outcome, *, command: str) -> str:
    """Stripped stdout of a successful outcome; raises on failure."""
    del command
    raise_for_failure(outcome)
    assert isinstance(outcome, Response)
    return outcome.body.strip()


def expect_json(outcome: Outcome, *, command: str) -> Any:
    """Parsed JSON body of a successful outcome; raises on failure.

    An empty body is an error, not ``None`` -- it cannot be told apart from an
    unreadable one (issue #756).
    """
    output = expect_ok(outcome, command=command)
    if not output:
        raise GitHubError(
            f"gh exited 0 with empty stdout for command: {command}; "
            "cannot distinguish an empty result from an unreadable one"
        )
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise GitHubError(f"Expected JSON from gh command: {command}") from exc


__all__ = [
    "CIRCUIT_OPEN_RETURNCODE",
    "GitHubNotFoundError",
    "TIMEOUT_RETURNCODE",
    "expect_json",
    "expect_ok",
    "failure_text",
    "is_not_found_gh_error",
    "raise_for_failure",
    "to_run_result",
]
