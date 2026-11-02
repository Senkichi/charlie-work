"""``charlie wait-pr``: block until a PR's checks are terminal, over REST only (#2444).

Operator and agent sessions used to poll with ``sleep`` loops around
``gh pr view`` / ``gh pr checks``, both GraphQL-backed, which burned a large
share of the shared 5,000-point/hour GraphQL budget. This command issues only
REST GETs (``pulls/{n}``, ``commits/{sha}/check-runs``, ``commits/{sha}/status``
and, for ``--required-only``, the branch-protection required-contexts route)
through the guarded transport, whose HTTP adapter sends ``If-None-Match`` from
the on-disk ETag cache (#1834): an unchanged resource is a ``304`` that costs
no rate-limit quota. It never builds a ``GraphQLRequest``.

Exit codes (also in the ``--help`` text):

* ``0`` -- every check reached a passing terminal state
* ``1`` -- at least one check failed (reported as soon as it is seen, without
  waiting for the rest)
* ``2`` -- timeout, or the PR could not be read (not found, no access)

The poll loop (:func:`wait_for_pr`) takes the transport ``send`` callable and
the clock/sleep as parameters so tests drive it without a network.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TextIO

from .command_result import CommandResult
from .github_transport.outcome import Outcome, Response
from .github_transport.request import RestRequest

EXIT_PASSED = 0
EXIT_FAILED = 1
EXIT_TIMEOUT = 2

DEFAULT_TIMEOUT_SECONDS = 1800.0
MIN_POLL_SECONDS = 10.0
MAX_POLL_SECONDS = 60.0
BACKOFF_FACTOR = 1.5
# Consecutive unreadable polls tolerated before giving up early (the overall
# timeout still bounds everything; this avoids spinning on a dead token).
MAX_CONSECUTIVE_READ_ERRORS = 5

_PASSING_CONCLUSIONS = frozenset({"success", "neutral", "skipped"})
_FAILING_STATUS_STATES = frozenset({"failure", "error"})
_CHECK_RUN_PAGE_CAP = 10

SendFn = Callable[[RestRequest], Outcome]


@dataclass(frozen=True)
class CheckState:
    name: str
    state: str  # "pass" | "fail" | "pending"
    detail: str = ""


@dataclass(frozen=True)
class Snapshot:
    head_sha: str
    base_ref: str
    checks: tuple[CheckState, ...]

    @property
    def failed(self) -> tuple[CheckState, ...]:
        return tuple(c for c in self.checks if c.state == "fail")

    @property
    def pending(self) -> tuple[CheckState, ...]:
        return tuple(c for c in self.checks if c.state == "pending")

    @property
    def passed(self) -> tuple[CheckState, ...]:
        return tuple(c for c in self.checks if c.state == "pass")

    def only(self, names: frozenset[str]) -> "Snapshot":
        """Restrict to *names*; a name with no registered check is PENDING, not dropped.

        Dropping an unregistered required context would let the run exit 0 before
        the check ever appears (false green).
        """
        kept = [c for c in self.checks if c.name in names]
        seen = {c.name for c in kept}
        kept.extend(CheckState(n, "pending", "not registered") for n in sorted(names - seen))
        return Snapshot(self.head_sha, self.base_ref, tuple(sorted(kept, key=lambda c: c.name)))

    def summary(self) -> str:
        return (
            f"{self.head_sha[:8]}: {len(self.passed)} passed, "
            f"{len(self.pending)} pending, {len(self.failed)} failed"
        )


@dataclass(frozen=True)
class WaitResult:
    exit_code: int
    reason: str
    snapshot: Snapshot | None
    polls: int


class _ReadError(Exception):
    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def _get_json(
    send: SendFn, route: str, query: dict[str, Any] | None = None
) -> tuple[Any, Response]:
    outcome = send(RestRequest.of("GET", route, query=query))
    if not isinstance(outcome, Response):
        raise _ReadError(f"GET {route}: {outcome.detail}")
    if outcome.status in (401, 404):
        raise _ReadError(f"GET {route}: HTTP {outcome.status}", fatal=True)
    if not outcome.ok:
        raise _ReadError(f"GET {route}: HTTP {outcome.status}")
    try:
        return outcome.json(), outcome
    except ValueError as exc:
        raise _ReadError(f"GET {route}: malformed JSON ({exc})") from exc


def _check_run_state(run: dict[str, Any]) -> CheckState:
    name = str(run.get("name") or "?")
    if run.get("status") != "completed":
        return CheckState(name, "pending", str(run.get("status") or "queued"))
    conclusion = str(run.get("conclusion") or "")
    state = "pass" if conclusion in _PASSING_CONCLUSIONS else "fail"
    return CheckState(name, state, conclusion)


def _status_state(status: dict[str, Any]) -> CheckState:
    name = str(status.get("context") or "?")
    raw = str(status.get("state") or "pending")
    if raw == "success":
        return CheckState(name, "pass", raw)
    if raw in _FAILING_STATUS_STATES:
        return CheckState(name, "fail", raw)
    return CheckState(name, "pending", raw)


def _fetch_check_runs(send: SendFn, owner: str, repo: str, sha: str) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for page in range(1, _CHECK_RUN_PAGE_CAP + 1):
        body, _ = _get_json(
            send,
            f"repos/{owner}/{repo}/commits/{sha}/check-runs",
            {"per_page": 100, "page": page},
        )
        page_runs = body.get("check_runs") if isinstance(body, dict) else None
        if not isinstance(page_runs, list):
            raise _ReadError(f"check-runs for {sha[:8]}: unexpected payload")
        runs.extend(r for r in page_runs if isinstance(r, dict))
        total = body.get("total_count")
        if not page_runs or not isinstance(total, int) or len(runs) >= total:
            break
    return runs


def fetch_required_contexts(
    send: SendFn, owner: str, repo: str, base_ref: str
) -> frozenset[str] | None:
    """Required status-check contexts for *base_ref*, or ``None`` if unreadable.

    ``None`` (no protection, no admin scope, transport error) means "cannot
    narrow": the caller considers every check rather than reading an
    unreadable list as an empty required set.
    """
    route = f"repos/{owner}/{repo}/branches/{base_ref}/protection/required_status_checks"
    outcome = send(RestRequest.of("GET", route))
    if not isinstance(outcome, Response) or not outcome.ok:
        return None
    try:
        body = outcome.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    names: set[str] = set()
    contexts = body.get("contexts")
    if isinstance(contexts, list):
        names.update(str(c) for c in contexts)
    checks = body.get("checks")
    if isinstance(checks, list):
        names.update(str(c["context"]) for c in checks if isinstance(c, dict) and c.get("context"))
    return frozenset(names)


def _poll_hint(*responses: Response) -> float | None:
    hints: list[float] = []
    for response in responses:
        raw = response.header("x-poll-interval")
        if raw is None:
            continue
        try:
            hints.append(float(raw))
        except ValueError:
            continue
    return max(hints) if hints else None


def take_snapshot(
    send: SendFn, owner: str, repo: str, number: int
) -> tuple[Snapshot, float | None]:
    """One poll: PR head -> check-runs + commit statuses. Returns (snapshot, poll hint)."""
    pr, pr_response = _get_json(send, f"repos/{owner}/{repo}/pulls/{number}")
    head = pr.get("head") if isinstance(pr, dict) else None
    base = pr.get("base") if isinstance(pr, dict) else None
    sha = head.get("sha") if isinstance(head, dict) else None
    if not isinstance(sha, str) or not sha:
        raise _ReadError(f"PR #{number}: no head sha in payload", fatal=True)
    base_ref = str(base["ref"]) if isinstance(base, dict) and base.get("ref") else "main"

    checks = [_check_run_state(r) for r in _fetch_check_runs(send, owner, repo, sha)]
    combined, status_response = _get_json(send, f"repos/{owner}/{repo}/commits/{sha}/status")
    statuses = combined.get("statuses") if isinstance(combined, dict) else None
    if isinstance(statuses, list):
        checks.extend(_status_state(s) for s in statuses if isinstance(s, dict))
    snapshot = Snapshot(sha, base_ref, tuple(sorted(checks, key=lambda c: c.name)))
    return snapshot, _poll_hint(pr_response, status_response)


def wait_for_pr(
    send: SendFn,
    owner: str,
    repo: str,
    number: int,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    required_only: bool = False,
    sleep: Callable[[float], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    out: TextIO | None = None,
) -> WaitResult:
    """Poll until all checks are terminal; see the module docstring for exit codes."""
    stream = out if out is not None else sys.stderr
    sleep = sleep if sleep is not None else time.sleep  # resolved at call time
    start = monotonic()
    interval = MIN_POLL_SECONDS
    last_line = ""
    last: Snapshot | None = None
    polls = 0
    read_errors = 0
    required: frozenset[str] | None = None
    required_base: str | None = None
    while True:
        polls += 1
        hint: float | None = None
        snapshot: Snapshot | None = None
        try:
            snapshot, hint = take_snapshot(send, owner, repo, number)
            read_errors = 0
        except _ReadError as exc:
            read_errors += 1
            print(f"wait-pr: PR #{number}: {exc}", file=stream)
            if exc.fatal or read_errors >= MAX_CONSECUTIVE_READ_ERRORS:
                return WaitResult(EXIT_TIMEOUT, f"unreadable: {exc}", last, polls)

        if snapshot is None:
            interval = min(interval * BACKOFF_FACTOR, MAX_POLL_SECONDS)
        else:
            if last is not None and snapshot.head_sha != last.head_sha:
                print(
                    f"wait-pr: head moved {last.head_sha[:8]} -> {snapshot.head_sha[:8]}",
                    file=stream,
                )
            if required_only:
                if required_base != snapshot.base_ref:
                    required_base = snapshot.base_ref
                    required = fetch_required_contexts(send, owner, repo, snapshot.base_ref)
                    if required is None:
                        print(
                            "wait-pr: required contexts unreadable; considering all checks",
                            file=stream,
                        )
                    elif not required:
                        print(
                            "wait-pr: base branch has no required contexts; "
                            "considering all checks",
                            file=stream,
                        )
                if required:
                    snapshot = snapshot.only(required)
            line = snapshot.summary()
            changed = line != last_line
            if changed:
                print(f"wait-pr: {line}", file=stream)
                last_line = line
            last = snapshot
            if snapshot.failed:
                names = ", ".join(f"{c.name} ({c.detail})" for c in snapshot.failed)
                return WaitResult(EXIT_FAILED, f"failed: {names}", snapshot, polls)
            # No checks yet is "not registered yet", never "all passed".
            if snapshot.checks and not snapshot.pending:
                return WaitResult(EXIT_PASSED, "all checks passed", snapshot, polls)
            interval = (
                MIN_POLL_SECONDS if changed else min(interval * BACKOFF_FACTOR, MAX_POLL_SECONDS)
            )

        delay = max(interval, hint or 0.0)
        remaining = timeout - (monotonic() - start)
        if remaining <= 0:
            return WaitResult(EXIT_TIMEOUT, f"timed out after {timeout:.0f}s", last, polls)
        sleep(min(delay, remaining))


def register_wait_pr_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "wait-pr",
        help=(
            "Wait for a PR's checks to finish using REST + ETag only (zero GraphQL). "
            "Exit codes: 0 all checks passed, 1 a check failed, 2 timeout/unreadable PR."
        ),
        epilog="exit codes: 0 all passed; 1 any failed; 2 timeout or PR unreadable",
    )
    parser.add_argument("number", type=int, help="PR number")
    parser.add_argument(
        "--repo",
        dest="wait_repo",
        default=None,
        metavar="OWNER/NAME",
        help="GitHub repository (default: this checkout's origin)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Give up after this many seconds (default: {DEFAULT_TIMEOUT_SECONDS:.0f}); exit 2",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help=(
            "Only consider checks required by the base branch's protection; a required "
            "check that has not registered yet counts as pending. If the required list "
            "is unreadable or empty, all checks are considered (a note is printed)."
        ),
    )


def _split_slug(slug: str) -> tuple[str, str] | None:
    owner, _, name = slug.partition("/")
    if not owner or not name or "/" in name:
        return None
    return owner, name


def _usage_error(message: str) -> CommandResult:
    return CommandResult(False, f"wait-pr: {message}", {"wait_pr_exit_code": EXIT_TIMEOUT})


def run_wait_pr_command(args: argparse.Namespace) -> CommandResult:
    """CLI entry: resolve owner/repo, wait, carry the exit code in ``data``."""
    from . import cli  # deferred: circular-import guard, as in the sibling *_command modules
    from .config import ConfigError
    from .github import GitHub
    from .paths import RepoNotFoundError

    slug = getattr(args, "wait_repo", None)
    target = _split_slug(slug) if slug else None
    if slug and target is None:
        return _usage_error(f"--repo must be OWNER/NAME, got {slug!r}")

    try:
        gh = cli.bootstrap_command(args).gh
    except (RepoNotFoundError, ConfigError):
        if target is None:
            raise
        # An explicit OWNER/NAME needs no fleet checkout: a bare client is enough.
        gh = GitHub(repo_root=Path.cwd())
    if target is None:
        target = gh._repo_owner_name()
    owner, repo = target

    transport = getattr(gh, "_transport_v2", None)
    if transport is None:
        return _usage_error(
            f"{type(gh).__name__} has no REST transport; wait-pr needs the GitHub client"
        )
    result = wait_for_pr(
        transport.send,
        owner,
        repo,
        args.number,
        timeout=args.timeout,
        required_only=args.required_only,
    )
    snapshot = result.snapshot
    return CommandResult(
        result.exit_code == EXIT_PASSED,
        f"wait-pr {owner}/{repo}#{args.number}: {result.reason} (exit {result.exit_code})",
        {
            "wait_pr_exit_code": result.exit_code,
            "polls": result.polls,
            "head_sha": snapshot.head_sha if snapshot else None,
            "checks": [
                {"name": c.name, "state": c.state, "detail": c.detail}
                for c in (snapshot.checks if snapshot else ())
            ],
        },
    )
