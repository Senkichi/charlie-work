"""``GitHub.run(argv)`` as a shim over the transport (ADR-0006, design 3.3).

``run`` keeps its gh-argv signature because ``FakeGitHub`` and a few callers
outside the capability layer pin it. Internally it is now three steps:
``legacy_argv.request_for_argv`` maps the argv to a request, the guarded
transport sends it, and ``_outcome`` renders the outcome into today's
contract (raw value / ``str`` / ``GitHubRunResult`` / raised
``GitHubError``). Retry, breaker, deadline, dry-run and fallback all live in
``GuardedTransport``; nothing here loops.

Under dry-run a mutating argv still returns ``[]`` / ``"DRY-RUN: gh ..."``,
because callers outside the package branch on ``isinstance(str)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..github_transport.legacy_argv import legacy_is_mutating, request_for_argv
from ..github_transport.outcome import Outcome
from ..github_transport.pagination import paginate_rest
from ..github_transport.request import RestRequest
from ._outcome import expect_json, expect_ok, to_run_result

if TYPE_CHECKING:
    from ..github import GitHub


def run_legacy(
    gh: "GitHub",
    args: list[str],
    *,
    json_output: bool,
    allow_failure: bool,
    long_call: bool,
) -> Any:
    command = " ".join(["gh", *args])
    if gh.dry_run and legacy_is_mutating(args):
        return [] if json_output else "DRY-RUN: " + command
    transport = gh._transport_v2
    translated = request_for_argv(
        args, long_call=long_call, use_requests=not transport.kill_switch
    )
    request = translated.request
    outcome: Outcome
    if translated.paginate and isinstance(request, RestRequest):
        outcome = paginate_rest(transport, request)
    else:
        outcome = transport.send(request)
    if allow_failure:
        return to_run_result(outcome, json_output=json_output, command=command)
    if json_output:
        return expect_json(outcome, command=command)
    return expect_ok(outcome, command=command)
