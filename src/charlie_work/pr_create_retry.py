"""Bounded outer retry for ``gh pr create``, with a duplicate-PR guard.

cw#1273 binding design (issue comment, phase-0-recon corrections):

``GitHub.run()`` already retries mutating commands (``pr create`` included)
on *pre-connection* failures only (TLS handshake timeout, connection
refused, could-not-connect, error-connecting-to) -- see ``github.py``'s
``_is_pre_connection_error``/``_should_retry``. That retry is deliberately
narrow: a mutation whose response is lost *after* it reached GitHub (a
post-send ambiguous timeout, a 5xx after headers) is never retried there,
because retrying an already-applied mutation risks a duplicate. Its total
span is short (``gh_max_retries`` * ``gh_retry_base_seconds``-scaled backoff,
~7s with the defaults) -- far shorter than the ~45s GitHub-side TLS blips
observed on this host (confirmed live, ``api.glb`` placeholder cert).

This module adds a second, OUTER retry layer on top, specifically for
``gh pr create``. It composes with the inner layer by construction rather
than by classifying error text itself: ``GitHub.pr_create`` collapses every
failure mode -- terminal-for-mutations immediately, or pre-connection
retried internally to exhaustion -- to the same ``None`` return (errors are
values in this codebase; see ``GitHub.pr_create``'s docstring). A pre-
connection error the inner layer retries *to success* never reaches this
module as a failure at all, so the outer ladder below only ever fires for
whatever the inner layer was unable to resolve on its own -- it can never
duplicate a success. What it can do is give a real mutation more total wall
clock than the inner layer's ~7s budget, spanning the observed ~45s blip.

Before every RETRY attempt (never the first, since nothing could exist yet)
this module checks whether a PR already exists for the head branch before
issuing another ``gh pr create`` -- the previous attempt may have been an
ambiguous post-send failure where the mutation actually landed and only the
response was lost, and a second create for the same branch would open a
duplicate PR. When a matching PR is found it is adopted (returned as
success) instead of creating a second one.

The precheck itself (``gh pr list``) is not infallible -- it is a live `gh`
call subject to the same rate limits and transient failures as everything
else in this module, and it *raises* ``GitHubError`` rather than returning
a value on failure (see ``GitHub.pr_list()``). Collapsing that failure into
"no PR found" -- as an earlier version of this module did -- is itself the
duplicate-PR bug: the guard exists precisely for the case where a create's
outcome is ambiguous, and a precheck failure right after an ambiguous
create is the *same* transient condition biting twice, not independent
evidence that nothing exists. So the precheck is three-state (found /
not-found / precheck-failed, see ``PrecheckStatus``), and a
``PRECHECK_FAILED`` result blocks the next create -- consuming the retry
slot with the normal backoff and re-checking next time -- whenever a create
has already been attempted this invocation, since that is exactly when the
ambiguity exists. On the very first attempt (nothing sent yet), a failed
precheck can never fire in practice (it is gated behind ``attempt > 0``,
which always follows attempt 0's unconditional create), but the asymmetry
is still written as an explicit condition rather than folded into that
gate, so it survives if the gating is ever refactored: blocking a create
that has no in-flight predecessor would trade an availability regression
for zero duplicate-risk reduction.

On final exhaustion, callers are expected to emit a
``pr_create_failed_branch_stranded`` event through the orphan-reap sweep's
existing ``_drift_fingerprint`` dedup path (workflow.py) -- this module does
not emit events itself; it has no ``state_file`` and no fingerprint state to
dedup against, and inventing a second dedup mechanism here would duplicate
machinery that already exists one layer up.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from .github import GitHubError

logger = logging.getLogger(__name__)

# Matches RuntimeConfig.gh_max_retries's naming: "N retries" means N
# *additional* attempts after the first, i.e. N + 1 total live `pr_create`
# invocations -- see RuntimeConfig.pr_create_retry_max_attempts.
DEFAULT_MAX_RETRIES = 3
# Backoff before retry attempts 1, 2, 3 (0-indexed exponent) with the default
# base and multiplier: 10s, 30s, 90s -- spans the ~45s blip within two
# retries in the common case, with a third as margin for a worse one.
DEFAULT_BASE_SECONDS = 10.0
_BACKOFF_MULTIPLIER = 3.0


def _default_sleep(seconds: float) -> None:
    """Indirection so ``create_pr_with_retry``'s backoff sleep is test-patchable.

    A default *parameter value* like ``sleep_fn: ... = time.sleep`` is
    evaluated exactly once, at module-import time, and the resulting
    reference is baked into the function's ``__defaults__``. Patching
    ``time.sleep`` on the shared stdlib module object afterward -- e.g. from
    a test fixture -- can never reach that already-captured reference, so it
    would silently fail to stub anything. Naming a function here instead and
    referencing it *by name inside the function body* (see
    ``create_pr_with_retry`` below) makes it a fresh global-namespace lookup
    on every call, which a fixture CAN patch via
    ``monkeypatch.setattr(pr_create_retry, "_default_sleep", ...)`` --
    without touching the shared ``time`` module, so unrelated tests calling
    ``time.sleep`` directly for their own reasons are never affected.
    """
    time.sleep(seconds)


class PrCreator(Protocol):
    """Narrow structural protocol for the `gh` surface this module needs.

    A subset of `github.GitHubLike` -- deliberately not importing that wider
    protocol, matching `closing_reference.py`'s `_IssueViewer` pattern. Test
    doubles can satisfy this with three methods instead of the full surface.
    """

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None: ...

    def pr_list(self) -> list[dict[str, Any]]: ...

    def invalidate_list_cache(self) -> None: ...


@dataclass(frozen=True)
class PrCreateRetryResult:
    """Outcome of `create_pr_with_retry`. Never raised; always returned.

    ``ok`` is True whenever a PR number was obtained at all -- either freshly
    created or adopted from an existing PR the duplicate-PR guard found.
    ``pr_number`` is that number, or ``None`` on exhaustion. ``adopted_existing``
    is True only on the duplicate-PR-guard path, so a caller/test can tell
    "we created it" apart from "we found and reused one that already
    existed" without inferring it from ``attempts``. ``attempts`` counts live
    ``gh pr create`` invocations actually issued (the duplicate-PR precheck
    itself does not count, since it never calls ``pr_create`` -- nor does a
    retry slot consumed by a ``PRECHECK_FAILED`` block, since that path also
    never calls ``pr_create``; ``attempts`` can therefore be smaller than the
    number of loop iterations spent). ``error`` carries the last failure's
    message when ``ok`` is False; ``None`` on success.
    """

    ok: bool
    pr_number: int | None
    adopted_existing: bool
    attempts: int
    error: str | None


class PrecheckStatus(Enum):
    """Outcome of the duplicate-PR precheck (cw#1273 fail-open fix).

    ``FOUND``: a PR already exists for the head branch -- adopt it.
    ``NOT_FOUND``: no matching PR exists, or the ``gh`` double doesn't
    implement ``pr_list`` at all (some test fakes elsewhere in the suite) --
    safe to create.
    ``PRECHECK_FAILED``: ``pr_list()`` itself raised ``GitHubError`` -- the
    precheck could not determine whether a PR exists. Deliberately distinct
    from ``NOT_FOUND``: `create_pr_with_retry` treats the two differently
    once a create has already been attempted this invocation, since a
    precheck failure right after an ambiguous create carries no information
    either way, unlike a genuinely empty list.
    """

    FOUND = "found"
    NOT_FOUND = "not_found"
    PRECHECK_FAILED = "precheck_failed"


def _find_pr_for_head(gh: PrCreator, head: str) -> tuple[PrecheckStatus, int | None]:
    """Check whether a PR already exists for ``head``.

    Returns ``(PrecheckStatus, pr_number)``; ``pr_number`` is only
    meaningful when the status is ``FOUND``.

    Never raises. Two distinct "we can't tell" shapes are both tolerated,
    but classified differently:

    - A ``PrCreator`` double lacking ``pr_list`` entirely (structurally
      incomplete -- see
      ``test_duplicate_pr_guard_survives_a_gh_without_the_optional_methods``)
      is checked with ``getattr`` rather than ``except AttributeError``, so
      an ``AttributeError`` raised *from inside* a real ``pr_list()`` is
      never mistaken for a missing method. This maps to ``NOT_FOUND``,
      preserving this module's pre-existing tolerance for incomplete test
      doubles.
    - A live ``GitHubError`` from ``pr_list()`` (rate limit, transient `gh`
      failure -- see ``GitHub.pr_list()``) maps to ``PRECHECK_FAILED``, not
      ``NOT_FOUND``. Collapsing these together is the cw#1273 fail-open bug
      this function exists to close: the guard is consulted precisely when
      a previous create's outcome is ambiguous, and treating "the precheck
      also failed" as "nothing exists" lets a second `pr_create` fire while
      a matching PR may already be sitting there, opening a duplicate.
    """
    pr_list = getattr(gh, "pr_list", None)
    if pr_list is None:
        return (PrecheckStatus.NOT_FOUND, None)
    try:
        prs = pr_list()
    except GitHubError:
        return (PrecheckStatus.PRECHECK_FAILED, None)
    except Exception:
        # Defensive only: `pr_list()`'s one realistic raise path is
        # `GitHubError` (via `GitHub._list_json` -> `run(allow_failure=False)`).
        # An unanticipated exception type here is a bug elsewhere, not the
        # transient-failure case this module distinguishes -- fail open to
        # NOT_FOUND (this module's long-standing default for "can't tell")
        # rather than let it escape and break the "never raises" contract.
        return (PrecheckStatus.NOT_FOUND, None)
    for pr in prs:
        if pr.get("headRefName") == head:
            number = pr.get("number")
            if isinstance(number, int):
                return (PrecheckStatus.FOUND, number)
    return (PrecheckStatus.NOT_FOUND, None)


def _should_block_create_after_precheck_failure(status: PrecheckStatus, attempts: int) -> bool:
    """True when a ``PRECHECK_FAILED`` result should block this attempt's create.

    Blocks only once a create has already been attempted this invocation
    (``attempts > 0``) -- the state is ambiguous only once something has
    actually been sent to GitHub. On the very first attempt (``attempts ==
    0``, nothing sent yet), a failed precheck carries no in-flight ambiguity
    to protect against, so it must not block: blocking here would trade an
    availability regression for zero duplicate-risk reduction.

    Extracted as its own predicate -- rather than inlined into the loop's
    `attempt > 0` precheck gate -- specifically so this asymmetry is
    independently testable: today, `attempts` is always `> 0` wherever the
    caller invokes this (the precheck only ever runs after attempt 0's
    unconditional first `pr_create`), so the `attempts == 0` branch is
    unreachable through `create_pr_with_retry`'s public API alone. See
    `test_precheck_failure_blocks_creation_only_once_one_is_already_in_flight`.
    """
    return status is PrecheckStatus.PRECHECK_FAILED and attempts > 0


def create_pr_with_retry(
    gh: PrCreator,
    *,
    head: str,
    base: str,
    title: str,
    body: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_seconds: float = DEFAULT_BASE_SECONDS,
    sleep_fn: Callable[[float], None] | None = None,
) -> PrCreateRetryResult:
    """Create a PR for ``head`` into ``base``, retrying failures up to ``max_retries`` times.

    ``max_retries`` additional attempts follow the first on failure --
    ``max_retries + 1`` total live ``gh pr create`` calls at most, mirroring
    ``RuntimeConfig.gh_max_retries``'s naming. Backoff before retry attempt
    *n* (1-indexed) is ``base_seconds * (3 ** (n - 1))`` -- 10s/30s/90s with
    the defaults. ``sleep_fn`` is injectable so tests never really sleep;
    when omitted (``None``), resolves to this module's ``_default_sleep``
    (itself just ``time.sleep``) -- see that function's docstring for why
    the indirection exists instead of a plain default-parameter binding.
    Production callers should leave this unset.

    Before each retry (not the first attempt), checks whether a PR already
    exists for ``head`` and adopts it instead of creating a second one --
    see the module docstring for why this is necessary, not optional, once
    an outer retry sits on top of a mutating command. That precheck is
    three-state (`PrecheckStatus`): a `PRECHECK_FAILED` result -- the
    precheck's own `pr_list()` call raised `GitHubError` -- blocks this
    attempt's create (consuming the retry slot with the normal backoff and
    re-checking next time) whenever a create has already been attempted
    this invocation, since that is exactly when the previous outcome is
    ambiguous; see `_should_block_create_after_precheck_failure`. If every
    remaining attempt is consumed this way, exhaustion returns the same
    failure result as any other exhaustion path -- no second `pr_create` is
    ever issued while the precheck cannot confirm one way or the other.

    Never raises. Returns a `PrCreateRetryResult` in every case.
    """
    last_error: str | None = None
    attempts = 0
    resolved_sleep_fn = sleep_fn if sleep_fn is not None else _default_sleep

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Duplicate-PR guard (cw#1273): the previous attempt's failure
            # may have been ambiguous (mutation applied server-side, only the
            # response lost) -- check before retrying, not after, so a second
            # `gh pr create` is never issued once one already exists.
            # `invalidate_list_cache()` matters when present -- `pr_list()`
            # is cached per orchestrator pass (issue #812), so without this
            # the check would read a snapshot taken before the create
            # attempt that just failed and could never observe the PR it
            # exists to find. Not every `PrCreator` double in the test suite
            # implements it (only the real `GitHub` and the fuller fakes
            # do), so this call is advisory like `_find_pr_for_head`'s own
            # `pr_list()` guard below -- a caller lacking it just skips
            # straight to the (possibly stale) existence check instead of
            # crashing, which never makes the duplicate-PR guard less safe
            # than not having it at all.
            try:
                gh.invalidate_list_cache()
            except Exception:
                pass
            status, existing = _find_pr_for_head(gh, head)
            if status is PrecheckStatus.FOUND:
                logger.info(
                    "gh pr create: adopting existing PR #%d for head=%s instead of retrying "
                    "(previous attempt's outcome was ambiguous)",
                    existing,
                    head,
                )
                return PrCreateRetryResult(
                    ok=True,
                    pr_number=existing,
                    adopted_existing=True,
                    attempts=attempts,
                    error=None,
                )

            delay = base_seconds * (_BACKOFF_MULTIPLIER ** (attempt - 1))

            if _should_block_create_after_precheck_failure(status, attempts):
                # Duplicate-PR guard, precheck-failure branch (cw#1273): the
                # precheck itself just failed, so we cannot tell whether the
                # *previous* pr_create call's ambiguous outcome actually
                # landed server-side. Retrying blind here is exactly the
                # fail-open bug this branch closes -- a second `pr_create`
                # risks opening a duplicate PR if the first one silently
                # succeeded. Consume this attempt with the normal backoff
                # and re-precheck next time instead of creating; if attempts
                # exhaust while still PRECHECK_FAILED, the loop falls
                # through to the terminal failure result below unchanged,
                # and the caller's stranded-branch breadcrumb fires -- a
                # missed retry slot is a strictly safer outcome than a
                # silent duplicate PR.
                logger.warning(
                    "gh pr create: duplicate-PR precheck failed (attempt %d/%d, head=%s) "
                    "with a create already in flight this invocation -- not retrying blind; "
                    "consuming the attempt and re-checking next time instead",
                    attempt,
                    max_retries + 1,
                    head,
                )
                resolved_sleep_fn(delay)
                continue

            # NOT_FOUND, or a PRECHECK_FAILED with nothing sent yet this
            # invocation (unreachable today -- see
            # `_should_block_create_after_precheck_failure`'s docstring) --
            # proceed with the create attempt either way.
            logger.warning(
                "gh pr create failed (attempt %d/%d, head=%s): %s; retrying in %gs",
                attempt,
                max_retries + 1,
                head,
                last_error,
                delay,
            )
            resolved_sleep_fn(delay)

        attempts += 1
        pr_number = gh.pr_create(head=head, base=base, title=title, body=body)
        if pr_number is not None:
            return PrCreateRetryResult(
                ok=True,
                pr_number=pr_number,
                adopted_existing=False,
                attempts=attempts,
                error=None,
            )
        last_error = "gh pr create failed or returned no PR number"

    return PrCreateRetryResult(
        ok=False,
        pr_number=None,
        adopted_existing=False,
        attempts=attempts,
        error=last_error,
    )
