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
from typing import Any, Callable, Protocol

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
    itself does not count, since it never calls ``pr_create``). ``error``
    carries the last failure's message when ``ok`` is False; ``None`` on
    success.
    """

    ok: bool
    pr_number: int | None
    adopted_existing: bool
    attempts: int
    error: str | None


def _find_pr_for_head(gh: PrCreator, head: str) -> int | None:
    """Return the PR number already open for ``head``, or ``None``.

    Never raises: a failure to list PRs (rate limit, transient `gh` error)
    is indistinguishable here from "no PR exists yet" -- the caller falls
    through to another create attempt either way, which is the same
    fail-open behavior `pr_list()` callers elsewhere in this codebase accept
    when checking "does X already exist" is advisory, not authoritative.
    """
    try:
        prs = gh.pr_list()
    except Exception:
        return None
    for pr in prs:
        if pr.get("headRefName") == head:
            number = pr.get("number")
            if isinstance(number, int):
                return number
    return None


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
    an outer retry sits on top of a mutating command.

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
            existing = _find_pr_for_head(gh, head)
            if existing is not None:
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
