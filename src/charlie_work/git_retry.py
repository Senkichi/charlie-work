"""Retry primitive for idempotent, network-touching ``git`` subprocess calls.

``GitHub.run()`` (``github.py``) already retries every ``gh`` CLI invocation
against transient network blips (backoff + jitter, classified via
:mod:`charlie_work.transient_errors`). Raw ``git`` subprocess calls --
``git fetch``, ``git pull --ff-only``, ``git ls-remote`` -- go through
``subprocess_runner.run_captured`` directly and hit the identical GitHub
edge over the identical network path, but historically got zero retry: one
``subprocess.run``, whatever the result. A single TLS blip failed a whole
``main_ci_reclaim`` pass, or, for ``self_deploy`` (a ~5-minute cadence),
cost up to 5 minutes of staleness for something that resolves in well under
a second (see the design investigation this module implements).

``run_git_with_retry`` mirrors ``GitHub.run()``'s backoff+jitter shape
against the *same* shared classifier, but deliberately drops its
mutation-safety split (``_is_pre_connection_error``): that split exists so a
*mutating* gh call (merge, label, comment) is never retried after an
ambiguous post-send failure, to avoid double-applying it server-side. Every
git command this module is meant for -- fetch, ff-only pull, ls-remote -- is
read-only/idempotent from git's own perspective (a ff-only pull that fails
to fast-forward never partially applies), so there is no at-most-once
concern to preserve and no such split is needed.

Scope, deliberately narrow -- callers must not point this at anything else:

- Never wrap ``git push`` or any other command that mutates the remote.
- A non-fast-forward pull, an auth failure, and a merge conflict are all
  terminal on the very first attempt: none of their error text matches
  :func:`charlie_work.transient_errors.is_transient_network_error`'s
  allowlist, so the loop below never retries them -- this is enforced by
  the classifier, not by a second check here.
- ``worktree.py``'s ``_run_remote_captured`` -- the repo's own chokepoint
  fronting ~15 more read-only ``git fetch``/``git ls-remote`` call sites,
  including the phantom-reap ``ls-remote`` probes -- was wired in a follow-up
  landing (issue #1778): it runs this primitive around ``run_captured`` with
  its own ``_REMOTE_TIMEOUT_SECONDS``, keeping its pre-existing
  retry-once-on-timeout second chance layered on top (a bare timeout has no
  classifiable stderr, so the loop below would never retry one itself).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import subprocess_runner
from .subprocess_runner import RunResult
from .transient_errors import is_transient_network_error

logger = logging.getLogger(__name__)

# Fractional jitter applied to each retry backoff (e.g. 0.25 => +/- 25%),
# matching GitHub.run()'s own jitter fraction.
_JITTER_FRACTION = 0.25

#: Matches ``RuntimeConfig.gh_max_retries``'s default -- 4 total attempts
#: (1 initial + 3 retries).
DEFAULT_MAX_RETRIES = 3
#: Matches ``RuntimeConfig.gh_retry_base_seconds``'s default.
DEFAULT_BASE_DELAY_SECONDS = 1.0
#: Independent wall-clock ceiling on top of ``max_retries`` (see
#: ``run_git_with_retry``'s docstring for why both bounds exist). A fleet
#: pass must never hang on retries: the supervisor's own pass budget is
#: measured in minutes, and this keeps one call's added retry latency to a
#: small fraction of a second-to-low-single-digit-seconds budget.
DEFAULT_MAX_ELAPSED_SECONDS = 20.0


@dataclass(frozen=True)
class RetryOutcome:
    """Summary of one ``run_git_with_retry`` call, handed to ``on_retry``.

    Only constructed when ``attempts > 1`` -- ``run_git_with_retry`` itself
    enforces "one event per retried call, never one per attempt" by only
    invoking ``on_retry`` once, after the loop concludes, and only when a
    retry actually happened. A caller that wires ``on_retry`` to
    ``log_event``/``append_event``/``_record_event`` gets that discipline
    for free rather than having to remember to gate it itself.
    """

    attempts: int
    ok: bool
    error: str | None


def run_git_with_retry(
    command: list[str],
    *,
    cwd: Path | str,
    timeout_seconds: int,
    run_command: Callable[..., RunResult] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_elapsed_seconds: float = DEFAULT_MAX_ELAPSED_SECONDS,
    is_retryable: Callable[[str], bool] = is_transient_network_error,
    sleep: Callable[[float], None] | None = None,
    random_uniform: Callable[[float, float], float] | None = None,
    on_retry: Callable[[RetryOutcome], None] | None = None,
) -> RunResult:
    """Run one idempotent, network-touching ``git`` command with retry.

    Never raises: ``run_command`` (default ``subprocess_runner.run_captured``)
    already guarantees "errors come back as values", and this wrapper
    preserves that contract exactly -- the returned ``RunResult`` is whatever
    the last attempt produced.

    Classification runs against ``result.stderr`` first, falling back to
    ``result.error`` only when stderr is empty (mirrors
    ``subprocess_runner.command_failure_message``'s own fix for the same
    shadowing bug, issue #817 item 3: ``run_captured`` always sets ``.error``
    to a generic ``"command exited N"`` on any non-zero exit, which would
    otherwise permanently shadow git's actual, classifiable stderr text).

    ``run_command``, ``sleep``, and ``random_uniform`` each default to
    ``None`` and are resolved -- to ``subprocess_runner.run_captured``,
    ``time.sleep``, and ``random.uniform`` respectively -- *inside* the
    function body below, rather than bound as a default-argument value (the
    literal ``= run_captured`` this signature used to carry). A default
    argument value is evaluated exactly once, at import time, and captured
    into the function object; monkeypatching the module attribute afterward
    (e.g. ``charlie_work.subprocess_runner.run_captured`` or
    ``charlie_work.git_retry.time.sleep`` -- the usual seams in this repo)
    then has no effect on a caller that does not pass the keyword explicitly,
    because the stale, already-bound reference is what actually gets called.
    Resolving inside the body instead means the *module attribute* is looked
    up fresh on every call, so both seams work.

    Bounded two ways, independently:

    - ``max_retries`` caps the number of *additional* attempts (so
      ``max_retries=3`` means at most 4 attempts total, matching
      ``GitHub.run()``'s ``gh_max_retries`` default).
    - ``max_elapsed_seconds`` gates *entering* another retry: elapsed time so
      far plus the backoff sleep about to happen must still fit the budget,
      checked immediately before that sleep (not merely "elapsed so far",
      which would let the loop commit to a sleep that itself blows the
      budget). It is not a wall-clock guarantee on the call's total
      duration, and cannot be made one without either capping each
      individual attempt's own ``timeout_seconds`` against the remaining
      budget (shrinking a caller-chosen value) or abandoning long
      per-attempt timeouts for genuinely slow-but-working connections
      entirely -- either of which would defeat the call sites (e.g.
      main_ci_reclaim's 120s fetch timeout, nearly 6x this bound's own
      default) that most need the retry. A single slow-but-not-yet-failed
      attempt can still occupy its full ``timeout_seconds`` regardless of
      this bound; in practice a hard timeout is not retried at all --
      ``run_captured`` reports it as ``"command timed out after Ns"``, which
      matches none of the transient allowlist's substrings -- so the common
      case this bound actually guards is repeated *fast* transient failures
      (each well under a second) piling up more backoff sleep than the
      budget allows, not a slow attempt's own timeout.

    ``on_retry``, when given, is called exactly once, after the loop
    concludes, and only if at least one retry happened (``attempts > 1``).
    It never fires for a call that succeeded or failed terminally on the
    first attempt.
    """
    if run_command is None:
        run_command = subprocess_runner.run_captured
    if sleep is None:
        sleep = time.sleep
    if random_uniform is None:
        random_uniform = random.uniform

    start = time.monotonic()
    result = run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)
    attempts = 1

    while (
        not result.ok
        and attempts <= max_retries
        and is_retryable((result.stderr or result.error or "").strip())
    ):
        delay = base_delay_seconds * (2 ** (attempts - 1))
        jitter = random_uniform(-_JITTER_FRACTION * delay, _JITTER_FRACTION * delay)
        sleep_seconds = max(0.0, delay + jitter)
        if (time.monotonic() - start) + sleep_seconds >= max_elapsed_seconds:
            break
        logger.warning(
            "Transient git error (attempt %d/%d): %s; retrying in %.2fs",
            attempts,
            max_retries + 1,
            result.stderr or result.error,
            sleep_seconds,
        )
        sleep(sleep_seconds)
        result = run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)
        attempts += 1

    if attempts > 1 and on_retry is not None:
        on_retry(RetryOutcome(attempts=attempts, ok=result.ok, error=result.error))

    return result
