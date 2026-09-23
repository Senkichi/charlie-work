"""Free-function circuit-breaker helpers for ``Transport`` (issue #1833).

Split out of ``transport.py`` to keep that module under the file-size
ratchet's 800-line cap (issue #1442) without growing ``config.py`` (already
over its own cap) instead. These are plain module-level functions, not
``Transport`` methods, because none of them is a ``self.<name>()`` call
target reached through ``GitHub``'s owner-to-collaborator delegation --
``github_delegation._build_routes()`` only routes members declared directly
in a collaborator class's own ``__dict__`` (``vars(collab_cls)``, not
inherited/mixin members), so anything never called that way is free to live
here as an ordinary function. ``Transport`` keeps thin wrapper methods for
the handful of members that ARE reached via delegation or that other
``Transport`` methods call by name: ``_circuit_breaker_allow_call``,
``_circuit_breaker_open_message``, ``_circuit_breaker_note_result``, and
``reset_circuit_breaker``.

``note_circuit_breaker_result`` inlines what used to be a separate
``Transport._emit_circuit_event`` helper: both of its call sites pass a
string literal ``kind`` ("github_circuit_opened" / "github_circuit_closed"),
so writing the ``log_event(...)`` calls directly here (rather than through an
indirection that forwards a ``kind`` parameter) keeps both of this repo's
AST-based event-kind scanners (``tests/test_instrumentation_event_kind_registry.py``,
``tests/test_event_kind_consumers.py``) resolving the kind by simple literal
matching, with no wrapper-function registration needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import layout
from ..instrumentation import log_event
from .circuit_breaker import CircuitBreakerState, GhFailureClass, classify_gh_failure

if TYPE_CHECKING:
    # Type-only: importing charlie_work.config at module level would cycle
    # (config -> github -> github_capabilities -> transport /
    # circuit_breaker_transport). Mirrors transport.py's identical note.
    from ..config import RuntimeConfig

# Circuit breaker defaults (issue #1833). Owner directive: ships enabled by
# default with sensible thresholds -- config exists only to retune or
# disable it (threshold set high enough, or cooldown 0), never as an
# opt-in flag. Five consecutive transport-class failures is enough above
# normal single-blip noise to avoid false trips, while still catching the
# #1832 pattern (every call in a pass failing the same way) within the
# first handful of calls rather than the whole pass. 60s cooldown is long
# enough to ride out a brief network blip without cycling to a probe every
# few seconds, short enough that a fleet pass isn't blocked for the bulk of
# its runtime once the network recovers. Moved from ``transport.py`` verbatim
# alongside ``build_circuit_breaker_state`` (issue #1833 follow-up, file-size
# ratchet issue #1442).
_DEFAULT_GH_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
_DEFAULT_GH_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 60.0


def build_circuit_breaker_state(runtime: "RuntimeConfig | None") -> CircuitBreakerState:
    """Construct the per-``GitHub``-instance breaker state (issue #1833).

    Called exactly once, from ``GitHub.__post_init__`` -- before that
    dataclass's ``_COLLABORATORS`` loop installs the collaborator instances,
    so it cannot go through ``self._transport``/``CapabilityCollaborator``
    delegation yet, and must be a plain module-level function rather than a
    ``Transport`` method. Kept here (not duplicated in ``github.py``) as the
    single place that knows the ``gh_circuit_breaker`` config keys and their
    defaults, matching how ``_max_retries``/``_retry_base_seconds``/
    ``_timeout_seconds`` are the single source for their own knobs.
    """
    if runtime is not None:
        breaker_config = runtime.gh_circuit_breaker
        return CircuitBreakerState(
            failure_threshold=breaker_config.failure_threshold,
            cooldown_seconds=breaker_config.cooldown_seconds,
        )
    return CircuitBreakerState(
        failure_threshold=_DEFAULT_GH_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        cooldown_seconds=_DEFAULT_GH_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    )


def circuit_breaker_state_path(runtime: "RuntimeConfig | None", repo_root: Path) -> Path:
    """Compute ``state.json``'s path without going through ``paths.py``
    (which imports ``.config`` and would cycle -- see the module docstring).
    Mirrors ``paths.runtime_paths``' resolution of ``runtime.state_dir``
    exactly (absolute as-is, relative joined to ``repo_root``), using only
    ``layout.py`` primitives. Moved verbatim from
    ``Transport._circuit_breaker_state_path``, which took only ``self`` and
    resolved ``runtime``/``repo_root`` through owner delegation; this free
    function takes the same two values as explicit parameters instead.
    """
    state_dir = runtime.state_dir if runtime is not None else layout.DEFAULT_STATE_DIR
    root = Path(state_dir)
    if not root.is_absolute():
        root = repo_root / root
    return layout.state_file_path(root.resolve())


def circuit_breaker_open_message(breaker: CircuitBreakerState, command: list[str]) -> str:
    """Build the refusal message for a call the breaker is short-circuiting.

    Moved verbatim (module-arity change only) from
    ``Transport._circuit_breaker_open_message``.
    """
    return (
        f"GitHub circuit breaker open: refusing to run {' '.join(command)} "
        f"after {breaker.consecutive_failures} consecutive transport-class "
        f"gh failure(s) (threshold {breaker.failure_threshold}); cooldown "
        f"{breaker.cooldown_seconds:g}s"
    )


def note_circuit_breaker_result(
    breaker: CircuitBreakerState,
    state_path: Path,
    *,
    error: str | None = None,
    timed_out: bool = False,
) -> None:
    """Classify one completed ``gh`` invocation's outcome and update the
    breaker (issue #1833).

    Call with ``error=None`` for a genuine success. Call with the failure
    text (or ``timed_out=True`` for a hang, or when the failure is known
    transport-class by construction -- e.g. ``gh`` not found -- regardless
    of its message text) for any failure; ``classify_gh_failure`` decides
    whether it counts toward the breaker.

    Called exactly once per logical ``gh`` invocation, at the point its
    FINAL outcome is known -- never per internal retry attempt within
    ``GitHub.run()``'s loop -- so "N consecutive failures" counts
    invocations, matching the acceptance criteria's framing ("further calls
    in the same pass fail immediately").

    Moved from ``Transport._circuit_breaker_note_result`` with its
    ``self._emit_circuit_event(...)`` calls inlined as direct, literal-kind
    ``log_event(...)`` calls (module docstring above explains why).
    """
    payload: dict[str, Any]
    if classify_gh_failure(error, timed_out=timed_out) is GhFailureClass.TRANSPORT:
        transition = breaker.record_transport_failure()
        if transition == "opened":
            payload = {
                "consecutive_failures": breaker.consecutive_failures,
                "failure_threshold": breaker.failure_threshold,
                "cooldown_seconds": breaker.cooldown_seconds,
            }
            log_event(state_path, "github_circuit_opened", payload)
    else:
        transition = breaker.record_success()
        if transition == "closed":
            payload = {"cooldown_seconds": breaker.cooldown_seconds}
            log_event(state_path, "github_circuit_closed", payload)
