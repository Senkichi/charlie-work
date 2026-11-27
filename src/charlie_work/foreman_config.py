"""The ``foreman:`` fleet config section (issue #3117).

Fleet-wide settings for the foreman layer — which harness and model run a
foreman's model call, the role chain around it, and the pass cadence. The
dataclass lives in its own module so it does not land in the over-cap
``config.py`` monolith (file-size ratchet, issue #1442); ``config.py``
re-exports it and wires it into ``OrchestratorConfig`` exactly like
``SlotPoolConfig``/``FleetSupervisorConfig``.

The role is deliberately *reviewer-harness-shaped* — ``REVIEWER_HARNESSES``
is the read-only harness set, which is what a foreman's Decide step needs —
and its ``fallbacks`` reuse ``role_entries``/``check_role_chain`` so the
role-chain machinery (``role_chain.chain_of``, ``role_selection``) resolves
it without a second chain implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from .config_validation import NonEmpty, NotNull, NullIsDefault, OneOf, Positive, Typed
from .harnesses import REVIEWER_HARNESSES
from .role_chain import RoleEntry, chain_of, role_entries

#: Sonnet-tier primary (design decision #3). Tracks ``config._DEFAULT_CLAUDE_MODEL``;
#: spelled here to keep this module importable without the config monolith.
FOREMAN_DEFAULT_MODEL = "claude-sonnet-5-5"

#: Default tool allowlist for the foreman's read-only model call.
FOREMAN_DEFAULT_TOOLS = "Read,Grep,Glob"


@dataclass(frozen=True)
class ForemanConfig:
    """Fleet-wide foreman settings under the ``foreman:`` section.

    ``enabled`` is the kill switch (default on): an empty ``foremen/``
    directory is already a no-op, so the switch exists for operators who want
    the layer off entirely. ``harness`` must be one of
    ``REVIEWER_HARNESSES``; ``fallbacks`` is the same ``RoleEntry`` chain the
    worker/reviewer roles use, minus ``effort`` (a foreman call pins no
    effort knob).
    """

    enabled: Annotated[bool, Typed] = True
    harness: Annotated[str, Typed, NotNull, OneOf(*sorted(REVIEWER_HARNESSES))] = "claude-code"
    model: Annotated[str, Typed, NotNull] = FOREMAN_DEFAULT_MODEL
    fallbacks: Annotated[
        tuple[RoleEntry, ...],
        NullIsDefault,
        role_entries(REVIEWER_HARNESSES, allow_effort=False),
    ] = ()
    tools: Annotated[str, Typed, NotNull, NonEmpty] = FOREMAN_DEFAULT_TOOLS
    max_model_turns: Annotated[int, Typed, Positive] = 40
    pass_interval_minutes: Annotated[int, Typed, Positive] = 5
    # Wall-clock ceiling on one Decide model call (issue #3120) -- the
    # ``run_captured`` timeout_seconds of the blocking launch. The launch is
    # synchronous, so this is also the worst-case stall a single Decide step
    # adds to a Turn: with the default 40 max_model_turns a slow model could
    # otherwise outlast a pass_interval window.
    decide_timeout_seconds: Annotated[int, Typed, Positive] = 1800
    chain = property(chain_of)
