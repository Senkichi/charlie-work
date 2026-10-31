"""Generic per-issue model tier: the ``model:<tier>`` label (TIS-CW-6).

Any issue may carry ``model:<tier>``; nothing special-cases who filed it. The
worker launch gate (``worker_launch_gate._launch_workers``, the one place every
worker lane launches) splits its batch by tier and launches each tier on the
first worker chain entry of that model family that the fleet quota ledger does
not restrict. Families come from the configured model ids themselves --
``claude-opus-5-5`` is in family ``opus`` (and ``claude``) -- so no model list
lives here.

When the chain cannot serve a tier (no entry of the family, every such entry
restricted, two tiers on one issue, or -- issue #2757 -- the entry's harness is
at its ``fleet.max_concurrent_sessions_by_harness`` ceiling) the issue launches on the normal chain,
exactly as an unlabelled one would, and ``worker_model_tier_fallback`` records
why. An unlabelled issue never reaches this module's selection: it launches on
the permit's selection, the fleet's default tier.

The label prefix is ``LabelConfig.model_tier_prefix``; an empty prefix turns the
routing off.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from . import role_quota_ledger
from .role_chain import RoleEntry
from .role_selection import RoleSelection, harness_auth_enabled, ledger_for_chain
from .write_gate import WriteGate, require_write_gate

logger = logging.getLogger(__name__)

EVENT_TIER_SELECTED = "worker_model_tier_selected"
EVENT_TIER_FALLBACK = "worker_model_tier_fallback"

# ``worker_model_tier_fallback`` reasons.
NO_ENTRY = "no_entry"
RESTRICTED = "restricted"
CONFLICTING = "conflicting_labels"
# Issue #2757: the tier's entry runs on a harness whose
# ``fleet.max_concurrent_sessions_by_harness`` ceiling has no room for the group.
HARNESS_CAP = "harness_cap"

_TOKEN = re.compile(r"[^a-z0-9]+")


def model_tiers(labels: Iterable[str], prefix: str) -> tuple[str, ...]:
    """The distinct tiers named by ``<prefix><tier>`` labels, lower-cased and sorted.

    Empty when the issue is unlabelled or ``prefix`` is empty (routing off).
    """
    if not prefix:
        return ()
    folded = prefix.lower()
    tiers = {
        label[len(prefix) :].strip().lower()
        for label in labels
        if label.lower().startswith(folded)
    }
    return tuple(sorted(tier for tier in tiers if tier))


def model_in_tier(model: str, tier: str) -> bool:
    """True when ``tier`` names ``model``'s family: the whole id or one of its tokens.

    Tokens split on anything that is not a letter or digit, so ``opus`` matches
    ``claude-opus-5-5`` but ``code`` does not match ``gpt-5-codex``. An empty
    model id (the harness default) belongs to no family.
    """
    wanted = tier.strip().lower()
    name = model.strip().lower()
    if not wanted or not name:
        return False
    return name == wanted or wanted in _TOKEN.split(name)


def select_tier_entry(
    chain: Sequence[RoleEntry],
    tier: str,
    ledger: Mapping[tuple[str, str], datetime],
    now: datetime,
) -> tuple[RoleSelection | None, str | None]:
    """The first unrestricted chain entry in ``tier``'s family, or why there is none.

    Pure. ``(selection, None)`` on a match; ``(None, NO_ENTRY)`` when no entry
    is in the family; ``(None, RESTRICTED)`` when every one of them is
    restricted at ``now``. A ledger ``until`` at or before ``now`` has expired.
    """
    chain_tuple = tuple(chain)
    family = [
        (index, entry)
        for index, entry in enumerate(chain_tuple)
        if model_in_tier(entry.model, tier)
    ]
    if not family:
        return None, NO_ENTRY
    for index, entry in family:
        until, _harness_wide = role_quota_ledger.effective_restriction(
            ledger, entry.harness, entry.model, now
        )
        if until is None:
            selection = RoleSelection(
                chain=chain_tuple,
                entry=entry,
                index=index,
                skipped=(),
                restrictions=MappingProxyType(dict(ledger)),
            )
            return selection, None
    return None, RESTRICTED


def select_tier_for_launch(
    chain: Sequence[RoleEntry],
    tier: str,
    now: datetime | None = None,
    *,
    harness_auth: bool = True,
) -> tuple[RoleSelection | None, str | None]:
    """:func:`select_tier_entry` against the fleet quota ledger. Never raises.

    The ledger is the one ``role_selection.select_for_launch`` reads
    (:func:`role_selection.ledger_for_chain`): a length-1 chain sees only the
    harness-wide blocks, so a tier never launches on a harness whose
    credential is dead (issue #2810).
    """
    chain_tuple = tuple(chain)
    ledger = ledger_for_chain(chain_tuple, harness_auth=harness_auth)
    return select_tier_entry(chain_tuple, tier, ledger, now or datetime.now(UTC))


def select_for_config(
    chain: Sequence[RoleEntry], tier: str, config: Any, now: datetime | None = None
) -> tuple[RoleSelection | None, str | None]:
    """:func:`select_tier_for_launch` honoring ``config``'s harness-auth kill switch."""
    return select_tier_for_launch(chain, tier, now, harness_auth=harness_auth_enabled(config))


def emit_tier_selected(
    write_gate: WriteGate, *, tier: str, selection: RoleSelection, numbers: Sequence[int]
) -> None:
    """``worker_model_tier_selected`` for launches a ``model:<tier>`` label routed. Never raises."""
    require_write_gate(write_gate)
    try:
        write_gate.log_event(
            kind="worker_model_tier_selected",  # event-consumer: audit-only -- the routing already happened at launch; this is the per-launch record of which tier picked which chain entry, and the rollout step 5 canary's evidence (TIS-CW-6)
            payload={
                "role": "worker",
                "tier": tier,
                "numbers": list(numbers),
                **selection.report_fields(),
            },
        )
    except Exception as exc:  # noqa: BLE001 - instrumentation must never block a launch
        logger.warning("worker_model_tier_selected event failed: %s", exc)


def emit_tier_fallback(
    write_gate: WriteGate, *, tier: str, reason: str | None, numbers: Sequence[int]
) -> None:
    """``worker_model_tier_fallback``: labelled issues launch on the normal chain. Never raises."""
    require_write_gate(write_gate)
    try:
        write_gate.log_event(
            kind="worker_model_tier_fallback",  # event-consumer: audit-only -- the issues already launch on the normal chain; this records the tier the chain could not serve and why, at warning level so the heartbeat's warning listing surfaces a chain missing a requested family (TIS-CW-6)
            payload={"role": "worker", "tier": tier, "reason": reason, "numbers": list(numbers)},
        )
    except Exception as exc:  # noqa: BLE001 - instrumentation must never block a launch
        logger.warning("worker_model_tier_fallback event failed: %s", exc)
