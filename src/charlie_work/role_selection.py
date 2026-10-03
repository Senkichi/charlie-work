"""Launch-time role-chain selection (issue #2086).

:func:`select_role_entry` is the pure core: given a role chain and the
fleet-scoped quota ledger, it returns the first entry whose
``(harness, model)`` is not restricted at ``now``, plus the entries it
skipped. Every worker and reviewer launch site calls it through
:func:`select_for_launch`.

A chain of length 1 (no ``fallbacks:`` configured) short-circuits to its
primary without reading the ledger -- the pre-#2086 behavior, byte for byte.

The per-repo throttle windows (``state["throttled_until"]`` for workers,
``state["reviewer_quota"]`` for reviewers) keep their meaning of "this repo
may not launch the role". For a chained role, a per-repo window that the
ledger fully explains (it ends no later than the latest ledger restriction on
some chain entry) is **covered**: selection already routed around it, so the
launch proceeds on the selected entry (:func:`window_covered`). Explained means
the stored window is quota-derived (it carries the ``reason`` /
``adapter_kind`` provenance the writers stamp) **and** attributable to an
entry selection skipped on that adapter -- never merely "shorter than some
ledger restriction somewhere in the chain".

Issue #2279: a fallback's quota death can leave a per-repo window that
outlives the *primary's* restriction -- selection then returns the primary
with nothing skipped, and adapter-level provenance attributes the window to
nobody. The throttle writers therefore stamp the window with the dead
session's own role-chain ``(harness, model)`` (``throttle_harness`` /
``throttle_model``, or ``harness`` / ``model`` on ``reviewer_quota``); a
stamped window is covered when the stamp differs from the selected entry and
the ledger restricts the stamped entry at least as long. A per-repo window
the ledger does not explain -- an operator hold, a window from a session
launched before the ledger existed, an unstamped window -- still blocks, so
nothing that blocked before this module can silently stop blocking. When
every entry is restricted, selection returns ``None`` and the caller defers
exactly as the per-repo window always did.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from . import role_quota_ledger
from .harnesses import HARNESS_REGISTRY
from .iso_timestamp import parse_iso_timestamp
from .role_chain import RoleEntry
from .write_gate import WriteGate, require_write_gate

logger = logging.getLogger(__name__)

EVENT_ROLE_FALLBACK_SELECTED = "role_fallback_selected"

# Shared immutable default for ``RoleSelection.restrictions``.
_EMPTY_RESTRICTIONS: Mapping[tuple[str, str], datetime] = MappingProxyType({})


def _format_z(moment: datetime) -> str:
    aware = moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    return aware.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SkippedEntry:
    index: int
    entry: RoleEntry
    until: datetime

    def to_payload(self) -> dict[str, Any]:
        return {**self.entry.to_payload(), "index": self.index, "until": _format_z(self.until)}


@dataclass(frozen=True)
class RoleSelection:
    """The outcome of one selection over a role chain.

    ``entry`` is ``None`` when every entry is restricted. ``skipped`` holds the
    restricted entries ahead of the selected one; :func:`window_covered`
    attributes a per-repo window to one of them.

    ``restrictions`` is the ledger snapshot the selection ran against
    (issue #2279): a window *stamped* with the ``(harness, model)`` whose
    death produced it is attributed even when selection skipped nothing --
    e.g. a fallback's quota window outliving the recovered primary. It is
    excluded from equality and hashing so a hand-built selection in a test
    compares equal to a ledger-read one.
    """

    chain: tuple[RoleEntry, ...]
    entry: RoleEntry | None
    index: int | None
    skipped: tuple[SkippedEntry, ...]
    restrictions: Mapping[tuple[str, str], datetime] = field(
        default=_EMPTY_RESTRICTIONS, compare=False
    )

    @property
    def is_fallback(self) -> bool:
        return self.index is not None and self.index > 0

    @property
    def exhausted(self) -> bool:
        return self.entry is None

    @property
    def earliest_until(self) -> datetime | None:
        """When the first restricted entry frees up (the all-exhausted retry time)."""
        return min((item.until for item in self.skipped), default=None)

    def report_fields(self) -> dict[str, Any]:
        """Event/deferral payload fields describing this selection."""
        fields: dict[str, Any] = {
            "chain_length": len(self.chain),
            "chain_index": self.index,
            "skipped": [item.to_payload() for item in self.skipped],
        }
        if self.entry is not None:
            fields["harness"] = self.entry.harness
            fields["model"] = self.entry.model
        if self.exhausted and self.earliest_until is not None:
            fields["chain_retry_at"] = _format_z(self.earliest_until)
        return fields

    def chain_report_fields(self) -> dict[str, Any]:
        """:meth:`report_fields` for a chained role; ``{}`` for a length-1 chain.

        Keeps every pre-#2086 payload byte-identical when no ``fallbacks`` are
        configured.
        """
        return self.report_fields() if len(self.chain) > 1 else {}


def select_role_entry(
    chain: Sequence[RoleEntry],
    ledger: Mapping[tuple[str, str], datetime],
    now: datetime,
) -> tuple[RoleEntry | None, tuple[SkippedEntry, ...]]:
    """First chain entry not restricted at ``now``, and the restricted ones before it.

    Pure. A ledger ``until`` at or before ``now`` has expired and restricts
    nothing. Returns ``(None, skipped)`` when every entry is restricted.
    """
    skipped: list[SkippedEntry] = []
    for index, entry in enumerate(chain):
        until = ledger.get(entry.key)
        if until is not None and until > now:
            skipped.append(SkippedEntry(index=index, entry=entry, until=until))
            continue
        return entry, tuple(skipped)
    return None, tuple(skipped)


def build_selection(
    chain: Sequence[RoleEntry], ledger: Mapping[tuple[str, str], datetime], now: datetime
) -> RoleSelection:
    chain_tuple = tuple(chain)
    entry, skipped = select_role_entry(chain_tuple, ledger, now)
    return RoleSelection(
        chain=chain_tuple,
        entry=entry,
        index=None if entry is None else chain_tuple.index(entry),
        skipped=skipped,
        restrictions=MappingProxyType(dict(ledger)),
    )


def select_for_launch(chain: Sequence[RoleEntry], now: datetime | None = None) -> RoleSelection:
    """Selection for a real launch: reads the fleet ledger for chains longer than 1.

    Never raises -- ``load_restrictions`` degrades to an empty ledger.
    """
    chain_tuple = tuple(chain)
    if len(chain_tuple) <= 1:
        return RoleSelection(chain=chain_tuple, entry=chain_tuple[0], index=0, skipped=())
    resolved_now = now if now is not None else datetime.now(UTC)
    return build_selection(chain_tuple, role_quota_ledger.load_restrictions(), resolved_now)


def window_covered(
    per_repo_until: Any,
    selection: RoleSelection,
    *,
    reason: str | None,
    adapter_kind: str | None,
    harness: str | None = None,
    model: str | None = None,
) -> bool:
    """True when the per-repo throttle window is explained by a restricted chain entry.

    Covered only when all of these hold; anything else (length-1 chains,
    operator holds, windows from unstamped pre-ledger sessions, a
    ``provider_auth`` cooldown, a window on an adapter no attributable entry
    uses, a window outlasting the attributable entry's restriction) is not,
    so the per-repo gate still blocks:

    * the role is chained and selection found an entry to launch on;
    * the window is quota-derived: ``reason`` is one of the failure kinds the
      ledger restricts on (``role_quota_ledger.RESTRICTING_FAILURE_KINDS``);
    * some **skipped** entry runs on ``adapter_kind`` and its ledger
      restriction lasts at least as long as ``per_repo_until`` -- or, when
      the window is stamped with the ``(harness, model)`` whose death
      produced it (issue #2279), the stamp differs from the selected entry,
      runs on ``adapter_kind``, and the ledger restricts it at least as long.

    The stamped-entry rule exists because selection can attribute nothing
    once the dying fallback's window outlives the *primary's* restriction:
    ``skipped`` is empty and adapter-level provenance cannot tell the two
    models apart. A stamp equal to the selected entry is never covered --
    an entry's own death window keeps blocking it -- and a stamp the ledger
    does not restrict, or restricts for less than the window, explains only
    part of the hold, so the window blocks.
    """
    if len(selection.chain) <= 1 or selection.entry is None:
        return False
    if reason not in role_quota_ledger.RESTRICTING_FAILURE_KINDS or adapter_kind is None:
        return False
    until = parse_iso_timestamp(per_repo_until)
    if until is None:
        return False
    until = until.replace(microsecond=0)  # the ledger stores whole seconds
    for item in selection.skipped:
        capabilities = HARNESS_REGISTRY.get(item.entry.harness)
        if capabilities is not None and capabilities.adapter_kind == adapter_kind:
            if until <= item.until:
                return True
    if isinstance(harness, str) and isinstance(model, str):
        capabilities = HARNESS_REGISTRY.get(harness)
        if (
            capabilities is not None
            and capabilities.adapter_kind == adapter_kind
            and (harness, model) != selection.entry.key
        ):
            stamped_until = selection.restrictions.get((harness, model))
            if stamped_until is not None and until <= stamped_until:
                return True
    return False


def reviewer_window_covered(state: Mapping[str, Any], selection: RoleSelection) -> bool:
    """:func:`window_covered` for the state's ``reviewer_quota`` record."""
    quota = state.get("reviewer_quota") or {}
    return window_covered(
        quota.get("throttled_until"),
        selection,
        reason=quota.get("reason"),
        adapter_kind=quota.get("adapter_kind"),
        harness=quota.get("harness"),
        model=quota.get("model"),
    )


def resolve_reviewer(config: Any) -> tuple[RoleSelection, Any]:
    """Select the reviewer chain entry and the config derived for it (issue #2086)."""
    selection = select_for_launch(config.reviewer.chain)
    return selection, reviewer_config_for(config, selection)


def selection_adapter_kind(selection: RoleSelection) -> str | None:
    """The ``WorkerView.adapter_kind`` of the selected entry's harness, if any."""
    if selection.entry is None:
        return None
    capabilities = HARNESS_REGISTRY.get(selection.entry.harness)
    return None if capabilities is None else capabilities.adapter_kind


# --- derived per-launch config ------------------------------------------------


def worker_config_for(config: Any, selection: RoleSelection) -> Any:
    """``config`` with ``worker`` pointed at the selected entry (identity at index 0)."""
    if not selection.is_fallback or selection.entry is None:
        return config
    entry = selection.entry
    return replace(config, worker=replace(config.worker, harness=entry.harness, model=entry.model))


def reviewer_config_for(config: Any, selection: RoleSelection) -> Any:
    """``config`` with ``reviewer`` pointed at the selected entry (identity at index 0).

    A fallback entry carries its own ``effort``; the primary's effort A/B
    experiment is a property of the primary, so it is switched off for a
    fallback launch rather than silently applied to a different model.
    """
    if not selection.is_fallback or selection.entry is None:
        return config
    entry = selection.entry
    reviewer = replace(
        config.reviewer,
        harness=entry.harness,
        model=entry.model,
        effort=entry.effort,
        effort_experiment_fraction=0.0,
    )
    return replace(config, reviewer=reviewer)


def worker_settings_for(app: Any, selection: RoleSelection) -> Any:
    """``AdapterSettings`` for a worker launch on the selected entry.

    Index 0 is ``app._adapter_settings()`` unchanged. A fallback reuses the
    per-harness venv/env resolution for the entry's harness and pins the
    entry's model through both carriers the adapters read (``worker_model``
    for devin-shell, ``config.worker.model`` for claude-code).
    """
    if not selection.is_fallback or selection.entry is None:
        return app._adapter_settings()
    entry = selection.entry
    base = app._adapter_settings(adapter=entry.harness)
    return replace(
        base,
        worker_model=entry.model,
        config=worker_config_for(app.config, selection),
        role="worker",
    )


# --- post-launch bookkeeping ---------------------------------------------------


def stamp_launch(
    sessions_dir: Path | None, number: int, role: str, selection: RoleSelection
) -> bool:
    """Stamp the selected entry onto the launched session's sidecar. Never raises."""
    if sessions_dir is None or selection.entry is None or selection.index is None:
        return False
    entry = selection.entry
    sidecar = role_quota_ledger.sidecar_path_for(Path(sessions_dir), entry.harness, number)
    stamp = role_quota_ledger.session_stamp(role, entry.harness, entry.model, selection.index)
    return role_quota_ledger.stamp_session(sidecar, stamp)


def emit_fallback_selected(
    write_gate: WriteGate,
    *,
    role: str,
    selection: RoleSelection,
    numbers: Sequence[int],
) -> None:
    """``role_fallback_selected`` for a launch past the primary. Never raises.

    Routed through the caller's ``WriteGate`` (issue #1264): a dry-run pass
    emits nothing, and the gate carries the state path and repo.
    """
    if not selection.is_fallback:
        return
    require_write_gate(write_gate)
    try:
        write_gate.log_event(
            kind="role_fallback_selected",  # event-consumer: audit-only -- the actionable state is the fleet quota ledger selection already acted on; this is the per-launch record of which chain entries were skipped and until when (issue #2086)
            payload={"role": role, "numbers": list(numbers), **selection.report_fields()},
        )
    except Exception as exc:  # noqa: BLE001 - instrumentation must never block a launch
        logger.warning("role_fallback_selected event failed: %s", exc)


def after_review_launch(
    reviews_dir: Path | None,
    write_gate: WriteGate,
    selection: RoleSelection,
    launched_prs: Sequence[int],
) -> None:
    """Stamp each launched reviewer session and emit the fallback event. Never raises."""
    for pr_number in launched_prs:
        stamp_launch(reviews_dir, pr_number, "reviewer", selection)
    if launched_prs:
        emit_fallback_selected(
            write_gate, role="reviewer", selection=selection, numbers=launched_prs
        )


def record_launch_quota_hit(
    selection: RoleSelection, until: Any, *, source: str = "launch_quota_hit"
) -> bool:
    """A reviewer launch hit the provider quota: restrict the entry it launched on."""
    entry = selection.entry
    if entry is None:
        return False
    return role_quota_ledger.record_restriction(
        entry.harness, entry.model, until, reason="quota_exhausted", source=source, role="reviewer"
    )


def record_error_quota_hit(
    selection: RoleSelection, error_text: str | None, config: Any, *, source: str
) -> bool:
    """A launch error matched the quota markers, with no per-repo backoff record.

    The local review lane has no ``reviewer_quota`` gate; its launch-time
    quota hit still teaches the fleet ledger. The window is the provider's
    stated reset (plus the resume margin) when the error names one, else
    ``review_dispatch.quota_reset_hours`` -- the same targets the remote
    lane's backoff uses.
    """
    from datetime import timedelta

    from .throttle_signatures import parse_reset_clock_time

    now = datetime.now(UTC)
    reset_at = parse_reset_clock_time(error_text, now) if error_text else None
    if reset_at is not None:
        until = reset_at + timedelta(seconds=config.runtime.throttle_resume_margin_s)
    else:
        until = now + timedelta(hours=config.review_dispatch.quota_reset_hours)
    return record_launch_quota_hit(selection, until, source=source)
