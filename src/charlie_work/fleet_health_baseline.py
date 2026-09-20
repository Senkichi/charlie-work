"""Fleet health baseline sidecar: ``notify_health_state.json`` I/O and reconciliation.

Extracted from ``fleet_dispatch`` (issue #1755 round-2 review): the fleet-wide
dedup baseline that ``fleet_dispatch._build_fleet_attention_digest`` filters
persistent-health attention entries against each pass. The sidecar records the
last-emitted health per ``"adapter_kind:issue_number"`` key so a standing
ERROR/STALLED condition does not re-fire with ``previous_health: null`` every
pass (issue #554).

The load/save helpers keep the project's atomic-write invariant (temp file +
``replace()``); ``reconcile_fleet_health_baselines`` is a pure function so the
issue #817 observed-repo reconciliation and the issue #1755 registry-membership
GC can be exercised without touching the filesystem.
``_filter_fleet_health_transitions`` is the stateful wrapper that owns the
``state_lock`` window and persists the reconciled result.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from . import layout
from .fleet_paths import warn_fleet_dir_virtualization_on_write
from .notify import AttentionEntry
from .state import state_lock, utc_now

logger = logging.getLogger(__name__)


def _fleet_health_state_path(fleet_dir_override: str | None) -> Path:
    """Return the fleet-level notify health baseline sidecar path.

    This sidecar persists the last-known health per (adapter_kind, issue_number)
    so the fleet digest can emit only on real transitions instead of re-firing
    every pass with ``previous_health: null`` (issue #554). It lives in the
    fleet directory alongside ``fleet.json``.
    """
    return layout.notify_health_state_path(override=fleet_dir_override)


def _load_fleet_health_state(path: Path) -> dict[str, str]:
    """Load the fleet health baseline sidecar.

    Returns an empty dict when the file is missing or unparseable (a corrupt
    sidecar is non-fatal: the worst case is one extra transition emission on
    the next pass, which is exactly the degraded mode we are fixing away from).
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, LookupError, ValueError, OSError):
        logger.warning("Fleet health state %s unreadable; starting fresh", path)
        return {}
    issues = data.get("issues") if isinstance(data, dict) else None
    if not isinstance(issues, dict):
        return {}
    # Coerce values to str; keys are already "adapter_kind:issue_number" strings.
    return {str(k): str(v) for k, v in issues.items() if isinstance(v, str)}


def _save_fleet_health_state(path: Path, issues: dict[str, str]) -> None:
    """Atomically persist the fleet health baseline sidecar.

    Temp-file + ``replace()`` per the project's atomic-write invariant. Warns
    on fleet-dir virtualization (issue #624) but never blocks the write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    warn_fleet_dir_virtualization_on_write(path.parent, context="writing notify_health_state.json")
    payload = {"version": 1, "generated_at": utc_now(), "issues": issues}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def reconcile_fleet_health_baselines(
    baselines: dict[str, str],
    *,
    touched_keys: set[str] | frozenset[str],
    observed_repo_keys: frozenset[str] | None = None,
    registered_repo_keys: frozenset[str] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Reconcile persisted fleet-health baselines. Pure: no I/O, no logging.

    ``baselines`` is the just-updated ``"adapter_kind:issue_number" -> health``
    map (entries emitted this pass have already been written into it by the
    caller); ``touched_keys`` are the baseline keys this pass's persistent
    entries affirmed, which are never reconciled away. Returns
    ``(new_baselines, dropped_keys)``; ``dropped_keys`` lists only the keys
    removed by the registry-membership GC, sorted, so the caller can log them.
    The #817 observed-repo drops below are silent by design (a healthy
    re-check produces no event to attach a recovery entry to) and are not
    listed.

    ``observed_repo_keys`` (issue #817 item 2): any untouched persisted key
    whose ``adapter_kind`` is in this set is cleared -- its repo's lane ran to
    completion this pass, so every issue it tracks was genuinely looked at and
    the absence of an unhealthy entry means it is healthy. A repo lane that
    did not run (missing repo_root, supervisor lock held, an unhandled
    per-repo exception) is not in this set, so its keys are left untouched --
    absence of a check is not evidence of health.

    ``registered_repo_keys`` (issue #1755): the set of repo keys currently in
    the fleet registry (``fleet.json``'s ``repos`` map). The
    ``observed_repo_keys`` rule can only ever clear a key once its repo's lane
    runs to completion -- a key whose ``adapter_kind`` names a repo that is
    not, or is no longer, a registry member can never appear in any pass's
    ``observed_repo_keys`` (only selected registry entries run lanes) and
    would otherwise be retained forever: the same "latched permanently"
    defect class #817 fixed for the repo-recovers case, but for the
    repo-removed / never-registered case. An untouched key whose
    ``adapter_kind`` is repo-shaped -- every ``name_with_owner`` contains
    ``/`` (``owner/name`` from GitHub, ``local/<name>`` for local-file
    repos), while fleet-internal namespaces like ``self-deploy`` never do --
    and is absent from ``registered_repo_keys`` is dropped outright: it is
    residue, not a health signal. Keys for repos that ARE registered but did
    not run this pass are still left untouched -- registry membership does
    not weaken #817's "absence of a check is not evidence of health"
    protection.

    FAIL CLOSED: ``registered_repo_keys=None`` disables the membership GC,
    and an EMPTY set is treated identically.
    ``fleet_registry._load_registry`` collapses a missing, corrupt, or empty
    ``fleet.json`` into ``{"repos": {}}``, so a zero-repo registry cannot be
    distinguished from an unreadable one -- and must never be grounds for
    deleting every repo-shaped baseline key. Membership GC therefore requires
    positive evidence of at least one registered repo; callers that compute
    the set should pass ``the_set or None``.
    """
    new_baselines = dict(baselines)
    dropped_keys: list[str] = []
    if not observed_repo_keys and not registered_repo_keys:
        return new_baselines, dropped_keys
    for key in list(new_baselines):
        if key in touched_keys:
            continue
        adapter_kind = key.split(":", 1)[0]
        if observed_repo_keys and adapter_kind in observed_repo_keys:
            del new_baselines[key]
        elif (
            registered_repo_keys
            and "/" in adapter_kind
            and adapter_kind not in registered_repo_keys
        ):
            dropped_keys.append(key)
            del new_baselines[key]
    return new_baselines, sorted(dropped_keys)


def _filter_fleet_health_transitions(
    entries: list[AttentionEntry],
    state_file: Path,
    *,
    persistent_mask: list[bool] | None = None,
    observed_repo_keys: frozenset[str] | None = None,
    registered_repo_keys: frozenset[str] | None = None,
) -> list[AttentionEntry]:
    """Stateful filter: keep only persistent-health entries whose health changed.

    Reads the fleet health baseline sidecar (``notify_health_state.json``)
    mapping ``"adapter_kind:issue_number"`` to the last-emitted health. For
    each *persistent* entry, only keeps it when the health differs from the
    persisted baseline, setting ``previous_health`` to that prior value.
    Updates and atomically persists the new baselines.

    This is the fleet-level analogue of ``workflow._build_attention_digest``'s
    per-issue transition dedup. Without it, every fleet pass re-emits the same
    ERROR/STALLED entries with ``previous_health: null`` (issue #554).

    ``persistent_mask`` (parallel to ``entries``) marks which entries represent
    a persistent health state subject to cross-pass dedup. Entries marked
    ``False`` are occurrence-style events (review_verdict_recorded/missed,
    skipped, live_worker_redispatch_averted, operational fallback) and pass
    through unchanged every call — they never consult nor update the baseline,
    so a constant-health confirmation keeps firing as a heartbeat (PR #669
    review). When ``persistent_mask`` is ``None`` every entry is treated as
    persistent (the self-deploy ERROR/REPAIRED path, which is always
    persistent).

    Entries are processed in order so a within-pass health change (e.g.
    ERROR then OK for the same issue) emits both transitions; the persisted
    baseline ends at the final health.

    ``observed_repo_keys`` (issue #817 item 2) reconciles the baseline
    against issues that were genuinely re-checked this pass and found
    healthy. Persistent-health entries only ever exist for *unhealthy*
    observations (stalled/error/health_transition) -- a healthy issue
    produces no event at all, so without this the baseline can only ever
    move *into* an unhealthy value and never back out, latching every
    tracked issue at its first failure forever (the same defect item 1 fixes
    for self-deploy, whose producer always has a distinct success value to
    feed; issue health has no such value to hang a recovery entry on). The
    actual reconciliation is :func:`reconcile_fleet_health_baselines`; any
    *other* persisted key whose ``adapter_kind`` (the part before the first
    ``:``) is in ``observed_repo_keys`` -- meaning that repo's lane ran to
    completion this pass, so every issue it tracks was genuinely looked at --
    and that was not itself re-affirmed unhealthy in this same call is
    silently cleared, not re-emitted as a recovery entry (there is no "issue
    confirmed healthy" event to attach one to). This does not create a false
    transition; it lets the *next* unhealthy observation for that key read
    ``previous_health: null`` and emit as a fresh incident instead of being
    suppressed by a latch that could never leave its last value. A repo lane
    that did not run this pass (missing repo_root, supervisor lock held, an
    unhandled per-repo exception) is not in ``observed_repo_keys``, so its
    keys are left untouched -- absence of a check is not evidence of health.

    ``registered_repo_keys`` (issue #1755) is the set of repo keys currently
    in the fleet registry, forwarded to
    :func:`reconcile_fleet_health_baselines` for the registry-membership GC:
    ``observed_repo_keys`` alone can never clear a key whose repo is not, or
    is no longer, a registry member, so a repo-shaped ``adapter_kind`` absent
    from the set is dropped as residue. Both ``None`` and an empty set
    disable the GC -- see that function's docstring for the fail-closed
    rationale (a zero-repo registry is indistinguishable from an unreadable
    one).
    """
    emitted: list[AttentionEntry] = []
    touched_keys: set[str] = set()
    # Ensure the parent directory exists before state_lock tries to create the
    # sibling .lock file (advisory_file_lock touches it directly).
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_lock(state_file):
        baselines = _load_fleet_health_state(state_file)
        for idx, entry in enumerate(entries):
            is_persistent = persistent_mask[idx] if persistent_mask is not None else True
            if not is_persistent:
                # Occurrence-style event: emit every time and leave the
                # persisted baseline untouched (it tracks persistent health
                # only, so a one-shot confirmation must not poison it).
                emitted.append(entry)
                continue
            key = f"{entry.adapter_kind}:{entry.issue_number}"
            touched_keys.add(key)
            last = baselines.get(key)
            if last == entry.health:
                continue
            emitted.append(replace(entry, previous_health=last))
            baselines[key] = entry.health
        baselines, dropped_unregistered = reconcile_fleet_health_baselines(
            baselines,
            touched_keys=touched_keys,
            observed_repo_keys=observed_repo_keys,
            registered_repo_keys=registered_repo_keys,
        )
        if dropped_unregistered:
            logger.info(
                "fleet health baseline: dropped %d key(s) whose repo is "
                "not in the current fleet registry: %s",
                len(dropped_unregistered),
                ", ".join(dropped_unregistered),
            )
        _save_fleet_health_state(state_file, baselines)
    return emitted
