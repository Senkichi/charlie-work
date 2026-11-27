"""The deterministic Collect step of a foreman Turn (issue #3118, design §8.1).

Collect assembles everything the Decide step's model call will see —
``briefing.md`` — from local state only: the repo's status snapshot, its
``events.db``, the foreman store (charter, instructions, memory, journal,
items, inbox, signals, prior Turn records) and the fleet's capacity files
(pause flag, quota ledger). It makes no GitHub call and persists nothing;
the Briefing it returns carries the new cursor values so the caller can
persist them after a successful Apply.

The module owns three things:

* :class:`Foreman` / :func:`load_foreman` — the resolved context one Collect
  runs against (charter, runtime state, the repo the charter names and the
  config the briefing needs).
* :func:`lifecycle_fingerprint` — the wake predicate's cheap change detector:
  one hash over the owned issue set's lifecycle labels, so a pass can tell
  "something the foreman cares about moved" without building a briefing.
* :func:`build_briefing` — assembles the :class:`Briefing` whose ``render()``
  produces ``briefing.md``. The document model lives in ``briefing.py`` and
  the per-section builders in ``briefing_sections.py`` — this module keeps
  the read order, the freshness gate and the cursor arithmetic. Instruction
  sections (charter, instructions, inbox — design §12) come first; everything
  else lands under a heading that says it is data and must not be followed
  as instruction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import layout
from ..config import LabelConfig, OrchestratorConfig
from ..config_validation import ConfigError
from ..github import label_names
from ..global_config import load_layered_config
from ..instrumentation import query_events
from ..iso_timestamp import parse_iso_timestamp
from ..labels import _label_state_map, _state_for_labels
from ..review_decision import resolve_decision_payload
from ..state_snapshot import read_state_snapshot as read_hub_state
from . import Result, briefing_sections
from .briefing import Briefing, OwnedIssue, Section, _fmt
from .charter import Charter, load_charter
from .journal import read_inbox
from .state import AuthoredIssue, ForemanState, read_state_snapshot
from .store import registered_repo_keys, resolve_foreman
from .tree import ensure_tree, sync_tree


@dataclass(frozen=True)
class Foreman:
    """The resolved context one Collect runs against.

    Built once per pass by :func:`load_foreman`; ``build_briefing`` and
    ``lifecycle_fingerprint`` take it as their ``foreman`` argument.
    ``state_root`` is the repo's runtime state dir (``state_dir`` in the
    fleet registry) — where ``status-snapshot.json``, ``state.json``,
    ``events.db`` and ``prs/pr-N/review-decision.json`` all live.
    """

    name: str
    root: Path  # foremen/<name>
    fleet_dir: Path
    charter: Charter
    state: ForemanState
    repo_key: str  # fleet registry key, e.g. "owner/repo"
    repo_root: Path
    state_root: Path
    state_file: Path  # <state_root>/state.json — events.db lives beside it
    labels: LabelConfig
    full_pass_interval_seconds: int
    #: The repo's resolved layered config — Decide needs the foreman role
    #: chain and timeout, Apply the issue-client settings (issue #3122).
    config: OrchestratorConfig


def _registry_entry(fleet_dir: Path, key: str) -> dict[str, Any] | None:
    """The ``repos[key]`` entry of ``fleet.json``, or None."""
    try:
        raw: Any = json.loads(
            (Path(fleet_dir) / layout.FLEET_REGISTRY_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    repos = raw.get("repos") if isinstance(raw, dict) else None
    entry = repos.get(key) if isinstance(repos, dict) else None
    return entry if isinstance(entry, dict) else None


def load_foreman(fleet_dir: Path, name: str) -> Result[Foreman]:
    """Resolve ``foremen/<name>`` into the context Collect runs against.

    Every failure — an unknown foreman, an invalid charter, a registry entry
    that lost its ``state_dir``, a config that fails validation — is a
    ``Result.error`` naming what broke, per the package's failures-as-values
    rule.
    """
    root_result = resolve_foreman(fleet_dir, name)
    if not root_result.ok:
        return Result(error=root_result.error)
    root = root_result.value
    keys = registered_repo_keys(fleet_dir)
    charter_result = load_charter(root, registered_repos=keys)
    if not charter_result.ok:
        return Result(error=charter_result.error)
    charter = charter_result.value
    repo_key = charter.repos[0]
    entry = _registry_entry(fleet_dir, repo_key)
    if entry is None:
        return Result(error=f"{repo_key}: no fleet registry entry at {fleet_dir}")
    repo_root_raw, state_root_raw = entry.get("repo_root"), entry.get("state_dir")
    if not isinstance(repo_root_raw, str) or not isinstance(state_root_raw, str):
        return Result(error=f"{repo_key}: registry entry lacks repo_root/state_dir")
    repo_root, state_root = Path(repo_root_raw), Path(state_root_raw)
    config_path = entry.get("config_path")
    explicit = (
        Path(config_path) if isinstance(config_path, str) and Path(config_path).is_file() else None
    )
    try:
        config = load_layered_config(repo_root, explicit, fleet_dir_override=str(fleet_dir))
    except (ConfigError, OSError) as exc:
        return Result(error=f"{repo_key}: config failed to load: {exc}")
    return Result(
        value=Foreman(
            name=name,
            root=root,
            fleet_dir=Path(fleet_dir),
            charter=charter,
            state=read_state_snapshot(root),
            repo_key=repo_key,
            repo_root=repo_root,
            state_root=state_root,
            state_file=layout.state_file_path(state_root),
            labels=config.labels,
            full_pass_interval_seconds=config.supervisor.full_pass_interval_seconds,
            config=config,
        )
    )


# --- snapshot read ----------------------------------------------------------


def _read_snapshot(state_root: Path) -> tuple[datetime | None, dict[str, Any] | None]:
    """Lock-free read of ``status-snapshot.json`` as (written_at, data).

    ``(None, None)`` covers every unusable shape — missing file, torn write,
    missing timestamp — the same "reads the last committed snapshot or
    nothing" contract the hub's own reader has. Collect never falls back to
    a live GitHub read (design §14).
    """
    try:
        envelope: Any = json.loads(
            layout.status_snapshot_path(state_root).read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        return None, None
    if not isinstance(envelope, dict):
        return None, None
    written = parse_iso_timestamp(envelope.get("snapshot_written_at"))
    data = envelope.get("data")
    if not isinstance(data, dict):
        return None, None
    return written, data


def _owned_from_snapshot(foreman: Foreman, data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Snapshot ``issues[]`` entries carrying this foreman's ownership label."""
    prefix = foreman.labels.foreman_prefix
    if not prefix:
        return {}
    owned_label = f"{prefix}{foreman.name}"
    owned: dict[int, dict[str, Any]] = {}
    issues = data.get("issues")
    if not isinstance(issues, list):
        return owned
    for entry in issues:
        if not isinstance(entry, dict):
            continue
        number = entry.get("number")
        if isinstance(number, int) and owned_label in label_names(entry):
            owned[number] = entry
    return owned


def _lifecycle_labels(foreman: Foreman, names: set[str]) -> tuple[str, ...]:
    """The label set's lifecycle labels, sorted — the fingerprint's per-issue tuple.

    Lifecycle labels are the ones the label state machine manages
    (``_label_state_map``'s key set) plus ``prose_only_deps`` and
    ``merge_hold`` — a disposition or an operator hold is lifecycle-relevant
    too. Everything else (priority, model tier, host, the ownership label
    itself) is not a lifecycle change and must not move the fingerprint.
    """
    managed = set(_label_state_map(foreman.labels))
    managed.add(foreman.labels.prose_only_deps)
    managed.add(foreman.labels.merge_hold)
    return tuple(sorted(names & managed))


def lifecycle_fingerprint(foreman: Foreman) -> str:
    """The wake predicate's change detector over the owned issue set.

    One hash of the sorted (number, lifecycle labels, present|absent) tuple
    per owned issue — owned now by the snapshot label, or recorded in
    ``state.issues`` at the last Turn and gone since (design §7 point 1).
    A missing or unreadable snapshot computes over the recorded register
    alone — every entry absent — which itself reads as a change worth a
    Turn's Collect erroring on (never a silent skip, and never a live
    GitHub read).
    """
    _written, data = _read_snapshot(foreman.state_root)
    present = _owned_from_snapshot(foreman, data) if data is not None else {}
    elements = []
    for number in sorted(set(present) | set(foreman.state.issues)):
        entry = present.get(number)
        if entry is None:
            elements.append((number, (), "absent"))
        else:
            elements.append((number, _lifecycle_labels(foreman, label_names(entry)), "present"))
    return hashlib.sha256(repr(tuple(elements)).encode("utf-8")).hexdigest()


def _linked_pr_map(data: dict[str, Any]) -> dict[int, int]:
    """``issue_number -> pr_number`` over the snapshot's linked-PR summaries."""
    linked: dict[int, int] = {}
    prs = data.get("prs")
    if not isinstance(prs, list):
        return linked
    for entry in prs:
        if not isinstance(entry, dict):
            continue
        issue_number, pr_number = entry.get("issue_number"), entry.get("number")
        if isinstance(issue_number, int) and isinstance(pr_number, int):
            linked[issue_number] = pr_number
    return linked


def _stage_seconds(state_file: Path, issue_number: int, now: datetime) -> float | None:
    """Time in the issue's current lifecycle state, from events.db.

    The stage opens at the last ``lifecycle_transition`` (or ``ready_observed``
    for an issue that only ever became Ready). None when the log records no
    lifecycle history for the issue — an unarmed tracked issue included.
    """
    opened: datetime | None = None
    for event in query_events(state_file, issue_number=issue_number):
        if event.get("kind") in ("lifecycle_transition", "ready_observed"):
            ts = parse_iso_timestamp(event.get("ts"))
            if ts is not None:
                opened = ts
    if opened is None:
        return None
    return max(0.0, (now - opened).total_seconds())


def _collect_owned(
    foreman: Foreman,
    present: dict[int, dict[str, Any]],
    linked: dict[int, int],
    issue_state: dict[str, Any],
    now: datetime,
) -> tuple[OwnedIssue, ...]:
    """The owned-issue register — present snapshot entries plus absent records."""
    owned: list[OwnedIssue] = []
    union = sorted(set(present) | set(foreman.state.issues))
    for number in union:
        prior = foreman.state.issues.get(number)
        body_hash = prior.body_hash if prior is not None else ""
        entry = present.get(number)
        names = label_names(entry) if entry is not None else set()
        lifecycle = _lifecycle_labels(foreman, names)
        state_name = _state_for_labels(foreman.labels, lifecycle)
        state_entry = issue_state.get(str(number))
        if not isinstance(state_entry, dict):
            state_entry = {}
        reasons: list[str] = []
        seen = state_entry.get("escalation_reasons_seen")
        if isinstance(seen, list):
            reasons.extend(str(code) for code in seen)
        latest_reason = state_entry.get("escalation_reason")
        if isinstance(latest_reason, str) and latest_reason not in reasons:
            reasons.append(latest_reason)
        pr_number = linked.get(number)
        verdict: str | None = None
        if pr_number is not None:
            payload = resolve_decision_payload(foreman.state_root / "prs" / f"pr-{pr_number}")
            decision = payload.get("decision")
            verdict = decision if isinstance(decision, str) and decision != "missing" else None
        if entry is None:
            owned.append(
                OwnedIssue(
                    number=number,
                    title="",
                    authored=bool(body_hash),
                    body_hash=body_hash,
                    present=False,
                    lifecycle_labels=(),
                    lifecycle_state="absent",
                    linked_pr=None,
                    verdict=None,
                    escalation_reasons=tuple(reasons),
                    stage_seconds=None,
                )
            )
            continue
        owned.append(
            OwnedIssue(
                number=number,
                title=str(entry.get("title") or ""),
                authored=bool(body_hash),
                body_hash=body_hash,
                present=True,
                lifecycle_labels=lifecycle,
                lifecycle_state=state_name or "unarmed",
                linked_pr=pr_number,
                verdict=verdict,
                escalation_reasons=tuple(reasons),
                stage_seconds=_stage_seconds(foreman.state_file, number, now),
            )
        )
    return tuple(owned)


def _sync_tree(foreman: Foreman) -> Result[Path]:
    """Move ``tree/`` to the default branch tip — the first Collect step.

    ``sync_tree`` is the specified call; the one gap it cannot cover is the
    first-ever Turn, where ``tree/`` does not exist yet — that falls back to
    ``ensure_tree``, which lands the same detached tip. Any other failure is
    an error value (a dirty tree is never clobbered).
    """
    synced = sync_tree(foreman.root, foreman.repo_root)
    if synced.ok:
        return synced
    if "not a registered worktree" in (synced.error or ""):
        return ensure_tree(foreman.root, foreman.repo_root)
    return synced


def build_briefing(foreman: Foreman, now: datetime) -> Result[Briefing]:
    """Assemble the Turn's briefing from local state only — no GitHub call.

    Order follows the issue: ``tree/`` sync first (a failure is an error
    value), then the snapshot freshness gate — a missing snapshot or one
    older than twice the fleet's full pass interval is an error value
    naming the repo, never a live-read fallback.
    """
    synced = _sync_tree(foreman)
    if not synced.ok:
        return Result(error=synced.error)
    written_at, data = _read_snapshot(foreman.state_root)
    if data is None or written_at is None:
        return Result(
            error=(
                f"{foreman.repo_key}: status snapshot missing or unreadable — "
                "Collect never falls back to a live read"
            )
        )
    bound = 2 * foreman.full_pass_interval_seconds
    age = (now - written_at).total_seconds()
    if age > bound:
        return Result(
            error=(
                f"{foreman.repo_key}: status snapshot stale — written {age:.0f}s ago, "
                f"bound is 2x the full pass interval ({bound}s)"
            )
        )
    charter_section = briefing_sections._charter_section(foreman)
    if not charter_section.ok:
        return Result(error=charter_section.error)
    instructions_section = briefing_sections._instructions_section(foreman)
    if not instructions_section.ok:
        return Result(error=instructions_section.error)

    hub_state = read_hub_state(foreman.state_file)
    issue_state = (hub_state or {}).get("issues", {})
    if not isinstance(issue_state, dict):
        issue_state = {}
    present = _owned_from_snapshot(foreman, data)
    linked = _linked_pr_map(data)
    owned = _collect_owned(foreman, present, linked, issue_state, now)

    unread_inbox = read_inbox(foreman.root, after_id=foreman.state.inbox_cursor)
    signals_section, unread_signals = briefing_sections._signals_section(foreman)

    delta = query_events(foreman.state_file, after_id=foreman.state.event_cursor)
    event_cursor = max(
        (event.get("id") for event in delta if isinstance(event.get("id"), int)),
        default=foreman.state.event_cursor,
    )
    inbox_cursor = max((msg.id for msg in unread_inbox), default=foreman.state.inbox_cursor)
    signal_cursor = max(
        (signal.id for signal in unread_signals), default=foreman.state.signal_cursor
    )

    instruction_sections = (
        charter_section.value,
        instructions_section.value,
        briefing_sections._inbox_section(foreman, unread_inbox),
    )
    data_sections = (
        briefing_sections._foreman_state_section(foreman),
        briefing_sections._memory_section(foreman),
        briefing_sections._journal_section(foreman),
        briefing_sections._owned_section(owned),
        briefing_sections._findings_section(foreman, owned),
        briefing_sections._event_delta_section(
            foreman,
            delta,
            owned_numbers={issue.number for issue in owned},
            pr_numbers={issue.linked_pr for issue in owned if issue.linked_pr is not None},
        ),
        briefing_sections._capacity_section(foreman, delta, now),
        briefing_sections._items_section(foreman, now, unread_inbox),
        signals_section,
        briefing_sections._refused_section(foreman),
        briefing_sections._bounds_section(foreman, owned, now),
    )
    return Result(
        value=Briefing(
            foreman=foreman.name,
            repo=foreman.repo_key,
            generated_at=_fmt(now),
            snapshot_written_at=_fmt(written_at),
            signal_marked=bool(unread_signals),
            lifecycle_fingerprint=lifecycle_fingerprint(foreman),
            event_cursor=event_cursor,
            inbox_cursor=inbox_cursor,
            signal_cursor=signal_cursor,
            issue_register=tuple(
                AuthoredIssue(number=issue.number, body_hash=issue.body_hash) for issue in owned
            ),
            instruction_sections=instruction_sections,
            data_sections=data_sections,
        )
    )


__all__ = [
    "Briefing",
    "Foreman",
    "OwnedIssue",
    "Section",
    "build_briefing",
    "lifecycle_fingerprint",
    "load_foreman",
]
