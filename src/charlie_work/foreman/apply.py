"""The Apply engine: the deterministic last step of a Turn (design §8.3, issue #3119).

The caller writes ``turns/<turn_id>/actions.json`` before Apply starts, so a
crashed Turn resumes from the same document and never asks the model again —
a resume cannot produce a different plan than the one already half-applied.
``apply_turn`` validates the whole document against ``TURN_OUTPUT_SCHEMA`` (a
schema failure fails the Turn output as one error value), then walks the
actions in blocker order and runs, per action: authority → bounds →
type-specific checks → effect → exactly one event,
``foreman_action_applied`` or ``foreman_action_refused`` with the reason.

Per-action discipline:

- **One refused action never stops the others.** Expected failures — charter
  misses, bounds, lint, guards, dup keys — are ``refused`` outcomes recorded
  in ``applied.jsonl``. Genuine read/write failures of the kind a resume
  should retry (a failed ``gh`` read, a torn effect) still propagate: Apply
  crashing mid-Turn is what ``applied.jsonl`` exists to resume from.
- **Replay never produces a second issue, comment or item.** Recorded
  outcomes skip outright; a crash between an effect and its record is
  repaired by the file-issue marker search, per-action item ids, and exact
  comment-body dedupe — never by a second issue, comment or item.
- **Pause is read before each action.** A ``paused`` foreman stops the walk;
  applied effects are never undone and unprocessed actions stay unrecorded
  so a later resume applies them.

A refused action is displayed as ``jabroni`` (see ``display_action_outcome``);
the event kind and stored reason stay semantic.

Module layout (the engine grew past the per-module size cap): this file
keeps the public surface — ``apply_turn``, the outcome dataclasses, the
``applied.jsonl`` bookkeeping, the processing order and the per-action
event emit. ``apply_context.py`` holds the shared run/verdict plumbing,
``apply_checks.py`` the guard layer every effect passes first, and
``apply_handlers.py`` the per-type handlers ``_process`` dispatches to.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import LabelConfig
from ..github import GitHubError
from ..instrumentation import log_event
from . import Result
from .actions import (
    Action,
    FileIssueAction,
    RaiseItemAction,
    TurnOutput,
    parse_turn_output,
)
from .apply_context import (
    OUTCOME_APPLIED,
    OUTCOME_REFUSED,
    OUTCOME_SKIPPED,
    _fingerprint,
    _marker_pattern,
    _Run,
    _Verdict,
)
from .apply_handlers import _process
from .charter import Charter
from .journal import _append, utc_now
from .paths import actions_path, applied_path, state_path
from .state import read_state_snapshot

#: Event kinds (registered in ``event_levels/``): one per processed action.
#: Module-level literals — the kind-registry scanner resolves the ``kind``
#: name in ``_emit`` against assignments in this file, so they must live here.
EVENT_ACTION_APPLIED = "foreman_action_applied"
EVENT_ACTION_REFUSED = "foreman_action_refused"


@dataclass(frozen=True)
class ActionOutcome:
    """What Apply did with one action: applied, refused, or skipped.

    ``reason`` names why a refusal happened (and which bound); ``detail`` is
    the human-readable applied effect ("filed #12, armed" / "would file …" in
    dry-run).
    """

    index: int
    type: str
    key: str | None
    outcome: str
    reason: str | None = None
    detail: str | None = None
    issue_number: int | None = None
    item_id: str | None = None


@dataclass(frozen=True)
class ApplyResult:
    """The Turn's apply pass: the parsed output plus one outcome per action."""

    turn_id: str
    turn_summary: str
    outcomes: tuple[ActionOutcome, ...]
    stopped_by_pause: bool = False
    dry_run: bool = False


# ---------------------------------------------------------------------------
# applied.jsonl — one JSON record per processed action, append-only
# ---------------------------------------------------------------------------


def _load_applied(path: Path) -> dict[int, dict[str, Any]]:
    """Recorded outcomes by action index; torn or corrupt lines are skipped."""
    records: dict[int, dict[str, Any]] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return records
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        index = record.get("index")
        if isinstance(index, int) and index not in records:
            records[index] = record
    return records


def _append_record(
    path: Path, index: int, fingerprint: str, verdict: _Verdict, raw_action: dict[str, Any]
) -> Result[None]:
    record = {
        "index": index,
        "fingerprint": fingerprint,
        "outcome": verdict.outcome,
        "type": raw_action.get("type"),
        "key": raw_action.get("key"),
        "issue_number": verdict.issue_number,
        "item_id": verdict.item_id,
        "reason": verdict.reason,
        "detail": verdict.detail,
        "ts": utc_now(),
    }
    return _append(path, record)


# ---------------------------------------------------------------------------
# Ordering: same-Turn key dependencies are filed blockers-first
# ---------------------------------------------------------------------------


def _file_issue_keys(actions: tuple[Action, ...]) -> dict[str, list[int]]:
    """key -> action indices for every ``file_issue`` (dups kept, refused later)."""
    keys: dict[str, list[int]] = {}
    for i, action in enumerate(actions):
        if isinstance(action, FileIssueAction):
            keys.setdefault(action.key, []).append(i)
    return keys


def _processing_order(
    actions: tuple[Action, ...],
) -> tuple[list[int], frozenset[str], frozenset[str], frozenset[str]]:
    """Stable topological order, plus the cycle/dup/known key sets.

    Only ``file_issue``→``file_issue`` edges exist (``blocked_by`` ``key:``
    refs). Non-file_issue actions keep their relative order; Kahn's with the
    original index as tie-break keeps the whole order deterministic. Actions
    left unordered are in (or behind) a dependency cycle — their keys are
    returned so dependents can name the refusal.
    """
    keys = _file_issue_keys(actions)
    indegree: dict[int, int] = {i: 0 for i in range(len(actions))}
    dependents: dict[int, list[int]] = {i: [] for i in range(len(actions))}
    for i, action in enumerate(actions):
        if not isinstance(action, FileIssueAction):
            continue
        for ref in action.blocked_by:
            if not ref.startswith("key:"):
                continue
            for dep in keys.get(ref[4:], []):
                # A self-edge counts too: it keeps the action cyclic, so the
                # refusal names a cycle rather than an unapplied dependency.
                indegree[i] += 1
                dependents[dep].append(i)
    ready = [i for i in range(len(actions)) if indegree[i] == 0]
    order: list[int] = []
    emitted: set[int] = set()
    while ready:
        i = min(ready)
        ready.remove(i)
        emitted.add(i)
        order.append(i)
        for dep in dependents[i]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                ready.append(dep)
    cyclic = [i for i in range(len(actions)) if i not in emitted]
    order.extend(sorted(cyclic))
    cycle_keys = frozenset(
        actions[i].key for i in cyclic if isinstance(actions[i], FileIssueAction)
    )
    dup_file = frozenset(k for k, idxs in keys.items() if len(idxs) > 1)
    item_keys: dict[str, int] = {}
    for a in actions:
        if isinstance(a, RaiseItemAction):
            item_keys[a.key] = item_keys.get(a.key, 0) + 1
    dup_item = frozenset(k for k, count in item_keys.items() if count > 1)
    return order, cycle_keys, dup_file, dup_item


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def _emit(run: _Run, index: int, raw_action: dict[str, Any], verdict: _Verdict) -> None:
    kind = EVENT_ACTION_APPLIED if verdict.outcome == OUTCOME_APPLIED else EVENT_ACTION_REFUSED
    payload: dict[str, Any] = {
        "foreman": run.charter.name,
        "turn_id": run.turn_id,
        "action_index": index,
        "action_type": raw_action.get("type"),
        "key": raw_action.get("key"),
        "reason": verdict.reason,
        "detail": verdict.detail,
    }
    if verdict.issue_number is not None:
        payload["issue_number"] = verdict.issue_number
    if verdict.item_id is not None:
        payload["item_id"] = verdict.item_id
    # write-gate-exempt(issue=3119): the foreman Apply engine has no write_gate receiver; _emit only runs on a live pass — apply_turn skips it under dry_run.
    log_event(  # event-consumer: audit-only — applied.jsonl is the per-action record Apply reads back on resume; the events.db mirror keeps a Turn's verdicts queryable
        state_path(run.foreman_root),
        kind,
        payload,
        repo=run.charter.repos[0] if run.charter.repos else None,
    )


def apply_turn(
    foreman_root: Path,
    turn_id: str,
    *,
    charter: Charter,
    gh: Any,
    labels: LabelConfig | None = None,
    signal_marked: bool = False,
    dry_run: bool = False,
) -> Result[ApplyResult]:
    """Apply one Turn's ``actions.json``; returns the outcome of every action.

    ``gh`` is the issue client for the charter's single repo (a ``GitHub``
    adapter or ``LocalFileGitHub`` in tests); ``labels`` is the live
    ``LabelConfig`` the Ready marker and active/terminal sets are read from;
    ``signal_marked`` mirrors the briefing flag — a signal-marked Turn files
    every issue unarmed. ``dry_run`` performs every check and reports what
    would happen, with no effect, event, or record.

    The whole Turn output fails as one error value when ``actions.json`` is
    missing, unparseable, or fails the schema.
    """
    labels = labels or LabelConfig()
    try:
        actions_file = actions_path(foreman_root, turn_id)
        applied_file = applied_path(foreman_root, turn_id)
    except ValueError as exc:
        return Result(error=f"turn path refused: {exc}")
    try:
        raw_text = actions_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Result(error=f"actions.json: not found at {actions_file}")
    except OSError as exc:
        return Result(error=f"actions.json: unreadable: {exc}")
    try:
        document = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return Result(error=f"actions.json: invalid JSON: {exc}")
    parsed = parse_turn_output(document)
    if not parsed.ok:
        return Result(error=f"actions.json: {parsed.error}")
    output: TurnOutput = parsed.value
    raw_actions = document["actions"]

    recorded = _load_applied(applied_file)
    owner_label = f"{labels.foreman_prefix}{charter.name}"
    try:
        open_owned = gh.issue_list(labels=[owner_label], state="open")
        all_owned = gh.issue_list(labels=[owner_label], state="all")
    except GitHubError as exc:
        return Result(error=f"could not read issues for {owner_label!r}: {exc}")

    marker_re = _marker_pattern(charter.name)
    marker_map: dict[tuple[str, str], int] = {}
    for issue in all_owned:
        match = marker_re.search(issue.get("body") or "")
        if match and isinstance(issue.get("number"), int):
            marker_map[(match.group(1), match.group(2))] = issue["number"]

    order, cycle_keys, dup_file, dup_item = _processing_order(output.actions)
    run = _Run(
        foreman_root=foreman_root,
        turn_id=turn_id,
        charter=charter,
        gh=gh,
        labels=labels,
        owner_label=owner_label,
        signal_marked=signal_marked,
        dry_run=dry_run,
        open_issues_used=len(open_owned),
        # ``file_issue`` outcomes already recorded still count toward the
        # per-Turn bound: resume is the same Turn.
        file_issues_applied=sum(
            1
            for record in recorded.values()
            if record.get("type") == FileIssueAction.type
            and record.get("outcome") == OUTCOME_APPLIED
        ),
        marker_map=marker_map,
        cycle_keys=cycle_keys,
        duplicate_file_keys=dup_file,
        duplicate_item_keys=dup_item,
        known_file_keys=frozenset(_file_issue_keys(output.actions)),
    )

    outcomes: list[ActionOutcome] = []
    stopped = False
    for index in order:
        action = output.actions[index]
        raw_action = raw_actions[index] if isinstance(raw_actions[index], dict) else {}
        fingerprint = _fingerprint(raw_actions[index])
        prior = recorded.get(index)
        if prior is not None and prior.get("fingerprint") == fingerprint:
            # Dependents still resolve against the recorded outcome: an
            # applied file_issue hands its number to blocked_by refs, a
            # refused one hands them its refusal.
            if isinstance(action, FileIssueAction):
                if prior.get("outcome") == OUTCOME_APPLIED and isinstance(
                    prior.get("issue_number"), int
                ):
                    run.key_numbers[action.key] = prior["issue_number"]
                elif prior.get("outcome") == OUTCOME_REFUSED:
                    run.failed_keys.add(action.key)
            outcomes.append(
                ActionOutcome(
                    index=index,
                    type=action.type,
                    key=getattr(action, "key", None),
                    outcome=OUTCOME_SKIPPED,
                    detail=f"already recorded as {prior.get('outcome')}",
                    issue_number=prior.get("issue_number"),
                    item_id=prior.get("item_id"),
                )
            )
            continue

        state = read_state_snapshot(foreman_root)
        if state.state == "paused":
            stopped = True
            break

        if dry_run:
            verdict = _process(run, action, index)
        else:
            verdict = _process(run, action, index)
            _emit(run, index, raw_action, verdict)
            _append_record(applied_file, index, fingerprint, verdict, raw_action)
        outcomes.append(
            ActionOutcome(
                index=index,
                type=action.type,
                key=getattr(action, "key", None),
                outcome=verdict.outcome,
                reason=verdict.reason,
                detail=verdict.detail,
                issue_number=verdict.issue_number,
                item_id=verdict.item_id,
            )
        )

    for index in order[len(outcomes) :]:
        action = output.actions[index]
        outcomes.append(
            ActionOutcome(
                index=index,
                type=action.type,
                key=getattr(action, "key", None),
                outcome=OUTCOME_SKIPPED,
                reason="foreman paused" if stopped else None,
            )
        )

    return Result(
        value=ApplyResult(
            turn_id=turn_id,
            turn_summary=output.turn_summary,
            outcomes=tuple(outcomes),
            stopped_by_pause=stopped,
            dry_run=dry_run,
        )
    )
