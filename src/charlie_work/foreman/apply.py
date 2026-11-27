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
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..atomic_write import write_text_atomic
from ..config import LabelConfig
from ..github import GitHubError, label_names
from ..github_body_scan import parse_blockers
from ..instrumentation import log_event
from ..issue_contract_lint import lint_issue_body, render_blocker_section
from ..outbound_body_guard import (
    OutboundBodyGuardError,
    check_outbound_write,
    refusal_summary,
    scan_outbound_text,
)
from ..private_slug_gate import count_slug_mentions_in_text
from ..ratchet_baseline import BaselineFormatError, load_set_baseline
from . import Result
from .actions import (
    Action,
    AmendIssueAction,
    CommentIssueAction,
    DeleteMemoryAction,
    FileIssueAction,
    ProposeDoneAction,
    RaiseItemAction,
    TurnOutput,
    WithdrawIssueAction,
    WriteMemoryAction,
    parse_turn_output,
)
from .charter import ARMING_AUTO, Charter
from .items import create_item
from .journal import _append, utc_now
from .memory import delete_memory as _delete_memory_file
from .memory import write_memory as _write_memory_file
from .paths import applied_path, actions_path, state_path, turn_dir
from .state import AuthoredIssue, read_state_snapshot, save_state, with_updates

#: Event kinds (registered in ``event_levels/``): one per processed action.
EVENT_ACTION_APPLIED = "foreman_action_applied"
EVENT_ACTION_REFUSED = "foreman_action_refused"

OUTCOME_APPLIED = "applied"
OUTCOME_REFUSED = "refused"
#: A recorded action on replay, or one the pause-stop left unprocessed.
OUTCOME_SKIPPED = "skipped"

#: The repo's private-slug baseline; its presence arms the public-repo slug
#: refusal (issue #1502's ratchet baseline, reused as a data source).
_SLUG_BASELINE_DIRNAME = ".private-slug-baseline"
_SLUG_SETS_DIRNAME = "slugs"

#: Color/description used when Apply bootstraps the ownership label. The
#: purple matches ``helpers_labels._label_color``'s non-ready convention.
_OWNER_LABEL_COLOR = "5319E7"

#: The hidden marker trailing every foreman-filed issue body. ``{turn}`` and
#: ``{key}`` carry the Turn id and action key, so a crash between create and
#: record is found by searching the ownership label for this exact string.
_MARKER_TEMPLATE = "<!-- foreman:{name} turn:{turn} key:{key} -->"


def _marker_pattern(foreman_name: str) -> re.Pattern[str]:
    """Match this foreman's hidden marker inside an authored issue body."""
    return re.compile(
        r"<!--\s*foreman:" + re.escape(foreman_name) + r"\s+turn:(\S+)\s+key:(\S+)\s*-->"
    )


def _marker(foreman_name: str, turn_id: str, key: str) -> str:
    return _MARKER_TEMPLATE.format(name=foreman_name, turn=turn_id, key=key)


def _body_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _fingerprint(raw_action: Any) -> str:
    """Canonical digest of one raw action object (replay-skip identity)."""
    return hashlib.sha256(
        json.dumps(raw_action, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


@dataclass
class _Run:
    """The mutable bookkeeping one apply walk threads through every action."""

    foreman_root: Path
    turn_id: str
    charter: Charter
    gh: Any
    labels: LabelConfig
    owner_label: str
    signal_marked: bool
    dry_run: bool
    open_issues_used: int
    file_issues_applied: int
    marker_map: dict[tuple[str, str], int]
    key_numbers: dict[str, int] = field(default_factory=dict)
    failed_keys: set[str] = field(default_factory=set)
    cycle_keys: frozenset[str] = frozenset()
    duplicate_file_keys: frozenset[str] = frozenset()
    duplicate_item_keys: frozenset[str] = frozenset()
    known_file_keys: frozenset[str] = frozenset()

    @property
    def repo_root(self) -> Path | None:
        return getattr(self.gh, "repo_root", None)

    @property
    def state_dir(self) -> str | None:
        state_dir = getattr(self.gh, "state_dir", None)
        if state_dir is None:
            state_dir = getattr(getattr(self.gh, "runtime", None), "state_dir", None)
        return state_dir


@dataclass(frozen=True)
class _Verdict:
    """A handler's per-action result before the index is folded in."""

    outcome: str
    reason: str | None = None
    detail: str | None = None
    issue_number: int | None = None
    item_id: str | None = None


def _refused(reason: str) -> _Verdict:
    return _Verdict(outcome=OUTCOME_REFUSED, reason=reason)


def _applied(detail: str, issue_number: int | None = None, item_id: str | None = None) -> _Verdict:
    return _Verdict(
        outcome=OUTCOME_APPLIED, detail=detail, issue_number=issue_number, item_id=item_id
    )


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
# Shared check helpers
# ---------------------------------------------------------------------------


def _repo_check(run: _Run, repo: str) -> str | None:
    if repo not in run.charter.repos:
        return f"repo {repo!r}: not in the charter ({list(run.charter.repos)})"
    return None


def _slug_baseline(repo_root: Path | None) -> frozenset[str] | None:
    """Listed slugs, or None when the repo carries no baseline (check skipped)."""
    if repo_root is None:
        return None
    directory = repo_root / _SLUG_BASELINE_DIRNAME / _SLUG_SETS_DIRNAME
    try:
        return load_set_baseline(directory)
    except BaselineFormatError:
        return None


def _slug_refusal(run: _Run, parts: tuple[tuple[str, str], ...]) -> str | None:
    """Refusal reason when any part names a baseline slug, else None."""
    slugs = _slug_baseline(run.repo_root)
    if not slugs:
        return None
    for part_name, text in parts:
        if text and count_slug_mentions_in_text(text, sorted(slugs)):
            return (
                f"{part_name} names a private slug from "
                f"{_SLUG_BASELINE_DIRNAME}/{_SLUG_SETS_DIRNAME}/ -- private repo "
                "names never reach a public repo"
            )
    return None


def _outbound_refusal(
    run: _Run, surface: str, parts: tuple[tuple[str, str], ...], issue_number: int | None
) -> str | None:
    """Refusal reason when the secret guard matches, else None.

    On a live run the guarded chokepoint itself scans and emits its refusal
    event; on dry-run the same scan runs without the event. A guard-machinery
    failure (``OutboundBodyGuardError``) fails closed as a refusal — the
    write is the thing being protected, so it can never proceed unchecked.
    """
    try:
        if run.dry_run:
            matches = tuple(
                match
                for part_name, text in parts
                for match in scan_outbound_text(text, part=part_name)
            )
        else:
            matches = check_outbound_write(
                surface=surface,
                parts=parts,
                repo_root=run.repo_root,
                state_dir=run.state_dir,
                issue_number=issue_number,
            )
    except OutboundBodyGuardError as exc:
        return f"outbound guard failure (refusing closed): {exc}"
    if matches:
        return refusal_summary(surface, matches)
    return None


def _live_authored(run: _Run, number: int) -> tuple[dict[str, Any], str | None]:
    """Live-fetch an issue and check authored-issue authority.

    Returns ``(issue, None)`` when the issue carries the ownership label, the
    foreman's marker, no active or terminal label, and is still open — or
    ``({}, reason)`` naming the first authority miss.
    """
    issue = run.gh.issue_view(number)
    if not issue:
        return {}, f"issue #{number}: not found in {run.charter.repos[0]}"
    names = label_names(issue)
    if run.owner_label not in names:
        return {}, f"issue #{number}: lacks ownership label {run.owner_label!r}"
    if str(issue.get("state", "")).upper() != "OPEN":
        return {}, f"issue #{number}: not open (state {issue.get('state')!r})"
    if not _marker_pattern(run.charter.name).search(issue.get("body") or ""):
        return {}, f"issue #{number}: tracked, not authored (no foreman marker)"
    blocked = names & (run.labels.active | run.labels.terminal)
    if blocked:
        return {}, f"issue #{number}: in active/terminal state {sorted(blocked)}"
    return issue, None


def _check_outbound_and_slug(
    run: _Run, surface: str, parts: tuple[tuple[str, str], ...], issue_number: int | None
) -> str | None:
    refusal = _slug_refusal(run, parts)
    if refusal is not None:
        return refusal
    return _outbound_refusal(run, surface, parts, issue_number)


def _record_authored(run: _Run, number: int, body: str) -> None:
    """Cache the issue's body hash in ``state.json`` (rebuildable; best-effort)."""
    state = read_state_snapshot(run.foreman_root)
    issues = dict(state.issues)
    issues[number] = AuthoredIssue(number=number, body_hash=_body_hash(body))
    save_state(run.foreman_root, with_updates(state, issues=issues))


def _blocker_tail(blockers: frozenset[int], marker: str) -> str:
    """The ``## Blocked by`` section with the hidden marker on its last line.

    The marker must end the body (design §8.3), but a bare trailing line
    under the section heading is scanned as a section *item* — neither a
    ``#N`` ref, a ``None`` sentinel, nor an inline declaration — and parks
    the issue as unreadable (``_scan_blocker_sections``). Riding the last
    line instead keeps every parse: ``Blocked by #N <!-- marker -->`` reads
    as the inline declaration the line already is, and ``None <!-- ... -->``
    still prefix-matches the sentinel.
    """
    return render_blocker_section(blockers).rstrip() + " " + marker


def _arm_or_raise(run: _Run, number: int, key: str, title: str) -> str:
    """The arming step: add Ready, or leave unarmed behind a decision item.

    Runs *after* the issue exists, per the design's "separate, final step":
    Ready only when the charter says ``arming: auto``, the foreman is not
    ``degraded``, and the Turn is not signal-marked. Every other path leaves
    the issue unarmed and raises a decision item whose ``command`` arms it.
    Returns the detail fragment for the applied outcome.
    """
    state = read_state_snapshot(run.foreman_root)
    why_unarmed: str | None = None
    if run.signal_marked:
        why_unarmed = "turn is signal-marked"
    elif run.charter.arming != ARMING_AUTO:
        why_unarmed = f"arming: {run.charter.arming}"
    elif state.state == "degraded":
        why_unarmed = "foreman is degraded"
    if why_unarmed is None:
        if run.dry_run:
            return "armed"
        if run.gh.add_issue_label(number, run.labels.ready):
            return "armed"
        why_unarmed = "ready label add failed"

    command = f'gh issue edit {number} --add-label "{run.labels.ready}"'
    item_id = f"{run.turn_id}:{key}:arm"
    if run.dry_run:
        return f"unarmed ({why_unarmed}); would raise arming item {item_id}"
    item = create_item(
        run.foreman_root,
        kind="decision",
        question=f"Arm issue #{number} ({title}) for dispatch?",
        options=(f"Arm it — add {run.labels.ready}", "Leave it unarmed"),
        recommended=0,
        command=command,
        item_id=item_id,
    )
    if item.ok:
        return f"unarmed ({why_unarmed}); arming item {item.value.id}"
    return f"unarmed ({why_unarmed}); arming item failed: {item.error}"


def _comment_body_file(run: _Run, index: int) -> Path:
    """The per-action comment body file ``issue_comment`` reads from."""
    return turn_dir(run.foreman_root, run.turn_id) / f"comment-{index}.md"


def _post_comment(run: _Run, number: int, index: int, text: str) -> str | None:
    """Append *text* as a comment, deduped on exact body; error string or None."""
    if _has_comment(run, number, text):
        return None
    path = _comment_body_file(run, index)
    write_text_atomic(path, text)
    try:
        run.gh.issue_comment(number, path)
    except GitHubError as exc:
        return f"comment on #{number} failed: {exc}"
    return None


def _comment_matches(stored: str, text: str) -> bool:
    """True when *stored* is the comment ``text`` — or carries it as the body.

    The local backend stores one comments-block chunk per entry: the
    ``**charlie-work** (<ts>)`` author line, a blank line, then the body —
    so an exact-body dedupe must accept the stored form ``author + "\\n\\n"
    + text``. On the ``gh`` backend ``comments[].body`` is the text alone.
    """
    stored = stored.strip().replace("\r\n", "\n")
    text = text.strip().replace("\r\n", "\n")
    return stored == text or stored.endswith("\n\n" + text)


def _has_comment(run: _Run, number: int, text: str) -> bool:
    """True when the exact comment body already sits on the issue."""
    live = run.gh.issue_view(number)
    existing = [str(c.get("body", "")) for c in live.get("comments", []) if isinstance(c, dict)]
    return any(_comment_matches(body, text) for body in existing)


# ---------------------------------------------------------------------------
# Per-type handlers — each returns a _Verdict
# ---------------------------------------------------------------------------


def _file_issue(run: _Run, action: FileIssueAction, index: int) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)

    # Idempotent create runs before the bounds: a crash between create and
    # record replays into this search, and the already-filed issue must
    # report applied — a bound refusal here would falsely cascade "the
    # action for key k was refused" to every dependent.
    existing_number = run.marker_map.get((run.turn_id, action.key))
    if existing_number is not None:
        if run.dry_run:
            return _applied(
                "would reuse existing issue "
                f"#{existing_number} ({_arm_or_raise(run, existing_number, action.key, action.title)})",
                issue_number=existing_number,
            )
        live = run.gh.issue_view(existing_number)
        _record_authored(run, existing_number, str(live.get("body") or ""))
        detail = f"reused existing issue #{existing_number}"
        if str(live.get("state", "")).upper() == "OPEN":
            detail += f", {_arm_or_raise(run, existing_number, action.key, action.title)}"
        run.key_numbers[action.key] = existing_number
        run.file_issues_applied += 1
        return _applied(detail, issue_number=existing_number)

    if action.key in run.duplicate_file_keys:
        return _refused(
            f"key {action.key!r}: duplicate file_issue key — markers and "
            "blocked_by refs must resolve to exactly one action"
        )
    bounds = run.charter.bounds
    if run.open_issues_used >= bounds.max_open_issues:
        return _refused(
            f"bound max_open_issues: {run.open_issues_used} open issues already "
            f"carry {run.owner_label} (limit {bounds.max_open_issues})"
        )
    if run.file_issues_applied >= bounds.max_issues_per_turn:
        return _refused(
            f"bound max_issues_per_turn: {run.file_issues_applied} file_issue "
            f"actions already applied this Turn (limit {bounds.max_issues_per_turn})"
        )

    blocker_numbers: list[int] = []
    for ref in action.blocked_by:
        if ref.startswith("key:"):
            dep = ref[4:]
            if dep in run.key_numbers:
                blocker_numbers.append(run.key_numbers[dep])
            elif dep in run.failed_keys:
                return _refused(f"blocked_by {ref!r}: the action for key {dep!r} was refused")
            elif dep in run.cycle_keys:
                return _refused(
                    f"blocked_by {ref!r}: key {dep!r} is in a dependency cycle, refused"
                )
            elif dep in run.known_file_keys:
                return _refused(f"blocked_by {ref!r}: key {dep!r} was not applied")
            else:
                return _refused(
                    f"blocked_by {ref!r}: no file_issue action in this Turn has key {dep!r}"
                )
        else:
            blocker_numbers.append(int(ref[1:]))

    problems = lint_issue_body(action.body, frozenset())
    if problems:
        return _refused("body fails the issue contract: " + "; ".join(problems))

    refusal = _check_outbound_and_slug(
        run,
        "create_issue",
        (("title", action.title), ("body", action.body)),
        issue_number=None,
    )
    if refusal is not None:
        return _refused(refusal)

    blocker_set = frozenset(blocker_numbers)
    marker = _marker(run.charter.name, run.turn_id, action.key)
    final_body = "\n\n".join(
        part for part in (action.body.strip(), _blocker_tail(blocker_set, marker)) if part
    )
    problems = lint_issue_body(final_body, blocker_set)
    if problems:
        return _refused("composed body fails the issue contract: " + "; ".join(problems))

    if run.dry_run:
        arming = _arm_or_raise(run, 0, action.key, action.title)
        return _applied(f"would file {action.title!r} ({arming})")

    known_labels = {entry.get("name") for entry in run.gh.label_list()}
    if run.owner_label not in known_labels:
        run.gh.label_create(
            run.owner_label,
            _OWNER_LABEL_COLOR,
            f"Authored by foreman {run.charter.name}",
        )
    number = run.gh.create_issue(action.title, final_body, labels=[run.owner_label])
    if number is None:
        return _refused("issue create failed (guard refusal or write failure)")
    run.marker_map[(run.turn_id, action.key)] = number
    live = run.gh.issue_view(number)
    _record_authored(run, number, str(live.get("body") or final_body))
    arming = _arm_or_raise(run, number, action.key, action.title)
    run.key_numbers[action.key] = number
    run.file_issues_applied += 1
    run.open_issues_used += 1
    return _applied(f"filed #{number}, {arming}", issue_number=number)


def _amend_issue(run: _Run, action: AmendIssueAction) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)
    issue, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)

    recorded = read_state_snapshot(run.foreman_root).issues.get(action.number)
    live_body = str(issue.get("body") or "")
    if recorded is None:
        return _refused(
            f"issue #{action.number}: no authored record in state.json — "
            "the body hash cannot be checked"
        )
    if _body_hash(live_body) != recorded.body_hash:
        return _refused(
            f"issue #{action.number}: live body hash differs from the recorded "
            "hash — the operator edited it and their text is never overwritten"
        )

    problems = lint_issue_body(action.body, frozenset())
    if problems:
        return _refused("body fails the issue contract: " + "; ".join(problems))
    refusal = _check_outbound_and_slug(
        run, "edit_issue_body", (("body", action.body),), issue_number=action.number
    )
    if refusal is not None:
        return _refused(refusal)

    blockers = frozenset(parse_blockers(live_body))
    marker_match = _marker_pattern(run.charter.name).search(live_body)
    marker = (
        marker_match.group(0) if marker_match else _marker(run.charter.name, run.turn_id, "amend")
    )
    new_body = "\n\n".join(
        part for part in (action.body.strip(), _blocker_tail(blockers, marker)) if part
    )
    problems = lint_issue_body(new_body, blockers)
    if problems:
        return _refused("composed body fails the issue contract: " + "; ".join(problems))

    if run.dry_run:
        return _applied(f"would amend #{action.number}", issue_number=action.number)

    # Labels, marker and openness are re-read live immediately before the
    # effect: an operator edit between check and write still refuses.
    _, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)
    if not run.gh.edit_issue_body(action.number, new_body):
        return _refused(f"issue #{action.number}: body edit failed")
    live = run.gh.issue_view(action.number)
    _record_authored(run, action.number, str(live.get("body") or new_body))
    return _applied(f"amended #{action.number}", issue_number=action.number)


def _withdraw_issue(run: _Run, action: WithdrawIssueAction, index: int) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)
    _, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)

    comment = (
        f"Withdrawn by foreman `{run.charter.name}` in turn `{run.turn_id}`.\n\n{action.reason}"
    )
    refusal = _check_outbound_and_slug(
        run, "issue_comment", (("body", comment),), issue_number=action.number
    )
    if refusal is not None:
        return _refused(refusal)

    if run.dry_run:
        return _applied(f"would withdraw #{action.number}", issue_number=action.number)

    _, refusal = _live_authored(run, action.number)
    if refusal is not None:
        return _refused(refusal)
    # Per the design: the Ready label goes first, then the close carries the
    # reason as a comment — an unarmed-then-closed issue can never dispatch.
    if not run.gh.remove_issue_label(action.number, run.labels.ready):
        return _refused(f"issue #{action.number}: could not remove {run.labels.ready}")
    comment_failure = _post_comment(run, action.number, index, comment)
    if comment_failure is not None:
        return _refused(comment_failure)
    if not run.gh.close_issue(action.number):
        return _refused(f"issue #{action.number}: close failed")
    return _applied(f"withdrew #{action.number}", issue_number=action.number)


def _comment_issue(run: _Run, action: CommentIssueAction, index: int) -> _Verdict:
    refusal = _repo_check(run, action.repo)
    if refusal is not None:
        return _refused(refusal)
    issue = run.gh.issue_view(action.number)
    if not issue:
        return _refused(f"issue #{action.number}: not found in {run.charter.repos[0]}")
    if run.owner_label not in label_names(issue):
        return _refused(f"issue #{action.number}: lacks ownership label {run.owner_label!r}")
    refusal = _check_outbound_and_slug(
        run, "issue_comment", (("body", action.body),), issue_number=action.number
    )
    if refusal is not None:
        return _refused(refusal)
    if run.dry_run:
        return _applied(f"would comment on #{action.number}", issue_number=action.number)
    if _has_comment(run, action.number, action.body):
        return _applied(
            f"comment already present on #{action.number} (deduped)",
            issue_number=action.number,
        )
    failure = _post_comment(run, action.number, index, action.body)
    if failure is not None:
        return _refused(failure)
    return _applied(f"commented on #{action.number}", issue_number=action.number)


def _raise_item(run: _Run, action: RaiseItemAction) -> _Verdict:
    if action.key in run.duplicate_item_keys:
        return _refused(
            f"key {action.key!r}: duplicate raise_item key — item ids are "
            "derived from the key and must be unique per Turn"
        )
    item_id = f"{run.turn_id}:{action.key}"
    if run.dry_run:
        return _applied(f"would raise {action.kind} item {item_id}", item_id=item_id)
    # For a report the question field carries the report body; options and
    # recommended are forwarded regardless so the item store's own per-kind
    # validation refuses a shape that doesn't fit the kind.
    result = create_item(
        run.foreman_root,
        kind=action.kind,
        question=action.question if action.kind != "report" else None,
        text=action.question if action.kind == "report" else None,
        options=action.options,
        recommended=action.recommended,
        command=action.command,
        item_id=item_id,
    )
    if not result.ok:
        if result.error and "already exists" in result.error:
            return _applied(f"item {item_id} already exists (deduped)", item_id=item_id)
        return _refused(f"item store refused: {result.error}")
    return _applied(f"raised {action.kind} item {result.value.id}", item_id=result.value.id)


def _write_memory(run: _Run, action: WriteMemoryAction) -> _Verdict:
    if run.dry_run:
        return _applied(f"would write memory {action.path!r}")
    result = _write_memory_file(run.foreman_root, action.path, action.content)
    if not result.ok:
        return _refused(f"memory refused: {result.error}")
    return _applied(f"wrote memory {action.path!r}")


def _delete_memory(run: _Run, action: DeleteMemoryAction) -> _Verdict:
    if run.dry_run:
        return _applied(f"would delete memory {action.path!r}")
    result = _delete_memory_file(run.foreman_root, action.path)
    if not result.ok:
        return _refused(f"memory refused: {result.error}")
    return _applied(f"deleted memory {action.path!r}")


def _propose_done(run: _Run, action: ProposeDoneAction, index: int) -> _Verdict:
    if run.charter.standing:
        return _refused("charter is standing: a standing foreman never proposes done")
    item_id = f"{run.turn_id}:propose-done:{index}"
    if run.dry_run:
        return _applied(f"would raise done-confirmation item {item_id}", item_id=item_id)
    result = create_item(
        run.foreman_root,
        kind="decision",
        question=f"Foreman {run.charter.name} proposes done. Is the outcome met?",
        text=action.evidence,
        options=("Done — mark the foreman done", "Not done — keep working"),
        recommended=0,
        item_id=item_id,
    )
    if not result.ok:
        if result.error and "already exists" in result.error:
            return _applied(f"item {item_id} already exists (deduped)", item_id=item_id)
        return _refused(f"item store refused: {result.error}")
    return _applied(f"raised done-confirmation item {result.value.id}", item_id=result.value.id)


def _process(run: _Run, action: Action, index: int) -> _Verdict:
    if isinstance(action, FileIssueAction):
        verdict = _file_issue(run, action, index)
        if verdict.outcome == OUTCOME_REFUSED:
            run.failed_keys.add(action.key)
        return verdict
    if isinstance(action, AmendIssueAction):
        return _amend_issue(run, action)
    if isinstance(action, WithdrawIssueAction):
        return _withdraw_issue(run, action, index)
    if isinstance(action, CommentIssueAction):
        return _comment_issue(run, action, index)
    if isinstance(action, RaiseItemAction):
        return _raise_item(run, action)
    if isinstance(action, WriteMemoryAction):
        return _write_memory(run, action)
    if isinstance(action, DeleteMemoryAction):
        return _delete_memory(run, action)
    if isinstance(action, ProposeDoneAction):
        return _propose_done(run, action, index)
    return _refused(f"unknown action type {type(action).__name__}")  # unreachable


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
    log_event(
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
