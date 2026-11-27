"""``charlie gang`` subcommands (issue #3117, ADR-0010): create and inspect foremen.

``gang`` is the operator's front door to the foreman layer: ``new`` scaffolds
``foremen/<name>/``, ``list`` shows every foreman with its state and bounds
usage, and ``show`` prints one foreman's charter, cached state, and open
items. Issue #3121 adds the talk-and-answer commands: ``say`` and ``signal``
write the append-only logs, ``interventions``/``intervene``/``snooze``/
``dismiss`` work the item lifecycle, ``pepe-silvia`` and ``dayman`` read the
newest briefing and report, and ``pause``/``resume`` set the foreman state.

This module is the *only* seam into ``charlie_work.foreman``: every import of
that package happens inside the handlers below, so importing the CLI never
pulls the foreman layer in and nothing else can reach it. The import-graph
guard lives in ``tests/test_foreman_import_direction.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .command_result import CommandResult
from .fleet_paths import fleet_dir

GANG_COMMAND = "gang"

_LOCAL_TIME_RE = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


def register_gang_subparsers(subparsers: Any) -> None:
    """Add every ``gang`` subcommand to the top-level subparsers."""
    gang = subparsers.add_parser(
        GANG_COMMAND, help="Manage foremen — chartered outcome managers (ADR-0010)"
    )
    sub = gang.add_subparsers(dest="gang_command", required=True)
    new = sub.add_parser("new", help="Create a foreman under foremen/<name>/")
    new.add_argument("name", help="foreman name ([a-z0-9][a-z0-9-]*)")
    new.add_argument(
        "--from",
        dest="from_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="copy charter.yaml and instructions.md from DIR instead of the built-in minimum",
    )
    sub.add_parser("list", help="List foremen with state and bounds usage")
    show = sub.add_parser("show", help="Print one foreman's charter, state, and open items")
    show.add_argument("name", help="foreman name")
    say = sub.add_parser("say", help="Append an operator message to the foreman's inbox")
    say.add_argument("name", help="foreman name")
    say.add_argument("text", help="the instruction text (operator authority)")
    signal = sub.add_parser(
        "signal", help="Append an external event to signals.jsonl (data, never an instruction)"
    )
    signal.add_argument("name", help="foreman name")
    signal.add_argument("--source", required=True, help="label naming the event source")
    payload = signal.add_mutually_exclusive_group(required=True)
    payload.add_argument("--file", dest="file", metavar="PATH", help="payload file to read")
    payload.add_argument("--text", dest="text", metavar="TEXT", help="inline payload text")
    interventions = sub.add_parser(
        "interventions",
        help="List open items — age, recommended option and command (one foreman or all)",
    )
    interventions.add_argument("name", nargs="?", default=None, help="foreman name")
    intervene = sub.add_parser("intervene", help="Answer an open item")
    intervene.add_argument("name", help="foreman name")
    intervene.add_argument("item", help="item id (e.g. it-3)")
    intervene.add_argument(
        "--choose",
        required=True,
        help="the option index (0-based) or the option's exact text",
    )
    intervene.add_argument("--note", default=None, help="optional note recorded with the answer")
    snooze = sub.add_parser("snooze", help="Hide an item until a local time")
    snooze.add_argument("name", help="foreman name")
    snooze.add_argument("item", help="item id (e.g. it-3)")
    snooze.add_argument(
        "--until",
        required=True,
        help="local time: ISO-8601 (e.g. 2026-11-28 09:00) or bare HH:MM for next occurrence",
    )
    dismiss = sub.add_parser("dismiss", help="Close an item without an answer")
    dismiss.add_argument("name", help="foreman name")
    dismiss.add_argument("item", help="item id (e.g. it-3)")
    pepe = sub.add_parser(
        "pepe-silvia", help="Print the newest turn briefing, or say no Turn has run"
    )
    pepe.add_argument("name", help="foreman name")
    dayman = sub.add_parser("dayman", help="Print the newest report item (one foreman or all)")
    dayman.add_argument("name", nargs="?", default=None, help="foreman name")
    pause = sub.add_parser("pause", help="Pause a foreman (no Turns while paused)")
    pause.add_argument("name", help="foreman name")
    resume = sub.add_parser(
        "resume", help="Resume a foreman — also clears faulted and the failure count"
    )
    resume.add_argument("name", help="foreman name")


def _plain(value: Any) -> Any:
    """Dataclass tree -> JSON-ready dict."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _gang_new(args: argparse.Namespace, store: Any, fleet: Any) -> CommandResult:
    from_dir = Path(args.from_dir) if args.from_dir else None
    created = store.scaffold_foreman(fleet, args.name, from_dir=from_dir)
    if not created.ok:
        return CommandResult(False, f"gang new: {created.error}", {})
    return CommandResult(
        True,
        f"created foreman {args.name!r} at {created.value}",
        {"name": args.name, "dir": str(created.value)},
    )


def _gang_list(
    fcharter: Any, fitems: Any, fstate: Any, store: Any, display_state: Any, fleet: Any
) -> CommandResult:
    foremen = store.list_foremen(fleet)
    if not foremen:
        return CommandResult(True, "no foremen", {"foremen": []})
    repos = store.registered_repo_keys(fleet)
    lines: list[str] = []
    data: list[dict[str, Any]] = []
    for name, fdir in foremen:
        state = fstate.read_state_snapshot(fdir)
        open_count = len(fitems.open_items(fitems.load_items(fdir)))
        entry: dict[str, Any] = {
            "name": name,
            "state": state.state,
            "display_state": display_state(state.state, state.degraded_reasons),
            "open_items": open_count,
        }
        chartered = fcharter.load_charter(fdir, registered_repos=repos)
        if chartered.ok:
            bounds = chartered.value.bounds
            entry["issues"] = f"{len(state.issues)}/{bounds.max_open_issues}"
            entry["turns_today"] = f"{state.turns_today}/{bounds.max_turns_per_day}"
            lines.append(
                f"{name}: {display_state(state.state, state.degraded_reasons)}, issues {entry['issues']}, "
                f"turns today {entry['turns_today']}, items {open_count} open"
            )
        else:
            entry["charter_error"] = chartered.error
            lines.append(
                f"{name}: {display_state(state.state, state.degraded_reasons)}, items {open_count} open "
                f"(charter invalid: {chartered.error})"
            )
        data.append(entry)
    return CommandResult(True, "\n".join(lines), {"foremen": data})


def _gang_show(
    fcharter: Any, fitems: Any, fstate: Any, store: Any, display_state: Any, fleet: Any, name: str
) -> CommandResult:
    resolved = store.resolve_foreman(fleet, name)
    if not resolved.ok:
        return CommandResult(False, f"gang show: {resolved.error}", {})
    fdir = resolved.value
    state = fstate.read_state_snapshot(fdir)
    chartered = fcharter.load_charter(fdir, registered_repos=store.registered_repo_keys(fleet))
    open_list = fitems.open_items(fitems.load_items(fdir))
    data: dict[str, Any] = {
        "name": name,
        "dir": str(fdir),
        "state": _plain(state),
        "display_state": display_state(state.state, state.degraded_reasons),
        "open_items": [_plain(item) for item in open_list],
    }
    lines = [f"{name}: {display_state(state.state, state.degraded_reasons)}", f"  dir: {fdir}"]
    if chartered.ok:
        charter = chartered.value
        data["charter"] = _plain(charter)
        lines += [
            f"  outcome: {charter.outcome}",
            f"  repos: {list(charter.repos)}",
            f"  standing: {charter.standing}",
            f"  arming: {charter.arming}",
            f"  bounds: issues {len(state.issues)}/{charter.bounds.max_open_issues}, "
            f"per turn {charter.bounds.max_issues_per_turn}, "
            f"turns today {state.turns_today}/{charter.bounds.max_turns_per_day}, "
            f"min interval {charter.bounds.min_turn_interval_minutes}m, "
            f"quiet {charter.bounds.max_quiet_hours}h, "
            f"unanswered {charter.bounds.max_unanswered_days}d",
        ]
        if charter.done_when is not None:
            lines.append(f"  done_when.check: {list(charter.done_when.check)}")
        if charter.cadence is not None and charter.cadence.report_at:
            lines.append(f"  cadence.report_at: {charter.cadence.report_at}")
    else:
        data["charter_error"] = chartered.error
        lines.append(f"  charter invalid: {chartered.error}")
    if open_list:
        lines.append("  open items:")
        for item in open_list:
            label = item.question or item.text or ""
            lines.append(f"    {item.id} [{item.kind}] {label}")
    else:
        lines.append("  open items: none")
    return CommandResult(True, "\n".join(lines), data)


def _fail(verb: str, error: str | None) -> CommandResult:
    return CommandResult(False, f"gang {verb}: {error}", {})


def _foreman_targets(store: Any, fleet: Path, name: str | None) -> tuple[Any, str | None]:
    """The ``(name, dir)`` pairs a command walks: one foreman, or all of them.

    Returns ``(targets, error)`` — exactly one is meaningful: a non-empty
    target list, or an error string when ``name`` failed to resolve.
    """
    if name is not None:
        resolved = store.resolve_foreman(fleet, name)
        if not resolved.ok:
            return (), resolved.error
        return ((name, resolved.value),), None
    return store.list_foremen(fleet), None


def _parse_utc(ts: str) -> datetime | None:
    """Aware ISO-8601 -> UTC ``datetime``; ``None`` on anything else."""
    text = ts.strip() if ts else ""
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return None
    try:
        return parsed.astimezone(UTC)
    except (ValueError, OverflowError):
        return None


def _age(seconds: float | None) -> str:
    """A compact age: ``40s``, ``12m``, ``3h20m``, ``2d4h``; ``?`` when unknown."""
    if seconds is None:
        return "?"
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600}h"


def _local_stamp(ts: str | None) -> str:
    """A stored UTC timestamp rendered in local time — times are shown local."""
    moment = _parse_utc(ts or "")
    if moment is None:
        return ts or ""
    return moment.astimezone().replace(microsecond=0).isoformat()


def _local_until(text: str) -> str | None:
    """A local time entry -> the UTC ``Z`` form the item store keeps.

    Accepts ISO-8601 with or without an offset (a naive value reads as local
    time) and a bare ``HH:MM`` meaning the next occurrence of that local time.
    ``None`` on anything else — the caller turns that into the refusal.
    """
    t = text.strip()
    if _LOCAL_TIME_RE.fullmatch(t):
        now = datetime.now().astimezone()
        candidate = now.replace(hour=int(t[:2]), minute=int(t[3:]), second=0, microsecond=0)
        if candidate <= now:
            candidate += timedelta(days=1)
        parsed = candidate
    else:
        try:
            parsed = datetime.fromisoformat(t)
        except ValueError:
            return None
        if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
            parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    try:
        return parsed.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except (ValueError, OverflowError):
        return None


def _gang_say(fjournal: Any, store: Any, fleet: Path, name: str, text: str) -> CommandResult:
    resolved = store.resolve_foreman(fleet, name)
    if not resolved.ok:
        return _fail("say", resolved.error)
    written = fjournal.append_inbox(resolved.value, text)
    if not written.ok:
        return _fail("say", written.error)
    entry = written.value
    return CommandResult(
        True, f"{name}: inbox message #{entry.id}", {"name": name, "id": entry.id}
    )


def _gang_signal(
    fjournal: Any, store: Any, fleet: Path, args: argparse.Namespace
) -> CommandResult:
    resolved = store.resolve_foreman(fleet, args.name)
    if not resolved.ok:
        return _fail("signal", resolved.error)
    source = (args.source or "").strip()
    if not source:
        return _fail("signal", "--source: a non-empty label naming the event source")
    if args.file is not None:
        path = Path(args.file)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return _fail("signal", f"--file {path}: unreadable: {exc}")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            return _fail("signal", f"--file {path}: not UTF-8 text: {exc}")
    else:
        content = args.text or ""
    size = len(content.encode("utf-8"))
    if size > fjournal.SIGNAL_MAX_PAYLOAD_BYTES:
        return _fail(
            "signal",
            f"payload: at most {fjournal.SIGNAL_MAX_PAYLOAD_BYTES} bytes, got {size}",
        )
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = None
    # A JSON object carries its own fields; anything else is payload text.
    data = parsed if isinstance(parsed, dict) else {"text": content}
    appended = fjournal.append_signal(resolved.value, source, data)
    if not appended.ok:
        return _fail("signal", appended.error)
    entry = appended.value
    return CommandResult(
        True,
        f"{args.name}: signal #{entry.id} from {source!r}",
        {"name": args.name, "id": entry.id},
    )


def _gang_interventions(fitems: Any, store: Any, fleet: Path, name: str | None) -> CommandResult:
    targets, error = _foreman_targets(store, fleet, name)
    if error is not None:
        return _fail("interventions", error)
    now = datetime.now(UTC)
    lines: list[str] = []
    data: list[dict[str, Any]] = []
    for fname, fdir in targets:
        for item in fitems.open_items(fitems.load_items(fdir)):
            created = _parse_utc(item.created_at)
            age_seconds = (now - created).total_seconds() if created is not None else None
            recommended = (
                item.options[item.recommended]
                if item.recommended is not None and 0 <= item.recommended < len(item.options)
                else None
            )
            data.append(
                {
                    "foreman": fname,
                    "id": item.id,
                    "kind": item.kind,
                    "age_seconds": age_seconds,
                    "question": item.question,
                    "text": item.text,
                    "options": list(item.options),
                    "recommended": recommended,
                    "command": item.command,
                    "created_at": item.created_at,
                }
            )
            lines.append(f"{fname} {item.id} [{item.kind}] {_age(age_seconds)} old")
            if item.question:
                lines.append(f"  question: {item.question}")
            elif item.text:
                lines.append(f"  report: {item.text}")
            if item.options:
                options = " · ".join(
                    f"[{index}] {option}{' (recommended)' if index == item.recommended else ''}"
                    for index, option in enumerate(item.options)
                )
                lines.append(f"  options: {options}")
            if item.command:
                lines.append(f"  command: {item.command}")
    if not lines:
        return CommandResult(True, "no open items", {"items": []})
    return CommandResult(True, "\n".join(lines), {"items": data})


def _gang_intervene(
    fitems: Any, store: Any, fleet: Path, args: argparse.Namespace
) -> CommandResult:
    resolved = store.resolve_foreman(fleet, args.name)
    if not resolved.ok:
        return _fail("intervene", resolved.error)
    choice: int | str = args.choose
    try:
        choice = int(args.choose)
    except (TypeError, ValueError):
        pass  # an option's exact text is also a valid choice
    answered = fitems.answer_item(resolved.value, args.item, choice=choice, note=args.note)
    if not answered.ok:
        return _fail("intervene", answered.error)
    item = answered.value
    picked = (
        item.options[item.answer]
        if item.answer is not None and 0 <= item.answer < len(item.options)
        else item.answer
    )
    return CommandResult(
        True,
        f"{args.name}: {item.id} answered [{picked}]",
        {"name": args.name, "item": _plain(item)},
    )


def _gang_snooze(fitems: Any, store: Any, fleet: Path, args: argparse.Namespace) -> CommandResult:
    resolved = store.resolve_foreman(fleet, args.name)
    if not resolved.ok:
        return _fail("snooze", resolved.error)
    until = _local_until(args.until)
    if until is None:
        return _fail("snooze", f"--until {args.until!r}: a local time (ISO-8601 or bare HH:MM)")
    snoozed = fitems.snooze_item(resolved.value, args.item, until=until)
    if not snoozed.ok:
        return _fail("snooze", snoozed.error)
    item = snoozed.value
    return CommandResult(
        True,
        f"{args.name}: {item.id} snoozed until {_local_stamp(item.snoozed_until)}",
        {"name": args.name, "item": _plain(item)},
    )


def _gang_dismiss(fitems: Any, store: Any, fleet: Path, args: argparse.Namespace) -> CommandResult:
    resolved = store.resolve_foreman(fleet, args.name)
    if not resolved.ok:
        return _fail("dismiss", resolved.error)
    dismissed = fitems.dismiss_item(resolved.value, args.item)
    if not dismissed.ok:
        return _fail("dismiss", dismissed.error)
    return CommandResult(
        True,
        f"{args.name}: {dismissed.value.id} dismissed",
        {"name": args.name, "item": _plain(dismissed.value)},
    )


def _gang_pepe_silvia(store: Any, fpaths: Any, fleet: Path, name: str) -> CommandResult:
    resolved = store.resolve_foreman(fleet, name)
    if not resolved.ok:
        return _fail("pepe-silvia", resolved.error)
    try:
        turns = fpaths.turns_root(resolved.value)
    except ValueError as exc:
        return _fail("pepe-silvia", str(exc))
    briefings = (
        [path for path in turns.glob(f"*/{fpaths.BRIEFING_FILENAME}") if path.is_file()]
        if turns.is_dir()
        else []
    )
    if not briefings:
        return CommandResult(True, f"{name}: no Turn has run", {"briefing": None})

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return -1.0  # a briefing racing a delete sorts last, never raises

    newest = max(briefings, key=lambda p: (_mtime(p), p.parent.name))
    try:
        text = newest.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return _fail("pepe-silvia", f"{newest}: unreadable: {exc}")
    return CommandResult(
        True, f"{newest}:\n\n{text}", {"name": name, "path": str(newest), "briefing": text}
    )


def _gang_dayman(fitems: Any, store: Any, fleet: Path, name: str | None) -> CommandResult:
    targets, error = _foreman_targets(store, fleet, name)
    if error is not None:
        return _fail("dayman", error)
    blocks: list[str] = []
    data: list[dict[str, Any]] = []
    for fname, fdir in targets:
        reports = [item for item in fitems.load_items(fdir) if item.kind == fitems.ITEM_REPORT]
        if not reports:
            continue
        newest = max(
            reports,
            key=lambda item: (
                _parse_utc(item.created_at) or datetime.min.replace(tzinfo=UTC),
                item.id,
            ),
        )
        data.append({"foreman": fname, "item": _plain(newest)})
        blocks.append(
            f"{fname} {newest.id} [{_local_stamp(newest.created_at)}]\n{newest.text or ''}"
        )
    if not blocks:
        scope = f"{name}: " if name else ""
        return CommandResult(True, f"{scope}no report items", {"reports": []})
    return CommandResult(True, "\n\n".join(blocks), {"reports": data})


def _gang_set_state(
    fstate: Any, store: Any, fleet: Path, name: str, *, resume: bool
) -> CommandResult:
    verb = "resume" if resume else "pause"
    resolved = store.resolve_foreman(fleet, name)
    if not resolved.ok:
        return _fail(verb, resolved.error)
    fdir = resolved.value
    state = fstate.read_state_snapshot(fdir)
    updates: dict[str, Any] = {"state": "active" if resume else "paused"}
    if resume:
        # Resuming also clears ``faulted`` (which becomes active here) and the
        # consecutive-Turn-failure count that put it there (design §9).
        updates["consecutive_failures"] = 0
    # The foreman-state store is not the orchestrator's gated mutator layer —
    # write-gate-exempt(issue=3121): name-matched only, no write_gate receiver here.
    saved = fstate.save_state(fdir, fstate.with_updates(state, **updates))
    if not saved.ok:
        return _fail(verb, saved.error)
    return CommandResult(
        True,
        f"{name}: {saved.value.state}",
        {"name": name, "state": saved.value.state},
    )


def run_gang_command(args: argparse.Namespace) -> CommandResult:
    """Dispatch ``charlie gang <sub>``; all failures come back as values."""
    # The one sanctioned edge into charlie_work.foreman: a function-local
    # import so `import charlie_work.cli` never loads the foreman package.
    from .foreman import charter as fcharter
    from .foreman import display_state
    from .foreman import items as fitems
    from .foreman import journal as fjournal
    from .foreman import paths as fpaths
    from .foreman import state as fstate
    from .foreman import store

    fleet = fleet_dir(override=args.fleet_dir)
    command = args.gang_command
    if command == "new":
        return _gang_new(args, store, fleet)
    if command == "list":
        return _gang_list(fcharter, fitems, fstate, store, display_state, fleet)
    if command == "show":
        return _gang_show(fcharter, fitems, fstate, store, display_state, fleet, args.name)
    if command == "say":
        return _gang_say(fjournal, store, fleet, args.name, args.text)
    if command == "signal":
        return _gang_signal(fjournal, store, fleet, args)
    if command == "interventions":
        return _gang_interventions(fitems, store, fleet, args.name)
    if command == "intervene":
        return _gang_intervene(fitems, store, fleet, args)
    if command == "snooze":
        return _gang_snooze(fitems, store, fleet, args)
    if command == "dismiss":
        return _gang_dismiss(fitems, store, fleet, args)
    if command == "pepe-silvia":
        return _gang_pepe_silvia(store, fpaths, fleet, args.name)
    if command == "dayman":
        return _gang_dayman(fitems, store, fleet, args.name)
    if command in ("pause", "resume"):
        return _gang_set_state(fstate, store, fleet, args.name, resume=(command == "resume"))
    return CommandResult(False, f"unknown gang command: {command}", {})
