"""``charlie gang`` talk-and-answer commands (issue #3121).

``say``/``signal`` write the append-only logs, ``interventions``/``intervene``/
``snooze``/``dismiss`` work the item lifecycle, ``pepe-silvia``/``dayman`` read
the newest briefing and report, and ``pause``/``resume`` set the foreman state.
Every command is exercised end to end through ``run_gang_command`` on a real
temp fleet dir, and every failure comes back as a ``CommandResult`` value.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pytest

from charlie_work.cli import build_parser
from charlie_work.foreman import items as fitems
from charlie_work.foreman.journal import (
    SIGNAL_MAX_PAYLOAD_BYTES,
    read_inbox,
    read_signals,
)
from charlie_work.foreman.state import ForemanState, read_state_snapshot, save_state
from charlie_work.foreman.store import scaffold_foreman
from charlie_work.gang_command import run_gang_command


def _args(fleet: Path, command: str, **kw) -> argparse.Namespace:
    base = dict(
        fleet_dir=str(fleet),
        gang_command=command,
        name=None,
        from_dir=None,
        text=None,
        source=None,
        file=None,
        item=None,
        choose=None,
        note=None,
        until=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _foreman(fleet: Path, name: str = "g1") -> Path:
    created = scaffold_foreman(fleet, name)
    assert created.ok, created.error
    return created.value


def _decision(
    root: Path,
    item_id: str = "it-1",
    *,
    question: str = "Retry the failed issue?",
    options: tuple[str, ...] = ("retry smaller", "drop it"),
    recommended: int | None = 0,
    command: str | None = "charlie requeue",
) -> None:
    created = fitems.create_item(
        root,
        kind="decision",
        question=question,
        options=options,
        recommended=recommended,
        command=command,
        item_id=item_id,
    )
    assert created.ok, created.error


def test_parser_accepts_talk_and_answer_subcommands() -> None:
    parser = build_parser()
    a = parser.parse_args(["gang", "say", "g1", "focus on #12"])
    assert (a.gang_command, a.name, a.text) == ("say", "g1", "focus on #12")
    a = parser.parse_args(["gang", "signal", "g1", "--source", "ci", "--text", "{}"])
    assert (a.gang_command, a.source, a.text, a.file) == ("signal", "ci", "{}", None)
    a = parser.parse_args(["gang", "signal", "g1", "--source", "ci", "--file", "p.json"])
    assert (a.file, a.text) == ("p.json", None)
    assert parser.parse_args(["gang", "interventions"]).name is None
    assert parser.parse_args(["gang", "interventions", "g1"]).name == "g1"
    a = parser.parse_args(["gang", "intervene", "g1", "it-2", "--choose", "1", "--note", "why"])
    assert (a.item, a.choose, a.note) == ("it-2", "1", "why")
    a = parser.parse_args(["gang", "snooze", "g1", "it-2", "--until", "09:00"])
    assert a.until == "09:00"
    a = parser.parse_args(["gang", "dismiss", "g1", "it-2"])
    assert a.item == "it-2"
    assert parser.parse_args(["gang", "pepe-silvia", "g1"]).gang_command == "pepe-silvia"
    assert parser.parse_args(["gang", "dayman"]).name is None
    assert parser.parse_args(["gang", "pause", "g1"]).gang_command == "pause"
    assert parser.parse_args(["gang", "resume", "g1"]).gang_command == "resume"


def test_signal_requires_source_and_exactly_one_payload() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["gang", "signal", "g1", "--text", "{}"])  # no --source
    with pytest.raises(SystemExit):
        parser.parse_args(["gang", "signal", "g1", "--source", "ci"])  # no payload
    with pytest.raises(SystemExit):
        parser.parse_args(["gang", "signal", "g1", "--source", "ci", "--text", "x", "--file", "f"])


def test_gang_say_writes_inbox_and_never_signals(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    result = run_gang_command(_args(tmp_path, "say", name="g1", text="focus on #12"))
    assert result.ok, result.message
    assert [m.text for m in read_inbox(root)] == ["focus on #12"]
    assert read_signals(root) == ()
    assert not (root / "signals.jsonl").exists()


def test_gang_signal_writes_signals_and_never_inbox(tmp_path: Path) -> None:
    """A signal is data for the next briefing — it must never land in the
    operator-instruction stream the foreman reads as commands."""
    root = _foreman(tmp_path)
    result = run_gang_command(
        _args(tmp_path, "signal", name="g1", source="ci", text='{"job": "red"}')
    )
    assert result.ok, result.message
    signals = read_signals(root)
    assert len(signals) == 1
    assert signals[0].source == "ci"
    assert signals[0].data == {"job": "red"}
    assert read_inbox(root) == ()
    assert not (root / "inbox.jsonl").exists()


def test_gang_signal_wraps_plain_text_and_reads_payload_files(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    result = run_gang_command(
        _args(tmp_path, "signal", name="g1", source="cron", text="disk at 91%")
    )
    assert result.ok, result.message
    payload = tmp_path / "payload.json"
    payload.write_text('{"run": 7}', encoding="utf-8")
    result = run_gang_command(_args(tmp_path, "signal", name="g1", source="ci", file=str(payload)))
    assert result.ok, result.message
    signals = read_signals(root)
    assert signals[0].data == {"text": "disk at 91%"}
    assert signals[1].data == {"run": 7}
    missing = run_gang_command(
        _args(tmp_path, "signal", name="g1", source="ci", file=str(tmp_path / "none.json"))
    )
    assert not missing.ok
    assert "unreadable" in missing.message


def test_gang_signal_refuses_payload_over_cap(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    big = "x" * (SIGNAL_MAX_PAYLOAD_BYTES + 1)
    result = run_gang_command(_args(tmp_path, "signal", name="g1", source="ci", text=big))
    assert not result.ok
    assert str(SIGNAL_MAX_PAYLOAD_BYTES) in result.message
    assert read_signals(root) == ()


def test_gang_interventions_lists_open_items(tmp_path: Path) -> None:
    root = _foreman(tmp_path, "g1")
    _foreman(tmp_path, "g2")
    _decision(root)
    result = run_gang_command(_args(tmp_path, "interventions", name="g1"))
    assert result.ok, result.message
    assert "g1 it-1 [decision]" in result.message
    assert "(recommended)" in result.message
    assert "command: charlie requeue" in result.message
    # Without a name every foreman lists; g2 simply contributes nothing.
    result = run_gang_command(_args(tmp_path, "interventions"))
    assert result.ok
    assert "it-1" in result.message and "g2" not in result.message
    assert result.data["items"][0]["recommended"] == "retry smaller"
    # An answered item leaves the list.
    assert fitems.answer_item(root, "it-1", choice=1).ok
    result = run_gang_command(_args(tmp_path, "interventions"))
    assert result.ok and "no open items" in result.message


def test_gang_intervene_answers_by_index_and_option_text(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    _decision(root)
    result = run_gang_command(_args(tmp_path, "intervene", name="g1", item="it-1", choose="1"))
    assert result.ok and "answered [drop it]" in result.message
    item = fitems.load_items(root)[0]
    assert item.status == "answered" and item.answer == 1
    # An option's exact text is a valid choice too.
    _decision(root, "it-2")
    result = run_gang_command(
        _args(tmp_path, "intervene", name="g1", item="it-2", choose="retry smaller", note="try it")
    )
    assert result.ok, result.message
    item = fitems.load_items(root)[1]
    assert item.answer == 0 and item.answer_note == "try it"
    # Failures are values: already answered, bad choice, unknown item.
    assert not run_gang_command(
        _args(tmp_path, "intervene", name="g1", item="it-1", choose="0")
    ).ok
    _decision(root, "it-3")
    assert not run_gang_command(
        _args(tmp_path, "intervene", name="g1", item="it-3", choose="bogus")
    ).ok
    result = run_gang_command(_args(tmp_path, "intervene", name="g1", item="it-9", choose="0"))
    assert not result.ok and "not found" in result.message


def test_gang_snooze_hides_until_then_resurfaces(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    _decision(root)
    result = run_gang_command(
        _args(tmp_path, "snooze", name="g1", item="it-1", until="2099-01-01 09:00")
    )
    assert result.ok, result.message
    assert "snoozed until" in result.message
    result = run_gang_command(_args(tmp_path, "interventions", name="g1"))
    assert "no open items" in result.message
    # Local input is stored as UTC.
    assert fitems.load_items(root)[0].snoozed_until.endswith("Z")
    # A deadline in the past leaves the item open.
    result = run_gang_command(
        _args(tmp_path, "snooze", name="g1", item="it-1", until="2000-01-01 00:00")
    )
    assert result.ok
    result = run_gang_command(_args(tmp_path, "interventions", name="g1"))
    assert "it-1" in result.message
    # Bare HH:MM is a valid local time.
    result = run_gang_command(_args(tmp_path, "snooze", name="g1", item="it-1", until="23:59"))
    assert result.ok
    # Garbage input is a clean refusal; the item is unchanged.
    result = run_gang_command(_args(tmp_path, "snooze", name="g1", item="it-1", until="banana"))
    assert not result.ok
    result = run_gang_command(
        _args(tmp_path, "snooze", name="g1", item="it-9", until="2099-01-01")
    )
    assert not result.ok and "not found" in result.message


def test_gang_dismiss_closes_item(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    _decision(root)
    result = run_gang_command(_args(tmp_path, "dismiss", name="g1", item="it-1"))
    assert result.ok and "dismissed" in result.message
    assert fitems.load_items(root)[0].status == "dismissed"
    assert not run_gang_command(_args(tmp_path, "dismiss", name="g1", item="it-1")).ok
    assert not run_gang_command(_args(tmp_path, "dismiss", name="g1", item="it-9")).ok


def test_gang_pepe_silvia_prints_newest_briefing(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    result = run_gang_command(_args(tmp_path, "pepe-silvia", name="g1"))
    assert result.ok and "no Turn has run" in result.message
    turns = root / "turns"
    (turns / "t-001").mkdir()
    (turns / "t-002").mkdir()
    old = turns / "t-001" / "briefing.md"
    old.write_text("OLD BRIEFING", encoding="utf-8")
    new = turns / "t-002" / "briefing.md"
    new.write_text("NEW BRIEFING", encoding="utf-8")
    os.utime(old, (1_000_000_000, 1_000_000_000))
    os.utime(new, (1_700_000_000, 1_700_000_000))
    result = run_gang_command(_args(tmp_path, "pepe-silvia", name="g1"))
    assert result.ok
    assert "NEW BRIEFING" in result.message and "OLD BRIEFING" not in result.message


def test_gang_dayman_prints_newest_report(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    result = run_gang_command(_args(tmp_path, "dayman", name="g1"))
    assert result.ok and "no report items" in result.message
    assert fitems.create_item(
        root, kind="report", text="first", created_at="2026-01-01T00:00:00Z"
    ).ok
    assert fitems.create_item(
        root, kind="report", text="latest", created_at="2026-02-01T00:00:00Z"
    ).ok
    _decision(root, "it-7")  # a decision is not a report
    result = run_gang_command(_args(tmp_path, "dayman", name="g1"))
    assert result.ok
    assert "latest" in result.message and "first" not in result.message


def test_gang_pause_and_resume(tmp_path: Path) -> None:
    root = _foreman(tmp_path)
    result = run_gang_command(_args(tmp_path, "resume", name="g1"))
    assert result.ok and result.data["state"] == "active"
    result = run_gang_command(_args(tmp_path, "pause", name="g1"))
    assert result.ok and read_state_snapshot(root).state == "paused"
    # Resume clears faulted and the consecutive-failure count that put it there.
    assert save_state(root, ForemanState(state="faulted", consecutive_failures=5)).ok
    result = run_gang_command(_args(tmp_path, "resume", name="g1"))
    assert result.ok, result.message
    state = read_state_snapshot(root)
    assert state.state == "active" and state.consecutive_failures == 0


def test_gang_commands_on_missing_foreman_are_clean_errors(tmp_path: Path) -> None:
    cases = [
        ("say", {"text": "hi"}),
        ("signal", {"source": "ci", "text": "{}"}),
        ("interventions", {}),
        ("intervene", {"item": "it-1", "choose": "0"}),
        ("snooze", {"item": "it-1", "until": "2099-01-01"}),
        ("dismiss", {"item": "it-1"}),
        ("pepe-silvia", {}),
        ("dayman", {}),
        ("pause", {}),
        ("resume", {}),
    ]
    for command, kw in cases:
        result = run_gang_command(_args(tmp_path, command, name="ghost", **kw))
        assert not result.ok, command
        assert "not found" in result.message, command
