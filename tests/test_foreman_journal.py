"""Append-only logs: journal, inbox, signals (issue #3117)."""

from __future__ import annotations

from pathlib import Path

from charlie_work.foreman import paths
from charlie_work.foreman.journal import (
    append_inbox,
    append_journal,
    append_signal,
    read_inbox,
    read_journal,
    read_signals,
)


def _foreman(tmp_path: Path) -> Path:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    return fdir


def test_journal_append_and_read(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    first = append_journal(fdir, "turn opened", turn_id="t-1")
    second = append_journal(fdir, "turn closed", ts="2026-11-25T10:00:00Z")
    assert first.ok and second.ok
    assert (first.value.id, second.value.id) == (1, 2)
    entries = read_journal(fdir)
    assert [e.text for e in entries] == ["turn opened", "turn closed"]
    assert entries[0].turn_id == "t-1"
    assert entries[1].ts == "2026-11-25T10:00:00Z"
    assert read_journal(fdir, limit=1)[0].text == "turn closed"


def test_inbox_unread_since(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    append_inbox(fdir, "first")
    append_inbox(fdir, "second")
    append_inbox(fdir, "third")
    assert [m.text for m in read_inbox(fdir)] == ["first", "second", "third"]
    assert [m.text for m in read_inbox(fdir, after_id=2)] == ["third"]
    assert read_inbox(fdir, after_id=3) == ()


def test_signals_unread_since(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    append_signal(fdir, "github-push", {"ref": "main"})
    append_signal(fdir, "schedule", {})
    unread = read_signals(fdir, after_id=1)
    assert len(unread) == 1
    assert unread[0].source == "schedule"
    assert unread[0].id == 2
    assert read_signals(fdir)[0].data == {"ref": "main"}


def test_torn_final_line_is_skipped(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    append_inbox(fdir, "kept")
    inbox = fdir / "inbox.jsonl"
    with inbox.open("a", encoding="utf-8") as handle:
        handle.write('{"id": 2, "ts": "x", "text": "torn')
    assert [m.text for m in read_inbox(fdir)] == ["kept"]
    # A torn write does not reset the id sequence.
    appended = append_inbox(fdir, "next")
    assert appended.ok and appended.value.id == 2


def test_missing_files_read_empty(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    assert read_journal(fdir) == ()
    assert read_inbox(fdir) == ()
    assert read_signals(fdir) == ()
