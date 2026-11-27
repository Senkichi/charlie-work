"""Append-only logs: journal, inbox, signals (issue #3117)."""

from __future__ import annotations

from pathlib import Path

from charlie_work.foreman import paths
from charlie_work.foreman.journal import (
    SIGNAL_MAX_PAYLOAD_BYTES,
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


def test_signal_payload_cap_refuses_oversized_and_accepts_at_cap(
    tmp_path: Path,
) -> None:
    """The 16 KB cap is on the serialized ``data`` payload (issue #3121): over
    is a refusal that appends nothing, at is accepted."""
    fdir = _foreman(tmp_path)
    oversized = {"blob": "x" * SIGNAL_MAX_PAYLOAD_BYTES}
    refused = append_signal(fdir, "ci", oversized)
    assert not refused.ok
    assert str(SIGNAL_MAX_PAYLOAD_BYTES) in refused.error
    assert read_signals(fdir) == ()
    # Largest payload still under the cap: encoded size exactly at the limit.
    fit = "y" * (SIGNAL_MAX_PAYLOAD_BYTES - len('{"blob": ""}'))
    accepted = append_signal(fdir, "ci", {"blob": fit})
    assert accepted.ok, accepted.error
    assert read_signals(fdir)[0].data["blob"] == fit


def test_signal_unserializable_payload_is_an_error_value(tmp_path: Path) -> None:
    """A payload ``json.dumps`` cannot encode refuses instead of raising."""
    fdir = _foreman(tmp_path)
    refused = append_signal(fdir, "ci", {"when": object()})
    assert not refused.ok and "serializable" in refused.error
    assert read_signals(fdir) == ()


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


def test_append_after_torn_line_is_readable(tmp_path: Path) -> None:
    """A torn final line cannot swallow the next append: the new record lands
    on its own line (a leading ``\\n`` is written first) and reads back."""
    fdir = _foreman(tmp_path)
    journal = fdir / "journal.jsonl"
    append_journal(fdir, "kept")
    with journal.open("a", encoding="utf-8") as handle:
        handle.write('{"id": 2, "ts": "x", "text": "torn')
    appended = append_journal(fdir, "after-torn")
    assert appended.ok, appended.error
    entries = read_journal(fdir)
    assert [e.text for e in entries] == ["kept", "after-torn"]
    assert appended.value.id == 2


def test_undecodable_middle_line_is_skipped(tmp_path: Path) -> None:
    """Decoding is per line: one invalid byte costs its line, not the stream.

    A whole-file strict decode would read the log as empty, let the next
    append report ``ok`` with a duplicate id 1, and strand every
    ``after_id >= 1`` cursor forever.
    """
    fdir = _foreman(tmp_path)
    append_inbox(fdir, "first")
    append_inbox(fdir, "second")
    inbox = fdir / "inbox.jsonl"
    first, second = inbox.read_bytes().split(b"\n")[:2]
    inbox.write_bytes(first + b"\n" + b'\xff\xfe{"id": 9' + b"\n" + second + b"\n")
    assert [m.text for m in read_inbox(fdir)] == ["first", "second"]
    appended = append_inbox(fdir, "third")
    assert appended.ok and appended.value.id == 3


def test_unreadable_log_is_an_error_not_an_id_reset(tmp_path: Path, monkeypatch) -> None:
    """An OSError reading an existing log is an error value — the append must
    not restart ids at 1 and must not write."""
    fdir = _foreman(tmp_path)
    assert append_inbox(fdir, "kept").ok
    inbox = fdir / "inbox.jsonl"
    before = inbox.read_bytes()

    def _unreadable(self, *args, **kwargs):
        raise OSError("simulated read failure")

    monkeypatch.setattr(Path, "read_bytes", _unreadable)
    appended = append_inbox(fdir, "lost")
    assert not appended.ok
    assert "unreadable" in appended.error
    monkeypatch.undo()
    assert inbox.read_bytes() == before


def test_append_to_missing_foreman_dir_is_an_error(tmp_path: Path) -> None:
    """The lock acquire mkdirs the log's parent — appending to a removed or
    never-created foreman must refuse first, not silently re-materialize a
    charterless ``foremen/<name>/`` that ``gang list`` would then show."""
    fdir = paths.foreman_dir(tmp_path, "ghost")
    assert not append_journal(fdir, "x").ok
    assert not append_inbox(fdir, "x").ok
    assert not append_signal(fdir, "source").ok
    assert not fdir.exists()


def test_append_lock_busy_is_an_error_value(tmp_path: Path, monkeypatch) -> None:
    """A held log lock makes the append a bounded error — never a hang."""
    from charlie_work.file_lock import try_acquire_byte_range_lock
    from charlie_work.foreman import journal as journal_mod

    monkeypatch.setattr(journal_mod, "_LOCK_RETRIES", 2)
    fdir = _foreman(tmp_path)
    held = try_acquire_byte_range_lock(fdir / "journal.jsonl.lock")
    assert held is not None
    blocked = append_journal(fdir, "x")
    assert not blocked.ok
    assert "lock busy" in blocked.error
    held.release()
    assert append_journal(fdir, "x").ok


def test_concurrent_appends_get_unique_ids(tmp_path: Path) -> None:
    """``_next_id`` + ``_append`` run under the lock: no interleaved allocation."""
    import threading

    fdir = _foreman(tmp_path)
    results = []
    threads = [
        threading.Thread(target=lambda i=i: results.append(append_inbox(fdir, f"m{i}")))
        for i in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(r.ok for r in results)
    ids = sorted(r.value.id for r in results)
    assert ids == list(range(1, 9))


def test_missing_files_read_empty(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    assert read_journal(fdir) == ()
    assert read_inbox(fdir) == ()
    assert read_signals(fdir) == ()
