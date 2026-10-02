"""Regression tests for ``charlie_work.atomic_write`` (issue #2265).

The shared primitive replaced ~30 hand-rolled ``path.with_suffix(".tmp")``
writers that all shared ONE temp name per destination: two writers saving
the same file at once opened/truncated the same tmp file and raced on the
rename, surfacing on Windows as ``PermissionError`` [WinError 5] (observed
on ``http-etag-cache.json``; the retry mirrors ``state.save_state``'s,
which has carried it since issue #1062).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from charlie_work import atomic_write
from charlie_work.atomic_write import (
    write_bytes_atomic,
    write_json_atomic,
    write_text_atomic,
)


def test_write_json_atomic_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"  # parent does not exist yet
    write_json_atomic(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    # indent=2 + trailing newline, matching the writers it replaced.
    assert path.read_text(encoding="utf-8").endswith("}\n")
    assert not list(tmp_path.rglob("*.tmp"))


def test_write_text_and_bytes_atomic(tmp_path: Path) -> None:
    text_path = tmp_path / "note.md"
    write_text_atomic(text_path, "hello\nworld")
    assert text_path.read_text(encoding="utf-8") == "hello\nworld"

    bin_path = tmp_path / "blob.bin"
    write_bytes_atomic(bin_path, b"\x00\x01\xff")
    assert bin_path.read_bytes() == b"\x00\x01\xff"


def test_each_write_uses_a_unique_temp_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two writes to the same destination must not share a temp file name --
    the fixed ``<name>.tmp`` name was the issue-#2265 collision vector."""
    path = tmp_path / "cache.json"
    used: list[Path] = []
    real_replace = Path.replace

    def _spy(self: Path, target: object) -> Path:
        used.append(self)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _spy)
    write_json_atomic(path, {"n": 1})
    write_json_atomic(path, {"n": 2})

    assert len(used) == 2
    assert used[0] != used[1]
    for tmp in used:
        # Same directory as the destination (a rename must stay on one
        # filesystem), prefixed by the destination stem, and still matching
        # the ``*.json.tmp`` orphan-sweep glob.
        assert tmp.parent == path.parent
        assert tmp.name.startswith(f"{path.stem}.")
        assert tmp.name.endswith(f"{path.suffix}.tmp")


def test_concurrent_writers_use_distinct_temp_files(tmp_path: Path) -> None:
    """Two threads writing the same destination at once must each rename a
    DIFFERENT tmp file; both writes succeed, the result is one intact
    payload, and no tmp file survives."""
    path = tmp_path / "cache.json"
    barrier = threading.Barrier(2, timeout=30)
    rendezvous_done = threading.Event()
    hook_entered = threading.Event()
    used: list[Path] = []
    real_replace = Path.replace
    errors: list[BaseException] = []

    # A foreign Path.replace fired from a background thread inside the
    # patched window -- the api-worker terminal-record writer shape that
    # flaked this test on #2283. Lives under tmp_path/"foreign" so its
    # parent differs from the test writers' directory.
    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    foreign_tmp = foreign_dir / "issue-1514.api.terminal.0000.json.tmp"
    foreign_dst = foreign_dir / "issue-1514.api.terminal.json"
    foreign_tmp.write_text("{}\n", encoding="utf-8")

    def _blocking_replace(self: Path, target: object) -> Path:
        # The patch is on the Path CLASS, so every Path.replace in the
        # process lands here -- including a leftover background thread from
        # another test on the same xdist worker. Foreign calls must pass
        # straight through: recording one inflates `used`, and letting one
        # reach the barrier steals a rendezvous slot from the real writers
        # (issue #2284).
        if self.parent != tmp_path:
            return real_replace(self, target)
        used.append(self)
        hook_entered.set()
        # Rendezvous the writers' first replace calls *while both tmp files
        # exist* -- with a shared tmp name (the pre-#2265 code) both threads
        # would be holding the same file. A retried replace (transient
        # PermissionError -- the exact Windows condition the retry exists
        # for) re-enters this hook, so it must pass straight through or the
        # barrier would deadlock on its second phase.
        if not rendezvous_done.is_set():
            barrier.wait()
            rendezvous_done.set()
        return real_replace(self, target)

    def _write(payload: dict) -> None:
        try:
            write_json_atomic(path, payload)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    def _foreign_replace() -> None:
        try:
            # Fire while a writer is inside the hook -- the interleaving
            # that flaked #2283. Without the tmp_path filter in
            # _blocking_replace this call is recorded in `used` and can
            # steal a barrier slot.
            hook_entered.wait(timeout=30)
            foreign_tmp.replace(foreign_dst)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(Path, "replace", _blocking_replace)
    threads = [threading.Thread(target=_write, args=({"writer": i},)) for i in range(2)]
    foreign = threading.Thread(target=_foreign_replace)
    try:
        foreign.start()
        for thread in threads:
            thread.start()
        # foreign is joined inside the patched window so its replace is
        # guaranteed to run under the monkeypatch.
        for thread in (*threads, foreign):
            thread.join(timeout=30)
    finally:
        monkeypatch.undo()

    assert not any(thread.is_alive() for thread in (*threads, foreign)), "barrier deadlock"
    assert errors == []
    # The foreign write passed through untouched: it renamed its own file
    # and was never recorded or rendezvoused.
    assert foreign_dst.read_text(encoding="utf-8") == "{}\n"
    assert not foreign_tmp.exists()
    # Two DISTINCT tmp files coexisted at the barrier (a PermissionError
    # retry re-uses the same tmp, so compare the set, not the count).
    assert len(set(used)) == 2
    assert used[0] != used[1]
    # Last writer wins -- the file parses and is one complete payload,
    # never a torn mix of the two.
    assert json.loads(path.read_text(encoding="utf-8")) in ({"writer": 0}, {"writer": 1})
    assert not list(tmp_path.glob("*.tmp"))


def test_permission_error_on_replace_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient PermissionError on the rename (a lock-free reader or AV
    scanner holding the destination open on Windows) retries and succeeds."""
    path = tmp_path / "cache.json"
    path.write_text('{"old": true}', encoding="utf-8")
    calls: list[Path] = []
    real_replace = Path.replace

    def _flaky(self: Path, target: object) -> Path:
        calls.append(self)
        if len(calls) <= 2:
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", _flaky)
    write_json_atomic(path, {"new": True})

    assert len(calls) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_permission_error_exhaustion_raises_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persistent PermissionError surfaces after REPLACE_ATTEMPTS tries;
    the destination stays intact and the unique tmp file is unlinked."""
    path = tmp_path / "cache.json"
    path.write_text('{"old": true}', encoding="utf-8")
    calls: list[Path] = []

    def _failing(self: Path, target: object) -> Path:
        calls.append(self)
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", _failing)
    with pytest.raises(PermissionError):
        write_json_atomic(path, {"new": True})

    assert len(calls) == atomic_write.REPLACE_ATTEMPTS
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_non_permission_oserror_does_not_retry_but_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-transient OSError is not a sharing violation: propagate on the
    first attempt (no pointless backoff) and still remove the tmp file."""
    path = tmp_path / "cache.json"
    calls: list[Path] = []

    def _failing(self: Path, target: object) -> Path:
        calls.append(self)
        raise OSError("disk gone")

    monkeypatch.setattr(Path, "replace", _failing)
    with pytest.raises(OSError, match="disk gone"):
        write_json_atomic(path, {"x": 1})

    assert len(calls) == 1
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_serialization_failure_leaves_no_destination_and_no_orphan(tmp_path: Path) -> None:
    """A value that fails mid-``json.dump`` (e.g. a Path not coerced to str --
    the issue-#1184 failure class) must leave neither the destination nor a
    stranded tmp file."""
    path = tmp_path / "record.json"
    with pytest.raises(TypeError):
        write_json_atomic(path, {"bad": Path("/not/serializable")})

    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))
