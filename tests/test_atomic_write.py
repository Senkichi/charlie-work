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

import conftest
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


def test_each_write_uses_a_unique_temp_name(tmp_path: Path, patch_path_replace) -> None:
    """Two writes to the same destination must not share a temp file name --
    the fixed ``<name>.tmp`` name was the issue-#2265 collision vector."""
    path = tmp_path / "cache.json"
    used: list[Path] = []
    real_replace = Path.replace

    def _spy(self: Path, target: object) -> Path:
        used.append(self)
        return real_replace(self, target)

    patch_path_replace(_spy, scope=tmp_path)
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


def test_concurrent_writers_use_distinct_temp_files(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
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
    # flaked this test on #2283. Lives in its own basetemp dir so it is
    # outside the patch's scope, exactly like a leftover thread from a
    # different test.
    foreign_dir = tmp_path_factory.mktemp("foreign")
    foreign_tmp = foreign_dir / "issue-1514.api.terminal.0000.json.tmp"
    foreign_dst = foreign_dir / "issue-1514.api.terminal.json"
    foreign_tmp.write_text("{}\n", encoding="utf-8")

    def _blocking_replace(self: Path, target: object) -> Path:
        # Only in-scope calls arrive here: patch_path_replace delegates
        # every out-of-scope Path.replace (like the foreign thread's below)
        # to the real method, so it can neither inflate `used` nor steal a
        # barrier rendezvous slot (issue #2284).
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
            # that flaked #2283. The path is outside the patch scope, so the
            # scoped wrapper delegates it untouched; under a raw class-wide
            # patch it would be recorded in `used` and could steal a
            # barrier slot.
            hook_entered.wait(timeout=30)
            foreign_tmp.replace(foreign_dst)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    monkeypatch = pytest.MonkeyPatch()
    conftest.patch_path_replace(monkeypatch, _blocking_replace, scope=tmp_path)
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
    tmp_path: Path, patch_path_replace
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

    patch_path_replace(_flaky, scope=tmp_path)
    write_json_atomic(path, {"new": True})

    assert len(calls) == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_permission_error_exhaustion_raises_and_cleans_up(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, patch_path_replace
) -> None:
    """Persistent PermissionError surfaces after REPLACE_ATTEMPTS tries;
    the destination stays intact and the unique tmp file is unlinked.

    The call count is asserted against a FOREIGN ``Path.replace`` fired on
    a background thread during the patched window (issue #2290): the
    class-wide patch is process-wide, and the pre-helper version of this
    test flaked when another thread's replace landed in ``_failing`` (run
    37005463625 saw 4 calls, not 3)."""
    path = tmp_path / "cache.json"
    path.write_text('{"old": true}', encoding="utf-8")
    calls: list[Path] = []
    hook_entered = threading.Event()
    foreign_done = threading.Event()
    errors: list[BaseException] = []

    # Outside the patch scope, like a leftover writer from another test.
    foreign_dir = tmp_path_factory.mktemp("foreign")
    foreign_tmp = foreign_dir / "foreign.json.tmp"
    foreign_dst = foreign_dir / "foreign.json"
    foreign_tmp.write_text("{}\n", encoding="utf-8")

    def _failing(self: Path, target: object) -> Path:
        calls.append(self)
        # Hold the window open on the first call so the foreign thread's
        # replace is guaranteed to land while the patch is installed and a
        # fake call is in flight.
        hook_entered.set()
        foreign_done.wait(timeout=30)
        raise PermissionError(5, "Access is denied")

    def _foreign_replace() -> None:
        try:
            hook_entered.wait(timeout=30)
            foreign_tmp.replace(foreign_dst)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)
        finally:
            foreign_done.set()

    patch_path_replace(_failing, scope=tmp_path)
    foreign = threading.Thread(target=_foreign_replace)
    foreign.start()
    try:
        with pytest.raises(PermissionError):
            write_json_atomic(path, {"new": True})
    finally:
        foreign.join(timeout=30)

    assert not foreign.is_alive(), "foreign replace thread wedged"
    assert errors == []
    # The foreign replace passed through the scoped wrapper untouched.
    assert foreign_dst.read_text(encoding="utf-8") == "{}\n"
    assert not foreign_tmp.exists()
    assert len(calls) == atomic_write.REPLACE_ATTEMPTS
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_non_permission_oserror_does_not_retry_but_cleans_up(
    tmp_path: Path, patch_path_replace
) -> None:
    """A non-transient OSError is not a sharing violation: propagate on the
    first attempt (no pointless backoff) and still remove the tmp file."""
    path = tmp_path / "cache.json"
    calls: list[Path] = []

    def _failing(self: Path, target: object) -> Path:
        calls.append(self)
        raise OSError("disk gone")

    patch_path_replace(_failing, scope=tmp_path)
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
