"""The pooled HTTP transport's ETag cache (``github_capabilities.http_cache``).

No network and no transport here: these tests drive the cache's own functions
against files under ``tmp_path``. The transport's use of the cache (a
conditional GET served from a 304) is covered in ``test_http_transport.py``.

Covers: round trip and fail-closed reads; the version 2 layout (a small index
plus one body file per entry, issue #3138); what a lookup and a record each
touch on disk; eviction by entry count and by total body bytes; and concurrent
index saves (issue #2265).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from charlie_work.github_capabilities import http_cache


def test_cache_round_trip(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    link = '<https://api.github.com/repos/a/b?page=2>; rel="next"'
    assert http_cache.get_cached(path, "/repos/a/b") is None
    http_cache.record_response(
        path, "/repos/a/b", etag='"abc"', status=200, body='{"ok":true}', link=link
    )
    cached = http_cache.get_cached(path, "/repos/a/b")
    assert cached is not None
    assert cached.etag == '"abc"'
    assert cached.body == '{"ok":true}'
    assert cached.link == link


def test_cache_corrupt_file_fails_closed(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    path.write_text("not json{{{", encoding="utf-8")
    assert http_cache.load_cache(path) == {}
    assert http_cache.get_cached(path, "/x") is None


def test_cache_evicts_oldest_beyond_max_entries(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    monkeypatch.setattr(http_cache, "_MAX_ENTRIES", 2)
    http_cache.record_response(path, "/a", etag="1", status=200, body="a")
    http_cache.record_response(path, "/b", etag="2", status=200, body="b")
    http_cache.record_response(path, "/c", etag="3", status=200, body="c")
    entries = http_cache.load_cache(path)
    assert len(entries) == 2
    assert "/c" in entries  # most recent survives


def _etag_index(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _etag_body_files(path: Path) -> list[Path]:
    return sorted(http_cache._bodies_dir(path).iterdir())


def test_cache_index_holds_no_body_text(tmp_path: Path):
    """Issue #3138: bodies live out of line, so the index stays small however
    large the responses are."""
    path = tmp_path / "http-etag-cache.json"
    bodies = {f"/repos/a/b/pulls?page={n}": f"page-{n}-" + "x" * 200_000 for n in range(4)}
    for request_path, body in bodies.items():
        http_cache.record_response(path, request_path, etag=f'"{body[:7]}"', status=200, body=body)

    raw = path.read_text(encoding="utf-8")
    index = _etag_index(path)
    assert index["version"] == 2
    assert set(index["entries"]) == set(bodies)
    assert all("body" not in entry for entry in index["entries"].values())
    assert "x" * 100 not in raw
    assert len(raw.encode("utf-8")) < 1024 * len(bodies)
    assert {key: cached.body for key, cached in http_cache.load_cache(path).items()} == bodies


def test_cache_record_touches_only_its_own_body_file(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    http_cache.record_response(path, "/a", etag='"a1"', status=200, body="a" * 1000)
    (other,) = _etag_body_files(path)
    before = other.stat().st_mtime_ns

    http_cache.record_response(path, "/b", etag='"b1"', status=200, body="b" * 1000)

    assert other.stat().st_mtime_ns == before
    assert len(_etag_body_files(path)) == 2


def test_cache_get_opens_only_its_own_body_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    http_cache.record_response(path, "/a", etag='"a1"', status=200, body="body-a")
    http_cache.record_response(path, "/b", etag='"b1"', status=200, body="body-b")
    wanted = http_cache._bodies_dir(path) / _etag_index(path)["entries"]["/a"]["body_file"]
    read: list[Path] = []
    real_read_bytes = Path.read_bytes

    def _spy(self: Path) -> bytes:
        read.append(self)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _spy)
    cached = http_cache.get_cached(path, "/a")

    assert cached is not None and cached.body == "body-a"
    assert read == [wanted]


def test_cache_get_parses_an_unchanged_index_once(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    http_cache.record_response(path, "/a", etag='"a1"', status=200, body="body-a")
    parses: list[Path] = []
    real_parse = http_cache._parse_index

    def _counting(cache_path: Path):
        parses.append(cache_path)
        return real_parse(cache_path)

    monkeypatch.setattr(http_cache, "_parse_index", _counting)
    for _ in range(5):
        assert http_cache.get_cached(path, "/a") is not None
    assert len(parses) <= 1

    # A changed index is picked up: the memo is keyed on the file, not on time.
    http_cache.record_response(path, "/b", etag='"b1"', status=200, body="body-b")
    assert http_cache.get_cached(path, "/b") is not None


def test_cache_missing_body_file_is_a_miss(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    http_cache.record_response(path, "/a", etag='"a1"', status=200, body="body-a")
    (body_file,) = _etag_body_files(path)
    body_file.unlink()

    assert http_cache.get_cached(path, "/a") is None
    assert http_cache.load_cache(path) == {}


def test_cache_truncated_body_file_is_a_miss(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    http_cache.record_response(path, "/a", etag='"a1"', status=200, body="body-a")
    (body_file,) = _etag_body_files(path)
    body_file.write_bytes(b"bod")

    assert http_cache.get_cached(path, "/a") is None


def test_cache_body_file_name_outside_the_bodies_dir_is_ignored(tmp_path: Path):
    """A corrupt index must never make the cache read an arbitrary file."""
    path = tmp_path / "http-etag-cache.json"
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    entry = {
        "etag": '"a1"',
        "status": 200,
        "stored_at": "t",
        "link": None,
        "body_file": "../secret.txt",
        "body_bytes": 6,
    }
    path.write_text(json.dumps({"version": 2, "entries": {"/a": entry}}), encoding="utf-8")

    assert http_cache.get_cached(path, "/a") is None


def test_cache_version_1_file_reads_empty_and_is_replaced(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    old_entry = {
        "etag": '"o"',
        "status": 200,
        "body": "inline body",
        "stored_at": "2026-01-01T00:00:00Z",
        "link": None,
    }
    path.write_text(json.dumps({"version": 1, "entries": {"/old": old_entry}}), encoding="utf-8")
    assert http_cache.load_cache(path) == {}
    assert http_cache.get_cached(path, "/old") is None

    http_cache.record_response(path, "/new", etag='"n"', status=200, body="new body")

    index = _etag_index(path)
    assert index["version"] == 2
    assert set(index["entries"]) == {"/new"}
    assert "inline body" not in path.read_text(encoding="utf-8")


def test_cache_eviction_by_count_removes_the_body_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    monkeypatch.setattr(http_cache, "_MAX_ENTRIES", 2)
    for name in ("/a", "/b", "/c"):
        http_cache.record_response(path, name, etag=f'"{name}"', status=200, body=name)

    index = _etag_index(path)["entries"]
    assert set(index) == {"/b", "/c"}
    assert {p.name for p in _etag_body_files(path)} == {e["body_file"] for e in index.values()}


def test_cache_eviction_by_byte_budget_removes_the_body_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    monkeypatch.setattr(http_cache, "_MAX_BODY_BYTES", 250)
    for name in ("/a", "/b", "/c"):
        http_cache.record_response(path, name, etag=f'"{name}"', status=200, body=name * 50)

    index = _etag_index(path)["entries"]
    assert set(index) == {"/b", "/c"}
    assert sum(entry["body_bytes"] for entry in index.values()) <= 250
    assert {p.name for p in _etag_body_files(path)} == {e["body_file"] for e in index.values()}


def test_cache_keeps_a_single_body_larger_than_the_byte_budget(tmp_path: Path, monkeypatch):
    path = tmp_path / "http-etag-cache.json"
    monkeypatch.setattr(http_cache, "_MAX_BODY_BYTES", 10)
    http_cache.record_response(path, "/small", etag='"s"', status=200, body="s")
    http_cache.record_response(path, "/big", etag='"b"', status=200, body="b" * 100)

    assert set(_etag_index(path)["entries"]) == {"/big"}
    cached = http_cache.get_cached(path, "/big")
    assert cached is not None and cached.body == "b" * 100


def test_cache_rerecord_replaces_the_superseded_body(tmp_path: Path):
    path = tmp_path / "http-etag-cache.json"
    http_cache.record_response(path, "/a", etag='"v1"', status=200, body="one")
    http_cache.record_response(path, "/a", etag='"v2"', status=200, body="two")

    cached = http_cache.get_cached(path, "/a")
    assert cached is not None and (cached.etag, cached.body) == ('"v2"', "two")
    assert len(_etag_body_files(path)) == 1


def test_cache_body_round_trips_exactly(tmp_path: Path):
    """Binary body files: no newline translation, non-ASCII intact."""
    path = tmp_path / "http-etag-cache.json"
    body = json.dumps({"a": "line1\r\nline2\n", "b": "café ☃"}, ensure_ascii=False)
    body = body + "\r\n"
    http_cache.record_response(path, "/a", etag='"e"', status=200, body=body)
    cached = http_cache.get_cached(path, "/a")
    assert cached is not None and cached.body == body


def test_save_cache_concurrent_writes_use_distinct_temps(tmp_path: Path, patch_path_replace):
    """Issue #2265: two concurrent ``_save_index`` calls shared one fixed
    ``http-etag-cache.json.tmp`` name, colliding on open/truncate/rename and
    surfacing ``PermissionError`` [WinError 5] on Windows. Each save now
    stages a unique temp file, so both complete and the destination holds
    one intact payload with no ``.tmp`` orphans."""
    path = tmp_path / "http-etag-cache.json"
    barrier = threading.Barrier(2, timeout=30)
    rendezvous_done = threading.Event()
    used: list[Path] = []
    real_replace = Path.replace
    errors: list[BaseException] = []

    def _blocking_replace(self: Path, target: object) -> Path:
        used.append(self)
        # Rendezvous the writers' first replace calls while BOTH temp files
        # exist -- under the fixed-name scheme this is exactly where the two
        # writers shared one file. A retried replace (transient
        # PermissionError -- the exact Windows condition the retry exists
        # for) re-enters this hook, so it must pass straight through or the
        # barrier would deadlock on its second phase.
        if not rendezvous_done.is_set():
            barrier.wait()
            rendezvous_done.set()
        return real_replace(self, target)

    def _save(etag: str) -> None:
        try:
            entry = http_cache._IndexEntry(
                etag=etag, status=200, stored_at="t", body_file="0" * 40 + ".body", body_bytes=2
            )
            http_cache._save_index(path, {"/p": entry})
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    patch_path_replace(_blocking_replace, scope=tmp_path)
    threads = [threading.Thread(target=_save, args=(f'"e{i}"',)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not any(thread.is_alive() for thread in threads), "barrier deadlock"
    assert errors == []
    # Two DISTINCT tmp files coexisted at the barrier (a PermissionError
    # retry re-uses the same tmp, so compare the set, not the count).
    assert len(set(used)) == 2
    assert used[0] != used[1]
    loaded = http_cache._load_index(path)
    assert set(loaded) == {"/p"}
    assert loaded["/p"].etag in {'"e0"', '"e1"'}
    assert not list(tmp_path.glob("*.tmp"))
