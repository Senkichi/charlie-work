"""``items.json`` typed mutations: create/answer/snooze/dismiss (issue #3117)."""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.foreman import paths
from charlie_work.foreman.items import (
    answer_item,
    create_item,
    dismiss_item,
    load_items,
    mark_notified,
    open_items,
    snooze_item,
)


def _foreman(tmp_path: Path) -> Path:
    fdir = paths.foreman_dir(tmp_path, "g1")
    fdir.mkdir(parents=True)
    return fdir


def test_create_decision_round_trip(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    created = create_item(
        fdir,
        kind="decision",
        question="Ship the new transport?",
        options=["ship", "hold"],
        recommended=0,
        created_at="2026-11-25T09:00:00Z",
    )
    assert created.ok, created.error
    assert created.value.id == "it-1"
    loaded = load_items(fdir)
    assert len(loaded) == 1
    item = loaded[0]
    assert item.kind == "decision"
    assert item.status == "open"
    assert item.options == ("ship", "hold")
    assert item.recommended == 0
    assert open_items(loaded) == (item,)


def test_ids_increment(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="one")
    second = create_item(fdir, kind="report", text="two")
    assert second.value.id == "it-2"


def test_validation_errors(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    assert not create_item(fdir, kind="wish", text="x").ok
    assert not create_item(fdir, kind="decision", question="q", options=["only one"]).ok
    assert not create_item(fdir, kind="decision", options=["a", "b"]).ok
    assert not create_item(fdir, kind="report", options=["a", "b"], text="t").ok
    assert not create_item(fdir, kind="report").ok
    bad_rec = create_item(fdir, kind="decision", question="q", options=["a", "b"], recommended=5)
    assert not bad_rec.ok and "recommended" in bad_rec.error
    # Errors leave the store untouched.
    assert load_items(fdir) == ()


def test_duplicate_id_rejected(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    assert create_item(fdir, kind="report", text="x", item_id="it-9").ok
    assert not create_item(fdir, kind="report", text="y", item_id="it-9").ok


def test_answer_by_index_and_text(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="decision", question="pick", options=["a", "b"])
    create_item(fdir, kind="permission", question="run it?", options=["approve", "deny"])
    answered = answer_item(fdir, "it-1", choice=1, note="because")
    assert answered.ok
    assert answered.value.status == "answered"
    assert answered.value.answer == 1
    assert answered.value.answer_note == "because"
    answered2 = answer_item(fdir, "it-2", choice="approve")
    assert answered2.ok and answered2.value.answer == 0
    # Answering a closed item is an error; a bad choice is too.
    assert not answer_item(fdir, "it-1", choice=0).ok
    create_item(fdir, kind="decision", question="again", options=["x", "y"])
    assert not answer_item(fdir, "it-3", choice="nope").ok
    assert not answer_item(fdir, "it-3", choice=7).ok


def test_snooze_then_reopen_on_expiry(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="later", created_at="t0")
    snoozed = snooze_item(fdir, "it-1", until="2026-11-26T09:00:00Z")
    assert snoozed.ok and snoozed.value.status == "snoozed"
    items = load_items(fdir)
    # Still snoozed before the deadline, open again after it.
    assert open_items(items, now="2026-11-25T00:00:00Z") == ()
    assert len(open_items(items, now="2026-11-27T00:00:00Z")) == 1


def test_dismiss_and_missing_item(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="bye")
    dismissed = dismiss_item(fdir, "it-1")
    assert dismissed.ok and dismissed.value.status == "dismissed"
    assert open_items(load_items(fdir)) == ()
    assert not dismiss_item(fdir, "it-1").ok  # already closed
    assert not dismiss_item(fdir, "it-99").ok  # never existed
    assert not snooze_item(fdir, "it-1", until="t").ok


def test_mark_notified(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="hi")
    marked = mark_notified(fdir, "it-1", at="2026-11-25T09:00:00Z")
    assert marked.ok
    assert marked.value.notified_at == "2026-11-25T09:00:00Z"
    assert load_items(fdir)[0].notified_at == "2026-11-25T09:00:00Z"


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    assert load_items(_foreman(tmp_path)) == ()


def test_corrupt_file_reads_empty(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    (fdir / "items.json").write_text("[{corrupt", encoding="utf-8")
    assert load_items(fdir) == ()


def test_corrupt_items_file_survives_create_item(tmp_path: Path) -> None:
    """A corrupt ``items.json`` is an error value — never silently overwritten."""
    fdir = _foreman(tmp_path)
    path = fdir / "items.json"
    path.write_text("[{corrupt", encoding="utf-8")
    created = create_item(fdir, kind="report", text="x")
    assert not created.ok
    assert "items.json" in created.error
    # Mutators must not clobber what they could not read.
    assert path.read_text(encoding="utf-8") == "[{corrupt"
    assert not answer_item(fdir, "it-1", choice=0).ok
    assert not dismiss_item(fdir, "it-1").ok
    assert path.read_text(encoding="utf-8") == "[{corrupt"


def test_undecodable_record_blocks_mutators(tmp_path: Path) -> None:
    """A record that fails to decode makes the strict load an error — mutators
    must not rewrite ``items.json`` without it. Readers still tolerate it and
    surface the decodable records."""
    fdir = _foreman(tmp_path)
    path = fdir / "items.json"
    payload = json.dumps(
        {
            "version": 1,
            "items": [
                {"id": "it-1", "kind": "report", "status": "open", "text": "x"},
                {"id": "it-2", "kind": "wish", "status": "open", "text": "y"},
            ],
        }
    )
    path.write_text(payload, encoding="utf-8")
    assert [item.id for item in load_items(fdir)] == ["it-1"]
    created = create_item(fdir, kind="report", text="z")
    assert not created.ok
    assert "items.json" in created.error
    dismissed = dismiss_item(fdir, "it-1")
    assert not dismissed.ok
    assert "items.json" in dismissed.error
    assert path.read_text(encoding="utf-8") == payload


def test_items_lock_busy_is_an_error_value(tmp_path: Path, monkeypatch) -> None:
    """A held ``items.json`` lock makes mutators a bounded error — never a
    hang. The lock covers the strict load AND the write, so ``create_item``
    (validation inside the locked step) and ``answer_item``/``mark_notified``
    (a single locked ``_update`` load) all see it."""
    import types

    from charlie_work import file_lock
    from charlie_work.file_lock import try_acquire_byte_range_lock

    monkeypatch.setattr(file_lock, "time", types.SimpleNamespace(sleep=lambda _s: None))
    fdir = _foreman(tmp_path)
    assert create_item(fdir, kind="decision", question="q", options=["a", "b"]).ok
    held = try_acquire_byte_range_lock(fdir / "items.json.lock")
    assert held is not None
    try:
        created = create_item(fdir, kind="report", text="y")
        assert not created.ok
        assert "lock busy" in created.error
        answered = answer_item(fdir, "it-1", choice=0)
        assert not answered.ok
        assert "lock busy" in answered.error
        notified = mark_notified(fdir, "it-1")
        assert not notified.ok
        assert "lock busy" in notified.error
    finally:
        held.release()
    assert create_item(fdir, kind="report", text="y").ok


def test_threaded_create_item_ids_are_unique(tmp_path: Path, monkeypatch) -> None:
    """The strict load runs under the lock: concurrent creators cannot mint
    the same id off a shared snapshot. The barrier widens the unprotected
    window to certainty — every thread is held at the load until all arrive,
    so without the lock they all read the same pre-write state."""
    import threading

    from charlie_work.foreman import items as items_mod

    fdir = _foreman(tmp_path)
    workers = 8
    barrier = threading.Barrier(workers, timeout=0.5)
    real_load = items_mod._load_items_strict

    def gated_load(foreman_root: Path):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            # The lock serializes callers, so under the fix each thread
            # reaches the barrier alone and it breaks — proceed anyway.
            pass
        return real_load(foreman_root)

    monkeypatch.setattr(items_mod, "_load_items_strict", gated_load)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(create_item(fdir, kind="report", text="x")))
        for _ in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(r.ok for r in results)
    assert sorted(r.value.id for r in results) == [f"it-{i}" for i in range(1, workers + 1)]
    assert len(load_items(fdir)) == workers


def test_answer_fields_round_trip_from_disk(tmp_path: Path) -> None:
    """``command``, ``created_at``, ``answer_note`` and ``answered_at`` reload off disk."""
    fdir = _foreman(tmp_path)
    create_item(
        fdir,
        kind="permission",
        question="run the command?",
        options=["approve", "deny"],
        command="uv run pytest -q",
        created_at="2026-11-25T09:00:00Z",
    )
    answered = answer_item(fdir, "it-1", choice=0, note="lgtm", answered_at="2026-11-25T10:00:00Z")
    assert answered.ok
    reloaded = load_items(fdir)[0]
    assert reloaded.command == "uv run pytest -q"
    assert reloaded.created_at == "2026-11-25T09:00:00Z"
    assert reloaded.answer == 0
    assert reloaded.answer_note == "lgtm"
    assert reloaded.answered_at == "2026-11-25T10:00:00Z"


def test_snooze_rejects_unparseable_and_naive(tmp_path: Path) -> None:
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="x")
    assert not snooze_item(fdir, "it-1", until="tomorrow").ok
    # A naive timestamp (no offset) is ambiguous — rejected too.
    assert not snooze_item(fdir, "it-1", until="2026-11-26T09:00:00").ok


def test_snooze_offset_form_normalizes_and_compares(tmp_path: Path) -> None:
    """An offset-form ``until`` is stored UTC-normalized and compares as an instant."""
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="x")
    snoozed = snooze_item(fdir, "it-1", until="2026-11-26T11:00:00+02:00")
    assert snoozed.ok, snoozed.error
    assert snoozed.value.snoozed_until == "2026-11-26T09:00:00Z"
    items = load_items(fdir)
    assert open_items(items, now="2026-11-26T08:59:00Z") == ()
    assert len(open_items(items, now="2026-11-26T09:01:00Z")) == 1
    # Same instant in another offset form — datetime compare, not string order.
    assert len(open_items(items, now="2026-11-26T10:01:00+01:00")) == 1


def test_unparseable_stored_snooze_resurfaces(tmp_path: Path) -> None:
    """A stored ``snoozed_until`` that fails to parse fails visible — the item resurfaces."""
    fdir = _foreman(tmp_path)
    create_item(fdir, kind="report", text="x")
    path = fdir / "items.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "items": [
                    {
                        "id": "it-1",
                        "kind": "report",
                        "status": "snoozed",
                        "text": "x",
                        "snoozed_until": "garbage",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    items = load_items(fdir)
    assert len(open_items(items, now="2026-11-26T09:00:00Z")) == 1
