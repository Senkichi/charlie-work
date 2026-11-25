"""``items.json`` typed mutations: create/answer/snooze/dismiss (issue #3117)."""

from __future__ import annotations

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
