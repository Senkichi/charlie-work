"""Foreman items on the Now dashboard (issue #3121).

``now_read_foremen`` decodes open ``decision``/``permission`` items straight
from ``foremen/*/items.json`` — read-only, no ``charlie_work.foreman`` import
(the import-direction guard covers the seam structurally). ``now_needs_me``
folds them into the "Foremen" group and the page draws the fourth tile.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from _dashboard_page_fixtures import _item, _model, _page

from charlie_work.dashboard.now_needs_me import needs_me_items
from charlie_work.dashboard.now_read_foremen import read_foreman_items
from charlie_work.dashboard.now_types import ForemanItemRead, SourcesRead

NOW_DT = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)


def _record(item_id: str, kind: str = "decision", status: str = "open", **kw) -> dict:
    record = {
        "id": item_id,
        "kind": kind,
        "status": status,
        "question": f"Q {item_id}?",
        "options": ["a", "b"],
        "recommended": 0,
        "created_at": "2026-11-20T00:00:00Z",
    }
    record.update(kw)
    return record


def _items_file(fleet: Path, foreman: str, records: list[dict]) -> Path:
    directory = fleet / "foremen" / foreman
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "items.json"
    path.write_text(json.dumps({"version": 1, "items": records}), encoding="utf-8")
    return path


def _tree(path: Path) -> dict[str, str]:
    """A fingerprint of everything under ``path`` (names + file bytes)."""
    if not path.exists():
        return {}
    return {
        str(p.relative_to(path)): (p.read_bytes().hex() if p.is_file() else "<dir>")
        for p in path.rglob("*")
    }


def test_read_foreman_items_lists_open_decisions_and_permissions(tmp_path: Path) -> None:
    _items_file(
        tmp_path,
        "g1",
        [
            _record("it-1"),
            _record("it-2", kind="permission"),
            _record("it-3", status="answered"),
            _record("it-4", status="dismissed"),
            _record("it-5", kind="report", text="a report, not a question"),
            _record("it-6", status="snoozed", snoozed_until="2099-01-01T00:00:00Z"),
            _record("it-7", status="snoozed", snoozed_until="2000-01-01T00:00:00Z"),
        ],
    )
    _items_file(tmp_path, "g2", [_record("it-1", kind="permission")])
    reads = read_foreman_items(tmp_path, NOW_DT)
    by_id = {(r.foreman, r.item_id): r for r in reads}
    assert set(by_id) == {("g1", "it-1"), ("g1", "it-2"), ("g1", "it-7"), ("g2", "it-1")}
    first = by_id[("g1", "it-1")]
    assert first.kind == "decision"
    assert first.text == "Q it-1?"
    assert first.recommended == 0
    assert first.created_at == datetime(2026, 11, 20, tzinfo=UTC)


def test_read_foreman_items_validates_recommended_and_commands(tmp_path: Path) -> None:
    _items_file(
        tmp_path,
        "g1",
        [
            _record("it-1", recommended=9, command="charlie requeue"),  # out of range
            _record("it-2", recommended="x"),  # not an int
        ],
    )
    reads = {r.item_id: r for r in read_foreman_items(tmp_path, NOW_DT)}
    assert reads["it-1"].recommended is None
    assert reads["it-1"].command == "charlie requeue"
    assert reads["it-2"].recommended is None


def test_read_foreman_items_tolerates_missing_and_corrupt(tmp_path: Path) -> None:
    assert read_foreman_items(tmp_path, NOW_DT) == ()
    _items_file(tmp_path, "g1", [_record("it-1")])
    bad = tmp_path / "foremen" / "g2"
    bad.mkdir()
    (bad / "items.json").write_text("not json", encoding="utf-8")
    (tmp_path / "foremen" / "_shared").mkdir()  # never a foreman
    stray = tmp_path / "foremen" / "README.md"
    stray.write_text("hi", encoding="utf-8")
    assert [r.foreman for r in read_foreman_items(tmp_path, NOW_DT)] == ["g1"]
    # A malformed record cannot hide the decodable one beside it.
    _items_file(tmp_path, "g3", [_record("it-9"), "garbage", {"no": "id"}])
    assert [r.item_id for r in read_foreman_items(tmp_path, NOW_DT) if r.foreman == "g3"] == [
        "it-9"
    ]


def test_read_foreman_items_is_read_only(tmp_path: Path) -> None:
    """The read must never write: the directory tree is byte-identical after."""
    _items_file(tmp_path, "g1", [_record("it-1")])
    before = _tree(tmp_path)
    assert read_foreman_items(tmp_path, NOW_DT)
    assert _tree(tmp_path) == before


def test_needs_me_folds_foreman_items_into_their_group() -> None:
    created = datetime(2026, 11, 24, tzinfo=UTC)
    with_command = ForemanItemRead(
        "g1", "it-1", "decision", "Retry?", "charlie requeue", 0, created
    )
    without_command = ForemanItemRead(
        "g2", "it-3", "permission", "Arm the fleet?", None, 1, created
    )
    items = needs_me_items(
        SourcesRead(repos=(), foreman_items=(with_command, without_command)),
        (),
        (),
        NOW_DT,
        600.0,
        None,
    )
    assert [i.kind for i in items] == ["foreman_item", "foreman_item"]
    assert {i.group for i in items} == {"Foremen"}
    by_repo = {i.repo: i for i in items}
    # The item's own command is the suggested remedy.
    assert by_repo["g1"].command == "charlie requeue"
    # Without one the remedy is to answer the item, prefilled with the
    # recommended option index.
    assert by_repo["g2"].command == "charlie gang intervene g2 it-3 --choose 1", by_repo[
        "g2"
    ].command
    assert by_repo["g1"].reason == "Decision item it-1: Retry?"
    assert by_repo["g2"].age_seconds == (NOW_DT - created).total_seconds()


def test_page_renders_the_foremen_tile_and_row() -> None:
    items = _model().needs_me + (
        _item(
            "Foremen",
            kind="foreman_item",
            repo="g1",
            reason="Decision item it-1: Retry the failed issue?",
            command="charlie gang intervene g1 it-1 --choose 0",
            number=None,
            as_of_snapshot=False,
        ),
    )
    page = _page(_model(items=items))
    assert 'id="tile-foremen" data-g="foremen"' in page
    assert '<section class="grp" data-g="foremen" aria-label="Foremen"' in page
    assert re.search(r'class="why"[^>]*>[^<]*Decision item it-1: Retry', page), (
        "the foreman row's question must render"
    )
