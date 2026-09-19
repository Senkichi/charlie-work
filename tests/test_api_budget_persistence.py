"""Ledger persistence: save/load roundtrip, quarantine, schema (issue #480).

Also carries the frozen-dataclass value-type invariant that spans every
persisted type.

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from charlie_work import api_budget
from charlie_work.api_budget import (
    DayBucket,
    Ledger,
    Usage,
    ledger_path,
    load_ledger,
    save_ledger,
    settle_session,
)

from _api_budget_unit_fixtures import _entry


def test_save_then_load_roundtrip(tmp_path: Path) -> None:
    entry = _entry()
    ledger = settle_session(Ledger(), entry)
    path = ledger_path(tmp_path)
    save_ledger(path, ledger)
    loaded = load_ledger(path)
    assert loaded.lifetime_usd == pytest.approx(ledger.lifetime_usd)
    assert dict(loaded.days) == dict(ledger.days)
    assert len(loaded.sessions) == 1
    assert loaded.sessions[0] == entry


def test_load_ledger_missing_file_returns_empty(tmp_path: Path) -> None:
    ledger = load_ledger(ledger_path(tmp_path))
    assert ledger == Ledger()


def test_load_ledger_corrupt_file_preserves_original_and_recovers_empty(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("ERROR", logger="charlie_work.api_budget"):
        ledger = load_ledger(path)
    assert ledger == Ledger()
    # The corrupt original must be preserved on disk (moved aside, not destroyed).
    remaining = list(tmp_path.glob("api-budget.json.corrupt-*"))
    assert len(remaining) == 1
    assert remaining[0].read_text(encoding="utf-8") == "{not valid json"
    # The original path no longer holds the corrupt content.
    assert not path.exists()
    assert any("unrecoverable" in rec.message for rec in caplog.records)


def test_load_ledger_partial_json_preserves_original(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"days": {"2026-07-22": {', encoding="utf-8")  # truncated
    ledger = load_ledger(path)
    assert ledger == Ledger()
    assert list(tmp_path.glob("api-budget.json.corrupt-*"))


def test_load_ledger_wrong_typed_lifetime_usd_quarantines(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Valid JSON with a wrong-typed (non-numeric) ``lifetime_usd`` is structural
    corruption: ``_ledger_from_dict``'s ``float()`` coercion raises, and the
    file must go through the quarantine path (forensic log + preserved original)
    rather than propagating uncaught and wedging every future settlement.

    Regression guard for the review-#540 finding that ``load_ledger``'s guard
    only wrapped ``json.load`` and left ``_ledger_from_dict`` outside it.
    """
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original = '{"lifetime_usd": "not-a-number", "days": {}, "sessions": []}'
    path.write_text(original, encoding="utf-8")
    with caplog.at_level("ERROR", logger="charlie_work.api_budget"):
        ledger = load_ledger(path)
    assert ledger == Ledger()
    # Quarantined — original preserved on disk for forensics.
    remaining = list(tmp_path.glob("api-budget.json.corrupt-*"))
    assert len(remaining) == 1
    assert remaining[0].read_text(encoding="utf-8") == original
    assert not path.exists()
    assert any("unrecoverable" in rec.message for rec in caplog.records)


def test_load_ledger_wrong_typed_day_bucket_usd_quarantines(tmp_path: Path) -> None:
    """A wrong-typed ``usd`` inside a day bucket is the same structural
    corruption class and must also quarantine (not wedge)."""
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"lifetime_usd": 1.0, "days": {"2026-07-22": {"usd": ["bad"]}}, "sessions": []}',
        encoding="utf-8",
    )
    ledger = load_ledger(path)
    assert ledger == Ledger()
    assert list(tmp_path.glob("api-budget.json.corrupt-*"))


def test_load_ledger_wrong_typed_session_usd_drops_entry_leniently(tmp_path: Path) -> None:
    """A wrong-typed ``usd`` on a session entry is caught by the per-entry
    try/except in ``_ledger_from_dict`` (which guards this exact failure class
    by dropping the bad entry, per the review-#540 note) — so it does NOT
    quarantine; the entry is dropped and the rest of the ledger loads.

    This documents the intentional split: per-entry malformation = drop the
    entry; structural (top-level / day-bucket) malformation = quarantine.
    """
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"lifetime_usd": 0.0, "days": {}, "sessions": ['
        '{"issue": 1, "session_id": "s", "usd": {"x": 1}}]}',
        encoding="utf-8",
    )
    ledger = load_ledger(path)
    # Bad entry dropped leniently; no quarantine (the file is otherwise valid).
    assert ledger == Ledger()
    assert not list(tmp_path.glob("api-budget.json.corrupt-*"))


def test_load_ledger_lenient_on_missing_keys(tmp_path: Path) -> None:
    path = ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"version": 1}', encoding="utf-8")  # valid JSON, no ledger keys
    ledger = load_ledger(path)
    assert ledger == Ledger()


def test_save_ledger_uses_atomic_temp_replace(tmp_path: Path) -> None:
    """Atomicity: the write goes through temp + replace. After save, no .tmp
    leftover remains and the file is valid JSON (a concurrent reader never
    observes a half-written file)."""
    path = ledger_path(tmp_path)
    save_ledger(path, settle_session(Ledger(), _entry()))
    # No leftover temp file.
    assert not (tmp_path / "api-budget.json.tmp").exists()
    # The file is valid, parseable JSON.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "days" in data
    assert "lifetime_usd" in data
    assert "sessions" in data


def test_save_ledger_no_plain_open_write_on_ledger_path(tmp_path: Path) -> None:
    """Invariant: the ledger path is never written with a plain open(path, 'w').

    Inspect the module source to assert the atomic temp+replace pattern is used
    and no bare ``open(<ledger path>, "w")`` exists for the ledger file itself.
    """
    src = Path(api_budget.__file__).read_text(encoding="utf-8")
    # save_ledger opens the TMP path for writing, then replaces — never the
    # ledger path directly. Assert the canonical pattern is present.
    assert 'tmp_path = path.with_suffix(path.suffix + ".tmp")' in src
    assert "tmp_path.replace(path)" in src
    # No plain open(path, "w") on the ledger path itself.
    assert 'open(path, "w")' not in src
    assert "open(path, 'w')" not in src


def test_ledger_to_dict_and_schema(tmp_path: Path) -> None:
    entry = _entry()
    ledger = settle_session(Ledger(), entry)
    data = api_budget.ledger_to_dict(ledger)
    assert isinstance(data["days"], dict)
    day = data["days"]["2026-07-22"]
    assert set(day.keys()) == {"input_tokens", "output_tokens", "cached_tokens", "usd"}
    assert isinstance(data["sessions"], list)
    sess = data["sessions"][0]
    assert set(sess.keys()) == {
        "issue",
        "session_id",
        "provider",
        "model",
        "started_at",
        "ended_at",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "usd",
        "duration_s",
        "outcome",
    }


def test_value_types_are_frozen() -> None:
    for obj in (Usage(), DayBucket(), _entry(), Ledger()):
        # Frozen dataclasses raise FrozenInstanceError (an AttributeError
        # subclass) on attribute assignment.
        with pytest.raises(AttributeError):
            obj.input_tokens = 999  # type: ignore[misc]
            obj.usd = 999.0  # type: ignore[misc]
