"""Tests for issue #1460: read_advisories / advisory_log_exists.

Positive control: write a record via the same JSONL shape ``hook_entry``
writes, then read it back through ``read_advisories`` -- this is the load-
bearing control for the "no rows" case elsewhere (an empty read alone proves
nothing without a positive-control read that succeeds).
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.attachment_contracts.hook_entry import (
    _ADVISORY_LOG_REL,
    advisory_log_exists,
    read_advisories,
)
from charlie_work.attachment_contracts.model import AdvisoryRecord


def _write_log(root: Path, lines: list[str]) -> None:
    log_path = root / _ADVISORY_LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_advisory_log_exists_false_when_missing(tmp_path: Path) -> None:
    assert advisory_log_exists(tmp_path) is False
    assert read_advisories(tmp_path) == ()


def test_positive_control_write_then_read_back(tmp_path: Path) -> None:
    """A record written in the current (post-#1460) shape round-trips."""
    record = {
        "severity": "block",
        "file": "src/foo.py",
        "identity": "Foo",
        "message": "Foo is saturated",
        "redirect": "src/foo_extra.py",
        "timestamp": "2026-08-25T00:00:00+00:00",
    }
    _write_log(tmp_path, [json.dumps(record)])

    assert advisory_log_exists(tmp_path) is True
    records = read_advisories(tmp_path)
    assert records == (
        AdvisoryRecord(
            severity="block",
            file="src/foo.py",
            identity="Foo",
            message="Foo is saturated",
            redirect="src/foo_extra.py",
            timestamp="2026-08-25T00:00:00+00:00",
        ),
    )


def test_old_shape_record_without_redirect_or_timestamp_parses(tmp_path: Path) -> None:
    """A record written before #1460 (no redirect/timestamp keys) must still
    parse -- read_advisories tolerates the old shape."""
    old_record = {
        "severity": "error",
        "file": "src/bar.py",
        "identity": "Bar",
        "message": "Bar tamper detected",
    }
    _write_log(tmp_path, [json.dumps(old_record)])

    records = read_advisories(tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec.severity == "error"
    assert rec.file == "src/bar.py"
    assert rec.identity == "Bar"
    assert rec.message == "Bar tamper detected"
    assert rec.redirect is None
    assert rec.timestamp is None


def test_malformed_lines_are_skipped_best_effort(tmp_path: Path) -> None:
    good = {
        "severity": "block",
        "file": "src/ok.py",
        "identity": "Ok",
        "message": "fine",
    }
    _write_log(
        tmp_path,
        [
            "not json at all {{{",
            json.dumps(["not", "an", "object"]),
            json.dumps({"severity": "block"}),  # missing required fields
            json.dumps(good),
            "",
        ],
    )

    records = read_advisories(tmp_path)
    assert len(records) == 1
    assert records[0].identity == "Ok"


def test_multiple_records_preserve_order(tmp_path: Path) -> None:
    records_in = [
        {"severity": "block", "file": f"src/f{i}.py", "identity": f"I{i}", "message": "m"}
        for i in range(3)
    ]
    _write_log(tmp_path, [json.dumps(r) for r in records_in])

    records_out = read_advisories(tmp_path)
    assert [r.identity for r in records_out] == ["I0", "I1", "I2"]
