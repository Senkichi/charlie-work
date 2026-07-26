"""Tests for the marker-wait helper that removes the create-then-write race.

The helper exists because adapter launch tests polled for ``path.exists()`` and
then read, which can observe a zero-byte or half-written file. These tests
reproduce that window deterministically — an empty file that gains content later
— so a regression back to existence-polling fails here instead of failing
randomly on a loaded CI runner.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from _worker_marker_wait import read_worker_marker


def _write_after(path: Path, text: str, delay: float) -> threading.Thread:
    """Create ``path`` empty now, fill it after ``delay`` — the exact race window."""
    path.write_text("", encoding="utf-8")

    def _later() -> None:
        time.sleep(delay)
        path.write_text(text, encoding="utf-8")

    thread = threading.Thread(target=_later, daemon=True)
    thread.start()
    return thread


def test_waits_through_the_empty_file_window(tmp_path: Path) -> None:
    """A file that exists but is empty must not end the wait.

    This is the failure that took down test_launch_claude_worker_prompt_path_
    placeholder_skips_stdin: `assert '' == 'prompt payload for argv'`.
    """
    path = tmp_path / "marker.txt"
    thread = _write_after(path, "payload", delay=0.3)

    assert path.exists(), "precondition: the path is already there, so exists() is useless here"
    assert read_worker_marker(path, timeout=5) == "payload"
    thread.join(timeout=5)


def test_waits_for_the_exact_expected_value_not_a_prefix(tmp_path: Path) -> None:
    """A partial write must keep the wait going, not satisfy it.

    Waiting for merely non-empty content is not enough for a multi-field probe:
    this is the shape that produced `ValueError: not enough values to unpack
    (expected 2, got 1)` when a `a|b` probe was read as `a`.
    """
    path = tmp_path / "probe.txt"
    path.write_text("2", encoding="utf-8")  # non-empty but incomplete

    def _complete() -> None:
        time.sleep(0.3)
        path.write_text("2|kimi-k3", encoding="utf-8")

    thread = threading.Thread(target=_complete, daemon=True)
    thread.start()

    xdist, model = read_worker_marker(path, expected="2|kimi-k3", timeout=5).split("|")
    assert (xdist, model) == ("2", "kimi-k3")
    thread.join(timeout=5)


def test_missing_file_raises_with_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "never.txt"
    with pytest.raises(AssertionError, match="never created"):
        read_worker_marker(path, timeout=0.2)


def test_empty_file_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    """Timing out must fail loudly, never hand back the empty read."""
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    with pytest.raises(AssertionError, match="empty"):
        read_worker_marker(path, timeout=0.2)


def test_wrong_value_reports_expected_and_observed(tmp_path: Path) -> None:
    path = tmp_path / "wrong.txt"
    path.write_text("actual-value", encoding="utf-8")
    with pytest.raises(AssertionError) as exc:
        read_worker_marker(path, expected="wanted-value", timeout=0.2)
    message = str(exc.value)
    assert "wanted-value" in message
    assert "actual-value" in message


def test_reason_is_included_in_the_failure(tmp_path: Path) -> None:
    """`reason` replaces the tautological `assert text == expected` that would
    otherwise follow a call with `expected=`, so it must survive into the error."""
    path = tmp_path / "reasoned.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(AssertionError, match="VIRTUAL_ENV must be stripped"):
        read_worker_marker(
            path,
            expected="yes",
            reason="VIRTUAL_ENV must be stripped",
            timeout=0.2,
        )


def test_returns_immediately_when_content_is_already_there(tmp_path: Path) -> None:
    """The helper must not add latency to the common case."""
    path = tmp_path / "ready.txt"
    path.write_text("done", encoding="utf-8")

    started = time.monotonic()
    assert read_worker_marker(path, timeout=5) == "done"
    assert time.monotonic() - started < 1.0
