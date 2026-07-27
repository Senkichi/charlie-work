from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from charlie_work.fleet_paths import fleet_dir
from charlie_work.logging_setup import (
    LOG_RETENTION_DAYS,
    _log_file_path,
    _prune_old_logs,
    configure_logging,
)


class _CollectingHandler(logging.Handler):
    """A handler attached directly to a named (non-root) logger.

    pytest's own log capturing attaches a LogCaptureHandler to the ROOT
    logger for the duration of each test's "call" phase -- which is exactly
    what configure_logging()'s idempotency guard (`if logging.root.handlers:
    return`) trips on. Exercising configure_logging's real logic therefore
    requires clearing logging.root.handlers, which also detaches pytest's own
    handler and makes caplog blind for the rest of the test. Attaching this
    collector to the specific logger a record is emitted through sidesteps
    that: handler dispatch happens at every level a record propagates
    through, not only at whichever handlers are on the root logger, so this
    still sees the record regardless of what root's handler list holds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def _clean_root_logger() -> Iterator[None]:
    """Restore the root logger's handlers/level after a test clears them.

    configure_logging() is deliberately idempotent once handlers exist -- the
    same rule logging.basicConfig follows, necessary so repeated in-process
    CLI invocations (cli.main() is called 13+ times across test_cli.py in one
    pytest process) don't open and abandon a new file handle every call.
    Exercising the "first call" path therefore requires each test to clear
    logging.root.handlers itself, as the FIRST statement in the test body --
    doing it in a fixture's setup code runs too early (before pytest's own
    per-test log-capture handler is attached for the "call" phase) and gets
    silently overwritten before the test body executes. Any FileHandler
    opened during the test must be closed before this restores the original
    list, or the open handle blocks Windows from deleting the underlying
    tmp_path during fixture teardown.
    """
    saved_handlers = list(logging.root.handlers)
    saved_level = logging.root.level
    try:
        yield
    finally:
        for handler in logging.root.handlers:
            if handler not in saved_handlers:
                handler.close()
        logging.root.handlers = saved_handlers
        logging.root.level = saved_level


def test_log_file_path_embeds_utc_date_under_fleet_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))

    path = _log_file_path(datetime(2026, 7, 26, 3, 0, tzinfo=UTC))

    assert path == tmp_path / "fleet" / "logs" / "charlie-work-20260726.log"


def test_configure_logging_installs_stream_and_file_handlers_and_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _clean_root_logger: None,
) -> None:
    """The primary #646 follow-up guarantee: a census-level log record must
    land on disk under fleet_dir(), independent of stderr, independent of
    CWD, and independent of whether any launcher redirected a stream."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    logging.root.handlers = []  # see _clean_root_logger docstring

    configure_logging(verbose=False)

    file_handlers = [h for h in logging.root.handlers if isinstance(h, logging.FileHandler)]
    stream_handlers = [
        h
        for h in logging.root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    assert len(file_handlers) == 1, "expected exactly one FileHandler to be installed"
    assert stream_handlers, "stderr handler must still be present alongside the file handler"

    marker = "issue-646 census probe: n_alive=1 worktree=/tmp/x pid=1234 cap=2"
    logging.getLogger("charlie_work.workflow").info(marker)
    for handler in logging.root.handlers:
        handler.flush()

    expected_path = fleet_dir() / "logs" / f"charlie-work-{datetime.now(UTC):%Y%m%d}.log"
    assert expected_path == Path(file_handlers[0].baseFilename)
    content = expected_path.read_text(encoding="utf-8")
    assert marker in content


def test_configure_logging_is_idempotent_when_handlers_already_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _clean_root_logger: None,
) -> None:
    """Regression: a second in-process call (every CLI test invokes main())
    must not open a second file handle -- basicConfig's own no-op rule is
    what configure_logging mirrors, and this pins that it actually does."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    logging.root.handlers = []  # see _clean_root_logger docstring
    sentinel = logging.NullHandler()
    logging.root.addHandler(sentinel)

    configure_logging(verbose=False)

    assert logging.root.handlers == [sentinel]
    assert not (tmp_path / "fleet" / "logs").exists()


def test_configure_logging_falls_back_to_stream_only_on_file_open_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    _clean_root_logger: None,
) -> None:
    """A log directory that can never be created (here: an ancestor path
    component is a plain file, so mkdir(parents=True) raises
    NotADirectoryError) must not crash logging setup -- same principle as
    the try/except wrapped around _log_worker_census in workflow.py.

    Uses _CollectingHandler rather than caplog: caplog relies on a handler
    attached to the root logger, which this test must clear to get past
    configure_logging's idempotency guard in the first place.
    """
    blocker_file = tmp_path / "blocker"
    blocker_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(blocker_file / "fleet"))
    logging.root.handlers = []  # see _clean_root_logger docstring

    collector = _CollectingHandler()
    setup_logger = logging.getLogger("charlie_work.logging_setup")
    setup_logger.addHandler(collector)
    try:
        configure_logging(verbose=False)  # must not raise
    finally:
        setup_logger.removeHandler(collector)

    assert any(isinstance(h, logging.StreamHandler) for h in logging.root.handlers)
    assert not any(isinstance(h, logging.FileHandler) for h in logging.root.handlers)
    assert any(
        "could not open charlie-work log file" in record.getMessage()
        for record in collector.records
    )


def test_prune_old_logs_deletes_stale_files_but_keeps_recent(tmp_path: Path) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    stale = logs_dir / "charlie-work-20200101.log"
    recent = logs_dir / "charlie-work-20260726.log"
    other = logs_dir / "not-a-charlie-log.txt"
    for path in (stale, recent, other):
        path.write_text("x", encoding="utf-8")

    now = datetime.now(UTC).timestamp()
    stale_time = now - (LOG_RETENTION_DAYS + 1) * 86400
    recent_time = now - 86400  # yesterday, well inside the retention window
    os.utime(stale, (stale_time, stale_time))
    os.utime(recent, (recent_time, recent_time))
    os.utime(other, (stale_time, stale_time))

    _prune_old_logs(logs_dir)

    assert not stale.exists()
    assert recent.exists()
    assert other.exists(), "pruning must only touch charlie-work-*.log files"


def test_prune_old_logs_never_raises_on_missing_dir(tmp_path: Path) -> None:
    _prune_old_logs(tmp_path / "does-not-exist")  # must return quietly, not raise
