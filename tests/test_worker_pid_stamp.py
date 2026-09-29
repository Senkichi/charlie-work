"""Unit tests for ``worker_pid_stamp.stamp_worker_process``."""

from __future__ import annotations

from types import SimpleNamespace

from charlie_work.worker_pid_stamp import stamp_worker_process


def test_stamps_pid_and_start_time() -> None:
    entry = {"worker_pid": 1, "worker_process_start_time": 1.0, "status": "dispatched"}
    stamp_worker_process(entry, SimpleNamespace(pid=99, process_start_time=5.5))
    assert entry == {"worker_pid": 99, "worker_process_start_time": 5.5, "status": "dispatched"}


def test_missing_pid_clears_previous_epoch() -> None:
    entry = {"worker_pid": 1, "worker_process_start_time": 1.0, "status": "dispatched"}
    stamp_worker_process(entry, SimpleNamespace(pid=None, process_start_time=None))
    assert entry == {"status": "dispatched"}


def test_missing_result_clears_previous_epoch() -> None:
    entry = {"worker_pid": 1, "worker_process_start_time": 1.0}
    stamp_worker_process(entry, None)
    assert entry == {}
