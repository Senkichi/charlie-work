"""``charlie dashboard serve``: kill switch, HEAD-drift restart, bind failure, CLI parse."""

from __future__ import annotations

import json
import socket
import threading

import pytest

from charlie_work import cli
from charlie_work.dashboard import serve
from charlie_work.dashboard.config import DashboardConfig
from charlie_work.dashboard.serve import serve_dashboard, watch_head_drift
from charlie_work.supervise_loop import EXIT_RESTART_REQUESTED


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_kill_switch_does_not_bind_and_exits_zero(tmp_path, capsys) -> None:
    (tmp_path / "config.yaml").write_text("dashboard:\n  enabled: false\n", encoding="utf-8")
    rc = cli.main(["--json", "--fleet-dir", str(tmp_path), "dashboard", "serve"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["ok"] is True
    assert out["message"] == serve.DISABLED_MESSAGE


def test_head_drift_returns_restart_request(tmp_path) -> None:
    heads = iter(["aaa", "aaa"])
    result = serve_dashboard(
        DashboardConfig(port=_free_port()),
        str(tmp_path),
        read_head=lambda: next(heads, "bbb"),
        drift_interval_seconds=0.05,
    )
    assert result.ok and result.data == {"restart_requested": True}
    assert "HEAD changed" in result.message


def test_drift_through_cli_exits_with_imported_constant(tmp_path, monkeypatch) -> None:
    (tmp_path / "config.yaml").write_text(
        f"dashboard:\n  port: {_free_port()}\n", encoding="utf-8"
    )
    heads = iter(["aaa"])
    monkeypatch.setattr(serve, "_default_head_reader", lambda: next(heads, "bbb"))
    monkeypatch.setattr(serve, "DEFAULT_DRIFT_INTERVAL_SECONDS", 0.05)
    rc = cli.main(["--fleet-dir", str(tmp_path), "dashboard", "serve"])
    assert rc == EXIT_RESTART_REQUESTED


def test_none_reads_never_fake_a_drift() -> None:
    stop, drifted = threading.Event(), threading.Event()
    reads = iter([None, "aaa", None, "aaa"])

    def read() -> str | None:
        try:
            return next(reads)
        except StopIteration:
            stop.set()
            return "aaa"

    watch_head_drift(read, stop, drifted, 0.01)
    assert not drifted.is_set()


def test_port_in_use_is_an_error_value(tmp_path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = serve_dashboard(DashboardConfig(port=port), str(tmp_path), read_head=lambda: "a")
    assert result.ok is False
    assert f"cannot bind 127.0.0.1:{port}" in result.message


def test_serve_help_parses(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["dashboard", "serve", "--help"])
    assert exc.value.code == 0
    assert "serve" in capsys.readouterr().out
