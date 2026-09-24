"""Notify-digest freshness consumer tests for ``scripts/heartbeat_check.py``.

Issue #1859: the notify digest shipped as a signal without a consumer -- the
live daemon's file-sink writer was dead for three weeks while nothing read
the file. ``check_notify_digest_freshness`` is that consumer: it merges the
``notify:`` section across the global fleet layer and the checkout's own
``orchestrator.config.yaml`` exactly the way ``load_layered_config`` does,
then flags an enabled file sink whose digest is missing or stale.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import _iso, _load_heartbeat_check


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


@pytest.fixture()
def fleet_dir(tmp_path: Path, monkeypatch: Any) -> Path:
    """Point the script's fleet_dir() at a hermetic directory."""
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
    return tmp_path / "fleet"


def _write_yaml(path: Path, notify: dict[str, Any] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if notify is None:
        path.write_text("runtime:\n  state_dir: .var/x\n", encoding="utf-8")
        return
    lines = ["notify:"]
    for key, value in notify.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = f"'{value}'"
        else:
            rendered = str(value)
        lines.append(f"  {key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_digest(path: Path, generated_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"generated_at": generated_at, "repo": "fleet", "transitions": []}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")


def test_notify_digest_warn_when_section_absent(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """No notify: section in either layer -> WARN (not anomaly): a fleet that
    never opted in must not flip the exit code, but the line is visible."""
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")


def test_notify_digest_warn_when_explicitly_disabled(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    _write_yaml(fleet_dir / "config.yaml", {"enabled": False})
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")


def test_notify_digest_anomaly_when_enabled_but_file_missing(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """The 2026-08-31 shape: enabled file sink, no digest on disk."""
    digest = tmp_path / "state" / "notify" / "digest.jsonl"
    _write_yaml(
        fleet_dir / "config.yaml",
        {"enabled": True, "sink": "file", "file_path": str(digest)},
    )
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert report.anomaly
    assert "does not exist" in report.lines[0]


def test_notify_digest_anomaly_when_last_entry_stale(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    digest = tmp_path / "digest.jsonl"
    _write_yaml(
        fleet_dir / "config.yaml",
        {"enabled": True, "sink": "file", "file_path": str(digest)},
    )
    old_ts = _iso(60 * (hb.NOTIFY_DIGEST_STALE_HOURS + 1))
    _write_digest(digest, old_ts)
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert report.anomaly
    assert "writer looks dead" in report.lines[0]


def test_notify_digest_ok_when_last_entry_fresh(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    digest = tmp_path / "digest.jsonl"
    _write_yaml(
        fleet_dir / "config.yaml",
        {"enabled": True, "sink": "file", "file_path": str(digest)},
    )
    _write_digest(digest, _iso(5))
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert report.lines[0].startswith("OK notify-digest")


def test_notify_digest_mtime_fallback_when_tail_unparseable(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """A digest file with no parseable generated_at still gets a verdict from
    its mtime -- a file merely touched is not mistaken for live output."""
    digest = tmp_path / "digest.jsonl"
    _write_yaml(
        fleet_dir / "config.yaml",
        {"enabled": True, "sink": "file", "file_path": str(digest)},
    )
    digest.write_text("{broken json\n", encoding="utf-8")
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly  # mtime is fresh (just written)
    assert "mtime" in report.lines[0]


def test_notify_digest_anomaly_when_file_path_unset(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """enabled + sink=file + no file_path is the incoherent combination every
    emit fails on; surface it as an anomaly."""
    _write_yaml(fleet_dir / "config.yaml", {"enabled": True, "sink": "file"})
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert report.anomaly
    assert "file_path is unset" in report.lines[0]


def test_notify_digest_ok_for_non_file_sink(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    _write_yaml(fleet_dir / "config.yaml", {"enabled": True, "sink": "webhook"})
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert "no digest file" in report.lines[0]


def test_notify_digest_checkout_layer_overrides_global(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """The per-repo file wins per-key over the global layer, matching
    load_layered_config: the global layer's file_path must not be probed when
    the checkout layer sets its own."""
    global_digest = tmp_path / "global.jsonl"
    checkout_digest = tmp_path / "checkout.jsonl"
    _write_yaml(
        fleet_dir / "config.yaml",
        {"enabled": True, "sink": "file", "file_path": str(global_digest)},
    )
    _write_yaml(
        tmp_path / "orchestrator.config.yaml",
        {"enabled": True, "file_path": str(checkout_digest)},
    )
    # Checkout-layer path is fresh; global-layer path is missing. If the
    # merge order were wrong this would anomaly on the missing global path.
    _write_digest(checkout_digest, _iso(2))
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert str(checkout_digest) in report.lines[0]


def test_notify_digest_relative_file_path_resolves_against_checkout(
    hb: ModuleType, fleet_dir: Path, tmp_path: Path
) -> None:
    """The production spelling is relative (``.var/charlie-work/notify/
    digest.jsonl``); _file_sink interprets it against the supervisor's cwd,
    which in production is the checkout this script runs from."""
    _write_yaml(
        fleet_dir / "config.yaml",
        {
            "enabled": True,
            "sink": "file",
            "file_path": ".var/charlie-work/notify/digest.jsonl",
        },
    )
    _write_digest(
        tmp_path / ".var" / "charlie-work" / "notify" / "digest.jsonl",
        _iso(3),
    )
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert report.lines[0].startswith("OK notify-digest")


def test_notify_digest_warns_when_package_unimportable(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Guarded-leaf contract: a broken package install degrades the check to
    a WARN line instead of crashing the script (scripts/README invariant)."""
    monkeypatch.setattr(hb, "_ndc", None)
    report = hb.Report()
    hb.check_notify_digest_freshness(report, checkout_root=tmp_path)
    assert not report.anomaly
    assert report.lines[0].startswith("WARN notify-digest")
    assert "not importable" in report.lines[0]
