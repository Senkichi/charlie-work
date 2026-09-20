"""Orchestrator-config tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``load_orchestrator_config`` / ``check_orchestrator_config``
(absent/valid/invalid-YAML/non-UTF8/non-mapping paths),
``get_mergequeue_label`` broken-config default, and
``check_merge_flow`` mergequeue-stall detection.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _gh_dispatch,
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def test_load_orchestrator_config_ok_when_no_config_path_registered(
    hb: ModuleType, tmp_path: Path
) -> None:
    # load_repos() represents "no config registered for this repo" as
    # Path("") -- not a real file, must not be treated as cwd (Path("")
    # stringifies to "." and Path(".").exists() is True).
    config, error = hb.load_orchestrator_config(Path(""))
    assert config == {}
    assert error is None


def test_load_orchestrator_config_ok_when_file_absent(hb: ModuleType, tmp_path: Path) -> None:
    config, error = hb.load_orchestrator_config(tmp_path / "does-not-exist.yaml")
    assert config == {}
    assert error is None


def test_load_orchestrator_config_ok_when_valid(hb: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    config, error = hb.load_orchestrator_config(path)
    assert config == {"dispatch": {"max_concurrent_sessions": 3}}
    assert error is None


def test_load_orchestrator_config_error_on_invalid_yaml(hb: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("dispatch: [unterminated\n", encoding="utf-8")
    config, error = hb.load_orchestrator_config(path)
    assert config == {}
    assert error is not None
    assert str(path) in error


def test_load_orchestrator_config_error_on_non_utf8_bytes(hb: ModuleType, tmp_path: Path) -> None:
    # A concurrent partial write can leave non-UTF-8 bytes on disk. The
    # original implementation only caught (OSError, yaml.YAMLError) --
    # UnicodeDecodeError is a ValueError subclass, so this used to raise
    # straight out of read_text() instead of degrading. Must not raise.
    path = tmp_path / "orchestrator.config.yaml"
    path.write_bytes(b"\xff\xfe\x00garbage")
    config, error = hb.load_orchestrator_config(path)
    assert config == {}
    assert error is not None
    assert str(path) in error


def test_load_orchestrator_config_error_on_non_mapping_top_level(
    hb: ModuleType, tmp_path: Path
) -> None:
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    config, error = hb.load_orchestrator_config(path)
    assert config == {}
    assert error is not None
    assert "list" in error


def test_get_mergequeue_label_and_dispatch_cap_default_quietly_on_broken_config(
    hb: ModuleType, tmp_path: Path
) -> None:
    # get_mergequeue_label/get_dispatch_cap must keep degrading to None on a
    # broken config rather than raising or propagating the error -- callers
    # of these two functions are not where issue #703's signal should
    # surface; check_orchestrator_config is.
    path = tmp_path / "orchestrator.config.yaml"
    path.write_text("dispatch: [unterminated\n", encoding="utf-8")
    assert hb.get_mergequeue_label(path) is None
    assert hb.get_dispatch_cap(path) is None


def test_check_orchestrator_config_ok_when_not_registered(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo = replace(repo, config_path=Path(""))
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert not report.anomaly
    assert "no config_path registered" in report.lines[0]


def test_check_orchestrator_config_ok_when_absent(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)  # default config_path does not exist
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert not report.anomaly
    assert "not present" in report.lines[0]


def test_check_orchestrator_config_ok_when_valid(hb: ModuleType, tmp_path: Path) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert not report.anomaly
    assert "readable" in report.lines[0]


def test_check_orchestrator_config_anomaly_when_invalid_yaml(
    hb: ModuleType, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_text("dispatch: [unterminated\n", encoding="utf-8")
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert report.anomaly
    assert str(repo.config_path) in report.lines[0]


def test_check_orchestrator_config_anomaly_when_non_utf8(hb: ModuleType, tmp_path: Path) -> None:
    # Behavioral red case: pre-fix, this raised UnicodeDecodeError out of
    # main()'s per-repo loop instead of degrading -- the strongest evidence
    # that the fix, not just its return-type signature, changed behavior.
    repo = _make_repo(hb, tmp_path)
    repo.config_path.write_bytes(b"\xff\xfe\x00garbage")
    report = hb.Report()
    hb.check_orchestrator_config(report, repo)
    assert report.anomaly
    assert str(repo.config_path) in report.lines[0]


def test_check_merge_flow_ok_when_no_mergequeue_label(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    _gh_dispatch(
        monkeypatch,
        hb,
        lambda args, cwd: (True, [], ""),
    )
    report = hb.Report()
    hb.check_merge_flow(report, repo, {}, {}, skip_delta=False)
    assert not report.anomaly
    assert "merge-flow" in report.lines[0]


def test_check_merge_flow_anomaly_when_mergequeue_stalled(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _make_repo(hb, tmp_path)
    # Write a config with a mergequeue label so the check counts it.
    repo.config_path.parent.mkdir(parents=True, exist_ok=True)
    repo.config_path.write_text("auto_merge:\n  mergequeue_label: mergequeue\n", encoding="utf-8")

    merged_at = "2020-01-01T00:00:00Z"

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        if "--state" in args and "merged" in args[args.index("--state") + 1]:
            return True, [{"number": 9, "mergedAt": merged_at}], ""
        # open PRs: one carrying the mergequeue label
        return True, [{"number": 1, "labels": [{"name": "mergequeue"}]}], ""

    _gh_dispatch(monkeypatch, hb, handler)
    prev = {
        "mergequeue_count": 1,
        "mergequeue_unchanged_streak": 1,
        "last_merged_at": merged_at,
    }
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_merge_flow(report, repo, prev, new, skip_delta=False)
    assert report.anomaly
    assert "mergequeue count stuck" in report.lines[0]
