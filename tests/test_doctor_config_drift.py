"""Tests for the config-drift doctor checks (issue #1849).

Covers the two ``doctor_config_drift`` checks end-to-end: the triage-label
map (``docs/agents/triage-labels.md`` vs ``LabelConfig.ready``) and the
Aviator required-checks mirror (``.aviator/config.yml`` vs
``auto_merge.required_checks``), plus their ``run_doctor`` wiring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.config import AutoMergeConfig, LabelConfig
from charlie_work.doctor import run_doctor
from charlie_work.doctor_config_drift import (
    _check_aviator_required_checks,
    _check_triage_label_map,
)
from charlie_work.paths import runtime_paths
from _doctor_fixtures import FakeDoctorGitHub, _config, _write_workflow

_TRIAGE_MAP = "docs/agents/triage-labels.md"


def _collect(fn: Any, repo_root: Path, config: Any) -> list[tuple[str, bool, str, str]]:
    collected: list[tuple[str, bool, str, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        collected.append((name, ok, detail, severity))

    fn(add, repo_root, config)
    return collected


def _write_triage_map(repo_root: Path, ready_row: str) -> None:
    target = repo_root / "docs" / "agents" / "triage-labels.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "# Triage labels\n\n"
        "| Label in mattpocock/skills | Label in our tracker | Meaning |\n"
        "| --- | --- | --- |\n"
        "| `needs-triage` | `needs-triage` | Maintainer evaluates |\n"
        f"{ready_row}\n",
        encoding="utf-8",
    )


def _write_aviator(repo_root: Path, required_checks: list[str]) -> None:
    target = repo_root / ".aviator" / "config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(f'      - "{name}"' for name in required_checks)
    target.write_text(
        f"merge_rules:\n  preconditions:\n    required_checks:\n{lines}\n",
        encoding="utf-8",
    )


# -- triage-label map ---------------------------------------------------------


def test_triage_label_map_absent_file_passes_not_adopted(tmp_path: Path) -> None:
    results = _collect(_check_triage_label_map, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert name == "triage-label map"
    assert ok is True
    assert "not adopted" in detail


def test_triage_label_map_matching_row_passes(tmp_path: Path) -> None:
    _write_triage_map(tmp_path, "| `ready-for-agent` | `automated-ready` | ready |")

    results = _collect(_check_triage_label_map, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is True
    assert "automated-ready" in detail


def test_triage_label_map_missing_row_is_error(tmp_path: Path) -> None:
    _write_triage_map(tmp_path, "| `wontfix` | `wontfix` | Will not be actioned |")

    results = _collect(_check_triage_label_map, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "error"
    assert "ready-for-agent" in detail
    assert "automated-ready" in detail


def test_triage_label_map_mismatch_is_error_naming_both(tmp_path: Path) -> None:
    _write_triage_map(tmp_path, "| `ready-for-agent` | `agent:ready` | ready |")

    results = _collect(_check_triage_label_map, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "error"
    assert "agent:ready" in detail
    assert "automated-ready" in detail


def test_triage_label_map_backticks_optional_in_first_cell(tmp_path: Path) -> None:
    _write_triage_map(tmp_path, "| ready-for-agent | automated-ready | ready |")

    results = _collect(_check_triage_label_map, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is True


def test_triage_label_map_reads_ready_from_config_not_literal(tmp_path: Path) -> None:
    # The expected label comes from the loaded LabelConfig, never a
    # hardcoded "automated-ready" string.
    config = _config(labels=LabelConfig(ready="custom-ready"))
    _write_triage_map(tmp_path, "| `ready-for-agent` | `custom-ready` | ready |")

    results = _collect(_check_triage_label_map, tmp_path, config)

    [(name, ok, detail, severity)] = results
    assert ok is True
    assert "custom-ready" in detail


def test_triage_label_map_invalid_utf8_is_error_not_exception(tmp_path: Path) -> None:
    target = tmp_path / "docs" / "agents" / "triage-labels.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xff\xfe# not utf-8\n")

    results = _collect(_check_triage_label_map, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "error"
    assert "could not read" in detail


# -- aviator required checks ---------------------------------------------------


def test_aviator_required_checks_absent_file_passes(tmp_path: Path) -> None:
    results = _collect(_check_aviator_required_checks, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert name == "aviator required checks"
    assert ok is True


def test_aviator_required_checks_equal_sets_pass(tmp_path: Path) -> None:
    _write_aviator(tmp_path, ["Lint", "Tests", "Collect-only gate"])
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=("Tests", "Lint", "Collect-only gate"))
    )

    results = _collect(_check_aviator_required_checks, tmp_path, config)

    [(name, ok, detail, severity)] = results
    assert ok is True


def test_aviator_required_checks_difference_is_warning_naming_both_sides(
    tmp_path: Path,
) -> None:
    _write_aviator(tmp_path, ["Lint", "Tests"])
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=("Lint", "Tests", "Collect-only gate"))
    )

    results = _collect(_check_aviator_required_checks, tmp_path, config)

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "warning"
    assert "Collect-only gate" in detail
    assert "auto_merge.required_checks" in detail

    # And the mirror-image drift: a check only on the Aviator side.
    _write_aviator(tmp_path, ["Lint", "Tests", "Collect-only gate", "Aviator-only"])
    results = _collect(_check_aviator_required_checks, tmp_path, config)

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "warning"
    assert "Aviator-only" in detail
    assert ".aviator" in detail


def test_aviator_required_checks_no_list_passes(tmp_path: Path) -> None:
    target = tmp_path / ".aviator" / "config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("merge_rules:\n  labels:\n    trigger: mergequeue\n", encoding="utf-8")
    # Non-empty config side: if the absent Aviator list were silently read as
    # an empty list, the set comparison would report a drift, not a pass.
    config = _config(auto_merge=AutoMergeConfig(required_checks=("Tests",)))

    results = _collect(_check_aviator_required_checks, tmp_path, config)

    [(name, ok, detail, severity)] = results
    assert ok is True
    assert severity == "warning"
    assert "merge_rules.preconditions.required_checks" in detail


def test_aviator_required_checks_unparseable_yaml_is_warning(tmp_path: Path) -> None:
    target = tmp_path / ".aviator" / "config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("merge_rules: [unclosed\n", encoding="utf-8")

    results = _collect(_check_aviator_required_checks, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "warning"
    assert "could not parse" in detail


def test_aviator_required_checks_invalid_utf8_is_warning(tmp_path: Path) -> None:
    target = tmp_path / ".aviator" / "config.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"\xff\xfe not utf-8\n")

    results = _collect(_check_aviator_required_checks, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is False
    assert severity == "warning"
    assert "could not parse" in detail


# -- run_doctor wiring ---------------------------------------------------------


def test_run_doctor_reports_both_config_drift_checks(tmp_path: Path) -> None:
    _write_workflow(tmp_path, "jobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n")
    _write_triage_map(tmp_path, "| `ready-for-agent` | `automated-ready` | ready |")
    _write_aviator(tmp_path, ["Tests"])
    config = _config(auto_merge=AutoMergeConfig(required_checks=("Tests",)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    triage = by_name["triage-label map"]
    aviator = by_name["aviator required checks"]
    assert triage.ok is True
    # Only the adopted path emits the mapped label — the "not adopted"
    # detail would name the file, never `automated-ready`.
    assert "automated-ready" in triage.detail
    assert aviator.ok is True
    assert "1 required check(s) match" in aviator.detail
    assert ok is True


def test_run_doctor_triage_drift_fails_overall_aviator_drift_does_not(
    tmp_path: Path,
) -> None:
    _write_workflow(tmp_path, "jobs:\n  test:\n    name: Tests\n    runs-on: ubuntu-latest\n")
    config = _config(auto_merge=AutoMergeConfig(required_checks=("Tests",)))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    # A mismatched triage row is severity=error: it hard-fails the run.
    _write_triage_map(tmp_path, "| `ready-for-agent` | `agent:ready` | ready |")
    _write_aviator(tmp_path, ["Tests"])
    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["triage-label map"].ok is False
    assert by_name["triage-label map"].severity == "error"
    assert "agent:ready" in by_name["triage-label map"].detail
    assert ok is False

    # Aviator drift alone is severity=warning: reported, never gating.
    _write_triage_map(tmp_path, "| `ready-for-agent` | `automated-ready` | ready |")
    _write_aviator(tmp_path, ["Tests", "Lint"])
    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["aviator required checks"].ok is False
    assert by_name["aviator required checks"].severity == "warning"
    assert "Lint" in by_name["aviator required checks"].detail
    assert ok is True
