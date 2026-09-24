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

    results = _collect(_check_aviator_required_checks, tmp_path, _config())

    [(name, ok, detail, severity)] = results
    assert ok is True


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
    assert by_name["triage-label map"].ok is True
    assert by_name["aviator required checks"].ok is True
    assert ok is True
