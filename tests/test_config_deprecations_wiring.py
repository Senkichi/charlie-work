"""Layer-pairing contract and ``fleet_loop`` wiring tests for the
deprecated-config-key retirement sweep (issue #1976).

Split out of ``tests/test_config_deprecations.py`` when that module grew past
the 800-line file-size-ratchet cap. These tests pin the integration seams:
``load_layered_config`` and the sweep enumerate the same layer slots through
``config_layer_paths``, and ``fleet_loop`` invokes
``_run_fleet_config_retirement_sweep`` once per pass, surfacing its attention
entries (and degrading its failures) in the consolidated digest. The registry,
read-time signal, sweep-behavior, and label-edge tests remain in the sibling
module; shared helpers live in ``tests/_config_deprecations_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from charlie_work import fleet_lanes
from charlie_work.config import OrchestratorConfig
from charlie_work.fleet_dispatch import fleet_loop
from charlie_work.global_config import config_layer_paths, load_layered_config
from _config_deprecations_fixtures import _DOTTED, _NOW, _write_repo_config
from _fleet_dispatch_fixtures import (
    _patch_ci_fleet_dirty_for_hermetic_tests as _patch_ci_fleet_dirty_for_hermetic_tests,
    _patch_self_deploy_for_fleet_tests as _patch_self_deploy_for_fleet_tests,
)


# ---------------------------------------------------------------------------
# Layer-pairing contract (loader <-> sweep)
# ---------------------------------------------------------------------------


def test_load_layered_config_consumes_config_layer_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """The loader resolves its layer files through config_layer_paths — the
    sweep enumerates the same function, so repointing it repoints the loader's
    reads too (this is the pin that keeps the two from drifting)."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    global_layer = fleet_dir / "config.yaml"
    global_layer.write_text("labels:\n  ready: fleet-ready\n", encoding="utf-8")
    repo_layer = tmp_path / "elsewhere.yaml"
    repo_layer.write_text("dispatch:\n  default_limit: 9\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()

    def _layers(repo_root, explicit=None, *, fleet_dir_override=None):
        return (("user-global", global_layer), ("repo", repo_layer))

    monkeypatch.setattr("charlie_work.global_config.config_layer_paths", _layers)

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    assert config.labels.ready == "fleet-ready"
    assert config.dispatch.default_limit == 9
    assert set(config.sources) == {str(global_layer), str(repo_layer)}


def test_config_layer_paths_match_what_the_loader_reads(tmp_path: Path):
    """The set of existing layer files the loader reports in ``sources`` is
    exactly the existing subset of config_layer_paths; absent layers are still
    enumerated as slots for the sweep."""
    fleet_dir = tmp_path / "fleet"
    fleet_dir.mkdir()
    (fleet_dir / "config.yaml").write_text("labels:\n  ready: fleet-ready\n", encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo_config(repo, "dispatch:\n  default_limit: 9\n")

    config = load_layered_config(repo, fleet_dir_override=str(fleet_dir))

    slot_paths = {
        str(path) for _, path in config_layer_paths(repo, fleet_dir_override=str(fleet_dir))
    }
    assert set(config.sources) == slot_paths

    # An absent repo layer is still a slot (the sweep counts it "checked").
    empty_repo = tmp_path / "empty"
    empty_repo.mkdir()
    slots = dict(config_layer_paths(empty_repo, fleet_dir_override=str(fleet_dir)))
    assert slots["repo"] == empty_repo / "orchestrator.config.yaml"
    assert slots["user-global"] == fleet_dir / "config.yaml"


# ---------------------------------------------------------------------------
# fleet_loop wiring
# ---------------------------------------------------------------------------


@patch("charlie_work.fleet_dispatch.compute_api_worker_fleet_report")
@patch("charlie_work.fleet_dispatch._run_fleet_config_retirement_sweep")
@patch("charlie_work.fleet_dispatch._load_registry")
@patch("charlie_work.fleet_dispatch.load_layered_config")
@patch("charlie_work.fleet_dispatch.runtime_paths")
@patch("charlie_work.fleet_dispatch.GitHub")
@patch("charlie_work.fleet_dispatch.OrchestratorApp")
def test_fleet_loop_runs_retirement_sweep_and_surfaces_attention(
    mock_app_class: MagicMock,
    mock_gh_class: MagicMock,
    mock_runtime_paths: MagicMock,
    mock_load_layered_config: MagicMock,
    mock_load_registry: MagicMock,
    mock_sweep: MagicMock,
    mock_compute_report: MagicMock,
    tmp_path: Path,
):
    """fleet_loop invokes the retirement sweep once per pass and its
    attention entries reach the consolidated digest (issue #1976)."""
    mock_load_registry.return_value = {"repos": {}}
    mock_compute_report.return_value = None
    mock_sweep.return_value = [
        {
            "repo_key": "fleet",
            "type": "config_retirement_armed",
            "key": _DOTTED,
            "issue_number": 1977,
            "reason": f"{_DOTTED} absent everywhere for 7.0d; removal issue marked Ready",
        }
    ]

    result = fleet_loop(
        fleet_dir_override=str(tmp_path / "fleet"),
        global_config=None,
        repos=None,
        limit=1,
        merge=None,
        dry_run=True,
        work_only=True,
    )

    mock_sweep.assert_called_once()
    sweep_args = mock_sweep.call_args.args
    assert sweep_args[0] == str(tmp_path / "fleet")  # fleet_dir_override
    digest = result.data["digest"]
    assert digest["count"] >= 1
    assert any(e.get("type") == "config_retirement_armed" for e in digest["events"])


def test_retirement_sweep_wrapper_survives_client_construction_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """If the GitHub client cannot even be constructed, the fleet-loop helper
    degrades to a config_retirement_error attention entry instead of raising
    into the pass (issue #1976 — 'the sweep never raises' at the call site)."""

    def _boom(**_kwargs):
        raise RuntimeError("gh ctor boom")

    monkeypatch.setattr(fleet_lanes, "GitHub", _boom)

    entries = fleet_lanes._run_fleet_config_retirement_sweep(
        str(tmp_path / "fleet"), OrchestratorConfig(), True, _NOW
    )

    assert [e["type"] for e in entries] == ["config_retirement_error"]
