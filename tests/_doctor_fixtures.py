"""Shared fakes/helpers for the doctor test modules.

Hoisted verbatim out of ``tests/test_doctor.py`` (issue #1563, Track 1
shoulder) when that module was split into seam-named siblings -- the
``tests/_*.py`` hoisted-fixture convention is the sanctioned import target
for shared test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from charlie_work.config import (
    ApiBudgetConfig,
    OrchestratorConfig,
    ReviewDispatchConfig,
)
from charlie_work.config import (
    ApiProviderConfig,
    ApiWorkerConfig,
)


def _write_workflow(repo_root: Path, body: str) -> None:
    workflows = repo_root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "ci.yml").write_text(body, encoding="utf-8")


def _collect_allocation_checks(
    config: Any, fleet_dir: Path, *, now: Any = None
) -> list[tuple[str, bool, str]]:
    from charlie_work.doctor_allocation import _check_runner_allocation

    collected: list[tuple[str, bool, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        collected.append((name, ok, detail))

    _check_runner_allocation(add, config, fleet_dir_override=str(fleet_dir), now=now)
    return collected


def _patch_resolve_to_diverge(monkeypatch: Any, literal: Path, redirected: Path) -> None:
    """Make ``Path.resolve()`` return ``redirected`` for ``literal`` only.

    Every other path resolves normally, so the rest of ``run_doctor`` is
    unaffected. This is the "patch the resolution step" injection the issue
    prescribes: the real ``fleet_dir_virtualization`` logic runs end-to-end,
    only the filesystem's answer is forged.
    """
    import os as _os
    import pathlib

    real_resolve = pathlib.Path.resolve

    def fake_resolve(self, *args, **kwargs):
        result = real_resolve(self, *args, **kwargs)
        if _os.path.normcase(_os.fspath(result)) == _os.path.normcase(_os.fspath(literal)):
            return redirected
        return result

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)


def _api_provider(
    *,
    api_key_env: str = "MOONSHOT_API_KEY",
    base_url: str = "https://api.moonshot.ai/anthropic",
) -> ApiProviderConfig:
    return ApiProviderConfig(
        base_url=base_url,
        api_key_env=api_key_env,
        model="kimi-k3",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.30,
    )


def _write_sidecar(sessions_dir: Path, name: str, payload: dict) -> None:
    (sessions_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def _write_workflow_named(repo_root: Path, filename: str, body: str) -> None:
    """Write a workflow file by name, tolerating an existing workflows dir.

    Unlike ``_write_workflow`` (which always writes ``ci.yml`` and would raise
    on a second ``mkdir``), this lets a test plant several workflow files in
    one repo -- the multi-workflow scenario the issue #1508 matrix-scoping fix
    targets.
    """
    workflows = repo_root / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / filename).write_text(body, encoding="utf-8")


def _doctor_allocation_config(*, enabled: bool = True, budget: int = 8) -> Any:
    from dataclasses import replace

    from charlie_work.config import OrchestratorConfig

    base = OrchestratorConfig()
    return replace(
        base,
        runner_allocation=replace(
            base.runner_allocation, enabled=enabled, max_running_runners=budget
        ),
    )


def _write_allocation_stamp(
    fleet_dir: Path,
    *,
    age_seconds: float,
    source: Any,
    full_pass_interval_seconds: int | None = None,
    skip_reason: str | None = None,
    now: Any = None,
) -> None:
    """Write a state file aged ``age_seconds`` (negative = future-dated).

    ``now`` is the reference instant the age is computed against; defaults to
    the real wall clock. Pass a frozen value (issue #828) when the caller
    also passes the same value to ``_collect_allocation_checks`` so a tight
    downstream assertion cannot race an unbounded CI stall between the write
    here and the probe's own clock read.
    """
    import datetime

    from ci_fleet.charlie_work_adapter import ALLOCATION_STATE_FILENAME

    reference = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    when = reference - datetime.timedelta(seconds=age_seconds)
    payload: dict[str, Any] = {"version": 1, "updated_at": when.isoformat(), "repos": {}}
    if source is not None:
        payload["source"] = source
    if full_pass_interval_seconds is not None:
        payload["full_pass_interval_seconds"] = full_pass_interval_seconds
    if skip_reason is not None:
        payload["skip_reason"] = skip_reason
    (fleet_dir / ALLOCATION_STATE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _write_supervisor_heartbeat(fleet_dir: Path, payload: dict) -> None:
    """Plant a ``supervisor-heartbeat.json`` sidecar in the fleet dir.

    The doctor probe's ``fleet_dir_override`` makes ``fleet_dir`` the
    host-wide fleet directory, so the heartbeat lands beside the allocation
    stamp exactly as the real supervisor writes it.
    """
    from charlie_work.supervisor_lifecycle import HEARTBEAT_FILENAME

    (fleet_dir / HEARTBEAT_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _config(**kwargs) -> OrchestratorConfig:
    # The real default (review_dispatch.enabled=False, rescue.enabled=False)
    # is exactly the "no automated review-to-verdict path" gap the new
    # doctor check flags -- so every pre-existing test in this module that
    # doesn't care about that check would otherwise trip it incidentally.
    # Default review_dispatch on here; tests that specifically exercise the
    # new check override it explicitly (see
    # test_doctor_flags_no_automated_review_to_verdict_path).
    kwargs.setdefault("review_dispatch", ReviewDispatchConfig(enabled=True))
    return OrchestratorConfig(**kwargs)


class FakeDoctorGitHub:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels if labels is not None else []

    def run(self, args, **kwargs):
        return ""

    def label_list(self):
        return [{"name": name} for name in self.labels]


def _api_worker_config(
    *,
    enabled: bool = True,
    provider: ApiProviderConfig | None = None,
    budget: ApiBudgetConfig | None = None,
    provider_name: str = "kimi-k3",
) -> ApiWorkerConfig:
    return ApiWorkerConfig(
        enabled=enabled,
        provider=provider_name,
        providers={provider_name: provider or _api_provider()},
        budget=budget or ApiBudgetConfig(),
    )


def _make_fleet_json(tmp_path: Path, state_dir: Path) -> Path:
    """Create a fleet.json with one repo pointing at the given state_dir."""
    fleet_dir_path = tmp_path / "fleet"
    fleet_dir_path.mkdir(parents=True, exist_ok=True)
    fleet_json = fleet_dir_path / "fleet.json"
    fleet_json.write_text(
        json.dumps(
            {
                "version": 1,
                "repos": {
                    "owner/repo": {
                        "repo_root": str(tmp_path),
                        "state_dir": str(state_dir),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return fleet_json
