"""Shared fixtures for api-worker budget settlement wiring tests (issue #480).

These helpers build the on-disk shape an ``adapter_kind == "api"`` worker
leaves behind (an ``issue-<n>.api.json`` sidecar + a sibling
``issue-<n>.events.jsonl``) so integration-level tests can drive the
production reap call sites in ``reconcile.py`` and ``workflow.py`` and
assert that spend is settled into the api-budget ledger.

Kept side-effect-free and dependency-light so it can be imported from any
test module without pulling the full ``test_api_budget`` namespace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.api_budget import LEDGER_FILENAME, ledger_path, load_ledger
from charlie_work.claude_code import _sidecar_path
from charlie_work.config import ApiProviderConfig, ApiWorkerConfig


def api_provider(
    *,
    input_usd_per_mtok: float = 3.0,
    output_usd_per_mtok: float = 15.0,
    cached_input_usd_per_mtok: float = 0.30,
) -> ApiProviderConfig:
    """A pricing-valid provider named ``example`` (model ``example-model``)."""
    return ApiProviderConfig(
        base_url="https://api.example.com/anthropic",
        api_key_env="EXAMPLE_API_KEY",
        model="example-model",
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        cached_input_usd_per_mtok=cached_input_usd_per_mtok,
    )


def api_worker_config(provider_name: str = "example") -> ApiWorkerConfig:
    """An ``enabled`` ApiWorkerConfig with one registered provider.

    ``enabled=True`` exercises the same config-validation path production
    runs, so a wiring test that constructs this config also guards against a
    regression that would make the reap path receive an invalid registry.
    """
    return ApiWorkerConfig(
        enabled=True,
        provider=provider_name,
        providers={provider_name: api_provider()},
    )


def write_api_sidecar(
    sessions_dir: Path,
    issue_number: int,
    *,
    provider: str = "example",
    pid: int | None = None,
    error: str | None = None,
    failure_kind: str | None = None,
    session_id: str = "sess-1",
    started_at: str = "2026-07-22T10:00:00Z",
) -> Path:
    """Write an ``issue-<n>.api.json`` sidecar that ``iter_workers`` discovers.

    The ``log_path`` points at ``issue-<n>.claude.log`` so
    ``_events_path_from_log`` derives the sibling ``issue-<n>.events.jsonl``.
    """
    sidecar = _sidecar_path(sessions_dir, issue_number, "api")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    payload: dict[str, Any] = {
        "issue_number": issue_number,
        "branch": f"agent/issue-{issue_number}",
        "worktree_path": "",
        "prompt_path": "",
        "command": [],
        "pid": pid,
        "started_at": started_at,
        "log_path": str(log_path),
        "error": error,
        "failure_kind": failure_kind,
        "process_start_time": None,
        "reclaimed": None,
        "adapter_kind": "api",
        "provider": provider,
        "session_id": session_id,
    }
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    # The log file must exist so stat()-based helpers (update_worker_log_stat,
    # _log_is_stalled_at_shim) don't short-circuit; its content is irrelevant
    # for settlement, which reads the events.jsonl sibling.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    return sidecar


def write_api_events(
    sessions_dir: Path,
    issue_number: int,
    *,
    input_tokens: int = 1_000_000,
    output_tokens: int = 200_000,
    cache_read: int = 500_000,
) -> Path:
    """Write a terminal ``result`` event the settlement parser consumes.

    With the default ``api_provider`` pricing the settled USD is
    ``1M*3 + 0.2M*15 + 0.5M*0.30 = 6.15``.
    """
    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "total_cost_usd": 0.0,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": cache_read,
        },
    }
    events_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
    return events_path


def ledger_entries(state_dir: Path):
    """Load the api-budget ledger under ``state_dir`` (empty list if absent)."""
    return load_ledger(ledger_path(state_dir)).sessions


def ledger_file(state_dir: Path) -> Path:
    return Path(state_dir) / LEDGER_FILENAME
