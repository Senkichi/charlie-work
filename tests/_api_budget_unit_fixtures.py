"""Shared helpers for the api-budget unit-test siblings (issue #1571).

Hoisted verbatim out of ``tests/test_api_budget.py`` when that module was
split into seam-named siblings (Track 1 shoulder) -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.api_budget import SessionEntry
from charlie_work.config import ApiProviderConfig


def _provider(
    *,
    input_usd_per_mtok: float = 3.0,
    output_usd_per_mtok: float = 15.0,
    cached_input_usd_per_mtok: float = 0.30,
) -> ApiProviderConfig:
    return ApiProviderConfig(
        base_url="https://api.example.com/anthropic",
        api_key_env="EXAMPLE_API_KEY",
        model="example-model",
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        cached_input_usd_per_mtok=cached_input_usd_per_mtok,
    )


def _entry(
    *,
    issue: int = 42,
    session_id: str = "sess-1",
    provider: str = "example",
    model: str = "example-model",
    started_at: str = "2026-07-22T10:00:00Z",
    ended_at: str = "2026-07-22T10:30:00Z",
    input_tokens: int = 1_000_000,
    output_tokens: int = 200_000,
    cached_tokens: int = 500_000,
    usd: float = 9.0,
    duration_s: float = 1800.0,
    outcome: str = "completed",
) -> SessionEntry:
    return SessionEntry(
        issue=issue,
        session_id=session_id,
        provider=provider,
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        usd=usd,
        duration_s=duration_s,
        outcome=outcome,
    )


def _assistant_event(
    msg_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> dict:
    return {
        "type": "assistant",
        "session_id": "sess-1",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        },
    }


def _result_event(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> dict:
    return {
        "type": "result",
        "subtype": "success",
        "session_id": "sess-1",
        "total_cost_usd": 0.0,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
    }


def _write_api_sidecar(sessions_dir: Path, issue_number: int, provider: str) -> Path:
    from charlie_work.claude_code import _sidecar_path

    sidecar = _sidecar_path(sessions_dir, issue_number, "api")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(
            {
                "issue_number": issue_number,
                "branch": "agent/x",
                "worktree_path": "",
                "prompt_path": "",
                "command": [],
                "pid": None,
                "started_at": "2026-07-22T10:00:00Z",
                "log_path": str(sessions_dir / f"issue-{issue_number}.claude.log"),
                "error": None,
                "failure_kind": None,
                "process_start_time": None,
                "reclaimed": None,
                "adapter_kind": "api",
                "provider": provider,
                "session_id": "sess-1",
            }
        ),
        encoding="utf-8",
    )
    return sidecar


def _write_events(sessions_dir: Path, issue_number: int, events: list[dict]) -> Path:
    events_path = sessions_dir / f"issue-{issue_number}.events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return events_path


def _api_worker_view(sessions_dir: Path, issue_number: int, provider: str) -> "object":
    from charlie_work.worker import WorkerView

    return WorkerView(
        adapter_kind="api",
        issue_number=issue_number,
        repo_key="",
        pid=None,
        started_at="2026-07-22T10:00:00Z",
        process_start_time=None,
        log_path=str(sessions_dir / f"issue-{issue_number}.claude.log"),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        session_id="sess-1",
    )
