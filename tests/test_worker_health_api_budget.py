"""API per-session budget tests for ``classify_worker_health``.

Split out of ``tests/test_worker_health.py`` (issue #1568, Track-1):
the issue #484 Signal 6.5 in-flight api-budget kill -- over-cap kill,
dormant cap unset, below-cap/non-api/no-provider/no-events healthy
paths. The ``_api_worker_view``/``_write_api_events`` helpers moved
verbatim with their only consumers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from _api_budget_fixtures import api_provider
from charlie_work.config import (
    ApiBudgetConfig,
    ApiWorkerConfig,
    OrchestratorConfig,
)
from charlie_work.worker import (
    WorkerHealth,
    WorkerView,
    classify_worker_health,
)


# ---------------------------------------------------------------------------
# Issue #484: in-flight api per-session budget kill (Signal 6.5)
# ---------------------------------------------------------------------------


def _api_worker_view(tmp_path: Path, *, provider: str = "example") -> WorkerView:
    """Build a live api WorkerView whose log_path points at issue-1.claude.log."""
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("Working on task...", encoding="utf-8")
    recent_start = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    return WorkerView(
        adapter_kind="api",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
        provider=provider,
    )


def _write_api_events(tmp_path: Path, cost_usd: float = 6.15) -> Path:
    """Write an events.jsonl whose usage yields ``cost_usd`` at default pricing.

    With the default ``api_provider`` pricing (input=3.0, output=15.0,
    cached=0.30 per MTok), 1M input + 0.2M output + 0.5M cached = 6.15 USD.
    """
    events_file = tmp_path / "sessions" / "issue-1.events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)
    events_file.write_text(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 200_000,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 500_000,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return events_file


def test_classify_worker_health_api_budget_kill_over_cap(tmp_path: Path) -> None:
    """Issue #484: an api worker whose accumulated cost exceeds
    ``max_usd_per_session`` is classified RUNAWAY (kill)."""
    _write_api_events(tmp_path)  # 6.15 USD at default pricing
    view = _api_worker_view(tmp_path)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=5.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.RUNAWAY


def test_classify_worker_health_api_budget_dormant_cap_unset(tmp_path: Path) -> None:
    """Issue #484: when ``max_usd_per_session`` is 0/unset the check is entirely
    dormant — no kill at any cost level (HEALTHY)."""
    _write_api_events(tmp_path)  # 6.15 USD — would exceed a 5.0 cap, but cap is 0
    view = _api_worker_view(tmp_path)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=0.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_below_cap_healthy(tmp_path: Path) -> None:
    """Issue #484: an api worker under the cap is HEALTHY."""
    _write_api_events(tmp_path)  # 6.15 USD
    view = _api_worker_view(tmp_path)

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=10.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_non_api_never_evaluated(
    tmp_path: Path,
) -> None:
    """Issue #484: non-api workers are never budget-evaluated via the api
    per-session cap, even with high-cost events and a low cap configured."""
    _write_api_events(tmp_path)  # 6.15 USD
    log_file = tmp_path / "sessions" / "issue-1.claude.log"
    recent_start = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    view = WorkerView(
        adapter_kind="claude-code",
        issue_number=1,
        repo_key="",
        pid=12345,
        started_at=recent_start,
        process_start_time=1710000000.0,
        log_path=str(log_file),
        worktree_path="",
        error=None,
        failure_kind=None,
        reclaimed=None,
    )

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        # api budget cap is low, but the worker is claude-code, not api.
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=1.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        # Claude Code's own Signal 6 uses self-reported cost_usd; the events
        # fixture has no cost_usd field, so Signal 6 is dormant too → HEALTHY.
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_no_provider_healthy(tmp_path: Path) -> None:
    """Issue #484: an api worker with an empty/unknown provider is not
    budget-evaluated (no pricing → cannot compute cost → HEALTHY)."""
    _write_api_events(tmp_path)
    view = _api_worker_view(tmp_path, provider="")

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=1.0),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY


def test_classify_worker_health_api_budget_no_events_healthy(tmp_path: Path) -> None:
    """Issue #484: an api worker with no events.jsonl yet (young session) is
    not budget-killed — absence is not over-budget."""
    view = _api_worker_view(tmp_path)
    # No events.jsonl written.

    with patch("charlie_work.worker.is_worker_alive", return_value=True):
        config = OrchestratorConfig(
            api_worker=ApiWorkerConfig(
                enabled=True,
                provider="example",
                providers={"example": api_provider()},
                budget=ApiBudgetConfig(max_usd_per_session=0.01),
            )
        )
        now = datetime.now(UTC)
        health = classify_worker_health(view, config, now)
        assert health == WorkerHealth.HEALTHY
