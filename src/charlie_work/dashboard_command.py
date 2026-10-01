"""``charlie dashboard`` subcommands (ADR-0008): read-only views over fleet sources.

``rollup`` runs one derived-table pass into ``dashboard.db`` (the only write, and
only to that derived file); ``now`` prints the Now model built from local sources
with no alarm findings. Both honour the ``dashboard.enabled`` kill switch and
return ``CommandResult`` values -- failures never raise.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import UTC, datetime, timedelta
from typing import Any

import yaml

from .config import build_config_from_data
from .config_validation import ConfigError
from .dashboard.config import DASHBOARD_SECTION, DashboardConfig
from .fleet_paths import fleet_dir
from .layout import GLOBAL_CONFIG_FILENAME
from .workflow import CommandResult

DASHBOARD_COMMAND = "dashboard"


def register_dashboard_subparsers(subparsers: Any) -> None:
    """Add ``dashboard {rollup,now}`` to the top-level subparsers."""
    dashboard = subparsers.add_parser(
        DASHBOARD_COMMAND, help="Read-only fleet dashboard data (ADR-0008)"
    )
    sub = dashboard.add_subparsers(dest="dashboard_command", required=True)
    sub.add_parser("rollup", help="Run one rollup pass into dashboard.db and print the result")
    sub.add_parser("now", help="Print the Now model built from local sources")
    history = sub.add_parser("history", help="Print History metric series with takeaways")
    history.add_argument("--days", type=int, default=7, help="Window length in days")
    history.add_argument("--bucket-hours", type=int, default=24, help="Bucket size in hours")


def _load_dashboard_config(fleet_dir_override: str | None) -> DashboardConfig:
    """Read just the host-wide ``dashboard:`` section of the global fleet config."""
    path = fleet_dir(override=fleet_dir_override) / GLOBAL_CONFIG_FILENAME
    if not path.exists():
        return DashboardConfig()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = raw.get(DASHBOARD_SECTION) if isinstance(raw, dict) else None
    if section is None:
        return DashboardConfig()
    return build_config_from_data({DASHBOARD_SECTION: section}).dashboard


def _plain(value: Any) -> Any:
    """Dataclass tree -> JSON-ready dict (datetimes become ISO strings)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _run_history(
    days: int, bucket_hours: int, now: datetime, fleet_dir_override: str | None
) -> CommandResult:
    """Every History series over the last ``days`` plus its takeaway vs the prior window."""
    from . import layout
    from .dashboard.metrics import MetricQuery, all_series, open_dashboard_ro
    from .dashboard.takeaways import takeaway

    if days < 1 or bucket_hours < 1:
        return CommandResult(False, "--days and --bucket-hours must be at least 1", {})
    query = MetricQuery(now - timedelta(days=days), now, timedelta(hours=bucket_hours))
    db, error = open_dashboard_ro(layout.dashboard_db_path(override=fleet_dir_override))
    if db is None:
        return CommandResult(
            False, f"dashboard.db unavailable ({error}); run `dashboard rollup`", {}
        )
    try:
        current = all_series(db, query)
        prior = all_series(db, query.prior())
    finally:
        db.close()
    tabs = {}
    for tab, metrics in current.items():
        tabs[tab] = {}
        for metric, series in metrics.items():
            # Category metrics may carry different categories per window: pair by name.
            old = {s.name: s for s in prior[tab][metric]}
            tabs[tab][metric] = [
                {**_plain(cur), "takeaway": takeaway(cur, old[cur.name])}
                if cur.name in old
                else {**_plain(cur), "takeaway": "not enough data"}
                for cur in series
            ]
    return CommandResult(True, f"dashboard history ({days}d)", {"tabs": tabs})


def run_dashboard_command(args: argparse.Namespace) -> CommandResult:
    """Dispatch ``charlie dashboard <sub>``; config/IO failures come back as values."""
    override = args.fleet_dir
    try:
        config = _load_dashboard_config(override)
    except (ConfigError, ValueError, OSError, yaml.YAMLError) as exc:
        return CommandResult(False, f"dashboard config error: {exc}", {})
    if not config.enabled:
        return CommandResult(
            False, "dashboard is disabled (dashboard.enabled: false in the fleet config)", {}
        )
    now = datetime.now(UTC)
    if args.dashboard_command == "rollup":
        from .dashboard.rollup import run_rollup, rollup_sources

        result = run_rollup(rollup_sources(override), now)
        data = {**_plain(result), "ingested": result.ingested, "errors": list(result.errors)}
        message = (
            f"rollup ingested {result.ingested} event(s)"
            if not result.errors
            else f"rollup finished with errors: {'; '.join(result.errors)}"
        )
        return CommandResult(not result.errors, message, data)
    if args.dashboard_command == "now":
        from .dashboard.now_collect import collect_sources_read
        from .dashboard.now_model import build_now_model

        model = build_now_model(collect_sources_read(now, override), now, findings=[])
        return CommandResult(True, "dashboard now", _plain(model))
    if args.dashboard_command == "history":
        return _run_history(args.days, args.bucket_hours, now, override)
    return CommandResult(False, f"unknown dashboard command: {args.dashboard_command}", {})
