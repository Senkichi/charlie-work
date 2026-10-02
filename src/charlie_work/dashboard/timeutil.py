"""UTC timestamp helpers shared by the dashboard metric modules."""

from __future__ import annotations

from datetime import UTC, datetime, tzinfo


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts).astimezone(UTC)


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def local_iso(ts: str, tz: tzinfo | None = None) -> str:
    """UTC ``ts`` rendered in ``tz`` (the host's local zone when None), offset included."""
    return parse_ts(ts).astimezone(tz).replace(microsecond=0).isoformat()
