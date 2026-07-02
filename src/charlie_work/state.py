from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STATE_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def empty_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "generated_at": utc_now(),
        "issues": {},
        "prs": {},
        "events": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_state()
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        # Never crash the orchestrator on a truncated/corrupt state file, and
        # never silently discard it either — quarantine it for forensics.
        quarantine = path.with_name(f"{path.name}.corrupt-{utc_now().replace(':', '')}")
        try:
            path.replace(quarantine)
        except OSError:
            pass
        return empty_state()
    if not isinstance(data, dict):
        return empty_state()
    data.setdefault("version", STATE_VERSION)
    data.setdefault("generated_at", utc_now())
    data.setdefault("issues", {})
    data.setdefault("prs", {})
    data.setdefault("events", [])
    return data


def save_state(path: Path, data: dict[str, Any]) -> None:
    data["generated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def append_event(data: dict[str, Any], kind: str, payload: dict[str, Any]) -> None:
    events = data.setdefault("events", [])
    events.append({"at": utc_now(), "kind": kind, "payload": payload})
    if len(events) > 200:
        del events[:-200]
