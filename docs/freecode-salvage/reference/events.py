"""Decision logging (implementation plan section 8).

Each emitted record has top-level ``schema: "freecode/decision/1"``, ``ts`` (UTC
ISO8601), and ``event_type``. Additional keys are payload fields (scrubbed).

Stable ``event_type`` strings and required payload fields are documented in
``docs/decision_events_schema.md``. Do not rename event types or
payload keys without a schema bump and changelog entry.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from freecode.config import logs_dir
from freecode.observability.redaction import scrub
from freecode.observability.sinks import HmacChainSink, JsonlFileSink

_DECISIONS: list[dict[str, Any]] = []
_DEFAULT_SINK: Any | None = None


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class DecisionEvent:
    """Structured routing / policy decision (no secrets, no prompts)."""

    event_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        payload = scrub(dict(self.payload))
        return {
            "schema": "freecode/decision/1",
            "ts": _now_iso(),
            "event_type": self.event_type,
            **payload,
        }

    def to_json_line(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True, default=str)


def configure_decision_logging(*, audit_hmac_seed: bytes | None = None) -> None:
    """Configure process-local durable decision logging."""

    global _DEFAULT_SINK
    base = JsonlFileSink(logs_dir())
    _DEFAULT_SINK = (
        HmacChainSink(base, seed=audit_hmac_seed) if audit_hmac_seed is not None else base
    )


def reset_decision_logging_for_tests() -> None:
    global _DEFAULT_SINK
    _DEFAULT_SINK = None


def emit_decision(event: DecisionEvent, sink: Any | None = None) -> None:
    """Emit a schema-versioned, redacted decision record."""

    record = event.to_record()
    line = json.dumps(record, sort_keys=True, default=str)
    target = sink if sink is not None else _default_sink()
    if target is not None:
        if hasattr(target, "write_record"):
            target.write_record(record)
        else:
            target.write(line + "\n")
    else:
        print(line, file=sys.stderr, flush=True)
    # Preserve the legacy stderr stream for local tooling even when durable logging is active.
    if sink is not None:
        return
    if target is not None:
        print(line, file=sys.stderr, flush=True)
    _DECISIONS.append(record)


def _default_sink() -> Any | None:
    global _DEFAULT_SINK
    if _DEFAULT_SINK is None:
        try:
            _DEFAULT_SINK = JsonlFileSink(logs_dir())
        except OSError:
            return None
    return _DEFAULT_SINK


def drained_test_decisions() -> list[dict[str, Any]]:
    """Test helper: return accumulated decisions and clear buffer."""

    out = list(_DECISIONS)
    _DECISIONS.clear()
    return out
