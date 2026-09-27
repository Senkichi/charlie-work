"""The notify digest's operator-facing freshness consumer (issue #1859).

The digest shipped as a signal without a consumer: the live daemon's
file-sink writer was dead for three weeks (last real entry 2026-08-31)
while nothing read the file at all. This module holds the pieces
``scripts/heartbeat_check.py``'s ``check_notify_digest_freshness`` composes
once per beat:

* the staleness bound both sides share, and
* :func:`probe_digest_file`, the guarded read-only probe of the digest path
  the *supervisor* resolved -- published in its ``notify_resolution``
  event -- never a path this check derives from the script's own checkout
  (round-1 review: the script's checkout and the daemon's config root are
  different trees on this host, so re-deriving answered the wrong question).

Leaf module, same contract as ``charlie_work.event_kinds``: stdlib-only,
no ``charlie_work`` or ``ci_fleet`` imports of its own, so the heartbeat
script can import it behind its guarded try/except and stay importable
even when the package install itself is broken -- the invariant
``scripts/README.md`` states as "a broken package install can never break
the check that would detect it".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: A digest that has produced nothing for this many hours is a dead writer,
#: not a quiet fleet. ``notify_freshness.NOTIFY_DIGEST_STALE_SECONDS`` (the
#: writer-side tripwire the fleet supervisor emits) is derived from this so
#: the two bounds can never drift apart.
#:
#: Sized from the real daemon digest's own gap distribution (measured
#: 2026-09-24): the longest healthy silence between attention transitions
#: was 47.1h and 2 of 575 gaps exceeded 24h, so the original day bound
#: alarmed on a healthy fleet. 72h sits ~1.5x above the observed maximum;
#: a fleet legitimately quieter than that rides the line down via
#: heartbeat-suppressions.yaml rather than weakening the default.
NOTIFY_DIGEST_STALE_HOURS = 72

#: How much of the digest's tail to scan for the last entry's generated_at.
#: A real digest line is ~300 bytes; 1 MiB covers thousands of entries.
_TAIL_BYTES = 1024 * 1024


@dataclass(frozen=True)
class DigestProbe:
    """Read-only probe of the supervisor-resolved digest path.

    ``exists`` is False when the file cannot be ``stat()``ed at all -- absent,
    unready volume, or unresolvable path are deliberately collapsed into one
    outcome, the same collapse ``_file_sink`` would hit on its next append.
    ``age_hours`` is only ever set when ``exists`` is True; ``age_source``
    records whether the age came from the last entry's ``generated_at`` or
    fell back to mtime. ``error`` carries the ``OSError`` detail when the
    file could not be stat()ed for a reason other than absence.
    """

    exists: bool
    age_hours: float | None
    age_source: str
    error: str | None = None


def probe_digest_file(
    path: Path,
    *,
    now: datetime | None,
    parse_iso: Callable[[str | None], datetime | None],
) -> DigestProbe:
    """Probe *path* (the digest location the supervisor published) read-only.

    Never raises: a hostile or vanished filesystem degrades to a
    ``DigestProbe`` with ``error`` set so the caller can decide WARN vs
    ANOMALY instead of crashing the beat before ``save_state``/report output
    (round-1 review: ``exists()`` and a non-string ``generated_at`` were
    both unguarded). Never creates the file or its parent -- a probe that
    writes would mask the writer-dead condition it exists to detect.

    Staleness is measured on the last entry's ``generated_at`` when one
    parses, falling back to the file's mtime -- the line timestamp
    distinguishes "real writer's last output" from a file merely touched by
    an unrelated writer (the false-freshness shape the dev checkout's
    digest exhibited on pytest writes).
    """
    resolved_now = now if now is not None else datetime.now(timezone.utc)

    try:
        stat_result = path.stat()
    except FileNotFoundError:
        return DigestProbe(exists=False, age_hours=None, age_source="missing")
    except OSError as exc:
        return DigestProbe(exists=False, age_hours=None, age_source="unreadable", error=str(exc))

    last_generated = None
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, stat_result.st_size - _TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="surrogateescape")
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                # A non-string generated_at would crash parse_iso's
                # str.replace -- treat it like an absent field.
                raw = entry.get("generated_at")
                if isinstance(raw, str):
                    last_generated = parse_iso(raw)
            break
    except OSError:
        pass  # mtime fallback below still produces a verdict

    if last_generated is not None:
        age = resolved_now - last_generated
        age_source = f"generated_at={last_generated.isoformat()}"
    else:
        age = resolved_now - datetime.fromtimestamp(stat_result.st_mtime, timezone.utc)
        age_source = "mtime (no parseable generated_at in tail)"

    return DigestProbe(exists=True, age_hours=age.total_seconds() / 3600.0, age_source=age_source)
