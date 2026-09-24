"""The notify digest's operator-facing freshness consumer (issue #1859).

The digest shipped as a signal without a consumer: the live daemon's
file-sink writer was dead for three weeks (last real entry 2026-08-31)
while nothing read the file at all. This module holds the verdict logic
``scripts/heartbeat_check.py`` calls once per beat.

Leaf module, same contract as ``charlie_work.event_kinds``: stdlib-only,
no ``charlie_work`` or ``ci_fleet`` imports of its own, so the heartbeat
script can import it behind its guarded try/except and stay importable
even when the package install itself is broken -- the invariant
``scripts/README.md`` states as "a broken package install can never break
the check that would detect it".

The primitives this check shares with the rest of the heartbeat script --
the fleet-dir location, the YAML config loader, the ISO-8601 parser -- are
passed in as callables rather than duplicated here, because a second copy
is exactly the kind of drift source the leaf's existence is meant to
avoid. ``report`` is duck-typed: anything with the ``ok`` / ``warn`` /
``anom`` methods of the script's ``Report`` class works.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: A digest that has produced nothing for this many hours is a dead writer,
#: not a quiet fleet. ``notify_freshness.NOTIFY_DIGEST_STALE_SECONDS`` (the
#: writer-side tripwire the fleet supervisor emits) is derived from this so
#: the two bounds can never drift apart. A quiet-but-healthy fleet that
#: legitimately emits no attention transitions for a day rides the line down
#: via heartbeat-suppressions.yaml rather than weakening the default.
NOTIFY_DIGEST_STALE_HOURS = 24

#: How much of the digest's tail to scan for the last entry's generated_at.
#: A real digest line is ~300 bytes; 1 MiB covers thousands of entries.
_TAIL_BYTES = 1024 * 1024


def merged_notify_section(
    *,
    fleet_dir: Path,
    checkout_root: Path,
    load_orchestrator_config: Callable[[Path], tuple[dict[str, Any], str | None]],
) -> dict[str, Any]:
    """Merge the ``notify:`` section the way ``load_layered_config`` does.

    ``run_fleet_supervise`` builds its notify config from two layers: the
    global fleet layer (``<fleet_dir>/config.yaml``) underneath the
    checkout's own ``orchestrator.config.yaml``, per-key with the per-repo
    file winning. Unreadable/absent files contribute nothing, matching the
    layered loader's silent-{} treatment of a missing layer (a *corrupt*
    per-repo file is surfaced separately by check_orchestrator_config).
    """
    merged: dict[str, Any] = {}
    for path in (fleet_dir / "config.yaml", checkout_root / "orchestrator.config.yaml"):
        data, _error = load_orchestrator_config(path)
        section = data.get("notify")
        if isinstance(section, dict):
            merged.update(section)
    return merged


def check_notify_digest_freshness(
    report: Any,
    *,
    now: datetime | None,
    checkout_root: Path,
    fleet_dir: Path,
    load_orchestrator_config: Callable[[Path], tuple[dict[str, Any], str | None]],
    parse_iso: Callable[[str | None], datetime | None],
) -> None:
    """Flag an enabled file sink whose digest has gone missing or stale.

    Three outcomes, in order:

    * ``enabled`` resolves False (or the section is absent) -> ``WARN``. A
      fleet that never opted in shows this line every beat, so it cannot
      flip the exit code -- but on a fleet that DID opt in, this is the
      "config section lost" signal: enabled silently reverting to the
      dataclass default is exactly how the 2026-08-31 outage stayed
      invisible.
    * ``enabled`` + ``sink=file`` + digest missing/stale -> ``ANOMALY``: the
      writer is configured to produce this file and is not producing it.
    * ``enabled`` + non-file sink -> ``OK`` (nothing on disk to tail).

    Staleness is measured on the last entry's ``generated_at`` when one
    parses, falling back to the file's mtime -- the line timestamp
    distinguishes "real writer's last output" from a file merely touched by
    an unrelated writer (the false-freshness shape the dev checkout's
    digest exhibited on pytest writes).
    """
    check = "notify-digest"
    notify = merged_notify_section(
        fleet_dir=fleet_dir,
        checkout_root=checkout_root,
        load_orchestrator_config=load_orchestrator_config,
    )

    if not notify.get("enabled"):
        report.warn(
            check,
            "notify resolves to enabled=false (or the notify: section is absent "
            "from every config layer) -- the digest writer is off. If this fleet "
            "opted in to notifications, its notify: block was lost",
        )
        return

    sink = str(notify.get("sink") or "file").lower()
    if sink != "file":
        report.ok(check, f"enabled with sink={sink} (no digest file to tail)")
        return

    raw_path = notify.get("file_path")
    if not raw_path or not isinstance(raw_path, str):
        report.anom(
            check,
            "notify enabled with sink=file but file_path is unset -- "
            "every emit fails 'file_path is empty'",
        )
        return

    # The fleet supervisor hands emit_digest the raw (unresolved) file_path;
    # _file_sink interprets a relative path against the supervisor's cwd --
    # the checkout this script runs from in production.
    digest_path = Path(raw_path)
    if not digest_path.is_absolute():
        digest_path = checkout_root / digest_path

    resolved_now = now if now is not None else datetime.now(timezone.utc)

    if not digest_path.exists():
        report.anom(
            check,
            f"notify enabled but {digest_path} does not exist -- "
            "the writer has never landed a line",
        )
        return

    try:
        stat_result = digest_path.stat()
    except OSError as exc:
        report.anom(check, f"{digest_path} unreadable: {exc}")
        return

    last_generated = None
    try:
        with digest_path.open("rb") as handle:
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
                last_generated = parse_iso(entry.get("generated_at"))
            break
    except OSError:
        pass  # mtime fallback below still produces a verdict

    if last_generated is not None:
        age = resolved_now - last_generated
        age_source = f"generated_at={last_generated.isoformat()}"
    else:
        age = resolved_now - datetime.fromtimestamp(stat_result.st_mtime, timezone.utc)
        age_source = "mtime (no parseable generated_at in tail)"

    age_hours = age.total_seconds() / 3600.0
    facts = (
        f"last entry {age_hours:.1f}h old ({age_source}); "
        f"threshold={NOTIFY_DIGEST_STALE_HOURS}h path={digest_path}"
    )
    if age_hours > NOTIFY_DIGEST_STALE_HOURS:
        report.anom(
            check,
            f"notify digest writer looks dead: {facts} -- the enabled file "
            "sink has produced nothing past the staleness bound",
        )
    else:
        report.ok(check, facts)
