"""Monitor events.db for new events and report what's being captured.

Runs in a loop, checking every 60 seconds for new events since the last
check. Reports event counts by kind, level, and correlation ID, plus
any errors or warnings seen.
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from charlie_work.instrumentation import (
    close_db,
    event_counts_by_kind,
    query_events,
    read_event_log,
)


def monitor(state_path: Path, interval: int = 60) -> None:
    last_count = 0
    last_check = datetime.now(UTC)

    print(f"Monitoring events.db at {state_path.parent / 'events.db'}")
    print(f"Checking every {interval}s. Press Ctrl+C to stop.\n")

    while True:
        try:
            events = read_event_log(state_path)
            current_count = len(events)

            if current_count > last_count:
                new_events = events[last_count:]
                print(f"\n[{datetime.now(UTC).isoformat()}] {len(new_events)} new event(s) "
                      f"(total: {current_count})")

                # Group by kind
                kinds: dict[str, int] = {}
                levels: dict[str, int] = {}
                cids: set[str] = set()
                for e in new_events:
                    k = e.get("kind", "?")
                    kinds[k] = kinds.get(k, 0) + 1
                    lv = e.get("level", "?")
                    levels[lv] = levels.get(lv, 0) + 1
                    cid = e.get("correlation_id")
                    if cid:
                        cids.add(cid)

                print(f"  Kinds: {dict(sorted(kinds.items(), key=lambda x: -x[1]))}")
                print(f"  Levels: {levels}")
                if cids:
                    print(f"  Correlation IDs: {cids}")

                # Show errors and warnings in detail
                for e in new_events:
                    if e.get("level") in ("error", "warning"):
                        print(f"  [{e.get('ts')}] {e.get('kind')} "
                              f"pr={e.get('pr_number')} issue={e.get('issue_number')} "
                              f"cid={e.get('correlation_id')} "
                              f"payload={e.get('payload')}")

                last_count = current_count
            elif current_count < last_count:
                # DB was reset or migration happened
                print(f"\n[{datetime.now(UTC).isoformat()}] Event count reset "
                      f"({last_count} -> {current_count})")
                last_count = current_count

            # Full summary every 5 minutes
            elapsed = (datetime.now(UTC) - last_check).total_seconds()
            if elapsed >= 300:
                counts = event_counts_by_kind(state_path)
                print(f"\n[{datetime.now(UTC).isoformat()}] === 5-min summary ===")
                print(f"  Total events: {current_count}")
                for kind, count in sorted(counts.items(), key=lambda x: -x[1]):
                    print(f"    {kind}: {count}")
                last_check = datetime.now(UTC)

        except Exception as exc:
            print(f"[ERROR] Monitor exception: {exc}", file=sys.stderr)

        time.sleep(interval)


if __name__ == "__main__":
    state_path = Path(r"C:\Users\senki\repos\charlie-work\.var\charlie-work\state.json")
    try:
        monitor(state_path, interval=60)
    except KeyboardInterrupt:
        print("\nStopping monitor.")
        close_db(state_path)
