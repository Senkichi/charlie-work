"""One-shot verification of events.db health."""

from pathlib import Path
from charlie_work.instrumentation import (
    read_event_log,
    event_counts_by_kind,
    _get_db,
    close_db,
)

sp = Path(r"C:\Users\senki\repos\charlie-work\.var\charlie-work\state.json")
evts = read_event_log(sp)
print("=== Final Verification ===")
print(f"Total events captured: {len(evts)}")
print()

counts = event_counts_by_kind(sp)
print("Event counts by kind:")
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print()

levels = {}
for e in evts:
    lv = e.get("level", "unknown")
    levels[lv] = levels.get(lv, 0) + 1
print("Event counts by level:")
for lv, cnt in sorted(levels.items()):
    print(f"  {lv}: {cnt}")
print()

cids = set(e.get("correlation_id") for e in evts if e.get("correlation_id"))
print(f"Unique correlation IDs: {len(cids)}")
print()

conn = _get_db(sp)
cursor = conn.execute("SELECT COUNT(*) FROM loop_passes")
lp_count = cursor.fetchone()[0]
cursor = conn.execute("SELECT * FROM loop_passes ORDER BY started_at DESC LIMIT 5")
rows = cursor.fetchall()
print(f"Loop passes recorded: {lp_count}")
print("Last 5 loop passes:")
for r in rows:
    print(
        f"  cid={r[0]} started={r[1]} completed={r[2]} ok={r[3]} elapsed={r[4]}s errors={r[5]} merges={r[6]} reviews={r[7]}"
    )
print()

db_path = sp.parent / "events.db"
wal_path = sp.parent / "events.db-wal"
print(f"DB file: {db_path}")
print(f"DB size: {db_path.stat().st_size} bytes")
print(f"WAL file exists: {wal_path.exists()}")
if wal_path.exists():
    print(f"WAL size: {wal_path.stat().st_size} bytes")

close_db(sp)
print()
print("=== Verification PASSED ===")
