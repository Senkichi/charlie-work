"""One-shot verification of events.db health.

Usage:
    python scripts/verify_events.py [path/to/state.json]

The state.json path is optional. When omitted, it is resolved from the
current repo's layered config (``runtime.state_dir``, default
``.var/charlie-work``) via ``charlie_work.global_config.load_layered_config``,
so this script targets the right tree on any repo that overrides
``runtime.state_dir`` instead of silently assuming the default.
"""

import sys
from pathlib import Path
from charlie_work.instrumentation import (
    read_event_log,
    event_counts_by_kind,
    _db_path,
    _get_db,
    close_db,
)

if len(sys.argv) > 1:
    sp = Path(sys.argv[1])
else:
    from charlie_work.global_config import load_layered_config
    from charlie_work.paths import find_repo_root, runtime_paths

    repo_root = find_repo_root()
    config = load_layered_config(repo_root)
    sp = runtime_paths(repo_root, config.runtime.state_dir).state_file
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
if conn is None:
    # _get_db is best-effort and returns None when the database cannot be
    # opened. Without this guard the next line raises a bare AttributeError on
    # NoneType -- an unreadable failure in the one tool an operator reaches for
    # *because* the database is suspect. Fail loudly on stderr instead.
    print(
        f"ERROR: could not open events database for {sp}\n"
        f"  expected at: {_db_path(sp)}\n"
        "  (missing, locked by another process, or corrupt)",
        file=sys.stderr,
    )
    sys.exit(1)
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

# Ask instrumentation where the database is rather than re-deriving it here:
# a second spelling of this path is how it drifts out of agreement with the
# module that actually opens it.
db_path = _db_path(sp)
wal_path = db_path.with_name(db_path.name + "-wal")
print(f"DB file: {db_path}")
print(f"DB size: {db_path.stat().st_size} bytes")
print(f"WAL file exists: {wal_path.exists()}")
if wal_path.exists():
    print(f"WAL size: {wal_path.stat().st_size} bytes")

close_db(sp)
print()
print("=== Verification PASSED ===")
