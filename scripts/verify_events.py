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

# A verifier must not manufacture the artifact it is verifying. Every read
# helper below (read_event_log, event_counts_by_kind, and the explicit
# _get_db call further down) resolves its connection through
# instrumentation._get_db, which creates a brand-new empty events.db --
# plus any missing parent directories -- the instant it is pointed at a
# path that doesn't exist yet. That behaviour is correct and load-bearing
# for the live orchestrator (first-run provisioning); it is wrong here,
# because it means simply pointing this script at the wrong tree would
# silently fabricate the very evidence it's supposed to check for and then
# report a clean pass against it. So confirm both the state path and its
# events.db genuinely pre-exist BEFORE touching anything that can create
# them.
if not sp.exists():
    print(
        f"ERROR: state.json not found at {sp}\n"
        "  refusing to verify -- opening the events database for a "
        "nonexistent state path would silently create an empty one",
        file=sys.stderr,
    )
    sys.exit(1)

expected_db_path = _db_path(sp)
if not expected_db_path.exists():
    print(
        f"ERROR: no events.db found at {expected_db_path}\n"
        f"  (derived from state path {sp})\n"
        "  refusing to verify a freshly-created empty database -- run the "
        "orchestrator at least once against this state path first",
        file=sys.stderr,
    )
    sys.exit(1)

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

# Reuse the path confirmed to exist above rather than re-deriving it again:
# a second spelling of this path is how it drifts out of agreement with the
# module that actually opens it.
db_path = expected_db_path
wal_path = db_path.with_name(db_path.name + "-wal")
print(f"DB file: {db_path}")
print(f"DB size: {db_path.stat().st_size} bytes")
print(f"WAL file exists: {wal_path.exists()}")
if wal_path.exists():
    print(f"WAL size: {wal_path.stat().st_size} bytes")
print()

# The existence checks above only prove the file isn't fresh -- they don't
# prove it has ever recorded anything. A database that pre-exists but holds
# zero events AND zero loop passes is indistinguishable, from an operator's
# point of view, from the exact false positive this script was fixed to
# stop reporting: after a state-dir rename or repo-root mixup, an operator
# could easily be pointed at some other real-but-unrelated empty database.
# The one case this does NOT fail is a genuinely mixed reading (events
# logged but no loop pass yet, or vice versa) -- that reflects a real
# partial run, not an all-zero non-signal, so it still prints PASSED.
if len(evts) == 0 and lp_count == 0:
    print(
        f"ERROR: {sp} has a pre-existing events.db, but it contains zero "
        "events and zero loop passes\n"
        "  refusing to report PASSED against an all-zero result -- this is "
        "indistinguishable from being pointed at the wrong tree\n"
        "  (if this is genuinely a brand-new instance, run the orchestrator "
        "at least once first, then re-run this script)",
        file=sys.stderr,
    )
    close_db(sp)
    sys.exit(1)

close_db(sp)
print("=== Verification PASSED ===")
