"""Fleet registry ``last_seen`` bookkeeping (issues #1832, #2754).

Extracted verbatim from ``fleet_dispatch`` (file-size ratchet, issue #1442):
``fleet_dispatch`` is an over-cap monolith whose high-water mark may not grow,
and issue #2754's retirement wiring needs a few more lines there. Moving this
registry write out keeps the monolith under its mark. ``fleet_dispatch``
re-imports the name so ``fleet_dispatch._touch_registry_last_seen`` stays a
valid import path for ``fleet_loop`` and existing callers.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from .fleet_registry import _load_registry
from .state import save_state, state_lock

logger = logging.getLogger(__name__)


def _touch_registry_last_seen(
    fleet_json_path: Path,
    touched_keys: list[str],
    now: datetime.datetime,
    retired_by_key: dict[str, bool] | None = None,
) -> None:
    """Bump ``last_seen`` for every repo actually attempted this fleet pass.

    Issue #1832: ``_select_repos`` orders an implicit (no explicit ``repos``)
    pass by oldest ``last_seen`` first, but nothing previously bumped
    ``last_seen`` after a fleet pass ran a repo's lane -- only the
    single-repo CLI's ``fleet_registry.touch_repo`` did (cli.py), which a
    fleet pass never calls. Without this, the ordering never changed pass to
    pass: a repo whose lane runs to completion every time and a repo that
    gets deferred every time would sort identically next pass, so a slow
    repo that eats the whole in-pass deadline can starve the same later
    repos every single pass. Bumping every attempted repo's ``last_seen`` to
    "now" (including the slow one) means it sorts toward the back next pass,
    while a deferred (never-attempted) repo keeps its old timestamp and
    sorts toward the front -- rotation instead of static starvation.

    Deliberately excludes stale entries (repo_root missing) and
    deadline-deferred entries: a stale entry's ``last_seen`` drives issue
    #1372's grace-period auto-prune, and bumping it here would mask a dead
    registry entry as freshly active forever; a deferred entry was never
    attempted, so touching it would defeat the rotation this function exists
    to provide.

    Atomic and best-effort, mirroring :func:`_prune_stale_registry_entries`:
    the write happens under ``state_lock`` via ``save_state``'s temp-file +
    ``replace()``, and a failure is logged (never raised) so a registry
    write hiccup cannot fail the pass that already completed its lanes.
    """
    if not touched_keys:
        return
    stamp = now.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        with state_lock(fleet_json_path):
            data = _load_registry(fleet_json_path)
            repos = data.get("repos", {})
            changed = False
            for key in touched_keys:
                entry = repos.get(key)
                if entry is None:
                    continue
                entry["last_seen"] = stamp
                # Issue #2754: persist the derived flag for raw readers.
                if retired_by_key is not None and key in retired_by_key:
                    entry["retired"] = bool(retired_by_key[key])
                changed = True
            if changed:
                data["repos"] = repos
                # write-gate-exempt(issue=1832): standalone bookkeeping predates the WriteGate wave
                save_state(fleet_json_path, data)
    except Exception:
        logger.exception("Failed to bump fleet registry last_seen for touched repos")
