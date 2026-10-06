"""Per-repo ``state.json`` + reviewer-sidecar reads for the Now collector (read-only).

``state.json`` is replaced atomically by its writer, so a lock-free read never sees a torn
file (``dispatch_selection._count_live_reviews`` takes the state lock; a read-only page
must not contend for it). Unreadable inputs come back as ``None`` / empty, never as zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import host as _host
from .. import layout
from ..worker import iter_workers
from . import sources as src

# state.json ``issues[N]`` keys that stamp when an issue entered its sink, newest-meaning
# first: ``terminal_since`` (escalation) else the merged-PR-mention flag (mention-flagged
# human-needed issues carry no ``terminal_since``).
_SINCE_KEYS = ("terminal_since", "merged_pr_mention_flagged_at")
_DISPATCHED = "review_dispatch_dispatched"


@dataclass(frozen=True)
class RepoStateRead:
    escalated_since: tuple[tuple[int, datetime], ...]
    reviewers_live: int | None  # None = could not be measured


def _escalated_since(issues: Any) -> tuple[tuple[int, datetime], ...]:
    out: list[tuple[int, datetime]] = []
    for key, entry in (issues if isinstance(issues, dict) else {}).items():
        if not isinstance(entry, dict) or not str(key).isdigit():
            continue
        for name in _SINCE_KEYS:
            when = src._parse_utc(entry.get(name))
            if when is not None:
                out.append((int(key), when))
                break
    return tuple(sorted(out))


def _live_reviewers(reviews_dir: Path, prs: Any) -> int:
    """Sidecar-alive reviewers plus state.json-dispatched ghosts with a live pid.

    Same corroboration as ``dispatch_selection._count_live_reviews``: a missing sidecar
    must not make a live reviewer invisible.
    """
    live_prs: set[int] = set()
    for worker in iter_workers(reviews_dir):
        if worker.is_alive():
            live_prs.add(worker.issue_number)
    count = len(live_prs)
    for key, entry in (prs if isinstance(prs, dict) else {}).items():
        if not isinstance(entry, dict) or entry.get("review_dispatch_status") != _DISPATCHED:
            continue
        if not str(key).isdigit() or int(key) in live_prs:
            continue
        pid = entry.get("reviewer_pid")
        if pid is not None and _host.current().probe.is_alive(
            pid, entry.get("reviewer_process_start_time")
        ):
            count += 1
    return count


def read_repo_state(state_dir: Path, reviews_dir_override: str = "") -> RepoStateRead:
    raw = src.read_json_file(layout.state_file_path(state_dir))
    if raw.data is None:
        return RepoStateRead((), None)
    reviews_dir = (
        Path(reviews_dir_override)
        if reviews_dir_override
        else layout.reviews_dir_default(state_dir)
    )
    return RepoStateRead(
        _escalated_since(raw.data.get("issues")),
        _live_reviewers(reviews_dir, raw.data.get("prs")),
    )
