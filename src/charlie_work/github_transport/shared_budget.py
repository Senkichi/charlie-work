"""The observed GitHub budget, shared across processes (issue #2442).

A rate-limit budget belongs to a *token*, not to a client: the fleet builds a
fresh client per repo per pass, lanes run on pool threads, and CLI commands
(``charlie wait-pr``, mop-up, doctor) are separate processes. Each client only
saw its own responses, so none of them knew the fleet had already drained the
hour. ``SharedBudgetFile`` is the small JSON snapshot they all publish to and
read from, in the fleet-level state directory.

Contract:

* **Keyed by a token fingerprint** (``token_key``), never the token itself.
* **Newest wins**, by the same server-side numbers ``observe_github_rate``
  uses (``window_supersedes``): publishing merges into what is on disk, so a
  late writer cannot move a window backwards.
* **A snapshot from a past window is ignored**: a window whose ``reset`` has
  passed is dropped on read and pruned on write.
* **Best effort, fail open.** It is an optimisation over the per-process
  budget. Every I/O error degrades to "no shared knowledge"; it never raises
  into a request. Concurrent publishers are not locked: the worst case is one
  lost observation, corrected by the next response.
* Writes go through ``write_json_atomic`` and are throttled (a response burst
  would otherwise be one file write per request); a new window, or an
  exhausted one, always publishes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..api_budget import GitHubRateBudget, GitHubRateWindow, window_supersedes
from ..atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_TOKEN_KEY = "default"
_READ_TTL_SECONDS = 2.0
_PUBLISH_INTERVAL_SECONDS = 1.0


def token_fingerprint(token: str | None) -> str:
    """A stable, non-reversible key for *token* (``default`` when unknown)."""
    if not token:
        return DEFAULT_TOKEN_KEY
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def merge_github_rate(budget: GitHubRateBudget, other: GitHubRateBudget) -> GitHubRateBudget:
    """*budget* with each window of *other* applied under newest-wins."""
    merged = {w.resource: w for w in budget.windows}
    for window in other.windows:
        current = merged.get(window.resource)
        if current is None or window_supersedes(window, current):
            merged[window.resource] = window
    return GitHubRateBudget(windows=tuple(merged.values()))


def _live(budget: GitHubRateBudget, now: float) -> GitHubRateBudget:
    return GitHubRateBudget(tuple(w for w in budget.windows if w.reset_epoch > now))


def _window_from_dict(raw: Any) -> GitHubRateWindow | None:
    try:
        return GitHubRateWindow(
            str(raw["resource"]),
            int(raw["limit"]),
            int(raw["remaining"]),
            int(raw["reset"]),
            float(raw["observed"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _window_to_dict(window: GitHubRateWindow) -> dict[str, Any]:
    return {
        "resource": window.resource,
        "limit": window.limit,
        "remaining": window.remaining,
        "reset": window.reset_epoch,
        "observed": window.observed_epoch,
    }


class SharedBudgetFile:
    """Read and publish the shared budget snapshot at *path*."""

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], float] = time.time,
        read_ttl: float = _READ_TTL_SECONDS,
        publish_interval: float = _PUBLISH_INTERVAL_SECONDS,
    ) -> None:
        self.path = path
        self._now = now
        self._read_ttl = read_ttl
        self._publish_interval = publish_interval
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, GitHubRateBudget]] = {}
        self._last_publish: dict[str, tuple[float, dict[str, int]]] = {}

    def _load_all(self) -> dict[str, GitHubRateBudget]:
        """Every token's live windows; ``{}`` for a missing or unreadable file."""
        now = self._now()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            tokens = data["tokens"] if data.get("version") == SCHEMA_VERSION else {}
            out: dict[str, GitHubRateBudget] = {}
            for key, entry in tokens.items():
                windows = tuple(
                    w for w in map(_window_from_dict, entry.get("windows", ())) if w is not None
                )
                live = _live(GitHubRateBudget(windows), now)
                if live.windows:
                    out[str(key)] = live
            return out
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            logger.debug(
                "shared GitHub budget %s unreadable; ignoring it", self.path, exc_info=True
            )
            return {}

    def read(self, token_key: str) -> GitHubRateBudget:
        """The live shared windows for *token_key* (cached for a couple of seconds)."""
        now = self._now()
        with self._lock:
            cached = self._cache.get(token_key)
            if cached is not None and now - cached[0] < self._read_ttl:
                return _live(cached[1], now)
        budget = self._load_all().get(token_key, GitHubRateBudget())
        with self._lock:
            self._cache[token_key] = (now, budget)
        return budget

    def publish(self, token_key: str, budget: GitHubRateBudget) -> bool:
        """Merge *budget* into the file; ``False`` when throttled or the write failed."""
        now = self._now()
        marks = {w.resource: w.reset_epoch for w in budget.windows}
        exhausted = any(w.remaining <= 0 for w in budget.windows)
        with self._lock:
            last = self._last_publish.get(token_key)
            if (
                last is not None
                and not exhausted
                and last[1] == marks
                and now - last[0] < self._publish_interval
            ):
                return False
            self._last_publish[token_key] = (now, marks)
        try:
            everything = self._load_all()
            merged = merge_github_rate(everything.get(token_key, GitHubRateBudget()), budget)
            everything[token_key] = _live(merged, now)
            document = {
                "version": SCHEMA_VERSION,
                "tokens": {
                    key: {"windows": [_window_to_dict(w) for w in b.windows]}
                    for key, b in everything.items()
                    if b.windows
                },
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.path, document)
        except Exception:  # noqa: BLE001 - the shared snapshot must never break a request
            logger.debug("failed to publish shared GitHub budget %s", self.path, exc_info=True)
            return False
        with self._lock:
            self._cache[token_key] = (now, everything.get(token_key, GitHubRateBudget()))
        return True
