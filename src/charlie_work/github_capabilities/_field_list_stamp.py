"""Remember that the field lists passed schema validation (issue #2438).

``Transport.validate_field_lists`` costs ~8 GraphQL points and used to run on
every ``OrchestratorApp`` construction -- every lane pass, reap sweep and CLI
command -- about 30% of the hourly GraphQL budget. The answer only changes when
the compiled field-list constants change (a code deploy) or GitHub's schema
changes (rare), so a pass is remembered:

* in-process, keyed by ``(state dir, repo slug)``; and
* on disk, as ``field-list-probe.json`` next to ``state.json``, keyed by a
  digest of the field-list constants plus the repo slug, valid for 24 hours.

Only a fully clean validation (every probe accepted, none skipped or
inconclusive) is stamped. The cache is trusted, not proven, so a schema
rejection on a *real* call (``_send.send_read``) clears the stamp and raises
``ConfigError`` -- the same outcome the probes would have produced. Set
``CHARLIE_FIELD_LIST_PROBE_CACHE=off`` to always probe.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..atomic_write import write_json_atomic

logger = logging.getLogger(__name__)

STAMP_FILENAME = "field-list-probe.json"
STAMP_TTL_SECONDS = 24 * 60 * 60
KILL_SWITCH_ENV = "CHARLIE_FIELD_LIST_PROBE_CACHE"

# (state dir, "owner/repo") -> digest validated in this process.
_VALIDATED: dict[tuple[str, str], str] = {}


def enabled() -> bool:
    return os.environ.get(KILL_SWITCH_ENV, "").strip().lower() not in {"off", "0", "false"}


def digest_of(probes: list[tuple[str, str, Any]], slug: str) -> str:
    """Stable digest of every ``(constant, fields)`` pair plus the repo slug."""
    material = json.dumps([[c, f] for c, f, _ in probes] + [slug])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _key(state_dir: Path, slug: str) -> tuple[str, str]:
    return (str(state_dir), slug)


def is_fresh(state_dir: Path, slug: str, digest: str, *, now: float | None = None) -> bool:
    """True iff this exact digest was validated in-process or stamped < 24h ago."""
    if not enabled():
        return False
    key = _key(state_dir, slug)
    if _VALIDATED.get(key) == digest:
        return True
    try:
        stamp = json.loads((state_dir / STAMP_FILENAME).read_text(encoding="utf-8"))
        validated_at = float(stamp["validated_at"])
        matches = stamp["digest"] == digest and stamp["repo"] == slug
    except (OSError, ValueError, KeyError, TypeError):
        return False
    age = (time.time() if now is None else now) - validated_at
    if matches and 0 <= age < STAMP_TTL_SECONDS:
        _VALIDATED[key] = digest
        return True
    return False


def record(state_dir: Path, slug: str, digest: str, *, now: float | None = None) -> None:
    """Remember a clean validation (process + disk). Disk failure is non-fatal."""
    if not enabled():
        return
    _VALIDATED[_key(state_dir, slug)] = digest
    stamp = {
        "digest": digest,
        "repo": slug,
        "validated_at": time.time() if now is None else now,
    }
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(state_dir / STAMP_FILENAME, stamp)
    except OSError as exc:
        logger.warning("Could not write field-list probe stamp: %s", exc)


def clear(state_dir: Path, slug: str) -> None:
    _VALIDATED.pop(_key(state_dir, slug), None)
    try:
        (state_dir / STAMP_FILENAME).unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove field-list probe stamp: %s", exc)


def check_real_call(collab: Any, owner: str, name: str, outcome: Any) -> None:
    """Raise ``ConfigError`` if a real call's schema rejection contradicts a trusted pass.

    A no-op unless this process is relying on a cached validation for the repo,
    so callers that never validated keep their errors-as-values behavior.
    """
    from ..config import ConfigError
    from ._field_probes import probe_verdict
    from .circuit_breaker_transport import circuit_breaker_state_path

    slug = f"{owner}/{name}"
    state_dir = circuit_breaker_state_path(collab.runtime, collab.repo_root).parent
    if _key(state_dir, slug) not in _VALIDATED:
        return
    rejected = probe_verdict(outcome).rejected
    if rejected is None:
        return
    clear(state_dir, slug)
    raise ConfigError(
        "GitHub rejected a field the cached schema validation had accepted "
        f"({', '.join(rejected)}); cleared the field-list probe stamp"
    )
