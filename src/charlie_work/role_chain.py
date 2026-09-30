"""Role chain (model waterfall) config: ``worker.fallbacks`` / ``reviewer.fallbacks``.

Issue #2086. A role (``worker:`` or ``reviewer:``) names a primary
``(harness, model)`` and, optionally, up to :data:`MAX_FALLBACKS` ordered
fallback entries. The **role chain** is the primary followed by its fallbacks;
at launch time :func:`charlie_work.role_selection.select_role_entry` picks the
first entry whose ``(harness, model)`` is not restricted in the fleet-scoped
quota ledger (:mod:`charlie_work.role_quota_ledger`).

This module owns the config half: parsing and validating the ``fallbacks:``
list (every entry is checked exactly like the primary -- harness membership,
string types -- plus no duplicate ``(harness, model)`` pairs across the whole
chain), the ``chain`` accessor both role dataclasses expose, and the warn-only
cross-family guard. It lives outside ``config.py`` because of the file-size
ratchet; ``config.py`` calls :func:`normalize_role_section` at the two role
build sites.

An empty ``fallbacks`` list (the default) is a chain of length 1, which
behaves exactly as a role did before this module existed.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Mapping
from dataclasses import MISSING, dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)

# At most 3 fallbacks, so a chain holds at most 4 entries including the primary.
MAX_FALLBACKS = 3

# Synchronous harnesses never produce a session sidecar, so no quota or
# rate-limit death can ever be recorded against them -- a chain containing
# one could never move past (or back to) it. Rejected in any chain that has
# fallbacks, as primary or as fallback.
_NON_SESSION_HARNESSES = frozenset({"manual", "command"})


@dataclass(frozen=True)
class RoleEntry:
    """One ``(harness, model)`` entry of a role chain.

    ``effort`` is the reviewer's ``--effort`` pin for this entry (claude-code
    only; empty means the harness default). Worker entries never carry one.
    """

    harness: str
    model: str = ""
    effort: str = ""

    @property
    def key(self) -> tuple[str, str]:
        """The quota-ledger key: a provider restriction is per (harness, model)."""
        return (self.harness, self.model)

    def to_payload(self) -> dict[str, str]:
        return {"harness": self.harness, "model": self.model}


def chain_of(role: Any) -> tuple[RoleEntry, ...]:
    """The role chain for a ``WorkerRoleConfig`` / ``ReviewerRoleConfig``.

    Primary first, then ``role.fallbacks`` in configured order. Bound as the
    ``chain`` property on both role dataclasses.
    """
    primary = RoleEntry(role.harness, role.model, str(getattr(role, "effort", "") or ""))
    return (primary, *tuple(getattr(role, "fallbacks", ()) or ()))


def _config_error(message: str) -> Exception:
    # Lazy: config.py imports this module at load time.
    from .config import ConfigError

    return ConfigError(message)


def _field_default(cls: type, name: str) -> Any:
    for item in fields(cls):
        if item.name == name:
            if item.default is not MISSING:
                return item.default
            return None
    return None


def _parse_entry(
    section: str, index: int, raw: Any, *, harnesses: Collection[str], allow_effort: bool
) -> RoleEntry:
    where = f"config section '{section}' key 'fallbacks[{index}]'"
    if not isinstance(raw, Mapping):
        raise _config_error(f"{where} must be a mapping, got {type(raw).__name__}")
    allowed = {"harness", "model", "effort"} if allow_effort else {"harness", "model"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _config_error(
            f"unknown key(s) in {where}: {', '.join(map(str, unknown))} "
            f"(valid: {', '.join(sorted(allowed))})"
        )
    harness = raw.get("harness")
    if not isinstance(harness, str) or harness not in harnesses:
        raise _config_error(
            f"{where} key 'harness' must be one of {sorted(harnesses)}, got {harness!r}"
        )
    values: dict[str, str] = {}
    for key in sorted(allowed - {"harness"}):
        value = raw.get(key, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise _config_error(
                f"{where} key '{key}' must be a string, got {type(value).__name__}"
            )
        values[key] = value
    return RoleEntry(harness=harness, **values)


def parse_fallbacks(
    section: str,
    raw: Any,
    primary: RoleEntry,
    *,
    harnesses: Collection[str],
    allow_effort: bool,
) -> tuple[RoleEntry, ...]:
    """Validate a raw ``fallbacks:`` value and return it as a tuple of entries.

    Raises ``ConfigError`` for a non-list, more than :data:`MAX_FALLBACKS`
    entries, a bad entry (see :func:`_parse_entry`), a synchronous harness
    anywhere in a chain that has fallbacks, or a ``(harness, model)`` pair
    that appears twice in the chain (primary included).
    """
    if raw is None:
        return ()
    if isinstance(raw, tuple) and all(isinstance(item, RoleEntry) for item in raw):
        entries = raw  # already parsed (e.g. a config round-trip)
    else:
        if not isinstance(raw, list | tuple):
            raise _config_error(
                f"config section '{section}' key 'fallbacks' must be a list, "
                f"got {type(raw).__name__}"
            )
        entries = tuple(
            item
            if isinstance(item, RoleEntry)
            else _parse_entry(section, index, item, harnesses=harnesses, allow_effort=allow_effort)
            for index, item in enumerate(raw)
        )
    if len(entries) > MAX_FALLBACKS:
        raise _config_error(
            f"config section '{section}' key 'fallbacks' allows at most {MAX_FALLBACKS} "
            f"entries, got {len(entries)}"
        )
    if not entries:
        return ()
    chain = (primary, *entries)
    for entry in chain:
        if entry.harness in _NON_SESSION_HARNESSES:
            raise _config_error(
                f"config section '{section}': harness {entry.harness!r} cannot be part of a "
                "role chain with fallbacks (it produces no session a quota restriction "
                "could be recorded against)"
            )
    seen: set[tuple[str, str]] = set()
    for entry in chain:
        if entry.key in seen:
            raise _config_error(
                f"config section '{section}' key 'fallbacks': duplicate (harness, model) "
                f"pair {entry.key!r} in the role chain"
            )
        seen.add(entry.key)
    return entries


def normalize_role_section(
    cls: type, section: str, data: Mapping[str, Any], harnesses: Collection[str]
) -> dict[str, Any]:
    """Return ``data`` with its ``fallbacks`` list parsed into ``RoleEntry`` tuples.

    Called by ``config.build_config_from_data`` on the raw ``worker:`` /
    ``reviewer:`` section before ``_build_section``. The primary is resolved
    from ``data`` with ``cls``'s own field defaults, so the duplicate check
    sees the same primary the dataclass will. A section without
    ``fallbacks`` is returned unchanged (as a copy).
    """
    result = dict(data)
    if "fallbacks" not in result:
        return result
    allow_effort = any(item.name == "effort" for item in fields(cls))

    def _primary(name: str) -> str:
        value = result.get(name, _field_default(cls, name))
        return value if isinstance(value, str) else ""

    primary = RoleEntry(_primary("harness"), _primary("model"), _primary("effort"))
    result["fallbacks"] = parse_fallbacks(
        section,
        result["fallbacks"],
        primary,
        harnesses=harnesses,
        allow_effort=allow_effort,
    )
    return result


# --- warn-only cross-family guard -------------------------------------------

# Model-name prefixes -> vendor family. Deliberately partial: an unknown model
# yields no family and therefore no warning (this guard only warns; it must
# never reject a config it cannot classify).
_FAMILY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("opus", "anthropic"),
    ("sonnet", "anthropic"),
    ("haiku", "anthropic"),
    ("swe-", "cognition"),
    ("gpt", "openai"),
    ("codex", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "google"),
    ("kimi", "moonshot"),
    ("glm", "zhipu"),
    ("qwen", "alibaba"),
    ("deepseek", "deepseek"),
)


def model_family(entry: RoleEntry) -> str | None:
    """Best-effort vendor family for an entry's model; ``None`` when unknown.

    An empty claude-code model resolves to the harness's pinned Claude default,
    so it is ``anthropic``; any other empty model is unknown.
    """
    model = entry.model.strip().lower()
    if not model:
        return "anthropic" if entry.harness == "claude-code" else None
    for prefix, family in _FAMILY_PREFIXES:
        if model.startswith(prefix):
            return family
    return None


def same_family_pairs(
    worker_chain: tuple[RoleEntry, ...], reviewer_chain: tuple[RoleEntry, ...]
) -> list[tuple[int, int, str]]:
    """``(worker_index, reviewer_index, family)`` for every same-family pair
    that involves at least one fallback entry.

    The primary/primary pair is excluded: ``charlie doctor``'s cross-family
    row already reports it, and this guard exists for the case the issue
    names -- a *fallback* silently putting worker and reviewer on one family.
    """
    pairs: list[tuple[int, int, str]] = []
    for w_index, w_entry in enumerate(worker_chain):
        w_family = model_family(w_entry)
        if w_family is None:
            continue
        for r_index, r_entry in enumerate(reviewer_chain):
            if w_index == 0 and r_index == 0:
                continue
            if model_family(r_entry) == w_family:
                pairs.append((w_index, r_index, w_family))
    return pairs


_WARNED: set[str] = set()


def warn_same_family(worker: Any, reviewer: Any) -> None:
    """Log (once per process per message) each same-family chain pair. Never raises."""
    worker_chain, reviewer_chain = chain_of(worker), chain_of(reviewer)
    for w_index, r_index, family in same_family_pairs(worker_chain, reviewer_chain):
        w_entry, r_entry = worker_chain[w_index], reviewer_chain[r_index]
        message = (
            f"role chain: worker entry {w_index} ({w_entry.harness}/{w_entry.model or '-'}) "
            f"and reviewer entry {r_index} ({r_entry.harness}/{r_entry.model or '-'}) are "
            f"both {family}-family models; worker and reviewer should differ in family"
        )
        if message not in _WARNED:
            _WARNED.add(message)
            logger.warning(message)
