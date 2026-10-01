"""Role chain (model waterfall) config: ``worker.fallbacks`` / ``reviewer.fallbacks``.

Issue #2086. A role (``worker:`` or ``reviewer:``) names a primary
``(harness, model)`` and, optionally, up to :data:`MAX_FALLBACKS` ordered
fallback entries. The **role chain** is the primary followed by its fallbacks;
at launch time :func:`charlie_work.role_selection.select_role_entry` picks the
first entry whose ``(harness, model)`` is not restricted in the fleet-scoped
quota ledger (:mod:`charlie_work.role_quota_ledger`).

This module owns the config half, expressed in the section-validation
vocabulary (ADR-0007) rather than a parser of its own:

* :func:`role_entries` is the use-site marker that validates the ``fallbacks:``
  list as it is built -- list shape, at most :data:`MAX_FALLBACKS` entries,
  every entry checked like the primary (harness membership for the role, string
  types, unknown keys), and ``effort`` accepted only on a reviewer entry.
* :func:`check_role_chain` / :func:`check_reviewer_role_chain` are the
  ``Check`` hooks for the rules that need the primary: no synchronous harness in
  a chain with fallbacks, and no duplicate ``(harness, model)`` pair across the
  whole chain. The reviewer hook also runs the warn-only cross-family guard.
* :func:`chain_of`, the ``chain`` accessor both role dataclasses expose.

An empty ``fallbacks`` list (the default) is a chain of length 1, which
behaves exactly as a role did before this module existed.
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import Annotated, Any

from .config_validation import Entries, FieldError, NotNull, NullIsDefault, OneOf, Typed

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
    only; empty means the harness default). Worker entries never carry one --
    the worker role's :func:`role_entries` marker forbids the key.
    ``harness`` membership is per role, so it is added at the use site.
    """

    harness: Annotated[str, Typed]
    model: Annotated[str, Typed, NullIsDefault] = ""
    effort: Annotated[str, Typed, NullIsDefault] = ""

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


def role_entries(harnesses: Collection[str], *, allow_effort: bool) -> Entries:
    """The ``fallbacks`` marker for one role: harness set, length cap, effort policy."""
    return Entries(
        max_len=MAX_FALLBACKS,
        forbid=() if allow_effort else ("effort",),
        harness=(Typed, NotNull, OneOf(*sorted(harnesses))),
    )


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def check_role_chain(role: Any, config: Any = None) -> None:  # noqa: ARG001 (Check signature)
    """Section hook: the chain-wide rules that need the primary. A chain with no
    fallbacks is never checked, so a role without them behaves as before #2086."""
    entries = tuple(role.fallbacks or ())
    if not entries:
        return
    chain = (RoleEntry(_text(role.harness), _text(role.model)), *entries)
    for index, entry in enumerate(chain):
        if entry.harness in _NON_SESSION_HARNESSES:
            raise FieldError(
                "harness" if index == 0 else f"fallbacks[{index - 1}].harness",
                "a harness that records sessions in a role chain with fallbacks "
                f"(not {', '.join(sorted(_NON_SESSION_HARNESSES))}: a quota "
                "restriction cannot be recorded against them)",
                entry.harness,
            )
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(chain):
        if entry.key in seen:
            raise FieldError(
                f"fallbacks[{index - 1}]",
                "a (harness, model) pair not already in the role chain",
                entry.key,
            )
        seen.add(entry.key)


def check_reviewer_role_chain(reviewer: Any, config: Any) -> None:
    """Reviewer hook: the chain rules, then the warn-only worker/reviewer family guard."""
    check_role_chain(reviewer)
    warn_same_family(config.worker, reviewer)


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
