"""Single source of truth for which harnesses the worker/reviewer roles accept.

Issue #1513: before this module existed, ``config.py`` hand-maintained a
worker-harness allowlist (``_VALID_WORKER_HARNESSES``) and separately
hard-pinned the reviewer role to ``"claude-code"`` only, while
``adapters.dispatch_sessions`` kept its own, textually separate if/elif
chain of the harnesses it actually knew how to launch. Nothing enforced that
the three stayed in sync -- a harness could be "valid" per config yet
unreachable from dispatch, or vice versa.

This module is now the only place harness support is declared. Both
``config.py`` (validation, for both roles) and ``adapters.py`` (the worker
dispatch table, with a drift-guard assertion) import their allowed harness
sets from here rather than maintaining their own copies. Adding a harness --
or adding review support to an existing worker-only harness -- means editing
one entry in ``HARNESS_REGISTRY``; nothing else needs to change to stay in
sync.

Naming note: the harness name used in config (``worker.harness`` /
``reviewer.harness``, e.g. ``"devin-shell"``) is not always identical to the
``WorkerView.adapter_kind`` value carried by that harness's session sidecars
(e.g. ``"devin"``) -- ``adapter_kind`` is the field that lets code enumerate
live sessions with ``worker.iter_workers()`` without caring which harness
launched them. ``HarnessCapabilities.adapter_kind`` records that mapping
explicitly, once, here -- so no caller needs a second hardcoded translation
table between the two names.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarnessCapabilities:
    """What one harness can be used for, and how its sessions are identified.

    ``worker``: this harness may be selected as ``worker.harness`` (or
    ``rescue.worker.harness``) and dispatched by ``adapters.dispatch_sessions``.

    ``review``: this harness may be selected as ``reviewer.harness`` (or
    ``rescue.reviewer.harness``) and its launch is wired into
    ``Workflow.dispatch_reviews``.

    ``adapter_kind``: the value this harness's session records carry in
    ``WorkerView.adapter_kind`` (see ``worker.py``). Harnesses that never
    produce a ``WorkerView``-tracked session (synchronous, no sidecar --
    ``"manual"``, ``"command"``) carry their own harness name here as a
    harmless placeholder; nothing filters on it for those.
    """

    worker: bool
    review: bool
    adapter_kind: str


HARNESS_REGISTRY: dict[str, HarnessCapabilities] = {
    "claude-code": HarnessCapabilities(worker=True, review=True, adapter_kind="claude-code"),
    "devin-shell": HarnessCapabilities(worker=True, review=True, adapter_kind="devin"),
    "api": HarnessCapabilities(worker=True, review=True, adapter_kind="api"),
    "command": HarnessCapabilities(worker=True, review=False, adapter_kind="command"),
    "manual": HarnessCapabilities(worker=True, review=False, adapter_kind="manual"),
}

# Harnesses valid for `worker.harness` / `rescue.worker.harness`.
WORKER_HARNESSES: frozenset[str] = frozenset(
    name for name, cap in HARNESS_REGISTRY.items() if cap.worker
)

# Harnesses valid for `reviewer.harness` / `rescue.reviewer.harness`.
REVIEWER_HARNESSES: frozenset[str] = frozenset(
    name for name, cap in HARNESS_REGISTRY.items() if cap.review
)

# WorkerView.adapter_kind values produced by review-capable harnesses. Used to
# filter `iter_workers(reviews_dir)` results without re-deriving the
# harness-name-to-adapter_kind mapping at each call site.
REVIEWER_ADAPTER_KINDS: frozenset[str] = frozenset(
    cap.adapter_kind for cap in HARNESS_REGISTRY.values() if cap.review
)
