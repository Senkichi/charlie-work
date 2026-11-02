"""``fleet.max_concurrent_sessions_by_harness``: per-harness live-session ceilings (issue #2757).

``fleet.global_max_concurrent_sessions`` counts every live worker session, whatever
its harness. But the quota it was sized for -- a shared Devin account's rate limit --
binds only the sessions that draw on that account, so a free-model worker (say an
``opencode`` entry) occupied a Devin-ceiling slot it did not spend. This map scopes a
ceiling to the harness it protects::

    fleet:
      global_max_concurrent_sessions: 10   # host-level total, all harnesses
      max_concurrent_sessions_by_harness:
        devin-shell: 3                     # the shared-account ceiling

Keys are harness names (``harnesses.HARNESS_REGISTRY``), values non-negative ints
(``0`` blocks the harness outright); a harness not named has no ceiling of its own and
is bounded only by the total. Absent or empty -- the default -- changes nothing.

Where each half lives:

* the fleet-wide live count per harness is measured by the concurrency governor
  (``reap_dispatch._apply_concurrency_governor``) from sidecar ``adapter_kind`` mapped
  through the registry -- no harness list is hardcoded anywhere;
* the cap is enforced after role-chain selection, because only then is the launch's
  harness known: a selected entry on a full harness falls through to the next entry
  with room, else the launch is clamped to zero with ``clamped_by: harness_max``
  (``worker_launch_gate._apply_harness_caps``).

Only this module validates and freezes the map, so a parsed ``FleetConfig`` and a
hand-built one are held to the same rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from .config_validation import FieldError
from .harnesses import HARNESS_REGISTRY

KEY = "max_concurrent_sessions_by_harness"


def freeze_harness_caps(raw: object) -> Mapping[str, int]:
    """Validate ``raw`` and return it as an immutable ``{harness: cap}`` view.

    Raises :class:`FieldError` (keyed ``max_concurrent_sessions_by_harness``; the
    section parser prefixes ``fleet.``) for a non-mapping, an unregistered harness
    name, or a cap that is not a non-negative int (a bool is not an int here).
    """
    if not isinstance(raw, Mapping):
        raise FieldError(KEY, "mapping of harness name to non-negative int", raw, "type")
    caps: dict[str, int] = {}
    for harness, cap in raw.items():
        if not isinstance(harness, str) or harness not in HARNESS_REGISTRY:
            raise FieldError(
                KEY,
                f"a registered harness name (valid: {', '.join(sorted(HARNESS_REGISTRY))})",
                harness,
            )
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise FieldError(f"{KEY}.{harness}", "non-negative int", cap, "type")
        if cap < 0:
            raise FieldError(f"{KEY}.{harness}", ">= 0", cap)
        caps[harness] = cap
    return MappingProxyType(caps)
