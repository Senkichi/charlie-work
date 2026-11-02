"""The host half of ``ci_fleet``'s secret-name strip (issue #2902).

The allocation pass runs inside the supervisor, so a runner listener starts with
a copy of the supervisor's environment, and every self-hosted CI job step
inherits that copy. The supervisor holds provider API keys under configured names
(``api_worker.providers[*].api_key_env``) because the ``api`` adapter reads them
from there. ``ci_fleet`` can't know those names: they live in charlie-work config
and the dependency points one way. So ``ci_fleet.env_strip`` takes a source, and
``instrumentation.py`` installs this one beside the other seams.

**Every registered repo, plus the global layer.** One supervisor serves every
repo and one environment reaches every listener, so a name any single repo
configures has to go. The global layer is read on its own as well, so the
fleet-wide names survive a missing or unreadable registry. The registry is read
directly because ``managed_repo_roots`` collapses two repos that share a name.

**Called per launch, never cached.** ``ci_fleet`` asks again for each listener,
so a config change needs no restart here. A running listener keeps the
environment it started with, though, so a key already copied into one stays there
until that listener restarts.

**One bad config costs only its own names.** ``ci_fleet`` treats a raising source
as "strip nothing", so a failure must not escape from here. The log line gives
the exception type, not its message, because a config parse error can quote the
file. Only names pass through this module, never values.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


def fleet_provider_key_names(fleet_dir_override: str | None = None) -> frozenset[str]:
    """Return the provider-key variable names configured across the fleet.

    ``ci_fleet`` calls this with no arguments. ``fleet_dir_override`` exists for
    tests and for a fleet run outside the default directory.
    """
    # Imported here because ``instrumentation`` installs this function while it
    # is still being imported, and these modules import ``instrumentation``.
    from . import layout
    from .config import load_config
    from .env_sanitize import provider_api_key_env_vars
    from .fleet_registry import _load_registry
    from .global_config import load_layered_config

    def names_from(label: object, load: Callable[[], OrchestratorConfig]) -> tuple[str, ...]:
        try:
            return provider_api_key_env_vars(load())
        except Exception as exc:  # see "One bad config" in the module docstring
            logger.warning(
                "env strip: skipped %s, its provider key names stay in listener environments: %s",
                label,
                type(exc).__name__,
            )
            return ()

    global_path = layout.global_config_path(override=fleet_dir_override)
    names = set(names_from(global_path, partial(load_config, global_path)))

    registry = _load_registry(layout.fleet_registry_path(override=fleet_dir_override))
    for entry in registry.get("repos", {}).values():
        root = entry.get("repo_root") if isinstance(entry, dict) else None
        if not root:
            continue
        load = partial(load_layered_config, Path(root), fleet_dir_override=fleet_dir_override)
        names.update(names_from(root, load))
    return frozenset(names)
