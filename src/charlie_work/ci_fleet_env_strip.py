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

**A failure costs only the names it hides.** ``ci_fleet`` treats a raising source
as "strip nothing", so a bad repo entry or a bad registry must not take the
readable names down with it. Every skip logs a warning, because a skipped name
stays in listener environments. The log line gives the exception type, not its
message, because a config parse error can quote the file. Only names pass through
this module, never values.

Imports from this package sit inside the functions because ``instrumentation``
installs the source while it is still being imported, and these modules import
``instrumentation``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import OrchestratorConfig

logger = logging.getLogger(__name__)


def _names_from(label: object, load: Callable[[], OrchestratorConfig]) -> tuple[str, ...]:
    from .env_sanitize import provider_api_key_env_vars

    try:
        return provider_api_key_env_vars(load())
    except Exception as exc:  # see "A failure costs only" in the module docstring
        logger.warning(
            "env strip: skipped %s, its provider key names stay in listener environments: %s",
            label,
            type(exc).__name__,
        )
        return ()


def _registered_repos(registry_path: Path) -> dict[str, Any]:
    """Return the registry's repo entries, or none if it can't be read.

    ``_load_registry`` turns a corrupt file into an empty registry and says
    nothing, so an empty result from a file that exists gets its own warning.
    """
    from .fleet_registry import _load_registry

    try:
        repos = dict(_load_registry(registry_path)["repos"])
        if not repos and registry_path.exists():
            logger.warning(
                "env strip: %s lists no repos, so repo-level provider key names "
                "stay in listener environments",
                registry_path,
            )
        return repos
    except Exception as exc:  # see "A failure costs only" in the module docstring
        logger.warning(
            "env strip: skipped %s, repo-level provider key names stay in listener "
            "environments: %s",
            registry_path,
            type(exc).__name__,
        )
        return {}


def _load_repo_config(entry: dict[str, Any], fleet_dir_override: str | None) -> OrchestratorConfig:
    from .global_config import load_layered_config

    root = Path(entry["repo_root"])
    explicit = Path(entry["config_path"]) if entry.get("config_path") else None
    # A missing path loads as plain defaults, which would look like a repo that
    # configures no keys rather than one whose names couldn't be read.
    for path in (root, explicit):
        if path is not None and not path.exists():
            raise FileNotFoundError(path)
    return load_layered_config(root, explicit, fleet_dir_override=fleet_dir_override)


def fleet_provider_key_names(fleet_dir_override: str | None = None) -> frozenset[str]:
    """Return the provider-key variable names configured across the fleet.

    ``ci_fleet`` calls this with no arguments, so ``layout`` picks the fleet dir
    and ``CHARLIE_WORK_FLEET_DIR`` is honoured. A ``--fleet-dir`` flag does not
    reach this call: a supervisor started with one still reads the default
    fleet's configs. ``fleet_dir_override`` is for tests.
    """
    from . import layout
    from .config import load_config

    global_path = layout.global_config_path(override=fleet_dir_override)
    names = set(_names_from(global_path, partial(load_config, global_path)))

    registry_path = layout.fleet_registry_path(override=fleet_dir_override)
    for name, entry in _registered_repos(registry_path).items():
        names.update(_names_from(name, partial(_load_repo_config, entry, fleet_dir_override)))
    return frozenset(names)
