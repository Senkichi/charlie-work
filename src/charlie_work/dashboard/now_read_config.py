"""Per-repo layered config reads for the Now collector (read-only; errors are values).

Caps must come from ``global_config.load_layered_config`` -- the loader the fleet itself
uses -- never from a single YAML file: the per-repo ``orchestrator.config.yaml`` overrides
the global ``<fleet_dir>/config.yaml`` key by key, so any single-source read is wrong.
A repo whose config cannot be loaded yields ``None`` (unknown), never a default cap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .. import global_config
from ..config_validation import ConfigError


@dataclass(frozen=True)
class RepoConfigRead:
    """The slice of one repo's merged config the Now view needs (0 = unlimited)."""

    worker_cap: int
    review_cap: int
    reviews_dir: str  # "" = derive from the state dir
    global_worker_cap: int  # fleet.global_max_concurrent_sessions
    global_review_cap: int  # fleet.global_max_concurrent_reviews
    managed_root: str  # runner_allocation.managed_root or runner_scaling.managed_root


def read_repo_config(repo_root: Path, fleet_dir_override: str | None) -> RepoConfigRead | None:
    try:
        cfg = global_config.load_layered_config(repo_root, fleet_dir_override=fleet_dir_override)
    except (ConfigError, OSError, ValueError, yaml.YAMLError):
        return None
    return RepoConfigRead(
        worker_cap=cfg.dispatch.max_concurrent_sessions,
        review_cap=cfg.review_dispatch.max_concurrent_reviews,
        reviews_dir=cfg.review_dispatch.reviews_dir,
        global_worker_cap=cfg.fleet.global_max_concurrent_sessions,
        global_review_cap=cfg.fleet.global_max_concurrent_reviews,
        managed_root=cfg.runner_allocation.managed_root or cfg.runner_scaling.managed_root,
    )
