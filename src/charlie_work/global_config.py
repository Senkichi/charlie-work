from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import (
    OrchestratorConfig,
    find_config_path,
    load_config,
)
from .fleet_paths import fleet_dir


def _deep_merge(base: Any, override: Any) -> Any:
    """Recursively merge two dicts; non-dict overrides win.

    This keeps mapping-valued defaults from the global layer when a per-repo
    config only overrides a subset (e.g. ``api_worker.budget.max_usd_per_session``
    without redeclaring the other caps, or ``api_worker.providers`` additions).
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged.get(key), value)
        return merged
    return override


def load_layered_config(
    repo_root: Path,
    explicit: Path | None = None,
    *,
    fleet_dir_override: str | None = None,
) -> OrchestratorConfig:
    """Load config with a global fleet layer and per-repo override.

    Global config (if present) at <fleet_dir>/config.yaml supplies fleet-wide
    defaults. The per-repo orchestrator.config.yaml (resolved via
    find_config_path) wins on any key present in both. Absent global file ->
    no-op (identical to today's per-repo-only behavior).

    The merge happens at the raw YAML dict level before validation, so unknown
    keys in the global file raise ConfigError exactly like unknown keys in the
    per-repo file.

    Args:
        repo_root: The repository root path.
        explicit: Optional explicit path to the per-repo config file.
        fleet_dir_override: Optional override for the fleet directory path.

    Returns:
        The merged OrchestratorConfig.
    """
    # Load per-repo config path
    repo_config_path = find_config_path(repo_root, explicit)

    # Load global config if present
    global_config_path = fleet_dir(override=fleet_dir_override) / "config.yaml"
    global_raw = (
        yaml.safe_load(global_config_path.read_text(encoding="utf-8"))
        if global_config_path.exists()
        else {}
    )
    global_data = global_raw if isinstance(global_raw, dict) else {}

    # Load per-repo config if present
    repo_raw = (
        yaml.safe_load(repo_config_path.read_text(encoding="utf-8"))
        if repo_config_path and repo_config_path.exists()
        else {}
    )
    repo_data = repo_raw if isinstance(repo_raw, dict) else {}

    # Merge: global as base, per-repo as override (section-by-section, deep)
    merged_data: dict[str, Any] = {}
    all_sections = set(global_data.keys()) | set(repo_data.keys())

    for section in all_sections:
        global_section = global_data.get(section, {})
        repo_section = repo_data.get(section, {})

        # Both should be dicts for a proper merge
        global_section = global_section if isinstance(global_section, dict) else {}
        repo_section = repo_section if isinstance(repo_section, dict) else {}

        # Merge: repo values override global values. The api_worker section is
        # deep-merged so partial per-repo overrides (e.g. budget caps or provider
        # additions) do not drop global defaults. All other sections keep the
        # original shallow-merge semantics: repo keys fully replace global keys.
        if section == "api_worker":
            merged_section = _deep_merge(global_section, repo_section)
        else:
            merged_section = {**global_section, **repo_section}
        if merged_section:
            merged_data[section] = merged_section

    # If no config at all, delegate to the original load_config for consistency
    if not merged_data:
        return load_config(repo_config_path)

    # Use the existing load_config logic by writing a merged dict to a temp file
    # This ensures we reuse all validation logic (unknown keys, type checks, etc.)
    # without duplicating it here.
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as tmp:
        yaml.dump(merged_data, tmp)
        tmp_path = Path(tmp.name)

    try:
        return load_config(tmp_path)
    finally:
        # Clean up the temp file
        try:
            tmp_path.unlink()
        except OSError:
            pass
