from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .config import (
    ConfigError,
    OrchestratorConfig,
    find_config_path,
    load_config,
)
from .fleet_paths import fleet_dir

logger = logging.getLogger(__name__)


def describe_config_file(path: Path) -> str:
    """Describe a config file's readability for provenance logging.

    ``Path.exists()`` does not report *why* it says no. It swallows every error
    in ``pathlib._ignore_error`` -- ``ENOENT``, ``ENOTDIR``, ``EBADF``, ``ELOOP``
    and, on Windows, the device-not-ready / invalid-name / cannot-resolve-filename
    winerrors -- and returns a bare ``False`` for all of them. So "the file was
    never created" is indistinguishable from "the volume wasn't ready" or "the
    path could not be resolved", and *every one* of those takes the silent-``{}``
    branch in :func:`load_layered_config`, yielding a config of pristine
    dataclass defaults with no error raised anywhere.

    That is the whole of issue #590's remaining unknown, so callers log this
    string rather than an ``exists=`` flag. One ``stat()``, distinguishable
    outcomes, and no second filesystem call that could disagree with the first.

    Note that ``EACCES`` is *not* in that ignored set: a config that exists but
    is permission-denied makes ``exists()`` raise rather than return False, so
    it would crash the caller instead of silently defaulting. Permissions are
    therefore ruled out as a cause of a silently-defaulted config -- worth
    knowing, because it is the first thing one reaches for.
    """
    try:
        return f"present bytes={path.stat().st_size}"
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        return f"UNREADABLE ({type(exc).__name__}: {exc})"


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
    require_global: bool = False,
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
        require_global: When True, treat an unreachable global fleet config as
            a hard error rather than an empty mapping. A plain single-repo
            checkout has no global layer and that absence is legitimate, so the
            default is False; fleet entry points (``run_fleet_supervise``,
            ``run_fleet_work``, ``run_fleet_bash_rats``) pass True because every
            fleet-wide knob silently reverting to its dataclass default while
            passes keep reporting success is the #590/#623 failure shape. The
            raised ``ConfigError`` names the path and the
            ``describe_config_file`` cause so an unready volume or an
            unresolvable path is distinguishable from a file that was never
            created -- ``Path.exists()`` collapses all of those into a bare
            ``False`` and this is the one place that un-collapses them.

    Returns:
        The merged OrchestratorConfig.
    """
    # Load per-repo config path
    repo_config_path = find_config_path(repo_root, explicit)

    # Load global config if present
    global_config_path = fleet_dir(override=fleet_dir_override) / "config.yaml"
    global_exists = global_config_path.exists()
    if require_global and not global_exists:
        # The silent-{} branch below is the whole of issue #623: every
        # fleet-wide knob reverts to its dataclass default with no error
        # raised anywhere, and ``exists()`` swallows ENOENT/ENOTDIR/EBADF/
        # ELOOP plus the Windows device-not-ready / unresolvable-path
        # winerrors into a bare False, so a genuinely absent file is
        # indistinguishable from an unready volume. ``describe_config_file``
        # does one ``stat()`` and keeps the cause, so the error message
        # separates "never created" from "could not be reached" -- the two
        # demand opposite fixes. ``EACCES`` is not in the ignored set, so a
        # permission-denied config raises from ``exists()`` before reaching
        # here and is therefore not a silent-default mechanism.
        raise ConfigError(
            "global fleet config layer is required but was not readable: "
            f"{global_config_path} — {describe_config_file(global_config_path)}"
        )
    global_raw = (
        yaml.safe_load(global_config_path.read_text(encoding="utf-8")) if global_exists else {}
    )
    global_data = global_raw if isinstance(global_raw, dict) else {}

    # Provenance, not values. An absent global layer is legitimate (a plain
    # single-repo checkout has none), so this cannot be a warning here -- but
    # its effect is that every fleet-wide knob silently reverts to its dataclass
    # default while passes keep reporting success. #590 was indistinguishable
    # from "the feature was never wired up" for hours because the resolved
    # config records what a section *became*, never whether the file that
    # declares it was read. Callers that do expect a global layer log this at
    # INFO themselves (see run_fleet_supervise).
    # One vocabulary for both provenance lines, so this and the supervisor's INFO
    # line are directly comparable. A bare "present" would collapse a truncated
    # 0-byte file into the same token as a populated one, and "present
    # sections=(none)" is *exactly* the #590 signature -- the byte count is what
    # separates "the file is empty" from "the file has content that did not
    # parse into any section".
    #
    # ``exists()`` stays the read gate on purpose: it swallows a specific set of
    # errors (see describe_config_file) and changing which failures reach the
    # caller is a behaviour change, not a logging one. The description is a
    # second, independent observation used only for the message -- if the two
    # ever disagree, sections= and the description are both printed, so the
    # contradiction is visible in the log rather than resolved silently.
    logger.debug(
        "Layered config: global path=%s %s sections=%s; repo path=%s",
        global_config_path,
        describe_config_file(global_config_path),
        sorted(global_data) if global_data else "(none)",
        repo_config_path,
    )

    # Load per-repo config if present
    repo_raw = (
        yaml.safe_load(repo_config_path.read_text(encoding="utf-8"))
        if repo_config_path and repo_config_path.exists()
        else {}
    )
    repo_data = repo_raw if isinstance(repo_raw, dict) else {}

    # ``runner_allocation`` is host-wide only (see RunnerAllocationConfig's
    # docstring): three repos must not hold three opinions about how many jobs
    # one machine can run. The merge below is section-by-section with the
    # per-repo file winning per key, so without this rejection a per-repo
    # ``orchestrator.config.yaml`` could silently override a host-wide knob --
    # the exact confusion that made #590 expensive to diagnose. Reject the key
    # outright so the invalid state is unrepresentable rather than merely
    # unused (issue #600).
    if "runner_allocation" in repo_data:
        raise ConfigError(
            "config section 'runner_allocation' is host-wide only and must not "
            f"appear in a per-repo config ({repo_config_path}); declare it in "
            "the global fleet layer (<fleet_dir>/config.yaml) instead"
        )

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
        try:
            return load_config(tmp_path)
        except ConfigError:
            # A present-but-invalid global layer (e.g. an unknown key) makes
            # the merged load raise, and callers (fleet_dispatch) catch
            # ConfigError and skip the repo -- silently discarding a *valid*
            # per-repo config. That is the #623 failure shape (host-wide knobs
            # silently disabled) via a different trigger (issue #665). When a
            # per-repo config exists, retry with it alone so the global layer's
            # breakage does not take the per-repo config down with it. With no
            # per-repo config to rescue, propagate the original error --
            # silently defaulting would itself reproduce the #623 shape.
            if not global_exists or not repo_data:
                raise
            repo_only = load_config(repo_config_path)
            logger.warning(
                "Layered config: merged load failed validation; the global "
                "layer was discarded and the per-repo config used alone. "
                "global path=%s",
                global_config_path,
            )
            return repo_only
    finally:
        # Clean up the temp file
        try:
            tmp_path.unlink()
        except OSError:
            pass
