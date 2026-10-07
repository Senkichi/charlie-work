from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import Any, TypeVar

import yaml

from .config import (
    ConfigError,
    OrchestratorConfig,
    build_config_from_data,
    default_config_path,
    known_config_sections,
    load_config,
)
from . import layout
from .config_deprecations import emit_deprecated_key_reads, repo_state_path
from .config_validation import ConstructionError, host_wide_error, host_wide_sections
from .fleet_paths import fleet_dir

from .paths import RepoNotFoundError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def config_layer_paths(
    repo_root: Path,
    explicit: Path | None = None,
    *,
    fleet_dir_override: str | None = None,
) -> tuple[tuple[str, Path], ...]:
    """The config-layer files ``load_layered_config`` reads, in merge order.

    Each entry is ``(layer, path)`` where *layer* is ``"user-global"`` (the
    fleet dir's ``config.yaml``) or ``"repo"`` (the resolved per-repo file).
    Paths come back whether or not the file exists -- the fleet retirement
    sweep (issue #1976) needs the *slots*, not just the files that happen to
    be present, so an absent layer is still "checked". ``load_layered_config``
    itself resolves its layer paths through this function, so the sweep and
    the loader share one source set by construction rather than by parallel
    resolution that could drift.
    """
    return (
        ("user-global", layout.global_config_path(override=fleet_dir_override)),
        ("repo", default_config_path(repo_root)) if explicit is None else ("repo", explicit),
    )


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


def peek_runtime_state_dir(
    repo_root: Path,
    explicit: Path | None = None,
    *,
    fleet_dir_override: str | None = None,
) -> str:
    """Best-effort read of ``runtime.state_dir`` from the config layers.

    Exists for one caller: the boot-time pending-sync repair (issue #2312)
    must locate the marker *before* ``load_fleet_global_config`` runs -- the
    dependency skew it repairs is exactly what can crash that load. Reads the
    same two layer slots ``load_layered_config`` uses and applies the same
    repo-wins-per-key precedence, but only for ``runtime.state_dir``, and
    tolerates every failure the real loader would surface: absent files,
    malformed YAML, non-mapping documents, and non-mapping ``runtime``
    sections all contribute nothing.

    A layer that declares ``state_dir`` wins even when the value is unusable:
    the real loader would raise ``ConfigError`` on it and the fleet entry
    points fall back to defaults, so the peek's own default matches where
    that fallback puts the marker. The string comes back unresolved --
    resolution belongs to ``runtime_paths``/``supervisor_runtime_paths``.
    """
    state_dir: Any = None
    for _layer, path in config_layer_paths(
        repo_root, explicit, fleet_dir_override=fleet_dir_override
    ):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict):
            continue
        runtime = raw.get("runtime")
        if isinstance(runtime, dict) and "state_dir" in runtime:
            state_dir = runtime["state_dir"]
    return state_dir if isinstance(state_dir, str) else layout.DEFAULT_STATE_DIR


_AVIATOR_CONFIG = Path(".aviator") / "config.yml"
_AVIATOR_WARNED: set[Path] = set()


def warn_if_aviator_config_missing(config: OrchestratorConfig, repo_root: Path) -> bool:
    """Warn (once per repo per process) when ``mergequeue_label`` is set but the repo has no
    ``.aviator/config.yml`` (#2441): the hand-off labels PRs for a queue that has no rules
    to merge them, so they sit labelled forever. Returns True when it warned.
    """
    label = config.auto_merge.mergequeue_label
    if not label or (repo_root / _AVIATOR_CONFIG).is_file():
        return False
    key = repo_root.resolve()
    if key in _AVIATOR_WARNED:
        return False
    _AVIATOR_WARNED.add(key)
    logger.warning(
        "auto_merge.mergequeue_label=%r is set but %s has no %s; PRs handed to the merge "
        "queue will not be merged until that file exists",
        label,
        repo_root,
        _AVIATOR_CONFIG.as_posix(),
    )
    return True


def load_layered_config(
    repo_root: Path,
    explicit: Path | None = None,
    *,
    fleet_dir_override: str | None = None,
    require_global: bool = False,
) -> OrchestratorConfig:
    """``_load_layered_config`` plus the missing-``.aviator/config.yml`` warning (#2441)."""
    config = _load_layered_config(
        repo_root, explicit, fleet_dir_override=fleet_dir_override, require_global=require_global
    )
    warn_if_aviator_config_missing(config, repo_root)
    return config


def _load_layered_config(
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
    # Both layer paths come from config_layer_paths (issue #1976): the fleet
    # retirement sweep enumerates exactly these slots, so the loader and the
    # sweep share one source set by construction. The repo slot is the
    # *candidate* -- explicit wins unconditionally, otherwise the default
    # filename whether or not it exists; existence is gated at ``repo_source``
    # below and ``load_config`` treats a nonexistent path as no config.
    layer_map = dict(
        config_layer_paths(repo_root, explicit, fleet_dir_override=fleet_dir_override)
    )
    global_config_path = layer_map["user-global"]
    repo_config_path = layer_map["repo"]
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
    # Issue #1976: a registered deprecated key in the global layer is worth a
    # fleet-level event the moment it is read. Emitted here rather than after
    # the merge because the merged path is also reached via the
    # discarded-global-layer rescue below, where this layer was still read
    # even though it did not contribute.
    if global_exists:
        try:
            emit_deprecated_key_reads(
                global_data,
                source_path=global_config_path,
                state_path=layout.state_file_path(fleet_dir(override=fleet_dir_override)),
            )
        except Exception:  # noqa: BLE001 — deprecation telemetry must never break config load
            logger.debug("config_key_deprecated_read emit failed for %s", global_config_path)

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

    # Load per-repo config if present. Bound once rather than re-tested, so the
    # provenance recorded below cannot disagree with what was actually read
    # (issue #943).
    repo_source = repo_config_path if repo_config_path and repo_config_path.exists() else None
    repo_raw = (
        yaml.safe_load(repo_source.read_text(encoding="utf-8")) if repo_source is not None else {}
    )
    repo_data = repo_raw if isinstance(repo_raw, dict) else {}

    # Provenance in merge order: global layer first, per-repo last, matching the
    # override precedence below. Keyed off the *reads* above, not a fresh
    # exists() check. A file that was read but contributed no sections still
    # belongs here -- "read a 0-byte global layer" and "there is no global
    # layer" are the two readings of #590 that took hours to separate, and they
    # are only distinguishable if an empty-but-present file leaves a trace.
    layer_sources: tuple[str, ...] = tuple(
        str(p) for p in (global_config_path if global_exists else None, repo_source) if p
    )

    # Host-wide sections (declared with ``HostWideOnly`` on the ``OrchestratorConfig``
    # field, never listed here): one physical machine / one fleet supervisor daemon, so
    # three repos must not hold three opinions about it. The merge below is
    # section-by-section with the per-repo file winning per key, so without this
    # rejection a per-repo ``orchestrator.config.yaml`` could silently override a
    # host-wide knob -- the exact confusion that made #590 expensive to diagnose. Reject
    # the key outright so the invalid state is unrepresentable rather than merely unused
    # (issues #600, #763, #1978). A legacy ``supervisor.<key>`` spelling of a
    # moved fleet-supervisor knob needs no carve-out here: since #1979 it is
    # just an unknown key, and the ordinary validation below rejects it like
    # any other.
    for host_wide in sorted(host_wide_sections()):
        if host_wide in repo_data:
            raise host_wide_error(
                host_wide, fleet_dir=global_config_path.parent, repo_path=repo_config_path
            )

    # Merge: global as base, per-repo as override (section-by-section, deep)
    merged_data: dict[str, Any] = {}
    all_sections = set(global_data.keys()) | set(repo_data.keys())
    known_sections = known_config_sections()

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
        # A falsy merged section (both layers empty/absent for this name) is
        # dropped so it doesn't shadow a dataclass default -- load_config's own
        # `_section()` already defaults a *known* section that's absent
        # entirely, so dropping an empty-but-known one changes nothing.
        #
        # An *unknown* section name must survive this filter even when its
        # body is empty (`{}`, `null`, `[]` all coerce to `{}` above), or it
        # never reaches load_config's unknown-section check at all -- that was
        # issue #962: `bogus_section: {}` merged to falsy and vanished from
        # merged_data before validation ever saw the name, so a typo'd section
        # with no body was silently accepted here while load_config (which
        # checks raw key presence, not truthiness) rejected the identical
        # file. Keeping the name (with its coerced-empty value) lets
        # build_config_from_data raise "unknown config section(s)" exactly
        # as it does for a non-empty bogus section, and keeps the #665
        # discarded-global-layer rescue below in play for it.
        if merged_section or section not in known_sections:
            merged_data[section] = merged_section

    # If no config at all, delegate to the original load_config for consistency.
    # Its own provenance covers only the per-repo path it is handed, so restate
    # the full layer list: reaching here with a *present* global layer means that
    # file was read and parsed to nothing, which is a different diagnosis from
    # never having had one.
    if not merged_data:
        return replace(load_config(repo_config_path), sources=layer_sources)

    # Build the merged config in memory. This reuses load_config's exact
    # section-validation logic (unknown keys, type checks, etc.) via
    # build_config_from_data -- the shared helper extracted from load_config
    # that takes a raw dict instead of a path -- so the two entry points
    # cannot drift apart, but neither performs a filesystem round-trip to get
    # there (issue #704: the previous implementation wrote merged_data to a
    # NamedTemporaryFile and read it straight back through load_config, which
    # cost a write+read on every config load and left a temp file to clean
    # up).
    try:
        # build_config_from_data leaves ``sources`` at its dataclass default
        # (it only ever sees a dict); attach the real layer provenance here,
        # the same way load_config attaches a single path's provenance.
        merged = replace(build_config_from_data(merged_data), sources=layer_sources)
        # Issue #1976: emit the repo layer's deprecated reads here -- the two
        # delegation branches (``not merged_data``, discarded global layer)
        # get theirs from ``load_config`` instead, so this is the one place
        # the merged path's repo-layer read is recorded.
        if repo_source is not None:
            try:
                emit_deprecated_key_reads(
                    repo_data,
                    source_path=repo_source,
                    state_path=repo_state_path(repo_root, merged.runtime.state_dir),
                )
            except Exception:  # noqa: BLE001 — deprecation telemetry must never break config load
                logger.debug("config_key_deprecated_read emit failed for %s", repo_source)
        return merged
    except ConfigError as exc:
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
        # A section constructor's own rejection (ci_fleet's host-wide
        # ``__post_init__`` rules, ``rescue.worker``) is the global layer's
        # *host-wide* breakage. Discarding that layer would silently default
        # every host-wide knob -- the #623 shape -- so it stays loud.
        if isinstance(exc, ConstructionError):
            raise
        # Provenance is the per-repo file alone, deliberately: the global
        # layer was *discarded*, so listing it would claim a contribution
        # that was rolled back. This is the case the field earns its keep
        # on -- the warning below scrolls away, the value does not.
        repo_only = replace(load_config(repo_config_path), sources=(str(repo_config_path),))
        logger.warning(
            "Layered config: merged load failed validation; the global "
            "layer was discarded and the per-repo config used alone. "
            "global path=%s",
            global_config_path,
        )
        return repo_only


def load_fleet_global_config(
    load: Callable[..., OrchestratorConfig],
    cwd: Path,
    *,
    fleet_dir_override: str | None,
    fallback: _T,
    report: Callable[[Exception], None],
) -> OrchestratorConfig | _T:
    """Load the fleet entry points' global config: require the global layer, degrade softly.

    The one place ``fleet supervise`` / ``fleet work`` / ``fleet bash-rats`` decide what a
    failed global load means. A missing or invalid global layer is reported through
    *report*, then the per-repo config is reloaded without the global requirement so it
    survives (the #623 silent-disable shape); only if that also fails does the caller's
    *fallback* apply.

    A :class:`ConstructionError` is never degraded. It is a *host-wide* section's own
    constructor rejecting the global layer (ci_fleet's ``__post_init__`` rules, design F1),
    which main surfaced as a raw ``ValueError`` that these handlers did not catch, so the
    process refused to start. Continuing on defaults would silently turn runner allocation
    and the fleet caps off (#623 / #590). ``ConstructionError`` subclasses ``ConfigError``,
    so it must be re-raised explicitly ahead of that catch.

    *load* is the caller's own ``load_layered_config`` binding (injected, not imported here,
    so the caller module remains the seam its tests patch). The per-repo call sites that
    skip a repo on ``ConfigError`` (autoscale prologue, fleet status) stay as they are: they
    are non-fatal per repo, and skipping one repo is not a host-wide fail-open.
    """
    try:
        return load(cwd, None, fleet_dir_override=fleet_dir_override, require_global=True)
    except ConstructionError:
        raise
    except (ConfigError, RepoNotFoundError) as exc:
        report(exc)
        try:
            return load(cwd, None, fleet_dir_override=fleet_dir_override)
        except ConstructionError:
            raise
        except (ConfigError, RepoNotFoundError):
            return fallback
