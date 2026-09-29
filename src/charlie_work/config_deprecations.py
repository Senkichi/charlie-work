"""Deprecated-config-key registry (issue #1976).

A config key is deprecated iff it appears in ``DEPRECATED_CONFIG_KEYS`` --
there is deliberately no second list, marker, or flag anywhere else. The
registry entry carries the ``removal_issue`` that deletes the key, so the
fleet retirement sweep (``config_retirement_sweep``) can mark that issue
Ready once the key has been absent from every config layer of every
registered repo for the ``runtime.config_retirement_quiet_days`` quiet
window.

``config.py`` imports this module at load time, so it must stay free of
``charlie_work`` imports that could cycle back -- ``log_event`` is imported
lazily inside ``emit_deprecated_key_reads`` for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class DeprecatedConfigKey:
    """One deprecated config key and the issue that retires it.

    ``replacement`` is ``section.key`` form (e.g. ``"dispatch.max_limit"``),
    or ``None`` when the knob was retired without a successor -- the
    ``dispatch.require_worker_github_token`` case, where the gate was
    removed outright (issue #1853).
    """

    section: str
    key: str
    replacement: str | None
    removal_issue: int

    @property
    def dotted(self) -> str:
        return f"{self.section}.{self.key}"


#: THE registry. Membership is what makes a key "deprecated"; tests and the
#: retirement sweep read this and nothing else.
DEPRECATED_CONFIG_KEYS: tuple[DeprecatedConfigKey, ...] = (
    # Issue #1853 made this a no-op kept only so existing config files parse;
    # issue #1977 removes it once the quiet window proves it is set nowhere.
    DeprecatedConfigKey(
        section="dispatch",
        key="require_worker_github_token",
        replacement=None,
        removal_issue=1977,
    ),
)


def deprecated_keys_in(data: Mapping[str, Any]) -> list[DeprecatedConfigKey]:
    """Return the registered keys present in one raw config-layer mapping.

    Presence means ``data[section][key]`` exists -- the *value* is irrelevant
    to deprecation (a key set to its default is still a live reference the
    removal PR would break).
    """
    found: list[DeprecatedConfigKey] = []
    for entry in DEPRECATED_CONFIG_KEYS:
        section = data.get(entry.section)
        if isinstance(section, Mapping) and entry.key in section:
            found.append(entry)
    return found


def repo_state_path(repo_root: Path, state_dir: str) -> Path:
    """Resolve a repo's ``state.json`` path for ``runtime.state_dir``.

    Mirrors ``paths.runtime_paths``'s root resolution (relative anchors on
    ``repo_root``) without its phantom-dir warning -- config loading runs
    long before any state exists on a fresh checkout, so warning here would
    be the false positive #648's heuristic exists to avoid.
    """
    from . import layout

    root = Path(state_dir)
    if not root.is_absolute():
        root = repo_root / root
    return layout.state_file_path(root)


def emit_deprecated_key_reads(
    data: Mapping[str, Any],
    *,
    source_path: Path,
    state_path: Path,
    repo: str | None = None,
) -> None:
    """Emit ``config_key_deprecated_read`` for each registered key in *data*.

    Called by the config loaders at the point a layer file has just been
    read into ``data``, so ``source_path`` is the file the key actually came
    from -- never the merged view. Best-effort like every ``log_event`` call;
    the retirement sweep re-derives presence from the files themselves, so a
    dropped event never corrupts retirement state.
    """
    from .instrumentation import log_event

    for entry in deprecated_keys_in(data):
        try:
            # write-gate-exempt(issue=1976): read-time observability; config load has no write_gate and must fire under dry-run.
            log_event(
                state_path,
                "config_key_deprecated_read",
                {
                    "section": entry.section,
                    "key": entry.key,
                    "source": str(source_path),
                    "replacement": entry.replacement,
                    "issue_number": entry.removal_issue,
                },
                repo=repo,
            )
        except Exception:  # noqa: BLE001 — instrumentation never breaks config load
            pass
