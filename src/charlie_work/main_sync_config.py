"""``main_sync:`` config section -- keep a registered repo's local main on origin.

Lives outside ``config.py`` (which sits near its file-size ratchet mark) and is
re-exported there as the type of ``OrchestratorConfig.main_sync``. Config sections do
not nest, so the opt-in post-sync hook is flat ``post_sync_*`` keys. See
docs/specs/main-sync.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from .config_validation import NotNull, Positive, Typed


@dataclass(frozen=True)
class MainSyncConfig:
    """Per-repo (and fleet-wide, via ``<fleet_dir>/config.yaml``) main-sync settings.

    ``enabled``: the kill switch; on by default.

    ``post_sync_command``: argv run with cwd = the checkout root after a working-tree
    fast-forward whose range touches ``post_sync_when_paths``; never through a shell.
    Empty means no post_sync. Exit 0 keeps the advance, exit 75 (EX_TEMPFAIL) defers
    it, anything else (or a timeout) fails it; both roll back with
    ``git reset --keep``.

    ``post_sync_when_paths``: ``fnmatch`` patterns over repo-relative POSIX paths
    (``*`` crosses ``/``). Empty means every advance runs ``post_sync_command``.
    """

    enabled: Annotated[bool, Typed] = True
    post_sync_command: Annotated[tuple[str, ...], Typed, NotNull] = ()
    post_sync_when_paths: Annotated[tuple[str, ...], Typed, NotNull] = ()
    post_sync_timeout_seconds: Annotated[int, Typed, Positive] = 300
