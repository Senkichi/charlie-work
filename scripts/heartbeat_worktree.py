"""Config + worktree path resolution helpers for ``heartbeat_check.py``.

The config loader (``load_orchestrator_config``) and the worktree-path
derivation family (``_slugify_branch``, ``_resolved_worktrees_dir``,
``_registered_worktree_for_branch``, ``_worktree_path_for_branch``) live
here. ``_worktree_path_for_branch`` mirrors
``charlie_work.worktree.worktree_path_for_branch``: a branch already checked
out in a registered worktree wins over the managed-path computation — an
adopted foreign checkout (issue #1476) lives outside the managed root, and
reporting "no worktree found" for a live adopted worker would be a false
ANOMALY in the in-progress-staleness check.

Loaded from ``heartbeat_check.py`` via ``importlib`` from the sibling script
path, never a bare ``import`` — matching how #1895 loads
``heartbeat_event_alarms.py`` and #1879 loads ``heartbeat_local_repo.py``:
``scripts/`` is not a package and is deliberately kept off ``sys.path`` by
the test harness (``tests/_script_loader.py``). ``heartbeat_check``
re-exports the names its own code and tests reference, so ``hb.*``
attribute references keep resolving unchanged. This module is never run
standalone.

Extracted out of ``heartbeat_check.py`` to create file-size ratchet headroom
(``file_size_ratchet_baseline/scripts/heartbeat_check.py.count``) — no
behavior change beyond the issue #1476 registry-first resolution, which is
the feature the extraction serves.

Stdlib-only, same constraint as ``heartbeat_check.py`` itself
(``scripts/README.md``): no ``charlie_work`` or third-party imports beyond
``yaml``, and never an import back into ``heartbeat_check`` — that would
cycle through its loader block, which is also why ``RepoInfo`` is a
``TYPE_CHECKING``-only name and ``CREATE_NO_WINDOW``/``GH_TIMEOUT_SECONDS``
are mirrored below rather than shared.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    # ``heartbeat_check`` is not importable as a module at runtime
    # (``scripts/`` is not a package and stays off ``sys.path``); this name
    # exists so the moved functions keep their original annotations
    # byte-identically. A runtime import would cycle through
    # ``heartbeat_check``'s own loader block.
    from heartbeat_check import RepoInfo

# Mirrors heartbeat_check's CREATE_NO_WINDOW (``getattr``-guarded: the
# attribute only exists on Windows).
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# Mirrors heartbeat_check's GH_TIMEOUT_SECONDS — the same bounded-subprocess
# posture applies to `git worktree list`.
GH_TIMEOUT_SECONDS = 30


def load_orchestrator_config(config_path: Path) -> tuple[dict[str, Any], str | None]:
    """Load an orchestrator.config.yaml.

    Returns (config, error). error is None when the config is legitimately
    absent -- including an unset config_path, which load_repos() represents
    as Path("") (== Path("."), the "no config registered for this repo"
    sentinel -- deliberately not treated as cwd-relative) -- or when the file
    parses cleanly to a mapping. error is a message when config_path is set
    and points at something that exists but fails to read, isn't valid UTF-8,
    fails to parse as YAML, or parses to something other than a mapping.
    That "present but broken" case (issue #703) must not be treated the same
    as "absent" -- callers that need it surfaced use check_orchestrator_config
    below rather than reading the error here.
    """
    if config_path == Path(""):
        return {}, None
    try:
        if not config_path.exists():
            return {}, None
        raw = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return {}, f"{config_path}: {exc}"
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return {}, f"{config_path}: invalid YAML: {exc}"
    if not isinstance(data, dict):
        return {}, f"{config_path}: expected a mapping at top level, got {type(data).__name__}"
    return data, None


def _slugify_branch(branch: str) -> str:
    """Mirror ``charlie_work.worktree._slugify`` (stdlib-only reimplementation).

    The production function lives in ``charlie_work.worktree``; this script
    cannot import it (stdlib-only invariant, scripts/README.md). The two must
    agree so the worktree path derived here matches the one the orchestrator
    created. ``tests/test_heartbeat_check_in_progress_stale.py`` exercises
    the same derivation against real branch names, so a drift surfaces as a
    missing-dir test failure rather than a silent false ANOMALY.
    """
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", branch).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].rstrip("-") or "worktree"


def _resolved_worktrees_dir(repo: RepoInfo) -> Path:
    """Resolve the worktrees root for ``repo``, honouring ``claude_code.worktrees_dir``.

    Mirrors ``charlie_work.paths.resolved_layout``'s worktrees resolution
    (issue #1379 review): ``claude_code.worktrees_dir`` is a sentinel-style
    override -- ``None``/empty means "derive from ``runtime.state_dir``"
    (``<state_dir>/worktrees``), a non-empty value is an explicit path
    (absolute returned as-is, relative joined to ``repo_root``). This script
    cannot import ``charlie_work.config``/``paths`` (stdlib-only invariant,
    scripts/README), so the resolution is reimplemented locally against the
    config dict ``load_orchestrator_config`` already returns -- the same
    reimplement-locally treatment ``fleet_dir`` and the stale-open-issue-mention
    primitives use. A broken/unreadable config degrades to the default
    (fail-toward-flagging: a missing worktree dir reads as ANOMALY, not OK).
    """
    default = repo.state_dir / "worktrees"
    config, _error = load_orchestrator_config(repo.config_path)
    raw = config.get("claude_code", {}).get("worktrees_dir")
    if not raw or not isinstance(raw, str):
        return default
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo.repo_root / candidate


def _registered_worktree_for_branch(repo: RepoInfo, branch: str) -> Path | None:
    """The path where ``branch`` is checked out per ``git worktree list``, or
    None on no match or any git failure (fails closed — callers degrade to
    the managed path, matching ``charlie_work.worktree.list_worktrees``)."""
    try:
        proc = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=str(repo.repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=GH_TIMEOUT_SECONDS,
            creationflags=CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    # Porcelain order within a record puts `worktree <path>` before `branch`.
    wt_path: str | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            wt_path = line[len("worktree ") :]
        elif line.startswith("branch "):
            if wt_path and line[len("branch ") :].removeprefix("refs/heads/") == branch:
                return Path(wt_path)
    return None


def _worktree_path_for_branch(
    repo: RepoInfo, branch: str, worktrees_dir: Path | None = None
) -> Path:
    """Return the worktree dir for ``branch`` under ``repo``'s worktrees root.

    Mirrors ``charlie_work.worktree.worktree_path_for_branch``. A branch
    already checked out in a registered worktree wins — an adopted foreign
    checkout (issue #1476) lives outside the managed root, and reporting
    "no worktree found" for a live adopted worker would be a false ANOMALY.
    The worktrees root defaults to ``_resolved_worktrees_dir(repo)`` (which
    honours ``claude_code.worktrees_dir``); pass ``worktrees_dir`` to
    override it once (e.g. a caller that resolves it once for many
    branches). ``repo.state_dir`` is the state root (the directory holding
    ``state.json``, as registered in fleet.json).
    """
    registered = _registered_worktree_for_branch(repo, branch)
    if registered is not None:
        return registered
    root = worktrees_dir if worktrees_dir is not None else _resolved_worktrees_dir(repo)
    return root / _slugify_branch(branch)
