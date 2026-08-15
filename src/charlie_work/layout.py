"""Single source of truth for every well-known path name the orchestrator uses.

Why this module exists
----------------------
Before this module, well-known filenames were re-spelled at each use site:
``"supervisor.lock"`` appeared in five modules, ``"fleet.json"`` in four, and
the default state-dir literal ``.var/charlie-work`` was hardcoded in the
package defaults of four separate config dataclasses. That duplication is not
merely untidy — it produced a live split-brain bug. ``dispatch`` created
worktrees under one root while ``charlie worktree-clean`` swept a different
one, so 74 worktrees accumulated uncollected on a repo that had overridden
``runtime.state_dir``: the override moved the *clean* side and left the
*create* side pinned to the hardcoded default.

The fix is structural rather than a set of scattered corrections. Every
filename lives here exactly once, and every path is composed from a root that
the caller passes in. A path can then only be wrong in one place instead of
five, and the two sides of any create/clean pair are spelled by the same
function.

Two distinct roots
------------------
Callers must not conflate them:

* **State root** — per-repo orchestrator state, ``runtime.state_dir`` resolved
  against the repo root (default ``.var/charlie-work``). Use
  :func:`charlie_work.paths.runtime_paths` to obtain it; the ``state_root``
  helpers below take that ``.root``.
* **Fleet dir** — host-wide, shared by every registered repo
  (``%LOCALAPPDATA%\\charlie-work`` on Windows). Obtained from
  :func:`charlie_work.fleet_paths.fleet_dir`; the ``*_path`` helpers below
  wrap it and take the same ``override`` parameter.

A test (``tests/test_no_path_literals.py``) enforces that these names are not
re-spelled elsewhere in the package.
"""

from __future__ import annotations

from pathlib import Path

from .fleet_paths import fleet_dir

# --- state-dir layout ------------------------------------------------------

#: Default value of ``runtime.state_dir``, relative to the repo root.
#:
#: ``config.RuntimeConfig.state_dir`` reads this rather than re-spelling the
#: literal, so the default exists in exactly one place. Note that it is a
#: *relative* POSIX-style string, not a Path: it is a config default that the
#: operator may override with either a relative or an absolute path, and
#: ``runtime_paths`` performs the resolution.
DEFAULT_STATE_DIR = ".var/charlie-work"

STATE_FILENAME = "state.json"
SUPERVISOR_LOCK_FILENAME = "supervisor.lock"
PENDING_SYNC_FILENAME = "pending-sync.json"
SELF_DEPLOY_FAILURE_STATE_FILENAME = "self-deploy-failures.json"
ZERO_PASS_STREAK_STATE_FILENAME = "zero-pass-streak.json"

ISSUES_DIRNAME = "issues"
PRS_DIRNAME = "prs"
LOGS_DIRNAME = "logs"
DISPATCHES_DIRNAME = "dispatches"
SESSIONS_DIRNAME = "sessions"
REVIEWS_DIRNAME = "reviews"
WORKTREES_DIRNAME = "worktrees"
CROSS_FAMILY_DIRNAME = "cross-family"
NOTIFY_DIRNAME = "notify"
NOTIFY_DIGEST_FILENAME = "digest.jsonl"
SESSION_MANIFEST_FILENAME = "session-manifest.json"
SESSION_RESULTS_FILENAME = "session-results.json"

#: Worktree-local ``gh`` configuration directory name components.
#:
#: Deliberately *outside* the orchestrator state dir: this isolates each
#: worktree's ``gh`` config from the host's, so a worker cannot inherit or
#: clobber ambient credentials. It is keyed on the worktree path, not on
#: ``runtime.state_dir``, and must stay that way — see :func:`gh_config_dir`.
GH_CONFIG_DIRNAME = "gh-config"

_VAR_DIRNAME = ".var"

#: Directory names whose *re-spelling* is a real divergence hazard, and which
#: ``tests/test_no_path_literals.py`` therefore forbids outside this module.
#:
#: Deliberately narrower than the full set of dirnames above. The generic
#: per-repo subdirectories (``issues``, ``prs``, ``logs``, ``dispatches``) are
#: already exposed as members on :class:`charlie_work.paths.RuntimePaths`, so
#: the correct fix at those sites is to *use the member* (``paths.prs``) rather
#: than to swap one literal for a constant — substituting
#: ``root / PRS_DIRNAME`` for ``root / "prs"`` adds an indirection while
#: preventing nothing. Their constants exist solely so ``runtime_paths()`` can
#: define the members in one place.
#:
#: The names below are different: each is either a bare filename with no
#: owning member, or a root that actually diverged into two spellings in
#: production (``worktrees`` being the one that cost 74 uncollected worktrees).
_ENFORCED_DIRNAMES = (
    WORKTREES_DIRNAME,
    CROSS_FAMILY_DIRNAME,
    SESSIONS_DIRNAME,
    GH_CONFIG_DIRNAME,
    _VAR_DIRNAME,
)


def default_state_root(repo_root: Path) -> Path:
    """Return the *default* state root for ``repo_root``.

    This is ``<repo_root>/.var/charlie-work`` — the value a repo gets when it
    does not override ``runtime.state_dir``.

    Prefer ``runtime_paths(repo_root, config.runtime.state_dir).root`` wherever
    a config is in hand: this helper ignores any override and is therefore only
    correct as a last-resort fallback for library callers that genuinely have
    no config (see :func:`charlie_work.worktree.worktree_path_for_branch`).
    """
    return repo_root / _VAR_DIRNAME / Path(DEFAULT_STATE_DIR).name


def state_file_path(state_root: Path) -> Path:
    """Return the ``state.json`` path under ``state_root``."""
    return state_root / STATE_FILENAME


def supervisor_lock_path(state_root: Path) -> Path:
    """Return the per-repo supervisor lock path under ``state_root``.

    Distinct from :func:`fleet_supervisor_lock_path`, which is host-wide. This
    one serialises loops driving a *single* repo; that one serialises the fleet
    supervisor across every repo on the host.
    """
    return state_root / SUPERVISOR_LOCK_FILENAME


def pending_sync_path(state_root: Path) -> Path:
    """Return the deferred-``uv sync`` marker path under ``state_root``."""
    return state_root / PENDING_SYNC_FILENAME


def self_deploy_failure_state_path(state_root: Path) -> Path:
    """Return the consecutive self-deploy-failure counter path under ``state_root``.

    Tracks how many ``self_deploy`` attempts in a row have failed, so a
    persistent deploy outage can escalate rather than merely repeating the
    same latched digest entry (issue #817 item 5). Sibling of
    :func:`pending_sync_path` -- same directory, same atomic-write contract.
    """
    return state_root / SELF_DEPLOY_FAILURE_STATE_FILENAME


def zero_pass_streak_state_path(state_root: Path) -> Path:
    """Return the consecutive-zero-repo-pass-cycle counter path under ``state_root``.

    Tracks how many fleet-supervisor cycles in a row completed with zero repo
    passes despite at least one repo being configured, so a supervisor that
    keeps restarting and exiting before doing any repo work (issue #855, the
    general shape behind #851) can escalate rather than repeating a silent
    exit-code-0 success forever. Sibling of
    :func:`self_deploy_failure_state_path` -- same directory, same
    atomic-write contract.
    """
    return state_root / ZERO_PASS_STREAK_STATE_FILENAME


def worktrees_dir(state_root: Path) -> Path:
    """Return the worktrees root under ``state_root``.

    Both sides of the create/clean pair must route through this function:
    ``dispatch`` (which creates worktrees) and ``charlie worktree-clean``
    (which removes them). They disagreed before this module existed, which is
    the bug described in the module docstring.
    """
    return state_root / WORKTREES_DIRNAME


def cross_family_dir(state_root: Path) -> Path:
    """Return the cross-family review-artifact dir under ``state_root``."""
    return state_root / CROSS_FAMILY_DIRNAME


def dispatches_dir(state_root: Path) -> Path:
    """Return the dispatch-record root under ``state_root``."""
    return state_root / DISPATCHES_DIRNAME


def sessions_dir_default(state_root: Path) -> Path:
    """Return the default launched-session dir under ``state_root``.

    ``_default`` marks this as the value used when ``devin.sessions_dir`` is
    not explicitly configured. Callers holding a config must honour an explicit
    override in preference to this.
    """
    return dispatches_dir(state_root) / SESSIONS_DIRNAME


def reviews_dir_default(state_root: Path) -> Path:
    """Return the default review-dispatch dir under ``state_root``.

    ``_default`` marks this as the value used when
    ``review_dispatch.reviews_dir`` is not explicitly configured.
    """
    return dispatches_dir(state_root) / REVIEWS_DIRNAME


def session_manifest_default(state_root: Path) -> Path:
    """Return the default session-manifest path under ``state_root``.

    ``_default`` marks this as the value used when ``devin.session_manifest``
    is not explicitly configured.
    """
    return dispatches_dir(state_root) / SESSION_MANIFEST_FILENAME


def session_results_default(state_root: Path) -> Path:
    """Return the default session-results path under ``state_root``.

    ``_default`` marks this as the value used when ``devin.session_results``
    is not explicitly configured.
    """
    return dispatches_dir(state_root) / SESSION_RESULTS_FILENAME


def notify_digest_default(state_root: Path) -> Path:
    """Return the default notification digest path under ``state_root``.

    ``_default`` marks this as the value used when ``notify.file_path`` is not
    explicitly configured.
    """
    return state_root / NOTIFY_DIRNAME / NOTIFY_DIGEST_FILENAME


def resolve_state_child(value: str, *, repo_root: Path, default: Path) -> Path:
    """Resolve a sentinel-style state-child config value.

    Several config fields (``devin.sessions_dir``, ``devin.session_manifest``,
    ``devin.session_results``, ``review_dispatch.reviews_dir``,
    ``notify.file_path``) use an empty string to mean "derive this from
    ``runtime.state_dir``" rather than re-spelling the historical default in
    every config dataclass. An empty *value* therefore returns *default*
    (already an absolute path under the repo's resolved state root, e.g.
    :func:`sessions_dir_default`). A non-empty *value* is an explicit override,
    resolved against *repo_root* the same way config paths always have been
    (an absolute override is returned as-is; a relative one is joined to
    *repo_root*).

    This is the single place that understands the sentinel convention — see
    :func:`charlie_work.paths.resolved_layout`, which calls this once per
    sentinel field instead of re-implementing the "empty means derive" check
    at each of the dozen or so call sites across the package.
    """
    if not value:
        return default
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def gh_config_dir(target_path: Path) -> Path:
    """Return the worktree-local ``gh`` config dir for ``target_path``.

    Intentionally keyed on the *worktree* path and intentionally outside the
    orchestrator state dir: the point is that each worktree gets an isolated
    ``gh`` configuration, so this must not follow ``runtime.state_dir``. It is
    centralised here for discoverability, not to make it configurable.
    """
    return target_path / _VAR_DIRNAME / GH_CONFIG_DIRNAME


# --- fleet-dir layout ------------------------------------------------------

GLOBAL_CONFIG_FILENAME = "config.yaml"
FLEET_REGISTRY_FILENAME = "fleet.json"
FLEET_LOCK_FILENAME = "fleet.lock"
FLEET_SUPERVISOR_LOCK_FILENAME = "fleet-supervisor.lock"
NOTIFY_HEALTH_STATE_FILENAME = "notify_health_state.json"

# The fleet heartbeat state file (``heartbeat-state.json``) also lives in the
# fleet dir, but is deliberately NOT centralised here. Its sole owner is
# ``scripts/heartbeat_check.py``, which is stdlib-only by design so that a
# broken package install cannot break the check that detects it — meaning it
# cannot import this module. Adding a constant here that its only consumer is
# structurally unable to use would create a second spelling with nothing
# enforcing agreement between them, which is the exact drift this module
# exists to prevent. If a future consumer inside the package needs that path,
# add the constant then and update the script to match in the same change.

#: ``*_FILENAME`` constants deliberately *excluded* from Rule 1 enforcement.
#:
#: The enforced filename set is derived automatically by
#: ``tests/test_no_path_literals.py`` -- it sweeps ``layout`` for every
#: module-level attribute whose name ends in ``_FILENAME`` and whose value is
#: a ``str``, then subtracts the names listed here. This is fail-closed: a
#: newly added ``*_FILENAME`` constant receives automatic Rule 1 protection
#: with zero author effort, and the only way to opt *out* is to append the
#: name here -- a deliberate, visible act with a test attached (see
#: ``test_filename_sweep_rederives_from_dir_layout``).
#:
#: ``GLOBAL_CONFIG_FILENAME`` (``config.yaml``) is excluded because it is a
#: bare, generic name that Rule 1 can only match by exact string membership,
#: so it would false-positive on any unrelated ``some_dir / "config.yaml"``
#: and induce the wrong import. Every other ``*_FILENAME`` constant is
#: specific enough that a re-spelling elsewhere is almost always a real
#: divergence hazard.
_FILENAME_EXCLUSIONS = (GLOBAL_CONFIG_FILENAME,)


def global_config_path(override: str | None = None) -> Path:
    """Return the host-wide global config path in the fleet dir.

    Distinct from the per-repo ``orchestrator.config.yaml``
    (``config.DEFAULT_CONFIG_FILENAME``): this is the fleet-wide default layer
    that per-repo config merges over.
    """
    return fleet_dir(override=override) / GLOBAL_CONFIG_FILENAME


def fleet_registry_path(override: str | None = None) -> Path:
    """Return the fleet registry (``fleet.json``) path in the fleet dir."""
    return fleet_dir(override=override) / FLEET_REGISTRY_FILENAME


def fleet_lock_path(override: str | None = None) -> Path:
    """Return the fleet-wide dispatch lock path in the fleet dir."""
    return fleet_dir(override=override) / FLEET_LOCK_FILENAME


def fleet_supervisor_lock_path(override: str | None = None) -> Path:
    """Return the host-wide fleet-supervisor lock path in the fleet dir.

    See :func:`supervisor_lock_path` for the per-repo counterpart.
    """
    return fleet_dir(override=override) / FLEET_SUPERVISOR_LOCK_FILENAME


def notify_health_state_path(override: str | None = None) -> Path:
    """Return the fleet health-notification baseline sidecar path."""
    return fleet_dir(override=override) / NOTIFY_HEALTH_STATE_FILENAME
