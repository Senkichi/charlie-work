"""Git-write containment for the test suite (issue #2060).

Incident: a ``git config user.email`` run with ``cwd`` inside a directory
that is not itself a repository ascends into the enclosing checkout; when the
checkout is a linked worktree that ``--local`` write lands on the *shared*
``<git-common-dir>/config`` of the main repo. The observed leak was
``user.email=test@example.test`` into the live checkout's shared config,
which the ``git_identity`` preflight (#1950) then refused until the key was
unset by hand. Fleet workers hit it because ``env_sanitize`` (#1767) points
worker ``TMP``/``TEMP``/``TMPDIR`` at a worktree-local directory, so every
pytest ``tmp_path`` sits inside a linked worktree and any ``git init`` /
``git worktree add`` / ``git clone`` that fails (or is skipped) leaves a
``cwd`` whose parent chain ends in that worktree.

This module supplies the machinery ``tests/conftest.py`` wires in, in three
layers:

* :func:`install_session_git_isolation` — the structural fix, applied from
  ``pytest_configure`` so collection-time and module/session-scoped fixture
  subprocesses are covered too: ``GIT_CEILING_DIRECTORIES`` is rebuilt from
  ``os.path.realpath`` of the basetemp and every parent above it (a git
  discovery walk starting anywhere under the temp root stops at the nearest
  listed ancestor and can never reach an enclosing checkout — verified on
  git-for-windows 2.45: a ceiling directory itself is not examined), and
  ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` point at per-session *copies*
  of the ambient files so ``--global``/``--system`` writes land on a
  throwaway file while reads still see the operator's real values.
* :func:`enclosing_repo_config_path` — resolves, once at session start, the
  shared config file of whatever repository encloses the pytest invocation
  cwd; the conftest's per-test guard fixture byte-compares it before and
  after every test so a mutation is attributed to the offending test and
  restored instead of silently persisting.
* :func:`assert_repo_scoped_config` — the call-site belt, wired into the
  shared ``_git`` runner in ``tests/_worktree_fixtures.py``: a repo-scoped
  ``git config`` mutation raises unless ``cwd`` resolves to the very
  repository anchor (worktree toplevel or bare gitdir) the write would land
  on, so the enclosing repo's config is unreachable even where no ceiling
  applies.

:data:`GIT_ISOLATION_ENV_VARS` names the variables the session layer
installs; :func:`protective_git_env` re-exports them for the handful of test
helpers that deliberately scrub ``GIT_*`` from subprocess environments (an
ambient ``GIT_DIR`` redirect is the hazard those scrubs exist for, but the
scrub must not also strip the containment that keeps a stray ``git config``
inside the sandbox).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

#: Environment variables the session isolation layer owns. Tests that
#: rebuild a subprocess env by dropping ``GIT_*`` keys must preserve these --
#: they are the containment, not the redirect hazard the scrub targets.
GIT_ISOLATION_ENV_VARS = (
    "GIT_CEILING_DIRECTORIES",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_SYSTEM",
    "GIT_CONFIG_NOSYSTEM",
)


def _canonical(path: Path | str) -> str:
    """Resolved, case-normalized spelling for path comparison on any OS."""
    return os.path.normcase(os.path.realpath(str(path)))


def ceiling_directories(*roots: Path | str) -> list[str]:
    """``realpath`` of each root plus every ancestor, dedup'd, in order.

    ``GIT_CEILING_DIRECTORIES`` entries are directories git will not ascend
    into during repository discovery — and the ceiling entry itself is not
    examined either. Listing each root *and* all of its parents means a cwd
    anywhere under a temp root stops ascending at the nearest listed
    ancestor: tmp_path cases stop at the basetemp entry, ``tempfile``-created
    siblings of basetemp stop at the temp root, and nothing below the temp
    root can ever reach an enclosing checkout. ``realpath`` is required so a
    junction/8.3-spelled basetemp resolves to the same spelling git computes
    while ascending (the pre-#2060 function-scoped fixture listed raw
    spellings, which can silently fail to match).
    """
    entries: list[str] = []
    for root in roots:
        resolved = Path(os.path.realpath(str(root)))
        for directory in (resolved, *resolved.parents):
            spelling = str(directory)
            if spelling not in entries:
                entries.append(spelling)
    return entries


def merge_ceiling_directories(*roots: Path | str) -> str:
    """Ceiling value for ``roots`` merged over the ambient variable.

    Ambient entries (e.g. the worker-sandbox ceiling ``env_sanitize`` sets,
    or an operator-exported value) are preserved — ours are added, never
    replacing a ceiling somebody else installed.
    """
    entries = ceiling_directories(*roots)
    for existing in os.environ.get("GIT_CEILING_DIRECTORIES", "").split(os.pathsep):
        if existing and existing not in entries:
            entries.append(existing)
    return os.pathsep.join(entries)


def protective_git_env() -> dict[str, str]:
    """The installed isolation variables, for GIT_*-scrubbing test helpers."""
    return {name: os.environ[name] for name in GIT_ISOLATION_ENV_VARS if name in os.environ}


def _global_config_content() -> bytes:
    """Content to seed the session ``GIT_CONFIG_GLOBAL`` file with.

    A copy of the ambient global config — the file an ambient
    ``GIT_CONFIG_GLOBAL`` already names, else ``$XDG_CONFIG_HOME/git/config``
    and ``~/.gitconfig`` concatenated in git's own read order (later wins) —
    so tests that read a global identity see exactly what they saw before the
    redirect. Writes land on the copy, never on the operator's files.
    """
    ambient = os.environ.get("GIT_CONFIG_GLOBAL")
    if ambient:
        source = Path(ambient)
        return source.read_bytes() if source.is_file() else b""
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    parts = [
        candidate.read_bytes()
        for candidate in (xdg / "git" / "config", Path.home() / ".gitconfig")
        if candidate.is_file()
    ]
    return b"\n".join(parts)


def _system_config_content() -> bytes:
    """Content to seed the session ``GIT_CONFIG_SYSTEM`` file with.

    Same copy-don't-share approach as the global file; the real system config
    path is discovered by asking git itself (``--show-origin``), which fails
    soft to an empty file when there is no system config.
    """
    ambient = os.environ.get("GIT_CONFIG_SYSTEM")
    if ambient:
        source = Path(ambient)
        return source.read_bytes() if source.is_file() else b""
    result = subprocess.run(
        ["git", "config", "--system", "--list", "--show-origin"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.startswith("file:"):
        source = Path(result.stdout.split("\t", 1)[0].removeprefix("file:"))
        if source.is_file():
            return source.read_bytes()
    return b""


def install_session_git_isolation(*, basetemp: Path | None = None) -> Path:
    """Point the session's git config files and discovery ceiling at temp space.

    Creates a fresh scratch dir under ``tempfile.gettempdir()`` holding the
    ``GIT_CONFIG_GLOBAL``/``GIT_CONFIG_SYSTEM`` copies, sets
    ``GIT_CEILING_DIRECTORIES`` from the realpath'd ``basetemp`` (or the temp
    root alone when no explicit basetemp was given — the default basetemp
    lives under it anyway) merged over the ambient value, and returns the
    scratch dir so the caller can remove it at session end. Must run before
    any other ``GIT_CONFIG_*`` variable is sampled: the ambient values are
    read first precisely because they are about to be shadowed.
    """
    global_config = _global_config_content()
    system_config = _system_config_content()
    scratch = Path(tempfile.mkdtemp(prefix="pytest-git-isolation-"))
    global_file = scratch / "global.gitconfig"
    system_file = scratch / "system.gitconfig"
    global_file.write_bytes(global_config)
    system_file.write_bytes(system_config)
    os.environ["GIT_CONFIG_GLOBAL"] = str(global_file)
    os.environ["GIT_CONFIG_SYSTEM"] = str(system_file)
    roots: list[Path | str] = [basetemp] if basetemp is not None else []
    roots.append(tempfile.gettempdir())
    os.environ["GIT_CEILING_DIRECTORIES"] = merge_ceiling_directories(*roots)
    return scratch


def enclosing_repo_config_path(cwd: Path | None = None) -> Path | None:
    """The shared ``config`` file of the repo enclosing *cwd* — or None.

    ``--git-common-dir`` is the linked-worktree-aware answer: from a linked
    worktree it resolves to the main checkout's ``.git``, which is exactly
    the shared file a stray ``git config --local`` writes through the
    worktree's ``.git`` file. ``None`` when the session was not launched from
    inside a repository (nothing to watch).
    """
    command = ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    common_dir = result.stdout.strip()
    if not common_dir:
        return None
    return Path(os.path.realpath(common_dir)) / "config"


# ---------------------------------------------------------------------------
# Repo-scoped `git config` write guard for the shared `_git` runner
# ---------------------------------------------------------------------------

# Read modes mutate nothing, so they are exempt from the anchor check — a
# `--get` from a subdirectory is a legitimate, harmless lookup.
_CONFIG_READ_FLAGS = frozenset(
    {"--get", "--get-all", "--get-regexp", "--get-urlmatch", "--list", "-l"}
)

_CONFIG_WRITE_FLAGS = frozenset(
    {
        "--add",
        "--append",
        "--replace-all",
        "--unset",
        "--unset-all",
        "--rename-section",
        "--remove-section",
        "--edit",
        "-e",
    }
)

# Scopes that do not consult repository discovery at all. `--local` and
# `--worktree` are deliberately absent: they name exactly the repo-bound
# scopes the guard exists to contain.
_CONFIG_EXTERNAL_SCOPE_FLAGS = frozenset({"--global", "--system", "--file", "-f", "--blob"})

# Flags that consume the following argv element as their value (only when
# spelled without ``=``).
_CONFIG_VALUE_FLAGS = frozenset(
    {"--file", "-f", "--blob", "--type", "--value", "--default", "--comment"}
)


def _config_mutation_targets_repo(args: Sequence[str]) -> bool:
    """True when ``git config <args>`` performs a repository-scoped write.

    External scopes (``--global``/``--system``/``--file``/``--blob``) never
    consult repo discovery. Read modes mutate nothing — including the implied
    read of a bare ``git config <name>`` (single positional). Everything else
    — explicit write flags, section operations, a ``<name> <value>`` pair, or
    any argv shape this parser does not classify — is treated as a repo-bound
    write: fail-closed.
    """
    positionals = 0
    external_scope = False
    read = False
    write = False
    i = 0
    while i < len(args):
        arg = args[i]
        key = arg.split("=", 1)[0]
        if arg.startswith("-"):
            external_scope |= key in _CONFIG_EXTERNAL_SCOPE_FLAGS
            read |= key in _CONFIG_READ_FLAGS
            write |= key in _CONFIG_WRITE_FLAGS
            if key in _CONFIG_VALUE_FLAGS and "=" not in arg:
                i += 1  # skip the flag's separate value argument
        else:
            positionals += 1
        i += 1
    if external_scope:
        return False
    return write or (not read and positionals >= 2)


def _repo_anchor(cwd: Path) -> Path | None:
    """The repository root (or bare gitdir) discovery resolves from *cwd*.

    ``--show-toplevel`` covers work trees (including linked worktrees, which
    report their own path); bare repositories have no work tree, so they fall
    back to ``--absolute-git-dir`` — the gitdir itself, which for a bare repo
    root equals the directory the caller passed. ``None`` when no repository
    contains *cwd*.
    """
    top = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if top.returncode == 0 and top.stdout.strip():
        return Path(top.stdout.strip())
    bare = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--is-bare-repository"],
        capture_output=True,
        text=True,
    )
    if bare.returncode == 0 and bare.stdout.strip() == "true":
        gitdir = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
        )
        if gitdir.returncode == 0 and gitdir.stdout.strip():
            return Path(gitdir.stdout.strip())
    return None


def assert_repo_scoped_config(cwd: Path, argv: Iterable[str]) -> None:
    """Refuse a ``git config`` mutation whose target repo is not rooted at *cwd*.

    No-op for non-``config`` commands and for calls that cannot write a
    repository's config (external scopes, reads). For a repo-scoped write the
    repository that discovery resolves from *cwd* must be anchored exactly at
    *cwd* — otherwise the call would mutate a repo the caller never named,
    which is precisely how a non-repo temp dir inside a linked worktree
    writes the shared ``.git/config`` of the enclosing checkout (#2060).
    Raises ``AssertionError`` (a broken test premise, not a git failure) with
    the resolved anchor named so the failure attributes itself.
    """
    args = list(argv)
    if not args or args[0] != "config" or not _config_mutation_targets_repo(args[1:]):
        return
    anchor = _repo_anchor(cwd)
    resolved_cwd = _canonical(cwd)
    if anchor is not None and _canonical(anchor) == resolved_cwd:
        return
    resolved = f"the repo rooted at {anchor}" if anchor is not None else "no repository at all"
    raise AssertionError(
        f"refusing `git {' '.join(args)}` from {cwd}: repo discovery resolves "
        f"to {resolved}, not to that directory — a repo-scoped config write "
        "here would mutate an enclosing repo's config (issue #2060)"
    )


__all__ = [
    "GIT_ISOLATION_ENV_VARS",
    "assert_repo_scoped_config",
    "ceiling_directories",
    "enclosing_repo_config_path",
    "install_session_git_isolation",
    "merge_ceiling_directories",
    "protective_git_env",
]
