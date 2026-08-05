from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _paths_equal(a: Path, b: Path) -> bool:
    """Compare two paths for equality using the OS's own normalization.

    **Not** a case-sensitivity guard, despite what this docstring said until
    #899. ``PurePath.__eq__`` already applies flavour-appropriate case folding,
    and already normalizes separators, so for two ``Path`` inputs this is
    exactly equivalent to ``a == b``::

        PureWindowsPath("C:/Foo") == PureWindowsPath("c:/foo")   # True
        PurePosixPath("/Foo")     == PurePosixPath("/foo")       # False
        Path("C:/Foo")            == Path("C:\\\\Foo")             # True

    The original claim was never covered by a test, which is how it survived;
    ci_fleet's vendored copy caught it when a mutation replacing the body with
    ``a == b`` left all ten of its tests green.

    Kept, with the real rationale: the ``Path`` annotation is not enforced at
    runtime, and a ``str`` on either side makes ``==`` fail outright —
    ``"C:/Foo" == Path("C:/Foo")`` is ``False``, which the probe would read as
    a genuine literal-vs-resolved divergence and report as virtualization.
    ``os.fspath`` + ``normcase`` collapses that case. Deliberately not narrowed
    to ``a == b``: the guard costs nothing and the failure it prevents is a
    false positive on a security-adjacent probe (#624).
    """
    return os.path.normcase(os.fspath(a)) == os.path.normcase(os.fspath(b))


def detect_path_virtualization(literal: Path) -> tuple[Path, Path] | None:
    """Detect per-process virtualization of a path (issue #624).

    MSIX/container copy-on-write redirection sits *below* the environment
    variable: the literal path string is identical in both the container and
    the host, but ``Path.resolve()`` follows the reparse point to the
    redirected location. Reads pass through to the real file; the first write
    forks a private copy that daemons reading the same path string never see.

    The load-bearing signal is that the literal path and its resolved form
    disagree — never a hardcoded package moniker (which would rot on the next
    app update and only cover one container). Returns ``(literal, resolved)``
    when they diverge, else ``None``.

    Resolution errors are treated as "not virtualized" rather than raising: a
    probe that crashes the caller is worse than one that stays silent on an
    unreadable path. ``Path.resolve(strict=False)`` (the default) does not
    require the path to exist, so an absent fleet dir still resolves cleanly
    and only a real reparse-point divergence fires.
    """
    try:
        resolved = literal.resolve()
    except OSError:
        return None
    if not _paths_equal(resolved, literal):
        return (literal, resolved)
    return None


def fleet_dir_virtualization(*, override: str | None = None) -> tuple[Path, Path] | None:
    """Detect per-process virtualization of the fleet directory (issue #624).

    Thin wrapper over :func:`detect_path_virtualization` keyed on the fleet
    dir. Repo-agnostic by construction: the fleet dir is a host-wide
    per-process property, not a project layout, so the same probe fires for
    every registered repo on this host.
    """
    return detect_path_virtualization(fleet_dir(override=override))


def warn_fleet_dir_virtualization_on_write(literal: Path, *, context: str) -> None:
    """Log a warning when a host-wide write lands in a virtualized copy.

    Called from fleet-dir state writers so "I deployed it" cannot be reported
    when the write forked a private copy that daemons reading the same path
    string will never see (issue #624; this is the exact shape of the #590
    failure). Never raises and never blocks the write — the operator already
    asked for it; this only names where it actually landed.
    """
    diverged = detect_path_virtualization(literal)
    if diverged is None:
        return
    _literal, resolved = diverged
    logger.warning(
        "Fleet dir virtualization detected while %s: %s resolves to %s — "
        "this write lands in a private copy invisible to scheduled tasks and "
        "daemons reading the same path string (issue #624; see also #590). "
        "Daemon-visible state must be written via a non-redirected route "
        "(e.g. a UNC path).",
        context,
        literal,
        resolved,
    )


def fleet_dir(*, override: str | None = None) -> Path:
    """Return the user-level fleet directory for charlie-work.

    Platform-specific defaults:
    - Windows (win32): %LOCALAPPDATA%\\charlie-work\\
    - POSIX: ${XDG_STATE_HOME:-~/.local/state}/charlie-work/

    The override parameter (or CHARLIE_WORK_FLEET_DIR env var) allows
    test isolation without hardcoding platform-specific paths in fixtures.

    Args:
        override: Optional path string to use instead of the platform default.
                  If None, checks CHARLIE_WORK_FLEET_DIR env var, then uses
                  the platform default.

    Returns:
        Path to the fleet directory.
    """
    if override is not None:
        return Path(override)

    env_override = os.environ.get("CHARLIE_WORK_FLEET_DIR")
    if env_override:
        return Path(env_override)

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))

    return base / "charlie-work"
