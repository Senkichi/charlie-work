from __future__ import annotations

import logging
import os
import sys
import time
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


_ADMIN_SHARE_WARNED: set[str] = set()

# The per-file probe below is cached per process, keyed on the normalized
# literal dir, until this TTL expires: per-file MSIX detection is a directory
# scan plus one ``Path.resolve()`` per file, which this host measured at
# 13-32 ms against a ~95-file fleet dir (and seconds against the UNC route),
# and the callers (:func:`_route_around_fork` and
# :func:`warn_fleet_dir_virtualization_on_write`) run on every fleet read and
# every fleet write. A deliberately short TTL keeps mid-session fork
# detection: a fork written after the cached scan is noticed at the next TTL
# refresh. :func:`fleet_dir_virtualization` (the ``charlie doctor`` probe)
# deliberately bypasses this cache and reads fresh.
_FLEET_PROBE_CACHE_TTL_SECONDS = 30.0
_FLEET_PROBE_CACHE: dict[str, tuple[float, list[tuple[Path, Path]]]] = {}
_MONOTONIC = time.monotonic


def _fleet_file_findings(literal: Path) -> list[tuple[Path, Path]]:
    """Probe every file present in ``literal``; return the diverging pairs.

    MSIX copy-on-write redirection is **per file** (#2806): a packaged
    process forks a private copy of a file the first time it writes it, and
    reads of the same literal path then return the fork, but the *directory*
    itself is never redirected, so probing the directory always finds
    nothing. Iterate the directory instead of keeping a list of names, so a
    fork of any file present is caught regardless of its name.
    """
    try:
        entries = list(literal.iterdir())
    except OSError:
        return []
    findings: list[tuple[Path, Path]] = []
    for entry in entries:
        try:
            if not entry.is_file():
                continue
        except OSError:
            continue
        diverged = detect_path_virtualization(entry)
        if diverged is not None:
            findings.append(diverged)
    return findings


def _fleet_file_findings_cached(literal: Path) -> list[tuple[Path, Path]]:
    """TTL-cached :func:`_fleet_file_findings`, keyed on the normalized dir.

    Shared by :func:`_route_around_fork` and the write guard, which run per
    read/per write and therefore cannot afford a fresh directory scan per
    call. :func:`fleet_dir_virtualization` (``charlie doctor``) calls
    :func:`_fleet_file_findings` directly so the doctor always reports fresh
    state.
    """
    key = os.path.normcase(os.path.abspath(str(literal)))
    now = _MONOTONIC()
    cached = _FLEET_PROBE_CACHE.get(key)
    if cached is not None and (now - cached[0]) < _FLEET_PROBE_CACHE_TTL_SECONDS:
        return cached[1]
    findings = _fleet_file_findings(literal)
    _FLEET_PROBE_CACHE[key] = (now, findings)
    return findings


def _default_fleet_dir_literal() -> Path:
    """The platform-default fleet dir, without routing (see fleet_dir)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "charlie-work"


def fleet_dir_virtualization(
    *, override: str | None = None
) -> tuple[Path, Path] | list[tuple[Path, Path]] | None:
    r"""Detect per-process virtualization of the fleet directory (issue #624).

    Repo-agnostic by construction: the fleet dir is a host-wide per-process
    property, not a project layout, so the same probe fires for every
    registered repo on this host.

    Returns ``None`` when nothing diverges, the ``(literal, resolved)`` pair
    when the directory itself diverges (the unchanged #624 contract), or a
    list of ``(literal, resolved)`` pairs for every diverging file present
    in the directory. Since #2806 the probe covers files, not just the
    directory: MSIX redirection forks individual files, so the directory
    resolving to itself proves nothing. Per-file findings are reported as a
    list — there can be more than one, and ``charlie doctor`` names each
    one. Every file present in the literal dir is probed -- never a
    hardcoded name list, which would rot with the next file the dir gains.

    Keyed on the literal directory even after :func:`fleet_dir` learned to
    route around a fork (#2806): the routed path reads the real registry, so
    probing it would find nothing — accuracy here depends on reporting the
    situation itself.
    """
    if override is not None:
        literal = Path(override)
    else:
        env_override = os.environ.get("CHARLIE_WORK_FLEET_DIR")
        literal = Path(env_override) if env_override else _default_fleet_dir_literal()
    dir_diverged = detect_path_virtualization(literal)
    if dir_diverged is not None:
        return dir_diverged
    return _fleet_file_findings(literal) or None


def _administrative_share_form(literal: Path) -> Path | None:
    """The ``\\\\localhost\\<drive>$\\<rest>`` form of a local directory.

    That administrative share bypasses MSIX per-file redirection (redirection
    applies to paths under the virtualized APPDATA root, not to a UNC route
    to the same volume), so it is the only way a packaged session can read
    and write the daemon-visible copies of forked files. Returns ``None``
    when ``literal`` is not a local drive-letter directory (already a UNC
    path, no drive letter, or a drive root itself).
    """
    drive = literal.drive
    if len(drive) != 2 or not drive.endswith(":"):
        return None
    try:
        rest = literal.relative_to(literal.anchor)
    except ValueError:
        return None
    if not rest.parts:
        return None
    return Path("\\\\localhost\\", drive[0] + "$", *rest.parts)


def _route_around_fork(literal: Path) -> Path:
    """Route host-wide reads and writes around a per-file MSIX fork (#2806).

    When any file under the literal fleet dir is virtualized, a packaged
    session reading the literal path gets its private forked copies while
    daemons and scheduled tasks (which are never packaged) get the real
    files, and the two silently diverge. Return the administrative-share
    route instead when it exists, so every ``fleet_dir() / NAME`` reader in
    :mod:`charlie_work.layout` lands on the daemon-visible copy. When the
    route does not exist on this host, keep the literal path (preserving the
    #624 behavior) and rely on the warning below to surface the situation.

    Never routes an explicit ``override`` or the ``CHARLIE_WORK_FLEET_DIR``
    env var -- callers that pin a path mean it. Never raises: the probe is
    best-effort and any failure degrades to returning ``literal``.

    The per-file probe behind the routing decision is cached per process for
    ``_FLEET_PROBE_CACHE_TTL_SECONDS``, keyed on the literal dir: a fork
    written after the cached scan is picked up at the next TTL refresh rather
    than on every call.
    """
    findings = _fleet_file_findings_cached(literal)
    if not findings:
        return literal
    admin = _administrative_share_form(literal)
    route: Path | None = None
    if admin is not None:
        try:
            if admin.exists():
                route = admin
        except OSError:
            route = None
    names = ", ".join(pair[0].name for pair in findings)
    # Once per process per literal dir: daemons log this once, not per read.
    if str(literal) not in _ADMIN_SHARE_WARNED:
        _ADMIN_SHARE_WARNED.add(str(literal))
        if route is not None:
            logger.warning(
                "Fleet dir per-file virtualization detected (%s resolve elsewhere "
                "in %s); routing host-wide fleet state to the non-redirected "
                "route %s instead (issue #2806).",
                names,
                literal,
                route,
            )
        else:
            logger.warning(
                "Fleet dir per-file virtualization detected (%s resolve elsewhere "
                "in %s), but the non-redirected route does not exist; keeping the "
                "literal path — reads and writes hit the private fork and will "
                "diverge from scheduled tasks and daemons (issue #2806).",
                names,
                literal,
            )
    result = route if route is not None else literal
    return result


def warn_fleet_dir_virtualization_on_write(literal: Path, *, context: str) -> None:
    """Log a warning when a host-wide write lands in a virtualized copy.

    Called from fleet-dir state writers so "I deployed it" cannot be reported
    when the write forked a private copy that daemons reading the same path
    string will never see (issue #624; this is the exact shape of the #590
    failure). Never raises and never blocks the write — the operator already
    asked for it; this only names where it actually landed.

    Since #2806 the probe covers files, not just the given path: MSIX
    redirection forks individual files and leaves the directory resolving to
    itself, so a directory-level probe passes while the file the write is
    about to replace is already forked. Callers that pass the target's
    parent directory get a per-file probe of that directory; a caller that
    passes the file itself is covered by the direct probe.

    The directory-branch probe is TTL-cached (:func:`_fleet_file_findings_cached`)
    — this guard runs on every fleet write — and skipped entirely when the
    directory is already a UNC/routed path (its drive starts with ``\\``):
    the administrative-share route is non-redirected by construction, so the
    scan (which costs seconds on a UNC path) would always find nothing.

    ``Path.is_dir`` re-raises non-ENOENT ``OSError``s on Python >= 3.11 (for
    example a permission error or a bad reparse point), so the check is
    wrapped: a guard that crashed the writer would contradict the "never
    raises" contract, since :mod:`charlie_work.fleet_dispatch` and the other
    writers below call it unwrapped around the real write.
    """
    diverged = detect_path_virtualization(literal)
    if diverged is not None:
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
        return
    try:
        if not literal.is_dir():
            return
    except OSError:
        return
    if literal.drive.startswith("\\"):
        return
    for file_literal, resolved in _fleet_file_findings_cached(literal):
        logger.warning(
            "Fleet dir virtualization detected while %s: %s resolves to %s — "
            "a per-file MSIX fork (#2806): reading this path returns the private "
            "copy while scheduled tasks and daemons get the real file. "
            "Daemon-visible state must be written via a non-redirected route "
            "(e.g. a UNC path).",
            context,
            file_literal,
            resolved,
        )


def fleet_dir(*, override: str | None = None) -> Path:
    """Return the user-level fleet directory for charlie-work.

    Platform-specific defaults:
    - Windows (win32): %LOCALAPPDATA%\\charlie-work\\
    - POSIX: ${XDG_STATE_HOME:-~/.local/state}/charlie-work/

    The override parameter (or CHARLIE_WORK_FLEET_DIR env var) allows
    test isolation without hardcoding platform-specific paths in fixtures.

    Since #2806, the win32 default fleet dir is probed for per-file MSIX
    virtualization (copy-on-write redirection forks individual files; the
    directory itself is never redirected). When a file is forked, the
    returned directory is the non-redirected administrative-share route to
    the same location (``\\\\localhost\\<drive>$\\...``) if it exists, so
    host-wide state reads and writes reach the daemon-visible copies; when
    the route does not exist, the literal path is returned unchanged (the
    #624 behavior) and the condition is logged as a warning. Overrides and
    the env var are returned unchanged — the caller pinned that path. The
    probe is keyed on the literal default dir and cached for
    ``_FLEET_PROBE_CACHE_TTL_SECONDS`` per process (see
    :func:`_route_around_fork`), so a fork written mid-session is detected
    at the next TTL refresh, not only at process start; scheduled tasks,
    daemons, and unpackaged terminals are unaffected, since their fleet dir
    has no virtualized files and the probe finds nothing.

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
        return _route_around_fork(_default_fleet_dir_literal())

    return _default_fleet_dir_literal()
