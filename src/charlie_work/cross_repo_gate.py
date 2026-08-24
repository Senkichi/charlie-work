"""Pre-flight gate: refuse to dispatch when an issue's referenced code is not in the target repo.

Issue #1010: a dispatched worker edited a sibling repo's shared main checkout
because the issue's subject code (``suite_coverage.py``) does not exist in the
repo it was dispatched against.  The orchestrator created a proper isolated
worktree, but the file the worker was asked to change was not in it — so the
worker went to the sibling repo's shared checkout and worked there,
contaminating another agent's PR.

This module implements the "pre-flight gate" from the issue's proposed fix
(1): at dispatch time, extract file-path references from the issue body and
check whether any of them exist in the target repo.  If the issue references
file paths but *none* of them exist in the repo, the gate returns ``False``
— the caller should escalate to ``agent:human-needed`` with a
``cross_repo_target`` reason instead of burning a worker and a slot.

The gate is conservative by design:

- An issue that references **no** file paths passes (no evidence of a
  cross-repo target).
- An issue where **at least one** referenced path exists in the repo passes
  (the worker has something to work on here).
- An issue with **exactly one** referenced path, that path missing, and the
  path not shaped like a reference to this repo (not a relative path whose
  first segment names a real top-level directory here) also passes — a
  single ambiguous fragment pulled out of prose is weak evidence on its own,
  and escalating on it wastes a human triage action for no reason.
- Otherwise — every referenced path is missing, and either there are
  multiple such paths or the sole path is repo-shaped-but-missing — the
  issue is blocked.

Escalating to ``human-needed`` is a safe failure mode: a human can re-label
the issue after confirming the target repo, so a false positive costs one
manual triage action rather than a contaminated sibling checkout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import LAUNCHER_OWNED_DIRS
from .safe_path import contains

# A file extension: 1-10 word characters after a dot.  Bounds the length so
# the regex does not match version strings like ``1.2.3.4.5.6.7.8.9.0``.
_EXT = r"\.[a-zA-Z][a-zA-Z0-9]{0,9}"

# A relative path with at least one path separator and a file extension.
# Requires at least 2 segments (e.g. ``src/foo.py``, ``ci_fleet/suite_coverage.py``)
# to avoid matching bare filenames like ``main.py`` that may appear in prose
# without being file-path references.
_REL_PATH = rf"(?<![\w/.])((?:[\w.-]+/)+[\w.-]+{_EXT})(?![\w])"

# An absolute path: drive letter (Windows) or leading ``/`` (POSIX), followed
# by path segments and a file extension.
_ABS_WIN_PATH = rf"(?<![\w])([A-Za-z]:[\\/](?:[\w.-]+[\\/])+[\w.-]+{_EXT})(?![\w])"
_ABS_POSIX_PATH = rf"(?<![\w])(/(?:[\w.-]+/)+[\w.-]+{_EXT})(?![\w])"

# Backtick-quoted paths: `` `...ext` `` — catches paths quoted in markdown
# regardless of whether they are relative or absolute.
_TICK_PATH = rf"`([^`]*(?:/|\\)[^`]*{_EXT})`"

_PATH_RE = re.compile(
    "|".join((_TICK_PATH, _ABS_WIN_PATH, _ABS_POSIX_PATH, _REL_PATH)),
)

# A scheme-less domain-shaped token followed by a path, e.g.
# ``pultegroupinc.com/careers/default.aspx``.  This is not a file-path
# reference — it is a URL fragment with the ``https://`` scheme dropped (as
# happens routinely when a URL is pasted into prose or a markdown table
# cell). Mirrors the ``https?://`` strip below: the "host" portion is one or
# more dot-separated labels ending in an alpha-only 2-24 char label (TLD-
# shaped), immediately followed by ``/`` and the rest of the token. Stripped
# as a whole (host + path) before path extraction runs, exactly like the
# ``https?://\S+`` strip removes the scheme *and* its path together. The
# token stops at whitespace and markdown structure characters (``|``
# table-cell delimiters, backticks, closing brackets/parens) so a domain
# token packed tightly against a real path in a table cell —
# ``|domain.com/x.aspx|src/real.py |`` — does not swallow its neighbor.
_DOMAIN_PATH = r"\b(?:[\w-]+\.)+[a-zA-Z]{2,24}/[^\s|`)\]]*"

# A placeholder path segment: a template stand-in that can never name a real
# file. Issue #1343: a templated documentation path in an issue body (e.g. a
# runtime-state example ``<state-dir>/prs/pr-N/review-decision.json``) was
# extracted as a path candidate and, because the runtime state dir is a real
# top-level directory in the checkout, false-positived as a cross-repo
# target. A placeholder segment can never be a genuine cross-repo reference,
# so candidates containing one are dropped before existence checks.
#
# Matches either:
#   - an angle-bracket placeholder (``<state-dir>``, ``<pr-N>``, ``<...>``) —
#     any segment containing ``<`` or ``>``; or
#   - a placeholder-numbered segment (``pr-N``, ``issue-N``) — one or more
#     letters, a dash, then a literal capital ``N`` standing in for an
#     unknown number.
_PLACEHOLDER_SEGMENT = re.compile(r"^(?:[A-Za-z]+-N|.*[<>].*)$")

# Glob metacharacters: ``*``, ``?``, ``[``, ``]``. A candidate containing any
# of these is a glob pattern, not a literal file path — no file literally
# named ``*.py`` exists, so a glob candidate is always "missing" and would
# false-positive the gate. The backtick-quoted path regex (``_TICK_PATH``)
# captures globs because it matches anything inside backticks that contains a
# separator and ends with an extension; the non-tick regexes exclude ``*``
# from their character classes, so only backtick-quoted globs reach this
# filter (issue #1391).
_GLOB_METACHAR = re.compile(r"[*?\[\]]")


@dataclass(frozen=True)
class CrossRepoGateResult:
    """Outcome of the cross-repo pre-flight gate.

    Attributes:
        passed: ``True`` when the issue should be dispatched, ``False`` when
            it should be escalated to ``agent:human-needed``.
        referenced_paths: every candidate file path extracted from the issue
            body, as raw strings (may include paths that do not exist
            anywhere).
        missing_paths: the subset of ``referenced_paths`` that do not exist
            in the target repo.  When ``passed`` is ``False``, this equals
            ``referenced_paths`` (every referenced path was missing).
        reason: human-readable explanation for the gate's decision, suitable
            for an event payload.
    """

    passed: bool
    referenced_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    reason: str


def extract_referenced_paths(issue_body: str) -> list[str]:
    """Extract candidate file-path references from an issue body.

    Returns a de-duplicated list of raw path strings, preserving first-seen
    order.  Paths inside backtick quotes, absolute paths (Windows drive-letter
    or POSIX leading-slash), and relative paths with at least one ``/``
    separator and a file extension are all candidates.

    URLs (``http://``, ``https://``) are excluded — they are not file paths.
    Scheme-less domain-shaped tokens (e.g. a bare ``example.com/path`` pasted
    into prose, with no ``https://`` prefix) are excluded for the same
    reason.

    Candidates containing a placeholder path segment (``<state-dir>``,
    ``pr-N``, ``issue-N``, ``<...>``) are dropped — a template stand-in can
    never name a real file, so a templated documentation path in an issue
    body cannot fire the gate (issue #1343).

    Candidates containing glob metacharacters (``*``, ``?``, ``[``, ``]``)
    are dropped — a glob pattern is not a literal file path, and no file
    literally named ``*.py`` exists, so a glob candidate is always "missing"
    and would false-positive the gate (issue #1391).

    Candidates whose first path segment is a launcher-owned worktree
    directory (``.devin``, ``.git_worktree_dir``) are dropped — these paths
    live only inside agent worktrees, not in the repo tree, so they are not
    evidence of a cross-repo target (issue #1391).
    """
    # Strip URLs before matching so the POSIX absolute-path alternation does
    # not capture the path portion of ``https://example.com/foo.py``.
    stripped = re.sub(r"https?://\S+", "", issue_body)
    # Strip scheme-less domain-shaped tokens the same way — a "path" whose
    # leading segment is actually a hostname is not a file-path reference.
    stripped = re.sub(_DOMAIN_PATH, "", stripped)
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _PATH_RE.finditer(stripped):
        # The regex has four alternation groups; pick the one that matched.
        raw = next((g for g in match.groups() if g is not None), "")
        if not raw:
            continue
        # Drop templated/placeholder paths: a segment like ``pr-N`` or
        # ``<state-dir>`` is documentation template text, not a real file
        # reference, and can never be a genuine cross-repo target.
        if _has_placeholder_segment(raw):
            continue
        # Drop glob patterns: a candidate containing ``*``, ``?``, ``[``, or
        # ``]`` is a glob, not a literal path. No file named ``*.py`` exists,
        # so a glob is always "missing" and would false-positive the gate.
        if _GLOB_METACHAR.search(raw):
            continue
        # Drop launcher-owned worktree paths: paths under ``.devin/`` or
        # ``.git_worktree_dir/`` live only inside agent worktrees, not in the
        # repo tree, so they are not evidence of a cross-repo target.
        if _is_launcher_owned_path(raw):
            continue
        if raw not in seen:
            seen.add(raw)
            candidates.append(raw)
    return candidates


def _has_placeholder_segment(candidate: str) -> bool:
    """Return ``True`` when any segment of ``candidate`` is a template placeholder.

    A placeholder segment (``<state-dir>``, ``pr-N``, ``<...>``) can never
    name a real file — it is documentation template text, not a cross-repo
    reference.  Candidates containing one are dropped before existence
    checks so a templated example path in an issue body cannot fire the gate
    (issue #1343).
    """
    return any(_PLACEHOLDER_SEGMENT.match(seg) for seg in re.split(r"[\\/]+", candidate))


def _is_launcher_owned_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` is under a launcher-owned worktree dir.

    Paths under ``.devin/`` or ``.git_worktree_dir/`` live only inside agent
    worktrees — the shim materializes them on every dispatch — not in the
    repo tree. A candidate whose first path segment names one of these
    directories is not evidence of a cross-repo target: it will always be
    "missing" from the repo and would false-positive the gate (issue #1391).

    The launcher-owned directory set is sourced from
    :data:`charlie_work.config.LAUNCHER_OWNED_DIRS`, shared with
    :mod:`charlie_work.worktree`'s dirty check so the two modules share one
    definition of "launcher-owned, not evidence."
    """
    segments = re.split(r"[\\/]+", candidate, maxsplit=1)
    return bool(segments) and segments[0] in LAUNCHER_OWNED_DIRS


def _path_exists_in_repo(path_str: str, repo_root: Path) -> bool:
    """Return ``True`` when ``path_str`` resolves to an existing file inside ``repo_root``."""
    path = Path(path_str)
    if path.is_absolute():
        try:
            if not path.exists():
                return False
            return contains(repo_root, path)
        except (OSError, ValueError):
            return False
    # Relative path: resolve against the repo root.
    resolved = repo_root / path
    try:
        return resolved.exists()
    except OSError:
        return False


def _top_level_dirs(repo_root: Path) -> set[str]:
    """Return the names of ``repo_root``'s immediate subdirectories.

    Derived dynamically from the live filesystem — never a hardcoded list —
    so the "is this a real repo-relative path" check tracks the repo's
    actual structure.
    """
    try:
        return {entry.name for entry in repo_root.iterdir() if entry.is_dir()}
    except OSError:
        return set()


def _gitignored_top_level_names(repo_root: Path) -> set[str]:
    """Return top-level names ignored by the repo's ``.gitignore``.

    Derived from the repo's ``.gitignore`` — never a hardcoded name list —
    so the "repo-shaped" exclusion tracks the repo's actual ignore policy
    rather than a brittle hand-maintained set.  Issue #1343: the runtime
    state dir (``.var``) is a real top-level directory in the checkout but
    is gitignored; a templated documentation path keyed on its name must
    not count as "repo-shaped" evidence.

    Only simple literal patterns are collected — a bare ``name`` or
    ``name/`` line with no nested slash, no wildcard, and no negation.
    Complex patterns (globs, negations, nested paths) are left to git
    itself and do not participate in this exclusion; they cannot name a
    single top-level directory unambiguously, and over-collecting them
    would risk excluding a real tracked directory.
    """
    gitignore = repo_root / ".gitignore"
    if not gitignore.is_file():
        return set()
    ignored: set[str] = set()
    try:
        text = gitignore.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return set()
    for line in text.splitlines():
        pattern = line.strip()
        if not pattern or pattern.startswith("#") or pattern.startswith("!"):
            continue
        # Drop a trailing slash (directory marker); the bare name is what
        # would appear as the first segment of a relative path.
        name = pattern.rstrip("/")
        # Only collect simple literal top-level names: no slash (nested
        # patterns do not name a top-level dir), no wildcard/bracket sets
        # (they cannot name a single directory unambiguously).
        if "/" in name or "*" in name or "?" in name or "[" in name:
            continue
        ignored.add(name)
    return ignored


def _is_absolute_path(candidate: str) -> bool:
    """Return ``True`` when ``candidate`` is absolute, on any host platform.

    ``Path(candidate).is_absolute()`` alone is platform-dependent in a way
    that matters here: on Windows, ``PureWindowsPath`` only counts a path as
    absolute when it carries a drive letter, so a POSIX-style absolute path
    like ``/home/user/other-repo/foo.py`` reports ``is_absolute() is False``.
    Left unguarded, that misclassifies a genuinely absolute (and genuinely
    outside-the-repo) candidate as "relative", which would let it reach the
    repo-shape check, fail it (its "first segment" is the empty string
    before the leading slash), and incorrectly abstain instead of escalate.
    A leading path separator is unambiguously absolute regardless of host
    platform, so it is treated as absolute here even where ``Path.is_absolute``
    disagrees.
    """
    return Path(candidate).is_absolute() or bool(re.match(r"[\\/]", candidate))


def _is_repo_shaped_relative_candidate(candidate: str, repo_root: Path) -> bool:
    """Return ``True`` when ``candidate`` is a *relative* path whose first
    path segment names a directory that actually exists in ``repo_root`` and
    is not gitignored.

    Absolute candidates are never repo-shaped by this definition — callers
    that need to keep escalating on a missing absolute path must check
    ``_is_absolute_path(candidate)`` themselves before consulting this
    function (see the single-candidate exception in :func:`cross_repo_gate`).

    This is the shape test that distinguishes a genuine (but missing)
    repo-relative reference like ``src/charlie_work/nonexistent.py`` (first
    segment ``src`` is a real top-level directory) from an unrelated
    relative-looking token like ``Scripts/charlie.exe`` (first segment
    ``Scripts`` names no directory in this repo — it is a venv path, not a
    reference to this repo's code).

    A real top-level directory that is *gitignored* (the runtime state dir,
    a venv, build artifacts) does not count as repo-shaped evidence: a path
    keyed on such a name is typically a templated documentation path (e.g. a
    runtime-state example), not a reference to this repo's tracked code
    (issue #1343).  The exclusion is derived from the repo's ``.gitignore``,
    not a hardcoded name list, so it tracks the repo's actual ignore policy.
    """
    if _is_absolute_path(candidate):
        return False
    first_segment = re.split(r"[\\/]+", candidate, maxsplit=1)[0]
    if first_segment not in _top_level_dirs(repo_root):
        return False
    if first_segment in _gitignored_top_level_names(repo_root):
        return False
    return True


def cross_repo_gate(issue_body: str, repo_root: Path) -> CrossRepoGateResult:
    """Decide whether an issue should be dispatched or escalated as cross-repo.

    Returns a :class:`CrossRepoGateResult` with ``passed=True`` when the issue
    is safe to dispatch (it references no file paths, or at least one
    referenced path exists in ``repo_root``), and ``passed=False`` when every
    referenced path is missing from ``repo_root`` — the signal that the
    issue's subject code lives in a different repo.

    A single-candidate exception applies: when extraction found exactly one
    candidate and it is missing, the gate abstains (``passed=True``) unless
    that candidate is a relative path whose first segment names a directory
    that actually exists in ``repo_root`` (see
    :func:`_is_repo_shaped_relative_candidate`). One ambiguous fragment
    pulled out of prose (a venv-relative path, a config key, anything that
    merely *looks* path-shaped) is weak evidence of a cross-repo target on
    its own — escalating on it wastes a human triage action for no reason.
    An absolute path resolving outside the repo is not ambiguous in the same
    way and keeps escalating regardless of this exception.
    """
    referenced = extract_referenced_paths(issue_body)
    if not referenced:
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=(),
            missing_paths=(),
            reason="no file paths referenced in issue body",
        )
    missing = tuple(p for p in referenced if not _path_exists_in_repo(p, repo_root))
    # Block only when EVERY referenced path is absent — the issue's subject
    # code is not in this repo at all.  If even one referenced path exists,
    # the worker has something to work on here and the gate passes.
    if len(missing) < len(referenced):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(referenced),
            missing_paths=missing,
            reason="at least one referenced path exists in the target repo",
        )
    # NOTE: absoluteness is checked via ``_is_absolute_path``, not a bare
    # ``Path(...).is_absolute()`` — the latter is platform-dependent (a
    # POSIX-style absolute path is not "absolute" under Windows' drive-letter
    # rule) and would let a genuinely-absolute, outside-the-repo candidate
    # slip through as "relative" and incorrectly abstain. See
    # ``_is_absolute_path`` for the full rationale.
    if (
        len(referenced) == 1
        and not _is_absolute_path(referenced[0])
        and not _is_repo_shaped_relative_candidate(referenced[0], repo_root)
    ):
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=tuple(referenced),
            missing_paths=missing,
            reason=(
                "single ambiguous candidate "
                f"({referenced[0]!r}) is not a repo-shaped relative path — "
                "abstaining rather than escalating on weak evidence"
            ),
        )
    # Every referenced path is missing — the issue's subject code is not in
    # this repo.  Escalate instead of dispatching a worker that will wander.
    return CrossRepoGateResult(
        passed=False,
        referenced_paths=tuple(referenced),
        missing_paths=missing,
        reason=(
            f"cross_repo_target: all {len(missing)} referenced file path(s) "
            f"are absent from the target repo ({repo_root})"
        ),
    )


def cross_repo_scope_gate(
    issue_title: str,
    issue_body: str,
    dispatching_repo_name: str,
    managed_repo_names: frozenset[str],
) -> CrossRepoGateResult:
    """Decide whether an issue's *scope* targets another managed repo.

    This is the intake-time repo attribution from issue #1244 (Option 1):
    when an issue's deliverables live in a different managed repo, the lane
    that dispatches it can never finalize it — the worker either hops to
    the sibling repo's worktree (contaminating it) or exits with zero
    artifacts in the dispatching repo, and the single-remote orphan sweep
    declares it dead and redispatches, looping forever.

    The gate checks the issue **title** for a ``<repo-name>:`` prefix that
    names a managed repo other than the dispatching one.  This is the
    clearest signal that the issue's scope is that repo — the #709 case
    study had the title ``job-cannon: docs/devin-orchestration/ ... stale``
    in the charlie-work tracker, but every deliverable was a job-cannon
    file.

    The managed-repo set **must** derive from the fleet registry (see
    :func:`charlie_work.fleet_registry.managed_repo_names`), never a
    hardcoded list — a literal list would require manual updates for every
    fleet change and break silently when a repo is added or removed.

    Conservative by design (mirrors the file-path gate's philosophy):

    - An empty managed-repo set (no fleet registry, single-repo) passes —
      no other repos to attribute the issue to.
    - An issue whose title does not start with ``<other-repo>:`` passes —
      the scope is not unambiguously another repo.
    - The dispatching repo's own name is excluded — an issue in
      charlie-work that says ``charlie-work: fix X`` is in-repo by
      definition.
    - Only a title that starts with ``<repo-name>:`` for a managed repo
      *other than* the dispatching one is blocked.  The body is not
      scanned for repo-name mentions because a passing reference
      (``coordinate with job-cannon on this``) is not evidence of a
      cross-repo scope.

    Args:
        issue_title: The issue's title string.
        issue_body: The issue's body string (unused by the current
            title-prefix check, but accepted so callers can extend the
            detection without changing the call site).
        dispatching_repo_name: The repo-name segment of the dispatching
            repo (e.g. ``charlie-work``), extracted from the GitHub
            ``nameWithOwner`` or the repo root's directory name.
        managed_repo_names: The set of repo-name segments managed by the
            fleet, from :func:`fleet_registry.managed_repo_names`.

    Returns:
        A :class:`CrossRepoGateResult` with ``passed=False`` when the
        issue title names another managed repo, ``passed=True`` otherwise.
    """
    other_repos = managed_repo_names - {dispatching_repo_name}
    if not other_repos:
        return CrossRepoGateResult(
            passed=True,
            referenced_paths=(),
            missing_paths=(),
            reason="no other managed repos in the fleet registry",
        )
    title_lower = issue_title.lower().lstrip()
    for repo_name in sorted(other_repos):
        # Match "repo-name:" at the start of the title (case-insensitive).
        # This is the "job-cannon: docs/..." pattern from #709 — the
        # clearest signal that the issue's scope is that repo, not this
        # one.  A colon immediately after the repo name is the convention
        # for scope-prefixed issue titles in this fleet.
        prefix = f"{repo_name.lower()}:"
        if title_lower.startswith(prefix):
            return CrossRepoGateResult(
                passed=False,
                referenced_paths=(),
                missing_paths=(),
                reason=(
                    f"cross_repo_scope: issue title starts with "
                    f"{repo_name!r}: — the issue's deliverables target "
                    f"{repo_name}, not the dispatching repo "
                    f"({dispatching_repo_name})"
                ),
            )
    return CrossRepoGateResult(
        passed=True,
        referenced_paths=(),
        missing_paths=(),
        reason="issue title does not name another managed repo",
    )


__all__ = [
    "CrossRepoGateResult",
    "cross_repo_gate",
    "cross_repo_scope_gate",
    "extract_referenced_paths",
]
