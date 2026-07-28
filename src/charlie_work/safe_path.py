"""Single point of enforcement for path-containment checks.

Consolidates the ad hoc ``.resolve()`` + ``is_relative_to()`` pairs previously
duplicated across five independent call sites (``runner_slots.py``,
``worktree.py`` x3, ``supervise.py``) with no single source of truth for what
"safely contained" means. None of them resolved symlinks/junctions
defensively on *both* sides before comparing, and one (``worktree.py``'s
``_materialize_directory``) compared unresolved paths, which does not detect
a `..` segment escaping the base directory (``Path.is_relative_to`` is a
purely lexical part-prefix check; it does not collapse ``..``).

Ported from GSD's ``bin/lib/security.cjs`` (``validatePath``/
``requireSafePath``) during the 2026-07-27 GSD sunset -- the design, not the
file (Node fs API -> pathlib). See ``~/.claude/GSD-SUNSET-SALVAGE.md`` item 8.
"""

from __future__ import annotations

from pathlib import Path


def _resolve_defensive(path: Path) -> Path:
    """Resolve ``path``, collapsing symlinks/junctions on every existing segment.

    ``Path.resolve()`` does not require ``path`` to exist: it resolves every
    existing leading segment (following symlinks/junctions along the way) and
    leaves only trailing nonexistent components as literal names. That already
    matches GSD's ``realpathSync``-with-existing-parent-fallback design, so a
    candidate path that hasn't been created yet can still be checked for
    containment before it's materialized.
    """
    return path.resolve()


def contains(base: Path, candidate: Path) -> bool:
    """Return ``True`` when ``candidate`` resolves to a path inside ``base``.

    Both sides are resolved before comparing, so this catches a reparse point
    or symlink that makes ``candidate`` *look* contained lexically but
    actually escapes ``base`` on disk (the CLAUDE.md-declared ``managed_root``
    invariant this guards). Use at any site that can simply skip a
    non-contained entry (e.g. a discovery loop); use ``require_contained``
    where skipping isn't an option.
    """
    resolved_base = _resolve_defensive(base)
    resolved_candidate = _resolve_defensive(candidate)
    return resolved_candidate == resolved_base or resolved_candidate.is_relative_to(
        resolved_base
    )


def require_contained(base: Path, candidate: Path, *, context: str) -> Path:
    """Return the resolved ``candidate``, raising ``ValueError`` if it escapes ``base``.

    For boundaries where an externally-influenced path (config, JSON state, a
    subprocess's stdout) is about to be read from or written to, and silently
    skipping would hide the problem rather than surface it. ``context`` is
    included in the error message to identify the call site without needing a
    traceback.
    """
    resolved_base = _resolve_defensive(base)
    resolved_candidate = _resolve_defensive(candidate)
    if resolved_candidate != resolved_base and not resolved_candidate.is_relative_to(
        resolved_base
    ):
        raise ValueError(
            f"{context}: {candidate} resolves to {resolved_candidate}, which "
            f"escapes {base} (resolves to {resolved_base})"
        )
    return resolved_candidate


__all__ = ["contains", "require_contained"]
