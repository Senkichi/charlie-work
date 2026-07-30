"""Pure planner for merging a legacy ``runtime.state_dir`` tree into the canonical root.

Why this exists
----------------
A repo's orchestrator state can live in two places at once: the canonical
location derived from ``runtime.state_dir`` (default ``.var/charlie-work``,
see ``layout.py``), and a legacy override pointing somewhere else (job-cannon's
``.var/devin-orchestrator``). Merging the legacy tree into the canonical one
was originally written as imperative PowerShell prose in a plan doc, and that
prose accumulated roughly a dozen defects because shell prose has no
verifier. This module replaces the *decision* half of that migration --
"which children may be moved, and which cannot" -- with tested, pure Python,
following the plan/actuate split already established by
:mod:`charlie_work.runner_allocation`: a pure function turns a snapshot into
an immutable plan; a separate actuator (out of scope here -- this module is
planning only) applies it.

Two defects this module exists specifically to prevent
--------------------------------------------------------
1. **Path separator + case mismatch.** ``git worktree list --porcelain``
   always emits forward-slash paths, while ``Path.iterdir()`` yields
   backslash paths on Windows for the exact same location. A naive
   ``in``/``==``/``startswith`` comparison between the two silently finds
   zero matches -- this precise mismatch made a hand-written probe report 74
   "orphaned" worktrees that were, in fact, all registered. Every comparison
   in this module routes through :func:`_normalize_path_key`, which folds
   both separator spelling and case (Windows paths are case-insensitive) into
   one comparable string, rather than relying on ``pathlib``'s own
   platform-dependent equality semantics.

2. **Registered worktrees hiding in unexpected children.** ``git worktree
   list --porcelain`` can report registrations nested arbitrarily deep inside
   a state-dir child -- e.g. ``dispatches/reviews/pr-1384``, two levels below
   a child named ``dispatches``, not ``worktrees``. A worktree's registration
   is a pair of absolute back-references (its own ``.git`` file pointing at
   ``<main>/.git/worktrees/<name>``, and that admin dir's ``gitdir`` file
   pointing back); a directory move rewrites neither, silently corrupting the
   registration. The blocking rule below is therefore a derived predicate
   over ``registered_worktrees`` -- checked at *every* depth under every
   child -- never a hardcoded list of "risky" child names such as
   ``"worktrees"``, which cannot generalize to ``dispatches`` or any other
   name not anticipated in advance (CLAUDE.md invariant: no hardcoded lists).

A third, related hazard this module also guards against: **atomic sibling
groups**. A SQLite database in WAL mode is really three files -- ``<db>``,
``<db>-wal``, ``<db>-shm`` -- and moving only some of them loses
committed-but-not-yet-checkpointed transactions. A naive per-child planner
cannot see this relationship, so :func:`plan_state_dir_migration` derives it
from the ``-wal``/``-shm`` naming convention (never from a specific filename
such as ``events.db``) and forces every member of such a group to share the
most restrictive member's disposition: if any one member is blocked, the
whole group is blocked.

Errors as values
----------------
Per CLAUDE.md ("errors from external processes come back as values, never
raised"), nothing in this module raises for an ordinary planning outcome.
:func:`plan_state_dir_migration` is pure and cannot fail (no I/O, so no
external error can occur); its ``MigrationPlan.ok`` is always ``True``.
:func:`gather_migration_inputs` is where a real failure can happen (the git
worktree listing can fail), and it surfaces that as ``ok=False`` with a
message on ``.error`` -- never an exception.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .worktree import _list_worktrees_porcelain

#: SQLite WAL-mode side-file suffixes. A child ending in one of these is part
#: of an atomic sibling group with the child obtained by stripping the
#: suffix, *when that stripped name is also present* among the children --
#: this is a naming *convention* (any SQLite database gets these side files),
#: not a hardcoded reference to any specific database such as ``events.db``.
_SQLITE_SIDE_FILE_SUFFIXES = ("-wal", "-shm")

#: What the planner decided for one child: move it, or block it (see
#: ``MigrationChild.reasons`` for why). Deliberately a small closed set of
#: string literals rather than a bool, so a future third disposition (e.g.
#: "skip") can be added without changing every caller's truthiness check.
Disposition = Literal["move", "blocked"]


@dataclass(frozen=True)
class MigrationChild:
    """One top-level entry directly under ``src_root`` and its migration verdict.

    ``reasons`` and ``remediation`` are tuples (not the single ``reason``
    string suggested in the original sketch) because a child can be blocked
    for more than one independent reason at once -- a nested registered
    worktree AND a destination name collision -- and both must be reported,
    not just the first one found. ``remediation`` holds zero or more
    ``git worktree move <src> <dst>`` command strings, one per offending
    registration; a name collision has no ``git worktree move`` remediation
    of its own; its explanation lives in ``reasons`` instead.

    ``group`` is empty for a child that stands alone. For a child that is
    part of an atomic sibling group (see the module docstring's WAL/SHM
    example), it holds every member's name, sorted, including this child's
    own -- attached whether or not the group ended up blocked, so a CLI can
    always render "these N entries move/block together" for any group.
    """

    name: str
    src_path: Path
    dst_path: Path
    disposition: Disposition
    reasons: tuple[str, ...] = ()
    remediation: tuple[str, ...] = ()
    group: tuple[str, ...] = ()

    @property
    def movable(self) -> bool:
        """``True`` when this child may be moved as-is."""
        return self.disposition == "move"

    @property
    def blocked(self) -> bool:
        """``True`` when this child must not be moved without remediation first."""
        return self.disposition == "blocked"


@dataclass(frozen=True)
class MigrationPlan:
    """A complete, immutable migration decision for one (src_root, dst_root) pair.

    ``ok``/``error`` describe whether the plan itself could be produced at
    all -- e.g. :func:`gather_migration_inputs` could not determine the
    repo's registered worktrees -- not whether individual children are
    movable. A plan with ``ok=False`` carries no children: without the
    registered-worktree data, no child can be safely judged, so nothing is
    planned rather than something being planned unsafely.
    """

    src_root: Path
    dst_root: Path
    children: tuple[MigrationChild, ...]
    ok: bool = True
    error: str | None = None

    @property
    def movable(self) -> tuple[MigrationChild, ...]:
        """Children cleared to move, in the order they were planned."""
        return tuple(child for child in self.children if child.movable)

    @property
    def blocked(self) -> tuple[MigrationChild, ...]:
        """Children that must not be moved until their reasons are resolved."""
        return tuple(child for child in self.children if child.blocked)


def _normalize_path_key(path: Path) -> str:
    """Return a separator- and case-folded comparison key for *path*.

    Both normalizations are load-bearing on their own:

    * Separator folding (``\\`` -> ``/``) reconciles ``git worktree list
      --porcelain`` output (always forward slashes) with ``Path`` values
      built from local filesystem iteration (backslashes on Windows).
    * Case folding reconciles Windows' case-insensitive filesystem, where a
      registration and an on-disk directory can legitimately differ in case.

    Every path comparison in this module goes through this one helper on
    both sides, rather than through ``Path.__eq__``/``in``/``startswith``
    directly -- ``pathlib``'s own equality semantics differ by platform
    (case-sensitive ``PurePosixPath`` vs. case-insensitive ``WindowsPath``),
    so relying on them would make this module's behavior depend on which
    flavor happens to be active instead of being pinned by one explicit rule.
    """
    return str(path).replace("\\", "/").rstrip("/").casefold()


def _is_equal_or_nested(inner: Path, outer: Path) -> bool:
    """Return ``True`` when *inner* is *outer* itself, or lies anywhere below it.

    Purely lexical (string-prefix) containment on the normalized keys from
    :func:`_normalize_path_key` -- deliberately NOT ``safe_path.contains``,
    which resolves symlinks/junctions on both sides via ``Path.resolve()``
    and is therefore impure. This function must stay filesystem-free so
    :func:`plan_state_dir_migration` remains a pure function; a future
    actuator applying this plan should use ``safe_path.contains`` (or
    ``require_contained``) for its own on-disk safety check before actually
    moving anything.
    """
    inner_key = _normalize_path_key(inner)
    outer_key = _normalize_path_key(outer)
    return inner_key == outer_key or inner_key.startswith(f"{outer_key}/")


def _relative_parts(outer: Path, inner: Path) -> tuple[str, ...]:
    """Return the path components of *inner* that lie below *outer*.

    Components are matched case- and separator-insensitively (the same
    normalization :func:`_is_equal_or_nested` uses), but the components that are
    *returned* keep their original casing -- these end up in a ``git worktree
    move`` command an operator runs verbatim, so a case-folded path would be
    both ugly and, on a case-sensitive checkout, wrong.

    Returns ``()`` when *inner* denotes *outer* itself or is not below it. Pure
    and total: no filesystem access, and it never raises, so it cannot break
    :func:`plan_state_dir_migration`'s totality the way ``Path.relative_to``
    (which raises ``ValueError``) would.
    """
    outer_parts = Path(str(outer).replace("\\", "/")).parts
    inner_parts = Path(str(inner).replace("\\", "/")).parts
    if len(inner_parts) <= len(outer_parts):
        return ()
    for outer_part, inner_part in zip(outer_parts, inner_parts):
        if outer_part.casefold() != inner_part.casefold():
            return ()
    return tuple(inner_parts[len(outer_parts) :])


def _sqlite_group_base_name(name: str) -> str | None:
    """Return the main-file name *name* is a WAL/SHM side file of, or ``None``.

    Purely a suffix check -- ``"events.db-wal"`` -> ``"events.db"`` -- with no
    knowledge of any specific database name. The caller still must confirm the
    stripped base is actually present among the sibling children before
    treating this as a real group; a lone ``"foo-wal"`` with no ``"foo"``
    sibling is not grouped with anything.
    """
    for suffix in _SQLITE_SIDE_FILE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return None


def _atomic_groups(children: Sequence[MigrationChild]) -> dict[str, tuple[str, ...]]:
    """Return ``{member_name: sorted_group_names}`` for every multi-member group.

    A name absent from the returned mapping is not part of any atomic group.
    """
    names_present = {child.name for child in children}
    groups: dict[str, set[str]] = {}
    for name in names_present:
        base = _sqlite_group_base_name(name)
        if base is not None and base in names_present:
            groups.setdefault(base, {base}).add(name)

    member_to_group: dict[str, tuple[str, ...]] = {}
    for base, members in groups.items():
        ordered = tuple(sorted(members))
        for member in members:
            member_to_group[member] = ordered
    return member_to_group


def _apply_atomic_groups(children: Sequence[MigrationChild]) -> tuple[MigrationChild, ...]:
    """Force every member of an atomic sibling group to share one disposition.

    If any member of a group is independently blocked, every member becomes
    blocked -- a migration must never move part of a group (see the module
    docstring's SQLite WAL/SHM rationale). A member dragged into "blocked"
    only because of a sibling gains an extra reason naming that sibling and
    summarizing why *it* is blocked, so the plan is self-explanatory without
    needing to cross-reference another child by hand. Groups with no blocked
    member are left exactly as planned, apart from gaining the ``group`` tag
    so a renderer can still show the atomic set.

    Pure: builds new :class:`MigrationChild` values via ``dataclasses.replace``
    rather than mutating any input, consistent with the frozen-dataclass
    invariant.
    """
    by_name = {child.name: child for child in children}
    member_to_group = _atomic_groups(children)

    result: list[MigrationChild] = []
    for child in children:
        group = member_to_group.get(child.name)
        if group is None:
            result.append(child)
            continue

        blocked_others = [
            by_name[other_name]
            for other_name in group
            if other_name != child.name and by_name[other_name].disposition == "blocked"
        ]
        if not blocked_others and child.disposition != "blocked":
            result.append(dataclasses.replace(child, group=group))
            continue

        cross_reasons = tuple(
            f"grouped with {other.name}, which is blocked: {'; '.join(other.reasons)}"
            for other in blocked_others
        )
        result.append(
            dataclasses.replace(
                child,
                disposition="blocked",
                reasons=child.reasons + cross_reasons,
                group=group,
            )
        )
    return tuple(result)


def plan_state_dir_migration(
    *,
    repo_root: Path,
    src_root: Path,
    dst_root: Path,
    src_children: Sequence[Path],
    dst_names: Sequence[str],
    registered_worktrees: Sequence[Path],
) -> MigrationPlan:
    """Decide, for each top-level child of *src_root*, whether it may be moved.

    Pure: no filesystem or subprocess access, so this is fully unit-testable
    against hand-built ``Path``/``str`` sequences. Never raises; always
    returns ``ok=True`` (there is no external operation here that can fail --
    see :func:`gather_migration_inputs` for the impure counterpart that can).
    *repo_root* is used only to format the ``git -C <repo_root> worktree
    move ...`` remediation strings -- ``git worktree move`` is a repository
    admin operation and must be pointed at the main repo explicitly so the
    command is correct regardless of the operator's current directory; it is
    never read from or written to here.

    Rules, applied independently and cumulatively per child (a child can be
    blocked for both at once -- both are always reported, never only the
    first found):

    1. **Nested registration.** A child is blocked if any entry in
       *registered_worktrees* equals it, or lies anywhere below it at any
       depth (checked via :func:`_is_equal_or_nested`, so a registration two
       levels down inside a child named e.g. ``dispatches`` is caught exactly
       like one directly inside a child named ``worktrees`` -- the predicate
       is derived from ``registered_worktrees`` itself, never from a
       hardcoded set of "risky" child names). The reason names every
       offending registration path; the remediation is one
       ``git -C <repo_root> worktree move <registration> <new-path>`` command
       string per registration -- moving the *worktree's registration*, never
       the containing directory, which is what would silently corrupt it. The
       new path flattens any intermediate nesting to
       ``<dst_root>/<child-name>/<registration-leaf-name>`` (or just
       ``<dst_root>/<child-name>`` when the registration *is* the child
       itself) rather than mirroring the legacy tree's full depth -- the
       destination layout should not inherit however deeply a stray
       registration happened to be buried in the legacy tree.
    2. **Destination name collision.** A child is blocked if its name already
       exists (case-insensitively) among *dst_names*.

    A child with no reasons is ``"move"`` -- unless rule 3 below overrides it:

    3. **Atomic sibling groups.** After rules 1-2 are applied per child, any
       child that is a WAL/SHM side file of another present child (see
       :func:`_apply_atomic_groups`) is forced to share its group's most
       restrictive disposition: if one member is blocked, every member is
       blocked, with an added reason naming the blocked sibling.
    """
    dst_name_keys = {name.casefold() for name in dst_names}
    children: list[MigrationChild] = []

    for child in src_children:
        name = child.name
        dst_path = dst_root / name
        reasons: list[str] = []
        remediation: list[str] = []

        nested = sorted(
            (reg for reg in registered_worktrees if _is_equal_or_nested(reg, child)),
            key=str,
        )
        if nested:
            offending = ", ".join(str(reg) for reg in nested)
            plural = "" if len(nested) == 1 else "s"
            reasons.append(
                f"{len(nested)} registered git worktree{plural} nested inside this "
                f"child: {offending}"
            )
            for reg in nested:
                # Preserve the registration's position *relative to the child*, rather than
                # re-parenting it by leaf name: a worktree at
                # ``dispatches/reviews/pr-1384`` must land at ``<dst>/dispatches/reviews/
                # pr-1384``, not ``<dst>/dispatches/pr-1384``. Flattening puts it beside the
                # directory the rest of that content migrates into.
                relative = _relative_parts(child, reg)
                target = dst_path.joinpath(*relative) if relative else dst_path
                remediation.append(f'git -C "{repo_root}" worktree move "{reg}" "{target}"')

        if name.casefold() in dst_name_keys:
            reasons.append(f"name {name!r} already exists in the destination at {dst_path}")

        disposition: Disposition = "blocked" if reasons else "move"
        children.append(
            MigrationChild(
                name=name,
                src_path=child,
                dst_path=dst_path,
                disposition=disposition,
                reasons=tuple(reasons),
                remediation=tuple(remediation),
            )
        )

    grouped_children = _apply_atomic_groups(children)
    return MigrationPlan(src_root=src_root, dst_root=dst_root, children=grouped_children)


def gather_migration_inputs(*, repo_root: Path, src_root: Path, dst_root: Path) -> MigrationPlan:
    """Gather filesystem/git state for one repo and produce its migration plan.

    The only impure function in this module: it lists *src_root*'s top-level
    children, *dst_root*'s existing entry names, and *repo_root*'s registered
    git worktrees (via :func:`charlie_work.worktree._list_worktrees_porcelain`,
    which this module reuses rather than re-implementing porcelain parsing),
    then hands all three to :func:`plan_state_dir_migration`. All decision
    logic lives there; this function only assembles its inputs.

    Two "nothing to reconcile" cases are deliberately NOT treated as errors:

    * *src_root* does not exist -> ``ok=True`` with zero children. There is
      nothing to migrate, which is a normal steady state (e.g. re-running the
      plan after a previous migration already completed), not a failure.
    * *dst_root* does not exist yet -> treated as having no existing entries,
      so nothing can collide by name. Creating it is the actuator's job (out
      of scope for this planning-only module).

    A git-worktree-listing failure IS an error (``ok=False``, children empty,
    ``.error`` set to a message naming the failure): the entire point of
    ``registered_worktrees`` is to stop a directory move from silently
    breaking a live worktree's registration, so if that data cannot be
    obtained, planning must not proceed as though zero worktrees were
    registered -- that would be exactly the unsafe assumption this module
    exists to prevent.
    """
    if not src_root.exists():
        return MigrationPlan(src_root=src_root, dst_root=dst_root, children=())

    src_children = list(src_root.iterdir())
    dst_names = [child.name for child in dst_root.iterdir()] if dst_root.exists() else []

    registered_entries, list_error = _list_worktrees_porcelain(repo_root)
    if list_error is not None:
        return MigrationPlan(
            src_root=src_root,
            dst_root=dst_root,
            children=(),
            ok=False,
            error=f"could not list registered git worktrees for {repo_root}: {list_error}",
        )

    # Fail closed on an unexpected entry shape rather than filtering it out.
    # ``_list_worktrees_porcelain`` documents that every returned dict has a ``Path``
    # under "worktree", so this cannot fire today -- but a silent ``isinstance`` filter
    # degrades to "zero worktrees registered" if that ever changes, which is precisely
    # the reading that lets a directory move break a live registration.
    registered_worktrees: list[Path] = []
    for entry in registered_entries:
        value = entry.get("worktree")
        if not isinstance(value, Path):
            return MigrationPlan(
                src_root=src_root,
                dst_root=dst_root,
                children=(),
                ok=False,
                error=(
                    f"git worktree listing for {repo_root} returned an entry with no usable "
                    f"path ({value!r}); refusing to plan as though nothing is registered"
                ),
            )
        registered_worktrees.append(value)

    return plan_state_dir_migration(
        repo_root=repo_root,
        src_root=src_root,
        dst_root=dst_root,
        src_children=src_children,
        dst_names=dst_names,
        registered_worktrees=registered_worktrees,
    )
