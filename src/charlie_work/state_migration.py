"""Pure planner for merging a legacy ``runtime.state_dir`` tree into the canonical root.

Why this exists
----------------
A repo's orchestrator state can live in two places at once: the canonical
location derived from ``runtime.state_dir`` (default ``.var/charlie-work``,
see ``layout.py``), and a legacy override pointing somewhere else (job-cannon's
``.var/devin-orchestrator``). Merging the legacy tree into the canonical one
was originally written as imperative PowerShell prose in a plan doc, and that
prose accumulated roughly a dozen defects because shell prose has no
verifier. This module replaces both the *decision* half of that migration --
"which children may be moved, and which cannot" -- and the *actuation* half
that applies the decision, following the plan/actuate split already
established by :mod:`ci_fleet.runner_allocation_pass` (charlie-work's own
copy was deleted by issue #921; the split it pioneered lives on there):
:func:`plan_state_dir_migration` turns a snapshot into an immutable plan;
:func:`apply_state_dir_migration` applies it.

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
raised"), nothing in this module raises for an ordinary outcome.
:func:`plan_state_dir_migration` is pure and cannot fail (no I/O, so no
external error can occur); its ``MigrationPlan.ok`` is always ``True``.
:func:`gather_migration_inputs` is where a real failure can happen (the git
worktree listing can fail), and it surfaces that as ``ok=False`` with a
message on ``.error`` -- never an exception. :func:`apply_state_dir_migration`
is the third impure function here: a stale plan caught by its TOCTOU
re-check, or an ``OSError`` raised by the underlying move, both come back as
``MigrationOutcome(ok=False, error=...)`` -- never an exception either.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from . import layout, safe_path
from .state import StateLockBusy, load_state, save_state, state_lock
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
    :func:`plan_state_dir_migration` remains a pure function.
    :func:`apply_state_dir_migration` deliberately does not add its own
    ``safe_path.contains`` on-disk containment check before moving a child --
    its TOCTOU existence re-check (src still present, dst still absent) is
    judged sufficient for now; a stronger on-disk containment guard remains a
    possible future hardening, not implemented here.
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

        if not _is_lexically_contained(src_root, child):
            reasons.append(
                f"source {child} is not inside the source root {src_root} -- refusing to "
                "move a path from outside the tree being migrated"
            )
        if not _is_lexically_contained(dst_root, dst_path):
            reasons.append(
                f"destination {dst_path} escapes the destination root {dst_root} "
                f"(child name {name!r})"
            )

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
      so nothing can collide by name. Creating it is
      :func:`apply_state_dir_migration`'s job, not this function's.

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


def _is_lexically_contained(root: Path, candidate: Path) -> bool:
    """Exact-prefix containment decided *lexically*, without touching the filesystem.

    The planner is pure, so it cannot use :func:`charlie_work.safe_path.contains`
    (which resolves both sides on disk). The two checks are not redundant: this one
    catches a name whose *shape* escapes -- ``..`` (note ``Path("..").name`` is
    ``".."``, so ``dst_root / name`` genuinely climbs out) or a caller-supplied
    absolute path outside the tree -- while the actuator's on-disk check catches an
    escape that exists only after resolution, i.e. through a junction or symlink.
    Neither subsumes the other, and only the lexical one can run in a pure function.

    Exact-prefix, not a bare ``startswith``: the latter would treat a sibling
    ``<root>-old`` as contained, the hazard CLAUDE.md documents for ``$WT_ROOT``.
    """
    root_key = os.path.normcase(os.path.normpath(str(root)))
    candidate_key = os.path.normcase(os.path.normpath(str(candidate)))
    return candidate_key == root_key or candidate_key.startswith(root_key.rstrip("\\/") + os.sep)


def _default_mover(src: Path, dst: Path) -> None:
    """Move *src* to *dst* via ``Path.rename`` -- the default ``mover``.

    A same-volume atomic rename (a repo's two state-dir roots are always
    ``.var/...`` paths under the same checkout, never across a volume
    boundary). :func:`apply_state_dir_migration` accepts an injected ``mover``
    instead of calling this directly so it is unit-testable against a
    recording fake with zero real filesystem writes, mirroring the
    ``ProcessLister`` seam in ``quiesce.py``.
    """
    src.rename(dst)


@dataclass(frozen=True)
class StateRewriteResult:
    """Result of :func:`_rewrite_state_json_paths` -- the embedded-path rewrite.

    ``rewritten`` is the count of string leaves under the old state root that
    were rewritten to the new one. ``error`` is set on every ``ok=False``
    outcome: a string that looks like a path under the old root but whose
    rewritten target does not exist (a non-path that coincidentally contains
    the prefix, or a path that was not moved), a lock-acquisition failure, or
    an ``OSError`` from the load/save cycle.
    """

    ok: bool
    rewritten: int = 0
    error: str | None = None


@dataclass(frozen=True)
class MigrationOutcome:
    """Result of :func:`apply_state_dir_migration` -- what actually moved.

    ``moved`` lists child names in the order they were moved -- exactly
    ``plan.movable``'s names on full success, a strict prefix of them on an
    aborted run. ``aborted_at`` names the child whose pre-move re-check or
    move itself failed; it is ``None`` both on full success and on a
    pre-flight refusal (refusal happens before any child is touched, so
    there is no child to name). ``error`` is set on every ``ok=False``
    outcome and is the only field distinguishing *why* -- pre-flight
    refusal, a stale plan caught by the TOCTOU re-check, or an ``OSError``
    from ``mover`` -- since ``aborted_at`` alone cannot tell those apart.

    ``rewritten_paths`` is the count of embedded absolute paths inside
    ``state.json`` that were rewritten from the old root to the new one as
    part of the migration (issue #735). It is ``0`` when there is no
    ``state.json`` to rewrite, when no embedded paths named the old root,
    or when the rewrite failed (``ok=False``).
    """

    ok: bool
    moved: tuple[str, ...] = ()
    error: str | None = None
    aborted_at: str | None = None
    rewritten_paths: int = 0


def _try_rewrite_path_string(
    value: str, src_root: Path, dst_root: Path
) -> tuple[str, int, str | None]:
    """Rewrite *value* if it is a path under *src_root*; verify the target exists.

    A "hit" is a string that, interpreted as a :class:`Path`, is lexically
    contained in *src_root* (exact-prefix, via :func:`_is_lexically_contained` --
    the same pure containment check the planner uses, so separator and case
    differences are folded). For each hit the relative parts below *src_root*
    are re-anchored under *dst_root* and the resulting path is checked for
    existence: after a successful all-or-nothing migration every file that was
    under the old root is now under the new one, so a rewritten target that
    does not exist means either (a) the string was not really a path but
    coincidentally started with the old-root prefix, or (b) a path that was not
    moved. Both are refused -- the issue #735 design principle is that the
    inconsistent state must never be representable, and silently skipping a
    non-resolving hit would reintroduce it.

    Returns ``(rewritten_or_original_string, count, error)``. ``count`` is 1
    on a successful rewrite, 0 otherwise. ``error`` is set when a hit's
    rewritten target does not exist.
    """
    candidate = Path(value)
    if not _is_lexically_contained(src_root, candidate):
        return value, 0, None
    relative = _relative_parts(src_root, candidate)
    rewritten = dst_root.joinpath(*relative) if relative else dst_root
    if not rewritten.exists():
        return (
            value,
            0,
            (
                f"string {value!r} is under the old state root {src_root} but its "
                f"rewritten path {rewritten} does not exist -- refusing to rewrite "
                "a non-path or unmoved target"
            ),
        )
    return str(rewritten), 1, None


def _walk_and_rewrite(value: Any, src_root: Path, dst_root: Path) -> tuple[Any, int, str | None]:
    """Recursively rewrite string leaves under *src_root* to *dst_root*.

    Structural walk, not ``str.replace`` on the file text: a text replace
    cannot distinguish a path value from a branch name or issue title that
    happens to contain the token, and a ``json.loads`` round-trip guard catches
    corruption but not a semantically wrong substitution. Only ``str`` leaves
    are candidates; dict keys, ints, floats, bools, and ``None`` are passed
    through untouched.

    Returns ``(new_value, count, error)``. On error, *new_value* is the
    original *value* unchanged and *count* is 0 -- the caller must not save
    a partially-rewritten tree.
    """
    if isinstance(value, str):
        return _try_rewrite_path_string(value, src_root, dst_root)
    if isinstance(value, dict):
        new_dict: dict[str, Any] = {}
        total = 0
        for key, item in value.items():
            new_item, count, error = _walk_and_rewrite(item, src_root, dst_root)
            if error is not None:
                return value, 0, error
            new_dict[key] = new_item
            total += count
        return new_dict, total, None
    if isinstance(value, list):
        new_list: list[Any] = []
        total = 0
        for item in value:
            new_item, count, error = _walk_and_rewrite(item, src_root, dst_root)
            if error is not None:
                return value, 0, error
            new_list.append(new_item)
            total += count
        return new_list, total, None
    return value, 0, None


def _rewrite_state_json_paths(
    state_path: Path, src_root: Path, dst_root: Path
) -> StateRewriteResult:
    """Load ``state.json``, rewrite embedded paths, save -- all under ``state_lock``.

    The single-point-of-enforcement for the invariant *"pointers in state.json
    name the live state root"* (issue #735). After :func:`apply_state_dir_migration`
    moves every child, ``state.json`` itself is at ``dst_root/state.json`` but
    its *contents* still embed absolute paths naming ``src_root/...``. This
    function walks the parsed JSON structurally (via :func:`_walk_and_rewrite`),
    rewrites every string leaf under *src_root* to *dst_root*, verifies each
    rewritten target exists, and saves atomically (via :func:`save_state`, which
    uses the temp-file + ``replace`` pattern) under :func:`state_lock`.

    A missing ``state_path`` is not an error: a fresh tree with no state yet
    has nothing to rewrite, and the migration of the tree itself still
    succeeded. Returns ``ok=True, rewritten=0`` in that case.

    Per CLAUDE.md ("errors from external processes come back as values, never
    raised"), :class:`StateLockBusy` and :class:`OSError` are caught and
    returned as ``ok=False`` with ``.error`` set -- never propagated.
    """
    if not state_path.exists():
        return StateRewriteResult(ok=True, rewritten=0)

    try:
        with state_lock(state_path):
            data = load_state(state_path)
            new_data, count, error = _walk_and_rewrite(data, src_root, dst_root)
            if error is not None:
                return StateRewriteResult(ok=False, error=error)
            if count > 0:
                save_state(state_path, new_data)
            return StateRewriteResult(ok=True, rewritten=count)
    except StateLockBusy as exc:
        return StateRewriteResult(ok=False, error=f"could not acquire state lock: {exc}")
    except OSError as exc:
        return StateRewriteResult(ok=False, error=f"state rewrite I/O error: {exc}")


def apply_state_dir_migration(
    plan: MigrationPlan,
    *,
    mover: Callable[[Path, Path], None] | None = None,
    state_rewriter: Callable[[Path, Path, Path], StateRewriteResult] | None = None,
) -> MigrationOutcome:
    """Actuate *plan*: move every movable child from ``src_root`` to ``dst_root``.

    The actuator half of the plan/actuate split described in the module
    docstring. All-or-nothing by construction, for the same reason
    :func:`plan_state_dir_migration` forces a whole atomic sibling group to
    share one disposition: a partial migration is the dangerous state, not
    merely an incomplete one -- it leaves the source directory non-empty, no
    compat junction pointing anywhere, and both trees simultaneously "live"
    with no single answer for where a given child currently is.

    Pre-flight (refuses before touching anything)
    ----------------------------------------------
    * ``plan.ok is False`` -- refuse. The plan itself could not be produced
      (see :func:`gather_migration_inputs`), so there is nothing safe to act
      on.
    * ``plan.blocked`` non-empty -- refuse, naming every blocked child in
      ``.error``. Moving only ``plan.movable`` and silently leaving the
      blocked children behind is exactly the partial-migration hazard this
      function exists to prevent.

    TOCTOU re-check (immediately before each move)
    -------------------------------------------------
    *plan* is a snapshot taken at some time T; this function may run at
    T+n. Immediately before moving each child, this re-verifies that
    ``child.src_path`` still exists and ``child.dst_path`` still does not --
    mirroring ``ci_fleet.runner_slots.park_runner_slot``'s re-check for a live
    ``Runner.Worker`` immediately before stopping a listener (charlie-work's
    own copy was deleted by issue #921), because a plan
    is a snapshot and the world can change between planning and actuation.
    On divergence the whole run aborts (``ok=False``, ``aborted_at`` set,
    ``moved`` holding exactly what succeeded before this child) rather than
    skipping the one stale child and continuing -- skipping would produce
    the same forbidden partial state as ignoring ``plan.blocked`` above.

    Movement and failure handling
    ------------------------------
    Each move goes through *mover* (default :func:`_default_mover`, a
    same-volume ``Path.rename``); injecting it is what makes this function
    testable without touching a real filesystem tree. Per CLAUDE.md ("errors
    from external processes come back as values, never raised"), an
    ``OSError`` from either the pre-move existence re-checks or *mover*
    itself is caught and returned as ``ok=False`` with ``aborted_at`` naming
    the offending child -- never propagated to the caller.

    ``plan.dst_root`` is created (``mkdir(parents=True, exist_ok=True)``) the
    first time there is at least one child to move -- :func:`gather_migration_inputs`
    documents that creating it is this function's job, since a planning-only
    module cannot touch the filesystem. Nothing is created when
    ``plan.movable`` is empty, so a no-op run has no filesystem side effects.

    Embedded-path rewrite (issue #735)
    -----------------------------------
    After every child is moved, ``state.json`` (now at
    ``dst_root/state.json``) still embeds absolute paths naming the old root
    -- ``prompt_path``, ``decision_path``, ``cross_family_report``,
    ``verdict_source``, and any other string leaf that was stored as
    ``str(Path(...))``. The rewrite is part of the migration itself, not a
    separate manual gate, so the inconsistent state is never representable.
    *state_rewriter* (default :func:`_rewrite_state_json_paths`) loads the
    file under ``state_lock``, walks it structurally, rewrites every string
    leaf under ``src_root`` to ``dst_root``, verifies each rewritten target
    exists, and saves atomically. The count is reported in
    ``MigrationOutcome.rewritten_paths``. A rewrite failure (a non-path hit,
    a missing target, a lock timeout) returns ``ok=False`` -- the children
    have already moved, so this is an incomplete migration that needs manual
    attention, not a rollback point.

    Deliberately NOT done here: creating the compat junction (``mklink /J``)
    that makes the legacy location keep resolving after the move. Data
    movement and "make it look finished" must stay separately observable
    outcomes. The embedded-path rewrite removes the *need* for the junction
    (issue #735), but the junction itself remains a separate operator
    concern.
    """
    if not plan.ok:
        return MigrationOutcome(
            ok=False,
            error=f"refusing to apply migration: plan itself is not ok: {plan.error}",
        )

    blocked = plan.blocked
    if blocked:
        blocked_names = ", ".join(child.name for child in blocked)
        noun = "child" if len(blocked) == 1 else "children"
        return MigrationOutcome(
            ok=False,
            error=(
                f"refusing to apply migration: {len(blocked)} blocked {noun} "
                f"present (all-or-nothing): {blocked_names}"
            ),
        )

    move = mover if mover is not None else _default_mover
    rewrite = state_rewriter if state_rewriter is not None else _rewrite_state_json_paths
    movable = plan.movable
    moved: list[str] = []

    if movable:
        try:
            plan.dst_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return MigrationOutcome(
                ok=False,
                error=f"could not create destination root {plan.dst_root}: {exc}",
            )

    for child in movable:
        try:
            # On-disk containment, re-checked here and not only in the planner: a
            # junction or symlink can make a lexically-contained path resolve outside
            # the tree, and only a resolving check sees that. The rehearsal trees carry
            # .venv junctions, and CLAUDE.md's managed_root invariant is exactly this
            # hazard. Aborts the whole run like any other divergence.
            if not safe_path.contains(plan.src_root, child.src_path) or not safe_path.contains(
                plan.dst_root, child.dst_path
            ):
                return MigrationOutcome(
                    ok=False,
                    moved=tuple(moved),
                    aborted_at=child.name,
                    error=(
                        f"aborting migration: {child.name} resolves outside the migration "
                        f"roots on disk ({child.src_path} -> {child.dst_path}); a junction "
                        "or symlink escapes the tree"
                    ),
                )
            if not child.src_path.exists():
                return MigrationOutcome(
                    ok=False,
                    moved=tuple(moved),
                    aborted_at=child.name,
                    error=(
                        f"aborting migration: {child.name}'s source {child.src_path} no "
                        "longer exists -- plan is stale; nothing after it was moved"
                    ),
                )
            if child.dst_path.exists():
                return MigrationOutcome(
                    ok=False,
                    moved=tuple(moved),
                    aborted_at=child.name,
                    error=(
                        f"aborting migration: {child.name}'s destination {child.dst_path} "
                        "now exists -- plan is stale; nothing after it was moved"
                    ),
                )
            move(child.src_path, child.dst_path)
        except OSError as exc:
            return MigrationOutcome(
                ok=False,
                moved=tuple(moved),
                aborted_at=child.name,
                error=f"failed to move {child.name}: {exc}",
            )
        moved.append(child.name)

    # Issue #735: after every child is moved, rewrite embedded absolute paths
    # inside state.json so they name the new root. ``state.json`` was itself a
    # child and is now at ``dst_root/state.json``; its contents still point at
    # ``src_root/...``. The rewrite is part of the migration, not a separate
    # manual gate, so the inconsistent state is never representable.
    state_path = plan.dst_root / layout.STATE_FILENAME
    rewrite_result = rewrite(state_path, plan.src_root, plan.dst_root)
    if not rewrite_result.ok:
        return MigrationOutcome(
            ok=False,
            moved=tuple(moved),
            error=(f"children moved but state.json path rewrite failed: {rewrite_result.error}"),
        )

    return MigrationOutcome(ok=True, moved=tuple(moved), rewritten_paths=rewrite_result.rewritten)
