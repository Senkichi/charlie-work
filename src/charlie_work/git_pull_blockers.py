"""The two reasons git refuses to *start* a merge or fast-forward.

Git declines to move the worktree when doing so would clobber local work, and
it does not care which of two shapes that work takes:

* **(a)** an *untracked* file shadowing a path the incoming tree also carries
  ("The following untracked working tree files would be overwritten by merge")
* **(b)** a *locally-modified tracked* file the incoming tree also changes
  ("Your local changes to the following files would be overwritten by merge")

Both refusals happen before any merge state exists, so both look identical from
the outside — no ``MERGE_HEAD``, an empty ``--diff-filter=U``, and ``merge
--abort`` exiting 128. Handling only one leaves the other indistinguishable
from an unrecoverable conflict.

**Callers must consider the union of both classes** — but do not rely on the
refusal message to tell you which classes are present, because that varies by
command. Measured on this repo:

* ``git merge`` reported them *sequentially* — clearing every class (a) blocker
  simply revealed the class (b) blockers behind it, so a caller that repaired
  one class and retried was refused a second time (the #1084/#1085 case).
* ``git pull --ff-only`` reported *both at once*, as two ``error:`` blocks in a
  single refusal (the 2026-08-06 self-deploy outage).

Either way the union is the safe thing to compute, which is why these
predicates are derived independently rather than read off whichever shape the
message happened to take. An earlier version of this docstring asserted the
sequential behaviour universally; that was generalised from the merge path
alone and is corrected here rather than left to mislead.

These predicates are computed from git's own data rather than by parsing the
refusal text, which is localized and has been reworded across git versions.

The module exists because two consumers need the same answer about different
subjects: :mod:`charlie_work.worktree` asks it of an agent's worktree before a
rework merge, and :mod:`charlie_work.supervise` asks it of the orchestrator's
own checkout before a self-deploy fast-forward. They inject different runners
(one captures directly, the other threads a caller-supplied ``run_command`` for
testability), so the runner is a parameter. Two copies of a rule that decides
what may be destroyed are two chances for them to drift into disagreeing, and
only one of those outcomes is recoverable.
"""

from __future__ import annotations

from collections.abc import Callable

from .subprocess_runner import RunResult

#: Runs a git argv and returns the captured result. Callers bind cwd/timeout.
GitRunner = Callable[[list[str]], RunResult]


def _split_z(payload: str) -> set[str]:
    """Split a ``-z`` (NUL-delimited) git listing into a set of paths."""
    return {entry for entry in payload.split("\0") if entry}


def untracked_paths_shadowing_ref(run_git: GitRunner, ref: str) -> tuple[str, ...]:
    """Untracked paths that ``ref`` also tracks — refusal class (a).

    The untracked set intersected with the target ref's tree.

    ``--exclude-standard`` is deliberate: ignored files are silently
    overwritten by merge, so they never block one and must not be swept up
    into a deletion set.
    """
    others = run_git(
        ["git", "-c", "core.quotePath=off", "ls-files", "--others", "--exclude-standard", "-z"]
    )
    tracked = run_git(
        ["git", "-c", "core.quotePath=off", "ls-tree", "-r", "--name-only", "-z", ref]
    )
    if not others.ok or not tracked.ok:
        return ()
    return tuple(sorted(_split_z(others.stdout) & _split_z(tracked.stdout)))


def modified_paths_overwritten_by_ref(run_git: GitRunner, ref: str) -> tuple[str, ...]:
    """Locally-modified tracked paths that merging ``ref`` would change — class (b).

    The incoming side is computed against the **merge base**, not against
    ``ref`` directly: a path the local side changed and the base did not is not
    something the merge touches, so it cannot block one and must not be
    restored. Diffing ``HEAD..ref`` over-approximates.

    ``--diff-filter=M`` keeps this to paths present in both ``HEAD`` and the
    worktree, which are exactly the ones ``git checkout HEAD --`` can restore.
    A staged addition or a local deletion is deliberately left out rather than
    handed to a command that would fail on it — dropping it from the blocking
    set escalates with a diagnosis instead of attempting a repair that cannot
    work.
    """
    base_result = run_git(["git", "merge-base", "HEAD", ref])
    if not base_result.ok:
        return ()
    merge_base = base_result.stdout.strip()
    if not merge_base:
        return ()
    dirty = run_git(
        ["git", "-c", "core.quotePath=off", "diff", "--name-only", "--diff-filter=M", "-z", "HEAD"]
    )
    incoming = run_git(
        ["git", "-c", "core.quotePath=off", "diff", "--name-only", "-z", merge_base, ref]
    )
    if not dirty.ok or not incoming.ok:
        return ()
    return tuple(sorted(_split_z(dirty.stdout) & _split_z(incoming.stdout)))
