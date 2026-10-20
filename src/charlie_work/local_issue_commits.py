"""Commit the local issue tracker's accumulated file writes (issue #2434).

``LocalFileGitHub`` writes tracker state straight into the consumer repo's
tracked ``issues_dir`` files -- ``state: closed``, ``resolved:`` stamps,
label transitions and every appended ``issue_comment`` -- and nothing used
to commit those writes. The tracker's source of truth therefore lived only
in the working tree: ``git checkout -- .`` or a worker ``git add -A``
destroyed or smuggled it, and the permanent ``git status`` noise blocked
every consumer operation that sensibly requires a clean tree before a
fast-forward.

Single point of enforcement is this module: it is the only code that stages
and commits issue files. The backend's write sites (``_mutate`` and
``issue_comment`` in ``local_issues.py``) record what they touched on
``queue_issue_write``; the orchestrator drains the queue once per loop pass,
at the end of the pass, via ``orchestration.local_tracker_flush`` -- one
flush, not one commit per label edge. The flush is also self-healing: it
reads ``git status --porcelain -- <issues_dir>`` rather than trusting the
in-memory queue alone, so a pass that aborted before its flush, a
hand-edited frontmatter, or an operator issue-CLI write are all swept up by
the next flush. It never touches anything outside ``issues_dir``.

Skip semantics (always "leave it to the next pass"):

- ``commit_writes`` false or a non-git ``repo_root``: skipped deliberately,
  without an anomaly signal -- the kill switch is config, not a failure.
- HEAD detached, or a merge/rebase/cherry-pick in progress: deferred -- the
  flush reports a defer reason and ``orchestration.local_tracker_flush``
  records the ``local_tracker_writes_deferred`` event that ``charlie
  doctor``'s accumulation warning consumes.
- ``git add``/``git commit`` failing for one issue: every other issue still
  commits; the failure is reported in the result and rides the same event.
- Unresolved conflict dirt inside ``issues_dir`` (any status code carrying
  ``U``, or ``AA``/``DD``): the whole flush defers -- a commit on top of a
  conflicted index would record half a resolution.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from .local_issue_files import issue_number_from_name
from .subprocess_runner import RunResult, command_failure_message, run_captured

# Ordinary git plumbing bound (matches local_lane.GIT_OP_TIMEOUT_SECONDS):
# local and fast; the bound exists so a wedged git process cannot stall the
# pass.
_FLUSH_TIMEOUT_SECONDS = 120

# Per-repo queue of tracker writes waiting for the pass-end flush. Keyed by
# (repo_root, issues_dir): fleet lanes for different repos get separate
# entries, so concurrent lanes never share a key. Values map each touched
# issue file to the ordered unique verbs recorded against it; the commit
# message joins them ("close+label") rather than reverting to a generic verb.
_PER_REPO_PENDING: dict[tuple[str, str], dict[str, list[str]]] = {}
_PENDING_LOCK = threading.Lock()

# Status codes git uses for unresolved merge conflicts: any code containing
# "U", plus both double-action codes (both sides added / both deleted).
_CONFLICT_CODES = ("AA", "DD")


@dataclass(frozen=True)
class TrackerFlushResult:
    """What one pass-end flush did, as a value (never raises)."""

    # One entry per landed commit, e.g. "chore(issues): close+label #12".
    committed: tuple[str, ...] = ()
    # Issue files still dirty after this flush: the deferred skip's dirty set,
    # or the files of a failed commit group. Empty on a clean flush.
    left_dirty: tuple[str, ...] = ()
    # True when the flush tried to commit but could not -- unsafe git state
    # or every commit group failing. ``skip_reason`` is the benign
    # counterpart: the kill switch, a non-git repo_root (test doubles,
    # non-git consumers of the file backend), or an out-of-repo issues_dir.
    deferred: bool = False
    defer_reason: str = ""
    skipped: bool = False
    skip_reason: str = ""
    # git's failure detail for a partially- or fully-failed flush.
    reason: str = field(default="")

    @property
    def needs_attention(self) -> bool:
        """Whether the flush left wanted writes on the floor."""
        return self.deferred or bool(self.reason)


def queue_issue_write(repo_root: Path, issues_dir: Path, path: Path, verb: str) -> None:
    """Record one backend tracker write so the pass-end flush can commit it.

    Best-effort bookkeeping: never raises, and a verb lost here (the entry
    was already popped by an earlier flush) costs message fidelity, not data
    -- the flush's status rescan still commits the file.
    """
    try:
        with _PENDING_LOCK:
            entry = _PER_REPO_PENDING.setdefault((repo_root.as_posix(), issues_dir.as_posix()), {})
            verbs = entry.setdefault(path.as_posix(), [])
            if verb not in verbs:
                verbs.append(verb)
    except (OSError, ValueError):
        return


def _issues_pathspec(repo_root: Path, issues_dir: Path) -> str:
    """``issues_dir`` as a repo-relative pathspec, or plain posix if unrelocatable."""
    try:
        return issues_dir.relative_to(repo_root).as_posix()
    except ValueError:
        return issues_dir.as_posix()


def _status_payload(repo_root: Path, issues_dir: Path) -> RunResult:
    """``git status --porcelain -z`` scoped to ``issues_dir``."""
    return _run_git(
        repo_root,
        [
            "git",
            "-c",
            "core.quotePath=off",
            "status",
            "--porcelain",
            "-z",
            "--",
            _issues_pathspec(repo_root, issues_dir),
        ],
    )


def dirty_tracker_files(repo_root: Path, issues_dir: Path) -> tuple[str, ...]:
    """Issue-shaped paths git reports dirty under ``issues_dir``.

    Read from git's own data, not the in-memory queue, so the flush also
    sweeps writes whose queue entry was lost (aborted pass, hand edit,
    operator issue-CLI write) and doctor's check answers from real
    working-tree state.     Paths are repo-relative (``core.quotePath=off``),
    sorted; a wholly-untracked ``issues_dir`` (a brand-new consumer repo
    reaches the scan as one collapsed ``?? <dir>/`` entry) expands to the
    issue files actually inside it. Unresolved conflicts are included too --
    the flush's conflict gate defers on that signal rather than silently
    committing half a resolution.
    """
    try:
        issues_dir.relative_to(repo_root)
    except ValueError:
        return ()
    return _porcelain_issue_paths(_status_payload(repo_root, issues_dir).stdout, repo_root)


def _porcelain_issue_paths(payload: str, repo_root: Path) -> tuple[str, ...]:
    """Parse ``git status --porcelain -z`` into sorted issue-shaped repo paths."""
    names: list[tuple[str, bool]] = []  # (path, collapsed-untracked-dir)
    fields = payload.split("\0")
    index = 0
    while index < len(fields):
        entry = fields[index]
        index += 1
        if not entry:
            continue
        code = entry[:2]
        name = entry[3:]
        if code[0] in ("R", "C"):
            # Rename/copy pair: the original path is the next NUL field.
            index += 1
        if name.endswith("/"):
            names.append((name.rstrip("/"), True))
        elif issue_number_from_name(Path(name).name) is not None:
            names.append((name, False))
    return _expand_untracked_dirs(names, repo_root)


def _expand_untracked_dirs(names: list[tuple[str, bool]], repo_root: Path) -> tuple[str, ...]:
    """Turn collapsed ``?? dir/`` entries into the issue files inside them.

    The porcelain path is repo-relative (the probe ran with ``cwd=repo_root``),
    so expansion resolves it against ``repo_root`` -- resolving it against the
    calling process's own cwd would silently read the wrong tree.
    """
    out: list[str] = []
    for name, is_collapsed_dir in names:
        if not is_collapsed_dir:
            out.append(name)
            continue
        directory = repo_root / name
        if not directory.is_dir():
            continue
        try:
            children = sorted(directory.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_file() and issue_number_from_name(child.name) is not None:
                out.append(child.relative_to(repo_root).as_posix())
    return tuple(sorted(out))


def _porcelain_conflicted(payload: str) -> bool:
    """Whether a ``porcelain`` payload carries an unresolved merge conflict."""
    for entry in payload.split("\0"):
        code = entry[:2]
        if "U" in code or code in _CONFLICT_CODES:
            return True
    return False


def _unsafe_commit_reason(repo_root: Path) -> str | None:
    """Why committing tracker writes right now is unsafe, or None when safe.

    HEAD detached (``symbolic-ref -q HEAD`` fails) or any merge/rebase/
    cherry-pick state file present: committing now would land on a
    temporary branch or inside an interrupted operation. The three
    pseudo-refs probe in one spawn via ``rev-parse --git-path`` -- plumbing
    resolves each state file under the worktree's real gitdir, so the
    check stays worktree-agnostic without hardcoding ``.git/`` paths.
    """
    symbolic = _run_git(repo_root, ["git", "symbolic-ref", "-q", "--short", "HEAD"])
    if not symbolic.ok:
        return "HEAD is detached"
    refs = ("MERGE_HEAD", "REBASE_HEAD", "CHERRY_PICK_HEAD")
    probe_command = [
        "git",
        "rev-parse",
        *(arg for ref in refs for arg in ("--git-path", ref)),
    ]
    probe = _run_git(repo_root, probe_command)
    paths = probe.stdout.splitlines()
    if not probe.ok or len(paths) != len(refs):
        return command_failure_message(probe_command, probe, "interrupted-operation probe failed")
    for ref, raw in zip(refs, paths, strict=True):
        if (repo_root / raw).exists():
            return f"{ref} references an interrupted merge/rebase"
    return None


def _run_git(repo_root: Path, argv: list[str]) -> RunResult:
    return run_captured(argv, cwd=repo_root, timeout_seconds=_FLUSH_TIMEOUT_SECONDS)


def _pop_pending(repo_root: Path, issues_dir: Path) -> dict[str, list[str]]:
    with _PENDING_LOCK:
        entry = _PER_REPO_PENDING.pop((repo_root.as_posix(), issues_dir.as_posix()), None)
    if not entry:
        return {}
    return {name: list(verbs) for name, verbs in entry.items()}


def _queued_verbs(pending: dict[str, list[str]], repo_root: Path, names: tuple[str, ...]) -> str:
    """The ordered unique verbs this flush's queue recorded for ``names``.

    Falls back to "update" for files whose writes reached the tree through a
    channel without a queue entry (hand edit, operator issue-CLI write, a
    lost entry from an aborted pass).
    """
    verbs: list[str] = []
    for name in names:
        for verb in pending.get((repo_root / name).as_posix(), ()):
            if verb not in verbs:
                verbs.append(verb)
    return "+".join(verbs) if verbs else "update"


def flush_tracker_writes(
    repo_root: Path,
    issues_dir: Path,
    *,
    commit_writes: bool = True,
) -> TrackerFlushResult:
    """Commit every dirty issue-shaped file under ``issues_dir``, once per pass.

    Groups the dirty set by issue number -- one commit per issue per flush,
    not per write -- so a batched pass pays a small number of
    ``chore(issues): <verb> #<n>`` commits instead of one per label edge.
    Groups commit independently: one failure (e.g. a missing committer
    identity) never blocks the others. Never raises; git failures come back
    as values in the result.
    """
    pending = _pop_pending(repo_root, issues_dir)
    status = _status_payload(repo_root, issues_dir)
    if _porcelain_conflicted(status.stdout):
        return TrackerFlushResult(
            deferred=True,
            defer_reason="unresolved merge conflicts inside issues_dir",
            left_dirty=_porcelain_issue_paths(status.stdout, repo_root),
        )
    dirty = _porcelain_issue_paths(status.stdout, repo_root)
    # The non-git check comes before the empty-dirty check: from outside a
    # repository ``git status`` fails outright, so a test double or a
    # non-git consumer must still see its benign skip rather than a silent
    # "nothing to do" no-op.
    if not (repo_root / ".git").exists():
        return TrackerFlushResult(
            skipped=True, left_dirty=dirty, skip_reason="not a git repository"
        )
    if not commit_writes:
        return TrackerFlushResult(
            skipped=True,
            left_dirty=dirty,
            skip_reason="local_issues.commit_writes is false",
        )
    if not dirty:
        return TrackerFlushResult()
    unsafe = _unsafe_commit_reason(repo_root)
    if unsafe is not None:
        return TrackerFlushResult(deferred=True, defer_reason=unsafe, left_dirty=dirty)

    by_number: dict[int, list[str]] = {}
    for name in dirty:
        number = issue_number_from_name(Path(name).name)
        if number is None:
            continue  # filtered upstream; a stray non-issue name stays untouched
        by_number.setdefault(number, []).append(name)

    committed: list[str] = []
    failed: list[str] = []
    failed_names: list[str] = []
    for number in sorted(by_number):
        names = tuple(sorted(by_number[number]))
        message = f"chore(issues): {_queued_verbs(pending, repo_root, names)} #{number}"
        add_command = ["git", "add", "--", *names]
        commit_command = ["git", "commit", "--only", "-m", message, "--", *names]
        add = _run_git(repo_root, add_command)
        commit: RunResult | None = None
        if add.ok:
            commit = _run_git(repo_root, commit_command)
        if add.ok and commit is not None and commit.ok:
            committed.append(message)
            continue
        detail_command = add_command if not add.ok else (commit_command or [])
        failed.append(
            f"{message} — "
            f"{command_failure_message(detail_command, add if not add.ok else commit, 'git flush failed')}"
        )
        failed_names.extend(names)

    if failed:
        return TrackerFlushResult(
            committed=tuple(committed),
            left_dirty=tuple(failed_names),
            deferred=not committed,
            defer_reason="; ".join(failed) if not committed else "",
            reason="; ".join(failed),
        )
    return TrackerFlushResult(committed=tuple(committed))
