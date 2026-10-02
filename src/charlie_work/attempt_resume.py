"""Resume a fresh redispatch from the work a provider-throttle death preserved (#2289).

A provider rate limit kills every live worker at once, and the free tier
limits every ~45-60 minutes. The dead attempt's work is already durable:
``worktree._capture_worktree_work_to_rescue_ref`` saves a dirty tree to
``refs/charlie/rescue/issue-<n>-<ts>`` and ``attempt_refs.snapshot_attempt_ref``
saves a branch tip to ``refs/charlie/attempts/issue-<n>/attempt-<k>``. Before
this module nothing ever read either ref back, so every redispatch restarted
from the base and an issue needing more than one rate-limit window could never
finish (#2269 lost a 603-line rescue ref eight times).

:func:`seed_from_throttled_attempt` is the single point of enforcement. It runs
once, from ``worktree.create_worktree``'s fresh-dispatch branch, after the
worktree sits on the base and before the worker prompt is written:

1. **Was the last death a provider throttle?** The latest per-issue death event
   (``session_exited`` / ``session_stalled``, even when unclassified, which vetoes;
   follow-up relabel/escalate events only when classified) carries the
   classifier's ``failure_kind``. A worker launched after that death vetoes too.
   Only :data:`rework_attempt_exemption.PROVIDER_THROTTLE_EXEMPT_KINDS` kinds
   qualify. The issue entry's own ``dead_worker_failure_kind`` stamp is cleared
   at the dispatch claim, before this runs, so the event is the durable record.
   A worker that died of its own fault (stalled, crash, ...) must not inherit
   possibly-bad work, so it is a clean start.
2. **Which ref belongs to that death?** The newest rescue or attempt ref
   created at or after the death event. A rescue ref carries its creation time
   in its name; an attempt ref's creation time is its reflog entry
   (``attempt_refs`` writes it with ``--create-reflog``). A ref with no
   readable creation time is never used, and neither is one older than the
   death, so an earlier unrelated attempt's leftovers are never applied.
3. **Apply it onto the base.** A rescue ref is a snapshot commit of a dirty
   tree, so it is squash-merged and left as uncommitted changes, exactly the
   state the dead worker had. An attempt ref is applied the same way. Refs whose
   diff against the base is empty are skipped, and a rescue ref outranks an
   attempt ref (it is the superset for the same death).

Every failure (conflict, git error, unreadable ref, empty ref) restores the clean
base and returns ``None`` after emitting ``attempt_resume_failed``: the dispatch
is never blocked. The one exception is :class:`ResumeRestoreError`: if the clean
base cannot be proven after a failed seed, the caller must tear the worktree down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .attempt_refs import ATTEMPT_REF_PREFIX
from .rework_attempt_exemption import is_provider_throttle_rework_death
from .subprocess_runner import run_captured

_TIMEOUT_SECONDS = 120
RESCUE_REF_PREFIX = "refs/charlie/rescue"
# Event kinds that record one dead worker session of one issue. A classifying
# kind is the death record itself, so it counts even with no ``failure_kind``
# (an unclassified death is still the issue's newest death and must veto).
CLASSIFYING_DEATH_KINDS: tuple[str, ...] = ("session_exited", "session_stalled")


# Follow-up events about a death already recorded elsewhere. Without a
# ``failure_kind`` they say nothing about why the worker died and must not mask
# the classified death they follow.
# Built lazily: ``dead_worker_sweep`` imports ``worktree`` (via claude_code), which
# imports this module, so a module-level import of the kind constant would cycle.
def _follow_up_death_kinds() -> tuple[str, ...]:
    from .dead_worker_sweep.decide_common import SESSION_FAILED_RELABELED

    return (
        SESSION_FAILED_RELABELED,
        "session_failed_escalated",
        "session_failed_relabeled_sweep",
    )


# Events recording a worker launch; ``issue_numbers`` lists the launched issues.
LAUNCH_EVENT_KINDS: tuple[str, ...] = ("dispatch", "dispatch_rework")

_RESCUE_NAME = re.compile(r"/issue-(\d+)-(\d{8}T\d{6}\d*Z)$")
_RESCUE_TS_FORMAT = "%Y%m%dT%H%M%S%fZ"
_REFLOG_SELECTOR = re.compile(r"@\{(\d+)\}$")


@dataclass(frozen=True)
class ResumedAttempt:
    """A fresh worktree seeded from a throttle-killed attempt's preserved work."""

    ref: str
    ref_kind: str  # "rescue" | "attempt"
    files: int
    insertions: int


@dataclass(frozen=True)
class PreservedRef:
    name: str
    kind: str  # "rescue" | "attempt"
    created_at: float  # epoch seconds


def render_resume_notice(resumed: ResumedAttempt) -> str:
    """Prompt text telling the worker its tree already holds the prior attempt's work."""
    return (
        "## Previous attempt interrupted by a provider rate limit\n\n"
        "The previous attempt at this issue was interrupted by a provider rate "
        "limit, not by a problem with its work. Its partial work "
        f"({resumed.files} file(s), +{resumed.insertions} lines, from "
        f"`{resumed.ref}`) is already in your working tree. Continue that work: "
        "review what is there, finish it, and run the tests. Do not restart "
        "from scratch or discard it.\n"
    )


def apply_resume_notice(prompt_text: str, resumed: ResumedAttempt) -> str:
    """Append :func:`render_resume_notice` to ``prompt_text``."""
    return f"{prompt_text.rstrip()}\n\n{render_resume_notice(resumed)}"


def _epoch(ts: object) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def latest_death(state_file: Path, issue_number: int) -> tuple[str, float] | None:
    """``(failure_kind, epoch)`` of the issue's newest worker death.

    ``failure_kind`` is ``""`` when the newest death carries no classification,
    which never names a throttle: an unclassified death must veto a resume.
    """
    from .instrumentation import query_events

    follow_up = _follow_up_death_kinds()
    newest: tuple[float, str] | None = None
    for kind in (*CLASSIFYING_DEATH_KINDS, *follow_up):
        for event in query_events(state_file, kind=kind, issue_number=issue_number, limit=200):
            at = _epoch(event.get("ts"))
            payload = event.get("payload")
            if at is None or not isinstance(payload, dict):
                continue
            failure_kind = payload.get("failure_kind")
            if not isinstance(failure_kind, str):
                if kind in follow_up:
                    continue
                failure_kind = ""
            if newest is None or at >= newest[0]:
                newest = (at, failure_kind)
    if newest is None:
        return None
    return newest[1], newest[0]


def launched_since(state_file: Path, issue_number: int, since: float) -> bool:
    """True when a worker was launched for the issue at or after ``since``.

    Event timestamps are whole seconds, so ``>=`` fails safe: a same-second launch
    is treated as a later one (a clean start) rather than risking a stale resume.
    """
    from .instrumentation import query_events

    for kind in LAUNCH_EVENT_KINDS:
        for event in query_events(state_file, kind=kind, limit=500):
            at = _epoch(event.get("ts"))
            payload = event.get("payload")
            if at is None or at < since or not isinstance(payload, dict):
                continue
            issues = payload.get("issue_numbers")
            if isinstance(issues, list) and issue_number in issues:
                return True
    return False


def _git(repo: Path, *args: str):
    return run_captured(["git", *args], cwd=repo, timeout_seconds=_TIMEOUT_SECONDS)


def _rescue_refs(repo_root: Path, issue_number: int) -> list[PreservedRef]:
    out = _git(repo_root, "for-each-ref", "--format=%(refname)", RESCUE_REF_PREFIX)
    refs: list[PreservedRef] = []
    for line in out.stdout.splitlines() if out.ok else ():
        name = line.strip()
        match = _RESCUE_NAME.search(name)
        if not match or int(match.group(1)) != issue_number:
            continue
        try:
            stamp = datetime.strptime(match.group(2), _RESCUE_TS_FORMAT).replace(tzinfo=UTC)
        except ValueError:
            continue
        refs.append(PreservedRef(name, "rescue", stamp.timestamp()))
    return refs


def _attempt_refs(repo_root: Path, issue_number: int) -> list[PreservedRef]:
    out = _git(
        repo_root,
        "for-each-ref",
        "--format=%(refname)",
        f"{ATTEMPT_REF_PREFIX}/issue-{issue_number}",
    )
    refs: list[PreservedRef] = []
    for line in out.stdout.splitlines() if out.ok else ():
        name = line.strip()
        if not name:
            continue
        # The reflog entry is the only record of when the snapshot was taken (the
        # tip commit's own date is when the worker last committed, before death).
        # A ref with no reflog (written before #2289) has no usable time.
        log = _git(repo_root, "reflog", "show", "-1", "--format=%gd", "--date=unix", name)
        match = _REFLOG_SELECTOR.search(log.stdout.strip()) if log.ok else None
        if match:
            refs.append(PreservedRef(name, "attempt", float(match.group(1))))
    return refs


def preserved_ref_candidates(
    repo_root: Path, issue_number: int, *, not_before: float
) -> list[PreservedRef]:
    """Rescue/attempt refs for the issue made at or after ``not_before``, best first.

    A rescue ref outranks any attempt ref: its parent is the worktree HEAD at
    death and it snapshots the dirty tree, so it is a superset of the attempt ref
    ``snapshot_attempt_ref`` writes for the same death (whose whole-second reflog
    time can even read as newer). Within a kind, newest first.
    """
    # Reflog times are whole seconds; compare at that granularity so a ref made in
    # the same second as the death event is not rejected.
    floor = int(not_before)
    rescue = sorted(
        (r for r in _rescue_refs(repo_root, issue_number) if int(r.created_at) >= floor),
        key=lambda r: r.created_at,
        reverse=True,
    )
    attempt = sorted(
        (r for r in _attempt_refs(repo_root, issue_number) if int(r.created_at) >= floor),
        key=lambda r: r.created_at,
        reverse=True,
    )
    return [*rescue, *attempt]


def _diff_stats(repo: Path, base: str, ref: str) -> tuple[int, int] | None:
    result = _git(repo, "diff", "--numstat", base, ref)
    if not result.ok:
        return None
    files = insertions = 0
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files += 1
        insertions += int(parts[0]) if parts[0].isdigit() else 0
    return files, insertions


class ResumeRestoreError(RuntimeError):
    """The worktree could not be returned to the clean base after a failed seed."""


def _restore_base(worktree_path: Path, head: str, exclusions: tuple[str, ...]) -> None:
    """Return to the exact pre-seed state; raise :class:`ResumeRestoreError` if not provable."""
    for argv in (
        ("merge", "--abort"),
        ("cherry-pick", "--abort"),
        ("reset", "--hard", head),
        ("clean", "-fd", "--", ".", *exclusions),
    ):
        _git(worktree_path, *argv)  # the aborts legitimately fail when nothing is in progress
    now = _git(worktree_path, "rev-parse", "--verify", "HEAD")
    if not now.ok or now.stdout.strip() != head:
        raise ResumeRestoreError(f"worktree HEAD is not the base after restore: {now.stdout!r}")
    status = _git(worktree_path, "status", "--porcelain", "--", ".", *exclusions)
    if not status.ok or status.stdout.strip():
        raise ResumeRestoreError(
            f"worktree not clean after restore: {status.stdout.strip() or status.error}"
        )


def _apply(worktree_path: Path, ref: PreservedRef) -> str | None:
    """Squash ``ref`` onto the worktree's HEAD, unstaged; return a failure reason or None.

    One path for both ref kinds: a squash merge tolerates merge commits in an
    attempt branch and commits the base already holds, which a cherry-pick range
    does not.
    """
    merged = _git(worktree_path, "merge", "--squash", ref.name)
    if not merged.ok:
        return (
            f"conflict applying {ref.kind} ref: {merged.error or merged.stderr or merged.stdout}"
        )
    # Leave the changes unstaged, the dead worker's own state.
    unstaged = _git(worktree_path, "reset", "--mixed", "HEAD")
    if not unstaged.ok:
        return f"cannot unstage {ref.kind} ref: {unstaged.error or unstaged.stderr}"
    return None


def _emit_resumed(state_file: Path, issue_number: int, resumed: ResumedAttempt) -> None:
    try:
        from .instrumentation import log_event

        # write-gate-exempt(issue=2289): no write_gate receiver in create_worktree; best-effort event
        log_event(  # event-consumer: audit-only -- observability for the #2289 resume; the seeded worktree and the prompt notice are the enforcement, pinned by tests/test_attempt_resume.py.
            state_file,
            "attempt_resumed",
            {
                "issue_number": issue_number,
                "ref": resumed.ref,
                "ref_kind": resumed.ref_kind,
                "files": resumed.files,
                "insertions": resumed.insertions,
            },
        )
    except Exception:  # noqa: BLE001 -- instrumentation is best-effort
        pass


def _emit_resume_failed(state_file: Path, issue_number: int, ref: str | None, reason: str) -> None:
    try:
        from .instrumentation import log_event

        # write-gate-exempt(issue=2289): no write_gate receiver in create_worktree; best-effort event
        log_event(  # event-consumer: audit-only -- the dispatch already fell back to the clean base; this records why the preserved ref was not carried forward (pinned by tests/test_attempt_resume.py).
            state_file,
            "attempt_resume_failed",
            {"issue_number": issue_number, "ref": ref, "reason": reason},
        )
    except Exception:  # noqa: BLE001 -- instrumentation is best-effort
        pass


def seed_from_throttled_attempt(
    repo_root: Path,
    worktree_path: Path,
    issue_number: int | None,
    *,
    state_file: Path | None,
    scaffolding: tuple[str, ...] = (),
) -> ResumedAttempt | None:
    """Seed a fresh worktree from the preserved work of a throttle-killed attempt.

    Returns the :class:`ResumedAttempt` when work was applied, else ``None``
    (not a throttle death, no ref from that death, or any failure). Never
    raises and never blocks a dispatch; see the module docstring.
    """
    if issue_number is None or state_file is None:
        return None
    ref: PreservedRef | None = None
    try:
        death = latest_death(state_file, issue_number)
        if death is None or not is_provider_throttle_rework_death(death[0]):
            return None
        # The throttle death must also be the last thing that happened: a worker
        # launched since then produced the work now in any newer ref.
        if launched_since(state_file, issue_number, death[1]):
            return None
        candidates = preserved_ref_candidates(repo_root, issue_number, not_before=death[1])
        if not candidates:
            return None
        head_result = _git(worktree_path, "rev-parse", "--verify", "HEAD")
        if not head_result.ok:
            raise RuntimeError(f"cannot resolve worktree HEAD: {head_result.stderr}")
        head = head_result.stdout.strip()
        chosen: tuple[PreservedRef, tuple[int, int]] | None = None
        for candidate in candidates:
            ref = candidate
            base = _git(worktree_path, "merge-base", head, candidate.name)
            if not base.ok or not base.stdout.strip():
                raise RuntimeError("preserved ref shares no history with the base")
            stats = _diff_stats(worktree_path, base.stdout.strip(), candidate.name)
            if stats is None:
                raise RuntimeError("cannot diff the preserved ref")
            if stats[0] > 0:
                chosen = (candidate, stats)
                break
        if chosen is None:
            # An empty snapshot (e.g. an attempt ref with no commits) preserves nothing.
            raise RuntimeError("empty_ref: every preserved ref is empty against the base")
        ref, stats = chosen
        exclusions = tuple(f":(exclude){p}" for p in (".venv", *scaffolding))
        reason = _apply(worktree_path, ref)
        if reason is not None:
            try:
                _restore_base(worktree_path, head, exclusions)
            except ResumeRestoreError as exc:
                _emit_resume_failed(state_file, issue_number, ref.name, f"{reason}; {exc}")
                raise
            raise RuntimeError(reason)
    except ResumeRestoreError:
        raise  # never launch a worker on a tree that is not provably the base
    except Exception as exc:  # noqa: BLE001 -- resume must never block a dispatch
        _emit_resume_failed(state_file, issue_number, ref.name if ref else None, str(exc))
        return None
    resumed = ResumedAttempt(ref.name, ref.kind, stats[0], stats[1])
    _emit_resumed(state_file, issue_number, resumed)
    return resumed


__all__ = [
    "ResumeRestoreError",
    "ResumedAttempt",
    "apply_resume_notice",
    "render_resume_notice",
    "seed_from_throttled_attempt",
]
