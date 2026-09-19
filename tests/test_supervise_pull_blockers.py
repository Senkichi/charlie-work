"""``_repair_lossless_pull_blockers`` tests -- real git repos, no mocked git.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

from _supervise_fixtures import no_fleet_live_sessions as no_fleet_live_sessions
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult, run_captured
from charlie_work.supervise import (
    PullBlockerRepair,
    _BLOCKER_NAMES_IN_MESSAGE,
    _repair_lossless_pull_blockers,
    _repairable_blocker_path,
    _self_deploy_state_path,
    self_deploy,
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_origin(origin_root: Path, *, autocrlf: bool = False) -> None:
    origin_root.mkdir(parents=True, exist_ok=True)
    _git(origin_root, "init", "--initial-branch=main")
    _git(origin_root, "config", "user.email", "test@example.test")
    _git(origin_root, "config", "user.name", "Test User")
    _git(origin_root, "config", "core.autocrlf", "true" if autocrlf else "false")
    (origin_root / "README.md").write_bytes(b"hello\n")
    _git(origin_root, "add", "README.md")
    _git(origin_root, "commit", "-m", "initial commit")


def _clone(origin_root: Path, clone_root: Path, *, autocrlf: bool = False) -> None:
    _git(origin_root.parent, "clone", str(origin_root), str(clone_root))
    _git(clone_root, "config", "user.email", "test@example.test")
    _git(clone_root, "config", "user.name", "Test User")
    _git(clone_root, "config", "core.autocrlf", "true" if autocrlf else "false")


def _counting_git_runner() -> tuple[Callable[..., RunResult], list[list[str]]]:
    """Wrap the real ``run_captured`` so calls are recorded but still actually
    executed -- proves a precondition short-circuited before doing
    unnecessary work, without mocking git itself."""
    calls: list[list[str]] = []

    def runner(command: list[str], *, cwd: Path, timeout_seconds: int) -> RunResult:
        calls.append(command)
        return run_captured(command, cwd=cwd, timeout_seconds=timeout_seconds)

    return runner, calls


def test_repair_lossless_pull_blockers_clears_both_classes_and_pull_succeeds(
    tmp_path: Path,
) -> None:
    """One blocker of each refusal class, both byte-identical to what the
    incoming commit would write: both are cleared, retained is empty, and a
    real ``git pull --ff-only`` immediately after succeeds -- the actual
    outcome this feature exists to restore.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "tracked.txt").write_bytes(b"original content\n")
    _git(origin_root, "add", "tracked.txt")
    _git(origin_root, "commit", "-m", "add tracked.txt")
    _clone(origin_root, clone_root)

    # Commit B on origin: modifies tracked.txt and adds a new file.
    (origin_root / "tracked.txt").write_bytes(b"incoming content\n")
    (origin_root / "new_untracked.txt").write_bytes(b"new content\n")
    _git(origin_root, "add", "tracked.txt", "new_untracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    # Local blockers whose content already matches what B would write.
    (clone_root / "new_untracked.txt").write_bytes(b"new content\n")  # untracked shadow
    (clone_root / "tracked.txt").write_bytes(b"incoming content\n")  # modified, unstaged

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ("new_untracked.txt", "tracked.txt")
    assert repair.retained == ()
    assert repair.acted is True

    pull = subprocess.run(
        ["git", "pull", "--ff-only", "origin", "main"],
        cwd=clone_root,
        capture_output=True,
        text=True,
    )
    assert pull.returncode == 0, pull.stderr
    assert (clone_root / "tracked.txt").read_bytes() == b"incoming content\n"
    assert (clone_root / "new_untracked.txt").read_bytes() == b"new content\n"


def test_repair_lossless_pull_blockers_mixed_clears_lossless_retains_genuine_conflict(
    tmp_path: Path,
) -> None:
    """A lossless untracked blocker and a lossless modified blocker are
    cleared; a modified blocker whose local content genuinely differs from
    both HEAD and the incoming commit is retained, untouched, on disk.
    Partial clearing is the documented behaviour, not a bug to fix.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "tracked.txt").write_bytes(b"original content\n")
    (origin_root / "divergent.txt").write_bytes(b"original divergent\n")
    _git(origin_root, "add", "tracked.txt", "divergent.txt")
    _git(origin_root, "commit", "-m", "add tracked files")
    _clone(origin_root, clone_root)

    (origin_root / "tracked.txt").write_bytes(b"incoming content\n")
    (origin_root / "divergent.txt").write_bytes(b"incoming divergent v2\n")
    (origin_root / "new_untracked.txt").write_bytes(b"new content\n")
    _git(origin_root, "add", "tracked.txt", "divergent.txt", "new_untracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    (clone_root / "new_untracked.txt").write_bytes(b"new content\n")
    (clone_root / "tracked.txt").write_bytes(b"incoming content\n")
    # Genuinely different from BOTH HEAD's version and the incoming version.
    (clone_root / "divergent.txt").write_bytes(b"local unique edit, not incoming\n")

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ("new_untracked.txt", "tracked.txt")
    assert repair.retained == ("divergent.txt",)
    assert (clone_root / "divergent.txt").read_bytes() == b"local unique edit, not incoming\n"


def test_repair_lossless_pull_blockers_diverged_clears_nothing(tmp_path: Path) -> None:
    """A local commit origin lacks means this is not a pending fast-forward at
    all -- the precondition must refuse before touching anything, even a
    blocker that would otherwise be provably lossless. The call trace pins
    that the refusal happens specifically at the ancestor check (not, say,
    an accidental early return that would also mask a real bug), and the
    blocker's untouched content proves ``git checkout HEAD --`` never ran:
    if it had, it would restore HEAD's ("local-only" commit's inherited)
    version, not leave our injected content in place.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "tracked.txt").write_bytes(b"original content\n")
    _git(origin_root, "add", "tracked.txt")
    _git(origin_root, "commit", "-m", "add tracked.txt")
    _clone(origin_root, clone_root)

    # Origin advances...
    (origin_root / "tracked.txt").write_bytes(b"incoming content\n")
    _git(origin_root, "add", "tracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    # ...while the clone independently commits something origin never saw.
    (clone_root / "local_only.txt").write_bytes(b"local commit content\n")
    _git(clone_root, "add", "local_only.txt")
    _git(clone_root, "commit", "-m", "local-only commit")

    # A blocker that WOULD be lossless if the precondition were skipped.
    (clone_root / "tracked.txt").write_bytes(b"incoming content\n")

    runner, calls = _counting_git_runner()
    repair = _repair_lossless_pull_blockers(clone_root, run_command=runner, timeout=30)

    assert repair == PullBlockerRepair()
    assert len(calls) == 3
    assert calls[0] == ["git", "rev-parse", "HEAD"]
    assert calls[1] == ["git", "rev-parse", "origin/main"]
    assert calls[2][:3] == ["git", "merge-base", "--is-ancestor"]
    # Not reverted to HEAD's inherited content -- proves checkout never ran.
    assert (clone_root / "tracked.txt").read_bytes() == b"incoming content\n"


def test_repair_lossless_pull_blockers_up_to_date_clears_nothing(tmp_path: Path) -> None:
    """HEAD already at ``origin/main`` -- there is no pending fast-forward, so
    the precondition refuses immediately. The call trace pins that only the
    two rev-parse calls happen: no ancestor check, no blocker computation at
    all, proving this is the equality short-circuit specifically and not
    some other path that happens to also land on an empty result.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    runner, calls = _counting_git_runner()
    repair = _repair_lossless_pull_blockers(clone_root, run_command=runner, timeout=30)

    assert repair == PullBlockerRepair()
    assert calls == [
        ["git", "rev-parse", "HEAD"],
        ["git", "rev-parse", "origin/main"],
    ]


def test_repair_lossless_pull_blockers_crlf_worktree_matches_lf_blob(tmp_path: Path) -> None:
    """With ``core.autocrlf=true`` and CRLF bytes in the worktree, a file whose
    blob is LF-only must still be recognised as lossless. ``git hash-object``
    must run with filters ON (the default, never ``--no-filters``) or this
    sweep goes silently inert on every Windows checkout with autocrlf enabled
    -- a real trap on this machine, not a hypothetical one.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root, autocrlf=False)
    _clone(origin_root, clone_root, autocrlf=True)

    # Origin adds a file with LF-only line endings.
    (origin_root / "crlf_target.txt").write_bytes(b"line one\nline two\n")
    _git(origin_root, "add", "crlf_target.txt")
    _git(origin_root, "commit", "-m", "advance origin with LF file")
    _git(clone_root, "fetch", "origin")

    # The clone's untracked copy has the SAME content but CRLF line endings --
    # what a Windows checkout with autocrlf=true would actually produce.
    (clone_root / "crlf_target.txt").write_bytes(b"line one\r\nline two\r\n")

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ("crlf_target.txt",)
    assert repair.retained == ()
    assert not (clone_root / "crlf_target.txt").exists()


def test_repair_lossless_pull_blockers_staged_unique_work_is_retained(tmp_path: Path) -> None:
    """Worktree content that happens to equal the incoming blob is not enough
    when the INDEX holds something else: staging is where the unique work
    actually lives, and ``git checkout HEAD --`` would destroy it along with
    the worktree copy. Must be retained, and the staged content must survive
    untouched.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    (origin_root / "staged.txt").write_bytes(b"ancestor\n")
    _git(origin_root, "add", "staged.txt")
    _git(origin_root, "commit", "-m", "add staged.txt")
    _clone(origin_root, clone_root)

    (origin_root / "staged.txt").write_bytes(b"incoming staged content\n")
    _git(origin_root, "add", "staged.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    # Stage a locally-authored, unique change...
    (clone_root / "staged.txt").write_bytes(b"staged unique work\n")
    _git(clone_root, "add", "staged.txt")
    # ...then overwrite the WORKTREE (not the index) to match what origin
    # would write -- content-identical worktree, divergent index.
    (clone_root / "staged.txt").write_bytes(b"incoming staged content\n")

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ()
    assert repair.retained == ("staged.txt",)
    staged_index_content = subprocess.run(
        ["git", "show", ":staged.txt"],
        cwd=clone_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert staged_index_content == "staged unique work\n"


def test_repairable_blocker_path_refuses_venv_and_escaping_paths(tmp_path: Path) -> None:
    """The path gate every candidate runs through before content is even
    considered: ``.venv/*`` is refused unconditionally (repair must never
    touch the running interpreter), a path escaping ``repo_root`` is
    refused, and an ordinary contained path is the positive control proving
    the gate isn't just returning ``False`` for everything.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    assert _repairable_blocker_path(repo_root, ".venv/pyvenv.cfg") is False
    assert _repairable_blocker_path(repo_root, ".venv/Lib/site-packages/foo.py") is False
    assert _repairable_blocker_path(repo_root, "../outside.txt") is False
    assert _repairable_blocker_path(repo_root, "ordinary.txt") is True


def test_repair_lossless_pull_blockers_refuses_venv_path_even_when_lossless(
    tmp_path: Path,
) -> None:
    """A ``.venv/...`` blocker that is byte-identical to the incoming blob
    must still be retained and left on disk -- content-identity is not
    sufficient license to touch the orchestrator's own interpreter tree.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    (origin_root / ".venv").mkdir()
    (origin_root / ".venv" / "marker.txt").write_bytes(b"venv marker\n")
    _git(origin_root, "add", ".venv/marker.txt")
    _git(origin_root, "commit", "-m", "advance origin with venv marker")
    _git(clone_root, "fetch", "origin")

    (clone_root / ".venv").mkdir()
    (clone_root / ".venv" / "marker.txt").write_bytes(b"venv marker\n")  # byte-identical

    repair = _repair_lossless_pull_blockers(clone_root, run_command=run_captured, timeout=30)

    assert repair.cleared == ()
    assert repair.retained == (".venv/marker.txt",)
    assert (clone_root / ".venv" / "marker.txt").read_bytes() == b"venv marker\n"


def test_pull_blocker_repair_describe_and_acted() -> None:
    """``.acted`` reflects whether anything was cleared (drives the
    self-deploy retry decision); ``describe()`` names cleared and retained
    paths for the escalation message an operator reads.
    """
    empty = PullBlockerRepair()
    assert empty.acted is False
    assert empty.describe() == ""

    cleared_only = PullBlockerRepair(cleared=("a.txt", "b.txt"))
    assert cleared_only.acted is True
    description = cleared_only.describe()
    assert "a.txt" in description
    assert "b.txt" in description
    assert "auto-cleared" in description

    mixed = PullBlockerRepair(cleared=("a.txt",), retained=("c.txt",))
    assert mixed.acted is True
    mixed_description = mixed.describe()
    assert "a.txt" in mixed_description
    assert "c.txt" in mixed_description
    assert "still need a human" in mixed_description

    # Retained-only: something needs a human, but nothing on disk changed --
    # `.acted` must track "did disk change", not "is there anything to report".
    retained_only = PullBlockerRepair(retained=("d.txt",))
    assert retained_only.acted is False
    assert "d.txt" in retained_only.describe()


def test_pull_blocker_repair_describe_bounds_long_path_lists() -> None:
    """``describe()`` names at most ``_BLOCKER_NAMES_IN_MESSAGE`` paths per class.

    The string reaches ``SelfDeployResult.error`` and is re-rendered into a
    ``self_deploy_failed`` payload once per retry for the whole duration of a
    wedge, so an unbounded join is a slow leak and an unreadable operator
    message. The exact COUNT must survive truncation -- that is the number an
    operator acts on -- so it is asserted separately from the names.
    """
    many = tuple(f"f{i:02d}.txt" for i in range(25))
    description = PullBlockerRepair(cleared=many).describe()

    assert "25" in description, "the full count must survive truncation"
    assert "f00.txt" in description
    assert f"f{_BLOCKER_NAMES_IN_MESSAGE - 1:02d}.txt" in description
    assert f"f{_BLOCKER_NAMES_IN_MESSAGE:02d}.txt" not in description, "not truncated"
    assert f"+{25 - _BLOCKER_NAMES_IN_MESSAGE} more" in description

    # Boundary: exactly at the limit is rendered in full, with no "+N more".
    exact = tuple(f"g{i:02d}.txt" for i in range(_BLOCKER_NAMES_IN_MESSAGE))
    exact_description = PullBlockerRepair(retained=exact).describe()
    assert f"g{_BLOCKER_NAMES_IN_MESSAGE - 1:02d}.txt" in exact_description
    assert "more" not in exact_description


def test_self_deploy_end_to_end_repairs_lossless_blocker_and_retries_pull(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Wiring test through the public ``self_deploy`` entry point: a real
    clone with exactly one lossless blocker ends up ``ok`` via the single
    retry, rather than surfacing the original pull failure -- the actual
    outcome this feature exists to produce end to end, not just at the
    ``_repair_lossless_pull_blockers`` unit level.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    (origin_root / "new_untracked.txt").write_bytes(b"new content\n")
    _git(origin_root, "add", "new_untracked.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    (clone_root / "new_untracked.txt").write_bytes(b"new content\n")

    result = self_deploy(clone_root, run_command=run_captured)

    assert result.ok is True
    assert result.pulled is True
    assert (clone_root / "new_untracked.txt").read_bytes() == b"new content\n"

    # A repair that succeeds leaves an ordinary `self_deploy_succeeded` behind
    # and is otherwise indistinguishable from a pass where nothing was wrong.
    # The audit record is the only thing that keeps a RECURRING cause visible,
    # so assert it on the success path specifically -- the failure path already
    # names the paths in the error string.
    cleared_events = query_events(
        _self_deploy_state_path(clone_root), kind="self_deploy_blockers_cleared"
    )
    assert len(cleared_events) == 1, "successful repair must still be recorded"
    payload = cleared_events[0]["payload"]
    assert payload["cleared"] == ["new_untracked.txt"]
    assert payload["retained"] == []
    assert payload["pull_ok_after_retry"] is True


def test_self_deploy_records_nothing_cleared_when_no_repair_was_needed(
    tmp_path: Path, no_fleet_live_sessions: None
) -> None:
    """Negative control for the audit record above.

    Without this, the assertion that a repair emits exactly one
    ``self_deploy_blockers_cleared`` is equally consistent with the event
    firing on *every* deploy. An ordinary clean fast-forward must emit none.
    """
    origin_root = tmp_path / "origin"
    clone_root = tmp_path / "clone"
    _init_origin(origin_root)
    _clone(origin_root, clone_root)

    (origin_root / "clean.txt").write_bytes(b"clean\n")
    _git(origin_root, "add", "clean.txt")
    _git(origin_root, "commit", "-m", "advance origin")
    _git(clone_root, "fetch", "origin")

    result = self_deploy(clone_root, run_command=run_captured)

    assert result.ok is True
    assert (
        query_events(_self_deploy_state_path(clone_root), kind="self_deploy_blockers_cleared")
        == []
    )
