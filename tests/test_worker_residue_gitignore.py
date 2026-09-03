"""Repo-hygiene regression for issues #1138 and #1204.

Worker/orchestrator protocol residue (``.worker-outcome.json``,
``.charlie-writer.json``) is state written into each worktree root at runtime,
not source. Tracking either file on ``main`` materializes a phantom outcome in
every fresh worktree and turns a worker's own write into a modification to a
tracked file -- the exact shape of dirt that trips the ``worktree_unsafe``
escalation path. These tests pin the structural prevention: the names are
gitignored and not tracked.

Issue #1204 extended the same mechanism to worker PR-body scratch files
(``PR_BODY_*.md``): workers ad-hoc draft their PR body into a
``PR_BODY_<issue>.md`` file in the worktree root, and salvage squash #1197
committed ``PR_BODY_1010.md`` to main. The glob pattern closes the class for
any issue number.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.config import WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME

# The protocol residue filenames that every worker worktree has written into
# its root. Sourced from the canonical constants rather than re-declared, so a
# rename in config.py flows through here instead of silently rotting the guard.
_RESIDUE_NAMES = (WORKER_OUTCOME_FILENAME, WRITER_MARKER_FILENAME)

# Worker PR-body scratch glob pattern. The worker prompt does not name a file,
# but workers ad-hoc write ``PR_BODY_<issue>.md`` into the worktree root. This
# is a glob, not a fixed filename, so it is declared here rather than sourced
# from a config constant. A sample concrete name is used for ``check-ignore``
# and ``ls-files`` exercises below.
_PR_BODY_GLOB = "PR_BODY_*.md"
_PR_BODY_SAMPLE = "PR_BODY_9999.md"

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def test_worker_residue_filenames_are_gitignored() -> None:
    """Each protocol residue filename must appear in ``.gitignore``.

    Without this, a salvage that ``git add -A``s a worktree with residue
    re-commits the sentinel -- the recurrence path that put
    ``.worker-outcome.json`` on ``main`` via PR #1114 in the first place.
    """
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [name for name in _RESIDUE_NAMES if name not in gitignore]
    assert not missing, f"worker residue not in .gitignore: {missing}"


def test_worker_residue_filenames_are_not_tracked() -> None:
    """No protocol residue filename may be tracked on this branch.

    ``git ls-files`` against the current index must not list either sentinel.
    This is the positive control for the ``git rm`` half of the fix: it fails
    the moment a tracked copy reappears.
    """
    tracked = set(_git("ls-files").splitlines())
    tracked_residue = sorted(tracked.intersection(_RESIDUE_NAMES))
    assert not tracked_residue, f"worker residue tracked as source: {tracked_residue}"


def test_worker_residue_is_ignored_by_git_check_ignore() -> None:
    """``git check-ignore`` must report each residue filename as ignored.

    This exercises git's actual ignore resolution (not just a substring match
    on ``.gitignore``), so a malformed pattern that fails to match is caught.
    """
    result = subprocess.run(
        ["git", "check-ignore", *_RESIDUE_NAMES],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ignored = set(result.stdout.splitlines())
    not_ignored = sorted(set(_RESIDUE_NAMES) - ignored)
    assert not not_ignored, f"git check-ignore did not match: {not_ignored}"


def test_pr_body_scratch_glob_is_gitignored() -> None:
    """The ``PR_BODY_*.md`` glob must appear in ``.gitignore``.

    Salvage squash #1197 committed ``PR_BODY_1010.md`` to main (issue #1204)
    because the worker's ad-hoc PR-body scratch file was not gitignored. The
    glob pattern closes the class for any issue number.
    """
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert _PR_BODY_GLOB in gitignore, f"PR-body scratch glob {_PR_BODY_GLOB!r} not in .gitignore"


def test_pr_body_scratch_glob_is_ignored_by_git_check_ignore() -> None:
    """``git check-ignore`` must resolve a concrete ``PR_BODY_<n>.md`` name.

    Exercises git's actual ignore resolution against the glob, so a malformed
    pattern (e.g. a missing wildcard) is caught.
    """
    result = subprocess.run(
        ["git", "check-ignore", _PR_BODY_SAMPLE],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ignored = set(result.stdout.splitlines())
    assert _PR_BODY_SAMPLE in ignored, (
        f"git check-ignore did not match {_PR_BODY_SAMPLE!r} against {_PR_BODY_GLOB!r}"
    )


def test_no_pr_body_scratch_files_are_tracked() -> None:
    """No file matching ``PR_BODY_*.md`` may be tracked on this branch.

    This is the positive control for the ``git rm`` half of the #1204 fix: it
    fails the moment a tracked ``PR_BODY_*.md`` copy reappears.
    """
    tracked = _git("ls-files").splitlines()
    tracked_pr_body = sorted(
        name for name in tracked if name.startswith("PR_BODY_") and name.endswith(".md")
    )
    assert not tracked_pr_body, f"PR-body scratch files tracked as source: {tracked_pr_body}"


# The ad-hoc PR-body spellings worktree.py's _LAUNCHER_OWNED_PR_BODY_RE already
# recognizes (issue #1391) but .gitignore did not, so a `_pr_body.md` (issue
# #1541 rework) or `.pr_body_<issue>.md` (issue #1418) scratch file slipped
# past `git add`. Each entry is (glob_in_gitignore, concrete_sample_for_check_ignore).
_PR_BODY_VARIANT_GLOBS = (
    ("_pr_body*.md", "_pr_body.md"),
    (".pr_body*.md", ".pr_body_1418.md"),
    (".worker-pr-body*.md", ".worker-pr-body.md"),
)


def test_pr_body_variant_globs_are_gitignored() -> None:
    """Every ad-hoc PR-body spelling the launcher-owned regex recognizes must
    also be gitignored.

    The dirty-check regex (``_LAUNCHER_OWNED_PR_BODY_RE``) already ignores
    these for the unsafe/dirty path, but .gitignore is a separate layer. Issue
    #1541 committed ``_pr_body.md`` because the glob was missing here; this
    pins the ignore-layer coverage so the same class cannot recur via ``git
    add``.
    """
    gitignore = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    missing = [glob for glob, _ in _PR_BODY_VARIANT_GLOBS if glob not in gitignore]
    assert not missing, f"PR-body variant globs not in .gitignore: {missing}"


def test_pr_body_variant_globs_are_ignored_by_git_check_ignore() -> None:
    """``git check-ignore`` must resolve a concrete name for each variant glob.

    Exercises git's actual ignore resolution (not a substring match), so a
    malformed pattern that fails to match is caught.
    """
    samples = [sample for _, sample in _PR_BODY_VARIANT_GLOBS]
    result = subprocess.run(
        ["git", "check-ignore", *samples],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ignored = set(result.stdout.splitlines())
    not_ignored = sorted(set(samples) - ignored)
    assert not not_ignored, f"git check-ignore did not match: {not_ignored}"


def test_no_pr_body_variant_scratch_files_are_tracked() -> None:
    """No PR-body scratch file in any ad-hoc spelling may be tracked.

    Positive control for the ``git rm`` half of the #1541 rework fix: the
    ``_pr_body.md`` that was committed is removed, and this fails the moment
    any variant spelling reappears as a tracked file.
    """
    tracked = set(_git("ls-files").splitlines())
    samples = {sample for _, sample in _PR_BODY_VARIANT_GLOBS}
    leaked = sorted(tracked & samples)
    assert not leaked, f"PR-body variant scratch files tracked as source: {leaked}"
