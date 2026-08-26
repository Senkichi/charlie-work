"""Cross-repo gate: neutral-candidate classification (issues #1452, #1460).

Ground truth (re-verified against the live issues, not assumed): issue
#1460's body opens with "BLOCKED ON: merge of PR #1459", but PR #1459
*merged* at 2026-08-25T22:20:56Z -- BEFORE both of #1460's escalations
(2026-08-25T22:23 and 2026-08-26T19:59). A "defer dispatch while the
blocker PR is open" fix (an earlier hypothesis for this issue) would never
have helped and is not implemented here -- it is not the discriminator.

The real shared shape between #1452's ORIGINAL body and #1460's body is
that every referenced candidate is either:

- a path cited as evidence/authority/rationale for the issue -- not code
  the worker is meant to touch (#1460: "Authority: llibrary/docs/plans/
  ...DECISION.md section 4 rows 5-6"; #1452's original body cited
  ``job_finder/*`` paths as the source of a false-alarm report); or
- a runtime artifact the issue's own future work will WRITE, which cannot
  exist yet by definition (#1460: "advisories are logged to
  .var/attachment-contracts/advisories.jsonl").

``cross_repo_gate`` now classifies both shapes as "neutral": excluded from
the pass/escalate decision and reported separately via
``CrossRepoGateResult.neutral_paths``. These tests drive ``cross_repo_gate``
directly -- the fix lives entirely in the gate's own candidate
classification, not in dispatch ordering, so a gate-level test is the right
level (unlike the "blocked on" hypothesis this replaced, which would have
needed a dispatch-path integration test to prove gate-ordering).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from charlie_work.cross_repo_gate import cross_repo_gate


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, raising on failure."""
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _init_git_repo_with_var_gitignore(repo: Path) -> None:
    """A real ``git init``-ed repo whose ``.gitignore`` ignores ``.var/`` --
    the same rule this repo itself carries (see ``.gitignore`` line 8)."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init")
    (repo / ".gitignore").write_text(".var/\n", encoding="utf-8")


# --- (a): #1460-shaped body abstains with both candidates neutral --------

_1460_AUTHORITY_PATH = "llibrary/docs/plans/2026-08-24-god-object-mitigation-DECISION.md"
_1460_ARTIFACT_PATH = ".var/attachment-contracts/advisories.jsonl"


def _1460_shaped_body() -> str:
    """Trimmed synthetic body matching #1460's real shape: an "Authority:"
    citation to a sibling-repo decision doc (wrapped onto its own line, plus
    a "section N rows N-M" suffix -- the real body's exact shape) and a
    runtime-artifact write destination (a gitignored ``.var/`` path
    introduced by "logged to")."""
    return (
        "Part of #1458 (pilot tracking). Authority:\n"
        f"{_1460_AUTHORITY_PATH} section 4 rows 5-6.\n\n"
        "When the PreToolUse advisory fires mid-session (advisories are "
        f"logged to {_1460_ARTIFACT_PATH}), the worker should take the "
        "redirect rather than bumping the baseline.\n"
    )


def test_1460_shaped_body_abstains_with_both_candidates_neutral(tmp_path: Path) -> None:
    """The #1460 fix: an evidence/authority citation and a gitignored
    write destination are both neutral -- the gate abstains (passes)
    instead of escalating, with zero survivors and no missing_paths."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)

    result = cross_repo_gate(_1460_shaped_body(), repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert set(result.neutral_paths) == {_1460_AUTHORITY_PATH, _1460_ARTIFACT_PATH}
    assert "abstaining" in result.reason


# --- (c): genuinely wrong-repo bodies still escalate ----------------------


def test_1452_original_shape_no_evidence_marker_still_escalates(tmp_path: Path) -> None:
    """The #1452-ORIGINAL-body shape: several absent paths, cited plainly
    (no evidence/authority marker words, no gitignored paths) -- still
    escalates with the unchanged reason string. This is the guard that the
    neutral classification does not swallow a genuine cross-repo report."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    foreign_paths = (
        "job_finder/web/ats_prober.py",
        "job_finder/web/careers_crawler/_cohort_legitimacy.py",
        "job_finder/db/_assessment_writer.py",
        "job_finder/db/_jobs.py",
        "job_finder/web/scheduler/_jobs.py",
        "job_finder/web/scheduler/_runner.py",
    )
    body = (
        "None of the files are missing. All flagged paths exist under real directories:\n"
        + "\n".join(f"- `{p}`" for p in foreign_paths)
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert set(result.referenced_paths) == set(foreign_paths)
    assert set(result.missing_paths) == set(foreign_paths)
    assert result.neutral_paths == ()
    assert (
        f"cross_repo_target: all {len(foreign_paths)} referenced file path(s) "
        "are absent from the target repo" in result.reason
    )


def test_plain_absent_path_body_still_escalates(tmp_path: Path) -> None:
    """Control from the brief: a body citing plain absent paths with no
    evidence framing and no gitignored paths still escalates -- the neutral
    classification does not swallow an ordinary cross-repo bug report."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()  # a real, non-ignored top-level dir -- keeps
    # this genuinely repo-shaped-but-missing rather than tripping the
    # unrelated single-candidate ambiguous-fragment exception.
    body = "The bug is in `src/foo.py` and the regression test is `tests/test_foo.py`."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert set(result.referenced_paths) == {"src/foo.py", "tests/test_foo.py"}
    assert result.neutral_paths == ()


def test_mixed_evidence_and_plain_absent_paths_still_escalates(tmp_path: Path) -> None:
    """(d) A body mixing one evidence-cited absent path (neutral) with one
    plain absent path (repo-shaped, survives) still escalates -- the plain
    candidate alone is enough to trigger escalation once neutral candidates
    are excluded."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()
    evidence_path = "llibrary/docs/plans/unrelated-design-note.md"
    plain_path = "src/foo.py"
    body = f"Authority: {evidence_path} section 2.\n\nThe actual bug is in `{plain_path}`.\n"

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == (plain_path,)
    assert result.missing_paths == (plain_path,)
    assert result.neutral_paths == (evidence_path,)
    assert "cross_repo_target: all 1 referenced file path(s)" in result.reason


# --- (e): git check-ignore fallback when repo_root is not a git repo -----


def test_check_ignore_fallback_when_repo_root_is_not_a_git_repo(tmp_path: Path) -> None:
    """When repo_root has no ``.git`` directory, ``git check-ignore`` fails
    ("not a git repository") and the fallback treats the candidate as NOT
    ignored -- the narrower, safer default (keep escalating) rather than
    silently suppressing a candidate a broken git invocation could not
    classify. A wrong-polarity fallback ("assume ignored on error") would
    flip this test's outcome to ``passed=True`` via the single-candidate
    exception, so this test discriminates the fallback's direction, not
    just its presence."""
    repo = tmp_path / "no_git_repo"
    repo.mkdir()
    (repo / ".gitignore").write_text(".var/\n", encoding="utf-8")
    # Deliberately no `git init` -- repo_root is a plain directory, not a
    # git repository.
    body = "The state file is `.var/charlie-work/state.json` and the code is in `src/charlie_work/nonexistent.py`."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert set(result.referenced_paths) == {
        ".var/charlie-work/state.json",
        "src/charlie_work/nonexistent.py",
    }
    assert result.neutral_paths == ()
