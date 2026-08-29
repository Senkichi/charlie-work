"""Pin test for issue #1452's CURRENT body shape against ``cross_repo_gate``.

Issue #1452 was escalated by the cross-repo gate at 2026-08-25T02:15:29Z. At
that moment the issue's body referenced only foreign ``job_finder/*``
evidence paths absent from this repo -- a correct escalation.

The issue body was later edited (2026-08-25T02:28:39Z) to also reference
in-repo files that back the fix (``src/charlie_work/citation_check.py``,
``src/charlie_work/workflow.py``, ``tests/test_citation_check.py``,
``tests/test_citation_drift_dispatch.py``). Running the gate against the
CURRENT body (verified empirically against the live issue #1452 body,
fetched via ``gh issue view 1452``) yields: 5 ``job_finder/*`` candidates
(missing, foreign to this repo, cited plainly via "->" arrows with no
evidence/authority marker words), 4 in-repo candidates (present), and one
gitignored runtime-artifact candidate
(``.var/charlie-work/issues/issue-1933/BLOCKED.md``, classified neutral --
see ``charlie_work.cross_repo_gate``'s module docstring) -- so the gate
PASSES (at least one referenced path exists), with the gitignored candidate
reported separately in ``neutral_paths`` rather than folded into
``missing_paths``. No change to ``cross_repo_gate`` or
``extract_referenced_paths`` was needed for #1452's own escalation
correctness; this test pins the current-body shape so a future regression in
path extraction, existence-checking, or neutral-candidate classification is
caught.

Uses a trimmed synthetic body of the same shape (not the full issue text) so
the test does not depend on the live GitHub issue body staying byte-for-byte
stable, and a real ``git init``-ed ``tmp_path`` repo (with the same
``.var/`` gitignore rule this repo carries) so the test exercises the
gitignore-based neutral classification the same way production does, rather
than a version where it silently no-ops for lack of a ``.git`` directory.
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


# The 4 paths that exist in this repo (from #1452's current body).
_IN_REPO_PATHS = (
    "src/charlie_work/citation_check.py",
    "src/charlie_work/workflow.py",
    "tests/test_citation_check.py",
    "tests/test_citation_drift_dispatch.py",
)

# The 5 job_finder/* evidence paths foreign to this repo (from #1452's
# current body's "Root cause" bullet list -- cited via "->" arrows, no
# evidence/authority marker words nearby, so they remain survivors of the
# neutral classification and are genuinely reported missing).
_FOREIGN_PATHS = (
    "job_finder/web/ats_prober.py",
    "job_finder/web/careers_crawler/_cohort_legitimacy.py",
    "job_finder/db/_assessment_writer.py",
    "job_finder/db/_jobs.py",
    "job_finder/web/scheduler/_jobs.py",
)

# The gitignored runtime-artifact path from #1452's current body ("A jc
# worker's analysis (jc `.var/.../BLOCKED.md`) ..."). Neutral via gitignore,
# not context -- this repo's own ``.gitignore`` ignores ``.var/``.
_GITIGNORED_PATH = ".var/charlie-work/issues/issue-1933/BLOCKED.md"


def _synthetic_1452_body() -> str:
    """Trimmed synthetic body matching #1452's current-body shape: an
    arrow-cited bullet list of foreign evidence paths (no marker words), a
    "Checker location" section citing the real in-repo fix files, and a
    parenthetical mention of the gitignored jc-worker-analysis artifact."""
    return (
        "citation-drift checker: file_missing false alarms for real files "
        "cited by basename; no ambiguity handling.\n\n"
        "None of the files are missing. All flagged paths exist under real "
        "directories:\n"
        f"- `ats_prober.py` -> `{_FOREIGN_PATHS[0]}`\n"
        f"- `_cohort_legitimacy.py` -> `{_FOREIGN_PATHS[1]}`\n"
        f"- `_assessment_writer.py` -> `{_FOREIGN_PATHS[2]}`\n"
        f"- `_jobs.py` -> AMBIGUOUS: `{_FOREIGN_PATHS[3]}` AND `{_FOREIGN_PATHS[4]}`\n\n"
        "## Checker location (this repo)\n\n"
        f"- `{_IN_REPO_PATHS[0]}` -- entry point\n"
        f"- `{_IN_REPO_PATHS[1]}` -- dispatch-time call site\n"
        f"- `{_IN_REPO_PATHS[2]}`, `{_IN_REPO_PATHS[3]}` -- existing tests\n\n"
        f"A jc worker's analysis (jc `{_GITIGNORED_PATH}`) independently "
        "confirmed the fix shape.\n"
    )


def test_1452_current_body_shape_passes_with_job_finder_paths_missing(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    for rel_path in _IN_REPO_PATHS:
        real_path = tmp_path / rel_path
        real_path.parent.mkdir(parents=True, exist_ok=True)
        real_path.write_text("# placeholder\n", encoding="utf-8")

    result = cross_repo_gate(_synthetic_1452_body(), tmp_path)

    assert result.passed is True
    assert set(result.referenced_paths) == set(_IN_REPO_PATHS) | set(_FOREIGN_PATHS)
    assert set(result.missing_paths) == set(_FOREIGN_PATHS)
    assert result.neutral_paths == (_GITIGNORED_PATH,)
    assert result.reason == "at least one referenced path exists in the target repo"
