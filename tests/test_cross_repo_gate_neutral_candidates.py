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


def test_authority_citation_alone_still_abstains_with_narrowed_markers(
    tmp_path: Path,
) -> None:
    """Review guard: dropping "see"/"per"/"line N" from the evidence
    vocabulary must not break the marker words that stayed. A single
    "Authority: <path> section N" citation -- no "see", no "per", no
    "logged to" -- is still enough on its own to classify the candidate
    neutral and abstain."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    evidence_path = "docs/decisions/x.md"
    body = f"Authority: `{evidence_path}` section 2"

    result = cross_repo_gate(body, repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert result.neutral_paths == (evidence_path,)
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


def test_see_and_line_pinpoint_citation_still_escalates(tmp_path: Path) -> None:
    """Review guard: "see `path` line N" is the ordinary way a bug report
    pinpoints the file the worker must EDIT, not an evidence/authority
    citation -- "see" and "line N" were deliberately dropped from the
    marker vocabulary so this body does not go all-neutral and abstain
    into dispatch against the wrong repo. Both candidates are repo-shaped
    (real, non-ignored `src/`/`tests/` dirs exist) but missing -- the gate
    must still escalate."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()
    (repo / "tests").mkdir()
    body = (
        "See `src/charlie_work/nonexistent_module.py` line 42 for the bug; "
        "also `tests/test_nonexistent.py` line 7"
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert set(result.referenced_paths) == {
        "src/charlie_work/nonexistent_module.py",
        "tests/test_nonexistent.py",
    }
    assert set(result.missing_paths) == {
        "src/charlie_work/nonexistent_module.py",
        "tests/test_nonexistent.py",
    }
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


# --- (f): citation-section headings (issue #1583) ------------------------
#
# A bullet list under a ``## Provenance`` heading is this fleet's house
# style for citing the evidence behind an issue. None of the clause-local
# marker words appear in a typical provenance bullet
# (``- Numbers: raw/analyses/.../foo.json (live)``), so before #1583 the
# most common citation shape in this repo's own issues was invisible to the
# neutral classifier and escalated at campaign scale (9 issues, #1538 +
# #1547-#1554). Section scope is a stronger signal than a clause-local word
# and is derived from the body's own structure.

# The two surviving paths from the #1538/#1547-#1554 campaign issues: a
# llibrary ``raw/analyses/...`` provenance citation and a repo-prefixed
# ``charlie-work/.attachment-budgets.json``. Both are absent from this repo
# and neither carries a clause-local marker word -- before #1583 they were
# the only survivors and the gate escalated.
_1583_PROVENANCE_PATH = (
    "raw/analyses/2026-09-god-object-paydown/charlie-work-priority-candidates.json"
)
_1583_BUDGET_PATH = "charlie-work/.attachment-budgets.json"


def _1583_shaped_body() -> str:
    """Trimmed synthetic body matching the #1538 campaign-issue shape: a
    ``## Provenance`` section whose bullets cite foreign analysis/artifact
    paths with no clause-local evidence marker words -- the exact shape
    that escalated 9 issues before #1583."""
    return (
        "## What happened\n\n"
        "The gate escalated because every referenced path was absent.\n\n"
        "## Provenance\n\n"
        f"- Numbers: {_1583_PROVENANCE_PATH} (live)\n"
        f"- Budgets: {_1583_BUDGET_PATH}\n"
    )


def test_1583_provenance_section_citations_abstain(tmp_path: Path) -> None:
    """Issue #1583: candidates under a ``## Provenance`` heading are
    neutral -- the gate abstains instead of escalating, with both
    provenance citations reported in ``neutral_paths`` and zero survivors.
    This is the #1538 campaign-issue shape (a ``## Provenance`` section
    citing foreign analysis/budget paths with no clause-local marker
    words) that escalated 9 issues before the fix."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)

    result = cross_repo_gate(_1583_shaped_body(), repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert set(result.neutral_paths) == {_1583_PROVENANCE_PATH, _1583_BUDGET_PATH}
    assert "abstaining" in result.reason


def test_provenance_section_foreign_path_abstains(tmp_path: Path) -> None:
    """A ``## Provenance`` section citing a genuinely foreign path (a
    sibling-repo analysis doc, absent from this repo, no clause-local
    marker word) abstains -- section scope is the citation signal, not the
    path's existence or a marker word. This guards the fix against a body
    that cites a foreign path the gate has never seen before."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    foreign_path = "llibrary/raw/analyses/2026-09/some-analysis.json"
    body = (
        "## Context\n\n"
        "The decision is documented elsewhere.\n\n"
        "## Provenance\n\n"
        f"- Source: {foreign_path}\n"
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert result.neutral_paths == (foreign_path,)
    assert "abstaining" in result.reason


def test_changes_section_missing_path_still_escalates(tmp_path: Path) -> None:
    """Guard that the citation-section fix does not widen the neutral set
    beyond citation sections: a body whose only path is in a ``## Changes``
    section and missing still escalates. ``Changes`` is not a citation
    heading, so the candidate survives and the gate blocks -- the fix must
    not neutralize a genuine dispatch target just because it sits under a
    heading."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()  # real, non-ignored top-level dir -- keeps the
    # candidate genuinely repo-shaped-but-missing rather than tripping the
    # single-candidate ambiguous-fragment exception.
    body = "## Changes\n\nThe fix is in `src/charlie_work/nonexistent.py`.\n"

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == ("src/charlie_work/nonexistent.py",)
    assert result.missing_paths == ("src/charlie_work/nonexistent.py",)
    assert result.neutral_paths == ()
    assert "cross_repo_target: all 1 referenced file path(s)" in result.reason


def test_references_and_see_also_headings_also_neutral(tmp_path: Path) -> None:
    """The citation-section vocabulary is ``provenance``, ``references``,
    ``sources``, and ``see also`` -- all four headings neutralize a
    candidate under them. This pins the full vocabulary, not just
    ``Provenance``, so a future narrowing that drops one word is caught."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    foreign_path = "other-repo/docs/analysis.json"
    for heading in ("References", "Sources", "See Also"):
        body = f"## {heading}\n\n- {foreign_path}\n"
        result = cross_repo_gate(body, repo)
        assert result.passed is True, f"heading {heading!r} should abstain"
        assert result.neutral_paths == (foreign_path,), (
            f"heading {heading!r} should neutralize the candidate"
        )
        assert result.referenced_paths == ()


def test_provenance_marker_word_alone_neutralizes(tmp_path: Path) -> None:
    """Issue #1583 fix part 2: ``provenance`` and ``origin`` were added to
    ``_EVIDENCE_MARKER_RE``. A candidate introduced by ``provenance:`` in
    its own clause -- with no ``## Provenance`` heading -- is neutral via
    the clause-local marker, the same rule as the existing four words."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    evidence_path = "llibrary/docs/analyses/note.json"
    body = f"Provenance: {evidence_path}"

    result = cross_repo_gate(body, repo)

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.missing_paths == ()
    assert result.neutral_paths == (evidence_path,)
    assert "abstaining" in result.reason


def test_origin_git_remote_usage_does_not_neutralize_dispatch_target(
    tmp_path: Path,
) -> None:
    """Review finding (PR #1584 round 1): ``origin`` was removed from
    ``_EVIDENCE_MARKER_RE`` because it is an ordinary git-remote name
    (``origin/main``, ``git push origin``) that appears constantly in issue
    bodies near genuine dispatch targets. A body where ``origin`` appears in
    the same clause as a genuinely missing repo-shaped path must still
    escalate -- the ``origin`` mention must NOT neutralize the dispatch
    target. This is the exact false-negative (cross-repo contamination) the
    module exists to prevent.

    The body deliberately puts ``origin/main`` in the same clause (no
    clause-boundary punctuation before the path) as the dispatch target, so
    the old unscoped ``origin`` marker would have neutralized it. With
    ``origin`` removed from ``_EVIDENCE_MARKER_RE`` the candidate survives
    and the gate escalates."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()  # real, non-ignored top-level dir -- keeps the
    # candidate genuinely repo-shaped-but-missing rather than tripping the
    # single-candidate ambiguous-fragment exception.
    dispatch_target = "src/charlie_work/nonexistent.py"
    body = f"After merging origin/main the bug in `{dispatch_target}` still reproduces."

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == (dispatch_target,)
    assert result.missing_paths == (dispatch_target,)
    assert result.neutral_paths == ()
    assert "cross_repo_target" in result.reason


def test_non_citation_heading_does_not_neutralize(tmp_path: Path) -> None:
    """A heading that is NOT in the citation vocabulary (``## Details``)
    does not neutralize a candidate under it -- the section-scope signal is
    the heading *word*, not the presence of any heading. Without this guard
    the fix would over-neutralize every path that happens to sit under any
    heading."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()
    body = "## Details\n\nThe bug is in `src/charlie_work/nonexistent.py`.\n"

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == ("src/charlie_work/nonexistent.py",)
    assert result.neutral_paths == ()


def test_heading_containing_citation_word_as_substring_does_not_neutralize(
    tmp_path: Path,
) -> None:
    """Review finding (PR #1584 round 1): the citation-section heading regex
    is anchored so a heading that merely *contains* a citation word as a
    substring does NOT neutralize paths under it. ``## Code References That
    Must Change`` contains ``references`` as a word, but the heading's core
    meaning is about code that must change -- the paths listed under it are
    genuine dispatch targets, not citations. The old unanchored
    ``\\b...\\b`` search regex would have neutralized them; the anchored
    ``^...$`` regex does not."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()  # real, non-ignored top-level dir -- keeps the
    # candidate genuinely repo-shaped-but-missing rather than tripping the
    # single-candidate ambiguous-fragment exception.
    dispatch_target = "src/charlie_work/nonexistent.py"
    body = (
        f"## Code References That Must Change\n\nThe primary fix target is `{dispatch_target}`.\n"
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == (dispatch_target,)
    assert result.missing_paths == (dispatch_target,)
    assert result.neutral_paths == ()
    assert "cross_repo_target" in result.reason


def test_fenced_code_block_citation_heading_does_not_neutralize(
    tmp_path: Path,
) -> None:
    """Review finding (PR #1584 round 3): a ``#``-prefixed line inside a
    fenced code block is code content, not a structural markdown heading.
    A code sample containing the literal line ``# References`` must NOT
    shadow the real preceding ``## Changes`` heading and neutralize the
    genuine dispatch target listed after the code block. The old
    ``_HEADING_LINE_RE.finditer`` walk matched every ``#``-prefixed line
    regardless of context, so the fenced ``# References`` became the
    "nearest preceding heading" and the dispatch target was wrongly
    neutralized -- the same cross-repo-contamination risk class round 1
    already blocked twice, reopened via a third vector."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()  # real, non-ignored top-level dir -- keeps the
    # candidate genuinely repo-shaped-but-missing rather than tripping the
    # single-candidate ambiguous-fragment exception.
    dispatch_target = "src/charlie_work/nonexistent.py"
    body = (
        "## Changes\n\n"
        "Reproducer:\n\n"
        "```python\n"
        "# References\n"
        "import foo\n"
        "```\n\n"
        f"The fix is in `{dispatch_target}`.\n"
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == (dispatch_target,)
    assert result.missing_paths == (dispatch_target,)
    assert result.neutral_paths == ()
    assert "cross_repo_target" in result.reason


def test_indented_code_block_citation_heading_does_not_neutralize(
    tmp_path: Path,
) -> None:
    """Review finding (PR #1584 round 3): a ``#``-prefixed line inside an
    indented code block (4+ leading spaces, per CommonMark) is code
    content, not a structural heading. A code sample containing the
    literal line ``    # References`` must NOT shadow the real preceding
    ``## Changes`` heading and neutralize the genuine dispatch target
    listed after it. The old ``_HEADING_LINE_RE`` allowed arbitrary
    leading whitespace, so the indented ``# References`` matched as a
    heading and became the nearest preceding heading -- neutralizing the
    dispatch target."""
    repo = tmp_path / "repo"
    _init_git_repo_with_var_gitignore(repo)
    (repo / "src").mkdir()  # real, non-ignored top-level dir -- keeps the
    # candidate genuinely repo-shaped-but-missing rather than tripping the
    # single-candidate ambiguous-fragment exception.
    dispatch_target = "src/charlie_work/nonexistent.py"
    body = (
        "## Changes\n\n"
        "Reproducer:\n\n"
        "    # References\n"
        "    import foo\n\n"
        f"The fix is in `{dispatch_target}`.\n"
    )

    result = cross_repo_gate(body, repo)

    assert result.passed is False
    assert result.referenced_paths == (dispatch_target,)
    assert result.missing_paths == (dispatch_target,)
    assert result.neutral_paths == ()
    assert "cross_repo_target" in result.reason
