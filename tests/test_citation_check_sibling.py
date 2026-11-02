"""Same-body sibling disambiguation tests for ``charlie_work.citation_check`` (issue #1881).

When a citation's literal path does not exist and several tracked files share
its basename, ``verify_citations`` consults *qualified* citations elsewhere in
the same body (citations whose asserted literal path exists) before flagging
``AMBIGUOUS_BASENAME``. Evidence is tiered, strongest first:

1. **Exact path** -- the same file is cited elsewhere in the body.
2. **Same directory** -- a *different* file in the same (non-root) directory is
   cited elsewhere in the body.

Each tier must pick out exactly one candidate; otherwise the citation stays
``AMBIGUOUS_BASENAME``. These tests live apart from
``tests/test_citation_check.py`` because that module is at its frozen
attachment-point ceiling (``.attachment-budgets/``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.citation_check import (
    CitationStatus,
    drift_fingerprint,
    drifted_verdicts,
    verify_citations,
)


def _write(repo: Path, rel: str, lines: list[str]) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _posix(resolved_path: str | None) -> str:
    assert resolved_path is not None
    return resolved_path.replace("\\", "/")


# --------------------------------------------------------------------------- #
# tier 1: exact-path sibling
# --------------------------------------------------------------------------- #


def test_verify_sibling_citation_disambiguates_ambiguous_basename(
    tmp_path: Path,
) -> None:
    # Issue #1881 scenario (from Senkichi/jobcannon#419): the body cites bare
    # ``config.py:259`` alongside an already-qualified
    # ``jobcannon/host/config.py:180-182``. The basename ``config.py`` matches
    # more than one tracked file, but the qualified sibling's resolved path
    # picks out exactly one candidate, so the ambiguous citation resolves
    # deterministically. The bare citation reports info-level
    # ``RESOLVED_BY_SIBLING`` -- NOT ``AMBIGUOUS_BASENAME`` -- so it stays out
    # of ``drifted_verdicts`` and produces an empty fingerprint. The qualified
    # sibling is cited AFTER the ambiguous one in the body: the evidence
    # pre-pass must collect siblings appearing after the ambiguous citation.
    _write(tmp_path, "jobcannon/host/config.py", ["a", "def f():", "    pass"])
    _write(tmp_path, "dashboard/config.py", ["b", "OTHER"])

    verdicts = verify_citations(
        "resolves config.py:2 from the sibling jobcannon/host/config.py:1-3",
        tmp_path,
    )

    assert len(verdicts) == 2
    sibling_verdict = verdicts[0]
    assert sibling_verdict.status is CitationStatus.RESOLVED_BY_SIBLING
    assert _posix(sibling_verdict.resolved_path).endswith("jobcannon/host/config.py")
    assert sibling_verdict.candidates is None
    assert verdicts[1].status is CitationStatus.OK
    assert drifted_verdicts(verdicts) == []
    assert drift_fingerprint(verdicts) == ""


def test_verify_sibling_evidence_requires_exactly_one_candidate(
    tmp_path: Path,
) -> None:
    # Two qualified siblings in different directories both match the
    # ambiguous basename's candidates -- no deterministic disambiguation
    # exists, so the citation falls back to ``AMBIGUOUS_BASENAME``.
    _write(tmp_path, "a/config.py", ["x"])
    _write(tmp_path, "b/config.py", ["y"])

    verdicts = verify_citations("config.py:2 with siblings a/config.py:1 b/config.py:1", tmp_path)

    assert len(verdicts) == 3
    ambiguous = [v for v in verdicts if v.status is CitationStatus.AMBIGUOUS_BASENAME]
    assert len(ambiguous) == 1
    candidates = ambiguous[0].candidates
    assert candidates is not None
    assert sorted(candidates) == ["a/config.py", "b/config.py"]
    assert len(drifted_verdicts(verdicts)) == 1


def test_verify_ambiguous_basenames_do_not_disambiguate_each_other(
    tmp_path: Path,
) -> None:
    # Self-disambiguation guard: two bare citations of the same ambiguous
    # basename must NOT resolve each other -- sibling evidence comes only
    # from citations whose asserted literal path exists in the tree. Both
    # stay ``AMBIGUOUS_BASENAME``.
    _write(tmp_path, "a/config.py", ["x"])
    _write(tmp_path, "b/config.py", ["y"])

    verdicts = verify_citations("config.py:1 and config.py:2 drift", tmp_path)

    assert len(verdicts) == 2
    assert all(v.status is CitationStatus.AMBIGUOUS_BASENAME for v in verdicts)


def test_verify_prefixed_ambiguous_sibling_resolves_to_stale_prefix(
    tmp_path: Path,
) -> None:
    # A prefixed citation whose asserted prefix is wrong resolves
    # deterministically via sibling evidence, but the false prefix is still
    # asserted: ``STALE_PREFIX`` (not ``RESOLVED_BY_SIBLING``) with
    # ``resolved_path`` pointing at the sibling-selected candidate.
    _write(tmp_path, "jobcannon/host/config.py", ["a", "b"])
    _write(tmp_path, "dashboard/config.py", ["c", "d"])

    verdicts = verify_citations(
        "wrong_dir/config.py:2 is stale; jobcannon/host/config.py:1 is right",
        tmp_path,
    )

    assert len(verdicts) == 2
    stale = verdicts[0]
    assert stale.status is CitationStatus.STALE_PREFIX
    assert _posix(stale.resolved_path).endswith("jobcannon/host/config.py")


def test_verify_sibling_resolved_line_range_still_validates(tmp_path: Path) -> None:
    # Sibling disambiguation picks the candidate; the line range then
    # validates against it like any other resolution. A range beyond the
    # resolved file's EOF is still ``OUT_OF_RANGE`` -- the coordinate defect
    # takes precedence over the info-level resolution.
    _write(tmp_path, "jobcannon/host/config.py", ["a", "b"])
    _write(tmp_path, "dashboard/config.py", ["c", "d"])

    verdicts = verify_citations(
        "config.py:99 out of range; jobcannon/host/config.py:1 qualified",
        tmp_path,
    )

    assert len(verdicts) == 2
    assert verdicts[0].status is CitationStatus.OUT_OF_RANGE
    assert len(drifted_verdicts(verdicts)) == 1


# --------------------------------------------------------------------------- #
# tier 2: same-directory sibling (a different file in the candidate's directory)
# --------------------------------------------------------------------------- #


def test_verify_same_directory_sibling_disambiguates_bare_basename(
    tmp_path: Path,
) -> None:
    # The body cites bare ``config.py:2`` and, elsewhere, a *different* file in
    # one candidate's directory: ``jobcannon/host/runner.py:1``. No sibling
    # cites ``config.py`` by exact path, but ``runner.py``'s directory matches
    # exactly one ``config.py`` candidate, so the bare citation resolves to
    # ``jobcannon/host/config.py`` as info-level ``RESOLVED_BY_SIBLING``.
    _write(tmp_path, "jobcannon/host/config.py", ["a", "def f():", "    pass"])
    _write(tmp_path, "jobcannon/host/runner.py", ["r"])
    _write(tmp_path, "dashboard/config.py", ["b", "OTHER"])

    verdicts = verify_citations(
        "see config.py:2 -- same dir as jobcannon/host/runner.py:1",
        tmp_path,
    )

    assert len(verdicts) == 2
    bare = verdicts[0]
    assert bare.status is CitationStatus.RESOLVED_BY_SIBLING
    assert _posix(bare.resolved_path).endswith("jobcannon/host/config.py")
    assert bare.candidates is None
    assert verdicts[1].status is CitationStatus.OK
    assert drifted_verdicts(verdicts) == []
    assert drift_fingerprint(verdicts) == ""


def test_verify_exact_path_sibling_wins_over_other_candidates_directory(
    tmp_path: Path,
) -> None:
    # Tier precedence: ``a/config.py`` is cited by exact path (tier 1 picks
    # exactly one candidate), while ``b/other.py`` sits in the *second*
    # candidate's directory (tier 2 alone would pick ``b/config.py``). The
    # exact-path tier wins -- it is consulted first and not diluted by the
    # directory evidence -- so the bare citation resolves to ``a/config.py``
    # instead of falling back to ``AMBIGUOUS_BASENAME``.
    _write(tmp_path, "a/config.py", ["x", "y"])
    _write(tmp_path, "b/config.py", ["p", "q"])
    _write(tmp_path, "b/other.py", ["o"])

    verdicts = verify_citations(
        "config.py:2 with a/config.py:1 and b/other.py:1",
        tmp_path,
    )

    assert len(verdicts) == 3
    bare = verdicts[0]
    assert bare.status is CitationStatus.RESOLVED_BY_SIBLING
    assert _posix(bare.resolved_path).endswith("a/config.py")
    assert drifted_verdicts(verdicts) == []


def test_verify_same_directory_evidence_matching_two_candidates_stays_ambiguous(
    tmp_path: Path,
) -> None:
    # Directory evidence from two different directories each matches a
    # candidate: no unique pick, so ``AMBIGUOUS_BASENAME`` with both
    # candidates listed.
    _write(tmp_path, "a/config.py", ["x"])
    _write(tmp_path, "b/config.py", ["y"])
    _write(tmp_path, "a/x.py", ["1"])
    _write(tmp_path, "b/y.py", ["2"])

    verdicts = verify_citations("config.py:1 with a/x.py:1 and b/y.py:1", tmp_path)

    assert len(verdicts) == 3
    assert verdicts[0].status is CitationStatus.AMBIGUOUS_BASENAME
    assert verdicts[0].candidates is not None
    assert sorted(verdicts[0].candidates) == ["a/config.py", "b/config.py"]


@pytest.mark.parametrize(
    ("candidate", "sibling"),
    [
        # The sibling sits in a *subdirectory* of the candidate's directory.
        ("a/config.py", "a/sub/x.py"),
        # The candidate sits in a *subdirectory* of the sibling's directory.
        ("a/sub/config.py", "a/x.py"),
    ],
)
def test_verify_same_directory_requires_the_same_directory_not_a_subdirectory(
    tmp_path: Path, candidate: str, sibling: str
) -> None:
    # "Same directory" is exact parent equality in both directions: a sibling
    # citation in a subdirectory of a candidate's directory, or in a parent
    # directory of it, is not evidence for that candidate.
    _write(tmp_path, candidate, ["x"])
    _write(tmp_path, "b/config.py", ["y"])
    _write(tmp_path, sibling, ["1"])

    verdicts = verify_citations(f"config.py:1 with {sibling}:1", tmp_path)

    assert verdicts[0].status is CitationStatus.AMBIGUOUS_BASENAME
    assert verdicts[0].candidates is not None
    assert sorted(verdicts[0].candidates) == sorted([candidate, "b/config.py"])


def test_verify_repo_root_sibling_is_not_directory_evidence(tmp_path: Path) -> None:
    # A root-level sibling (``setup.py``) names no directory, so it must not
    # pick the root-level ``config.py`` candidate. ``wrong/config.py`` is
    # ambiguous between the root file and ``pkg/config.py`` and stays so.
    _write(tmp_path, "config.py", ["root"])
    _write(tmp_path, "pkg/config.py", ["pkg"])
    _write(tmp_path, "setup.py", ["s"])

    verdicts = verify_citations("wrong/config.py:1 with setup.py:1", tmp_path)

    assert verdicts[0].status is CitationStatus.AMBIGUOUS_BASENAME
    assert verdicts[0].candidates is not None
    assert sorted(verdicts[0].candidates) == ["config.py", "pkg/config.py"]
