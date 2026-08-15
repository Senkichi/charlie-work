"""Tests for ``charlie_work.citation_check`` (issue #1000).

The check is the "cheap" backstop half of the fix; the filing convention is the
durable half. These tests pin both the parsing boundaries (what is and is not a
citation) and the verdict logic for each drift class.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.citation_check import (
    Citation,
    CitationStatus,
    drift_fingerprint,
    drifted_verdicts,
    parse_citations,
    verify_citations,
)


def _write(repo: Path, rel: str, lines: list[str]) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# parse_citations
# --------------------------------------------------------------------------- #


def test_parse_single_line_citation() -> None:
    cites = parse_citations("See workflow.py:4746 for the defect.")
    assert cites == [
        Citation(raw="workflow.py:4746", path="workflow.py", line=4746, end_line=4746)
    ]


def test_parse_range_citation() -> None:
    cites = parse_citations("workflow.py:3527-3535 is the loop.")
    assert cites == [
        Citation(raw="workflow.py:3527-3535", path="workflow.py", line=3527, end_line=3535)
    ]


def test_parse_path_with_directory_prefix() -> None:
    cites = parse_citations("src/charlie_work/workflow.py:9294 moved.")
    assert cites[0].path == "src/charlie_work/workflow.py"
    assert cites[0].line == 9294


def test_parse_deduplicates_repeated_citations() -> None:
    cites = parse_citations("workflow.py:10 and again workflow.py:10")
    assert len(cites) == 1


def test_parse_rejects_urls() -> None:
    # ``https://example.com:8080`` must not parse as a citation of example.com:8080.
    cites = parse_citations("see https://example.com:8080/path for details")
    assert cites == []


def test_parse_rejects_timestamps_without_extension() -> None:
    # ``13:34`` has no filename extension -> not a citation.
    cites = parse_citations("merged at 13:34Z today")
    assert cites == []


def test_parse_rejects_backward_range() -> None:
    cites = parse_citations("workflow.py:100-50 is invalid")
    assert cites == []


def test_parse_does_not_match_inside_a_longer_path() -> None:
    # The boundary prevents matching ``charlie_work/workflow.py:4746`` as a
    # separate citation when it is part of ``src/charlie_work/workflow.py:4746``.
    cites = parse_citations("src/charlie_work/workflow.py:4746")
    assert len(cites) == 1
    assert cites[0].path == "src/charlie_work/workflow.py"


def test_parse_multiple_distinct_citations() -> None:
    body = "checks.py:415 drifted; heartbeat_check.py:671 too; workflow.py:8856"
    cites = parse_citations(body)
    paths = [(c.path, c.line) for c in cites]
    assert paths == [("checks.py", 415), ("heartbeat_check.py", 671), ("workflow.py", 8856)]


# --------------------------------------------------------------------------- #
# verify_citations
# --------------------------------------------------------------------------- #


def test_verify_ok_for_valid_line(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["a", "def f():", "    pass"])
    # Resolve via bare basename under src/.
    verdicts = verify_citations("workflow.py:2 drifted", tmp_path)
    assert len(verdicts) == 1
    assert verdicts[0].status is CitationStatus.OK
    assert "def f():" in (verdicts[0].current_line_text or "")


def test_verify_out_of_range(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["only one line"])
    verdicts = verify_citations("workflow.py:5000 is gone", tmp_path)
    assert verdicts[0].status is CitationStatus.OUT_OF_RANGE


def test_verify_file_missing(tmp_path: Path) -> None:
    verdicts = verify_citations("renamed.py:42 no longer exists", tmp_path)
    assert verdicts[0].status is CitationStatus.FILE_MISSING


def test_verify_empty_line_is_drift(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["code()", "", "more()"])
    verdicts = verify_citations("workflow.py:2 was the call", tmp_path)
    assert verdicts[0].status is CitationStatus.EMPTY_LINE


def test_verify_range_ok(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", [f"line{i}" for i in range(10)])
    verdicts = verify_citations("workflow.py:3-5 is the block", tmp_path)
    assert verdicts[0].status is CitationStatus.OK


def test_verify_range_out_of_range_when_end_exceeds(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["a", "b", "c"])
    verdicts = verify_citations("workflow.py:2-99", tmp_path)
    assert verdicts[0].status is CitationStatus.OUT_OF_RANGE


def test_verify_resolves_full_repo_path(tmp_path: Path) -> None:
    _write(tmp_path, "src/charlie_work/workflow.py", ["x", "y", "z"])
    verdicts = verify_citations("src/charlie_work/workflow.py:2", tmp_path)
    assert verdicts[0].status is CitationStatus.OK


def test_verify_resolves_bare_basename_two_levels_deep(tmp_path: Path) -> None:
    # Regression for the reviewer finding on PR #1199: real source files in
    # this repo live two levels deep under src/ (src/charlie_work/<name>.py),
    # but the bare-filename fallback only searched one level. A bare citation
    # like ``checks.py:415`` -- one of issue #1000's own cited examples -- must
    # resolve to the real file, not fall through to FILE_MISSING. The fixture
    # mirrors the real package layout instead of the synthetic one-level tree
    # the original tests used.
    checks_lines = [f"line{i}" for i in range(416)]  # index 414 == line 415
    checks_lines[414] = "TARGET_CHECKS_LINE"
    _write(tmp_path, "src/charlie_work/checks.py", checks_lines)
    workflow_lines = [f"line{i}" for i in range(8857)]  # index 8855 == line 8856
    workflow_lines[8855] = "TARGET_WORKFLOW_LINE"
    _write(tmp_path, "src/charlie_work/workflow.py", workflow_lines)

    verdicts = verify_citations("checks.py:415 and workflow.py:8856 drifted", tmp_path)

    assert len(verdicts) == 2
    assert verdicts[0].citation.path == "checks.py"
    assert verdicts[0].status is CitationStatus.OK
    assert "TARGET_CHECKS_LINE" in (verdicts[0].current_line_text or "")
    assert verdicts[1].citation.path == "workflow.py"
    assert verdicts[1].status is CitationStatus.OK
    assert "TARGET_WORKFLOW_LINE" in (verdicts[1].current_line_text or "")


def test_verify_stale_directory_prefix_resolves_via_recursive_index(
    tmp_path: Path,
) -> None:
    # Regression for the reviewer finding on PR #1199: a citation that carries
    # a directory prefix whose basename matches a real file but whose prefix
    # does not (the file moved into a new subdirectory between filing and
    # dispatch) must resolve via the recursive basename index, not report
    # FILE_MISSING. The resolver's fallback strips the directory off *any*
    # unresolved citation via ``Path(path).name`` before consulting the index,
    # so the index must be built for prefixed citations too -- gating on "no
    # directory prefix" only would skip the walk and leave the citation
    # unresolved.
    lines = [f"line{i}" for i in range(11)]  # index 9 == line 10
    lines[9] = "TARGET_MOVED_LINE"
    _write(tmp_path, "src/charlie_work/workflow.py", lines)

    # The citation says ``old_dir/workflow.py:10`` but the file lives at
    # ``src/charlie_work/workflow.py``. The stale prefix does not exist; the
    # basename does, two levels deep under src/.
    verdicts = verify_citations("old_dir/workflow.py:10 drifted", tmp_path)

    assert len(verdicts) == 1
    assert verdicts[0].citation.path == "old_dir/workflow.py"
    assert verdicts[0].status is CitationStatus.OK
    assert "TARGET_MOVED_LINE" in (verdicts[0].current_line_text or "")


def test_verify_bare_basename_missing_still_file_missing_two_level_tree(
    tmp_path: Path,
) -> None:
    # The recursive fallback must not turn a genuinely absent file into a false
    # OK: a bare citation whose basename matches nothing under the source roots
    # still resolves to FILE_MISSING, even when the tree is shaped like the real
    # repo (so the index is actually built and queried).
    _write(tmp_path, "src/charlie_work/workflow.py", ["x", "y", "z"])
    verdicts = verify_citations("nonexistent.py:10 drifted", tmp_path)
    assert verdicts[0].status is CitationStatus.FILE_MISSING


def test_verify_does_not_raise_on_unreadable_file(tmp_path: Path) -> None:
    # A path that resolves to a directory (not a file) is treated as missing,
    # never as an exception.
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "workflow.py").mkdir()
    verdicts = verify_citations("workflow.py:1", tmp_path)
    assert verdicts[0].status is CitationStatus.FILE_MISSING


# --------------------------------------------------------------------------- #
# content drift (commit-stamp path)
# --------------------------------------------------------------------------- #


def test_verify_content_drift_when_stamped_content_differs(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["new code here"])
    original = ["old code here"]

    def fetch(path: str, sha: str) -> list[str] | None:
        assert sha == "deadbeef"
        return original

    verdicts = verify_citations(
        "workflow.py:1 changed", tmp_path, commit_sha="deadbeef", fetch_file_lines_at_commit=fetch
    )
    assert verdicts[0].status is CitationStatus.CONTENT_DRIFT
    assert verdicts[0].original_line_text == "old code here"
    assert verdicts[0].current_line_text == "new code here"


def test_verify_content_ok_when_stamped_content_matches(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["stable line"])

    def fetch(path: str, sha: str) -> list[str] | None:
        return ["stable line"]

    verdicts = verify_citations(
        "workflow.py:1", tmp_path, commit_sha="abc1234", fetch_file_lines_at_commit=fetch
    )
    assert verdicts[0].status is CitationStatus.OK


def test_verify_content_drift_skipped_when_fetch_returns_none(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["line"])

    def fetch(path: str, sha: str) -> list[str] | None:
        return None  # commit or file unavailable at that sha

    verdicts = verify_citations(
        "workflow.py:1", tmp_path, commit_sha="abc1234", fetch_file_lines_at_commit=fetch
    )
    # Falls back to coordinate-only check -> OK (line exists, non-blank).
    assert verdicts[0].status is CitationStatus.OK


def test_verify_content_drift_not_run_for_out_of_range(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["only"])
    calls = []

    def fetch(path: str, sha: str) -> list[str] | None:
        calls.append(path)
        return ["only"]

    verdicts = verify_citations(
        "workflow.py:99", tmp_path, commit_sha="abc1234", fetch_file_lines_at_commit=fetch
    )
    assert verdicts[0].status is CitationStatus.OUT_OF_RANGE
    # Coordinate verdict short-circuits before content comparison.
    assert calls == []


# --------------------------------------------------------------------------- #
# drifted_verdicts / drift_fingerprint
# --------------------------------------------------------------------------- #


def test_drifted_verdicts_filters_ok() -> None:
    v = verify_citations("workflow.py:1", Path("/nonexistent"))
    # No file -> FILE_MISSING, which is drift.
    assert len(drifted_verdicts(v)) == 1


def test_fingerprint_empty_when_no_drift() -> None:
    assert drift_fingerprint([]) == ""


def test_fingerprint_stable_for_same_drift(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["a"])
    v1 = verify_citations("workflow.py:99", tmp_path)
    v2 = verify_citations("workflow.py:99", tmp_path)
    assert drift_fingerprint(v1) == drift_fingerprint(v2)
    assert drift_fingerprint(v1) != ""


def test_fingerprint_differs_for_different_drift(tmp_path: Path) -> None:
    _write(tmp_path, "src/workflow.py", ["a"])
    a = verify_citations("workflow.py:99", tmp_path)  # out of range
    b = verify_citations("workflow.py:1", tmp_path)  # ok
    assert drift_fingerprint(a) != drift_fingerprint(b)
    assert drift_fingerprint(b) == ""


def test_fingerprint_differs_for_different_status_same_line(tmp_path: Path) -> None:
    # Same path/line but different status (OUT_OF_RANGE vs EMPTY_LINE) must
    # produce different fingerprints, so a status change re-alerts.
    c = Citation(raw="x.py:1", path="x.py", line=1, end_line=1)
    from charlie_work.citation_check import CitationVerdict

    fp_oor = drift_fingerprint([CitationVerdict(c, CitationStatus.OUT_OF_RANGE)])
    fp_empty = drift_fingerprint([CitationVerdict(c, CitationStatus.EMPTY_LINE)])
    assert fp_oor != fp_empty
