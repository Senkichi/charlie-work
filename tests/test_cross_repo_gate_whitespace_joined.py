"""Cross-repo gate: whitespace-joined citation split (issue #1790).

The embedded-whitespace filter (issue #1756) used to drop an entire
candidate the moment it contained any whitespace — including the deliberate
cw #1518 shape where two real, distinct paths are cited together inside one
backtick span separated by a single space (`` `tests/a.py tests/b.py` ``).
Both paths were discarded rather than classified, silently losing the
positive sibling-repo evidence either could have supplied.

Issue #1790: before dropping a whitespace-containing candidate, the gate
splits it on whitespace and re-runs each piece through the normal candidate
pipeline (extraction shape check, placeholder/glob/launcher-owned filters),
falling back to drop-the-whole-thing only when the split yields fewer than
2 path-shaped pieces. Split pieces share the whole span's ``(start, end)``
offsets — the span is a single citation unit, so evidence markers and
citation-section headings apply to every piece uniformly.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.cross_repo_gate import cross_repo_gate, extract_referenced_paths


def test_whitespace_joined_pair_extracted_as_two_candidates() -> None:
    """The cw #1518 shape — `` `tests/a.py tests/b.py` `` — now yields two
    candidates in citation order instead of zero."""
    paths = extract_referenced_paths("Run `tests/a.py tests/b.py` to reproduce.")
    assert paths == ["tests/a.py", "tests/b.py"]


def test_whitespace_joined_triple_extracted_as_three_candidates() -> None:
    """A three-path whitespace-joined span splits into three candidates."""
    paths = extract_referenced_paths("See `a/b.py c/d.py e/f.py`.")
    assert paths == ["a/b.py", "c/d.py", "e/f.py"]


def test_whitespace_joined_comma_separated_pieces_extracted() -> None:
    """Punctuation glued to a piece (`` `a/b.py, c/d.py` ``) does not defeat
    the split — each piece re-runs the normal extraction regex, which stops
    at the comma exactly as it does for unquoted prose."""
    paths = extract_referenced_paths("See `a/b.py, c/d.py`.")
    assert paths == ["a/b.py", "c/d.py"]


def test_whitespace_joined_duplicate_pieces_deduplicated() -> None:
    """The same path cited twice in one span extracts once — the usual
    first-seen-order dedup applies to split pieces."""
    paths = extract_referenced_paths("See `a/b.py a/b.py`.")
    assert paths == ["a/b.py"]


def test_single_path_shaped_piece_still_drops_whole_candidate() -> None:
    """Fallback: a span that yields only one path-shaped piece is still
    dropped whole — `` `tests/foo.py trailing.py` `` joins a real path with
    a bare filename (not path-shaped: bare filenames are never extracted,
    even alone in their own backticks), the same can-never-exist
    corrupted-candidate class as #1756's newline-wrapped path."""
    paths = extract_referenced_paths("See `tests/foo.py trailing.py`.")
    assert paths == []


def test_newline_wrapped_single_path_still_drops_whole_candidate() -> None:
    """The #1756 shape itself is preserved: a single path hard-wrapped
    mid-token yields zero path-shaped pieces, so the candidate drops whole
    rather than recovering a fragment."""
    paths = extract_referenced_paths("See `src/charlie_work/\ncorrupted_path.py`.")
    assert paths == []


def test_split_pieces_run_through_placeholder_filter() -> None:
    """Each split piece re-runs the normal pipeline: a ``pr-N`` placeholder
    piece is dropped by the placeholder filter while the real sibling piece
    survives — the real path is no longer lost with the span."""
    paths = extract_referenced_paths("See `dir/pr-N/state.json dir/real.py`.")
    assert paths == ["dir/real.py"]


def test_split_pieces_run_through_launcher_owned_filter() -> None:
    """A ``.devin/`` piece is launcher-owned (issue #1391) and filtered per
    the normal pipeline while the real sibling piece survives."""
    paths = extract_referenced_paths("See `.devin/hooks.v1.json src/real.py`.")
    assert paths == ["src/real.py"]


def test_split_pieces_inherit_span_evidence_marker(tmp_path: Path) -> None:
    """Split pieces share the whole span's clause context: an ``Evidence:``
    marker preceding the span neutralizes BOTH pieces — per-piece offsets
    would let the first piece's ``.md`` period sever the clause for the
    second, inconsistently neutralizing only half of one citation."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()

    body = "Evidence: `docs/a.md docs/b.md` backs this report."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.neutral_paths == ("docs/a.md", "docs/b.md")


def test_split_pieces_inherit_span_evidence_suffix(tmp_path: Path) -> None:
    """A ``section N`` suffix following the span neutralizes every piece —
    the whole span is cited as evidence, not just the last piece."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()

    body = "The claim comes from `docs/a.md docs/b.md` section 4."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.neutral_paths == ("docs/a.md", "docs/b.md")


def test_split_pieces_inherit_span_citation_section(tmp_path: Path) -> None:
    """A span under a citation-section heading (issue #1583) neutralizes
    every piece — section scope is derived from the span's position, which
    all pieces share."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()

    body = "## References\n\n- `docs/a.md docs/b.md`\n\n## Bug\n\nDetails here."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.neutral_paths == ("docs/a.md", "docs/b.md")


def test_split_pieces_under_non_citation_heading_survive(tmp_path: Path) -> None:
    """Contrast: the same span under a non-citation heading is NOT
    neutralized — ``## Code References That Must Change`` merely contains
    ``references`` as a substring and must not swallow genuine dispatch
    targets."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()

    body = "## Code References That Must Change\n\n- `docs/a.md docs/b.md`\n"

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ("docs/a.md", "docs/b.md")
    assert result.missing_paths == ("docs/a.md", "docs/b.md")


def test_split_piece_found_in_sibling_escalates(tmp_path: Path) -> None:
    """The precision gap issue #1790 closes: a genuine cross-repo citation
    written as a whitespace-joined pair now supplies its positive
    sibling-repo evidence — the piece found under exactly one registered
    sibling escalates with ``found_in_repo`` set."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    (sibling_repo / "src" / "ci_fleet").mkdir(parents=True)
    (sibling_repo / "src" / "ci_fleet" / "runner_slots.py").write_text(
        "# runner_slots", encoding="utf-8"
    )

    body = "Run `tests/a.py ci_fleet/runner_slots.py` to reproduce."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is False
    assert result.referenced_paths == ("tests/a.py", "ci_fleet/runner_slots.py")
    assert result.missing_paths == ("tests/a.py", "ci_fleet/runner_slots.py")
    assert result.found_in_repo == "ci_runners"
    assert "cross_repo_target" in result.reason


def test_split_piece_existing_here_passes(tmp_path: Path) -> None:
    """A split piece that exists in the target repo is ordinary pass
    evidence — the gate passes on at-least-one-survivor-exists exactly as
    it would for separately cited paths."""
    this_repo = tmp_path / "charlie-work"
    (this_repo / "tests").mkdir(parents=True)
    (this_repo / "tests" / "a.py").write_text("# a", encoding="utf-8")

    body = "Run `tests/a.py tests/missing.py` to reproduce."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ("tests/a.py", "tests/missing.py")
    assert result.missing_paths == ("tests/missing.py",)
    assert result.reason == "at least one referenced path exists in the target repo"


def test_dotdot_shorthand_piece_inside_span_still_neutral(tmp_path: Path) -> None:
    """A ``...``-prefixed piece inside a whitespace-joined span still hits
    the shorthand neutralization arm downstream (issue #1761) — split
    pieces flow through ``_split_survivors_and_neutral`` like any other
    candidate."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()

    body = "See `dir/a.json .../b.json` for the data."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ("dir/a.json",)
    assert result.neutral_paths == (".../b.json",)
