"""Cross-repo gate: positive-evidence sibling-repo lookup (issues #1756-#1758).

These tests drive :func:`cross_repo_gate`'s ``managed_repo_roots`` /
``dispatching_repo_name`` parameters directly — the decision rule this
module's docstring calls the "positive-evidence redesign": escalation now
requires a missing path to be positively found under exactly one *other*
managed fleet repo's root, rather than treating bare absence from this repo
as proof of a cross-repo target. See ``test_cross_repo_gate.py`` for the
mechanical flip of the pre-redesign "bare absence escalates" test suite.
"""

from __future__ import annotations

from pathlib import Path

from charlie_work.cross_repo_gate import cross_repo_gate


def test_found_in_single_sibling_repo_escalates(tmp_path: Path) -> None:
    """A missing path positively found under exactly one other managed
    repo's root escalates, with that repo's name recorded on
    ``found_in_repo`` and ``cross_repo_target`` in the reason."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    (sibling_repo / "src" / "ci_fleet").mkdir(parents=True)
    (sibling_repo / "src" / "ci_fleet" / "runner_slots.py").write_text(
        "# runner_slots", encoding="utf-8"
    )
    body = "The bug is in `src/ci_fleet/runner_slots.py`."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is False
    assert result.referenced_paths == ("src/ci_fleet/runner_slots.py",)
    assert result.missing_paths == ("src/ci_fleet/runner_slots.py",)
    assert result.found_in_repo == "ci_runners"
    assert "cross_repo_target" in result.reason


def test_1758_new_file_found_nowhere_abstains(tmp_path: Path) -> None:
    """Issue #1758's shape: an issue proposes a genuinely new file that does
    not exist yet anywhere in the fleet (not this repo, not any registered
    sibling). Absence alone is not evidence of a cross-repo target, so the
    gate abstains rather than escalating."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    sibling_repo.mkdir()

    body = "Add a new module `src/charlie_work/brand_new_feature.py` implementing the feature."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.missing_paths == ("src/charlie_work/brand_new_feature.py",)
    assert result.found_in_repo is None
    assert "abstaining" in result.reason


def test_1756_newline_corrupted_candidate_never_extracted(tmp_path: Path) -> None:
    """Issue #1756's shape: a hard-wrapped backtick-quoted path with an
    embedded newline is dropped at extraction (embedded-whitespace filter),
    so it never reaches the missing-path/sibling-lookup machinery at all —
    the gate passes with the "no file paths referenced" reason, even with a
    sibling repo registered that could otherwise have been searched."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    sibling_repo.mkdir()

    body = "See `src/charlie_work/\ncorrupted_path.py` for context."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.reason == "no file paths referenced in issue body"


def test_cw_1518_single_space_multi_path_span_never_extracted(tmp_path: Path) -> None:
    """cw #1518's shape: two paths cited together inside one backtick span,
    separated by a single space (`` `tests/a.py tests/b.py` ``), extracts as
    one candidate containing an embedded space. The embedded-whitespace
    filter (deliberately wider than #1756's own "2+ whitespace" proposal)
    drops it at extraction, so it never reaches the missing-path/sibling
    lookup and the gate passes."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    sibling_repo.mkdir()

    body = "Run `tests/a.py tests/b.py` to reproduce."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.referenced_paths == ()
    assert result.reason == "no file paths referenced in issue body"


def test_module_relative_suffix_path_present_only_in_dispatching_repo_abstains(
    tmp_path: Path,
) -> None:
    """Issue #1757's shape (module-relative citation, e.g.
    ``schemas/coach.py`` for the real, more deeply nested
    ``server/src/swole/schemas/coach.py``) — but present only in the
    *dispatching* repo itself, not a sibling. #1757's own same-repo suffix
    fallback for ``_path_exists_in_repo`` is explicitly NOT implemented by
    this change (see the module docstring) — only the shared
    ``_segment_boundary_suffix_match`` helper is provided for that future
    fix to reuse. With the only registered repo being the dispatching repo
    itself (excluded from the sibling search), the gate correctly abstains
    rather than crashing or escalating on its own nested file."""
    this_repo = tmp_path / "charlie-work"
    (this_repo / "server" / "src" / "swole" / "schemas").mkdir(parents=True)
    (this_repo / "server" / "src" / "swole" / "schemas" / "coach.py").write_text(
        "# coach", encoding="utf-8"
    )

    body = "The bug is in `schemas/coach.py`."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.missing_paths == ("schemas/coach.py",)
    assert result.found_in_repo is None
    assert "abstaining" in result.reason


def test_found_in_two_sibling_repos_abstains_ambiguous(tmp_path: Path) -> None:
    """A missing path positively found under TWO other managed repos'
    roots is ambiguous ownership — not positive evidence of a single
    cross-repo target — so the gate abstains rather than guessing."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_a = tmp_path / "ci_runners"
    (sibling_a / "shared").mkdir(parents=True)
    (sibling_a / "shared" / "utils.py").write_text("# a", encoding="utf-8")
    sibling_b = tmp_path / "job_finder"
    (sibling_b / "shared").mkdir(parents=True)
    (sibling_b / "shared" / "utils.py").write_text("# b", encoding="utf-8")

    body = "The helper lives in `shared/utils.py`."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_a, "job_finder": sibling_b},
        "charlie-work",
    )

    assert result.passed is True
    assert result.missing_paths == ("shared/utils.py",)
    assert result.found_in_repo is None
    assert "abstaining" in result.reason


def test_traversal_candidate_escaping_sibling_root_never_treated_as_found(
    tmp_path: Path,
) -> None:
    """Containment guard (explicit task requirement): a ``..``-traversal
    candidate that would resolve OUTSIDE a sibling repo's root must never
    be reported as "found" under that sibling, even when the resolved
    location happens to exist on disk. ``this_repo`` and ``sibling_repo``
    sit at different depths so a naive (containment-unaware) relative join
    against ``this_repo`` -- the pre-existing, out-of-scope gap in
    ``_path_exists_in_repo`` -- does not coincidentally also reach the
    outside file and mask what this test is isolating: the containment
    check inside ``_find_owning_repo``/``_resolve_within_root``, which
    resolves both sides via ``safe_path.contains`` before any ``exists()``
    call."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "deep" / "ci_runners"
    sibling_repo.mkdir(parents=True)
    # Sibling of `sibling_repo`, one level up -- reachable from inside
    # `sibling_repo` via `../outside/secret.py`, but NOT a descendant of
    # `sibling_repo`'s own root.
    outside_dir = tmp_path / "deep" / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.py").write_text("# secret", encoding="utf-8")

    body = "See `../outside/secret.py` for the fix."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is True
    assert result.missing_paths == ("../outside/secret.py",)
    assert result.found_in_repo is None
    assert "abstaining" in result.reason
