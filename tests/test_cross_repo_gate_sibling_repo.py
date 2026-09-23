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

import subprocess
from pathlib import Path

import pytest

import charlie_work.cross_repo_gate as cross_repo_gate_module
from charlie_work.cross_repo_gate import cross_repo_gate


@pytest.fixture(autouse=True)
def _git_discovery_isolated_from_enclosing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep git subprocesses run under ``tmp_path`` from discovering the
    enclosing checkout.

    In the sandboxed worker environment ``tmp_path`` itself lives inside
    the worktree's gitignored ``.var/`` tree, so ``git check-ignore`` and
    ``git ls-files`` run against a fake ``this_repo``/``sibling_repo`` —
    none of which are git repositories themselves — resolve the *outer*
    checkout's repository instead of failing closed the way they do on a
    normal CI host where the temp dir lives outside the checkout:
    ``check-ignore`` matches the outer ``.var/`` rule and classifies every
    candidate a "gitignored runtime artifact," and ``ls-files`` returns
    the outer listing instead of driving the ``_all_repo_files`` fallback.
    Setting ``GIT_CEILING_DIRECTORIES`` to ``tmp_path`` stops git's upward
    repository discovery there, restoring fail-closed behavior — a no-op
    on a host where discovery already finds nothing.

    ``core.longpaths`` (via ``GIT_CONFIG_KEY_0``/``GIT_CONFIG_VALUE_0``)
    keeps ``git init``/``add`` inside the deeply nested sandbox temp dir
    from hitting Windows MAX_PATH ("Filename too long", exit 128);
    harmless where paths are short.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.longpaths")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")


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
    sit at different depths so the relative-candidate existence check
    against ``this_repo`` (``_path_exists_in_repo``, containment-checked
    via ``_resolve_within_root`` since issue #1772) does not
    coincidentally also reach the outside file and mask what this test is
    isolating: the containment check inside
    ``_find_owning_repo``/``_resolve_within_root``, which resolves both
    sides via ``safe_path.contains`` before any ``exists()`` call."""
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


def test_traversal_candidate_escaping_dispatching_root_is_missing(tmp_path: Path) -> None:
    """Issue #1772: a ``..``-traversal relative candidate that escapes the
    *dispatching* repo's own root must be classified missing, not "exists
    in the target repo."

    ``_path_exists_in_repo`` previously joined a relative candidate onto
    ``repo_root`` and called ``exists()`` with no containment check — a
    ``../outside/secret.py`` candidate whose escaped location happens to
    exist on disk was wrongly classified "not missing," firing the gate's
    "at least one referenced path exists in the target repo" pass branch.
    The fix routes both branches through :func:`_resolve_within_root`
    (``safe_path.contains``), so the candidate is missing and — with no
    fleet registry supplied — the gate abstains.

    ``this_repo`` is a real ``git init``-ed repo so ``_is_gitignored``'s
    ``git check-ignore`` consults this repo's own rules — the production
    shape — rather than silently inheriting the ignore rules of whatever
    worktree happens to enclose ``tmp_path`` (which neutralizes the
    candidate before the gate logic under test is ever reached)."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    _git(this_repo, "init", "-q")
    # Sibling of `this_repo`, one level up -- reachable from inside
    # `this_repo` via `../outside/secret.py`, but NOT a descendant of
    # `this_repo`'s own root.
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.py").write_text("# secret", encoding="utf-8")

    body = "See `../outside/secret.py` for the fix."

    result = cross_repo_gate(body, this_repo)

    assert result.passed is True
    assert result.missing_paths == ("../outside/secret.py",)
    assert "abstaining" in result.reason


def test_path_exists_in_repo_containment_checks_relative_candidates(
    tmp_path: Path,
) -> None:
    """Issue #1772, function level: ``_path_exists_in_repo`` returns
    ``False`` for a relative candidate whose ``..`` segment escapes
    ``repo_root`` even when a file exists at the escaped location, while a
    genuinely contained relative path still returns ``True``."""
    this_repo = tmp_path / "charlie-work"
    (this_repo / "src").mkdir(parents=True)
    (this_repo / "src" / "real.py").write_text("# real", encoding="utf-8")
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.py").write_text("# secret", encoding="utf-8")

    assert cross_repo_gate_module._path_exists_in_repo("../outside/secret.py", this_repo) is False
    assert cross_repo_gate_module._path_exists_in_repo("src/real.py", this_repo) is True


def test_path_exists_in_repo_containment_checks_posix_absolute_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1772, absolute branch: ``_path_exists_in_repo`` returns ``False``
    for a POSIX-style absolute candidate (``/outside/secret.py``) even when a
    file exists at the location the uncontained existence check would consult.

    On Windows ``Path("/outside/secret.py").is_absolute()`` is ``False`` (no
    drive letter), so pre-#1772 the candidate took the *relative* branch:
    ``repo_root / candidate`` collapses to the drive root
    (``C:\\outside\\secret.py`` — never ``repo_root\\outside\\secret.py``, so a
    file planted inside ``tmp_path`` cannot stand in for the escaped location)
    and ``exists()`` ran with no containment check at all, reporting a
    coincidental file at the escaped location as "in the repo." Reporting
    ``True`` from ``Path.exists`` for whatever path is consulted reproduces
    that precondition portably: unfixed code consults the drive-root collapse
    and wrongly returns ``True``; fixed code containment-checks via
    :func:`_resolve_within_root` and returns ``False`` without consulting
    ``exists()`` at all."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()

    consulted: list[Path] = []

    def _always_exists(self: Path, *args: object, **kwargs: object) -> bool:
        consulted.append(self)
        return True

    monkeypatch.setattr(Path, "exists", _always_exists)

    assert cross_repo_gate_module._path_exists_in_repo("/outside/secret.py", this_repo) is False
    # Containment runs before existence: the escaped location is never even
    # consulted, so a file there can never leak an "exists in repo" verdict.
    assert consulted == []


def test_segment_boundary_match_ignores_untracked_worktree_copies(tmp_path: Path) -> None:
    """Review finding 2: sibling file listings come from ``git ls-files``
    (:func:`_repo_tracked_files`), so an untracked worktree/vendor copy of
    the same leaf filename does not make the segment-boundary suffix match
    ambiguous.

    Without the fix (an unpruned, untracked-inclusive ``os.walk`` listing
    every file on disk), both the real tracked file and its untracked
    ``.venv`` duplicate would satisfy the suffix match, and
    ``_segment_boundary_suffix_match`` treats 2+ matches the same as no
    match at all -- every managed repo on the real fleet has at least one
    nested ``.claude/worktrees/`` or ``.venv`` copy of *something*, which is
    exactly what made 2+ matches the norm, not the exception, before this
    fix. This test would abstain (``found_in_repo is None``) without it.
    """
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    (sibling_repo / "src" / "ci_fleet").mkdir(parents=True)
    (sibling_repo / "src" / "ci_fleet" / "runner_slots.py").write_text(
        "# tracked", encoding="utf-8"
    )
    _git(sibling_repo, "init", "-q")
    _git(sibling_repo, "add", "src/ci_fleet/runner_slots.py")
    _git(
        sibling_repo,
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "-q",
        "-m",
        "init",
    )
    # Untracked vendor/worktree-shaped duplicate of the same leaf filename --
    # never `git add`ed, so `git ls-files` does not list it.
    (sibling_repo / ".venv" / "site-packages" / "ci_fleet").mkdir(parents=True)
    (sibling_repo / ".venv" / "site-packages" / "ci_fleet" / "runner_slots.py").write_text(
        "# vendored copy, untracked", encoding="utf-8"
    )

    body = "The bug is in `ci_fleet/runner_slots.py`."

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is False
    assert result.found_in_repo == "ci_runners"
    assert "cross_repo_target" in result.reason


def test_sibling_file_listing_computed_once_per_root_per_gate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review finding 1: a sibling repo's tracked-file listing
    (:func:`_repo_tracked_files`) is computed at most once per root for the
    whole :func:`cross_repo_gate` call -- not once per missing candidate
    that falls through to the segment-boundary suffix fallback. An
    uncached, unpruned full-tree walk repeated per candidate measured 37
    seconds per call against this fleet's largest managed repo."""
    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_repo = tmp_path / "ci_runners"
    (sibling_repo / "src" / "ci_fleet").mkdir(parents=True)
    (sibling_repo / "src" / "ci_fleet" / "runner_slots.py").write_text(
        "# runner_slots", encoding="utf-8"
    )

    call_count = 0
    real_repo_tracked_files = cross_repo_gate_module._repo_tracked_files

    def _counting_repo_tracked_files(repo_root: Path) -> list[str]:
        nonlocal call_count
        call_count += 1
        return real_repo_tracked_files(repo_root)

    monkeypatch.setattr(
        cross_repo_gate_module, "_repo_tracked_files", _counting_repo_tracked_files
    )

    # Two missing candidates checked against the SAME sibling root within one
    # gate call: the first matches nothing there (forcing `_find_owning_repo`
    # to consult and cache the listing while still returning None), the
    # second matches via the segment-boundary suffix fallback -- both need
    # "ci_runners"'s listing.
    body = (
        "See `ci_fleet/decoy_missing_file.py` for background; "
        "the actual bug is in `ci_fleet/runner_slots.py`."
    )

    result = cross_repo_gate(
        body,
        this_repo,
        {"charlie-work": this_repo, "ci_runners": sibling_repo},
        "charlie-work",
    )

    assert result.passed is False
    assert result.found_in_repo == "ci_runners"
    assert call_count == 1


def test_dispatching_repo_excluded_by_root_when_name_mismatches(tmp_path: Path) -> None:
    """Review finding 3: the dispatching repo's own registered entry is
    excluded from the sibling search by resolved ROOT
    (:func:`_excludes_dispatching_root`), independent of whether the
    ``dispatching_repo_name`` argument happens to match the registry key
    that entry is stored under.

    ``_dispatching_repo_name`` falls back to ``repo_root.name`` (a
    directory name, e.g. a deployment dir like ``charlie-work-daemon``)
    when the ``gh`` lookup fails -- which need not match the fleet
    registry's ``owner/repo``-derived key (``charlie-work``). Without the
    root-based exclusion, a module-relative citation (``schemas/coach.py``
    for the real, more deeply nested
    ``server/src/swole/schemas/coach.py`` -- the same #1757 shape as
    ``test_module_relative_suffix_path_present_only_in_dispatching_repo_abstains``
    above) would match the dispatching repo's OWN entry via the
    segment-boundary suffix fallback, since name-exclusion alone fails to
    recognize the mismatched name as "self" -- escalating the repo against
    itself instead of abstaining."""
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
        "charlie-work-daemon",  # mismatches the registry key "charlie-work"
    )

    assert result.passed is True
    assert result.found_in_repo is None
    assert "abstaining" in result.reason


def test_issue_1791_driveless_absolute_path_classified_as_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1791: ``_path_exists_in_repo`` classifies a POSIX-style
    absolute candidate (leading ``/``, no drive letter) as absolute —
    containment-checked directly — matching how ``_resolve_within_root``
    treats the exact same string.

    ``_path_exists_in_repo`` previously branched on the raw,
    platform-dependent ``Path.is_absolute()``, which reports ``False``
    for a driveless absolute path on Windows. The candidate then fell
    into the "relative" branch and was joined onto ``repo_root``,
    collapsing to the drive root plus the candidate's tail
    (``Path("C:/repo") / Path("/x/y.py") == Path("C:/x/y.py")``) — an
    existence check against a foreign location with no containment check
    at all. Here that join lands on the real sibling file, so the
    unfixed function reports the out-of-repo path as present in the repo.

    The fixture spells a real on-disk sibling file in driveless form by
    stripping its own anchor — the only way to materialize a POSIX-style
    absolute path that genuinely exists on a Windows host without writing
    outside ``tmp_path`` (see ``test_posix_style_absolute_path_outside_repo_blocks``
    in ``test_cross_repo_gate.py`` for why a literal ``/home/...`` fixture
    cannot be built there). On POSIX hosts the same construction yields
    the file's ordinary absolute path verbatim, so both code paths agree
    there either way; the regression this pins is Windows-only.
    """
    # A driveless absolute path resolves against the *current drive* when
    # existence-checked on Windows, and the checkout's drive need not be
    # the temp dir's — the Windows CI runner keeps the checkout on ``D:``
    # but ``TEMP`` on ``C:``, where ``/Users/runneradmin/...`` resolves to
    # ``D:\Users\runneradmin\...`` (nonexistent) and the existence +
    # containment path this test pins is never exercised. ``chdir`` puts
    # the process CWD on ``tmp_path``'s drive so the driveless spelling
    # below resolves to the real file on any drive split; a no-op where
    # the two already share one.
    monkeypatch.chdir(tmp_path)

    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_root = tmp_path / "sibling_checkout"
    (sibling_root / "src" / "ci_fleet").mkdir(parents=True)
    sibling_file = sibling_root / "src" / "ci_fleet" / "suite_coverage.py"
    sibling_file.write_text("# suite_coverage", encoding="utf-8")

    # ``C:/.../suite_coverage.py`` -> ``/.../suite_coverage.py``: the same
    # file spelled without its drive letter — absolute per
    # ``_is_absolute_path`` on every platform, but reported non-absolute
    # by ``Path.is_absolute()`` on Windows (and resolving to the real file
    # under the current drive pinned by the ``chdir`` above).
    driveless = "/" + sibling_file.relative_to(sibling_file.anchor).as_posix()

    assert cross_repo_gate_module._is_absolute_path(driveless) is True
    # Absolute and outside ``this_repo``'s root — the same verdict
    # ``_resolve_within_root`` already gives this exact string.
    assert cross_repo_gate_module._resolve_within_root(this_repo, driveless) is None
    assert cross_repo_gate_module._path_exists_in_repo(driveless, this_repo) is False


def test_issue_1791_driveless_absolute_path_existing_outside_repo_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1791, end-to-end: a POSIX-style absolute candidate that
    exists on disk outside ``repo_root`` escalates via the founding
    #1010/#953 foreign-checkout arm
    (:func:`_is_confirmed_foreign_absolute_path`) — the same verdict
    ``test_absolute_path_outside_repo_blocks`` pins for its drive-letter
    spelling, reached through the missing-survivor path that
    ``_path_exists_in_repo`` decides.
    """
    # Same current-drive pin as the test above: without it, a
    # checkout/temp drive split (the Windows CI runner's ``D:`` checkout
    # vs. ``C:`` ``TEMP``) resolves the driveless candidate to a
    # nonexistent location — the file "exists nowhere," the
    # foreign-checkout arm never fires, and the gate abstains.
    monkeypatch.chdir(tmp_path)

    this_repo = tmp_path / "charlie-work"
    this_repo.mkdir()
    sibling_root = tmp_path / "sibling_checkout"
    (sibling_root / "src" / "ci_fleet").mkdir(parents=True)
    sibling_file = sibling_root / "src" / "ci_fleet" / "suite_coverage.py"
    sibling_file.write_text("# suite_coverage", encoding="utf-8")

    driveless = "/" + sibling_file.relative_to(sibling_file.anchor).as_posix()

    body = f"The file is at `{driveless}`."
    result = cross_repo_gate(body, this_repo)

    assert result.passed is False
    assert result.referenced_paths == (driveless,)
    assert result.missing_paths == (driveless,)
    assert "positive evidence of a foreign checkout" in result.reason
