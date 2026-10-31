"""Cross-repo gate: same-repo segment-boundary suffix fallback (issue #1757).

``_path_exists_in_repo`` used to check only the literal repo-root-relative
resolution of a candidate path. Issue bodies routinely cite paths in a
natural, module-relative form -- the way a developer discusses "the schema
module" or "the design system" -- rather than a full repo-root-relative
form. In a monorepo with nested project roots -- exactly swole's layout,
Python under ``server/src/swole/...`` and Kotlin under
``swole/app/<module>/src/main/kotlin/dev/senki/swole/...`` -- this made the
gate report a genuinely-existing file as "missing" and escalate a same-repo
issue as ``cross_repo_target`` (swole #92, #94, #213, #218).

The fix: when the direct repo-root-anchored resolution misses,
``_path_exists_in_repo`` segment-boundary suffix-matches the dispatching
repo's own tracked-file listing (:func:`_segment_boundary_suffix_match`,
the shared helper the sibling-ownership arm and this fix reuse) --
exactly one real file whose path ends with the candidate on a
path-segment boundary resolves it. An ambiguous suffix (2+ real files
share the candidate -- plausible once a real ``shared/`` module exists
alongside ``swole/app/mobile/``) must not silently pick one, so it
degrades to "missing" -- the conservative default the sibling arm and
the shorthand classifier already use. The listing is computed at most
once per :func:`cross_repo_gate` call (per-call ``file_listings``
memoization, mirroring the sibling search's own review-finding-1 cache).

Fixture bodies under ``tests/fixtures/cross_repo_gate_swole/`` are the
verbatim bodies of real swole issues, shared with
``test_cross_repo_gate_swole_regression.py``; the layout helper plants
the real nested monorepo shape those citations refer to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import charlie_work.cross_repo_gate as cross_repo_gate_module
from charlie_work.cross_repo_gate import cross_repo_gate

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cross_repo_gate_swole"


def _planted_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _swole_layout(root: Path, *, with_shared_settings: bool = False) -> None:
    """Plant a minimal swole-monorepo-shaped tree under ``root``.

    Nested project roots exactly as in the live repo: Python under
    ``server/src/swole/...``, Kotlin under
    ``swole/app/mobile/src/main/kotlin/dev/senki/swole/...``. Neither
    ``schemas/coach.py`` nor any of the other module-relative paths the
    issues cite exists at repo root. ``with_shared_settings`` adds a
    second real ``.../data/repo/SettingsRepository.kt`` under a
    ``shared/`` module -- the "plausible once a real ``shared/`` module
    exists alongside ``mobile/``" ambiguity the issue warns suffix
    matching must not silently resolve.
    """
    _planted_file(
        root / "server" / "src" / "swole" / "schemas" / "coach.py",
        "# coach\n",
    )
    _planted_file(
        root / "server" / "src" / "swole" / "coach" / "config.yaml",
        "restPolicy:\n",
    )
    _planted_file(
        root
        / "swole"
        / "app"
        / "mobile"
        / "src"
        / "main"
        / "kotlin"
        / "dev"
        / "senki"
        / "swole"
        / "data"
        / "repo"
        / "SettingsRepository.kt",
        "// settings\n",
    )
    if with_shared_settings:
        _planted_file(
            root
            / "shared"
            / "src"
            / "commonMain"
            / "kotlin"
            / "dev"
            / "senki"
            / "swole"
            / "data"
            / "repo"
            / "SettingsRepository.kt",
            "// shared settings\n",
        )


def test_real_swole_issue_92_module_relative_citations_resolve(tmp_path: Path) -> None:
    """Acceptance criterion: ``cross_repo_gate`` against #92's real body
    and a faithful swole-root fixture stops counting ``schemas/coach.py``
    as missing once the suffix-match fallback is in place -- the gate
    passes on the "at least one referenced path exists in the target
    repo" branch, with ``schemas/coach.py`` (real file:
    ``server/src/swole/schemas/coach.py``) and ``coach/config.yaml``
    (real file: ``server/src/swole/coach/config.yaml``) both resolved
    through the suffix fallback.

    ``docs/uat/feedback-2026-08-26.md`` stays missing here on purpose:
    per the maintainer's correction on this issue, that file exists in
    the diff of open sibling PR #91 only, absent from the checked-out
    default-branch tree -- a fourth citation shape neither the fallback
    nor any other gate arm addresses (PR #91's merge is #94's actual
    resolution path). The gate still passes: one existing survivor is
    pass evidence regardless of what else is missing.
    """
    root = tmp_path / "swole"
    _swole_layout(root)
    body = (FIXTURES_DIR / "swole_issue_92_body.txt").read_text(encoding="utf-8")

    result = cross_repo_gate(body, root)

    assert result.passed is True
    assert result.reason == "at least one referenced path exists in the target repo"
    assert result.referenced_paths == (
        "docs/uat/feedback-2026-08-26.md",
        "coach/config.yaml",
        "schemas/coach.py",
    )
    assert result.missing_paths == ("docs/uat/feedback-2026-08-26.md",)


def test_single_suffix_match_resolves_on_the_function(tmp_path: Path) -> None:
    """Function-level single match: a module-relative citation resolves
    against exactly one real nested file, on a path-segment boundary."""
    root = tmp_path / "swole"
    _swole_layout(root)

    assert cross_repo_gate_module._path_exists_in_repo("schemas/coach.py", root) is True
    assert cross_repo_gate_module._path_exists_in_repo("coach/config.yaml", root) is True


def test_kmp_module_relative_citation_resolves(tmp_path: Path) -> None:
    """The KMP shape from #213/#218: a citation missing only the
    ``swole/app/<module>/`` prefix (e.g. ``ui/design/Motion.kt`` for the
    real ``swole/app/mobile/src/main/kotlin/dev/senki/swole/ui/design/Motion.kt``)
    resolves through the same fallback, fixing the Python-nested-module
    and KMP-module cases with one shared rule and no per-repo
    configuration."""
    root = tmp_path / "swole"
    _swole_layout(root)
    _planted_file(
        root
        / "swole"
        / "app"
        / "mobile"
        / "src"
        / "main"
        / "kotlin"
        / "dev"
        / "senki"
        / "swole"
        / "ui"
        / "design"
        / "Motion.kt",
        "// motion\n",
    )
    body = "Replace `ui/design/Motion.kt`'s clock read with an expect/actual seam."

    result = cross_repo_gate(body, root)

    assert result.passed is True
    assert result.referenced_paths == ("ui/design/Motion.kt",)
    assert result.missing_paths == ()


def test_ambiguous_suffix_still_counts_as_missing(tmp_path: Path) -> None:
    """Acceptance criterion: a candidate whose suffix matches more than
    one real file is still treated as unresolved (missing), not silently
    accepted. Once a real ``shared/`` module exists alongside ``mobile/``,
    both carry ``.../data/repo/SettingsRepository.kt``; the cited
    ``data/repo/SettingsRepository.kt`` is ambiguous, so the fallback
    returns nothing and the gate abstains with the candidate still
    counted in ``missing_paths``."""
    root = tmp_path / "swole"
    _swole_layout(root, with_shared_settings=True)
    body = "Move `data/repo/SettingsRepository.kt` into the shared module."

    result = cross_repo_gate(body, root)

    assert result.passed is True
    assert result.missing_paths == ("data/repo/SettingsRepository.kt",)
    assert "abstaining" in result.reason


def test_ambiguous_suffix_is_missing_on_the_function(tmp_path: Path) -> None:
    root = tmp_path / "swole"
    _swole_layout(root, with_shared_settings=True)

    assert (
        cross_repo_gate_module._path_exists_in_repo("data/repo/SettingsRepository.kt", root)
        is False
    )


def test_raw_string_suffix_never_crosses_segment_boundary(tmp_path: Path) -> None:
    """The issue's own example: ``"ngs/coach.py"`` is a raw string suffix of
    ``".../settings/coach.py"`` but must not match it -- a suffix match
    has to start at a path-segment boundary, so the candidate stays
    missing."""
    root = tmp_path / "swole"
    _swole_layout(root)
    _planted_file(
        root / "server" / "src" / "swole" / "settings" / "coach.py",
        "# settings coach\n",
    )

    assert cross_repo_gate_module._path_exists_in_repo("ngs/coach.py", root) is False


def test_segment_boundary_match_helper_semantics() -> None:
    """Pin :func:`_segment_boundary_suffix_match`'s own contract:
    exact-path equality wins, a segment-boundary suffix resolves to the
    single matching file, a raw non-boundary suffix fails, and an
    ambiguous leaf name (2+ files share the candidate) resolves to
    nothing rather than guessing."""
    files = [
        "server/src/swole/schemas/coach.py",
        "swole/settings/coach.py",
        "src/real.py",
    ]
    match = cross_repo_gate_module._segment_boundary_suffix_match

    assert match("server/src/swole/schemas/coach.py", files) == (
        "server/src/swole/schemas/coach.py"
    )
    assert match("schemas/coach.py", files) == "server/src/swole/schemas/coach.py"
    assert match("settings/coach.py", files) == "swole/settings/coach.py"
    assert match("src/real.py", files) == "src/real.py"
    assert match("ngs/coach.py", files) is None
    assert match("coach.py", files) is None
    assert match("missing/only.py", files) is None


def test_repos_where_direct_resolution_succeeds_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion: repos where the direct repo-root resolution
    already succeeds see no behavior change -- the listing is never even
    computed, so no suffix search can overturn a direct verdict."""
    root = tmp_path / "swole"
    _planted_file(root / "schemas" / "coach.py", "# direct\n")

    call_count = 0
    real_repo_tracked_files = cross_repo_gate_module._repo_tracked_files

    def _counting_repo_tracked_files(repo_root: Path) -> list[str]:
        nonlocal call_count
        call_count += 1
        return real_repo_tracked_files(repo_root)

    monkeypatch.setattr(
        cross_repo_gate_module, "_repo_tracked_files", _counting_repo_tracked_files
    )

    assert cross_repo_gate_module._path_exists_in_repo("schemas/coach.py", root) is True
    assert call_count == 0


def test_listing_computed_once_per_gate_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The issue's "cache the full file listing ... to keep this cheap":
    the dispatching root's listing is computed at most once per
    :func:`cross_repo_gate` call -- lazily on the first direct miss, then
    shared across every survivor checked against the same root within the
    call, the same per-call memoization the sibling search uses for its
    roots (review finding 1: an uncached listing per candidate was the
    37-second-per-call liveness hazard)."""
    root = tmp_path / "swole"
    _swole_layout(root)
    _planted_file(
        root
        / "swole"
        / "app"
        / "mobile"
        / "src"
        / "main"
        / "kotlin"
        / "dev"
        / "senki"
        / "swole"
        / "ui"
        / "design"
        / "Motion.kt",
        "// motion\n",
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

    # Two candidates, both requiring the suffix fallback against the SAME
    # dispatching root within one gate call.
    body = "Fix `schemas/coach.py` and review `ui/design/Motion.kt`."

    result = cross_repo_gate(body, root)

    assert result.passed is True
    assert result.missing_paths == ()
    assert call_count == 1
