"""``instructions.md`` + ``_shared`` partial rendering (issue #3117)."""

from __future__ import annotations

from pathlib import Path

from charlie_work.foreman import paths
from charlie_work.foreman.instructions import load_instructions


def _setup(tmp_path: Path, name: str = "g1") -> Path:
    fdir = paths.foreman_dir(tmp_path, name)
    fdir.mkdir(parents=True)
    return fdir


def test_plain_instructions_round_trip(tmp_path: Path) -> None:
    fdir = _setup(tmp_path)
    (fdir / "instructions.md").write_text("do the thing\n", encoding="utf-8")
    result = load_instructions(tmp_path, fdir)
    assert result.ok
    assert result.value == "do the thing\n"


def test_missing_instructions_reads_empty(tmp_path: Path) -> None:
    fdir = _setup(tmp_path)
    result = load_instructions(tmp_path, fdir)
    assert result.ok and result.value == ""


def test_shared_partial_substitution(tmp_path: Path) -> None:
    fdir = _setup(tmp_path)
    shared = paths.shared_dir(tmp_path)
    shared.mkdir(parents=True)
    (shared / "issue-style.md").write_text("TITLE BODY LABELS\n", encoding="utf-8")
    (shared / "operator-facts.md").write_text("facts here\n", encoding="utf-8")
    (fdir / "instructions.md").write_text(
        "style: $section_issue-style\nfacts: $section_operator-facts\n", encoding="utf-8"
    )
    result = load_instructions(tmp_path, fdir)
    assert result.ok, result.error
    # Partial bodies are inserted verbatim, trailing newline included.
    assert result.value == "style: TITLE BODY LABELS\n\nfacts: facts here\n\n"


def test_missing_partial_is_an_error_value(tmp_path: Path) -> None:
    fdir = _setup(tmp_path)
    (fdir / "instructions.md").write_text("needs $section_nope\n", encoding="utf-8")
    result = load_instructions(tmp_path, fdir)
    assert not result.ok
    assert "nope" in result.error


def test_partial_from_another_foreman_is_not_shared(tmp_path: Path) -> None:
    """Only ``_shared`` resolves partials — a sibling foreman's files do not leak in."""
    fdir = _setup(tmp_path)
    other = _setup(tmp_path, "g2")
    (other / "instructions.md").write_text("private\n", encoding="utf-8")
    (fdir / "instructions.md").write_text("$section_instructions\n", encoding="utf-8")
    result = load_instructions(tmp_path, fdir)
    assert not result.ok


def test_symlinked_partial_resolving_outside_is_skipped(tmp_path: Path) -> None:
    fdir = _setup(tmp_path)
    shared = paths.shared_dir(tmp_path)
    shared.mkdir(parents=True)
    outside = tmp_path / "secret.md"
    outside.write_text("secret\n", encoding="utf-8")
    try:
        (shared / "planted.md").symlink_to(outside)
    except OSError:
        import pytest

        pytest.skip("symlinks unavailable")
    (fdir / "instructions.md").write_text("$section_planted\n", encoding="utf-8")
    result = load_instructions(tmp_path, fdir)
    assert not result.ok
    assert "planted" in result.error
