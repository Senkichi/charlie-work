"""G3: exclude-set resolution tests."""

from __future__ import annotations

from pathlib import Path

from charlie_work.attachment_contracts.excludes import Excludes, load_excludes


def test_load_excludes_no_pyproject_returns_empty(tmp_path: Path) -> None:
    excludes = load_excludes(tmp_path)
    assert excludes.exclude_globs == ()
    assert excludes.blame_ignore_shas == frozenset()


def test_load_excludes_reads_configured_globs(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.attachment-contracts]\nexclude_globs = ["legacy/**", "scripts/one_off.py"]\n',
        encoding="utf-8",
    )
    excludes = load_excludes(tmp_path)
    assert excludes.exclude_globs == ("legacy/**", "scripts/one_off.py")


def test_load_excludes_missing_section_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")
    excludes = load_excludes(tmp_path)
    assert excludes.exclude_globs == ()


def test_load_excludes_reads_blame_ignore_revs(tmp_path: Path) -> None:
    (tmp_path / ".git-blame-ignore-revs").write_text(
        "# comment\nabc123\n\ndef456\n", encoding="utf-8"
    )
    excludes = load_excludes(tmp_path)
    assert excludes.blame_ignore_shas == frozenset({"abc123", "def456"})


def test_structural_excludes_are_always_on() -> None:
    excludes = Excludes()
    assert excludes.is_excluded_path("src/.venv/pkg/x.py") is True
    assert excludes.is_excluded_path("src/pkg/__pycache__/x.py") is True
    assert excludes.is_excluded_path("node_modules/pkg/x.py") is True
    assert excludes.is_excluded_path(".var/state/x.py") is True
    assert excludes.is_excluded_path("src/pkg/generated/x.py") is True
    assert excludes.is_excluded_path("src/pkg/vendor/x.py") is True
    assert excludes.is_excluded_path(".claude/worktrees/foo/src/x.py") is True


def test_normal_path_not_excluded_by_default() -> None:
    excludes = Excludes()
    assert excludes.is_excluded_path("src/charlie_work/attachment_contracts/model.py") is False


def test_configured_glob_excludes_matching_path() -> None:
    excludes = Excludes(exclude_globs=("legacy/**",))
    assert excludes.is_excluded_path("legacy/old_module.py") is True
    assert excludes.is_excluded_path("src/current/module.py") is False


def test_is_excluded_dir_checks_bare_name() -> None:
    excludes = Excludes()
    assert excludes.is_excluded_dir(".venv") is True
    assert excludes.is_excluded_dir("vendor") is True
    assert excludes.is_excluded_dir("src") is False


def test_is_codemod_commit_threshold() -> None:
    excludes = Excludes()
    assert excludes.is_codemod_commit(20) is False
    assert excludes.is_codemod_commit(21) is True
