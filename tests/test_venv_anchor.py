"""Tests for charlie_work.venv_anchor — interpreter-anchored editable-.pth guard.

Everything is built from tmp_path fixtures. ``prefix`` is passed explicitly in
every test except the ``prefix=None`` cases, which monkeypatch ``sys.prefix``/
``sys.base_prefix`` instead of touching the real interpreter. No network, no
subprocesses, no real venv is required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from charlie_work.venv_anchor import VenvAnchorResult, verify_interpreter_anchored_editables


def _make_venv(root: Path) -> Path:
    """Create ``<root>/.venv/Lib/site-packages`` and return the venv path."""
    venv = root / ".venv"
    site_packages = venv / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    return venv


def _site_packages(venv: Path) -> Path:
    return venv / "Lib" / "site-packages"


def _write_pyproject(root: Path, sources_toml: str = "") -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fake-project"\nversion = "0.0.0"\n' + sources_toml,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 1. Clean case
# ---------------------------------------------------------------------------


def test_clean_pth_targets_ok(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    sibling_src = tmp_path / "sibling" / "src"
    sibling_src.mkdir(parents=True)
    (root / "src").mkdir()

    _write_pyproject(
        root,
        '\n[tool.uv.sources]\nsomedep = { path = "../sibling" }\n',
    )
    venv = _make_venv(root)
    site_packages = _site_packages(venv)
    (site_packages / "own_project.pth").write_text(str(root / "src") + "\n", encoding="utf-8")
    (site_packages / "somedep.pth").write_text(str(sibling_src) + "\n", encoding="utf-8")

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is True
    assert result.abstained is False
    assert isinstance(result, VenvAnchorResult)


# ---------------------------------------------------------------------------
# 2. Poisoned case — the incident shape
# ---------------------------------------------------------------------------


def test_poisoned_pth_target_is_violation(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "src").mkdir()
    _write_pyproject(root)
    venv = _make_venv(root)
    site_packages = _site_packages(venv)

    scratch_clone = tmp_path / "scratchpad" / "some-unrelated-clone" / "src"
    scratch_clone.mkdir(parents=True)
    poisoned_pth = site_packages / "charlie_work.pth"
    poisoned_pth.write_text(str(scratch_clone) + "\n", encoding="utf-8")

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is False
    assert result.abstained is False
    assert poisoned_pth.name in result.detail
    assert str(scratch_clone) in result.detail


# ---------------------------------------------------------------------------
# 3. Import lines / comments / blank lines ignored
# ---------------------------------------------------------------------------


def test_import_and_comment_and_blank_lines_ignored(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "src").mkdir()
    _write_pyproject(root)
    venv = _make_venv(root)
    site_packages = _site_packages(venv)

    scratch_clone = tmp_path / "scratchpad" / "unrelated" / "src"
    scratch_clone.mkdir(parents=True)

    # A .pth file whose only "content" lines are non-target lines, plus one
    # real (clean) target line — the import/comment/blank lines must not be
    # mistaken for violations or otherwise affect the outcome.
    (site_packages / "_virtualenv.pth").write_text(
        "import _virtualenv\n# a comment describing this file\n\n   \n" + str(root / "src") + "\n",
        encoding="utf-8",
    )

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is True
    assert result.abstained is False

    # Sanity: if those lines WERE treated as targets they'd resolve to bogus
    # paths outside any allowed root and should have tripped a violation —
    # confirm the scratch clone (untouched by this file) plays no role here.
    assert str(scratch_clone) not in result.detail


# ---------------------------------------------------------------------------
# 4. In-venv target is allowed
# ---------------------------------------------------------------------------


def test_target_inside_venv_itself_is_allowed(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "src").mkdir()
    _write_pyproject(root)
    venv = _make_venv(root)
    site_packages = _site_packages(venv)

    in_venv_target = site_packages / "some_vendored_pkg"
    in_venv_target.mkdir()
    (site_packages / "vendored.pth").write_text(str(in_venv_target) + "\n", encoding="utf-8")

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is True
    assert result.abstained is False


# ---------------------------------------------------------------------------
# 5. Abstentions
# ---------------------------------------------------------------------------


def test_abstains_when_no_pyproject_above_venv(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    venv = _make_venv(root)
    # deliberately no pyproject.toml written at root

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is True
    assert result.abstained is True
    assert "pyproject.toml" in result.detail


def test_abstains_on_invalid_pyproject_toml(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    venv = _make_venv(root)
    # Garbage bytes that tomllib will reject as invalid TOML.
    (root / "pyproject.toml").write_bytes(b"this is [not valid toml =::: \x00\xff")

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is True
    assert result.abstained is True
    # Detail should surface the exception type raised by tomllib so an
    # operator can distinguish "no pyproject" from "broken pyproject".
    assert "raised" in result.detail
    assert "Error" in result.detail or "Decode" in result.detail


def test_abstains_when_prefix_none_and_not_in_virtualenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_prefix = str(tmp_path / "not-a-venv")
    monkeypatch.setattr(sys, "prefix", fake_prefix)
    monkeypatch.setattr(sys, "base_prefix", fake_prefix)

    result = verify_interpreter_anchored_editables()

    assert result.ok is True
    assert result.abstained is True
    assert "not running inside a virtualenv" in result.detail


# ---------------------------------------------------------------------------
# 6. Relative .pth lines resolve against site-packages
# ---------------------------------------------------------------------------


def test_relative_pth_line_resolves_against_site_packages(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "src").mkdir()
    _write_pyproject(root)
    venv = _make_venv(root)
    site_packages = _site_packages(venv)

    # site-packages/../../../src  ==  venv/Lib/site-packages/../../../src
    # Lib/site-packages -> Lib -> venv -> root, so "../../../src" from
    # site-packages lands on root/src (three levels up from site-packages).
    relative_line = "../../../src"
    resolved_target = (site_packages / relative_line).resolve()
    assert resolved_target == (root / "src").resolve()

    (site_packages / "relative.pth").write_text(relative_line + "\n", encoding="utf-8")

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is True
    assert result.abstained is False


def test_relative_pth_line_escaping_root_is_violation(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    (root / "src").mkdir()
    _write_pyproject(root)
    venv = _make_venv(root)
    site_packages = _site_packages(venv)

    escape_target = tmp_path / "escaped-sibling"
    escape_target.mkdir()
    # Compute a relative path from site_packages to the escape target so this
    # test isn't tied to the exact directory depth of the venv layout.
    import os

    relative_line = os.path.relpath(escape_target, site_packages)
    (site_packages / "escape.pth").write_text(relative_line + "\n", encoding="utf-8")

    result = verify_interpreter_anchored_editables(prefix=venv)

    assert result.ok is False
    assert result.abstained is False
    assert "escape.pth" in result.detail


# ---------------------------------------------------------------------------
# 7. Real-venv smoke test
# ---------------------------------------------------------------------------


def test_real_process_smoke_no_args_never_raises() -> None:
    """Calling with no args in the real test process must not raise.

    Test-run contexts vary (worktree PYTHONPATH shadowing, non-venv
    interpreters, etc.), so this deliberately does not assert on ``ok`` —
    only that the call is well-typed and safe.
    """
    result = verify_interpreter_anchored_editables()

    assert isinstance(result, VenvAnchorResult)
    assert isinstance(result.ok, bool)
    assert isinstance(result.abstained, bool)
    assert isinstance(result.detail, str)
    assert result.detail != ""
