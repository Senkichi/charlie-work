"""``verify_shared_venv``: editable ``.pth`` poisoning detection."""

from __future__ import annotations

from pathlib import Path

from _worktree_fixtures import _init_repo
from _worktree_shared_fixtures import _create_shared_venv
from charlie_work.worktree import verify_shared_venv


def test_verify_shared_venv_catches_pth_pointing_outside_main_checkout(tmp_path: Path) -> None:
    """Editable .pth pointing at a worker worktree is the poisoned-venv case."""
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    worktree = tmp_path / "stolen-worktree"
    worktree.mkdir()
    _create_shared_venv(repo_root, pth_target=worktree / "src")

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert not ok
    # charlie_work's package root is derivable, so the per-package message names
    # the expected root rather than the generic "outside all roots" fallback.
    assert "_editable_impl_charlie_work.pth" in message
    assert "expected its package root" in message
    assert "uv sync --all-extras" in message


def test_verify_shared_venv_approves_pth_pointing_at_main_checkout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    _create_shared_venv(repo_root, pth_target=repo_root / "src")

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert ok
    assert "configured checkouts" in message


def _setup_repo_with_peer_dep(tmp_path: Path) -> tuple[Path, Path]:
    """Create a repo with a relative editable dep on a peer repo.

    Returns ``(repo_root, peer_src)`` where ``repo_root/pyproject.toml``
    declares ``ci-fleet = { path = "../ci_runners", editable = true }`` and
    ``peer_src`` is the peer repo's ``src`` directory containing
    ``ci_fleet/__init__.py``.
    """
    repo_root = tmp_path / "repo"
    _init_repo(repo_root)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "charlie-work"\nversion = "0.1.0"\n'
        '[tool.uv.sources]\nci-fleet = { path = "../ci_runners", editable = true }\n',
        encoding="utf-8",
    )
    peer_root = tmp_path / "ci_runners"
    peer_src = peer_root / "src"
    (peer_src / "ci_fleet").mkdir(parents=True)
    (peer_src / "ci_fleet" / "__init__.py").write_text("", encoding="utf-8")
    return repo_root, peer_src


def test_verify_shared_venv_detects_poisoned_foreign_editable(tmp_path: Path) -> None:
    """A peer-repo editable .pth pointing at a scratch dir is caught (issue #969 gap 2).

    The old filename filter excluded ``_editable_impl_ci_fleet.pth`` because
    ``ci_fleet`` is not a top-level package under this repo's ``src``.  The
    resolved-target test scans every ``.pth`` and flags any path line outside
    all configured roots.
    """
    repo_root, peer_src = _setup_repo_with_peer_dep(tmp_path)
    scratch = tmp_path / "scratch" / "src"
    scratch.mkdir(parents=True)
    site_packages = repo_root / "shared-venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    # Main repo .pth is healthy; foreign .pth is poisoned.
    (site_packages / "_editable_impl_charlie_work.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )
    (site_packages / "_editable_impl_ci_fleet.pth").write_text(
        str(scratch.resolve()) + "\n", encoding="utf-8"
    )

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert not ok
    assert "_editable_impl_ci_fleet.pth" in message
    # ci_fleet's package root is derivable, so the per-package message names the
    # expected peer root rather than the generic "outside all roots" fallback.
    assert "expected its package root" in message


def test_verify_shared_venv_approves_healthy_foreign_editable(tmp_path: Path) -> None:
    """A peer-repo editable .pth pointing at the correct peer src is approved."""
    repo_root, peer_src = _setup_repo_with_peer_dep(tmp_path)
    site_packages = repo_root / "shared-venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_editable_impl_charlie_work.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )
    (site_packages / "_editable_impl_ci_fleet.pth").write_text(
        str(peer_src.resolve()) + "\n", encoding="utf-8"
    )

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert ok
    assert "configured checkouts" in message


def test_verify_shared_venv_catches_foreign_editable_when_main_is_healthy(
    tmp_path: Path,
) -> None:
    """The false-green scenario from issue #969: main .pth healthy, foreign poisoned.

    The old filter + repair would flag the main .pth, repair it, re-verify
    only the main .pth, and report success while the foreign editable was still
    pointing into a scratch tree.  The resolved-target test scans every .pth,
    so the foreign mismatch is surfaced directly.
    """
    repo_root, peer_src = _setup_repo_with_peer_dep(tmp_path)
    scratch = tmp_path / "scratch" / "src"
    scratch.mkdir(parents=True)
    site_packages = repo_root / "shared-venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_editable_impl_charlie_work.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )
    (site_packages / "_editable_impl_ci_fleet.pth").write_text(
        str(scratch.resolve()) + "\n", encoding="utf-8"
    )

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert not ok
    assert "_editable_impl_ci_fleet.pth" in message


def test_verify_shared_venv_ignores_non_path_pth_lines(tmp_path: Path) -> None:
    """``import``/comment ``.pth`` lines are not treated as path targets.

    ``ci_fleet_probe.pth`` and ``_virtualenv.pth`` contain ``import`` lines
    that :func:`_resolve_pth_line` returns an empty path for.  They must not
    trigger a false mismatch under the resolved-target test.
    """
    repo_root, _peer_src = _setup_repo_with_peer_dep(tmp_path)
    site_packages = repo_root / "shared-venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_editable_impl_charlie_work.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )
    (site_packages / "ci_fleet_probe.pth").write_text(
        "import sys; exec(__import__('importlib').import_module('ci_fleet_probe')._probe())\n",
        encoding="utf-8",
    )
    (site_packages / "_virtualenv.pth").write_text("import _virtualenv\n", encoding="utf-8")

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert ok
    assert "configured checkouts" in message


def test_verify_shared_venv_catches_foreign_editable_repointed_at_wrong_root(
    tmp_path: Path,
) -> None:
    """Cross-root false-green (fast-follow #1180): a foreign editable repointed at a
    *different* configured root must be caught, not accepted.

    ``_editable_impl_ci_fleet.pth`` repointed at ``charlie-work/src`` would pass
    the old "any configured root" check because ``charlie-work/src`` IS a
    configured root -- yet import as a silent ``ImportError`` (no ``ci_fleet``
    package lives there).  The per-package detection validates the line against
    *the* root inferred from the ``.pth``'s package (``ci_runners/src``), so the
    wrong-root line is flagged.
    """
    repo_root, peer_src = _setup_repo_with_peer_dep(tmp_path)
    site_packages = repo_root / "shared-venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_editable_impl_charlie_work.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )
    # Foreign editable repointed at the WRONG configured root (charlie-work/src
    # instead of ci_runners/src).  This is the next vector after #969.
    (site_packages / "_editable_impl_ci_fleet.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert not ok
    assert "_editable_impl_ci_fleet.pth" in message
    assert "expected its package root" in message
    # The expected root (peer_src) must be named, not the wrong root.
    assert str(peer_src.resolve()) in message


def test_verify_shared_venv_unknown_package_falls_back_to_any_root(
    tmp_path: Path,
) -> None:
    """When no per-package root is derivable, the "any configured root" fallback holds.

    A ``.pth`` whose filename matches no known package (so ``_match_pth_to_root``
    returns ``None``) is still checked: a line outside *all* configured roots is
    poisoned and surfaces the generic "outside all configured checkouts" message.
    This preserves the #969 gap-2 behaviour for unknown packages.
    """
    repo_root, _peer_src = _setup_repo_with_peer_dep(tmp_path)
    scratch = tmp_path / "scratch" / "src"
    scratch.mkdir(parents=True)
    site_packages = repo_root / "shared-venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "_editable_impl_charlie_work.pth").write_text(
        str((repo_root / "src").resolve()) + "\n", encoding="utf-8"
    )
    # Unknown package name -> _match_pth_to_root returns None -> fallback path.
    (site_packages / "_editable_impl_mystery_pkg.pth").write_text(
        str(scratch.resolve()) + "\n", encoding="utf-8"
    )

    ok, message = verify_shared_venv(repo_root, repo_root / "shared-venv")

    assert not ok
    assert "_editable_impl_mystery_pkg.pth" in message
    assert "points outside all configured checkouts" in message
