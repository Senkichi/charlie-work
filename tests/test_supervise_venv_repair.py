"""Editable-``.pth`` self-heal tests: ``_check_venv``/``_repair_venv_pth``.

Split out of ``tests/test_supervise.py`` (issue #1562, Track 1) --
bodies are verbatim relocations; shared helpers live in
``tests/_supervise_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from _supervise_fixtures import _make_fake_runner
from charlie_work.instrumentation import query_events
from charlie_work.subprocess_runner import RunResult
from charlie_work.supervise import (
    SelfDeployResult,
    _check_venv,
    _repair_venv_pth,
    _self_deploy_state_path,
    self_deploy,
)
from charlie_work.worktree import _match_pth_to_root


def _setup_fake_venv(
    repo_root: Path,
    *,
    wrong_target: Path | None = None,
) -> Path:
    """Create a fake venv under ``repo_root/.venv`` with one editable .pth file.

    If ``wrong_target`` is provided, the .pth points there (mismatch).  If
    ``None``, it points at ``repo_root/src`` (healthy).
    """
    site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    pth = site_packages / "_editable_charlie_work.pth"
    target = wrong_target if wrong_target is not None else repo_root / "src"
    pth.write_text(str(target.resolve()) + "\n", encoding="utf-8")
    init_path = repo_root / "src" / "charlie_work" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    init_path.write_text("", encoding="utf-8")
    return pth


def test_check_venv_noop_when_no_venv_found(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """_check_venv is a no-op when _find_venv_path cannot locate a venv."""
    monkeypatch.setattr(
        "charlie_work.supervise._find_venv_path",
        lambda _repo_root: None,
    )

    result = _check_venv(tmp_path)

    assert result == SelfDeployResult(
        ok=True,
        pulled=False,
        changed=False,
        synced=False,
        message="no orchestrator venv found; pth check skipped",
    )


def test_self_deploy_repairs_venv_pth_mismatch(
    tmp_path: Path,
) -> None:
    """A poisoned editable .pth is atomically rewritten to repo_root/src."""
    wrong_target = tmp_path / "wrong" / "src"
    pth_path = _setup_fake_venv(tmp_path, wrong_target=wrong_target)

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "abc123\n", ""),  # after HEAD (no change)
        ],
    )

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is True
    assert result.venv_repaired is True
    assert result.pulled is True
    assert result.synced is False
    assert result.from_sha == "abc123"
    assert result.to_sha == "abc123"
    assert [c[0] for c in calls] == [
        ["git", "rev-parse", "HEAD"],
        ["git", "pull", "--ff-only", "origin", "main"],
        ["git", "rev-parse", "HEAD"],
    ]
    assert pth_path.read_text(encoding="utf-8").strip() == str((tmp_path / "src").resolve())


def test_self_deploy_repairs_venv_pth_mismatch_with_runners_active(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A .pth rewrite is not exe-locked and succeeds while charlie.exe is held open."""
    msvcrt = pytest.importorskip("msvcrt")

    wrong_target = tmp_path / "wrong" / "src"
    pth_path = _setup_fake_venv(tmp_path, wrong_target=wrong_target)

    # Simulate a live orchestrator process image by holding an exclusive byte-range
    # lock on a charlie.exe stand-in in the venv. The .pth rewrite must still succeed.
    charlie_exe = tmp_path / ".venv" / "Scripts" / "charlie.exe"
    charlie_exe.parent.mkdir(parents=True, exist_ok=True)
    charlie_exe.write_bytes(b"MZ fake executable content")
    handle = charlie_exe.open("r+b", encoding=None)

    monkeypatch.setattr(
        "charlie_work.fleet_registry.count_fleet_live_sessions",
        lambda _fleet_dir_override: (2, []),
    )

    runner, calls = _make_fake_runner(
        [
            RunResult(0, "abc123\n", ""),  # before HEAD
            RunResult(0, "", ""),  # pull ok
            RunResult(0, "abc123\n", ""),  # after HEAD (no change)
        ],
    )

    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        result = self_deploy(tmp_path, run_command=runner)

        assert result.ok is True
        assert result.venv_repaired is True
        assert result.pulled is True
        assert result.synced is False
        assert pth_path.read_text(encoding="utf-8").strip() == str((tmp_path / "src").resolve())
        assert [c[0] for c in calls] == [
            ["git", "rev-parse", "HEAD"],
            ["git", "pull", "--ff-only", "origin", "main"],
            ["git", "rev-parse", "HEAD"],
        ]
    finally:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def test_self_deploy_venv_repair_failure_is_non_fatal(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failed .pth repair is returned as a non-fatal error value."""
    wrong_target = tmp_path / "wrong" / "src"
    _setup_fake_venv(tmp_path, wrong_target=wrong_target)
    monkeypatch.setattr(
        "charlie_work.supervise._repair_venv_pth",
        lambda _repo_root, _venv_path: (False, "Access is denied", []),
    )

    runner, calls = _make_fake_runner([RunResult(0, "abc123\n", "")])

    result = self_deploy(tmp_path, run_command=runner)

    assert result.ok is False
    assert result.venv_repaired is False
    assert result.pulled is False
    assert result.changed is False
    assert result.synced is False
    assert result.error is not None
    assert "Access is denied" in result.error
    assert not calls


def _setup_repo_with_peer_dep_venv(
    tmp_path: Path,
    *,
    charlie_work_target: Path | None = None,
    ci_fleet_target: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Create a repo with a relative editable dep and a venv with two .pth files.

    Returns ``(repo_root, peer_src, charlie_pth, ci_fleet_pth)``.  Each .pth
    is written to the given target (or the correct root when ``None``).
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text(
        '[project]\nname = "charlie-work"\nversion = "0.1.0"\n'
        '[tool.uv.sources]\nci-fleet = { path = "../ci_runners", editable = true }\n',
        encoding="utf-8",
    )
    peer_src = tmp_path / "ci_runners" / "src"
    (peer_src / "ci_fleet").mkdir(parents=True)
    (peer_src / "ci_fleet" / "__init__.py").write_text("", encoding="utf-8")

    site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    charlie_pth = site_packages / "_editable_impl_charlie_work.pth"
    ci_fleet_pth = site_packages / "_editable_impl_ci_fleet.pth"
    charlie_pth.write_text(
        str((charlie_work_target or (repo_root / "src")).resolve()) + "\n",
        encoding="utf-8",
    )
    ci_fleet_pth.write_text(str((ci_fleet_target or peer_src).resolve()) + "\n", encoding="utf-8")
    return repo_root, peer_src, charlie_pth, ci_fleet_pth


def test_repair_venv_pth_rewrites_foreign_editable_to_peer_root(
    tmp_path: Path,
) -> None:
    """A poisoned foreign .pth is rewritten to its peer repo src, not repo_root/src (gap 1).

    The old repair used a single ``main_src`` constant and would have rewritten
    ``_editable_impl_ci_fleet.pth`` to ``charlie-work/src`` -- a hard
    ``ImportError`` because no ``ci_fleet`` package lives there.
    """
    scratch = tmp_path / "scratch" / "src"
    scratch.mkdir(parents=True)
    repo_root, peer_src, _charlie_pth, ci_fleet_pth = _setup_repo_with_peer_dep_venv(
        tmp_path, ci_fleet_target=scratch
    )

    ok, message, repaired = _repair_venv_pth(repo_root, repo_root / ".venv")

    assert ok
    assert ci_fleet_pth.read_text(encoding="utf-8").strip() == str(peer_src.resolve())
    assert "configured checkouts" in message
    assert "_editable_impl_ci_fleet.pth" in repaired


def test_repair_venv_pth_detects_and_rewrites_foreign_editable_repointed_at_wrong_root(
    tmp_path: Path,
) -> None:
    """Cross-root false-green repair (fast-follow #1180).

    ``_editable_impl_ci_fleet.pth`` repointed at ``charlie-work/src`` (a
    *different* configured root) is detected as poisoned -- the old "any
    configured root" detection pass would have accepted it because
    ``charlie-work/src`` IS a configured root, leaving a silent
    ``ImportError``.  The per-package detection flags it, and the rewrite
    restores the correct peer root.
    """
    # _setup_repo_with_peer_dep_venv creates repo_root at tmp_path / "repo".
    wrong_root = (tmp_path / "repo" / "src").resolve()
    repo_root, peer_src, _charlie_pth, ci_fleet_pth = _setup_repo_with_peer_dep_venv(
        tmp_path, ci_fleet_target=wrong_root
    )

    ok, message, repaired = _repair_venv_pth(repo_root, repo_root / ".venv")

    assert ok
    assert ci_fleet_pth.read_text(encoding="utf-8").strip() == str(peer_src.resolve())
    assert "_editable_impl_ci_fleet.pth" in repaired


def test_repair_venv_pth_refuses_unknown_foreign_editable(tmp_path: Path) -> None:
    """A poisoned .pth whose correct root is unknown is left untouched (gap 1).

    Refusing to write a wrong root is strictly safer than guessing: a missed
    repair surfaces as a verification mismatch on re-check, while a wrong
    repair surfaces as a silent ``ImportError`` that verifies clean.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    # No pyproject.toml -> no peer dep -> ci_fleet root is not derivable.
    scratch = tmp_path / "scratch" / "src"
    scratch.mkdir(parents=True)
    site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    unknown_pth = site_packages / "_editable_impl_ci_fleet.pth"
    unknown_pth.write_text(str(scratch.resolve()) + "\n", encoding="utf-8")
    original_content = unknown_pth.read_text(encoding="utf-8")

    ok, message, repaired = _repair_venv_pth(repo_root, repo_root / ".venv")

    assert not ok
    assert "could not determine correct root" in message
    assert "_editable_impl_ci_fleet.pth" in message
    # No files were repaired -- the unknown .pth was left untouched.
    assert repaired == []
    # The file is left untouched -- no ImportError written.
    assert unknown_pth.read_text(encoding="utf-8") == original_content


def test_repair_venv_pth_partial_repair_observable_despite_overall_failure(
    tmp_path: Path,
) -> None:
    """A mixed matchable/unmatchable scenario records the partial repair (PR #1176 review).

    When some poisoned .pth files are successfully rewritten but others are
    unrepairable, the overall call returns ``False`` -- but the successfully
    repaired files must not be invisible.  The return value's ``repaired_files``
    list makes the partial repair observable so it is indistinguishable neither
    from a no-op failure nor from a full success.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    # No pyproject.toml peer dep for ci_fleet -> its root is not derivable.
    correct_charlie_src = (repo_root / "src").resolve()
    wrong_charlie_target = (tmp_path / "wrong_charlie" / "src").resolve()
    wrong_charlie_target.mkdir(parents=True)
    wrong_ci_fleet_target = (tmp_path / "wrong_fleet" / "src").resolve()
    wrong_ci_fleet_target.mkdir(parents=True)

    site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    charlie_pth = site_packages / "_editable_impl_charlie_work.pth"
    charlie_pth.write_text(str(wrong_charlie_target) + "\n", encoding="utf-8")
    unknown_fleet_pth = site_packages / "_editable_impl_ci_fleet.pth"
    unknown_fleet_pth.write_text(str(wrong_ci_fleet_target) + "\n", encoding="utf-8")
    original_fleet_content = unknown_fleet_pth.read_text(encoding="utf-8")

    ok, message, repaired = _repair_venv_pth(repo_root, repo_root / ".venv")

    # Overall failure: the unmatchable ci_fleet .pth could not be repaired.
    assert not ok
    assert "could not determine correct root" in message
    assert "_editable_impl_ci_fleet.pth" in message
    # The matchable charlie_work .pth WAS successfully rewritten.
    assert charlie_pth.read_text(encoding="utf-8").strip() == str(correct_charlie_src)
    # The partial repair is observable in the return value.
    assert "_editable_impl_charlie_work.pth" in repaired
    assert "_editable_impl_ci_fleet.pth" not in repaired
    # The unmatchable file is left untouched.
    assert unknown_fleet_pth.read_text(encoding="utf-8") == original_fleet_content


def test_check_venv_partial_repair_event_records_repaired_files(
    tmp_path: Path,
) -> None:
    """A partial repair's venv_pth_repair_failed event includes repaired_files (PR #1176 review).

    Goes through ``_check_venv`` so the event path is exercised end-to-end.
    The matchable charlie_work .pth is rewritten; the unmatchable ci_fleet .pth
    is left untouched.  The overall result is failure, but the
    ``venv_pth_repair_failed`` event's payload carries ``repaired_files`` so the
    partial repair is not indistinguishable from a no-op failure in events.db.
    """
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    (repo_root / "src" / "charlie_work").mkdir(parents=True)
    (repo_root / "src" / "charlie_work" / "__init__.py").write_text("", encoding="utf-8")
    # No pyproject.toml -> no peer dep -> ci_fleet root is not derivable.
    correct_charlie_src = (repo_root / "src").resolve()
    wrong_charlie_target = (tmp_path / "wrong_charlie" / "src").resolve()
    wrong_charlie_target.mkdir(parents=True)
    wrong_ci_fleet_target = (tmp_path / "wrong_fleet" / "src").resolve()
    wrong_ci_fleet_target.mkdir(parents=True)

    site_packages = repo_root / ".venv" / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    charlie_pth = site_packages / "_editable_impl_charlie_work.pth"
    charlie_pth.write_text(str(wrong_charlie_target) + "\n", encoding="utf-8")
    unknown_fleet_pth = site_packages / "_editable_impl_ci_fleet.pth"
    unknown_fleet_pth.write_text(str(wrong_ci_fleet_target) + "\n", encoding="utf-8")

    state_path = _self_deploy_state_path(repo_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")

    result = _check_venv(repo_root)

    # Overall failure: the unmatchable ci_fleet .pth could not be repaired.
    assert result.ok is False
    assert result.venv_repaired is False
    assert "could not determine correct root" in (result.error or "")
    # The matchable charlie_work .pth WAS successfully rewritten.
    assert charlie_pth.read_text(encoding="utf-8").strip() == str(correct_charlie_src)
    # The partial repair is observable in the venv_pth_repair_failed event.
    failed_events = query_events(state_path, kind="venv_pth_repair_failed")
    assert len(failed_events) == 1
    payload = failed_events[0]["payload"]
    assert "repaired_files" in payload
    assert "_editable_impl_charlie_work.pth" in payload["repaired_files"]
    assert "_editable_impl_ci_fleet.pth" not in payload["repaired_files"]


def test_match_pth_to_root_returns_correct_root() -> None:
    """_match_pth_to_root maps a .pth filename to its configured src root."""
    repo_src = Path("C:/repo/src")
    peer_src = Path("C:/ci_runners/src")
    package_to_root = {"charlie_work": repo_src, "ci_fleet": peer_src}

    assert _match_pth_to_root(Path("_editable_impl_charlie_work.pth"), package_to_root) == repo_src
    assert _match_pth_to_root(Path("_editable_impl_ci_fleet.pth"), package_to_root) == peer_src


def test_match_pth_to_root_returns_none_for_unknown() -> None:
    """An unrecognized .pth filename yields None so the caller refuses to repair."""
    package_to_root = {"charlie_work": Path("C:/repo/src")}
    assert _match_pth_to_root(Path("_editable_impl_ci_fleet.pth"), package_to_root) is None


def test_check_venv_emits_mismatch_and_repaired_events(tmp_path: Path) -> None:
    """A detected mismatch emits venv_pth_mismatch; a successful repair emits venv_pth_repaired."""
    wrong_target = tmp_path / "wrong" / "src"
    pth_path = _setup_fake_venv(tmp_path, wrong_target=wrong_target)
    state_path = _self_deploy_state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")

    result = _check_venv(tmp_path)

    assert result.ok is True
    assert result.venv_repaired is True
    assert pth_path.read_text(encoding="utf-8").strip() == str((tmp_path / "src").resolve())

    mismatch_events = query_events(state_path, kind="venv_pth_mismatch")
    repaired_events = query_events(state_path, kind="venv_pth_repaired")
    assert len(mismatch_events) == 1
    assert len(repaired_events) == 1
    assert mismatch_events[0]["payload"]["detail"]


def test_check_venv_emits_repair_failed_event(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A failed repair emits venv_pth_repair_failed and returns a non-fatal error."""
    wrong_target = tmp_path / "wrong" / "src"
    _setup_fake_venv(tmp_path, wrong_target=wrong_target)
    state_path = _self_deploy_state_path(tmp_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "charlie_work.supervise._repair_venv_pth",
        lambda _repo_root, _venv_path: (False, "Access is denied", []),
    )

    result = _check_venv(tmp_path)

    assert result.ok is False
    assert result.venv_repaired is False
    assert "Access is denied" in result.error
    failed_events = query_events(state_path, kind="venv_pth_repair_failed")
    assert len(failed_events) == 1
    assert "Access is denied" in failed_events[0]["payload"]["detail"]


def test_check_venv_emits_mismatch_event_for_foreign_editable(
    tmp_path: Path,
) -> None:
    """A foreign editable mismatch emits venv_pth_mismatch even when the main .pth is healthy.

    This is the false-green scenario: the old filter would have repaired only
    the main .pth and reported success.  The resolved-target test catches the
    foreign mismatch, and the event makes it observable.
    """
    scratch = tmp_path / "scratch" / "src"
    scratch.mkdir(parents=True)
    repo_root, peer_src, _charlie_pth, _ci_fleet_pth = _setup_repo_with_peer_dep_venv(
        tmp_path, ci_fleet_target=scratch
    )
    state_path = _self_deploy_state_path(repo_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{}", encoding="utf-8")

    result = _check_venv(repo_root)

    assert result.ok is True
    assert result.venv_repaired is True
    mismatch_events = query_events(state_path, kind="venv_pth_mismatch")
    assert len(mismatch_events) == 1
    assert "_editable_impl_ci_fleet.pth" in mismatch_events[0]["payload"]["detail"]
    repaired_events = query_events(state_path, kind="venv_pth_repaired")
    assert len(repaired_events) == 1
