"""Tests for ``tests/_xdist_policy.py`` (HS-CW-1): small explicit runs stay serial."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from _xdist_policy import SMALL_RUN_MAX_ARGS, small_run_workers


def _touch(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def test_single_file_runs_serially(tmp_path: Path) -> None:
    _touch(tmp_path, "test_a.py")
    args = ["test_a.py"]
    assert small_run_workers(args, tmp_path, args, from_command_line=True) == 0


def test_up_to_three_nodeids_run_serially(tmp_path: Path) -> None:
    for name in ("test_a.py", "test_b.py", "test_c.py"):
        _touch(tmp_path, name)
    args = ["test_a.py::test_x", "test_b.py::test_y[1-2]", "test_c.py"]
    assert len(args) == SMALL_RUN_MAX_ARGS
    assert small_run_workers(args, tmp_path, args, from_command_line=True) == 0


def test_absolute_path_runs_serially(tmp_path: Path) -> None:
    target = _touch(tmp_path, "test_a.py")
    args = [str(target)]
    assert small_run_workers(args, tmp_path / "elsewhere", args, from_command_line=True) == 0


def test_more_than_three_targets_distribute(tmp_path: Path) -> None:
    args = [f"test_{i}.py" for i in range(SMALL_RUN_MAX_ARGS + 1)]
    for name in args:
        _touch(tmp_path, name)
    assert small_run_workers(args, tmp_path, args, from_command_line=True) is None


def test_directory_target_distributes(tmp_path: Path) -> None:
    _touch(tmp_path, "sub/test_a.py")
    args = ["sub"]
    assert small_run_workers(args, tmp_path, args, from_command_line=True) is None


def test_missing_path_distributes(tmp_path: Path) -> None:
    args = ["test_missing.py"]
    assert small_run_workers(args, tmp_path, args, from_command_line=True) is None


def test_testpaths_default_distributes(tmp_path: Path) -> None:
    _touch(tmp_path, "test_a.py")
    assert small_run_workers(["test_a.py"], tmp_path, [], from_command_line=False) is None


@pytest.mark.parametrize("flag", [["-n", "auto"], ["-nauto"], ["--numprocesses=auto"]])
def test_explicit_numprocesses_is_never_overridden(tmp_path: Path, flag: list[str]) -> None:
    _touch(tmp_path, "test_a.py")
    args = ["test_a.py"]
    assert small_run_workers(args, tmp_path, [*flag, *args], from_command_line=True) is None


def test_conftest_hook_delegates_to_the_policy(tmp_path: Path) -> None:
    import conftest

    target = _touch(tmp_path, "test_a.py")
    params = SimpleNamespace(dir=tmp_path, args=(str(target),))
    from_args = SimpleNamespace(
        args=[str(target)], invocation_params=params, args_source=pytest.Config.ArgsSource.ARGS
    )
    from_testpaths = SimpleNamespace(
        args=[str(target)],
        invocation_params=params,
        args_source=pytest.Config.ArgsSource.TESTPATHS,
    )
    assert conftest.pytest_xdist_auto_num_workers(from_args) == 0
    assert conftest.pytest_xdist_auto_num_workers(from_testpaths) is None
