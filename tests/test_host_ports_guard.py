"""Regrowth guard: no new direct clock/liveness reads in host-port consumers.

``workflow.py``, ``orchestration/``, ``merge_path/`` and ``dead_worker_sweep/``
read time and pid liveness through the host port (``host.current()`` /
``app.host``).  This AST scan counts the remaining direct ``datetime.now(``,
``time.monotonic(`` and ``is_pid_alive(`` calls per file and compares against
``tests/baselines/host_ports/<path>.count``.  The baseline is a ratchet: growth
fails, and so does shrinkage until the baseline is lowered (regenerate with
``HOST_PORTS_GUARD_WRITE=1``), so it can only ever move down.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "charlie_work"
BASELINES = Path(__file__).resolve().parent / "baselines" / "host_ports"
SCAN_TARGETS = ("workflow.py", "orchestration", "merge_path", "dead_worker_sweep")


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id if func.id == "is_pid_alive" else None
    if isinstance(func, ast.Attribute):
        if func.attr == "is_pid_alive":
            return "is_pid_alive"
        if isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            if pair in {("datetime", "now"), ("time", "monotonic")}:
                return f"{pair[0]}.{pair[1]}"
    return None


def count_direct_host_reads(source: str) -> int:
    """Count direct clock / pid-liveness call sites in ``source``."""
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and _call_name(node) is not None
    )


def _scan_files() -> list[Path]:
    files: list[Path] = []
    for target in SCAN_TARGETS:
        path = SRC / target
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def _baseline_path(rel: str) -> Path:
    return BASELINES / f"{rel}.count"


def test_counter_positive_control() -> None:
    assert count_direct_host_reads("x = datetime.now(UTC)\n") == 1
    assert count_direct_host_reads("x = time.monotonic()\nis_pid_alive(1)\n") == 2
    assert count_direct_host_reads("x = host.clock.now()\n") == 0


def test_scan_targets_exist() -> None:
    files = _scan_files()
    assert len(files) > 5 and all(f.exists() for f in files)


@pytest.mark.parametrize("path", _scan_files(), ids=_rel)
def test_no_new_direct_host_reads(path: Path) -> None:
    rel = _rel(path)
    count = count_direct_host_reads(path.read_text(encoding="utf-8"))
    baseline_file = _baseline_path(rel)
    if os.environ.get("HOST_PORTS_GUARD_WRITE") == "1":
        if count:
            baseline_file.parent.mkdir(parents=True, exist_ok=True)
            baseline_file.write_text(f"{count}\n", encoding="utf-8")
        elif baseline_file.exists():
            baseline_file.unlink()
        return
    baseline = (
        int(baseline_file.read_text(encoding="utf-8").strip()) if baseline_file.exists() else 0
    )
    assert count <= baseline, (
        f"{rel}: {count} direct datetime.now/time.monotonic/is_pid_alive "
        f"call(s), baseline {baseline}. Read time and liveness through "
        "host.current() / app.host instead."
    )
    assert count >= baseline, (
        f"{rel}: down to {count} (baseline {baseline}). Lower the ratchet: "
        "HOST_PORTS_GUARD_WRITE=1 uv run --no-sync pytest "
        "tests/test_host_ports_guard.py"
    )
