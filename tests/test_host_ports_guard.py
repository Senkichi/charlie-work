"""Regrowth guard: no new direct clock/liveness reads in host-port consumers.

``workflow.py``, ``orchestration/``, ``merge_path/`` and ``dead_worker_sweep/``
read time and pid liveness through the host port (``host.current()`` /
``app.host``).  This AST scan counts the remaining direct ``datetime.now(``,
``time.monotonic(`` and ``is_pid_alive(`` calls per file and compares against
``tests/baselines/host_ports/<path>.count``.  The baseline is a ratchet: growth
fails, and so does shrinkage until the baseline is lowered (regenerate with
``HOST_PORTS_GUARD_WRITE=1``), so it can only ever move down.

Liveness additionally has a repo-wide hard floor (issue #2228): after the host
probe migration, ``is_pid_alive(`` may be called directly only inside
``process_utils.py`` (the primitive) and ``host/liveness.py`` (the late-binding
adapter).  ``test_no_direct_is_pid_alive`` scans every file under
``src/charlie_work`` outside that two-file allowlist at baseline 0, so a
reintroduced direct call fails here rather than regrowing silently.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from _src_ast import parsed, source_files

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "charlie_work"
BASELINES = Path(__file__).resolve().parent / "baselines" / "host_ports"
SCAN_TARGETS = ("workflow.py", "orchestration", "merge_path", "dead_worker_sweep")

# Files allowed to call ``is_pid_alive`` directly: the primitive's own module
# and the late-binding host adapter.  Every other consumer reads liveness
# through ``host.current().probe`` / ``app.host.probe``.
DIRECT_LIVENESS_ALLOWLIST = ("host/liveness.py", "process_utils.py")

# Source modules deleted by issue #2479 (the History-redesign chart cleanup).
# Their parametrize ids are kept as skipped leaves so the collect-only gate
# sees no removed leaf -- same mechanism as ``_RETIRED_MODULES`` in
# test_sink_statuses_no_literal_pair.py.
_RETIRED_LIVENESS_MODULES = (
    "dashboard/charts/bullet.py",
    "dashboard/charts/line.py",
    "dashboard/charts/model.py",
    "dashboard/charts/multiples.py",
    "dashboard/charts/strip.py",
)


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


def _count_calls(tree: ast.AST, match) -> int:
    return sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.Call) and match(_call_name(node))
    )


def count_direct_host_reads(source: str) -> int:
    """Count direct clock / pid-liveness call sites in ``source``."""
    return _count_calls(ast.parse(source), lambda name: name is not None)


def _count_is_pid_alive_calls(tree: ast.AST) -> int:
    return _count_calls(tree, lambda name: name == "is_pid_alive")


def count_direct_is_pid_alive(source: str) -> int:
    """Count direct ``is_pid_alive(`` call sites in ``source``.

    Both the bare name and the ``module.is_pid_alive`` attribute form count —
    either one bypasses the host probe port.
    """
    return _count_is_pid_alive_calls(ast.parse(source))


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


def _liveness_scan_files() -> list[Path]:
    live = [p for p in source_files(SRC) if _rel(p) not in DIRECT_LIVENESS_ALLOWLIST]
    return [*live, *(SRC / rel for rel in _RETIRED_LIVENESS_MODULES)]


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


def test_is_pid_alive_counter_positive_control() -> None:
    assert count_direct_is_pid_alive("is_pid_alive(1)\n") == 1
    assert count_direct_is_pid_alive("pu.is_pid_alive(pid, start_time)\n") == 1
    assert count_direct_is_pid_alive("probe.is_alive(pid)\n") == 0
    assert count_direct_is_pid_alive("self.host.probe.is_alive(pid)\n") == 0
    assert count_direct_is_pid_alive("x = is_pid_alive  # referenced, not called\n") == 0


@pytest.mark.parametrize("rel", DIRECT_LIVENESS_ALLOWLIST)
def test_liveness_allowlist_still_calls_primitive(rel: str) -> None:
    """Non-vacuity check: the two allowlisted files really do contain direct
    ``is_pid_alive(`` calls, so a rename of the primitive — which would empty
    the scan's target — fails here instead of passing the repo-wide guard on
    zero real call sites."""
    count = _count_is_pid_alive_calls(parsed(SRC / rel))
    assert count >= 1, (
        f"{rel} no longer calls is_pid_alive directly; if the primitive was "
        "renamed, update the scan and this allowlist together."
    )


@pytest.mark.parametrize("path", _liveness_scan_files(), ids=_rel)
def test_no_direct_is_pid_alive(path: Path) -> None:
    """Hard floor at baseline 0 for every non-allowlisted file under
    ``src/charlie_work`` — including modules added after this guard."""
    if not path.exists():
        pytest.skip(f"{path.name} was deleted; id kept for the collect-only gate")
    rel = _rel(path)
    count = _count_is_pid_alive_calls(parsed(path))
    assert count == 0, (
        f"{rel}: {count} direct is_pid_alive( call(s). Route liveness through "
        "host.current().probe / app.host.probe instead; only process_utils.py "
        "(the primitive) and host/liveness.py (the adapter) may call it."
    )
