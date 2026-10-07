"""Regrowth guard: no new direct clock/liveness reads in host-port consumers.

``workflow.py``, ``orchestration/``, ``merge_path/`` and ``dead_worker_sweep/``
read time and pid liveness through the host port (``host.current()`` /
``app.host``).  This AST scan counts the remaining direct ``datetime.now(``,
``time.monotonic(``, ``time.time(`` and ``is_pid_alive(`` calls per file and
compares against ``tests/baselines/host_ports/<path>.count``.  The baseline is
a ratchet: growth fails, and so does shrinkage until the baseline is lowered
(regenerate with ``HOST_PORTS_GUARD_WRITE=1``), so it can only ever move down.

The scan resolves the file's own imports, so aliased spellings of the same
call still count: ``import datetime as _dt; _dt.datetime.now()``,
``from time import monotonic; monotonic()`` and
``from ... import is_pid_alive as alive; alive()`` all match (issue #2237).

Liveness additionally has a repo-wide hard floor (issue #2228): after the host
probe migration, ``is_pid_alive(`` may be called directly only inside
``process_utils.py`` (the primitive) and ``host/liveness.py`` (the late-binding
adapter).  ``test_no_direct_is_pid_alive`` scans every file under
``src/charlie_work`` outside that two-file allowlist at baseline 0, so a
reintroduced direct call fails here rather than regrowing silently.

Both file scans stay parametrized per scanned path so their leaf names persist
for the collect-only gate: deleting or renaming a scanned module drops a
parametrize id, which the gate reads as a removed test leaf (issue #1538).
The escape hatch is a skipped leaf with the retired id — see
``_RETIRED_LIVENESS_MODULES`` below and ``_RETIRED_MODULES`` in
test_sink_statuses_no_literal_pair.py — never an aggregate test, which would
drop every per-path leaf at once.  A baseline file whose module left the scan
is reported by ``test_no_orphaned_host_ports_baselines`` instead of silently
rotting (issue #2237).
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

# Dotted call names that bypass the host ports. ``is_pid_alive`` also matches
# any ``*.is_pid_alive`` attribute call — either form bypasses the probe.
_DIRECT_HOST_READS = frozenset(
    {
        "datetime.now",  # from datetime import datetime; datetime.now()
        "datetime.datetime.now",  # import datetime; datetime.datetime.now()
        "time.monotonic",
        "time.time",
        "is_pid_alive",
    }
)


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    """Map the file's imported names to their dotted source paths.

    ``import datetime as _dt`` binds ``_dt -> datetime``; ``from time import
    monotonic as m`` binds ``m -> time.monotonic``; ``import a.b`` binds
    ``a -> a`` (the top package).  Names no import binds resolve to
    themselves, so the conventional spellings still match unbound.
    """
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".", 1)[0]
                bindings[name] = alias.name.split(".", 1)[0] if not alias.asname else alias.name
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                target = f"{module}.{alias.name}" if module else alias.name
                bindings[alias.asname or alias.name] = target
    return bindings


def _dotted(node: ast.expr) -> str | None:
    """The dotted path of a Name/Attribute chain, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base is not None else None
    return None


def _matches(name: str | None) -> str | None:
    if name is None:
        return None
    if name == "is_pid_alive" or name.endswith(".is_pid_alive"):
        return "is_pid_alive"
    return name if name in _DIRECT_HOST_READS else None


def _call_name(node: ast.Call, bindings: dict[str, str]) -> str | None:
    literal = _dotted(node.func)
    head, _, tail = (literal or "").partition(".")
    resolved = (bindings[head] + ("." + tail if tail else "")) if head in bindings else None
    # Check both spellings: a file may bind an unrelated name onto a target
    # (``from m import helper as is_pid_alive``) and either shape is suspect.
    return _matches(literal) or _matches(resolved)


def _count_calls(tree: ast.AST, match) -> int:
    bindings = _import_bindings(tree)
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and match(_call_name(node, bindings))
    )


def count_direct_host_reads(source: str) -> int:
    """Count direct clock / pid-liveness call sites in ``source``."""
    return _count_calls(ast.parse(source), lambda name: name is not None)


def _count_is_pid_alive_calls(tree: ast.AST) -> int:
    return _count_calls(tree, lambda name: name == "is_pid_alive")


def count_direct_is_pid_alive(source: str) -> int:
    """Count direct ``is_pid_alive(`` call sites in ``source``.

    Both the bare name and the ``module.is_pid_alive`` attribute form count —
    either one bypasses the host probe port.  Import aliases resolve, so
    ``from ... import is_pid_alive as alive; alive(pid)`` counts too.
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
    # Aliased spellings of the same reads all count (#2237).
    assert count_direct_host_reads("import datetime\nx = datetime.datetime.now()\n") == 1
    assert count_direct_host_reads("import datetime as _dt\nx = _dt.datetime.now()\n") == 1
    assert count_direct_host_reads("from datetime import datetime as dt\nx = dt.now()\n") == 1
    assert count_direct_host_reads("import time\nx = time.time()\n") == 1
    assert count_direct_host_reads("import time as _t\nx = _t.monotonic()\n") == 1
    assert count_direct_host_reads("from time import monotonic\nx = monotonic()\n") == 1
    assert count_direct_host_reads("from time import monotonic as m\nx = m()\n") == 1
    assert count_direct_host_reads("from time import time\nx = time()\n") == 1
    assert count_direct_host_reads("x = other.monotonic()\n") == 0


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
        f"{rel}: {count} direct datetime.now/time.monotonic/time.time/"
        f"is_pid_alive call(s), baseline {baseline}. Read time and liveness "
        "through host.current() / app.host instead."
    )
    assert count >= baseline, (
        f"{rel}: down to {count} (baseline {baseline}). Lower the ratchet: "
        "HOST_PORTS_GUARD_WRITE=1 uv run --no-sync pytest "
        "tests/test_host_ports_guard.py"
    )


def test_no_orphaned_host_ports_baselines() -> None:
    """Report baseline files whose module left the scan (issue #2237).

    Aggregate in its own leaf — the per-file ratchet above stays parametrized
    so its ids persist for the collect-only gate, while this sweep keeps a
    deleted module's ``.count`` file from silently rotting.
    ``HOST_PORTS_GUARD_WRITE=1`` deletes the orphans.
    """
    write = os.environ.get("HOST_PORTS_GUARD_WRITE") == "1"
    scanned = {_rel(path) for path in _scan_files()}
    orphans = [
        baseline_file
        for baseline_file in sorted(BASELINES.rglob("*.count"))
        if baseline_file.relative_to(BASELINES).as_posix()[: -len(".count")] not in scanned
    ]
    if write:
        for baseline_file in orphans:
            baseline_file.unlink()
        return
    assert not orphans, "\n".join(
        f"{baseline_file.relative_to(BASELINES).as_posix()[: -len('.count')]}: "
        "baseline exists but the module is gone from the scan. Regenerate: "
        "HOST_PORTS_GUARD_WRITE=1 uv run --no-sync pytest tests/test_host_ports_guard.py"
        for baseline_file in orphans
    )


def test_is_pid_alive_counter_positive_control() -> None:
    assert count_direct_is_pid_alive("is_pid_alive(1)\n") == 1
    assert count_direct_is_pid_alive("pu.is_pid_alive(pid, start_time)\n") == 1
    assert (
        count_direct_is_pid_alive(
            "from charlie_work.process_utils import is_pid_alive as alive\nx = alive(pid)\n"
        )
        == 1
    )
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
