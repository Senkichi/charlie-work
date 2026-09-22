"""Maintain the file-size high-water-mark ratchet baseline (issue #1442).

The ratchet itself is enforced by ``tests/test_file_size_ratchet.py``, which
runs in CI on every push and fails any PR that leaves an over-cap tracked
``*.py`` file with MORE physical lines than its recorded high-water mark. This
script is the SOLE writer of the checked-in baseline
(``file_size_ratchet_baseline/`` at the repo root): it can create the
baseline from the live tree (one-time ``--init``) and lower existing marks
after a shrink (the default). The test suite never writes the baseline -- a
pytest run must leave the tree clean, so a PR only ever changes the baseline
as a deliberate, reviewed edit. This script NEVER raises a mark -- raising
requires an explicit, reviewed edit to the baseline entry, by design (#1442).

## Per-entry baseline layout (issue #1802)

The baseline is a directory of one ``.count`` file per covered path
(``file_size_ratchet_baseline/src/charlie_work/workflow.py.count`` records
the mark for ``src/charlie_work/workflow.py``), loaded and written via
``charlie_work.ratchet_baseline``. The previous single JSON document was a
shared append point: concurrent PRs editing different entries conflicted on
the same file, and a CONFLICTING PR gets no pull_request CI run. With one
file per entry, distinct-entry PRs merge cleanly; two PRs raising the SAME
entry still conflict on that file, which is the required control direction.

## Quantized marks

Marks are quantized to multiples of ``MARK_QUANTUM`` (200): every mark this
script writes is ``ceil(lines / 200) * 200``. Two reasons:

* **Merge-conflict damping.** Exact-count marks made the baseline the repo's
  hottest conflict site: any two concurrent PRs changing a monolith's line
  count wrote different values on the same JSON line. With quantized marks,
  growth within a bucket needs no baseline edit at all, and two PRs bumping
  the same file into the same bucket write the identical value (clean merge).
* **Deterministic convergence.** Every writer (this script, and a PR raising
  a mark by hand) uses the same rule -- next multiple of 200 -- so
  independent edits agree byte-for-byte.

A hand-raise in a growth PR must follow the same rule: raise the file's
``.count`` entry to the next multiple of 200, never to the exact line count.
If an entry still conflicts on merge, take the larger value.

Usage::

    # One-time initial generation (sets each over-cap file's mark to its
    # current line count quantized up to a multiple of MARK_QUANTUM).
    python scripts/refresh_file_size_ratchet.py --init

    # Lower-only maintenance: after shrinks, lower each mark to the current
    # line count quantized up (only when that is strictly lower than the
    # recorded mark) and drop entries for files that fell back under the cap.
    # Marks are never raised; new over-cap files are NOT added (those require
    # an explicit reviewed baseline edit).
    python scripts/refresh_file_size_ratchet.py

    # Dry run: print what would change without writing.
    python scripts/refresh_file_size_ratchet.py --dry-run

    # Fixed-point gate (issue #1675): recompute the lower-only refresh in
    # memory and exit non-zero -- listing the pending changes -- when the
    # checked-in baseline is not already a fixed point. Writes nothing. A PR
    # that shrinks an over-cap file or deletes one must lower/drop its mark
    # in the same diff; tests/test_file_size_ratchet.py enforces this mode
    # against the real tree so the tightening cannot be left unclaimed.
    python scripts/refresh_file_size_ratchet.py --check

The covered file set is derived from a live scan of tracked ``*.py`` files
(``git ls-files``) plus the baseline's own keys -- never a hardcoded list
(issue #1375: derive-what-is-covered fails closed). The size cap is the repo's
normal per-module cap of 800 lines.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Fallback so the script also runs under a bare interpreter without the
# package installed (same pattern scripts/ac1b_findings_actionability.py uses).
_FALLBACK_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_FALLBACK_SRC) not in sys.path:
    sys.path.insert(0, str(_FALLBACK_SRC))

from charlie_work.ratchet_baseline import (  # noqa: E402
    BaselineFormatError,
    load_count_baseline,
    write_count_baseline,
)

# The repo's normal per-module line cap. Over-cap files (lines > CAP) are the
# ratchet's covered set. This is the same cap the extraction lineage (#1283
# Phase A) records cap-exemption bands against (e.g. stalled_review_reap.py's
# [1308, 1391] band in tests/test_stalled_review_reap_split.py).
FILE_SIZE_CAP = 800

# Marks are recorded as multiples of this quantum (rounded UP from the live
# line count). Growth within a bucket needs no baseline edit; concurrent PRs
# bumping a file into the same bucket write the identical value and merge
# cleanly. tests/_ratchet_constants.py declares the same constant for the test
# side; tests/test_refresh_file_size_ratchet.py asserts the two stay equal.
MARK_QUANTUM = 200

_BASELINE_NAME = "file_size_ratchet_baseline"


def _quantize_mark(lines: int) -> int:
    """Round ``lines`` up to the next multiple of ``MARK_QUANTUM``.

    An exact multiple is preserved (26400 -> 26400); anything else rounds up
    (26401 -> 26600). This is the single mark-derivation rule every baseline
    writer -- this script, or a reviewed hand-raise in a growth PR -- must use,
    so independent edits produce byte-identical lines.
    """
    return -(-lines // MARK_QUANTUM) * MARK_QUANTUM


def _repo_root() -> Path:
    """Resolve the repo root from this script's location.

    Prefers ``git rev-parse --show-toplevel`` (correct under linked worktrees,
    where the script's parent dir is a worktree, not the common root); falls
    back to the script's ``..`` if git is unavailable.
    """
    script_dir = Path(__file__).resolve().parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return script_dir.parent


def _tracked_py_files(repo_root: Path) -> list[Path]:
    """Return every ``*.py`` file tracked by git under ``repo_root``.

    Uses ``git ls-files`` so untracked scratch files (which are not in any PR
    diff and not subject to the cap) are excluded -- the issue's "tracked
    *.py" scope. Paths are returned relative to ``repo_root``.
    """
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {out.stderr.strip()}")
    return [repo_root / line for line in out.stdout.splitlines() if line.endswith(".py")]


def _line_count(path: Path) -> int:
    """Physical line count of the blob (``len(text.splitlines())``).

    Counts the file as it sits at the PR head -- not diff arithmetic -- so a
    byte-identical extraction (issue #1317) passes trivially: the source file
    shrinks and the new module is a separate path. A trailing newline does not
    add a phantom line (``splitlines`` drops it).
    """
    return len(path.read_text(encoding="utf-8").splitlines())


def _scan_over_cap(repo_root: Path) -> dict[str, int]:
    """Live-scan tracked ``*.py`` files; return ``{rel_path: lines}`` for those
    over the cap."""
    over: dict[str, int] = {}
    for path in _tracked_py_files(repo_root):
        try:
            lines = _line_count(path)
        except OSError:
            continue
        if lines > FILE_SIZE_CAP:
            over[path.relative_to(repo_root).as_posix()] = lines
    return over


def _load_baseline(path: Path) -> dict[str, int]:
    """Load the per-entry baseline directory (issue #1802).

    A missing directory loads as an empty baseline (``--init``'s starting
    state); a present-but-malformed directory raises
    :class:`BaselineFormatError`, which ``main`` turns into a non-zero exit --
    fail closed, never silently a zero baseline.
    """
    if not path.is_dir():
        return {}
    return load_count_baseline(path)


def _write_baseline(path: Path, marks: dict[str, int]) -> None:
    """Sync the baseline directory to exactly *marks*.

    Each write is an atomic temp-file + ``replace`` inside
    ``write_count_baseline`` (CLAUDE.md's state-write invariant); stale
    entries are removed and emptied directories pruned.
    """
    write_count_baseline(path, marks)


def _read_baseline_or_report(baseline_path: Path) -> dict[str, int] | None:
    """``_load_baseline`` with the fail-closed boundary: a malformed directory
    reports and returns ``None`` so callers exit 1 rather than traceback or
    silently treat it as empty."""
    try:
        return _load_baseline(baseline_path)
    except BaselineFormatError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return None


def _init(repo_root: Path, baseline_path: Path, dry_run: bool) -> int:
    """One-time baseline generation: mark = current line count quantized up
    to a multiple of ``MARK_QUANTUM``, for every over-cap tracked ``*.py``
    file."""
    existing = _read_baseline_or_report(baseline_path)
    if existing is None:
        return 1
    if existing and not dry_run:
        print(
            f"ERROR: baseline already exists at {baseline_path} with "
            f"{len(existing)} entries. --init is a one-time setup; use the "
            "default (lower-only) mode to maintain it, or delete the "
            "directory first to regenerate from scratch.",
            file=sys.stderr,
        )
        return 1
    marks = {path: _quantize_mark(lines) for path, lines in _scan_over_cap(repo_root).items()}
    if dry_run:
        print(f"[dry-run] would write {len(marks)} entries to {baseline_path}")
        return 0
    _write_baseline(baseline_path, marks)
    print(f"wrote {len(marks)} high-water marks to {baseline_path}")
    return 0


def _lower_plan(
    baseline: dict[str, int], over: dict[str, int]
) -> tuple[dict[str, int], list[str]]:
    """Pure lower-only refresh plan, shared by ``_lower`` (which applies it)
    and ``_check`` (which verifies the checked-in baseline already equals it).

    Returns ``(updated_baseline, change_lines)``: each mark lowered to the
    live count quantized up to a multiple of ``MARK_QUANTUM`` -- only when
    that quantized value is strictly BELOW the recorded mark (never raise) --
    and each entry whose file fell under the cap or was deleted dropped.
    Never adds new entries; a new over-cap file requires an explicit reviewed
    baseline edit. ``change_lines`` is the human-readable plan in the
    script's ``-``/``~``/``!`` notation; ``!`` lines report growth the map
    leaves alone and do not alter ``updated_baseline``.
    """
    updated = dict(baseline)
    changes: list[str] = []
    for path, mark in baseline.items():
        current = over.get(path)
        if current is None:
            # File dropped below the cap or was deleted/untracked. Drop the
            # entry -- it is no longer covered. A future regrowth over the cap
            # is fail-closed (no entry -> implicit mark 0 -> violation).
            del updated[path]
            changes.append(f"  - {path}: dropped (no longer over cap)")
            continue
        target = _quantize_mark(current)
        if target < mark:
            updated[path] = target
            changes.append(f"  ~ {path}: {mark} -> {target} (lowered; live={current})")
        elif current > mark:
            # A growth past the mark is a ratchet violation the CI test catches.
            # This maintainer never raises, so leave the mark and report it.
            changes.append(
                f"  ! {path}: {mark} -> live={current} (GROWTH -- not raised; CI will "
                f"fail; a reviewed raise must use {target})"
            )
    return updated, changes


def _lower(repo_root: Path, baseline_path: Path, dry_run: bool) -> int:
    """Lower-only maintenance: lower each mark to the current line count
    quantized up to a multiple of ``MARK_QUANTUM`` -- and only when that
    quantized value is strictly BELOW the recorded mark (never raise). Drop
    entries for files that fell under the cap or were deleted. Never adds new
    entries -- a new over-cap file requires an explicit reviewed baseline
    edit."""
    baseline = _read_baseline_or_report(baseline_path)
    if baseline is None:
        return 1
    if not baseline:
        print(
            f"ERROR: no baseline at {baseline_path}. Run with --init first to "
            "create it from the live tree.",
            file=sys.stderr,
        )
        return 1
    updated, changes = _lower_plan(baseline, _scan_over_cap(repo_root))
    if not changes:
        print(f"no changes; baseline at {baseline_path} is current")
        return 0
    if dry_run:
        print(f"[dry-run] would apply {len(changes)} change(s):")
        for c in changes:
            print(c)
        return 0
    _write_baseline(baseline_path, updated)
    print(f"applied {len(changes)} change(s) to {baseline_path}:")
    for c in changes:
        print(c)
    return 0


def _check(repo_root: Path, baseline_path: Path) -> int:
    """Fixed-point gate (issue #1675): recompute the lower-only refresh in
    memory and exit non-zero -- listing the pending changes -- when the
    checked-in baseline is not already a fixed point. Writes nothing.

    "Fixed point" is judged on the baseline CONTENT the refresh would
    produce: pending lowerings (``~``) and dead-entry drops (``-``) fail the
    check. Growth past a mark (``!``) does not change the refresh's output --
    the lower-only map leaves that mark alone by design -- so a pure-growth
    state still exits 0 here; that violation is the ratchet keystone's
    (``test_over_cap_files_do_not_exceed_high_water_mark`` in
    tests/test_file_size_ratchet.py) to report, not this gate's.
    """
    baseline = _read_baseline_or_report(baseline_path)
    if baseline is None:
        return 1
    if not baseline:
        print(
            f"ERROR: no baseline at {baseline_path}. Run with --init first to "
            "create it from the live tree.",
            file=sys.stderr,
        )
        return 1
    updated, changes = _lower_plan(baseline, _scan_over_cap(repo_root))
    if updated == baseline:
        print(f"no pending lowerings; baseline at {baseline_path} is a fixed point")
        for c in changes:
            # Only '!' growth notices can be present here -- informational;
            # the ratchet keystone owns that failure, not this gate.
            print(c)
        return 0
    print(
        f"baseline at {baseline_path} is not a fixed point of the lower-only "
        "refresh; run `python scripts/refresh_file_size_ratchet.py` and commit "
        "the updated file_size_ratchet_baseline/ entries in the same PR to apply:"
    )
    for c in changes:
        print(c)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true", help="one-time baseline generation")
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "fixed-point gate: exit non-zero (listing pending changes) when the "
            "checked-in baseline is not already a fixed point of the lower-only "
            "refresh; writes nothing (issue #1675)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="print changes without writing")
    args = parser.parse_args()

    repo_root = _repo_root()
    baseline_path = repo_root / _BASELINE_NAME
    if args.init:
        return _init(repo_root, baseline_path, args.dry_run)
    if args.check:
        if args.dry_run:
            parser.error("--check never writes; --dry-run is redundant")
        return _check(repo_root, baseline_path)
    return _lower(repo_root, baseline_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
