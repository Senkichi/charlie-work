"""Maintain the file-size high-water-mark ratchet baseline (issue #1442).

The ratchet itself is enforced by ``tests/test_file_size_ratchet.py``, which
runs in CI on every push and fails any PR that leaves an over-cap tracked
``*.py`` file with MORE physical lines than its recorded high-water mark. This
script is the SOLE writer of the checked-in baseline file
(``file_size_ratchet_baseline.json`` at the repo root): it can create the
baseline from the live tree (one-time ``--init``) and lower existing marks
after a shrink (the default). The test suite never writes the baseline -- a
pytest run must leave the tree clean, so a PR only ever changes the baseline
as a deliberate, reviewed edit. This script NEVER raises a mark -- raising
requires an explicit, reviewed edit to the baseline file, by design (#1442).

## Quantized marks

Marks are quantized to multiples of ``MARK_QUANTUM`` (200): every mark this
script writes is ``ceil(lines / 200) * 200``. Two reasons:

* **Merge-conflict damping.** Exact-count marks made the baseline the repo's
  hottest conflict site: any two concurrent PRs changing a monolith's line
  count wrote different values on the same JSON line. With quantized marks,
  growth within a bucket needs no baseline edit at all, and two PRs bumping
  the same file into the same bucket write the identical line (clean merge).
* **Deterministic convergence.** Every writer (this script, and a PR raising
  a mark by hand) uses the same rule -- next multiple of 200 -- so
  independent edits agree byte-for-byte.

A hand-raise in a growth PR must follow the same rule: raise to the next
multiple of 200, never to the exact line count. If a baseline line still
conflicts on merge, take the larger value.

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

The covered file set is derived from a live scan of tracked ``*.py`` files
(``git ls-files``) plus the baseline's own keys -- never a hardcoded list
(issue #1375: derive-what-is-covered fails closed). The size cap is the repo's
normal per-module cap of 800 lines.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# The repo's normal per-module line cap. Over-cap files (lines > CAP) are the
# ratchet's covered set. This is the same cap the extraction lineage (#1283
# Phase A) records cap-exemption bands against (e.g. stalled_review_reap.py's
# [1308, 1391] band in tests/test_stalled_review_reap_split.py).
FILE_SIZE_CAP = 800

# Marks are recorded as multiples of this quantum (rounded UP from the live
# line count). Growth within a bucket needs no baseline edit; concurrent PRs
# bumping a file into the same bucket write the identical value and merge
# cleanly. tests/test_file_size_ratchet.py declares the same constant; the
# script tests assert the two stay equal.
MARK_QUANTUM = 200

_BASELINE_NAME = "file_size_ratchet_baseline.json"


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
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in data.items()}


def _write_baseline(path: Path, marks: dict[str, int]) -> None:
    """Atomic write (temp-file + ``replace``) per the project's JSON-write
    invariant (CLAUDE.md). Sorted keys for stable diffs."""
    payload = json.dumps(dict(sorted(marks.items())), indent=2, ensure_ascii=False)
    payload += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _init(repo_root: Path, baseline_path: Path, dry_run: bool) -> int:
    """One-time baseline generation: mark = current line count quantized up
    to a multiple of ``MARK_QUANTUM``, for every over-cap tracked ``*.py``
    file."""
    existing = _load_baseline(baseline_path)
    if existing and not dry_run:
        print(
            f"ERROR: baseline already exists at {baseline_path} with "
            f"{len(existing)} entries. --init is a one-time setup; use the "
            "default (lower-only) mode to maintain it, or delete the file "
            "first to regenerate from scratch.",
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


def _lower(repo_root: Path, baseline_path: Path, dry_run: bool) -> int:
    """Lower-only maintenance: lower each mark to the current line count
    quantized up to a multiple of ``MARK_QUANTUM`` -- and only when that
    quantized value is strictly BELOW the recorded mark (never raise). Drop
    entries for files that fell under the cap or were deleted. Never adds new
    entries -- a new over-cap file requires an explicit reviewed baseline
    edit."""
    baseline = _load_baseline(baseline_path)
    if not baseline:
        print(
            f"ERROR: no baseline at {baseline_path}. Run with --init first to "
            "create it from the live tree.",
            file=sys.stderr,
        )
        return 1
    over = _scan_over_cap(repo_root)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="one-time baseline generation")
    parser.add_argument("--dry-run", action="store_true", help="print changes without writing")
    args = parser.parse_args()

    repo_root = _repo_root()
    baseline_path = repo_root / _BASELINE_NAME
    if args.init:
        return _init(repo_root, baseline_path, args.dry_run)
    return _lower(repo_root, baseline_path, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
