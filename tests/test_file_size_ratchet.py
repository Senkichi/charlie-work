"""File-size high-water-mark ratchet (issue #1442).

## What this enforces

``workflow.py`` regrew +1,332 lines in 4 days (23,585 -> 24,917) because
salvage/rework PRs land new code into the monolith by default, and the
existing file-size discipline grandfathers it: the 800-line-per-module cap is
recorded as cap-exemption *bands* for extracted modules
(``tests/test_stalled_review_reap_split.py``'s [1308, 1391] band), but nothing
binds the monolith itself. Extraction (#1317) cannot converge while regrowth
outpaces it.

This ratchet closes that gap. For every tracked ``*.py`` file over the
800-line cap, a high-water mark is recorded in the checked-in baseline file
``file_size_ratchet_baseline.json`` (repo root). CI fails any PR whose tree
leaves an over-cap file with MORE physical lines than its recorded mark. The
mark may only be lowered -- never raised except via an explicit reviewed edit
to the baseline file.

## Derivation, not enumeration (issue #1375)

The covered file set is derived from the baseline file's own keys UNION a live
scan of tracked ``*.py`` files over the cap (``git ls-files`` + line count),
never a hardcoded list. This is the derive-what-is-covered / fail-closed
direction:

* A file over the cap that is NOT in the baseline is compared against an
  implicit mark of 0 -- any non-empty over-cap file with no baseline entry
  fails. A new over-cap file therefore requires an explicit, reviewed baseline
  edit (the only sanctioned way to "raise" a mark from 0). This is the same
  fail-closed shape ``_ratchet_violations`` in
  ``tests/test_write_gate_enforcement.py`` uses for a brand-new module.
* A file in the baseline that has since shrunk below the cap (or been deleted)
  is auto-dropped from the baseline -- a future regrowth over the cap is
  fail-closed again (no entry -> implicit 0).

## Counting

Physical lines of the blob at the PR head (``len(text.splitlines())``), not
diff arithmetic. A byte-identical extraction like #1317's passes trivially:
the source file shrinks (fewer lines -> under its mark -> green, and the mark
auto-lowers to tighten), and the extracted module is a separate path whose own
mark is established by the reviewed baseline edit that lands it. Rename/move
does not inflate the count because the blob is read at its current path.

## Auto-lowering

When a PR shrinks an over-cap file below its recorded mark, the keystone test
PASSES (green on shrink, per AC2) and auto-lowers the mark in the baseline
file to the new line count (atomic temp-file + ``replace`` write, per the
project's JSON-write invariant). This is the "auto-lowered when a PR shrinks
the file" behavior issue #1442 specifies -- the ratchet tightens on every
shrink so extraction converges rather than racing a stale high mark. In CI the
write is discarded (fresh checkout); locally the developer commits the
auto-lowered baseline as part of the PR. ``scripts/refresh_file_size_ratchet.py``
is the manual maintainer (one-time ``--init`` and lower-only refresh) for the
same operation outside the test.

## Failure message

The failure output names the extracted-module alternative -- the facade
re-export pattern and the domain modules -- so a worker gets an actionable
redirect, not just a red X. See ``_EXTRACTION_REMEDY`` below.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BASELINE_PATH = _REPO_ROOT / "file_size_ratchet_baseline.json"

# The repo's normal per-module line cap. Over-cap files (lines > CAP) are the
# ratchet's covered set. Same value the extraction lineage records cap-exemption
# bands against (tests/test_stalled_review_reap_split.py).
FILE_SIZE_CAP = 800

# The facade re-export pattern a worker must follow instead of growing an
# over-cap monolith. Names the domain modules the #1283 Phase-A extraction
# lineage already produced, so the redirect is concrete and actionable.
_EXTRACTION_REMEDY = (
    "New code must not land in an over-cap monolith. Extract it into a domain "
    "module under src/charlie_work/ and re-export it through workflow.py's "
    "facade import block -- the pattern the #1283 Phase-A extraction lineage "
    "established (see the .dispatch_selection / .escalation / .verdict_parsing "
    "/ .rework_prompts / .ci_findings / .backlog_reachability / "
    ".stalled_review_reap `from .X import (...)  # noqa: F401 (deliberate "
    "re-export)` blocks at the top of src/charlie_work/workflow.py). A "
    "byte-identical extraction shrinks the source file and passes this ratchet "
    "trivially. To record a deliberate, reviewed exception (e.g. a new "
    "extraction that is itself over the cap), edit "
    "file_size_ratchet_baseline.json directly as a reviewed change in the PR."
)


def _tracked_py_line_counts(repo_root: Path) -> dict[str, int]:
    """Live-scan every ``*.py`` file tracked by git under ``repo_root`` and
    return ``{posix_rel_path: physical_line_count}``.

    Uses ``git ls-files`` so untracked scratch files (not in any PR diff, not
    subject to the cap) are excluded -- the issue's "tracked *.py" scope.
    Physical line count is ``len(text.splitlines())`` (a trailing newline does
    not add a phantom line), matching the blob at the PR head, not diff
    arithmetic.
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
    counts: dict[str, int] = {}
    for line in out.stdout.splitlines():
        if not line.endswith(".py"):
            continue
        path = repo_root / line
        if not path.exists():
            # Deleted between index and worktree -- treat as 0 lines (covered
            # set union drops it via auto-clean). Should not happen in a clean
            # checkout, but stay robust.
            counts[line] = 0
            continue
        counts[line] = len(path.read_text(encoding="utf-8").splitlines())
    return counts


def _file_size_violations(
    actual_counts: dict[str, int], baseline: dict[str, int], cap: int
) -> list[tuple[str, int, int]]:
    """Pure ratchet check, factored out so the mechanics (hold/shrink pass,
    increase fails, brand-new-over-cap fails closed) are unit-testable with
    synthetic data independent of the real tree.

    ``actual_counts`` maps posix rel path -> current physical line count.
    ``baseline`` maps posix rel path -> recorded high-water mark. The covered
    set is ``baseline.keys() | {p for p, n in actual_counts if n > cap}`` --
    derived, never hardcoded. For each covered path:

    * ``actual > mark`` -> violation (growth past the high-water mark).
    * ``actual <= mark`` -> holds (hold or shrink); no violation.

    A covered path absent from ``baseline`` is compared against an implicit
    mark of 0, so a brand-new over-cap file (in ``actual_counts`` over the cap
    but not in ``baseline``) always violates -- fail-closed. Returns the sorted
    list of ``(path, mark, actual)`` violations, empty when the ratchet holds.
    """
    covered = set(baseline) | {p for p, n in actual_counts.items() if n > cap}
    violations: list[tuple[str, int, int]] = []
    for path in covered:
        actual = actual_counts.get(path, 0)
        mark = baseline.get(path, 0)
        if actual > mark:
            violations.append((path, mark, actual))
    return sorted(violations)


def _auto_lower(
    actual_counts: dict[str, int], baseline: dict[str, int], cap: int
) -> dict[str, int]:
    """Compute the auto-lowered baseline: for each covered path, if the file
    shrank below its mark, lower the mark to the new count; if it dropped to
    or below the cap (or was deleted), drop the entry entirely. Never raises.
    Returns the updated baseline dict (a new dict; ``baseline`` is not
    mutated)."""
    covered = set(baseline) | {p for p, n in actual_counts.items() if n > cap}
    updated = dict(baseline)
    for path in covered:
        actual = actual_counts.get(path, 0)
        mark = baseline.get(path, 0)
        if actual < mark:
            if actual <= cap:
                updated.pop(path, None)
            else:
                updated[path] = actual
    return updated


def _write_baseline_atomic(path: Path, marks: dict[str, int]) -> None:
    """Atomic write (temp-file + ``replace``) per the project's JSON-write
    invariant (CLAUDE.md). Sorted keys for stable diffs."""
    payload = json.dumps(dict(sorted(marks.items())), indent=2, ensure_ascii=False)
    payload += "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Pure-function mechanics (synthetic data, independent of the real tree).
# Mirrors the _ratchet_violations unit tests in test_write_gate_enforcement.py.
# ---------------------------------------------------------------------------


def test_ratchet_passes_when_count_holds_or_shrinks() -> None:
    """Hold and shrink both pass -- the ratchet only fails on growth."""
    baseline = {"src/charlie_work/workflow.py": 25442, "src/charlie_work/reconcile.py": 2952}
    holds = dict(baseline)
    shrinks = {"src/charlie_work/workflow.py": 25441, "src/charlie_work/reconcile.py": 2000}

    assert _file_size_violations(holds, baseline, FILE_SIZE_CAP) == []
    assert _file_size_violations(shrinks, baseline, FILE_SIZE_CAP) == []


def test_ratchet_fails_when_count_increases() -> None:
    """An increase in ANY covered file fails, naming that file (and only that
    file -- an unrelated file holding steady must not be swept in)."""
    baseline = {"src/charlie_work/workflow.py": 25442, "src/charlie_work/reconcile.py": 2952}
    one_regressed = {"src/charlie_work/workflow.py": 25443, "src/charlie_work/reconcile.py": 2952}

    assert _file_size_violations(one_regressed, baseline, FILE_SIZE_CAP) == [
        ("src/charlie_work/workflow.py", 25442, 25443)
    ]


def test_brand_new_over_cap_file_with_no_baseline_fails_closed() -> None:
    """A file over the cap with no baseline entry is compared against an
    implicit mark of 0 -- fail-closed (issue #1375). A new over-cap file must
    be added to the baseline via an explicit reviewed edit, not silently
    pass because it was never enumerated."""
    baseline: dict[str, int] = {}
    actual = {"src/charlie_work/brand_new_module.py": 900}

    assert _file_size_violations(actual, baseline, FILE_SIZE_CAP) == [
        ("src/charlie_work/brand_new_module.py", 0, 900)
    ]


def test_under_cap_file_not_in_covered_set() -> None:
    """A file at or below the cap that is not in the baseline is outside the
    covered set -- the ratchet does not bind files that are under the cap and
    were never over it. (A file in the baseline that shrank under the cap is
    auto-dropped by the keystone, not failed.)"""
    baseline = {"src/charlie_work/workflow.py": 25442}
    actual = {"src/charlie_work/small_module.py": 500}

    assert _file_size_violations(actual, baseline, FILE_SIZE_CAP) == []


def test_synthetic_plus_one_line_to_at_mark_file_trips_check() -> None:
    """AC3 mutation check: a synthetic +1 line to an at-mark file trips the
    ratchet (assert the failure, not just the wiring). The at-mark file holds
    at its mark; adding one line crosses it."""
    at_mark = 1000
    baseline = {"src/charlie_work/at_mark_file.py": at_mark}

    # At the mark: holds, no violation.
    assert (
        _file_size_violations(
            {"src/charlie_work/at_mark_file.py": at_mark}, baseline, FILE_SIZE_CAP
        )
        == []
    )
    # +1 line past the mark: trips.
    assert _file_size_violations(
        {"src/charlie_work/at_mark_file.py": at_mark + 1}, baseline, FILE_SIZE_CAP
    ) == [("src/charlie_work/at_mark_file.py", at_mark, at_mark + 1)]
    # -1 line below the mark: shrinks, no violation.
    assert (
        _file_size_violations(
            {"src/charlie_work/at_mark_file.py": at_mark - 1}, baseline, FILE_SIZE_CAP
        )
        == []
    )


def test_auto_lower_never_raises_and_drops_under_cap_entries() -> None:
    """The auto-lower helper only lowers (never raises) and drops entries for
    files that fell to or below the cap."""
    baseline = {
        "src/charlie_work/workflow.py": 25442,  # shrinks -> lower
        "src/charlie_work/reconcile.py": 2952,  # holds -> unchanged
        "src/charlie_work/small.py": 850,  # shrinks below cap -> drop
    }
    actual = {
        "src/charlie_work/workflow.py": 25400,
        "src/charlie_work/reconcile.py": 2952,
        "src/charlie_work/small.py": 790,
    }
    lowered = _auto_lower(actual, baseline, FILE_SIZE_CAP)

    assert lowered == {
        "src/charlie_work/workflow.py": 25400,
        "src/charlie_work/reconcile.py": 2952,
        # small.py dropped (fell under the cap).
    }
    # A growth is NOT silently raised by the auto-lowerer.
    grown_actual = {"src/charlie_work/workflow.py": 25500}
    assert (
        _auto_lower(grown_actual, baseline, FILE_SIZE_CAP)["src/charlie_work/workflow.py"] == 25442
    )


# ---------------------------------------------------------------------------
# Real-tree keystone: the CI gate itself.
# ---------------------------------------------------------------------------


def _load_baseline() -> dict[str, int]:
    if not _BASELINE_PATH.exists():
        pytest.fail(
            f"file_size_ratchet_baseline.json not found at {_BASELINE_PATH}. "
            "Run `python scripts/refresh_file_size_ratchet.py --init` to create it."
        )
    data = json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))
    return {k: int(v) for k, v in data.items()}


def test_baseline_includes_the_two_named_monoliths() -> None:
    """AC1: the baseline is derived (not hand-listed) but the two files the
    issue names -- workflow.py and test_charlie_work.py -- must be present.
    They are the dominant monoliths the ratchet exists to bind; their absence
    would mean the derivation is broken or the baseline was hand-pruned."""
    baseline = _load_baseline()
    assert "src/charlie_work/workflow.py" in baseline, (
        "workflow.py (the monolith issue #1442 is about) is missing from the "
        "baseline -- the derivation is broken or the file was hand-pruned"
    )
    assert "tests/test_charlie_work.py" in baseline, (
        "test_charlie_work.py (the test monolith issue #1442 names) is missing "
        "from the baseline -- the derivation is broken or the file was hand-pruned"
    )


def test_baseline_marks_are_at_or_above_the_live_tree() -> None:
    """Symmetric-direction guard: every baseline entry's mark must be >= the
    file's current live line count. A mark BELOW the live count is a stale-low
    baseline that would false-trip the keystone (the keystone would report a
    violation for a file that has not actually grown past its real high-water).
    This catches a hand-edited baseline that lowered a mark without the file
    actually shrinking."""
    baseline = _load_baseline()
    live = _tracked_py_line_counts(_REPO_ROOT)
    stale_low = [
        (path, mark, live.get(path, 0))
        for path, mark in baseline.items()
        if live.get(path, 0) > mark
    ]
    assert not stale_low, (
        "file_size_ratchet_baseline.json has mark(s) BELOW the live line count "
        "-- a stale-low baseline that false-trips the ratchet. Either the file "
        "grew (run the orchestrator's review lane) or the mark was hand-lowered "
        "without the file shrinking:\n"
        + "\n".join(f"  {p}: mark={m}, live={n}" for p, m, n in stale_low)
    )


def test_over_cap_files_do_not_exceed_high_water_mark() -> None:
    """The keystone CI gate (issue #1442 AC2). Loads the baseline, live-scans
    every tracked ``*.py`` file, and fails if any covered file grew past its
    recorded high-water mark. On shrink, auto-lowers the baseline (tightening
    the ratchet) and passes -- green on shrink, red on growth.

    The covered set is derived (baseline keys UNION live over-cap files), never
    hardcoded. The failure message names the facade/extraction remedy so a
    worker gets an actionable redirect.
    """
    baseline = _load_baseline()
    live = _tracked_py_line_counts(_REPO_ROOT)
    violations = _file_size_violations(live, baseline, FILE_SIZE_CAP)

    if violations:
        lines = "\n".join(
            f"  {path}: mark={mark}, now={actual} (+{actual - mark})"
            for path, mark, actual in violations
        )
        pytest.fail(
            "file-size high-water-mark ratchet violated (issue #1442): the "
            "following over-cap file(s) grew past their recorded mark, which "
            "the ratchet never allows:\n"
            f"{lines}\n\n"
            f"{_EXTRACTION_REMEDY}"
        )

    # Green on growth-fail path is unreachable here; on shrink, auto-lower the
    # baseline so the ratchet tightens toward convergence (issue #1442's goal:
    # extraction can converge only if shrinks bind). Atomic write per CLAUDE.md.
    lowered = _auto_lower(live, baseline, FILE_SIZE_CAP)
    if lowered != baseline:
        _write_baseline_atomic(_BASELINE_PATH, lowered)


def test_synthetic_plus_one_to_real_workflow_py_trips_check() -> None:
    """AC3 mutation check tied to the real monolith: take workflow.py's real
    live line count and recorded mark, and assert that +1 line past the
    at-mark position trips the ratchet while the current count holds. This
    exercises the real baseline + real file, not just synthetic data.

    Robust to the baseline being exactly at-mark (mark == live, the state the
    keystone's auto-lower maintains) or stale-high (mark > live): the +1 is
    taken past ``max(live, mark)`` so it always crosses the mark.
    """
    baseline = _load_baseline()
    path = "src/charlie_work/workflow.py"
    assert path in baseline, "workflow.py must be in the baseline (AC1)"
    live = _tracked_py_line_counts(_REPO_ROOT).get(path, 0)
    mark = baseline[path]

    # Current count holds (no violation) -- the baseline is not stale-low.
    assert _file_size_violations({path: live}, {path: mark}, FILE_SIZE_CAP) == [], (
        f"workflow.py live count {live} exceeds its mark {mark} -- the keystone "
        "should already have caught this; the baseline is stale-low"
    )
    # +1 past the at-mark position trips.
    at_mark = max(live, mark)
    assert _file_size_violations({path: at_mark + 1}, {path: mark}, FILE_SIZE_CAP) == [
        (path, mark, at_mark + 1)
    ], "a +1 line past workflow.py's at-mark position did not trip the ratchet"
