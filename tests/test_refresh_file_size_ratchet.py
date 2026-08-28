"""Tests for scripts/refresh_file_size_ratchet.py -- the SOLE writer of
``file_size_ratchet_baseline.json``.

The write-side behavior used to live inside ``tests/test_file_size_ratchet.py``
as an auto-lower-on-shrink side effect of the keystone test; that side effect
put an exact-count baseline diff into nearly every monolith-touching PR and
made the baseline the fleet's dominant merge-conflict source. The keystone is
now a pure assertion (guarded by ``test_keystone_never_writes_the_baseline``),
and the write-side properties -- never raises a mark, drops under-cap entries,
quantizes every written mark to a multiple of ``MARK_QUANTUM`` -- are asserted
here against the script, which owns them exclusively.

Loaded via ``tests/_script_loader.load_script_module`` (no sys.path
pollution). Git-dependent scanning (``_scan_over_cap``) is monkeypatched with
synthetic counts so these tests exercise the mark-derivation logic, not git.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _script_loader import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "refresh_file_size_ratchet.py"


@pytest.fixture()
def refresh_mod():
    return load_script_module(_SCRIPT_PATH, "refresh_file_size_ratchet_under_test")


def _write_json(path: Path, data: dict[str, int]) -> str:
    payload = json.dumps(dict(sorted(data.items())), indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return payload


# ---------------------------------------------------------------------------
# Quantization rule
# ---------------------------------------------------------------------------


def test_quantize_mark_rounds_up_and_preserves_exact_multiples(refresh_mod) -> None:
    q = refresh_mod.MARK_QUANTUM
    assert refresh_mod._quantize_mark(q) == q  # exact multiple preserved
    assert refresh_mod._quantize_mark(q + 1) == 2 * q  # rounds up, never down
    assert refresh_mod._quantize_mark(2 * q - 1) == 2 * q
    assert refresh_mod._quantize_mark(26400) == 26400
    assert refresh_mod._quantize_mark(26401) == 26600
    # A quantized mark is always >= the live count (stale-low is impossible
    # for a script-written mark -- the guard in test_file_size_ratchet.py
    # relies on this direction).
    for lines in (801, 999, 1000, 1001, 54082):
        assert refresh_mod._quantize_mark(lines) >= lines


def test_mark_quantum_matches_the_shared_test_constant(refresh_mod) -> None:
    """Drift guard: MARK_QUANTUM is declared in the script (the writer) and in
    tests/_ratchet_constants.py (imported by tests/test_file_size_ratchet.py
    for the remedy text -- test modules may not import each other, so the
    shared value lives in an underscore module). Deterministic convergence --
    concurrent PRs producing byte-identical baseline lines -- only holds if
    every writer uses the same quantum."""
    from _ratchet_constants import MARK_QUANTUM

    assert refresh_mod.MARK_QUANTUM == MARK_QUANTUM


# ---------------------------------------------------------------------------
# _init: one-time generation writes quantized marks
# ---------------------------------------------------------------------------


def test_init_writes_quantized_marks(refresh_mod, monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        refresh_mod,
        "_scan_over_cap",
        lambda repo_root: {"src/charlie_work/workflow.py": 26490, "src/big.py": 801},
    )
    baseline_path = tmp_path / "file_size_ratchet_baseline.json"

    rc = refresh_mod._init(tmp_path, baseline_path, dry_run=False)

    assert rc == 0
    written = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert written == {"src/charlie_work/workflow.py": 26600, "src/big.py": 1000}
    # Atomic-write invariant: no temp file left behind.
    assert not baseline_path.with_suffix(".json.tmp").exists()


def test_init_refuses_when_baseline_exists(refresh_mod, monkeypatch, tmp_path) -> None:
    baseline_path = tmp_path / "file_size_ratchet_baseline.json"
    original = _write_json(baseline_path, {"src/big.py": 1000})
    monkeypatch.setattr(refresh_mod, "_scan_over_cap", lambda repo_root: {"src/big.py": 900})

    rc = refresh_mod._init(tmp_path, baseline_path, dry_run=False)

    assert rc == 1
    assert baseline_path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# _lower: quantized, lower-only, drops under-cap, never raises
# ---------------------------------------------------------------------------


def test_lower_quantizes_and_only_lowers(refresh_mod, monkeypatch, tmp_path) -> None:
    baseline_path = tmp_path / "file_size_ratchet_baseline.json"
    _write_json(
        baseline_path,
        {
            "src/shrunk_cross_bucket.py": 2000,  # live 1500 -> lower to 1600
            "src/shrunk_same_bucket.py": 1600,  # live 1450 -> quantized 1600 == mark, hold
            "src/under_cap.py": 1000,  # live 700 (under cap) -> dropped
            "src/grown.py": 1000,  # live 1100 -> NOT raised
            "src/legacy_exact_mark.py": 1234,  # pre-quantization mark; live 900 -> 1000
        },
    )
    monkeypatch.setattr(
        refresh_mod,
        "_scan_over_cap",
        lambda repo_root: {
            "src/shrunk_cross_bucket.py": 1500,
            "src/shrunk_same_bucket.py": 1450,
            # under_cap.py absent: no longer over the cap.
            "src/grown.py": 1100,
            "src/legacy_exact_mark.py": 900,
        },
    )

    rc = refresh_mod._lower(tmp_path, baseline_path, dry_run=False)

    assert rc == 0
    written = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert written == {
        "src/shrunk_cross_bucket.py": 1600,
        "src/shrunk_same_bucket.py": 1600,
        "src/grown.py": 1000,  # never raised, even on growth
        "src/legacy_exact_mark.py": 1000,
    }
    assert "src/under_cap.py" not in written


def test_lower_reports_no_changes_when_current(refresh_mod, monkeypatch, tmp_path, capsys) -> None:
    """Same-bucket shrink and hold produce NO baseline change -- the property
    that keeps routine monolith-touching PRs free of baseline diffs."""
    baseline_path = tmp_path / "file_size_ratchet_baseline.json"
    original = _write_json(baseline_path, {"src/a.py": 1600, "src/b.py": 1000})
    monkeypatch.setattr(
        refresh_mod,
        "_scan_over_cap",
        lambda repo_root: {"src/a.py": 1401, "src/b.py": 1000},
    )

    rc = refresh_mod._lower(tmp_path, baseline_path, dry_run=False)

    assert rc == 0
    assert baseline_path.read_text(encoding="utf-8") == original
    assert "no changes" in capsys.readouterr().out


def test_lower_dry_run_does_not_write(refresh_mod, monkeypatch, tmp_path, capsys) -> None:
    baseline_path = tmp_path / "file_size_ratchet_baseline.json"
    original = _write_json(baseline_path, {"src/a.py": 2000})
    monkeypatch.setattr(refresh_mod, "_scan_over_cap", lambda repo_root: {"src/a.py": 1500})

    rc = refresh_mod._lower(tmp_path, baseline_path, dry_run=True)

    assert rc == 0
    assert baseline_path.read_text(encoding="utf-8") == original
    assert "[dry-run]" in capsys.readouterr().out
