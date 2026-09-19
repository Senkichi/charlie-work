"""State-dir / worktrees-root layout checks for ``run_doctor``.

Covers the state-dir split-brain detector and the worktrees-root
agreement check. Split out of ``tests/test_doctor.py`` (issue #1563,
Track 1 shoulder) -- bodies are verbatim relocations; shared helpers
live in ``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path
from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    RuntimeConfig,
)
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _config,
)


def test_doctor_state_dir_split_brain_silent_without_override(tmp_path: Path) -> None:
    """No ``state_dir`` override -> the default tree *is* the configured tree,
    so there is nothing to diagnose and the check must not appear at all."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []


def test_doctor_state_dir_split_brain_silent_when_explicit_state_dir_matches_default(
    tmp_path: Path,
) -> None:
    """``state_dir: .var/charlie-work`` spelled explicitly (charlie-work's own
    ``orchestrator.config.yaml`` does exactly this) must compare equal to
    ``layout.default_state_root`` and stay silent — a spelling-only match, not
    a real override. Guards a ``.resolve()`` asymmetry: ``runtime_paths``
    resolves ``paths.root`` but ``default_state_root`` does not, so a config
    that merely repeats the default value must not misread as a divergence.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir=".var/charlie-work"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []
    by_name = {c.name: c for c in checks}
    assert by_name["worktrees root"].ok is True


def test_doctor_state_dir_split_brain_silent_when_overridden_with_no_residue(
    tmp_path: Path,
) -> None:
    """``state_dir`` overridden but the default tree is empty/absent -> silent.

    Guards against a false positive firing on every overridden repo regardless
    of whether the default tree actually holds anything.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []


def test_doctor_state_dir_split_brain_fires_on_residue(tmp_path: Path) -> None:
    """Empirical proof this detector fires on the live #712 shape: an
    overridden ``state_dir`` plus a default tree still holding ``state.json``,
    a non-empty ``worktrees/``, and a non-empty ``events.db`` — exactly what
    job-cannon's ``.var/charlie-work`` looks like today (74 uncollected
    worktrees; 0-byte ``events.db`` was the pre-#718 shape, so this test uses a
    non-empty one to prove the size>0 branch too).
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    default_root = tmp_path / ".var" / "charlie-work"
    (default_root / "worktrees" / "some-branch").mkdir(parents=True)
    (default_root / "state.json").write_text("{}", encoding="utf-8")
    (default_root / "events.db").write_bytes(b"\x00" * 32)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["state dir split-brain"]
    assert check.ok is False
    assert check.severity == "warning"
    assert str(default_root) in check.detail
    assert str(paths.root) in check.detail
    assert "state.json" in check.detail
    assert "worktrees" in check.detail
    assert "events.db" in check.detail


def test_doctor_state_dir_split_brain_fires_on_worktrees_only_production_shape(
    tmp_path: Path,
) -> None:
    """The *actual* job-cannon shape, not just the easier all-three-signals
    shape above: no ``state.json`` (never written there), a 0-byte
    ``events.db`` (schema never applied — the #718 shape), and only a
    non-empty ``worktrees/`` holding the 74 uncollected worktrees. The
    ``events.db`` size>0 branch must NOT be required for this to fire — the
    ``worktrees/`` clause alone has to be sufficient, since that is the only
    signal present in production today.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    default_root = tmp_path / ".var" / "charlie-work"
    (default_root / "worktrees" / "some-branch").mkdir(parents=True)
    (default_root / "worktrees" / "other-branch").mkdir(parents=True)
    (default_root / "events.db").write_bytes(b"")

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["state dir split-brain"]
    assert check.ok is False
    assert check.severity == "warning"
    assert "worktrees" in check.detail
    assert "2 entries" in check.detail
    assert "state.json" not in check.detail
    assert "events.db" not in check.detail


def test_doctor_state_dir_split_brain_ignores_zero_byte_events_db(tmp_path: Path) -> None:
    """A freshly-``sqlite3.connect``-ed, schema-never-applied ``events.db`` is
    0 bytes (the #718 shape) — it must not itself count as residue, since a
    0-byte file proves nothing was ever written there."""
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    default_root = tmp_path / ".var" / "charlie-work"
    default_root.mkdir(parents=True)
    (default_root / "events.db").write_bytes(b"")

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    assert [c for c in checks if c.name == "state dir split-brain"] == []


def test_doctor_worktrees_root_agrees_by_default(tmp_path: Path) -> None:
    """No override -> the reported root is the plain default; ``ok`` is
    unconditionally True (see the ``_check_worktrees_root_agreement``
    docstring for why a pass/fail comparison is no longer meaningful post-A2)."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["worktrees root"]
    assert check.ok is True
    assert check.severity == "error"  # default severity; ok=True never blocks
    assert str(paths.root / "worktrees") in check.detail


def test_doctor_worktrees_root_no_longer_diverges_when_state_dir_overridden(
    tmp_path: Path,
) -> None:
    """Regression proof for #712: before A2, an overridden ``runtime.state_dir``
    with no explicit ``claude_code.worktrees_dir`` made dispatch's fallback
    (the unconditional default tree) and ``worktree-clean``'s sweep (the
    configured tree) resolve to two different directories, and this check
    reported ``ok=False``. A2 unified both call sites behind
    ``resolved_layout(config, repo_root).worktrees``, so the same config now
    reports a single root (the configured tree) and ``ok=True`` -- the stale
    default-tree path must no longer appear anywhere in the detail.
    """
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    check = by_name["worktrees root"]
    assert check.ok is True
    assert check.severity == "error"  # default severity; ok=True never blocks
    assert str(paths.root / "worktrees") in check.detail
    stale_default_worktrees = tmp_path / ".var" / "charlie-work" / "worktrees"
    assert str(stale_default_worktrees) not in check.detail


def test_doctor_worktrees_root_agrees_when_explicit_worktrees_dir_matches_state_dir(
    tmp_path: Path,
) -> None:
    """An explicit ``claude_code.worktrees_dir`` (axis 2) still reports a
    single agreeing root -- unaffected by which of the two override axes is
    in play, since both resolve through the same ``resolved_layout`` call."""
    config = _config(
        runtime=RuntimeConfig(state_dir="custom-state"),
        claude_code=ClaudeCodeConfig(worktrees_dir="custom-state/worktrees"),
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {c.name: c for c in checks}
    assert by_name["worktrees root"].ok is True
