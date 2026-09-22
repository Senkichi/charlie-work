"""runtime_paths layout: repo-relative paths and phantom-state-dir warning predicates.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
from charlie_work.paths import find_repo_root, runtime_paths


def test_runtime_paths_are_repo_relative(tmp_path: Path) -> None:
    paths = runtime_paths(tmp_path, ".var/charlie-work")

    assert paths.root == tmp_path / ".var" / "charlie-work"
    assert paths.state_file == paths.root / "state.json"


def test_runtime_paths_warns_on_phantom_state_dir(tmp_path: Path, caplog: Any) -> None:
    """Issue #648: a state dir that exists with sibling artifacts but no
    state.json is a phantom signal — runtime_paths must warn (non-blocking)
    so the operator notices instead of seeing a silent 'all clear'."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    # Mimic the stray artifacts described in the issue.
    (state_dir / "events.db").write_bytes(b"")
    (state_dir / "state.json.lock").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert str(state_dir.resolve()) in warnings[0].message
    assert "state.json" in warnings[0].message


def test_runtime_paths_check_phantom_false_suppresses_warning(tmp_path: Path, caplog: Any) -> None:
    """Issue #1754: the supervisor's own bookkeeping root (orchestrator_root())
    legitimately has events.db but never a state.json -- the per-repo loop
    never runs against it. ``check_phantom=False`` must suppress the warning
    under the exact same events.db-but-no-state.json fixture that proves the
    warning fires by default."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (state_dir / "events.db").write_bytes(b"")
    (state_dir / "state.json.lock").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work", check_phantom=False)

    assert not caplog.records


def test_supervisor_runtime_paths_suppresses_phantom_warning(
    tmp_path: Path, caplog: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #1754: ``supervisor_runtime_paths`` is the single binding of
    ``orchestrator_root()`` + ``check_phantom=False`` that every supervisor
    bookkeeping call site goes through.  Against a tmp orchestrator root
    whose state dir holds events.db but no state.json — the exact fixture
    that proves the warning fires by default — the helper must emit no
    phantom-state-dir warning.  Dropping the opt-out inside the helper (or
    reverting a call site to a bare ``runtime_paths(orchestrator_root(), ...)``)
    reintroduces the warning and fails this test."""
    from charlie_work import supervise

    monkeypatch.setattr(supervise, "_ORCHESTRATOR_ROOT", tmp_path)
    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (state_dir / "events.db").write_bytes(b"")
    (state_dir / "state.json.lock").write_text("", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        paths = supervise.supervisor_runtime_paths(".var/charlie-work")

    assert paths.root == state_dir.resolve()
    phantom_warnings = [r for r in caplog.records if "sibling artifacts" in r.message]
    assert not phantom_warnings


def test_runtime_paths_silent_without_sibling_artifacts(tmp_path: Path, caplog: Any) -> None:
    """Issue #648 review MINOR: a state dir that exists but has no state.json
    AND no sibling artifacts (events.db, state.json.lock) must NOT warn — it
    could be a pre-existing directory used for an unrelated purpose, not a
    phantom left by a misresolved invocation."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    # No sibling artifacts — just an empty dir.

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    assert not caplog.records


def test_runtime_paths_no_warn_for_absolute_unrelated_state_dir(
    tmp_path: Path, caplog: Any
) -> None:
    """Issue #648 review MINOR: an absolute state_dir pointing at a
    pre-existing directory without sibling artifacts must not trigger the
    phantom warning."""
    from charlie_work.paths import runtime_paths

    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "some-file.txt").write_text("data", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, str(unrelated))

    assert not caplog.records


def test_runtime_paths_silent_when_state_dir_absent(tmp_path: Path, caplog: Any) -> None:
    """A genuine first run has not created the state dir yet at runtime_paths
    call time — no phantom warning must fire."""
    from charlie_work.paths import runtime_paths

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    assert not caplog.records


def test_runtime_paths_silent_when_state_json_exists(tmp_path: Path, caplog: Any) -> None:
    """A populated state dir is the normal steady state — no warning."""
    from charlie_work.paths import runtime_paths

    state_dir = tmp_path / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (state_dir / "state.json").write_text("{}", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="charlie_work.paths"):
        runtime_paths(tmp_path, ".var/charlie-work")

    assert not caplog.records


def test_find_repo_root_explicit_marker_git_dir_returns_self(tmp_path: Path) -> None:
    """An explicit ``--repo`` naming a directory that contains a ``.git``
    *directory* must be honored as the root even when the ``.git`` dir is
    not a readable gitdir. Git discovery treats an empty ``.git`` as
    non-repository and walks past it into an enclosing repo — under a
    basetemp nested inside a checkout (worker-sandbox TMPDIR) that
    silently retargets ``--repo`` at the *enclosing* repo's state, the
    same misroute class issues #648/#895 close down. Test-fake repos
    (``_cli_fixtures._make_repo``) rely on the marker form — this test
    exercises exactly that shape, and in a repo-nested basetemp it
    distinguishes the fix from the absorption the git-discovery path
    used to produce."""
    marker_repo = tmp_path / "fake-repo"
    (marker_repo / ".git").mkdir(parents=True)

    assert find_repo_root(marker_repo, explicit=True) == marker_repo.resolve()
