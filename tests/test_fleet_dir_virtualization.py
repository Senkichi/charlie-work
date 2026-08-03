"""Tests for the fleet-dir virtualization probe and its write-side warnings.

Issue #624: MSIX/container redirection makes the literal fleet-dir path string
identical in both the container and the host while naming different files. The
probe keys on the literal path disagreeing with its resolved form -- never on a
hardcoded package moniker. Per the issue's test guidance, the divergence is
injected by patching the resolution step, not by building a real MSIX redirect.
"""

from __future__ import annotations

import logging
import os
import pathlib
from pathlib import Path
from typing import Any

from charlie_work.fleet_paths import (
    detect_path_virtualization,
    warn_fleet_dir_virtualization_on_write,
)


def _patch_resolve_to_diverge(monkeypatch: Any, literal: Path, redirected: Path) -> None:
    """Make ``Path.resolve()`` return ``redirected`` for ``literal`` only."""
    real_resolve = pathlib.Path.resolve

    def fake_resolve(self, *args, **kwargs):
        result = real_resolve(self, *args, **kwargs)
        if os.path.normcase(os.fspath(result)) == os.path.normcase(os.fspath(literal)):
            return redirected
        return result

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)


# ---------------------------------------------------------------------------
# detection helper
# ---------------------------------------------------------------------------


def test_detect_path_virtualization_returns_none_when_equal(tmp_path: Path) -> None:
    literal = tmp_path / "fleet"
    literal.mkdir(parents=True, exist_ok=True)
    assert detect_path_virtualization(literal) is None


def test_detect_path_virtualization_returns_pair_when_diverged(
    tmp_path: Path, monkeypatch: Any
) -> None:
    literal = tmp_path / "fleet"
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, literal, redirected)

    diverged = detect_path_virtualization(literal)
    assert diverged is not None
    assert diverged[0] == literal
    assert diverged[1] == redirected


def test_detect_path_virtualization_treats_case_only_difference_as_equal(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A case-only spelling difference is the same file, not virtualization.

    ``PurePath.__eq__`` is case-sensitive even on Windows; the probe must use
    the OS's own normalization so ``C:\\Foo`` and ``c:\\foo`` do not falsely
    fire. On a case-sensitive filesystem this is a no-op equality check.
    """
    literal = tmp_path / "fleet"
    literal.mkdir(parents=True, exist_ok=True)
    upper = Path(os.path.normcase(os.fspath(literal)).upper())
    # Forge resolve() to return a case-only variant of the same path.
    real_resolve = pathlib.Path.resolve

    def fake_resolve(self, *args, **kwargs):
        result = real_resolve(self, *args, **kwargs)
        if os.path.normcase(os.fspath(result)) == os.path.normcase(os.fspath(literal)):
            return upper
        return result

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)

    assert detect_path_virtualization(literal) is None


# ---------------------------------------------------------------------------
# write-side warning helper
# ---------------------------------------------------------------------------


def test_warn_on_write_logs_when_diverged(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    literal = tmp_path / "fleet"
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, literal, redirected)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        warn_fleet_dir_virtualization_on_write(literal, context="writing test-state.json")

    assert any(
        "Fleet dir virtualization" in record.message
        and "writing test-state.json" in record.message
        for record in caplog.records
    )
    # Both paths and the issue reference must appear in the warning text.
    joined = " ".join(record.message for record in caplog.records)
    assert str(literal) in joined
    assert str(redirected) in joined
    assert "#624" in joined


def test_warn_on_write_is_silent_when_equal(tmp_path: Path, caplog: Any) -> None:
    literal = tmp_path / "fleet"
    literal.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        warn_fleet_dir_virtualization_on_write(literal, context="writing test-state.json")

    assert not any("Fleet dir virtualization" in record.message for record in caplog.records)


def test_warn_on_write_never_raises_on_resolve_error(tmp_path: Path, monkeypatch: Any) -> None:
    """A resolve() failure must not crash the writer (probe is best-effort)."""
    literal = tmp_path / "fleet"

    def raising_resolve(self, *args, **kwargs):
        raise OSError("simulated unresolvable path")

    monkeypatch.setattr(pathlib.Path, "resolve", raising_resolve)

    # Must not raise.
    warn_fleet_dir_virtualization_on_write(literal, context="writing test-state.json")


# ---------------------------------------------------------------------------
# write-side integration: save_idle_streaks (the #590 "I deployed it" case)
# ---------------------------------------------------------------------------


def test_save_idle_streaks_warns_when_fleet_dir_is_virtualized(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """The runner-allocation.json write must warn when it lands in a copy.

    This is the exact #590 shape: ``charlie runners allocate`` reports success
    while the file the fleet supervisor reads never updates.
    """
    from charlie_work.runner_slots import save_idle_streaks

    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet, redirected)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        save_idle_streaks(
            fleet, {"owner/repo": 1}, source="prologue", full_pass_interval_seconds=300
        )

    assert any("runner-allocation.json" in record.message for record in caplog.records)
    # The write still lands (the warning never blocks it).
    assert (fleet / "runner-allocation.json").exists()


def test_save_idle_streaks_is_silent_when_not_virtualized(tmp_path: Path, caplog: Any) -> None:
    from charlie_work.runner_slots import save_idle_streaks

    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        save_idle_streaks(
            fleet, {"owner/repo": 1}, source="prologue", full_pass_interval_seconds=300
        )

    assert not any("Fleet dir virtualization" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# write-side integration: touch_repo (fleet.json registration)
# ---------------------------------------------------------------------------


def test_touch_repo_warns_when_fleet_dir_is_virtualized(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """Registering a repo must warn when fleet.json would land in a copy."""
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.paths import runtime_paths

    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet, redirected)

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    class _FakeGH:
        def name_with_owner(self) -> str:
            return "owner/repo"

    paths = runtime_paths(repo_root, ".var/charlie-work")

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        touch_repo(str(fleet), repo_root, paths, _FakeGH())  # type: ignore[arg-type]

    assert any("fleet.json" in record.message for record in caplog.records)


def test_touch_repo_is_silent_when_not_virtualized(tmp_path: Path, caplog: Any) -> None:
    from charlie_work.fleet_registry import touch_repo
    from charlie_work.paths import runtime_paths

    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    class _FakeGH:
        def name_with_owner(self) -> str:
            return "owner/repo"

    paths = runtime_paths(repo_root, ".var/charlie-work")

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        touch_repo(str(fleet), repo_root, paths, _FakeGH())  # type: ignore[arg-type]

    assert not any("Fleet dir virtualization" in record.message for record in caplog.records)


# --- _paths_equal semantics (#899) -------------------------------------------
# The helper's docstring claimed PurePath.__eq__ was case-sensitive on Windows.
# It is not, and nothing here covered the claim, so it rotted undetected until a
# mutation in ci_fleet's vendored copy (body -> `a == b`) left that suite green.
# These pin the behaviour the corrected rationale actually rests on.


def test_purepath_equality_already_folds_case_per_flavour() -> None:
    """The premise the old docstring got backwards -- pinned, not asserted in prose."""
    from pathlib import PurePosixPath, PureWindowsPath

    assert PureWindowsPath("C:/Foo/Bar") == PureWindowsPath("c:/foo/bar")
    assert PurePosixPath("/Foo/Bar") != PurePosixPath("/foo/bar")
    # Separators, too -- so for two Path inputs the helper adds nothing.
    assert PureWindowsPath("C:/Foo") == PureWindowsPath(r"C:\Foo")


def test_paths_equal_collapses_str_and_path_operands(tmp_path: Path) -> None:
    """The real reason the helper exists: a str operand makes ``==`` fail outright.

    This is the mutation-killer. Replacing the body with ``a == b`` fails here,
    because ``str`` and ``Path`` are never equal regardless of spelling -- and
    the probe would read that inequality as a literal-vs-resolved divergence and
    report virtualization that is not happening.
    """
    from charlie_work.fleet_paths import _paths_equal

    assert str(tmp_path) != tmp_path  # plain == is not sufficient across types
    assert _paths_equal(str(tmp_path), tmp_path)  # type: ignore[arg-type]
    assert _paths_equal(tmp_path, str(tmp_path))  # type: ignore[arg-type]


def test_paths_equal_still_reports_genuine_divergence(tmp_path: Path) -> None:
    """The guard must not collapse paths that really do differ."""
    from charlie_work.fleet_paths import _paths_equal

    assert not _paths_equal(tmp_path / "literal", tmp_path / "redirected")
