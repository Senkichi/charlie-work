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
import sys
from pathlib import Path
from typing import Any

import pytest

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


# Note: this file used to also carry a "save_idle_streaks" integration pair
# here (the #590 "I deployed it" case), exercising charlie_work's own
# runner_slots.save_idle_streaks against charlie_work.fleet_paths. Issue #921
# deleted charlie_work.runner_slots -- the live save_idle_streaks is now
# ci_fleet.runner_slots.save_idle_streaks, which calls ci_fleet's own vendored
# _vendor.fleet_paths, not this module's warn_fleet_dir_virtualization_on_write.
# Repointing the import would silently start asserting on a different logger
# ("ci_fleet._vendor.fleet_paths") under a docstring about *this* probe, and
# the integration is already covered where the live code lives: ci_runners'
# own tests/test_fleet_dir_virtualization.py exercises the same save_idle_streaks
# x vendored-warn pairing (confirmed present, not assumed).


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


# ---------------------------------------------------------------------------
# write-side integration: _run_fleet_allocation_prologue (fleet allocation events)
# ---------------------------------------------------------------------------


def test_allocation_prologue_warns_when_fleet_dir_is_virtualized(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """The allocation prologue must warn when fleet events would land in a copy.

    Mirrors ``test_touch_repo_warns_when_fleet_dir_is_virtualized``: the guard
    fires through the real ``warn_fleet_dir_virtualization_on_write``, proving
    ``_run_fleet_allocation_prologue`` is wired to the guard with
    ``resolved_fleet_dir`` (the ``fleet_dir()`` path, not a per-repo
    ``state_dir``). Issue #603 added the call site; this test pins that wiring
    so a future refactor that drops the call or passes the wrong path is caught.
    """
    import json
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from charlie_work.fleet_dispatch import (
        _CiFleetDirtyCheck,
        _run_fleet_allocation_prologue,
    )

    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True, exist_ok=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet, redirected)

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)

    # Minimal fleet.json registry with one valid repo root so the prologue
    # reaches the warn_fleet_dir_virtualization_on_write call site (it only
    # fires after the anchor-root check passes).
    (fleet / "fleet.json").write_text(
        json.dumps(
            {"version": 1, "repos": {"owner/repo": {"repo_root": str(repo_root)}}},
            indent=2,
        ),
        encoding="utf-8",
    )

    config = SimpleNamespace(
        runner_allocation=SimpleNamespace(
            enabled=True, max_running_runners=4, managed_root="C:/test-root"
        ),
        supervisor=SimpleNamespace(full_pass_interval_seconds=300),
        runtime=None,
        runner_scaling=SimpleNamespace(managed_root="C:/test-root"),
    )

    # Mock run_allocation_pass so no real CI fleet work runs; the skipped
    # result exercises the same state_path routing the fix changes without
    # needing a real runner pool.
    result = MagicMock(
        ok=True,
        skipped=True,
        notes=("no configured runners",),
        error=None,
        started=0,
        parked=0,
        results=(),
        plan=None,
    )
    with (
        patch(
            "charlie_work.fleet_dispatch.run_allocation_pass",
            return_value=result,
        ),
        patch("charlie_work.fleet_dispatch.GitHub"),
        patch(
            "charlie_work.fleet_dispatch._ci_fleet_worktree_dirty",
            return_value=_CiFleetDirtyCheck(is_dirty=False),
        ),
        patch(
            "charlie_work.fleet_dispatch.warn_fleet_dir_virtualization_on_write",
            wraps=warn_fleet_dir_virtualization_on_write,
        ) as warn_spy,
    ):
        with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
            _run_fleet_allocation_prologue(str(fleet), config, dry_run=False)

    # The guard was invoked with resolved_fleet_dir (the fleet_dir() path) and
    # the allocation-events context — proving the call site is wired.
    warn_spy.assert_called_once_with(fleet, context="writing fleet allocation events")
    # The real guard fired (via wraps), so the warning reached the logger.
    assert any(
        "Fleet dir virtualization" in record.message
        and "writing fleet allocation events" in record.message
        for record in caplog.records
    )


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


# ---------------------------------------------------------------------------
# per-file probe (#2806): MSIX redirection forks files, never the directory
# ---------------------------------------------------------------------------


def test_fleet_dir_virtualization_reports_virtualized_fleet_json(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A forked file inside the (self-resolving) dir is reported.

    The #2806 fixture: the directory resolves to itself (MSIX never
    redirects directories), while ``fleet.json`` inside it resolves to the
    package's private copy -- exactly how a packaged session reads a stale
    registry while ``detect_path_virtualization(dir)`` says all is well.
    """
    from charlie_work.fleet_paths import fleet_dir_virtualization

    fleet = tmp_path / "charlie-work"
    fleet.mkdir(parents=True)
    fleet_json = fleet / "fleet.json"
    fleet_json.write_text("{}", encoding="utf-8")
    redirected = (
        tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work" / "fleet.json"
    )
    _patch_resolve_to_diverge(monkeypatch, fleet_json, redirected)

    findings = fleet_dir_virtualization(override=str(fleet))
    assert findings is not None
    assert [pair[0] for pair in findings] == [fleet_json]
    assert [pair[1] for pair in findings] == [redirected]


def test_fleet_dir_virtualization_no_findings_without_diverging_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Planted control: every file resolving to itself -> no findings."""
    from charlie_work.fleet_paths import fleet_dir_virtualization

    fleet = tmp_path / "charlie-work"
    fleet.mkdir(parents=True)
    (fleet / "fleet.json").write_text("{}", encoding="utf-8")
    (fleet / "config.yaml").write_text("b: 1", encoding="utf-8")

    # No divergence patch: every path resolves to itself.
    assert fleet_dir_virtualization(override=str(fleet)) is None


def test_warn_on_write_reports_virtualized_file_inside_directory(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """A directory-level write guard must still catch an already-forked file.

    MSIX forks the file the first packaged write creates; from then on the
    probe-on-the-directory the #624 writers originally used reports nothing
    even though every subsequent write of ``fleet.json`` lands in the fork.
    """
    fleet = tmp_path / "charlie-work"
    fleet.mkdir(parents=True)
    fleet_json = fleet / "fleet.json"
    fleet_json.write_text("{}", encoding="utf-8")
    redirected = (
        tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work" / "fleet.json"
    )
    _patch_resolve_to_diverge(monkeypatch, fleet_json, redirected)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        warn_fleet_dir_virtualization_on_write(fleet, context="writing fleet.json")

    joined = " ".join(record.message for record in caplog.records)
    assert "#2806" in joined
    assert str(fleet_json) in joined
    assert str(redirected) in joined


# ---------------------------------------------------------------------------
# fleet_dir() routing around a per-file fork (#2806)
# ---------------------------------------------------------------------------


def _routing_fixture(
    monkeypatch: Any, tmp_path: Path, fleet_json_content: str = "{}"
) -> "tuple[Path, Path, Path]":
    """A win32 default fleet dir whose ``fleet.json`` diverges via resolve().

    Sets LOCALAPPDATA to the fixture root, clears the env override, keeps the
    real ``sys.platform`` (these tests only run on win32 -- the routing branch
    is win32-gated), and returns ``(literal_dir, fleet_json, redirected)``.
    """
    import sys as _sys

    localappdata = tmp_path / "AppData" / "Local"
    literal = localappdata / "charlie-work"
    literal.mkdir(parents=True)
    fleet_json = literal / "fleet.json"
    fleet_json.write_text(fleet_json_content, encoding="utf-8")
    redirected = localappdata / "Packages" / "app" / "LocalCache" / "charlie-work" / "fleet.json"
    _patch_resolve_to_diverge(monkeypatch, fleet_json, redirected)
    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    monkeypatch.delenv("CHARLIE_WORK_FLEET_DIR", raising=False)
    monkeypatch.setattr(_sys, "platform", "win32")
    return literal, fleet_json, redirected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_routes_to_administrative_share_when_route_exists(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """When a file in the default fleet dir is forked and the admin-share
    route exists, ``fleet_dir()`` returns the route."""
    from charlie_work import fleet_paths

    literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    admin_registry = fleet_paths._administrative_share_form(literal) / "fleet.json"
    real_exists = pathlib.Path.exists

    def fake_exists(self: Path) -> bool:
        if os.path.normcase(os.fspath(self)) == os.path.normcase(os.fspath(admin_registry)):
            return True
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    routed = fleet_paths.fleet_dir()
    expected = "\\\\localhost\\" + literal.drive[0] + "$" + str(literal)[len(literal.drive) :]
    assert os.path.normcase(str(routed)) == os.path.normcase(expected)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_returns_literal_when_administrative_share_absent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """No route on this host -> keep the literal path (the #624 fallback)."""
    from charlie_work import fleet_paths

    literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    real_exists = pathlib.Path.exists

    def fake_exists(self: Path) -> bool:
        if str(self).lower().startswith("\\\\localhost\\"):
            return False
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    assert fleet_paths.fleet_dir() == literal


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_env_and_override_bypass_routing(tmp_path: Path, monkeypatch: Any) -> None:
    """An explicit env var or override is returned unchanged, fork or not."""
    from charlie_work import fleet_paths

    _literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    pinned = tmp_path / "pinned-fleet-dir"
    pinned.mkdir()
    assert fleet_paths.fleet_dir(override=str(pinned)) == pinned

    env_pinned = tmp_path / "env-fleet-dir"
    env_pinned.mkdir()
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(env_pinned))
    assert fleet_paths.fleet_dir() == env_pinned


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_route_warning_logged_once_per_process(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """The routing choice is logged as a warning, once per process per dir."""
    from charlie_work import fleet_paths

    literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    admin_registry = fleet_paths._administrative_share_form(literal) / "fleet.json"
    real_exists = pathlib.Path.exists

    def fake_exists(self: Path) -> bool:
        if os.path.normcase(os.fspath(self)) == os.path.normcase(os.fspath(admin_registry)):
            return True
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        fleet_paths.fleet_dir()
        fleet_paths.fleet_dir()

    route_warnings = [
        record.message for record in caplog.records if "per-file virtualization" in record.message
    ]
    assert len(route_warnings) == 1
    assert str(literal) in route_warnings[0]
    assert "localhost" in route_warnings[0]
    assert "#2806" in route_warnings[0]


# ---------------------------------------------------------------------------
# charlie doctor check (#2806): state-file divergence is an error
# ---------------------------------------------------------------------------


def test_doctor_virtualization_check_is_error_for_forked_state_file(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The check names ``fleet.json`` and fails as an error.

    The merge-preflight hook and the config layering depend on ``fleet.json``
    being the daemon-visible copy; a forked one is not a "your writes are
    private" caveat but a silently wrong read, hence error severity.
    """
    from charlie_work import doctor

    fleet = tmp_path / "charlie-work"
    fleet.mkdir(parents=True)
    fleet_json = fleet / "fleet.json"
    fleet_json.write_text("{}", encoding="utf-8")
    redirected = (
        tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work" / "fleet.json"
    )
    _patch_resolve_to_diverge(monkeypatch, fleet_json, redirected)

    checks: list[tuple[str, bool, str, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        checks.append((name, ok, detail, severity))

    doctor._check_fleet_dir_virtualization(add, fleet_dir_override=str(fleet))

    assert [c[0] for c in checks] == ["fleet dir virtualization"]
    assert checks[0][1] is False
    assert checks[0][3] == "error"
    assert "fleet.json" in checks[0][2]
    assert str(redirected) in checks[0][2]


def test_doctor_virtualization_check_stays_warning_for_dir_divergence(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The #624 directory-level case keeps its warning severity."""
    from charlie_work import doctor

    fleet = tmp_path / "fleet"
    fleet.mkdir(parents=True)
    redirected = tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work"
    _patch_resolve_to_diverge(monkeypatch, fleet, redirected)

    checks: list[tuple[str, str]] = []

    def add(name: str, ok: bool, detail: str, *, severity: str = "error") -> None:
        checks.append((name, severity))

    doctor._check_fleet_dir_virtualization(add, fleet_dir_override=str(fleet))

    assert len(checks) == 1
    assert checks[0][1] == "warning"
