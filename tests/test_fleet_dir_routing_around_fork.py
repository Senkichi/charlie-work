"""Tests for ``fleet_dir()`` routing around a per-file MSIX fork (#2806).

Split out of ``tests/test_fleet_dir_virtualization.py`` so neither test
module grows past the file-size ratchet's 800-line high-water mark
(issue #1442). The probe itself is in ``charlie_work.fleet_paths``.

Issue #2806: MSIX/container redirection forks individual *files*, never the
directory, so a packaged session reads a stale private registry while the
directory probe reports healthy. The divergence is injected by patching
``Path.resolve`` (the way a packaged session's ``resolve()`` reveals the
LocalCache backing file), never a hardcoded package moniker.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest


@pytest.fixture(autouse=True)
def _fresh_fleet_route_cache() -> Iterator[None]:
    """Start and end every test with an empty route cache / warn set.

    ``_route_around_fork`` caches its per-file probe per literal dir for a TTL
    and its warning once per process; without this reset, one test's decision
    would leak into another test built on a different fixture for the same
    default-dir spelling.
    """
    from charlie_work import fleet_paths

    fleet_paths._FLEET_PROBE_CACHE.clear()
    fleet_paths._ADMIN_SHARE_WARNED.clear()
    yield
    fleet_paths._FLEET_PROBE_CACHE.clear()
    fleet_paths._ADMIN_SHARE_WARNED.clear()


def _patch_resolve_to_diverge(monkeypatch: Any, literal: Path, redirected: Path) -> None:
    """Make ``Path.resolve()`` return ``redirected`` for ``literal`` only.

    Copy of the helper in ``tests/test_fleet_dir_virtualization.py`` (the
    sibling module is a test monolith pinned by the file-size ratchet, so a
    shared helper module import is deliberately avoided here, per the #1283
    lineage's split pattern).
    """
    real_resolve = pathlib.Path.resolve

    def fake_resolve(self, *args, **kwargs):
        result = real_resolve(self, *args, **kwargs)
        if os.path.normcase(os.fspath(result)) == os.path.normcase(os.fspath(literal)):
            return redirected
        return result

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)


# ---------------------------------------------------------------------------
# _administrative_share_form: the literal route form (#2806 rework round 5)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_administrative_share_form_pins_literal_route_strings() -> None:
    """Pin the exact ``\\\\localhost\\<drive>$\\<rest>`` route strings.

    Every other routing test derives its expected route by calling
    :func:`fleet_paths._administrative_share_form` itself, so a mutation to
    the route construction (for example dropping the ``$``) leaves all of
    them green -- this test pins the literal expected strings instead, and
    asserts ``None`` for the non-routeable inputs (a UNC path, a drive
    root, a relative path, and a driveless rooted path).
    """
    from charlie_work import fleet_paths

    pinned = fleet_paths._administrative_share_form(Path(r"C:\Users\x\AppData\Local\charlie-work"))
    assert pinned is not None
    assert pinned == Path(r"\\localhost\C$\Users\x\AppData\Local\charlie-work")

    unc_input = Path(r"\\otherhost\share\charlie-work")
    assert fleet_paths._administrative_share_form(unc_input) is None
    drive_root = Path(r"C:\\")
    assert fleet_paths._administrative_share_form(drive_root) is None
    relative = Path(r"charlie-work")
    assert fleet_paths._administrative_share_form(relative) is None
    driveless_rooted = Path(r"\AppData\Local\charlie-work")
    assert fleet_paths._administrative_share_form(driveless_rooted) is None


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
    route exists, ``fleet_dir()`` returns the route.

    Hermetic by construction: ``Path.exists`` is faked for the admin-share
    DIRECTORY (the path the code actually probes) and every other
    ``\\\\localhost\\`` probe is False, so the result cannot depend on the
    host's admin-share loopback being up.
    """
    from charlie_work import fleet_paths

    literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    admin_dir = fleet_paths._administrative_share_form(literal)
    assert admin_dir is not None
    real_exists = pathlib.Path.exists

    def fake_exists(self: Path) -> bool:
        probe = os.path.normcase(os.fspath(self))
        if str(self).lower().startswith("\\\\localhost\\"):
            return probe == os.path.normcase(os.fspath(admin_dir))
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    routed = fleet_paths.fleet_dir()
    assert os.path.normcase(str(routed)) == os.path.normcase(str(admin_dir))


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
    """A forked pinned dir with an available route is still returned unchanged.

    Invariant under test: an explicit ``override`` and the
    ``CHARLIE_WORK_FLEET_DIR`` env var are never routed, even when the
    pinned dir holds a forked file AND its admin-share route exists —
    callers that pin a path mean it.
    """
    from charlie_work import fleet_paths

    def _pin_forked_and_routable(dir_pinned: Path) -> None:
        forked = dir_pinned / "config.yaml"
        forked.write_text("b: 1", encoding="utf-8")
        # Same fixture shape a packaged session exposes: a file under the
        # fleet dir resolving to the package's LocalCache backing copy.
        _patch_resolve_to_diverge(
            monkeypatch, dir_pinned / "config.yaml", dir_pinned / "Packages" / "fork"
        )
        admin_dir = fleet_paths._administrative_share_form(dir_pinned)
        assert admin_dir is not None
        real_exists = pathlib.Path.exists

        def fake_exists(self: Path) -> bool:
            probe = os.path.normcase(os.path.abspath(os.fspath(self)))
            if probe.startswith("\\\\localhost\\"):
                return probe == os.path.normcase(os.path.abspath(os.fspath(admin_dir)))
            return real_exists(self)

        monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    pinned = tmp_path / "pinned-fleet-dir"
    pinned.mkdir()
    _pin_forked_and_routable(pinned)
    routed = fleet_paths.fleet_dir(override=str(pinned))
    assert routed == pinned
    assert not str(routed).lower().startswith("\\\\localhost\\")

    env_pinned = tmp_path / "env-fleet-dir"
    env_pinned.mkdir()
    _pin_forked_and_routable(env_pinned)
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(env_pinned))
    routed_env = fleet_paths.fleet_dir()
    assert routed_env == env_pinned
    assert not str(routed_env).lower().startswith("\\\\localhost\\")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_route_warning_logged_once_per_process(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """The routing choice is logged as a warning, once per process per dir."""
    from charlie_work import fleet_paths

    literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    admin_dir = fleet_paths._administrative_share_form(literal)
    assert admin_dir is not None
    real_exists = pathlib.Path.exists

    def fake_exists(self: Path) -> bool:
        probe = os.path.normcase(os.fspath(self))
        if str(self).lower().startswith("\\\\localhost\\"):
            return probe == os.path.normcase(os.fspath(admin_dir))
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


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_route_probe_scans_once_within_ttl_and_reprobes_after_expiry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Two fleet_dir() calls inside the TTL scan the dir once; a call after
    expiry re-probes. The cache keeps the per-file probe off every fleet
    read; the clock is faked so the expiry branch is deterministic."""
    from charlie_work import fleet_paths

    literal, _fleet_json, _redirected = _routing_fixture(monkeypatch, tmp_path)

    admin_dir = fleet_paths._administrative_share_form(literal)
    assert admin_dir is not None
    real_exists = pathlib.Path.exists

    def fake_exists(self: Path) -> bool:
        probe = os.path.normcase(os.fspath(self))
        if str(self).lower().startswith("\\\\localhost\\"):
            return probe == os.path.normcase(os.fspath(admin_dir))
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)

    scans: list[Path] = []
    real_findings = fleet_paths._fleet_file_findings

    def counting_findings(literal_: Path) -> "list[tuple[Path, Path]]":
        scans.append(literal_)
        return real_findings(literal_)

    monkeypatch.setattr(fleet_paths, "_fleet_file_findings", counting_findings)

    clock = [1000.0]

    def fake_monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(fleet_paths, "_MONOTONIC", fake_monotonic)

    first = fleet_paths.fleet_dir()
    second = fleet_paths.fleet_dir()
    assert os.path.normcase(str(first)) == os.path.normcase(str(admin_dir))
    assert second == first
    assert len(scans) == 1, "two fleet_dir() calls inside the TTL must scan once"

    clock[0] = 1000.0 + fleet_paths._FLEET_PROBE_CACHE_TTL_SECONDS + 0.001
    third = fleet_paths.fleet_dir()
    assert third == first
    assert len(scans) == 2, "a call after TTL expiry must re-probe the fleet dir"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_fleet_dir_route_cache_reprobes_after_mid_session_fork_appears(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A fork appearing mid-session is picked up on a later TTL refresh, not
    only at restart (corrected fleet_dir() docstring claim)."""
    from charlie_work import fleet_paths

    localappdata = tmp_path / "AppData" / "Local"
    literal = localappdata / "charlie-work"
    literal.mkdir(parents=True)
    (literal / "config.yaml").write_text("b: 1", encoding="utf-8")
    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    monkeypatch.delenv("CHARLIE_WORK_FLEET_DIR", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")

    clock = [1000.0]

    def fake_monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(fleet_paths, "_MONOTONIC", fake_monotonic)

    no_fork = fleet_paths.fleet_dir()
    assert no_fork == literal

    # Now the packaged session forks a file: resolve() diverges from here on.
    forked = literal / "state.json"
    forked.write_text("{}", encoding="utf-8")
    redirected = localappdata / "Packages" / "app" / "LocalCache" / "charlie-work" / "state.json"
    _patch_resolve_to_diverge(monkeypatch, forked, redirected)

    # Keep the route decision hermetic: no admin share exists in this fixture
    # (every \\localhost\ probe is False) until the last step below.
    real_exists = pathlib.Path.exists
    admin_dir = fleet_paths._administrative_share_form(literal)
    assert admin_dir is not None

    def no_share_exists(self: Path) -> bool:
        if str(self).lower().startswith("\\\\localhost\\"):
            return False
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", no_share_exists)

    still_cached = fleet_paths.fleet_dir()
    assert still_cached == literal, "inside the TTL the old decision is kept"

    clock[0] = 1000.0 + fleet_paths._FLEET_PROBE_CACHE_TTL_SECONDS + 0.001
    refreshed = fleet_paths.fleet_dir()
    assert refreshed == literal, "no route on this host: literal kept, but re-probed"

    # The route now comes online: the next TTL refresh re-probes and routes.
    def routable_exists(self: Path) -> bool:
        probe = os.path.normcase(os.fspath(self))
        if str(self).lower().startswith("\\\\localhost\\"):
            return probe == os.path.normcase(os.fspath(admin_dir))
        return real_exists(self)

    monkeypatch.setattr(pathlib.Path, "exists", routable_exists)

    clock[0] += fleet_paths._FLEET_PROBE_CACHE_TTL_SECONDS + 0.001
    routed_later = fleet_paths.fleet_dir()
    assert os.path.normcase(str(routed_later)) == os.path.normcase(str(admin_dir))


# ---------------------------------------------------------------------------
# write guard: TTL-cached directory probe (#2806, PR #2817 rework round 3)
# ---------------------------------------------------------------------------


def test_warn_on_write_directory_probe_scans_once_within_ttl_and_reprobes_after_expiry(
    tmp_path: Path, monkeypatch: Any, caplog: Any
) -> None:
    """Two write-guard calls with a directory argument inside the TTL scan
    the dir once; a call after the clock expires re-probes."""
    from charlie_work import fleet_paths
    from charlie_work.fleet_paths import warn_fleet_dir_virtualization_on_write

    fleet = tmp_path / "charlie-work"
    fleet.mkdir(parents=True)
    fleet_json = fleet / "fleet.json"
    fleet_json.write_text("{}", encoding="utf-8")
    redirected = (
        tmp_path / "Packages" / "app" / "LocalCache" / "Local" / "charlie-work" / "fleet.json"
    )
    _patch_resolve_to_diverge(monkeypatch, fleet_json, redirected)

    scans: list[Path] = []
    real_findings = fleet_paths._fleet_file_findings

    def counting_findings(literal_: Path) -> "list[tuple[Path, Path]]":
        scans.append(literal_)
        return real_findings(literal_)

    monkeypatch.setattr(fleet_paths, "_fleet_file_findings", counting_findings)

    clock = [1000.0]

    def fake_monotonic() -> float:
        return clock[0]

    monkeypatch.setattr(fleet_paths, "_MONOTONIC", fake_monotonic)

    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        warn_fleet_dir_virtualization_on_write(fleet, context="writing test-a")
        warn_fleet_dir_virtualization_on_write(fleet, context="writing test-b")

    assert len(scans) == 1, "two write-guard calls inside the TTL must scan once"
    assert len(caplog.records) == 2, "the cache skips the scan, not the warning"

    clock[0] = 1000.0 + fleet_paths._FLEET_PROBE_CACHE_TTL_SECONDS + 0.001
    with caplog.at_level(logging.WARNING, logger="charlie_work.fleet_paths"):
        warn_fleet_dir_virtualization_on_write(fleet, context="writing test-c")

    assert len(scans) == 2, "a write-guard call after TTL expiry must re-probe the fleet dir"
    assert any("test-c" in record.message for record in caplog.records)


def test_warn_on_write_skips_per_file_probe_for_unc_directory(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A UNC/routed directory argument triggers zero per-file resolve() calls.

    The administrative-share route is non-redirected by construction, so the
    per-file probe there would be seconds of I/O that finds nothing. The
    UNC dir is made to exist and resolve cleanly so this test fails if the
    UNC skip is dropped (the probe spy would be called) regardless of
    whether this host's admin-share loopback is up.
    """
    from charlie_work import fleet_paths
    from charlie_work.fleet_paths import warn_fleet_dir_virtualization_on_write

    unc_dir = fleet_paths._administrative_share_form(tmp_path / "charlie-work")
    assert unc_dir is not None

    def raising_resolve(self, *args, **kwargs):
        raise OSError("simulated unresolvable path")

    monkeypatch.setattr(pathlib.Path, "resolve", raising_resolve)
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self, *args, **kwargs: True, raising=True)

    probe_calls: list[Path] = []

    def spy_findings(literal_: Path) -> "list[tuple[Path, Path]]":
        probe_calls.append(literal_)
        return []

    monkeypatch.setattr(fleet_paths, "_fleet_file_findings", spy_findings)

    # Must skip the per-file probe (and never raise: no real I/O happens).
    warn_fleet_dir_virtualization_on_write(unc_dir, context="writing routed state")

    assert probe_calls == [], "the per-file probe must be skipped for a UNC directory"


def test_warn_on_write_swallows_is_dir_oserror(tmp_path: Path, monkeypatch: Any) -> None:
    """A non-ENOENT ``OSError`` from ``Path.is_dir`` is swallowed, not raised.

    ``fleet_dispatch`` and the other fleet writers call the guard unwrapped
    around the real write, so a guard that re-raises (Python >= 3.11
    re-raises non-ENOENT OSErrors from ``is_dir``) would crash the write it
    was only supposed to sit beside.
    """
    from charlie_work.fleet_paths import warn_fleet_dir_virtualization_on_write

    guarded_dir = tmp_path / "charlie-work"
    guarded_dir.mkdir()

    def raising_is_dir(self: Path, *args: Any, **kwargs: Any) -> bool:
        raise OSError("simulated access denied / bad reparse point")

    monkeypatch.setattr(pathlib.Path, "is_dir", raising_is_dir)

    warn_fleet_dir_virtualization_on_write(guarded_dir, context="writing guarded state")
