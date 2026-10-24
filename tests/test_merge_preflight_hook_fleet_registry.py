"""Tests for ``charlie_work.merge_preflight_hook`` (#894).

Fleet-registry shapes: ``_load_fleet_roots`` (missing / corrupt /
non-dict / well-formed), ``_repo_for_cwd`` containment matching, and
the ``_decide`` failure modes they feed -- unreadable registry fails
closed on both the Bash and MCP paths, a genuinely empty fleet passes
through.

Everything is mocked: no network, no real fleet.json reads, no subprocesses,
no LLM processes. Split verbatim out of ``tests/test_merge_preflight_hook.py``
for the Track-1 attachment-budget split (#1564).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from charlie_work import merge_preflight_hook as hook


@pytest.fixture(autouse=True)
def _fresh_fleet_route_cache() -> Iterator[None]:
    """Start and end every test with an empty route cache / warn set.

    ``fleet_path._route_around_fork`` caches its per-file probe per literal
    dir for a TTL; without this reset one test's routing decision could leak
    into another test for the same default-dir spelling.
    """
    from charlie_work import fleet_paths

    fleet_paths._ROUTE_CACHE.clear()
    fleet_paths._ADMIN_SHARE_WARNED.clear()
    yield
    fleet_paths._ROUTE_CACHE.clear()
    fleet_paths._ADMIN_SHARE_WARNED.clear()


# ---------------------------------------------------------------------------
# _load_fleet_roots / _repo_for_cwd
# ---------------------------------------------------------------------------


def test_load_fleet_roots_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from charlie_work import layout

    registry = tmp_path / "fleet.json"
    monkeypatch.setattr(layout, "fleet_registry_path", lambda override=None: registry)

    # Missing file: no fleet configured -> {}.
    assert hook._load_fleet_roots() == {}

    # Corrupt file: read failure -> None (fail closed).
    registry.write_text("{not json", encoding="utf-8")
    assert hook._load_fleet_roots() is None

    # Non-dict JSON: also a read failure.
    registry.write_text("[]", encoding="utf-8")
    assert hook._load_fleet_roots() is None

    # Well-formed registry: owner/name lowercased -> repo_root Path.
    registry.write_text(
        '{"version": 1, "repos": {"Owner/Repo": {"repo_root": "C:/x/repo"}}}',
        encoding="utf-8",
    )
    roots = hook._load_fleet_roots()
    assert roots == {"owner/repo": Path("C:/x/repo")}


def test_repo_for_cwd_exact_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, root) == "o/repo"


def test_repo_for_cwd_nested_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    nested = root / "sub" / "dir"
    nested.mkdir(parents=True)
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, nested) == "o/repo"


def test_repo_for_cwd_sibling_prefix_does_not_match(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sibling = tmp_path / "repo-other"
    root.mkdir()
    sibling.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, sibling) is None


def test_repo_for_cwd_unrelated_cwd_returns_none(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    unrelated = tmp_path / "elsewhere"
    root.mkdir()
    unrelated.mkdir()
    roots = {"o/repo": root}
    assert hook._repo_for_cwd(roots, unrelated) is None


# ---------------------------------------------------------------------------
# _decide -- fleet-registry failure modes (fail closed on unreadable, pass
# through on genuinely empty)
# ---------------------------------------------------------------------------


def test_decide_bash_unreadable_registry_denies_mentioning_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # None = registry exists but cannot be read/parsed -> fail closed.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    reason = hook._decide("Bash", {"command": "gh pr merge 1 --squash"}, tmp_path)
    assert reason is not None
    assert "registry" in reason.lower()


def test_decide_bash_unreadable_registry_denies_even_with_explicit_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An unreadable registry cannot confirm ANY repo is outside the fleet.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    reason = hook._decide("Bash", {"command": "gh pr merge -R other/repo 1 --squash"}, tmp_path)
    assert reason is not None
    assert "registry" in reason.lower()


def test_decide_bash_genuinely_empty_fleet_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # {} = no registry file / no repos: nothing fleet-managed, nothing to guard.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {})
    assert hook._decide("Bash", {"command": "gh pr merge 1 --squash"}, tmp_path) is None
    assert (
        hook._decide("Bash", {"command": "gh pr merge -R other/repo 1 --squash"}, tmp_path) is None
    )


def test_decide_mcp_unreadable_registry_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression for the review finding on PR #1195: the MCP path must honor
    # the same fail-closed contract as the Bash path when the registry is
    # unreadable — otherwise a corrupt fleet.json silently disables
    # enforcement for MCP-initiated merges.
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: None)
    called: list[Any] = []
    monkeypatch.setattr(hook, "_run_merge_check", lambda *a: called.append(a) or (True, ""))
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "Senkichi", "repo": "charlie-work", "pullNumber": 5},
        tmp_path,
    )
    assert reason is not None
    assert "registry" in reason.lower()
    assert called == []


def test_decide_mcp_genuinely_empty_fleet_passes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hook, "_load_fleet_roots", lambda: {})
    reason = hook._decide(
        "mcp__github__merge_pull_request",
        {"owner": "o", "repo": "r", "pullNumber": 5},
        tmp_path,
    )
    assert reason is None


# ---------------------------------------------------------------------------
# _load_fleet_roots across the MSIX per-file fork (#2806)
# ---------------------------------------------------------------------------
#
# The #2806 fixture: the host's real fleet.json lists 9 repos, but a packaged
# (MSIX) session reading the same literal path gets a private fork from an
# earlier packaged write, listing only 3. Nothing flagged it: the directory
# resolves to itself (MSIX redirects files, never directories), so the
# #624 probe and the hook's own registry read both reported a healthy,
# smaller fleet -- bypassing the merge guard for 6 of 9 repos. Here the
# fork is simulated by patching ``Path.resolve`` so the literal fleet.json
# resolves elsewhere, the way a packaged session's ``resolve()`` reveals
# the LocalCache backing file.

_FLEET_REPOS = (
    "fleet-alpha",
    "fleet-bravo",
    "fleet-gamma",
    "fleet-delta",
    "fleet-epsilon",
    "fleet-zeta",
    "fleet-eta",
    "fleet-theta",
    "sub/nested-repo",
)


def _registry(repo_roots_dir: Path, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "version": 1,
        "repos": {f"Senkichi/{name}": {"repo_root": str(repo_roots_dir / name)} for name in names},
    }


def _virtualized_fleet_dir_with_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registry: dict[str, Any],
    *,
    route_exists: bool = True,
) -> None:
    """Plant a stale forked fleet.json and route reads to the real one.

    Patches, in order of where each matters:
    - ``Path.resolve``: the literal fleet.json diverges to a LocalCache-like
      copy (what a packaged session's resolve() reveals), everything under
      the administrative-share route short-circuits, and resolve failures
      degrade to the input path so the UNC route never trips on a host
      without admin shares.
    - ``Path.exists``: ONLY the administrative-share DIRECTORY exists (the
      path ``_route_around_fork`` actually probes, per the PR #2817 review
      finding -- the earlier fixture faked only ``<share>/fleet.json``), and
      every other ``\\\\localhost\\`` probe is False, so the routing result
      never depends on the host's admin-share loopback being up.
    - ``Path.read_text``: the real registry at the (not really existing)
      route reads ``registry`` when ``route_exists`` is True.
    """
    from charlie_work import fleet_paths

    localappdata = tmp_path / "AppData" / "Local"
    literal_dir = localappdata / "charlie-work"
    literal_dir.mkdir(parents=True)
    fleet_json = literal_dir / "fleet.json"
    # What the packaged session itself reads at the literal path: the stale
    # fork's content.
    fleet_json.write_text(
        json.dumps(_registry(tmp_path, tuple(_FLEET_REPOS[:3]))), encoding="utf-8"
    )

    admin_dir = fleet_paths._administrative_share_form(literal_dir)
    assert admin_dir is not None
    admin_registry = admin_dir / "fleet.json"

    real_resolve = pathlib.Path.resolve

    def fake_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
        if os.path.normcase(os.fspath(self)) == os.path.normcase(os.fspath(fleet_json)):
            return (
                tmp_path
                / "Packages"
                / "app"
                / "LocalCache"
                / "Local"
                / "charlie-work"
                / "fleet.json"
            )
        if str(self).lower().startswith("\\\\localhost\\"):
            return self
        try:
            return real_resolve(self, *args, **kwargs)
        except OSError:
            return self

    real_exists = pathlib.Path.exists
    real_read_text = pathlib.Path.read_text

    def fake_exists(self: Path) -> bool:
        probe = os.path.normcase(os.fspath(self))
        if str(self).lower().startswith("\\\\localhost\\"):
            if not route_exists:
                return False
            # The routed DIRECTORY (what _route_around_fork probes) and the
            # registry file inside it -- nothing else on the share.
            return probe in (
                os.path.normcase(os.fspath(admin_dir)),
                os.path.normcase(os.fspath(admin_registry)),
            )
        return real_exists(self)

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if route_exists and os.path.normcase(os.fspath(self)) == os.path.normcase(
            os.fspath(admin_registry)
        ):
            return json.dumps(registry)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", fake_resolve)
    monkeypatch.setattr(pathlib.Path, "exists", fake_exists)
    monkeypatch.setattr(pathlib.Path, "read_text", fake_read_text)
    monkeypatch.setenv("LOCALAPPDATA", str(localappdata))
    monkeypatch.delenv("CHARLIE_WORK_FLEET_DIR", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_load_fleet_roots_routes_around_virtualized_fleet_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forked fleet.json (3 repos) + routable admin share -> all 9 roots.

    The hook itself needs no changes: routing ``fleet_dir()`` to the
    non-redirected administrative share is enough for
    ``layout.fleet_registry_path()`` to read the daemon-visible registry.
    """
    real_registry = _registry(tmp_path / "roots", _FLEET_REPOS)
    (tmp_path / "roots").mkdir()
    _virtualized_fleet_dir_with_route(monkeypatch, tmp_path, real_registry)

    roots = hook._load_fleet_roots()
    assert roots is not None
    assert set(roots) == {f"senkichi/{name}" for name in _FLEET_REPOS}
    assert "senkichi/fleet-delta" in roots


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_load_fleet_roots_falls_back_to_forked_literal_when_route_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No admin share on this host -> the literal path is kept -> the fork read.

    PR #2817 review finding: the route-exists tests must not depend on the
    host's admin-share loopback. Here every ``\\\\localhost\\`` exists() is
    False, so ``fleet_dir()`` returns the literal path (the #624 fallback)
    and ``_load_fleet_roots`` reads the stale forked registry -- the exact
    degraded behavior a packaged session on a share-less host gets.
    """
    real_registry = _registry(tmp_path / "roots", _FLEET_REPOS)
    (tmp_path / "roots").mkdir()
    _virtualized_fleet_dir_with_route(monkeypatch, tmp_path, real_registry, route_exists=False)

    from charlie_work.fleet_paths import fleet_dir

    routed = fleet_dir()
    assert routed == Path(os.environ["LOCALAPPDATA"]) / "charlie-work"
    assert not str(routed).lower().startswith("\\\\localhost\\")

    roots = hook._load_fleet_roots()
    assert roots is not None
    assert set(roots) == {f"senkichi/{name}" for name in tuple(_FLEET_REPOS[:3])}
    assert "senkichi/fleet-delta" not in roots


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
def test_decide_bash_fleet_repo_merge_reaches_merge_check_despite_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``gh pr merge 5 --repo Senkichi/fleet-delta`` runs merge-check, not pass-through.

    Before #2806 the forked registry listed 3 repos, so fleet-delta looked
    "outside the fleet" and the hook left the call undecided — merging
    without a preflight on 6 of 9 fleet repos.
    """
    real_registry = _registry(tmp_path / "roots", _FLEET_REPOS)
    (tmp_path / "roots").mkdir()
    _virtualized_fleet_dir_with_route(monkeypatch, tmp_path, real_registry)

    called: list[tuple[Path, int]] = []

    def _recorded_merge_check(repo_root: Path, pr_number: int) -> tuple[bool, str]:
        called.append((repo_root, pr_number))
        return True, ""

    monkeypatch.setattr(hook, "_run_merge_check", _recorded_merge_check)

    reason = hook._decide(
        "Bash", {"command": "gh pr merge 5 --repo Senkichi/fleet-delta"}, tmp_path
    )
    assert reason is None
    assert called == [(Path(str(tmp_path / "roots" / "fleet-delta")), 5)]
