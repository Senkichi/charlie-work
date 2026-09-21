"""``RepoMeta.name_with_owner`` per-pass caching (issue #1770 review finding 7).

Mirrors ``test_githublike_protocol_l09.py``'s
``test_repo_owner_name_delegate_uses_owner_shared_list_cache`` precedent for
the same class of fix: a delegate method backed by the owner's shared
``_list_cache`` (design doc Section 3.3/3.4), so a hot-path caller (the
CI-headroom clamp's per-pass ``_apply_concurrency_governor`` call) does not
pay a fresh ``gh`` subprocess on every call within one pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work.github import GitHub, GitHubError


def test_name_with_owner_caches_within_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A second call within the same pass (no ``invalidate_list_cache()`` in
    between) must not issue a second ``gh`` subprocess call."""
    gh = GitHub(tmp_path)
    calls: list[list[str]] = []

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False, **_kwargs: object):
        calls.append(args)
        return {"nameWithOwner": "owner/repo"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    first = gh.name_with_owner()
    second = gh.name_with_owner()

    assert first == "owner/repo"
    assert second == "owner/repo"
    assert len(calls) == 1


def test_name_with_owner_cache_cleared_by_invalidate_list_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``invalidate_list_cache()`` (called at the start of every orchestrator
    pass) must force a fresh lookup on the next call."""
    gh = GitHub(tmp_path)
    calls: list[list[str]] = []

    def fake_run(self: GitHub, args: list[str], *, json_output: bool = False, **_kwargs: object):
        calls.append(args)
        return {"nameWithOwner": "owner/repo"}

    monkeypatch.setattr(GitHub, "run", fake_run)

    gh.name_with_owner()
    gh.invalidate_list_cache()
    gh.name_with_owner()

    assert len(calls) == 2


def test_name_with_owner_failure_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A transient failure must be retried on the next call within the same
    pass, never remembered as a permanent failure -- unlike
    ``branch_protection``, whose failure is itself a meaningful, cacheable
    answer, an ``nameWithOwner`` lookup failure is transport noise."""
    gh = GitHub(tmp_path)
    calls: list[list[str]] = []

    def fake_run_fails_once(
        self: GitHub, args: list[str], *, json_output: bool = False, **_kwargs: object
    ):
        calls.append(args)
        if len(calls) == 1:
            return {}
        return {"nameWithOwner": "owner/repo"}

    monkeypatch.setattr(GitHub, "run", fake_run_fails_once)

    with pytest.raises(GitHubError):
        gh.name_with_owner()

    assert gh.name_with_owner() == "owner/repo"
    assert len(calls) == 2


def test_name_with_owner_uses_the_owner_shared_list_cache(tmp_path: Path) -> None:
    """The cache read/write must hit the owner's ``_list_cache`` directly --
    the same shared-by-reference decoupling every cache-backed delegate
    relies on."""
    gh = GitHub(tmp_path)
    gh._list_cache[("name_with_owner",)] = "seeded/repo"

    assert gh.name_with_owner() == "seeded/repo"
