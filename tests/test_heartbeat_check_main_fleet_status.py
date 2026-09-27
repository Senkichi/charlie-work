"""Main-loop fleet-status and blocked-issue tests for ``scripts/heartbeat_check.py``.

Split out of ``tests/test_heartbeat_check.py`` (issue #1556, Track-1):
``main``'s once-per-beat ``charlie fleet status`` invocation and
degraded-status consumer annotation (issue #1438), plus
``get_blocked_issue_numbers`` cache-freshness warnings (issue #1463).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import _load_heartbeat_check
from charlie_work.config import RuntimeConfig


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


# ---------------------------------------------------------------------------
# Blocked-issue lookup economy (issue #1438)
# ---------------------------------------------------------------------------


def _write_fleet_json(
    hb: ModuleType, fleet_dir: Path, repos: list[tuple[str, Path, Path]]
) -> None:
    """Write a fleet.json registering ``repos`` under ``fleet_dir``.

    Each tuple is (slug, repo_root, state_dir). ``config_path`` is left
    empty so config-reading checks degrade to their no-config path rather
    than parsing a real file.
    """
    fleet_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "repos": {
            slug: {"repo_root": str(root), "state_dir": str(state)} for slug, root, state in repos
        }
    }
    (fleet_dir / "fleet.json").write_text(json.dumps(payload), encoding="utf-8")


def test_main_runs_charlie_fleet_status_exactly_once_per_beat(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1438: ``charlie fleet status --json`` must be invoked exactly
    once per heartbeat run, not once per consumer per repo.

    The fleet status snapshot cannot meaningfully change between checks
    within one beat, so main() fetches it once before the per-repo loop and
    threads the result (or the degraded-marker) into every consumer. This
    test installs a counting fake ``subprocess.run`` over the full ``main()``
    run with two registered repos -- which exercises both the
    dispatch-coverage and armable-backlog consumers for each repo (four
    consumer call-sites total) -- and asserts the charlie fleet status
    subprocess fires exactly once.
    """
    fleet_dir = tmp_path / "fleet"
    repo_a_root = tmp_path / "repo-a"
    repo_a_state = tmp_path / "state-a"
    repo_b_root = tmp_path / "repo-b"
    repo_b_state = tmp_path / "state-b"
    for d in (repo_a_root, repo_a_state, repo_b_root, repo_b_state):
        d.mkdir(parents=True, exist_ok=True)
    _write_fleet_json(
        hb,
        fleet_dir,
        [("owner/repo-a", repo_a_root, repo_a_state), ("owner/repo-b", repo_b_root, repo_b_state)],
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(tmp_path / "hb-state.json"))
    monkeypatch.setenv(
        "CHARLIE_WORK_HEARTBEAT_SUPPRESSIONS", str(tmp_path / "no-suppressions.yaml")
    )

    fleet_status_payload = json.dumps(
        {
            "data": {
                "repos": {
                    "owner/repo-a": {"blocked": []},
                    "owner/repo-b": {"blocked": []},
                }
            }
        }
    )

    charlie_status_calls: list[int] = []  # records the timeout kwarg per call

    class _FakeProc:
        def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, *a, **k):
        if args[:4] == ["charlie", "fleet", "status", "--json"]:
            charlie_status_calls.append(k.get("timeout"))
            return _FakeProc(returncode=0, stdout=fleet_status_payload, stderr="")
        # Every other subprocess (gh, git log, schtasks, ...) fails benignly;
        # the checks degrade to ANOMALY/OK lines without crashing.
        return _FakeProc(returncode=1, stdout="", stderr="fake subprocess disabled in test")

    monkeypatch.setattr(hb.subprocess, "run", fake_run)

    hb.main()

    assert len(charlie_status_calls) == 1, (
        f"expected exactly one charlie fleet status --json call per beat, "
        f"got {len(charlie_status_calls)}"
    )
    # Issue #1438 AC: timeout >= 120s so the bound is an outlier detector,
    # not the median runtime.
    assert charlie_status_calls[0] >= 120, (
        f"expected timeout >= 120s, got {charlie_status_calls[0]}"
    )


def test_main_annotates_all_consumers_when_fleet_status_degrades(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Issue #1438 AC: the degraded path still annotates all four consumer
    check lines (dispatch-coverage + armable-backlog, for each of two repos).

    When ``charlie fleet status --json`` fails, the single degraded-marker
    string is threaded into every consumer. This test captures main()'s
    stdout and asserts each of the four consumer check lines carries the
    degraded caveat.
    """
    fleet_dir = tmp_path / "fleet"
    repo_a_root = tmp_path / "repo-a"
    repo_a_state = tmp_path / "state-a"
    repo_b_root = tmp_path / "repo-b"
    repo_b_state = tmp_path / "state-b"
    for d in (repo_a_root, repo_a_state, repo_b_root, repo_b_state):
        d.mkdir(parents=True, exist_ok=True)
    _write_fleet_json(
        hb,
        fleet_dir,
        [("owner/repo-a", repo_a_root, repo_a_state), ("owner/repo-b", repo_b_root, repo_b_state)],
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(tmp_path / "hb-state.json"))
    monkeypatch.setenv(
        "CHARLIE_WORK_HEARTBEAT_SUPPRESSIONS", str(tmp_path / "no-suppressions.yaml")
    )

    degraded_marker = "timed out after 120 seconds"

    class _FakeProc:
        def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, *a, **k):
        if args[:4] == ["charlie", "fleet", "status", "--json"]:
            # Simulate the timeout path: subprocess.TimeoutExpired is a
            # SubprocessError subclass, which get_blocked_issue_numbers
            # catches and converts into the degraded-marker string.
            raise subprocess.TimeoutExpired(cmd=args, timeout=k.get("timeout", 120))
        # dispatch-coverage's run_gh_json must succeed with an empty issue
        # list so the check reaches its OK line (where the degraded caveat
        # is appended); other subprocesses fail benignly.
        if args[:2] == ["gh", "issue"] and "list" in args:
            return _FakeProc(returncode=0, stdout="[]", stderr="")
        return _FakeProc(returncode=1, stdout="", stderr="fake subprocess disabled in test")

    monkeypatch.setattr(hb.subprocess, "run", fake_run)

    import io

    captured = io.StringIO()
    monkeypatch.setattr(hb.sys, "stdout", captured)
    hb.main()
    output = captured.getvalue()

    for slug in ("owner/repo-a", "owner/repo-b"):
        assert any(
            f"dispatch-coverage {slug}:" in line and degraded_marker in line
            for line in output.splitlines()
        ), f"dispatch-coverage {slug} missing degraded caveat in:\n{output}"
        assert any(
            f"armable-backlog {slug}:" in line and degraded_marker in line
            for line in output.splitlines()
        ), f"armable-backlog {slug} missing degraded caveat in:\n{output}"


# ---------------------------------------------------------------------------
# Issue #1463 round-3: get_blocked_issue_numbers reads cache freshness
# ---------------------------------------------------------------------------


class _FakeStatusProc:
    def __init__(self, *, stdout: str) -> None:
        self.returncode = 0
        self.stdout = stdout
        self.stderr = ""


def _fleet_status_payload(*, cache_age_seconds: float | None) -> str:
    """Build a ``charlie fleet status --json`` payload with one repo."""
    repo_data: dict[str, Any] = {
        "ready_issue_count": 3,
        "blocked": [{"issue": 42, "blockers": []}],
        "cache_age_seconds": cache_age_seconds,
    }
    return json.dumps(
        {"ok": True, "message": "fleet status", "data": {"repos": {"owner/repo": repo_data}}}
    )


def _fleet_status_payload_multi(*, repo_ages: dict[str, float]) -> str:
    """Build a ``charlie fleet status --json`` payload with several repos.

    ``repo_ages`` maps slug -> served ``cache_age_seconds``; dict insertion
    order is preserved in the JSON, so a test controls which stale repo the
    payload lists first.
    """
    repos = {
        slug: {
            "ready_issue_count": 1,
            "blocked": [{"issue": 42, "blockers": []}],
            "cache_age_seconds": age,
        }
        for slug, age in repo_ages.items()
    }
    return json.dumps({"ok": True, "message": "fleet status", "data": {"repos": repos}})


def test_status_cache_stale_seconds_exceeds_default_serving_ttl(hb: ModuleType) -> None:
    """Issue #1886: ``STATUS_CACHE_STALE_SECONDS`` must sit strictly above
    the producer's default serving TTL. ``read_status_snapshot`` only
    serves a snapshot whose age is <=
    ``runtime.status_snapshot_ttl_seconds``, so a threshold at or below
    that default flags routine healthy refresh cadence on every repo —
    exactly the pairing #1464 shipped (600s inside the 900s ceiling).
    Pinning the ordering here makes CI, not a code comment, stop the two
    numbers from being re-paired in the wrong order again."""
    ttl = RuntimeConfig().status_snapshot_ttl_seconds
    assert hb.STATUS_CACHE_STALE_SECONDS > ttl, (
        f"STATUS_CACHE_STALE_SECONDS={hb.STATUS_CACHE_STALE_SECONDS}s must exceed "
        f"the default status_snapshot_ttl_seconds={ttl}s"
    )


def test_get_blocked_issue_numbers_warns_on_stale_cache(
    hb: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-3 review: ``get_blocked_issue_numbers`` must read
    ``cache_age_seconds`` and return a staleness warning when the cache is
    older than ``STATUS_CACHE_STALE_SECONDS``. The blocked data is still
    returned (it is the best available) — only the error string signals
    degradation so downstream checks annotate their output.

    Issue #1886 note: a 1900s served age is only reachable when a
    deployment widens ``runtime.status_snapshot_ttl_seconds`` past the
    900s default — under default config ``read_status_snapshot`` falls
    back to a live compute (``cache_age_seconds=None``) long before
    serving anything this old."""
    payload = _fleet_status_payload(cache_age_seconds=1900.0)
    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: _FakeStatusProc(stdout=payload))

    blocked, err = hb.get_blocked_issue_numbers(tmp_path)

    assert blocked == {"owner/repo": {42}}, "blocked data must still be returned"
    assert "stale" in err.lower(), f"expected staleness warning in err; got: {err!r}"
    assert "1900" in err, f"expected cache age in warning; got: {err!r}"
    # Issue #1886: the warning names the repo whose snapshot drove it.
    assert "owner/repo" in err, f"expected offending repo slug in warning; got: {err!r}"


def test_get_blocked_issue_numbers_no_warning_within_serving_ttl(
    hb: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1886: a served snapshot's age is bounded by the producer's own
    freshness contract — ``read_status_snapshot`` only serves a snapshot
    whose age is <= ``runtime.status_snapshot_ttl_seconds`` — so an age at
    the very top of that band is routine healthy refresh cadence, not
    degradation. The 881s age this issue observed firing on every repo,
    every tick sits in that band (default TTL 900s); the age is derived
    from the TTL itself so the test still pins the strongest servable bound
    if the default moves."""
    served_age = float(RuntimeConfig().status_snapshot_ttl_seconds)
    payload = _fleet_status_payload(cache_age_seconds=served_age)
    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: _FakeStatusProc(stdout=payload))

    blocked, err = hb.get_blocked_issue_numbers(tmp_path)

    assert blocked == {"owner/repo": {42}}
    assert err == "", f"served age inside the producer's serving TTL must not warn; got: {err!r}"


def test_get_blocked_issue_numbers_warns_names_worst_offender_repo(
    hb: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Issue #1886: when several repos serve over-threshold snapshots, the
    warning names the worst offender — the repo with the largest
    ``cache_age_seconds`` — not whichever stale repo the payload lists
    first. ``owner/repo-a`` (1900s) precedes ``owner/repo-b`` (2500s) in the
    payload, so a regression to 'first stale repo' picks repo-a and fails
    here."""
    payload = _fleet_status_payload_multi(
        repo_ages={"owner/repo-a": 1900.0, "owner/repo-b": 2500.0}
    )
    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: _FakeStatusProc(stdout=payload))

    blocked, err = hb.get_blocked_issue_numbers(tmp_path)

    assert blocked == {"owner/repo-a": {42}, "owner/repo-b": {42}}
    assert "2500" in err, f"expected worst-offender age in warning; got: {err!r}"
    assert "owner/repo-b" in err, f"expected worst-offender slug in warning; got: {err!r}"
    assert "owner/repo-a" not in err, f"warning must name only the worst offender; got: {err!r}"


def test_get_blocked_issue_numbers_no_warning_on_fresh_cache(
    hb: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fresh cache (age ≤ ``STATUS_CACHE_STALE_SECONDS``) must NOT produce a
    staleness warning — the error string is empty, same as a live response."""
    payload = _fleet_status_payload(cache_age_seconds=30.0)
    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: _FakeStatusProc(stdout=payload))

    blocked, err = hb.get_blocked_issue_numbers(tmp_path)

    assert blocked == {"owner/repo": {42}}
    assert err == "", f"fresh cache must not produce a warning; got: {err!r}"


def test_get_blocked_issue_numbers_no_warning_on_live_response(
    hb: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live response (``cache_age_seconds=None``) must NOT produce a
    staleness warning — there is no cache to be stale."""
    payload = _fleet_status_payload(cache_age_seconds=None)
    monkeypatch.setattr(hb.subprocess, "run", lambda *a, **k: _FakeStatusProc(stdout=payload))

    blocked, err = hb.get_blocked_issue_numbers(tmp_path)

    assert blocked == {"owner/repo": {42}}
    assert err == "", f"live response must not produce a warning; got: {err!r}"
