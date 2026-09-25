"""Local-only repo (``local_issues.enabled``) tests for ``scripts/heartbeat_check.py``.

Issue #1861: a ``local_issues.enabled`` repo (e.g. ``local/mdls``) has no
GitHub remote, so every ``gh <issue|pr> list -R <slug>`` fails with
"Could not resolve to a Repository" -- five permanent spurious ANOMALY
lines per beat (dispatch-coverage, armable-backlog, review-liveness,
merge-flow, stale-open-issue-mentions). ``RepoInfo.local_issues_enabled``
is resolved from the layered config in ``load_repos`` and each gh-based
check emits a ``skipped: local-only repo`` OK line instead of calling gh.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from _heartbeat_check_fixtures import (
    _gh_dispatch,
    _load_heartbeat_check,
    _make_repo,
)


@pytest.fixture(scope="module")
def hb() -> ModuleType:
    return _load_heartbeat_check()


def _write_fleet_json(fleet_dir: Path, entries: dict[str, dict[str, str]]) -> None:
    """Write a fleet.json registering ``entries`` (slug -> repo fields)."""
    fleet_dir.mkdir(parents=True, exist_ok=True)
    (fleet_dir / "fleet.json").write_text(json.dumps({"repos": entries}), encoding="utf-8")


def _local_repo(hb: ModuleType, tmp_path: Path) -> Any:
    """A RepoInfo flagged local-only, as ``load_repos`` would produce it."""
    return replace(_make_repo(hb, tmp_path), local_issues_enabled=True)


def _no_gh(monkeypatch: Any, hb: ModuleType) -> list[list[str]]:
    """Fail the test if any gh call is attempted; records attempted args."""
    attempted: list[list[str]] = []

    def handler(args: list[str], cwd: Path) -> tuple[bool, Any, str]:
        attempted.append(list(args))
        raise AssertionError(f"gh must not be called for a local-only repo: {args}")

    _gh_dispatch(monkeypatch, hb, handler)
    return attempted


# ---------------------------------------------------------------------------
# load_repos: local_issues_enabled resolution
# ---------------------------------------------------------------------------


def test_load_repos_marks_local_issues_enabled_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    fleet_dir = tmp_path / "fleet"
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("local_issues:\n  enabled: true\n", encoding="utf-8")
    _write_fleet_json(
        fleet_dir,
        {
            "local/mdls": {
                "repo_root": str(tmp_path),
                "state_dir": str(tmp_path / "state"),
                "config_path": str(config_path),
            }
        },
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    repos, err = hb.load_repos()
    assert err is None
    assert len(repos) == 1
    assert repos[0].local_issues_enabled is True


def test_load_repos_defaults_to_github_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """No ``local_issues`` section anywhere -> GitHub-backed (flag False)."""
    fleet_dir = tmp_path / "fleet"
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("dispatch:\n  max_concurrent_sessions: 3\n", encoding="utf-8")
    _write_fleet_json(
        fleet_dir,
        {
            "owner/repo": {
                "repo_root": str(tmp_path),
                "state_dir": str(tmp_path / "state"),
                "config_path": str(config_path),
            }
        },
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    repos, err = hb.load_repos()
    assert err is None
    assert repos[0].local_issues_enabled is False


def test_load_repos_local_issues_enabled_via_fleet_layer(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """The knob can live only in <fleet_dir>/config.yaml (fleet_registry's
    documented case): a bare per-repo read would resolve False."""
    fleet_dir = tmp_path / "fleet"
    _write_fleet_json(
        fleet_dir,
        {
            "local/mdls": {
                "repo_root": str(tmp_path),
                "state_dir": str(tmp_path / "state"),
                "config_path": str(tmp_path / "orchestrator.config.yaml"),
            }
        },
    )
    (fleet_dir / "config.yaml").write_text("local_issues:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    repos, err = hb.load_repos()
    assert err is None
    assert repos[0].local_issues_enabled is True


def test_load_repos_repo_config_overrides_fleet_layer(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Layered-config precedence: an explicit per-repo ``enabled: false``
    wins over a fleet-layer ``enabled: true``."""
    fleet_dir = tmp_path / "fleet"
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("local_issues:\n  enabled: false\n", encoding="utf-8")
    _write_fleet_json(
        fleet_dir,
        {
            "owner/repo": {
                "repo_root": str(tmp_path),
                "state_dir": str(tmp_path / "state"),
                "config_path": str(config_path),
            }
        },
    )
    (fleet_dir / "config.yaml").write_text("local_issues:\n  enabled: true\n", encoding="utf-8")
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    repos, err = hb.load_repos()
    assert err is None
    assert repos[0].local_issues_enabled is False


def test_load_repos_unreadable_config_reads_as_github_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """A config that cannot be parsed cannot prove the repo is local --
    treat it as GitHub-backed (the fleet_registry posture)."""
    fleet_dir = tmp_path / "fleet"
    config_path = tmp_path / "orchestrator.config.yaml"
    config_path.write_text("local_issues: [unterminated\n", encoding="utf-8")
    _write_fleet_json(
        fleet_dir,
        {
            "owner/repo": {
                "repo_root": str(tmp_path),
                "state_dir": str(tmp_path / "state"),
                "config_path": str(config_path),
            }
        },
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))

    repos, err = hb.load_repos()
    assert err is None
    assert repos[0].local_issues_enabled is False


# ---------------------------------------------------------------------------
# gh-based checks: skip instead of calling gh on a local-only repo
# ---------------------------------------------------------------------------


def test_check_dispatch_coverage_skips_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _local_repo(hb, tmp_path)
    _no_gh(monkeypatch, hb)
    prev = {"dispatchable_issues": [7]}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_dispatch_coverage(
        report, repo, prev, new, skip_delta=False, blocked_numbers=None, blocked_err=""
    )
    assert not report.anomaly
    assert report.lines[0] == (f"OK dispatch-coverage {repo.slug}: {hb.LOCAL_ONLY_SKIP_DETAIL}")
    # The delta snapshot is carried forward untouched (nothing measured).
    assert new["dispatchable_issues"] == [7]
    # The two non-gh sub-checks still emit their own lines.
    assert any(line.startswith(f"OK dispatch-throttle {repo.slug}:") for line in report.lines)
    assert any(
        line.startswith(f"OK in-progress-stale {repo.slug}:")
        and "skipped: local-only repo" in line
        for line in report.lines
    )


def test_check_armable_backlog_skips_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _local_repo(hb, tmp_path)
    _no_gh(monkeypatch, hb)
    report = hb.Report()
    hb.check_armable_backlog(report, repo, None, "")
    assert not report.anomaly
    assert report.lines == [f"OK armable-backlog {repo.slug}: {hb.LOCAL_ONLY_SKIP_DETAIL}"]


def test_check_review_liveness_skips_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _local_repo(hb, tmp_path)
    # The prs dir must exist so the check reaches the gh call on the
    # un-gated path (a missing dir already short-circuits OK).
    (repo.state_dir / "prs").mkdir(parents=True)
    _no_gh(monkeypatch, hb)
    report = hb.Report()
    hb.check_review_liveness(report, repo)
    assert not report.anomaly
    assert report.lines == [f"OK review-liveness {repo.slug}: {hb.LOCAL_ONLY_SKIP_DETAIL}"]


def test_check_merge_flow_skips_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _local_repo(hb, tmp_path)
    _no_gh(monkeypatch, hb)
    prev = {
        "mergequeue_count": 2,
        "mergequeue_unchanged_streak": 1,
        "last_merged_at": "2020-01-01T00:00:00Z",
    }
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_merge_flow(report, repo, prev, new, skip_delta=False)
    assert not report.anomaly
    assert report.lines == [f"OK merge-flow {repo.slug}: {hb.LOCAL_ONLY_SKIP_DETAIL}"]
    assert new == prev


def test_check_stale_open_issue_mentions_skips_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _local_repo(hb, tmp_path)
    _no_gh(monkeypatch, hb)
    report = hb.Report()
    hb.check_stale_open_issue_mentions(report, repo)
    assert not report.anomaly
    assert report.lines == [
        f"OK stale-open-issue-mentions {repo.slug}: {hb.LOCAL_ONLY_SKIP_DETAIL}"
    ]


def test_check_in_progress_staleness_skips_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    repo = _local_repo(hb, tmp_path)
    _no_gh(monkeypatch, hb)
    prev = {"in_progress": {"5": "2026-01-01T00:00:00Z"}}
    new: dict[str, Any] = {}
    report = hb.Report()
    hb.check_in_progress_staleness(report, repo, [], prev, new, skip_delta=False)
    assert not report.anomaly
    assert report.lines == [f"OK in-progress-stale {repo.slug}: {hb.LOCAL_ONLY_SKIP_DETAIL}"]
    # The last real beat's snapshot is carried forward untouched.
    assert new["in_progress"] == prev["in_progress"]


# ---------------------------------------------------------------------------
# main() end-to-end: a registered local-only repo produces no gh call and
# no ANOMALY line
# ---------------------------------------------------------------------------


def test_main_emits_no_anomaly_or_gh_repo_call_for_local_repo(
    hb: ModuleType, monkeypatch: Any, tmp_path: Path
) -> None:
    """Full-beat regression: with ``local/mdls`` registered and every gh
    call failing the way the real CLI does (``Could not resolve to a
    Repository``), no ANOMALY line may mention the local slug and no
    ``-R local/mdls`` argv may be attempted."""
    fleet_dir = tmp_path / "fleet"
    repo_root = tmp_path / "mdls"
    state_dir = tmp_path / "mdls-state"
    config_path = tmp_path / "orchestrator.config.yaml"
    for d in (repo_root, state_dir):
        d.mkdir(parents=True, exist_ok=True)
    # A present-but-empty prs dir matches production (the state dir ships
    # one) and forces review-liveness past its no-dir early return so the
    # local-only gate is what produces its line.
    (state_dir / "prs").mkdir()
    config_path.write_text(
        "local_issues:\n  enabled: true\n  issues_dir: docs/issues\n", encoding="utf-8"
    )
    _write_fleet_json(
        fleet_dir,
        {
            "local/mdls": {
                "repo_root": str(repo_root),
                "state_dir": str(state_dir),
                "config_path": str(config_path),
            }
        },
    )
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(fleet_dir))
    monkeypatch.setenv("CHARLIE_WORK_HEARTBEAT_STATE", str(tmp_path / "hb-state.json"))
    monkeypatch.setenv(
        "CHARLIE_WORK_HEARTBEAT_SUPPRESSIONS", str(tmp_path / "no-suppressions.yaml")
    )

    fleet_status_payload = json.dumps({"data": {"repos": {"local/mdls": {"blocked": []}}}})

    gh_repo_calls: list[list[str]] = []

    class _FakeProc:
        def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(args, *a, **k):
        if args[:4] == ["charlie", "fleet", "status", "--json"]:
            return _FakeProc(returncode=0, stdout=fleet_status_payload, stderr="")
        if args and args[0] == "gh":
            if "-R" in args:
                # Any -R <slug> call means a check ignored the local-only
                # gate; record it for the assertion below.
                gh_repo_calls.append(list(args))
                return _FakeProc(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "GraphQL: Could not resolve to a Repository with the "
                        "name 'local/mdls'. (repository)"
                    ),
                )
            # Non-repo-scoped gh (check_github_rate's `gh api rate_limit`)
            # still runs -- it is not repo-bound.
            return _FakeProc(
                returncode=0,
                stdout=json.dumps({"resources": {"graphql": {"remaining": 5000}}}),
                stderr="",
            )
        # git log / schtasks / everything else fails benignly.
        return _FakeProc(returncode=1, stdout="", stderr="fake subprocess disabled in test")

    monkeypatch.setattr(hb.subprocess, "run", fake_run)

    captured = io.StringIO()
    monkeypatch.setattr(hb.sys, "stdout", captured)
    hb.main()
    output = captured.getvalue()

    assert gh_repo_calls == [], (
        f"gh -R must never be attempted for a local-only repo; attempted: {gh_repo_calls}"
    )
    lines = output.splitlines()
    for check in (
        "dispatch-coverage",
        "armable-backlog",
        "review-liveness",
        "merge-flow",
        "stale-open-issue-mentions",
        "in-progress-stale",
    ):
        assert any(
            line == f"OK {check} local/mdls: {hb.LOCAL_ONLY_SKIP_DETAIL}" for line in lines
        ), f"missing skip line for {check} in:\n{output}"
        assert not any(line.startswith(f"ANOMALY {check} local/mdls:") for line in lines), (
            f"ANOMALY line for {check} in:\n{output}"
        )
    # The non-gh sub-check under dispatch-coverage still reports its real
    # state (its contract is that the line always prints).
    assert any(line.startswith("OK dispatch-throttle local/mdls:") for line in lines), (
        f"missing dispatch-throttle line in:\n{output}"
    )
