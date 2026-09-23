"""``run_doctor`` against a no-pull-requests backend (issue #1706).

A ``local_issues.enabled`` repo has no GitHub remote: ``gh auth status``,
``gh label list``, and the ``--live`` field probes are all unanswerable by
construction, so doctor must report them not-applicable rather than fail a
healthy repo. The backend discriminator is the capability probe
``local_work_park.publishes_pull_requests(gh)`` -- the same one
``dead_worker_reap`` already uses -- so the ``LocalFileGitHub`` below is a
real client, and the positive controls use ``FakeDoctorGitHub`` doubles that
keep the default ``publishes_pull_requests=True``.

Sibling doctor-test seam: fixtures come from ``tests/_doctor_fixtures.py``;
the small git/issue-file helpers are inlined here per this repo's convention
of keeping each new test file self-contained (see test_local_review_ready.py).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from charlie_work.config import AutoMergeConfig, LocalIssuesConfig, OrchestratorConfig
from charlie_work.doctor import run_doctor
from charlie_work.github import GitHubError
from charlie_work.local_issues import LocalFileGitHub
from charlie_work.paths import runtime_paths
from _doctor_fixtures import FakeDoctorGitHub, _config


def _init_repo(repo_root: Path) -> None:
    """A fresh git repo with one commit and NO origin remote."""
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "--allow-empty",
            "-m",
            "chore: seed",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _write_issue(issues_dir: Path, number: int, *, slug: str = "issue") -> Path:
    issues_dir.mkdir(parents=True, exist_ok=True)
    path = issues_dir / f"{number:03d}_{slug}.md"
    path.write_text(
        f'---\ntitle: "issue {number}"\nstate: open\nlabels: []\n---\nBody.\n',
        encoding="utf-8",
    )
    return path


def _local_config(**kwargs):
    """A config for a healthy local-file repo.

    ``auto_merge`` off is the honest setting for a no-remote repo (there is
    no merge gate to satisfy); ``review_dispatch`` stays on from ``_config``
    -- irrelevant either way once the check is capability-gated, but a real
    local config is free to leave the flag set.
    """
    kwargs.setdefault("auto_merge", AutoMergeConfig(enabled=False))
    kwargs.setdefault("local_issues", LocalIssuesConfig(enabled=True))
    return _config(**kwargs)


def _local_gh(repo_root: Path, issues_dir: Path) -> LocalFileGitHub:
    return LocalFileGitHub(repo_root=repo_root, issues_dir=issues_dir)


def _by_name(checks):
    return {check.name: check for check in checks}


def test_doctor_passes_on_healthy_local_file_repo(tmp_path: Path) -> None:
    """Acceptance (issue #1706): a correctly configured local repo is green.

    Every gh-shaped check reports not-applicable instead of failing, and the
    local-specific checks pass on a real LocalFileGitHub over a valid
    issues dir.
    """
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    _write_issue(issues_dir, 2, slug="second")
    config = _local_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert ok is True, [(c.name, c.detail) for c in checks if not c.ok and c.severity == "error"]
    # The gh-shaped checks report not-applicable rather than failing.
    assert by_name["gh on PATH"].ok is True
    assert "not applicable" in by_name["gh on PATH"].detail
    assert by_name["gh auth"].ok is True
    assert by_name["github labels"].ok is True
    assert "not applicable" in by_name["github labels"].detail
    assert by_name["worker GitHub token"].ok is True
    assert "not applicable" in by_name["worker GitHub token"].detail
    # And the local-specific checks ran and passed.
    assert by_name["local issues dir"].ok is True
    assert "2 issue file(s)" in by_name["local issues dir"].detail
    assert by_name["local issue files"].ok is True
    assert by_name["state dir gitignored"].ok is True


def test_doctor_stock_default_config_on_local_backend(tmp_path: Path) -> None:
    """Acceptance (issue #1706): a STOCK ``OrchestratorConfig()`` stays green.

    The healthy-repo test above runs ``_local_config()``, which overrides the
    two knobs this scenario is about (``auto_merge`` off, ``review_dispatch``
    on). A real first-time local repo runs on package defaults instead:
    ``auto_merge.enabled=True`` with empty ``required_checks`` and
    ``review_dispatch.enabled=False``/``rescue.enabled=False`` — the exact
    inputs that make ``required checks configured`` and
    ``review-to-verdict path`` fail on a PR-publishing backend. On a
    ``LocalFileGitHub`` both must report not-applicable, or a healthy local
    repo on stock defaults exits non-zero — the original #1706 bug.
    """
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    # Deliberately NOT _local_config()/_config(): the dataclass defaults are
    # the scenario under test. Pin them so a default change makes this test
    # say so rather than silently covering a different config.
    config = OrchestratorConfig()
    assert config.auto_merge.enabled is True
    assert config.auto_merge.required_checks == ()
    assert config.review_dispatch.enabled is False
    assert config.rescue.enabled is False
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["required checks configured"].ok is True
    assert by_name["required checks configured"].severity == "warning"
    assert "not applicable" in by_name["required checks configured"].detail
    assert by_name["review-to-verdict path"].ok is True
    assert by_name["review-to-verdict path"].severity == "warning"
    assert "not applicable" in by_name["review-to-verdict path"].detail
    assert ok is True, [(c.name, c.detail) for c in checks if not c.ok and c.severity == "error"]


def test_doctor_gh_auth_failure_still_fails_on_publishing_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: a real-client double with failing `auth status` fails.

    Proves the local-backend skip is capability-driven, not unconditional:
    a backend that publishes PRs (the default for every double) still runs
    the gh-shaped checks and still fails on a broken `gh auth`.
    """
    _init_repo(tmp_path)

    class _AuthFailingGitHub(FakeDoctorGitHub):
        def run(self, args, **kwargs):
            raise GitHubError("gh auth status: not logged in")

    # `gh` must resolve on PATH for the auth probe to run at all; fake the
    # resolution so the test does not depend on the host having gh installed.
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/gh" if name == "gh" else None)
    config = _config(auto_merge=AutoMergeConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _AuthFailingGitHub(labels=config.labels.all)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["gh auth"].ok is False
    assert "not logged in" in by_name["gh auth"].detail
    assert ok is False


def test_doctor_live_skips_gh_field_probes_on_local_backend(tmp_path: Path) -> None:
    """``--live`` must not call ``gh.run`` on a backend where it raises."""
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    config = _local_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        live=True,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["gh field lists"].ok is True
    assert "skipped" in by_name["gh field lists"].detail
    assert not any(name.startswith("gh field list:") for name in by_name)
    assert ok is True


def test_doctor_missing_issues_dir_is_blocking(tmp_path: Path) -> None:
    """A missing issues dir is the local backend's `gh outage`: blocking."""
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    config = _local_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, tmp_path / "docs" / "issues")  # never created

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["local issues dir"].ok is False
    assert by_name["local issues dir"].severity == "error"
    assert ok is False


def test_doctor_surfaces_issue_file_scan_problems_as_warning(tmp_path: Path) -> None:
    """Bad frontmatter / duplicate numbers drop issues from dispatch silently;
    doctor surfaces them as a warning (the backend itself warns-and-continues).
    """
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    (issues_dir / "002_broken.md").write_text(
        "---\nstate: [unclosed\n---\nbody\n", encoding="utf-8"
    )
    config = _local_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["local issue files"].ok is False
    assert by_name["local issue files"].severity == "warning"
    assert "002_broken.md" in by_name["local issue files"].detail
    assert ok is True  # warnings do not block


def test_doctor_flags_state_dir_not_gitignored(tmp_path: Path) -> None:
    """An un-ignored state dir pollutes the consumer repo's `git status` and
    is committable by a worker `git add -A` -- a blocking finding."""
    _init_repo(tmp_path)
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    config = _local_config()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["state dir gitignored"].ok is False
    assert by_name["state dir gitignored"].severity == "error"
    assert ".var" in by_name["state dir gitignored"].detail
    assert ok is False


def test_doctor_state_dir_outside_repo_is_not_applicable(tmp_path: Path) -> None:
    """An absolute runtime.state_dir outside the repo has nothing to ignore."""
    _init_repo(tmp_path)
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    config = _local_config()
    outside = tmp_path.parent / "outside-state"
    paths = runtime_paths(tmp_path, str(outside))
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["state dir gitignored"].ok is True
    assert "outside the repo" in by_name["state dir gitignored"].detail
    assert ok is True


def test_doctor_local_merge_queue_checks_when_enabled(tmp_path: Path) -> None:
    """The forward-declared local_merge_queue section lights up its checks:
    main checkout on a branch, verify command binaries on PATH."""
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    config = _local_config()
    # The config section does not exist on OrchestratorConfig yet; inject the
    # documented shape so the getattr-guarded checks are exercised.
    object.__setattr__(
        config,
        "local_merge_queue",
        SimpleNamespace(
            enabled=True,
            verify_commands=(("definitely-not-a-real-binary-1706", "--check"),),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["local merge queue branch"].ok is True  # unborn-or-main HEAD
    verify = by_name["local merge queue verify commands"]
    assert verify.ok is False
    assert "definitely-not-a-real-binary-1706" in verify.detail
    assert ok is False


def test_doctor_local_merge_queue_detached_head_fails(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".var/\n", encoding="utf-8")
    subprocess.run(
        ["git", "checkout", "--detach", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    issues_dir = tmp_path / "docs" / "issues"
    _write_issue(issues_dir, 1)
    config = _local_config()
    object.__setattr__(
        config,
        "local_merge_queue",
        SimpleNamespace(enabled=True, verify_commands=()),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = _local_gh(tmp_path, issues_dir)

    ok, checks = run_doctor(
        tmp_path,
        paths,
        config,
        tmp_path / "orchestrator.config.yaml",
        gh,
        fleet_dir_override=str(tmp_path / "fleet"),
    )

    by_name = _by_name(checks)
    assert by_name["local merge queue branch"].ok is False
    assert by_name["local merge queue verify commands"].ok is True  # empty is satisfiable
    assert ok is False
