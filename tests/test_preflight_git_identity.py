"""Tests for the ``git_identity`` preflight check (issue #1950).

A repo-local ``user.email``/``user.name`` that differs from the global
identity poisons every commit in every worktree of the checkout -- the live
``.git/config`` gained ``user.email = t@t`` and two days of worker commits
carried it. The check refuses the pass until the override is unset (fatal
by default) so no misattributed commit is minted.

Split out of ``test_preflight.py`` to keep that module under the 800-line
file-size cap; ``_paths``/``_disk_usage_free`` are duplicated deliberately
so both files stay self-contained.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from charlie_work.config import PreflightConfig
from charlie_work.instrumentation import close_db, query_events
from charlie_work.preflight import (
    GitIdentityProbe,
    PreflightCheck,
    PreflightPaths,
    emit_preflight_refusal,
    run_preflight,
)


@pytest.fixture(autouse=True)
def _close_db_after_test(tmp_path: Path) -> None:
    yield
    close_db(tmp_path / "state.json")


def _paths(tmp_path: Path) -> PreflightPaths:
    repo_root = tmp_path / "repo"
    state_dir = repo_root / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    (repo_root / ".venv" / "Scripts").mkdir(parents=True)
    return PreflightPaths(repo_root=repo_root, state_dir=state_dir)


def _disk_usage_free(free_bytes: int):
    def _fn(anchor: str) -> SimpleNamespace:
        return SimpleNamespace(total=1_000_000_000_000, used=0, free=free_bytes)

    return _fn


def _fixed_probe(probe: GitIdentityProbe):
    """Return a ``git_identity_probe`` callable yielding *probe*."""
    return lambda _root: probe


def _git_identity_check(paths: PreflightPaths, probe: GitIdentityProbe, cfg: PreflightConfig):
    result = run_preflight(
        paths,
        cfg,
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
        git_identity_probe=_fixed_probe(probe),
    )
    return result, next(c for c in result.checks if c.name == "git_identity")


def test_git_identity_ok_when_no_repo_local_override(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(global_email="op@example.com", global_name="Operator"),
        PreflightConfig(),
    )
    assert check.ok is True
    assert check.fatal is True
    assert result.ok is True


def test_git_identity_fails_on_divergent_repo_local_email(tmp_path: Path) -> None:
    """The incident shape: a local `user.email` shadowing the global identity."""
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(
            local_email="t@t",
            global_email="op@users.noreply.github.com",
            global_name="Operator",
        ),
        PreflightConfig(),
    )
    assert check.ok is False
    assert check.fatal is True
    assert result.ok is False
    assert result.fatal_failures == (check,)
    # Detail must name the key and both values (operator-actionable).
    assert "user.email" in check.detail
    assert "t@t" in check.detail
    assert "op@users.noreply.github.com" in check.detail


def test_git_identity_fails_on_divergent_repo_local_name(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(
            global_email="op@example.com",
            global_name="Operator",
            local_name="t",
        ),
        PreflightConfig(),
    )
    assert check.ok is False
    assert check.fatal is True
    assert result.ok is False
    assert "user.name" in check.detail


def test_git_identity_fails_when_local_set_but_global_unset(tmp_path: Path) -> None:
    """A repo-local identity with no global to fall back to is the same
    drift: commits carry the local pin, not the operator's identity."""
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(local_email="t@t"),
        PreflightConfig(),
    )
    assert check.ok is False
    assert check.fatal is True
    assert result.ok is False
    assert "<unset>" in check.detail


def test_git_identity_ok_when_local_override_matches_global(tmp_path: Path) -> None:
    """A redundant-but-equal local restatement is not the drift: commits
    carry the same identity either way."""
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(
            local_email="op@example.com",
            local_name="Operator",
            global_email="op@example.com",
            global_name="Operator",
        ),
        PreflightConfig(),
    )
    assert check.ok is True
    assert result.ok is True


def test_git_identity_probe_error_is_inert(tmp_path: Path) -> None:
    """A probe failure (not a git repo, missing git binary) must not refuse
    the pass -- the drift this check catches is durable and will still be
    there on the next measurable pass."""
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(error="git config read failed"),
        PreflightConfig(),
    )
    assert check.ok is True
    assert "skipped" in check.detail
    assert result.ok is True


def test_git_identity_respects_config_fatal_override(tmp_path: Path) -> None:
    """`runtime.preflight.git_identity_fatal: false` downgrades the refusal
    to a tripwire event while the pass proceeds."""
    paths = _paths(tmp_path)
    result, check = _git_identity_check(
        paths,
        GitIdentityProbe(local_email="t@t", global_email="op@example.com"),
        PreflightConfig(git_identity_fatal=False),
    )
    assert check.ok is False
    assert check.fatal is False
    assert result.ok is True
    assert result.non_fatal_failures == (check,)


def test_git_identity_refusal_emits_loop_refused_preflight_event(tmp_path: Path) -> None:
    """The triage's "emits an event" arm: a fatal git_identity failure goes
    through emit_preflight_refusal -> the `loop_refused_preflight` event."""
    state_path = tmp_path / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    check = PreflightCheck(
        name="git_identity",
        ok=False,
        detail="user.email: local 't@t' vs global 'op@example.com'",
        fatal=True,
    )

    emit_preflight_refusal(state_path, check)

    events = query_events(state_path, kind="loop_refused_preflight")
    assert len(events) == 1
    assert events[0]["payload"]["check"] == "git_identity"
    assert "t@t" in events[0]["payload"]["detail"]
    assert events[0]["level"] == "error"


# --- real-git acceptance shape (issue #1950 triage AC) ---------------------


def _git(repo: Path, *args: str) -> None:
    """Run a git command in *repo*, raising on failure."""
    subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _init_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "realrepo"
    repo.mkdir()
    _git(repo, "init")
    return repo


def _pin_global_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the probe's ``--global`` reads at a fixture file so the
    comparison never depends on the host's real ~/.gitconfig."""
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text(
        "[user]\n\tname = Test Operator\n\temail = operator@example.test\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


def _git_repo_paths(repo: Path) -> PreflightPaths:
    state_dir = repo / ".var" / "charlie-work"
    state_dir.mkdir(parents=True)
    return PreflightPaths(repo_root=repo, state_dir=state_dir)


def test_git_identity_fails_on_real_repo_with_local_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: a tmp repo carrying `git config user.email t@t` fails."""
    _pin_global_identity(tmp_path, monkeypatch)
    repo = _init_git_repo(tmp_path)
    _git(repo, "config", "user.email", "t@t")  # writes the --local scope
    paths = _git_repo_paths(repo)

    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(repo / ".venv" / "Scripts" / "python.exe"),
        package_file=str(repo / "src" / "charlie_work" / "preflight.py"),
    )

    check = next(c for c in result.checks if c.name == "git_identity")
    assert check.ok is False
    assert check.fatal is True
    assert result.ok is False
    assert "t@t" in check.detail
    assert "operator@example.test" in check.detail


def test_git_identity_passes_on_real_repo_without_local_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: a tmp repo with no repo-local override passes."""
    _pin_global_identity(tmp_path, monkeypatch)
    repo = _init_git_repo(tmp_path)
    paths = _git_repo_paths(repo)

    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(repo / ".venv" / "Scripts" / "python.exe"),
        package_file=str(repo / "src" / "charlie_work" / "preflight.py"),
    )

    check = next(c for c in result.checks if c.name == "git_identity")
    assert check.ok is True
    assert result.ok is True


def test_git_identity_ok_on_non_git_directory(tmp_path: Path) -> None:
    """The default probe fails inert when repo_root is not a git checkout
    (git exits non-1 nonzero) -- no false refusal from a guard that cannot
    measure."""
    paths = _paths(tmp_path)  # repo_root is a plain dir, no .git
    result = run_preflight(
        paths,
        PreflightConfig(),
        disk_usage=_disk_usage_free(50 * 1024**3),
        sys_executable=str(paths.venv_dir / "Scripts" / "python.exe"),
        package_file=str(paths.repo_root / "src" / "charlie_work" / "preflight.py"),
    )
    check = next(c for c in result.checks if c.name == "git_identity")
    assert check.ok is True
    assert "skipped" in check.detail
