"""Issue #2096: headless claude-code workers must not background their suite.

Two layers: the launch env disables background tasks (enforced at the
harness), and a fast clean exit with an uncommitted tree and no commit is
classified ``worker_exited_with_background_work`` (detection backstop).
"""

from __future__ import annotations

import json
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import pytest

import _git_templates
from _claude_adapter_fixtures import _install_fake_create_worktree
from _dead_session_fixtures import _git, _make_classify_state
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import _wg
from _worktree_fixtures import _init_bare_remote_and_clone

from charlie_work.claude_code import ClaudeWorkerRecord, launch_claude_worker
from charlie_work.config import DispatchConfig, OrchestratorConfig, PostMortemConfig
from charlie_work.dead_worker_sweep.decide_dead_sessions import (
    BACKGROUND_EXIT_FAILURE_KIND,
    dead_fallback_kind,
    exited_with_background_work,
)

_ENV = "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"


@pytest.fixture(scope="module", autouse=True)
def _warm_ebc_git_template(
    tmp_path_factory: pytest.TempPathFactory,
) -> _git_templates.EnvKey | None:
    """HS-CW-4: charge the ``ebc`` git-template cold build to module setup.

    ``test_fake_worker_exit_zero_dirty_no_commit_emits_classification_event``
    is this module's ``_init_bare_remote_and_clone`` consumer (via
    ``_git_templates.bare_remote_and_clone``), so when the module runs first
    in a worker process -- as it does in the ledger's selected ``test`` runs --
    its ``call`` phase absorbed the template's once-per-process nine-process
    ``git`` boot instead of the ~40 ms copy; the ledger flagged the test at
    2.3x baseline (issue #2603). Materializing one scratch remote+clone pair
    here keeps the same transition in the ``setup`` phase column, where
    resource acquisition belongs; when the template is already warm the
    warmup is one additional ~40 ms copy.

    ``TemplateRegistry`` keys templates by ``(shape, env_fingerprint())``, and
    the fingerprint includes every ``GIT_*`` variable except
    ``GIT_CEILING_DIRECTORIES``. The function-scoped autouse
    ``_isolate_git_env`` (tests/conftest.py) sets the
    ``GIT_CONFIG_COUNT/KEY_0/VALUE_0`` trio *after* this module fixture runs,
    so a warm-up without the trio would build a template under a fingerprint
    the call phase never looks up again. This fixture applies
    ``_git_templates.GIT_CONFIG_ENV`` -- the same mapping conftest reads, so
    the two cannot drift -- through a ``pytest.MonkeyPatch`` context before
    building, then returns the fingerprint it warmed under so
    ``test_ebc_git_template_warmed_by_module_setup`` can pin it against the
    call-phase environment directly. Under ``CI_FLEET_TEST_REUSE=off`` there
    is no registry to warm and the pin test skips itself, so the fixture
    returns ``None`` without spending the build.
    """
    if not _git_templates.reuse_enabled():
        return None
    with pytest.MonkeyPatch.context() as mp:
        for key, value in _git_templates.GIT_CONFIG_ENV.items():
            mp.setenv(key, value)
        _init_bare_remote_and_clone(tmp_path_factory.mktemp("ebc-template-warmup"))
        return _git_templates.env_fingerprint()


@pytest.mark.skipif(
    not _git_templates.reuse_enabled(),
    reason="CI_FLEET_TEST_REUSE=off forces the fresh path; there is no template to pin",
)
def test_ebc_git_template_warmed_by_module_setup(
    _warm_ebc_git_template: _git_templates.EnvKey | None, tmp_path: Path
) -> None:
    """Pin the module warm-up to the fingerprint the call phase looks up.

    The fixture returns the fingerprint it warmed under; asserting it equals
    this test's own call-phase fingerprint fails on any drift between the two
    environments regardless of what other modules warmed in this worker. The
    probe below then drives the real consumer path and proves it copied the
    warmed template: ``materialized["ebc"]`` must increment while the
    registry's key set stays fixed -- a cold build would add an
    ``("ebc", env)`` key.
    """
    warmed_env = _warm_ebc_git_template
    call_env = _git_templates.env_fingerprint()
    assert warmed_env == call_env, (
        "module warm-up ran under a different env fingerprint than the call "
        "phase; its template can never be looked up"
    )
    registry = _git_templates._REGISTRY
    assert registry.lookup("ebc") is not None, (
        "no ebc template under the call-phase fingerprint; the warm-up took the fresh path"
    )
    keys_before = set(registry._templates)
    copies_before = registry.materialized["ebc"]
    _init_bare_remote_and_clone(tmp_path)
    assert registry.materialized["ebc"] == copies_before + 1
    assert set(registry._templates) == keys_before, (
        "the call phase built a new ebc template instead of copying the warmed one"
    )


def _probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None) -> str:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _install_fake_create_worktree(monkeypatch, tmp_path)
    script = tmp_path / "probe.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import os
            from pathlib import Path
            Path("probe.txt").write_text(os.environ.get("{_ENV}", "<unset>"), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    record = launch_claude_worker(
        96,
        "agent/issue-96-bg",
        "prompt",
        repo_root=repo_root,
        sessions_dir=tmp_path / "sessions",
        command_template=(sys.executable, str(script)),
        env=env,
    )
    assert record.ok
    probe = Path(record.worktree_path) / "probe.txt"
    return _wait(probe)


def _wait(probe: Path) -> str:
    import time

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if probe.exists() and probe.read_text(encoding="utf-8"):
            return probe.read_text(encoding="utf-8")
        time.sleep(0.05)
    raise AssertionError("worker never wrote probe")


def test_worker_env_disables_background_tasks_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert _probe(tmp_path, monkeypatch, None) == "1"


def test_operator_worker_env_overrides_background_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert _probe(tmp_path, monkeypatch, {_ENV: "0"}) == "0"


def test_exited_with_background_work_predicate() -> None:
    ok = {"exit_code": 0, "duration_seconds": 38.0}
    kw = {"adapter_kind": "claude-code", "dirty": True, "ahead_count": 0, "pid": 7}
    ok = {**ok, "pid": 7}
    assert exited_with_background_work(ok, **kw)
    assert not exited_with_background_work({**ok, "pid": 8}, **kw)  # stale prior attempt
    assert not exited_with_background_work({k: v for k, v in ok.items() if k != "pid"}, **kw)
    assert not exited_with_background_work(ok, **{**kw, "pid": None})
    assert not exited_with_background_work(None, **kw)
    assert not exited_with_background_work({**ok, "exit_code": 1}, **kw)
    assert not exited_with_background_work({**ok, "duration_seconds": 5000.0}, **kw)
    assert not exited_with_background_work(ok, **{**kw, "dirty": False})
    assert not exited_with_background_work(ok, **{**kw, "ahead_count": 1})
    assert not exited_with_background_work(ok, **{**kw, "adapter_kind": "devin"})


def test_dead_fallback_kind_background_exit() -> None:
    assert (
        dead_fallback_kind(is_completed=False, worktree_unknown=False, background_exit=True)
        == BACKGROUND_EXIT_FAILURE_KIND
    )
    assert dead_fallback_kind(is_completed=False, worktree_unknown=False) == "stalled"


def test_fake_worker_exit_zero_dirty_no_commit_emits_classification_event(
    tmp_path: Path,
) -> None:
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    _remote, repo_root = _init_bare_remote_and_clone(tmp_path)
    # The clone itself is the "worktree": no `git worktree add`, which fails on
    # Windows when the pytest tmp path is long.
    worktree_path, branch = repo_root, "agent/issue-40"
    (worktree_path / "test_config_precedence.py").write_text("x = 1", encoding="utf-8")
    sessions_dir, state_file = _make_classify_state(tmp_path)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    record = ClaudeWorkerRecord(
        issue_number=40,
        branch=branch,
        worktree_path=str(worktree_path),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "-p"),
        pid=424242,
        started_at=now,
        log_path=str(sessions_dir / "issue-40.claude.log"),
        error=None,
    )
    (sessions_dir / "issue-40.claude.json").write_text(
        json.dumps(record.to_dict()), encoding="utf-8"
    )
    (sessions_dir / "issue-40.claude.terminal.json").write_text(
        json.dumps({"pid": 424242, "exit_code": 0, "duration_seconds": 38.0}), encoding="utf-8"
    )
    assert _git(worktree_path, "status", "--porcelain").stdout.strip()

    # Keep the reap pass off real host state (issue #2603): the default
    # post_mortem db_path resolves to the host's sessions.db under %APPDATA%
    # (a host-sized table scan inside the call phase), the default fleet dir
    # resolves to %LOCALAPPDATA%\charlie-work, and an empty dispatch.base_ref
    # pays a four-spawn origin/HEAD resolve-and-heal chain in
    # inspect_worktree_state that the remote-tracking ref already answers.
    config = OrchestratorConfig(
        dispatch=DispatchConfig(base_ref="origin/main"),
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "no-such-sessions.db")),
    )
    gh = FakeGitHub(repo_root=repo_root)
    gh.issues = [
        {
            "number": 40,
            "title": "t",
            "url": "https://example.test/issues/40",
            "body": "",
            "labels": [{"name": config.labels.in_progress}],
            "state": "OPEN",
        }
    ]
    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        state_file,
        gh,
        config,
        write_gate=_wg(state_file),
        fleet_dir_override=str(tmp_path / "no-fleet-dir"),
    )

    state = json.loads(state_file.read_text(encoding="utf-8"))
    events = [e for e in state["events"] if e["kind"] == BACKGROUND_EXIT_FAILURE_KIND]
    assert len(events) == 1
    assert events[0]["payload"]["issue_number"] == 40
