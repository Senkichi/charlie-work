"""Tests for the #1001 worker-GitHub-token gate — now retired by issue #1853.

Historical context (issue #1001): workers were dispatched without a
sanctioned GitHub credential (``sanitize_env`` strips GH_TOKEN/GITHUB_TOKEN
per issue #502, and no ``worker_env`` token was configured), so a doctor
preflight check and a dispatch gate warned/escalated — optionally deferring
dispatch under ``dispatch.require_worker_github_token``. Issue #1224 tracked
provisioning scoped worker PATs.

Issue #1853 retired all of it: the operator decided workers stay
credential-free by design, and ``.worker-outcome.json`` is the single
channel for PR changes — the authenticated orchestrator applies the edits
(see ``src/charlie_work/rework_outcome.py``). A missing worker token is the
intended state, not a defect.

What this file now pins:

1. Dispatch never emits ``worker_token_missing`` and never defers on a
   missing worker token — even with the deprecated
   ``dispatch.require_worker_github_token`` flag still set.
2. The retired config key still parses (backward compatibility) but must
   remain a bool.
3. ``sanitize_env``'s stripping behaviour is UNCHANGED — the security
   control the gate once warned about still stands on its own.
4. The retired predicate machinery (``worker_github_token_findings`` /
   ``WorkerTokenFinding`` / ``worker_github_token_findings_if_publishing``)
   is gone, not dead code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import (
    AutoMergeConfig,
    ConfigError,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
    build_config_from_data,
)
from charlie_work.env_sanitize import (
    STRIPPED_GH_TOKEN_VARS,
    sanitize_env,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state

from _fakes_github import FakeGitHub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    adapter: str = "devin-shell",
    devin_worker_env: dict[str, str] | None = None,
    require_token: bool = False,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig for the retirement tests."""
    return OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            sessions_dir="sessions",
            worker_env=devin_worker_env or {},
        ),
        worker=WorkerRoleConfig(harness=adapter),
        dispatch=DispatchConfig(
            require_worker_github_token=require_token,
        ),
    )


def _events_of_kind(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


# ---------------------------------------------------------------------------
# 1. The gate is retired: dispatch proceeds, no escalation, no deferral
# ---------------------------------------------------------------------------


def test_dispatch_proceeds_with_no_token_and_no_escalation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No worker token + default flag -> dispatch proceeds silently.

    The retired gate emitted ``worker_token_missing`` once per standing
    condition; post-#1853 it must not fire at all. ``dispatch_sessions`` is
    stubbed so no real worker is launched.
    """
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.workflow import OrchestratorApp

    config = _config(adapter="devin-shell")
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    def _fake_dispatch_sessions(repo_root, manifest_path, results_path, settings, requests):
        return [
            SessionDispatchResult(
                issue_number=req.issue_number,
                issue_title=req.issue_title,
                prompt_path=req.prompt_path,
                branch_name=req.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=999,
            )
            for req in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake_dispatch_sessions)

    fake_gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data.get("deferred_reason") != "worker_token_missing"
    state = load_state(paths.state_file)
    assert _events_of_kind(state, "worker_token_missing") == []
    assert state.get("worker_token_escalated") in (None, False)


def test_deprecated_require_flag_no_longer_defers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``require_worker_github_token=True`` is a deprecated no-op (#1853):
    dispatch must proceed with no token and emit no ``worker_token_missing``
    event — a config file carrying the old flag cannot accidentally re-arm
    the gate."""
    from charlie_work.adapters import SessionDispatchResult
    from charlie_work.workflow import OrchestratorApp

    config = _config(adapter="devin-shell", require_token=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=False)

    def _fake_dispatch_sessions(repo_root, manifest_path, results_path, settings, requests):
        return [
            SessionDispatchResult(
                issue_number=req.issue_number,
                issue_title=req.issue_title,
                prompt_path=req.prompt_path,
                branch_name=req.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=999,
            )
            for req in requests
        ]

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake_dispatch_sessions)

    fake_gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)

    assert result.ok is True
    assert result.data["selected_count"] == 1
    assert result.data.get("deferred_reason") is None or (
        result.data["deferred_reason"] != "worker_token_missing"
    )
    state = load_state(paths.state_file)
    assert _events_of_kind(state, "worker_token_missing") == []


# ---------------------------------------------------------------------------
# 2. Config backward compatibility: the retired key still parses
# ---------------------------------------------------------------------------


def test_config_with_retired_key_still_loads() -> None:
    """A config file that still sets ``dispatch.require_worker_github_token``
    must parse unchanged (issue #1853 acceptance criterion)."""
    config = build_config_from_data({"dispatch": {"require_worker_github_token": True}})
    assert config.dispatch.require_worker_github_token is True


def test_config_retired_key_must_still_be_bool() -> None:
    """Backward-compat parsing still validates the type."""
    with pytest.raises(ConfigError, match="require_worker_github_token.*must be a bool"):
        build_config_from_data({"dispatch": {"require_worker_github_token": "true"}})


# ---------------------------------------------------------------------------
# 3. sanitize_env stripping behaviour unchanged
# ---------------------------------------------------------------------------


def test_sanitize_env_still_strips_all_token_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sanitize_env must still strip every GH token variable — the security
    control (issue #502) is what makes workers credential-free; only the
    warning/gate built on top of it was retired."""
    worktree_path = tmp_path / "worktree"
    worktree_path.mkdir()

    for var in STRIPPED_GH_TOKEN_VARS:
        monkeypatch.setenv(var, f"secret-value-for-{var}")

    env = sanitize_env(worktree_path)

    for var in STRIPPED_GH_TOKEN_VARS:
        assert var not in env, f"{var} must be stripped by sanitize_env"


def test_sanitize_env_uses_shared_constant() -> None:
    """The strip loop in sanitize_env must use STRIPPED_GH_TOKEN_VARS, not an
    inline tuple — so the constant and the strip loop cannot drift.
    """
    import inspect

    src = inspect.getsource(sanitize_env)
    assert "STRIPPED_GH_TOKEN_VARS" in src, (
        "sanitize_env must use the STRIPPED_GH_TOKEN_VARS constant, not an inline tuple"
    )


# ---------------------------------------------------------------------------
# 4. Retired machinery is gone, not dead code
# ---------------------------------------------------------------------------


def test_token_gate_predicate_is_removed() -> None:
    """The #1001 predicate and its gated wrapper must not survive as dead
    code (issue #1853 retirement)."""
    import charlie_work.env_sanitize as env_sanitize_mod
    import charlie_work.local_work_park as lwp_mod

    assert not hasattr(env_sanitize_mod, "worker_github_token_findings")
    assert not hasattr(env_sanitize_mod, "WorkerTokenFinding")
    assert not hasattr(lwp_mod, "worker_github_token_findings_if_publishing")


def test_dispatch_impl_has_no_token_gate() -> None:
    """``_dispatch_impl`` must contain no token-gate remnants."""
    import inspect

    import charlie_work.orchestration.dispatch_state as dispatch_state_mod

    src = inspect.getsource(dispatch_state_mod._dispatch_impl)
    assert "worker_github_token" not in src
    assert "worker_token" not in src


def test_doctor_has_no_token_check() -> None:
    """``doctor.py`` must contain no ``_check_worker_github_token`` remnant."""
    import charlie_work.doctor as doctor_mod

    assert not hasattr(doctor_mod, "_check_worker_github_token")
