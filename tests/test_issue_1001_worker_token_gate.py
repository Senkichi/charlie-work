"""Tests for issue #1001: dispatch gate for missing worker GitHub token.

Workers are dispatched without a sanctioned GitHub credential (sanitize_env
strips GH_TOKEN/GITHUB_TOKEN per issue #502, and no worker_env token is
configured), so every worker fails ``gh pr create``, exits cleanly, and sits
undetected for ~45 minutes until a staleness watchdog reaps it. The
orchestrator then salvages the pushed branch into a PR itself — a fallback
running as the primary path.

The fix (issue #1001):
1. Extract the predicate from ``doctor._check_worker_github_token`` into
   ``env_sanitize.worker_github_token_findings`` — one function shared by
   both the doctor preflight and the dispatch gate.
2. Gate dispatch on the predicate: warn-only by default (escalate once,
   dispatch anyway), hard refusal behind
   ``dispatch.require_worker_github_token`` (defaults False).
3. The escalation fires once for a standing condition, not once per loop pass.
4. ``sanitize_env``'s stripping behaviour is unchanged.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from charlie_work.config import (
    AutoMergeConfig,
    ClaudeCodeConfig,
    DevinConfig,
    DispatchConfig,
    OrchestratorConfig,
    RescueConfig,
)
from charlie_work.env_sanitize import (
    STRIPPED_GH_TOKEN_VARS,
    WorkerTokenFinding,
    sanitize_env,
    worker_github_token_findings,
)
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state

from test_charlie_work import FakeGitHub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(
    *,
    adapter: str = "devin-shell",
    devin_worker_env: dict[str, str] | None = None,
    claude_worker_env: dict[str, str] | None = None,
    require_token: bool = False,
    rescue_enabled: bool = False,
    api_worker_enabled: bool = False,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig for token-gate tests."""
    from charlie_work.config import ApiBudgetConfig, ApiProviderConfig, ApiWorkerConfig

    api_worker = ApiWorkerConfig()
    if api_worker_enabled:
        provider_name = "kimi-k3"
        api_worker = ApiWorkerConfig(
            enabled=True,
            provider=provider_name,
            providers={
                provider_name: ApiProviderConfig(
                    base_url="https://api.moonshot.ai/anthropic",
                    api_key_env="MOONSHOT_API_KEY",
                    model="kimi-k3",
                    input_usd_per_mtok=3.0,
                    output_usd_per_mtok=15.0,
                    cached_input_usd_per_mtok=0.30,
                )
            },
            budget=ApiBudgetConfig(),
        )

    return OrchestratorConfig(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        devin=DevinConfig(
            adapter=adapter,
            sessions_dir="sessions",
            worker_env=devin_worker_env or {},
        ),
        claude_code=ClaudeCodeConfig(
            worker_env=claude_worker_env or {},
        ),
        dispatch=DispatchConfig(
            require_worker_github_token=require_token,
        ),
        rescue=RescueConfig(enabled=rescue_enabled),
        api_worker=api_worker,
    )


def _events_of_kind(state: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("kind") == kind]


# ---------------------------------------------------------------------------
# 1. Shared predicate: doctor and dispatch gate use one function
# ---------------------------------------------------------------------------


def test_doctor_and_dispatch_gate_share_one_predicate() -> None:
    """The doctor check and the dispatch gate must call the same function.

    This test would fail if either grew its own copy of the predicate —
    the issue #1001 acceptance criterion: "asserted by a test that would
    fail if either grew its own copy."
    """
    import inspect

    import charlie_work.doctor as doctor_mod
    import charlie_work.workflow as workflow_mod

    # doctor._check_worker_github_token must delegate to
    # env_sanitize.worker_github_token_findings (not inline its own copy).
    doctor_src = inspect.getsource(doctor_mod._check_worker_github_token)
    assert "worker_github_token_findings" in doctor_src, (
        "doctor._check_worker_github_token must call "
        "env_sanitize.worker_github_token_findings, not inline its own copy"
    )
    # The pre-#1001 private constant must be gone.
    assert not hasattr(doctor_mod, "_STRIPPED_GH_TOKEN_VARS"), (
        "doctor.py must not maintain a private _STRIPPED_GH_TOKEN_VARS copy — "
        "the shared STRIPPED_GH_TOKEN_VARS in env_sanitize.py is the single "
        "source of truth (issue #1001)"
    )

    # workflow._dispatch_impl must call the same function.
    dispatch_src = inspect.getsource(workflow_mod.OrchestratorApp._dispatch_impl)
    assert "worker_github_token_findings" in dispatch_src, (
        "workflow._dispatch_impl must call env_sanitize.worker_github_token_findings, "
        "not inline its own copy"
    )


def test_predicate_returns_findings_for_devin_shell_missing() -> None:
    config = _config(adapter="devin-shell")
    findings = worker_github_token_findings(config)
    assert len(findings) == 1
    assert findings[0].name == "worker GitHub token"
    assert findings[0].context == "devin-shell"
    assert findings[0].config_key == "devin.worker_env"
    assert findings[0].ok is False
    assert findings[0].configured_var is None


def test_predicate_returns_findings_for_devin_shell_ok() -> None:
    config = _config(
        adapter="devin-shell",
        devin_worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
    )
    findings = worker_github_token_findings(config)
    assert len(findings) == 1
    assert findings[0].ok is True
    assert findings[0].configured_var == "GH_TOKEN"


def test_predicate_returns_findings_for_claude_code_ok() -> None:
    config = _config(
        adapter="claude-code",
        claude_worker_env={"GITHUB_TOKEN": "placeholder-not-a-real-token"},
    )
    findings = worker_github_token_findings(config)
    assert len(findings) == 1
    assert findings[0].config_key == "claude_code.worker_env"
    assert findings[0].ok is True
    assert findings[0].configured_var == "GITHUB_TOKEN"


def test_predicate_omits_manual_and_command_adapters() -> None:
    for adapter in ("manual", "command"):
        config = _config(adapter=adapter)
        findings = worker_github_token_findings(config)
        assert findings == [], adapter


def test_predicate_fires_claude_code_routed_when_rescue_enabled() -> None:
    config = _config(
        adapter="devin-shell",
        devin_worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        rescue_enabled=True,
    )
    findings = worker_github_token_findings(config)
    assert len(findings) == 2
    assert findings[0].ok is True  # devin-shell path
    assert findings[1].name == "worker GitHub token (claude-code-routed)"
    assert findings[1].config_key == "claude_code.worker_env"
    assert findings[1].ok is False  # claude_code.worker_env has no token


def test_predicate_fires_claude_code_routed_when_api_worker_enabled() -> None:
    config = _config(
        adapter="devin-shell",
        devin_worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
        api_worker_enabled=True,
    )
    findings = worker_github_token_findings(config)
    assert len(findings) == 2
    assert findings[0].ok is True
    assert findings[1].ok is False


def test_predicate_detail_names_remediation_no_token_value() -> None:
    secret = "ghp_super-secret-token-value-1234567890"
    config = _config(
        adapter="devin-shell",
        devin_worker_env={"GH_TOKEN": secret},
    )
    findings = worker_github_token_findings(config)
    assert len(findings) == 1
    # The ok finding's detail must not contain the secret.
    assert secret not in findings[0].detail
    # The missing finding's detail must name the remediation.
    missing_config = _config(adapter="devin-shell")
    missing_findings = worker_github_token_findings(missing_config)
    detail = missing_findings[0].detail
    assert "devin.worker_env" in detail
    assert "sanitize_env" in detail
    assert "GH_TOKEN" in detail  # the variable NAME, not a value


# ---------------------------------------------------------------------------
# 2. Warn-only default: escalates once, dispatch proceeds
# ---------------------------------------------------------------------------


def test_dispatch_warn_only_escalates_once_and_proceeds(tmp_path: Path) -> None:
    """With no token and require_worker_github_token=False (default), dispatch
    must escalate once (emit a worker_token_missing event) and then proceed
    normally — not refuse.

    Uses dry_run=True so the dispatch proceeds to show what would be
    dispatched without actually creating worktrees or launching workers.
    The gate fires before the dry-run/real branch split, so the escalation
    event is emitted regardless.
    """
    from charlie_work.workflow import OrchestratorApp

    config = _config(adapter="devin-shell")
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    # First dispatch pass: should escalate and proceed.
    fake_gh.prs[0]["state"] = "CLOSED"
    result1 = app.dispatch(limit=1)
    assert result1.ok is True
    assert result1.data["selected_count"] == 1  # dispatch proceeded

    state = load_state(paths.state_file)
    events1 = _events_of_kind(state, "worker_token_missing")
    assert len(events1) == 1, "first pass must escalate once"
    # No token value in the event payload.
    payload_str = json.dumps(events1[0])
    assert "placeholder" not in payload_str
    assert "ghp_" not in payload_str

    # Second dispatch pass: must NOT re-escalate (once-only).
    # Clear the issue's dispatched status so it's re-dispatchable.
    from charlie_work.state import load_state as _ls, save_state, state_lock

    with state_lock(paths.state_file):
        s = _ls(paths.state_file)
        s["issues"].pop("123", None)
        save_state(paths.state_file, s)

    result2 = app.dispatch(limit=1)
    assert result2.ok is True

    state2 = load_state(paths.state_file)
    events2 = _events_of_kind(state2, "worker_token_missing")
    assert len(events2) == 1, "second pass must not re-escalate"


def test_dispatch_escalation_cleared_when_token_added(tmp_path: Path) -> None:
    """When the condition resolves (token added on a new instance), the
    escalation flag resets and no new event fires.
    """
    from charlie_work.workflow import OrchestratorApp

    # First instance: no token — escalates.
    config_no_token = _config(adapter="devin-shell")
    paths = runtime_paths(tmp_path, config_no_token.runtime.state_dir)
    fake_gh = FakeGitHub()
    app1 = OrchestratorApp(tmp_path, paths, config_no_token, fake_gh, dry_run=True)
    fake_gh.prs[0]["state"] = "CLOSED"
    app1.dispatch(limit=1)

    state1 = load_state(paths.state_file)
    assert len(_events_of_kind(state1, "worker_token_missing")) == 1

    # Second instance: token configured — no escalation, flag is False.
    config_with_token = _config(
        adapter="devin-shell",
        devin_worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
    )
    paths2 = runtime_paths(tmp_path, config_with_token.runtime.state_dir)
    fake_gh2 = FakeGitHub()
    app2 = OrchestratorApp(tmp_path, paths2, config_with_token, fake_gh2, dry_run=True)
    assert app2._worker_token_escalated is False

    # Clear issue state for re-dispatch.
    from charlie_work.state import load_state as _ls, save_state, state_lock

    with state_lock(paths2.state_file):
        s = _ls(paths2.state_file)
        s["issues"].pop("123", None)
        save_state(paths2.state_file, s)

    fake_gh2.prs[0]["state"] = "CLOSED"
    app2.dispatch(limit=1)

    state2 = load_state(paths2.state_file)
    assert len(_events_of_kind(state2, "worker_token_missing")) == 1, (
        "no new escalation when token is configured"
    )


# ---------------------------------------------------------------------------
# 3. Hard refusal when require_worker_github_token=True
# ---------------------------------------------------------------------------


def test_dispatch_refuses_when_require_flag_on_and_no_token(tmp_path: Path) -> None:
    """With require_worker_github_token=True and no token, dispatch must
    refuse (defer) rather than launching a worker.
    """
    from charlie_work.workflow import OrchestratorApp

    config = _config(adapter="devin-shell", require_token=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)
    assert result.data["deferred_reason"] == "worker_token_missing"
    assert result.data["selected_count"] == 0
    # The refusal message names the remediation.
    assert "devin.worker_env" in result.message or "claude_code.worker_env" in result.message
    # No token value in the message.
    assert "ghp_" not in result.message
    assert "placeholder" not in result.message


def test_dispatch_proceeds_when_require_flag_on_and_token_configured(
    tmp_path: Path,
) -> None:
    """With require_worker_github_token=True and a token configured, dispatch
    must proceed normally.
    """
    from charlie_work.workflow import OrchestratorApp

    config = _config(
        adapter="devin-shell",
        require_token=True,
        devin_worker_env={"GH_TOKEN": "placeholder-not-a-real-token"},
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, dry_run=True)

    fake_gh.prs[0]["state"] = "CLOSED"
    result = app.dispatch(limit=1)
    assert result.data.get("deferred_reason") != "worker_token_missing"
    assert result.data["selected_count"] == 1


def test_dispatch_refusal_names_remediation_no_token_value(tmp_path: Path) -> None:
    """The refusal message must name the remediation config key and must not
    log any token value or prefix (issue #1001 acceptance criterion).
    """
    from charlie_work.workflow import OrchestratorApp

    config = _config(adapter="claude-code", require_token=True)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.dispatch(limit=1)
    assert result.data["deferred_reason"] == "worker_token_missing"
    assert "claude_code.worker_env" in result.message
    # No token value or prefix.
    assert "ghp_" not in result.message
    assert "github_pat_" not in result.message
    # The missing_config_keys field carries config key names, not values.
    keys = result.data.get("missing_config_keys", [])
    assert "claude_code.worker_env" in keys


# ---------------------------------------------------------------------------
# 4. sanitize_env stripping behaviour unchanged
# ---------------------------------------------------------------------------


def test_sanitize_env_still_strips_all_token_vars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sanitize_env must still strip every GH token variable after the #1001
    refactor extracted STRIPPED_GH_TOKEN_VARS into a named constant.

    This test pins the last acceptance criterion: the fix cannot be
    'simplified' into re-inheriting the orchestrator's credentials.
    """
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
        "sanitize_env must use the STRIPPED_GH_TOKEN_VARS constant, not an "
        "inline tuple (issue #1001)"
    )


# ---------------------------------------------------------------------------
# 5. WorkerTokenFinding is a frozen dataclass
# ---------------------------------------------------------------------------


def test_worker_token_finding_is_frozen() -> None:
    """WorkerTokenFinding must be frozen (CLAUDE.md invariant: config / value
    objects are frozen dataclasses)."""
    f = WorkerTokenFinding(name="test", context="test", config_key="test.key", ok=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.ok = True  # type: ignore[misc]
