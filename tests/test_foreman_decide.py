"""The foreman Decide launch: role-chain resolution, read-only pins, deferred deaths (issue #3120).

Every test drives the host ``decide_launch`` port through ``FakeDecideLauncher``
-- nothing here starts a model. The single opt-in live test sits behind the
``live_decide`` marker plus ``CHARLIE_WORK_LIVE_DECIDE`` so it is never
selected in CI.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from charlie_work import role_quota_ledger
from charlie_work.config import ClaudeCodeConfig, OrchestratorConfig
from charlie_work.foreman import paths
from charlie_work.foreman.decide import DecideForeman, DecideResult, decide
from charlie_work.foreman_config import FOREMAN_DEFAULT_MODEL, ForemanConfig
from charlie_work.host.fakes import FakeDecideLauncher
from charlie_work.role_chain import RoleEntry
from charlie_work.subprocess_runner import RunResult, resolve_cli_binary

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "turn_summary": {"type": "string"},
        "actions": {"type": "array"},
    },
    "required": ["turn_summary"],
    "additionalProperties": False,
}

_BRIEFING = "Read the state below and produce the decision document."


def _envelope(**overrides: Any) -> str:
    envelope: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 1234,
        "num_turns": 3,
        "result": "",
        "total_cost_usd": 0.0123,
        "usage": {"input_tokens": 11, "output_tokens": 22},
        "structured_output": {"turn_summary": "no work pending", "actions": []},
    }
    envelope.update(overrides)
    return json.dumps(envelope)


def _foreman(tmp_path: Path, config: OrchestratorConfig | None = None) -> DecideForeman:
    fdir = paths.foreman_dir(tmp_path, "g1")
    (fdir / "tree").mkdir(parents=True)
    (fdir / "turns" / "t1").mkdir(parents=True)
    return DecideForeman(root=fdir, config=config or OrchestratorConfig())


def _turn_dir(foreman: DecideForeman) -> Path:
    return foreman.root / "turns" / "t1"


def _call(
    fake: FakeDecideLauncher, index: int = 0
) -> tuple[tuple[str, ...], Path, dict, int, str]:
    return fake.calls[index]


def test_decide_returns_parsed_output_and_metadata(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher([RunResult(returncode=0, stdout=_envelope(), stderr="")])
    ports = fake_host(decide_launch=launcher)
    assert ports.decide_launch is launcher
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert result.ok, result.error
    assert result.output == {"turn_summary": "no work pending", "actions": []}
    assert not result.deferred and result.failure_kind is None
    entry = result.entry
    assert entry is not None and entry.harness == "claude-code"
    assert entry.model == FOREMAN_DEFAULT_MODEL
    assert result.num_turns == 3
    assert result.input_tokens == 11 and result.output_tokens == 22
    assert result.cost_usd == pytest.approx(0.0123)

    argv, cwd, _env, timeout, stdin = _call(launcher)
    assert cwd == foreman.root / "tree"  # the tree/ worktree, never a registered checkout
    assert stdin == _BRIEFING
    assert timeout == foreman.config.foreman.decide_timeout_seconds
    assert "-p" in argv and "--output-format" in argv


def test_decide_writes_the_transcript_log(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher(
        [RunResult(returncode=0, stdout=_envelope(), stderr="a warning")]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    assert decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman)).ok

    log = (_turn_dir(foreman) / "decide.log").read_text(encoding="utf-8")
    assert "a warning" in log and "structured_output" in log


def test_decide_is_error_envelope_is_a_turn_failure_not_deferred(
    tmp_path: Path, fake_host
) -> None:
    launcher = FakeDecideLauncher(
        [
            RunResult(
                returncode=0,
                stdout=_envelope(
                    subtype="error_max_structured_output_retries",
                    is_error=True,
                    result="could not satisfy the schema",
                    structured_output=None,
                ),
                stderr="",
            )
        ]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and result.output is None
    assert not result.deferred  # a schema rejection is a fault, not a back-off
    assert "is_error" in (result.error or "")


def test_decide_missing_structured_output_is_a_turn_failure(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher(
        [RunResult(returncode=0, stdout=_envelope(structured_output=None), stderr="")]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and not result.deferred
    assert "schema" in (result.error or "")


def test_decide_timeout_is_a_turn_failure_not_deferred(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher(
        [
            RunResult(
                returncode=None,
                stdout="",
                stderr="",
                timed_out=True,
                error="command timed out after 1800s",
            )
        ]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and not result.deferred
    assert "timed out" in (result.error or "")
    # A timed-out launch still leaves its transcript for post-mortem.
    assert (_turn_dir(foreman) / "decide.log").is_file()


def test_decide_missing_binary_is_a_turn_failure(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher(
        [
            RunResult(
                returncode=None,
                stdout="",
                stderr="",
                error="[WinError 2] The system cannot find the file specified",
            )
        ]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and not result.deferred
    assert "WinError 2" in (result.error or "")


def test_decide_quota_death_is_deferred_and_restricts_the_entry(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher(
        [
            RunResult(
                returncode=1,
                stdout="",
                stderr="API Error: weekly usage quota has been exhausted",
            )
        ]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and result.deferred
    assert result.failure_kind == "quota_exhausted"
    # The same ledger entry a reviewer death would have written.
    restrictions = role_quota_ledger.load_restrictions()
    assert ("claude-code", FOREMAN_DEFAULT_MODEL) in restrictions


def test_decide_rate_limit_death_is_deferred(tmp_path: Path, fake_host) -> None:
    launcher = FakeDecideLauncher(
        [RunResult(returncode=1, stdout="", stderr="Error: rate limit reached")]
    )
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and result.deferred
    assert result.failure_kind == "rate_limited"
    assert ("claude-code", FOREMAN_DEFAULT_MODEL) in role_quota_ledger.load_restrictions()


def test_decide_selection_skips_a_restricted_primary(tmp_path: Path, fake_host) -> None:
    """A ledger restriction on the primary pushes Decide onto the fallback."""
    fallback_model = "claude-opus-4-7"
    config = OrchestratorConfig(
        foreman=ForemanConfig(fallbacks=(RoleEntry(harness="claude-code", model=fallback_model),))
    )
    role_quota_ledger.record_restriction(
        "claude-code",
        FOREMAN_DEFAULT_MODEL,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        source="test",
    )
    launcher = FakeDecideLauncher([RunResult(returncode=0, stdout=_envelope(), stderr="")])
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path, config)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert result.ok, result.error
    entry = result.entry
    assert entry is not None and entry.model == fallback_model
    assert result.selection is not None and len(result.selection.skipped) == 1


def test_decide_exhausted_chain_is_deferred_without_launching(tmp_path: Path, fake_host) -> None:
    role_quota_ledger.record_restriction(
        "claude-code",
        FOREMAN_DEFAULT_MODEL,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        source="test",
    )
    fallback_model = "claude-opus-4-7"
    role_quota_ledger.record_restriction(
        "claude-code",
        fallback_model,
        datetime.now(UTC) + timedelta(hours=1),
        reason="quota_exhausted",
        source="test",
    )
    config = OrchestratorConfig(
        foreman=ForemanConfig(fallbacks=(RoleEntry(harness="claude-code", model=fallback_model),))
    )
    launcher = FakeDecideLauncher()
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path, config)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and result.deferred
    assert "exhausted" in (result.error or "")
    assert launcher.calls == []  # nothing launched


def test_decide_unsupported_harness_is_a_config_error(tmp_path: Path, fake_host) -> None:
    """foreman.harness validates against REVIEWER_HARNESSES, which is wider
    than Decide's claude-code support -- so the check lives here."""
    config = OrchestratorConfig(foreman=ForemanConfig(harness="opencode"))
    launcher = FakeDecideLauncher()
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path, config)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))

    assert not result.ok and not result.deferred
    assert "claude-code only" in (result.error or "")
    assert launcher.calls == []


def test_decide_missing_tree_is_an_error_value(tmp_path: Path, fake_host) -> None:
    fdir = paths.foreman_dir(tmp_path, "g1")
    (fdir / "turns" / "t1").mkdir(parents=True)  # no tree/
    launcher = FakeDecideLauncher()
    fake_host(decide_launch=launcher)
    foreman = DecideForeman(root=fdir, config=OrchestratorConfig())

    result = decide(foreman, _BRIEFING, SCHEMA, fdir / "turns" / "t1")

    assert not result.ok and "tree" in (result.error or "")
    assert launcher.calls == []


def test_configured_template_cannot_widen_the_decide_contract(tmp_path: Path, fake_host) -> None:
    """Every Decide pin is strip-then-append: a command template carrying
    widened flags keeps the base argv but loses the widening."""
    config = OrchestratorConfig(
        claude_code=ClaudeCodeConfig(
            command=(
                "claude",
                "-p",
                "--permission-mode",
                "bypassPermissions",
                "--tools",
                "Bash",
                "--mcp-config",
                "evil.json",
                "--output-format",
                "text",
                "--json-schema",
                "{}",
                "--setting-sources",
                "user,project",
                "--model",
                "evil-model",
                "--max-turns",
                "999",
            )
        ),
        foreman=ForemanConfig(tools="Read"),
    )
    launcher = FakeDecideLauncher([RunResult(returncode=0, stdout=_envelope(), stderr="")])
    fake_host(decide_launch=launcher)
    foreman = _foreman(tmp_path, config)

    result = decide(foreman, _BRIEFING, SCHEMA, _turn_dir(foreman))
    assert result.ok, result.error

    argv, _cwd, _env, _timeout, _stdin = _call(launcher)

    def values(flag: str) -> list[str]:
        return [argv[i + 1] for i, token in enumerate(argv) if token == flag]

    assert argv[0] == resolve_cli_binary("claude")
    assert "-p" in argv
    # Read-only posture -- the one authoritative pin.
    assert values("--permission-mode") == ["plan"] and "bypassPermissions" not in argv
    # The foreman tool pin -- not the template's widened scope.
    assert values("--tools") == ["Read"] and "Bash" not in argv
    # MCP isolation -- the inline empty-server JSON, not the template's file.
    assert "--strict-mcp-config" in argv
    assert values("--mcp-config") == ['{"mcpServers":{}}'] and "evil.json" not in argv
    # The structured-output contract.
    assert values("--output-format") == ["json"]
    assert values("--json-schema") == [json.dumps(SCHEMA)]
    assert values("--setting-sources") == [""]
    # Role-chain model and the foreman turn cap -- the template's values lost.
    assert values("--model") == [FOREMAN_DEFAULT_MODEL] and "evil-model" not in argv
    assert values("--max-turns") == ["40"] and "999" not in argv


@pytest.mark.live_decide
@pytest.mark.skipif(
    not os.environ.get("CHARLIE_WORK_LIVE_DECIDE"),
    reason="opt-in live test: set CHARLIE_WORK_LIVE_DECIDE=1",
)
@pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not installed")
def test_decide_live_against_the_real_cli(tmp_path: Path) -> None:
    """One real ``claude -p`` Decide call through ``REAL`` ports -- never run in CI."""
    foreman = _foreman(tmp_path)
    schema = {
        "type": "object",
        "properties": {"turn_summary": {"type": "string"}},
        "required": ["turn_summary"],
        "additionalProperties": False,
    }

    result = decide(
        foreman,
        "Summarize the state of this empty foreman turn in one sentence.",
        schema,
        _turn_dir(foreman),
    )

    assert isinstance(result, DecideResult)
    assert result.ok, result.error
    assert result.output is not None and isinstance(result.output.get("turn_summary"), str)
    assert (_turn_dir(foreman) / "decide.log").is_file()
