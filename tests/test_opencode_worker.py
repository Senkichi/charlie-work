"""Tests for the opencode worker harness (``opencode_worker`` + its wiring).

Modelled on tests/test_api_worker.py: fake ``create_worktree``, a fake CLI
script (``sys.executable``) standing in for ``opencode``, sidecar/argv/env
assertions, and the no-key-material invariant for the forwarded host auth.
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _doctor_fixtures import FakeDoctorGitHub, _config

from charlie_work import claude_code, opencode_worker, role_quota_ledger
from charlie_work.adapter_fate_profile import profile_for
from charlie_work.config import OpenCodeConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.doctor import run_doctor
from charlie_work.harnesses import HARNESS_REGISTRY
from charlie_work.opencode_worker import (
    _REVIEWER_CONFIG,
    launch_opencode_worker,
    pin_flags,
    resolve_model,
    sanitize_review_command_template,
    worker_env_for,
)
from charlie_work.paths import runtime_paths
from charlie_work.role_chain import RoleEntry
from charlie_work.subprocess_runner import RunResult
from charlie_work.worker import iter_workers
from charlie_work.worktree import WorktreeInfo

SECRET = "sk-oc-secret-key-98765"
OTHER_SECRET = "oauth-refresh-OTHER-PROVIDER-4321"
GO_ENTRY = {"type": "api", "key": SECRET}
AUTH_JSON = json.dumps({"opencode-go": GO_ENTRY, "anthropic": {"refresh": OTHER_SECRET}})
MODEL = "glm-5.3-flash"


@pytest.fixture(autouse=True)
def _fake_create_worktree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_create_worktree(repo_root, branch, **_kwargs):
        path = tmp_path / "worktrees" / branch.replace("/", "-")
        path.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=path, branch=branch, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)


@pytest.fixture(autouse=True)
def _isolate_host_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the host-auth lookup at an empty dir unless a test seeds it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "host-data"))
    monkeypatch.delenv("OPENCODE_AUTH_CONTENT", raising=False)


def _seed_host_auth(tmp_path: Path, text: str = AUTH_JSON) -> None:
    auth = tmp_path / "host-data" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text(text, encoding="utf-8")


def _probe_script(tmp_path: Path) -> tuple[str, ...]:
    """A python script standing in for ``opencode``: records argv/stdin/env into cwd."""
    script = tmp_path / "fake_opencode.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, sys
            from pathlib import Path

            payload = {
                "argv": sys.argv[1:],
                "stdin": sys.stdin.read(),
                "env": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(("OPENCODE_", "XDG_DATA_HOME", "CW_"))
                },
            }
            tmp = Path("probe.json.partial")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace("probe.json")
            """
        ),
        encoding="utf-8",
    )
    return (sys.executable, str(script))


def _config_for(model: str = MODEL, **opencode: Any) -> OrchestratorConfig:
    return OrchestratorConfig(
        worker=WorkerRoleConfig(harness="opencode", model=model),
        opencode=OpenCodeConfig(**opencode),
    )


def _launch(tmp_path: Path, issue: int = 42, **kwargs: Any):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    kwargs.setdefault("command_template", _probe_script(tmp_path))
    kwargs.setdefault("config", _config_for())
    prompt = kwargs.pop("prompt", "Do the thing.")
    record = launch_opencode_worker(
        issue,
        f"agent/issue-{issue}-fix",
        prompt,
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        **kwargs,
    )
    return record, sessions_dir


def _read_probe(record) -> dict[str, Any]:
    """The fake CLI writes probe.json atomically (rename); poll for it."""
    path = Path(record.worktree_path) / "probe.json"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.05)
    raise AssertionError(f"probe never written: {path}")


# --- pure helpers -----------------------------------------------------------


def test_registry_declares_opencode_worker_and_reviewer() -> None:
    caps = HARNESS_REGISTRY["opencode"]
    assert caps.worker is True
    assert caps.review is True
    assert caps.adapter_kind == "opencode"
    assert caps.skill_dirs == (".opencode/skills", ".claude/skills")


@pytest.mark.parametrize(
    ("model", "provider", "expected"),
    [
        (MODEL, "opencode-go", f"opencode-go/{MODEL}"),
        ("anthropic/claude-x", "opencode-go", "anthropic/claude-x"),
        ("  glm  ", "p", "p/glm"),
        ("", "opencode-go", ""),
        (MODEL, "", MODEL),
    ],
)
def test_resolve_model(model: str, provider: str, expected: str) -> None:
    assert resolve_model(model, provider) == expected


def test_pin_flags_last_flag_wins_strips_operator_model_and_variant() -> None:
    command = ("opencode", "run", "-m", "x/y", "--model=a/b", "--model", "c/d", "--variant", "low")
    pinned = pin_flags(command, "opencode-go/glm", "high")
    assert pinned.count("--model") == 1
    assert "-m" not in pinned and "x/y" not in pinned and "a/b" not in pinned
    assert pinned[pinned.index("--model") + 1] == "opencode-go/glm"
    assert pinned.count("--variant") == 1
    assert pinned[pinned.index("--variant") + 1] == "high"


def test_pin_flags_without_variant_leaves_operator_variant_alone() -> None:
    pinned = pin_flags(("opencode", "run", "--variant", "low"), "p/m", "")
    assert pinned == ("opencode", "run", "--variant", "low", "--model", "p/m")


# --- worker_env_for ---------------------------------------------------------


def test_worker_env_for_posture_and_per_issue_data_dir(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    empty = {"XDG_DATA_HOME": str(tmp_path / "empty-host")}
    env = worker_env_for(7, sessions, environ=empty)
    assert env["XDG_DATA_HOME"] == str(sessions / "opencode-data" / "issue-7")
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] == "1"
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["permission"] == {"*": "allow"}
    assert config["autoupdate"] is False
    assert worker_env_for(8, sessions, environ=empty)["XDG_DATA_HOME"] != env["XDG_DATA_HOME"]
    assert "OPENCODE_AUTH_CONTENT" not in env  # no host login present


def test_worker_env_for_forwards_host_auth_json(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    env = worker_env_for(1, tmp_path / "s", environ={"XDG_DATA_HOME": str(tmp_path / "host-data")})
    # Only the selected provider's entry is forwarded, never the whole host file.
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in env["OPENCODE_AUTH_CONTENT"]


def test_worker_env_for_scopes_auth_to_requested_provider(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    environ = {"XDG_DATA_HOME": str(tmp_path / "host-data")}
    env = worker_env_for(1, tmp_path / "s", environ=environ, provider="anthropic")
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"anthropic": {"refresh": OTHER_SECRET}}
    assert SECRET not in env["OPENCODE_AUTH_CONTENT"]


def test_worker_env_for_forwards_nothing_for_absent_provider(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    environ = {"XDG_DATA_HOME": str(tmp_path / "host-data")}
    env = worker_env_for(1, tmp_path / "s", environ=environ, provider="not-logged-in")
    assert "OPENCODE_AUTH_CONTENT" not in env


@pytest.mark.parametrize("text", ["not json", "[1, 2]", ""])
def test_worker_env_for_forwards_nothing_for_invalid_auth_json(tmp_path: Path, text: str) -> None:
    _seed_host_auth(tmp_path, text)
    environ = {"XDG_DATA_HOME": str(tmp_path / "host-data")}
    assert "OPENCODE_AUTH_CONTENT" not in worker_env_for(1, tmp_path / "s", environ=environ)


def test_worker_env_for_skips_disk_read_when_auth_already_in_environ(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    env = worker_env_for(
        1,
        tmp_path / "s",
        environ={"XDG_DATA_HOME": str(tmp_path / "host-data"), "OPENCODE_AUTH_CONTENT": "x"},
    )
    # Already in the inherited environ: the child inherits it, nothing re-read.
    assert "OPENCODE_AUTH_CONTENT" not in env


def test_worker_env_for_operator_env_overrides_posture(tmp_path: Path) -> None:
    env = worker_env_for(
        1,
        tmp_path / "s",
        {"OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "0", "XDG_DATA_HOME": "/custom"},
        environ={"XDG_DATA_HOME": str(tmp_path / "empty-host")},
    )
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] == "0"
    assert env["XDG_DATA_HOME"] == "/custom"


# --- review posture ---------------------------------------------------------


def test_worker_env_for_review_denies_writes_and_allowlists_read_bash(
    tmp_path: Path,
) -> None:
    env = worker_env_for(
        3, tmp_path / "s", environ={"XDG_DATA_HOME": str(tmp_path / "h")}, review=True
    )
    permission = json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]
    assert permission["*"] == "deny"
    assert permission["edit"] == "deny"
    assert permission["read"] == "allow"
    bash = permission["bash"]
    assert bash["*"] == "deny"
    assert bash["git diff *"] == "allow"
    assert bash["gh pr view *"] == "allow"
    # The paired write verbs stay denied: no git push/commit or gh pr merge/edit.
    allowed = [pattern for pattern, effect in bash.items() if effect == "allow"]
    assert not any("push" in p or "commit" in p or "merge" in p or "edit" in p for p in allowed)


def test_worker_env_for_review_deny_policy_cannot_be_overridden(tmp_path: Path) -> None:
    """The reviewer's read-only invariant wins over operator worker_env --
    the OPENCODE_CONFIG_CONTENT counterpart of the --permission-mode plan pin."""
    hostile = json.dumps({"permission": {"*": "allow"}})
    env = worker_env_for(
        3,
        tmp_path / "s",
        {"OPENCODE_CONFIG_CONTENT": hostile},
        environ={"XDG_DATA_HOME": str(tmp_path / "h")},
        review=True,
    )
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["*"] == "deny"


def test_worker_env_for_review_pins_every_opencode_control_env(tmp_path: Path) -> None:
    """Every OPENCODE_* control surface is pinned after the worker_env merge:
    opencode merges OPENCODE_PERMISSION after OPENCODE_CONFIG_CONTENT, and the
    review checkout's .opencode/ dir, plugin list, or an ambient/extra config
    file could otherwise override the deny policy or run code at startup."""
    hostile_permission = json.dumps({"*": "allow"})
    env = worker_env_for(
        3,
        tmp_path / "s",
        {
            "OPENCODE_PERMISSION": hostile_permission,
            "OPENCODE_CONFIG": "/evil/opencode.json",
            "OPENCODE_CONFIG_DIR": "/evil/.opencode",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "0",
            "OPENCODE_PURE": "0",
        },
        environ={
            "XDG_DATA_HOME": str(tmp_path / "h"),
            # Ambient os.environ values flow through the sanitized base env --
            # they must lose to the pins too, not just hostile worker_env.
            "OPENCODE_PERMISSION": hostile_permission,
        },
        review=True,
    )
    # Pinned to the same deny policy the config content carries (pin, not
    # strip: deleting it would leave whatever ambient value survives).
    assert json.loads(env["OPENCODE_PERMISSION"])["*"] == "deny"
    assert json.loads(env["OPENCODE_PERMISSION"])["edit"] == "deny"
    # The untrusted review checkout's project config, agents and plugins never
    # load; external plugins are off entirely.
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_PURE"] == "1"
    # Extra config sources neutralized (empty == falsy in opencode's loader).
    assert env["OPENCODE_CONFIG"] == ""
    assert env["OPENCODE_CONFIG_DIR"] == ""


def _oc_match(command: str, pattern: str) -> bool:
    """opencode's Wildcard.match: ^pattern$ anchored, * -> .*, ? -> ., case-insensitive."""
    regex = re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".")
    return re.match(f"^{regex}$", command, re.IGNORECASE | re.DOTALL) is not None


def _oc_bash_effect(bash: dict[str, str], command: str) -> str:
    """findLast semantics: the last matching rule decides."""
    effect = "ask"
    for pattern, action in bash.items():
        if _oc_match(command, pattern):
            effect = action
    return effect


def test_review_bash_trailing_denies_close_wildcard_write_escapes() -> None:
    """The "verb *" allows are prefix-anchored; a command like "git diff > f"
    still matches "git diff *" (opencode checks the whole redirected_statement
    text), so the trailing deny patterns must win under findLast evaluation."""
    bash = _REVIEWER_CONFIG["permission"]["bash"]
    # Ordering is the enforcement mechanism: every entry after the last allow
    # must be a deny, or findLast could hand a write escape back to "allow".
    last_allow = max(i for i, v in enumerate(bash.values()) if v == "allow")
    assert all(v == "deny" for v in list(bash.values())[last_allow + 1 :])

    for attack in (
        "git diff > hacked.txt",
        "git diff --output=hacked.txt",
        "git log --output=../evil",
        "git show HEAD --output=/tmp/x",
        "git diff --ext-diff",
        "git log --exec=x",
        "gh pr view 1 > notes.md",
        "gh pr diff 1 > /tmp/pr.diff",
        "pytest tests --junitxml=C:/evil/x.xml",
        "pytest -o addopts=--junitxml=/tmp/x",
        "uv run --extra dev pytest -o addopts=x",
        "git push origin HEAD",
        "git -c core.editor=sh rebase -i",
        "sh -c 'rm -rf .'",
        "rm -rf .",
        "git commit -m x",
        "gh pr merge 1",
    ):
        assert _oc_bash_effect(bash, attack) == "deny", attack
    for allowed in (
        "git status",
        "git diff HEAD~1",
        "git diff --stat",
        "git log --oneline -20",
        "git show HEAD:file.py",
        "gh pr view 123 --json files",
        "gh pr diff 123",
        "pytest tests/test_x.py -x",
        "uv run --extra dev pytest -q --tb=short",
    ):
        assert _oc_bash_effect(bash, allowed) == "allow", allowed


def test_sanitize_review_command_template_strips_agent_flags() -> None:
    assert sanitize_review_command_template(
        ("opencode", "run", "--agent", "build", "--auto", "--agent=build")
    ) == ("opencode", "run", "--auto")


def _fake_review_checkout(
    repo_root: Path, pr_number: int, head_sha: str, *, reviews_dir: Path
) -> WorktreeInfo:
    checkout = reviews_dir / f"pr-{pr_number}"
    checkout.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=checkout, branch=head_sha, venv_junction=None)


def test_review_launch_pins_model_variant_and_deny_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review=True launch: --model from model_override, --variant from
    resolved_review_effort, --agent stripped, claude review pins absent,
    env carries the deny permission policy."""
    monkeypatch.setattr(claude_code, "create_review_checkout", _fake_review_checkout)
    record, sessions_dir = _launch(
        tmp_path,
        review=True,
        head_sha="b" * 40,
        model_override="glm-5.3-flash",
        resolved_review_effort="max",
        command_template=(*_probe_script(tmp_path), "--agent", "build"),
    )
    assert record.ok, record.error
    argv = _read_probe(record)["argv"]
    assert argv[argv.index("--model") + 1] == "opencode-go/glm-5.3-flash"
    assert argv[argv.index("--variant") + 1] == "max"
    for bad in ("--agent", "--permission-mode", "--max-turns", "--effort", "--output-format"):
        assert bad not in argv
    env = _read_probe(record)["env"]
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["*"] == "deny"
    assert json.loads(env["OPENCODE_PERMISSION"])["*"] == "deny"
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_PURE"] == "1"
    assert env["XDG_DATA_HOME"] == str(sessions_dir / "opencode-data" / "issue-42")
    sidecar = sessions_dir / "issue-42.opencode.json"
    assert sidecar.exists()
    assert (sessions_dir / "issue-42-review.claude.log").exists()
    assert not (sessions_dir / "issue-42.claude.log").exists()


def test_review_launch_requires_head_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(claude_code, "create_review_checkout", _fake_review_checkout)
    record, _ = _launch(tmp_path, review=True)
    assert not record.ok
    assert "head_sha" in record.error


# --- launch -----------------------------------------------------------------


def test_launch_writes_opencode_sidecar_and_pins_single_model(tmp_path: Path) -> None:
    record, sessions_dir = _launch(tmp_path)

    assert record.ok, record.error
    assert record.adapter_kind == "opencode"
    assert record.provider == "opencode-go"
    sidecar = sessions_dir / "issue-42.opencode.json"
    assert sidecar.exists()
    assert not (sessions_dir / "issue-42.claude.json").exists()
    assert not (sessions_dir / "issue-42.api.json").exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["adapter_kind"] == "opencode"
    assert payload["provider"] == "opencode-go"
    assert payload["error"] is None

    for argv in (tuple(record.command), tuple(payload["command"])):
        assert argv.count("--model") == 1
        assert argv[argv.index("--model") + 1] == f"opencode-go/{MODEL}"
        assert "--effort" not in argv
        assert "--permission-mode" not in argv
        assert "--variant" not in argv  # variant unset -> no pin


def test_launch_pins_variant_when_configured(tmp_path: Path) -> None:
    record, _ = _launch(tmp_path, config=_config_for(variant="high"))
    assert record.ok, record.error
    assert record.command.count("--variant") == 1
    assert record.command[record.command.index("--variant") + 1] == "high"


def test_launch_provider_passthrough_when_model_names_provider(tmp_path: Path) -> None:
    record, sessions_dir = _launch(tmp_path, config=_config_for(model="zen/big-pickle"))
    assert record.ok, record.error
    assert record.command[record.command.index("--model") + 1] == "zen/big-pickle"
    assert record.provider == "zen"
    payload = json.loads((sessions_dir / "issue-42.opencode.json").read_text(encoding="utf-8"))
    assert payload["provider"] == "zen"


def test_launch_strips_operator_supplied_model_flags(tmp_path: Path) -> None:
    script = _probe_script(tmp_path)
    record, _ = _launch(
        tmp_path, command_template=(*script, "-m", "evil/model", "--model=evil/other")
    )
    assert record.ok, record.error
    argv = _read_probe(record)["argv"]
    assert argv.count("--model") == 1
    assert argv[argv.index("--model") + 1] == f"opencode-go/{MODEL}"
    assert "-m" not in argv
    assert not any("evil" in token for token in argv)


def test_launch_feeds_prompt_on_stdin_and_writes_prompt_file(tmp_path: Path) -> None:
    record, _ = _launch(tmp_path, prompt="Fix the bug please.")
    assert record.ok, record.error
    assert _read_probe(record)["stdin"] == "Fix the bug please."
    prompt_path = Path(record.worktree_path) / ".orchestrator-prompt.md"
    assert prompt_path.read_text(encoding="utf-8") == "Fix the bug please."


def test_launch_uses_config_command_when_no_template_given(tmp_path: Path) -> None:
    script = _probe_script(tmp_path)
    record, _ = _launch(
        tmp_path, command_template=None, config=_config_for(command=(*script, "run", "--auto"))
    )
    assert record.ok, record.error
    assert _read_probe(record)["argv"][:2] == ["run", "--auto"]


def test_launch_child_env_carries_posture_and_per_issue_data_dir(tmp_path: Path) -> None:
    record, sessions_dir = _launch(tmp_path, issue=55)
    assert record.ok, record.error
    env = _read_probe(record)["env"]
    assert env["XDG_DATA_HOME"] == str(sessions_dir / "opencode-data" / "issue-55")
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] == "1"
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"] == {"*": "allow"}
    assert "OPENCODE_AUTH_CONTENT" not in env


def test_launch_operator_worker_env_overrides_posture(tmp_path: Path) -> None:
    record, _ = _launch(
        tmp_path, worker_env={"OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "0", "CW_MARK": "yes"}
    )
    assert record.ok, record.error
    env = _read_probe(record)["env"]
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] == "0"
    assert env["CW_MARK"] == "yes"


def test_launch_forwards_host_auth_but_never_persists_it(tmp_path: Path) -> None:
    """The host auth.json reaches the child via OPENCODE_AUTH_CONTENT only --
    the key never lands in a sidecar, record dict, argv, or any file under
    sessions_dir (no-key-material invariant, as for the api worker)."""
    _seed_host_auth(tmp_path)
    record, sessions_dir = _launch(tmp_path)
    assert record.ok, record.error

    forwarded = _read_probe(record)["env"]["OPENCODE_AUTH_CONTENT"]
    assert json.loads(forwarded) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in forwarded

    assert SECRET not in json.dumps(record.to_dict())
    assert SECRET not in " ".join(record.command)
    for path in sessions_dir.rglob("*"):
        if path.is_file():
            assert SECRET not in path.read_text(encoding="utf-8", errors="ignore"), path


def test_launch_empty_model_refused_without_popen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*_a, **_k):
        raise AssertionError("Popen must not be reached for an empty model")

    monkeypatch.setattr(claude_code.subprocess, "Popen", boom)
    record, sessions_dir = _launch(tmp_path, config=_config_for(model=""))

    assert not record.ok
    assert record.pid is None
    assert record.adapter_kind == "opencode"
    assert "worker.model is empty" in (record.error or "")
    sidecar = sessions_dir / "issue-42.opencode.json"
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["adapter_kind"] == "opencode"
    assert "worker.model is empty" in payload["error"]


def test_review_launch_empty_model_refused_names_reviewer(tmp_path: Path) -> None:
    """The refused-launch error names the role that owns the empty field --
    a reviewer launch's model comes from the reviewer chain entry."""
    record, sessions_dir = _launch(
        tmp_path,
        review=True,
        head_sha="b" * 40,
        model_override="",
        config=_config_for(model=""),
    )
    assert not record.ok
    assert "reviewer.model is empty" in (record.error or "")
    payload = json.loads((sessions_dir / "issue-42.opencode.json").read_text(encoding="utf-8"))
    assert "reviewer.model is empty" in payload["error"]


def test_launch_never_raises_on_missing_binary(tmp_path: Path) -> None:
    record, _ = _launch(tmp_path, command_template=(str(tmp_path / "no-such-opencode-bin"),))
    assert not record.ok
    assert record.error
    assert record.adapter_kind == "opencode"


def test_launch_delegates_with_opencode_adapter_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}
    real = claude_code.launch_claude_worker

    def capturing(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(opencode_worker, "launch_claude_worker", capturing)
    record, _ = _launch(tmp_path)
    assert record.ok, record.error
    assert captured["adapter_kind"] == "opencode"
    assert captured["provider"] == "opencode-go"
    assert callable(captured["cli_pins"])


# --- role chain / worker discovery ------------------------------------------


def test_role_quota_ledger_sidecar_path_for_opencode(tmp_path: Path) -> None:
    assert role_quota_ledger.sidecar_path_for(tmp_path, "opencode", 9) == (
        tmp_path / "issue-9.opencode.json"
    )


def test_classified_quota_death_on_stamped_opencode_session_restricts_ledger(
    tmp_path: Path,
) -> None:
    record, sessions_dir = _launch(tmp_path, issue=77)
    assert record.ok, record.error
    sidecar = role_quota_ledger.sidecar_path_for(sessions_dir, "opencode", 77)
    assert sidecar is not None and sidecar.exists()
    stamp = role_quota_ledger.session_stamp("worker", "opencode", MODEL, 1)
    assert role_quota_ledger.stamp_session(sidecar, stamp)

    # Dead session whose log tail carries a quota-exhausted signature.
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    Path(payload["log_path"]).write_text(
        '{"type":"step_start"}\n'
        '{"type":"error","error":{"name":"APIError","data":{"message":"Usage limit reached",'
        '"statusCode":429,"isRetryable":true}}}\n',
        encoding="utf-8",
    )
    payload["pid"] = 999_999_937
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    [view] = iter_workers(sessions_dir)
    assert view.adapter_kind == "opencode"
    profile = profile_for(view.adapter_kind)
    assert profile is not None and profile.record_failure is not None
    kind, until = profile.record_failure(
        sessions_dir, view.issue_number, fallback_kind="stalled", config=_config_for()
    )
    assert kind == "quota_exhausted"
    assert until is not None

    restrictions = role_quota_ledger.load_restrictions()
    assert ("opencode", MODEL) in restrictions
    assert restrictions[("opencode", MODEL)] > datetime.now(UTC) - timedelta(minutes=1)


def test_iter_workers_discovers_opencode_sidecar_and_is_alive_not_shortcircuited(
    tmp_path: Path,
) -> None:
    record, sessions_dir = _launch(tmp_path, issue=31)
    assert record.ok, record.error
    views = [v for v in iter_workers(sessions_dir) if v.issue_number == 31]
    assert len(views) == 1
    view = views[0]
    assert view.adapter_kind == "opencode"
    assert profile_for("opencode") is not None
    # An unknown adapter kind is conservatively dead; opencode has a profile, so
    # liveness follows the pid. Point the view at this (live) test process.
    assert replace(view, pid=os.getpid(), process_start_time=None).is_alive() is True
    assert replace(view, pid=999_999_937, process_start_time=None).is_alive() is False


def test_worker_view_reap_sidecar_removes_opencode_sidecar(tmp_path: Path) -> None:
    record, sessions_dir = _launch(tmp_path, issue=32)
    assert record.ok, record.error
    [view] = [v for v in iter_workers(sessions_dir) if v.issue_number == 32]
    sidecar = sessions_dir / "issue-32.opencode.json"
    assert sidecar.exists()
    view.reap_sidecar(sessions_dir)
    assert not sidecar.exists()


# --- doctor -----------------------------------------------------------------


def _doctor(tmp_path: Path, config: OrchestratorConfig):
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)
    _, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, adapter_probe=True)
    return {check.name: check for check in checks}


def test_doctor_probes_opencode_as_primary_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake_probe(repo_root, *, command=None, **_kw):
        seen.append(tuple(command))
        return RunResult(returncode=0, stdout="opencode 1.2.3\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    by_name = _doctor(tmp_path, _config(worker=WorkerRoleConfig(harness="opencode", model=MODEL)))
    assert seen == [("opencode", "--version")]
    assert by_name["opencode CLI probe"].ok is True
    assert "opencode 1.2.3" in by_name["opencode CLI probe"].detail


def test_doctor_probes_opencode_when_only_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, ...]] = []

    def fake_probe(repo_root, *, command=None, **_kw):
        seen.append(tuple(command))
        if command[0] == "opencode":
            return RunResult(returncode=None, stdout="", stderr="", error="opencode: not found")
        return RunResult(returncode=0, stdout="claude 9\n", stderr="")

    monkeypatch.setattr("charlie_work.claude_code.probe_claude", fake_probe)
    worker = WorkerRoleConfig(
        harness="claude-code",
        model="claude-sonnet-5-5",
        fallbacks=(RoleEntry("opencode", MODEL),),
    )
    by_name = _doctor(tmp_path, _config(worker=worker))
    assert ("opencode", "--version") in seen
    assert by_name["opencode CLI probe"].ok is False
    assert "not found" in by_name["opencode CLI probe"].detail


def test_doctor_does_not_probe_opencode_when_not_in_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "charlie_work.claude_code.probe_claude",
        lambda repo_root, **_kw: RunResult(returncode=0, stdout="ok", stderr=""),
    )
    worker = WorkerRoleConfig(harness="claude-code", model="claude-sonnet-5-5")
    by_name = _doctor(tmp_path, _config(worker=worker))
    assert "opencode CLI probe" not in by_name
