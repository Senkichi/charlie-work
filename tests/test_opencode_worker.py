"""Tests for the opencode worker harness (``opencode_worker`` + its wiring).

Modelled on tests/test_api_worker.py: fake ``create_worktree``, a fake CLI
script (``sys.executable``) standing in for ``opencode``, sidecar/argv/env
assertions, and the no-key-material invariant for the forwarded host auth.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from charlie_work import claude_code, opencode_worker, role_quota_ledger
from charlie_work.adapter_fate_profile import profile_for
from charlie_work.config import (
    OpenCodeConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.harnesses import HARNESS_REGISTRY
from charlie_work.opencode_worker import (
    launch_opencode_worker,
    pin_flags,
    resolve_model,
    sanitize_review_command_template,
)
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
    """No --variant flag exists on opencode 2.x (verified: `opencode run
    --variant x` prints `run --help`, exit 1): the variant is folded into the
    model string as provider/model#variant, and operator --variant/-m/--model
    tokens are stripped."""
    command = ("opencode", "run", "-m", "x/y", "--model=a/b", "--model", "c/d", "--variant", "low")
    pinned = pin_flags(command, "opencode-go/glm", "high")
    assert pinned.count("--model") == 1
    assert "-m" not in pinned and "x/y" not in pinned and "a/b" not in pinned
    assert "--variant" not in pinned and "low" not in pinned
    assert pinned[pinned.index("--model") + 1] == "opencode-go/glm#high"


def test_pin_flags_without_variant_leaves_operator_variant_alone() -> None:
    pinned = pin_flags(("opencode", "run", "--variant", "low"), "p/m", "")
    assert pinned == ("opencode", "run", "--model", "p/m")


def test_pin_flags_model_already_carrying_variant_keeps_it() -> None:
    pinned = pin_flags(("opencode", "run", "--variant", "low"), "p/m#max", "")
    assert pinned == ("opencode", "run", "--model", "p/m#max")


def test_pin_flags_model_with_variant_ignores_resolved_variant() -> None:
    pinned = pin_flags(("opencode", "run"), "p/m#max", "high")
    assert pinned == ("opencode", "run", "--model", "p/m#max")


def test_sanitize_review_command_template_strips_agent_flags() -> None:
    assert sanitize_review_command_template(
        ("opencode", "run", "--agent", "build", "--auto", "--agent=build")
    ) == ("opencode", "run", "--auto", "--standalone")


def test_sanitize_review_command_template_strips_session_movers() -> None:
    """--continue/--session/--worktree/--fork move the session (and --worktree
    the checkout) boundary -- the same strip class as --agent."""
    assert sanitize_review_command_template(
        ("opencode", "run", "--continue", "--session", "abc", "--fork", "--worktree")
    ) == ("opencode", "run", "--standalone")
    assert sanitize_review_command_template(
        ("opencode", "run", "--worktree", "side-name", "--auto")
    ) == ("opencode", "run", "--auto", "--standalone")  # optional worktree name goes too
    assert sanitize_review_command_template(
        ("opencode", "run", "-c", "-w=side", "-s", "abc", "--session=x")
    ) == ("opencode", "run", "--standalone")


def test_sanitize_review_command_template_strips_server_and_pins_standalone() -> None:
    """``--server`` is the env-posture bypass (the session would run inside a
    server whose env/config this launch does not own) and ``--standalone`` is
    what makes the pins apply -- both are boundary invariants, not defaults.

    An operator ``opencode.command`` replaces DEFAULT_COMMAND_TEMPLATE
    wholesale, so the pin must survive that path; it is appended once even
    when the template already carries it.
    """
    assert sanitize_review_command_template(
        ("opencode", "run", "--server", "http://127.0.0.1:4096", "--auto")
    ) == ("opencode", "run", "--auto", "--standalone")
    assert sanitize_review_command_template(("opencode", "run", "--server=http://x")) == (
        "opencode",
        "run",
        "--standalone",
    )
    assert sanitize_review_command_template(("opencode", "run", "--standalone")) == (
        "opencode",
        "run",
        "--standalone",
    )
    # A value-taking flag at the end is unaffected: the pin is appended, and
    # --standalone is parsed as a flag, never consumed as that value.
    assert sanitize_review_command_template(("opencode", "run", "--log-level", "warn")) == (
        "opencode",
        "run",
        "--log-level",
        "warn",
        "--standalone",
    )


def _fake_review_checkout(
    repo_root: Path, pr_number: int, head_sha: str, *, reviews_dir: Path
) -> WorktreeInfo:
    checkout = reviews_dir / f"pr-{pr_number}"
    checkout.mkdir(parents=True, exist_ok=True)
    return WorktreeInfo(path=checkout, branch=head_sha, venv_junction=None)


def test_review_launch_pins_model_variant_and_deny_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A review=True launch: --model from model_override with the resolved
    review effort folded in as provider/model#variant, --agent/session-movers
    stripped, --standalone pinned, claude review pins absent, env carries the
    deny permission policy."""
    monkeypatch.setattr(claude_code, "create_review_checkout", _fake_review_checkout)
    record, sessions_dir = _launch(
        tmp_path,
        review=True,
        head_sha="b" * 40,
        model_override="glm-5.3-flash",
        resolved_review_effort="max",
        command_template=(*_probe_script(tmp_path), "--agent", "build", "--worktree"),
    )
    assert record.ok, record.error
    argv = _read_probe(record)["argv"]
    assert argv[argv.index("--model") + 1] == "opencode-go/glm-5.3-flash#max"
    # The env posture only applies to a private server, so the pin is part of
    # the review launch, not of the template that happened to be configured.
    assert "--standalone" in argv
    assert argv.count("--standalone") == 1
    for bad in (
        "--variant",
        "--agent",
        "--continue",
        "--session",
        "--worktree",
        "--permission-mode",
        "--max-turns",
        "--effort",
        "--output-format",
    ):
        assert bad not in argv
    env = _read_probe(record)["env"]
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["*"] == "deny"
    assert json.loads(env["OPENCODE_PERMISSION"])["*"] == "deny"
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    # The 2.x alias (OPENCODE_CONFIG_PROJECT_DISABLE ?? legacy) is pinned too:
    # the alias wins on 2.3.0, and an unpinned alias re-enables the PR's
    # .opencode/ project config (verified live 2026-11-18).
    assert env["OPENCODE_CONFIG_PROJECT_DISABLE"] == "1"
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


def test_review_launch_rotates_stale_events_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A leftover issue-N-review.events.jsonl from an earlier claude/api
    round must be rotated before the opencode launch: opencode never tees
    into it, so otherwise the summary/metrics readers attribute the stale
    transcript to this session (the #2036 invariant devin_shell enforces)."""
    monkeypatch.setattr(claude_code, "create_review_checkout", _fake_review_checkout)
    sessions_dir = tmp_path / "sessions"  # the dir _launch passes through
    sessions_dir.mkdir(parents=True)
    stale = sessions_dir / "issue-42-review.events.jsonl"
    stale.write_text('{"type":"assistant","message":{"content":"stale"}}\n', encoding="utf-8")
    record, _ = _launch(tmp_path, review=True, head_sha="c" * 40)
    assert record.ok, record.error
    assert not stale.exists()
    rotated = sessions_dir / "issue-42-review.events.jsonl.1"
    assert rotated.exists()
    assert "stale" in rotated.read_text(encoding="utf-8")


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
        assert "--variant" not in argv  # variant unset -> model pin carries no #variant


def test_launch_folds_config_variant_into_model_pin(tmp_path: Path) -> None:
    record, _ = _launch(tmp_path, config=_config_for(variant="high"))
    assert record.ok, record.error
    assert record.command.count("--model") == 1
    assert record.command[record.command.index("--model") + 1] == f"opencode-go/{MODEL}#high"
    assert "--variant" not in record.command


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
    assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"  # the 2.x name for the same switch
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


def test_launch_seeds_scoped_auth_file_after_data_dir_wipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """opencode 2.x dropped OPENCODE_AUTH_CONTENT (0 occurrences in the
    installed 2.3.0 binary): credentials live in <XDG_DATA_HOME>/opencode/
    auth.json. The launcher wipes the per-issue data dir before every launch,
    so the scoped credential must be written back AFTER the wipe -- and only
    there, scoped to the selected provider."""
    _seed_host_auth(tmp_path)
    record, sessions_dir = _launch(tmp_path, issue=66)
    assert record.ok, record.error
    auth_file = sessions_dir / "opencode-data" / "issue-66" / "opencode" / "auth.json"
    assert auth_file.is_file()
    assert json.loads(auth_file.read_text(encoding="utf-8")) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in auth_file.read_text(encoding="utf-8")

    # A relaunch wipes the dir and re-seeds: stale session state never
    # accumulates, and the credential survives every wipe.
    record, _ = _launch(tmp_path, issue=66)
    assert record.ok, record.error
    assert json.loads(auth_file.read_text(encoding="utf-8")) == {"opencode-go": GO_ENTRY}

    # No host credential -> nothing seeded (the child may auth from env keys).
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "empty-host-data"))
    record2, sessions2 = _launch(tmp_path, issue=67)
    assert record2.ok, record2.error
    assert not (sessions2 / "opencode-data" / "issue-67" / "opencode" / "auth.json").exists()


def test_launch_seeds_auth_scoped_from_operator_env_content(tmp_path: Path) -> None:
    """An operator who authenticates through the documented OPENCODE_AUTH_CONTENT
    env channel gets that credential scoped to provider and seeded for 2.x --
    not dropped (the silent blanking an earlier review flagged)."""
    env_value = json.dumps({"opencode-go": GO_ENTRY, "anthropic": {"refresh": OTHER_SECRET}})
    record, sessions_dir = _launch(
        tmp_path,
        issue=68,
        worker_env={"OPENCODE_AUTH_CONTENT": env_value},
    )
    assert record.ok, record.error
    auth_file = sessions_dir / "opencode-data" / "issue-68" / "opencode" / "auth.json"
    assert auth_file.is_file()
    assert json.loads(auth_file.read_text(encoding="utf-8")) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in auth_file.read_text(encoding="utf-8")


def test_launch_forwards_host_auth_but_never_persists_it_outside_data_dir(tmp_path: Path) -> None:
    """The key reaches only two channels: OPENCODE_AUTH_CONTENT (opencode 1.x,
    child env) and the per-issue data dir's auth.json (opencode 2.x, removed
    by the reap with the data dir). It never lands in a sidecar, record dict,
    argv, log, or any file outside the per-issue data dir (no-key-material
    invariant, as for the api worker)."""
    _seed_host_auth(tmp_path)
    record, sessions_dir = _launch(tmp_path)
    assert record.ok, record.error

    forwarded = _read_probe(record)["env"]["OPENCODE_AUTH_CONTENT"]
    assert json.loads(forwarded) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in forwarded

    assert SECRET not in json.dumps(record.to_dict())
    assert SECRET not in " ".join(record.command)
    data_dir = sessions_dir / "opencode-data" / "issue-42"
    for path in sessions_dir.rglob("*"):
        if path.is_file() and data_dir not in path.parents:
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
