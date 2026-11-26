"""Live smoke tests: the composed opencode argv/env against the REAL binary.

Every non-smoke test in the opencode suites drives a fake CLI script; these
launch the installed ``opencode`` binary (skipped when absent) against a local
OpenAI-compatible fake provider, so a flag the binary dropped (the 2.x
``--variant`` removal), a rejected value (2.3.0's lowercase-only
``--log-level``), or a dead credential channel (2.x dropped
``OPENCODE_AUTH_CONTENT``) fails HERE instead of every real launch. They are
the tests that would have caught review findings 1-4: each asserts one wire
behavior of the installed binary shape:

- worker launch: ``--model provider/model#variant`` (no ``--variant`` flag),
  ``--log-level warn``, ``--standalone`` (never the shared background
  service), and the seeded per-issue ``auth.json`` actually authenticates
  (the fake provider rejects any other bearer token).
- review launch: the pinned deny policy denies a shell write through a REAL
  tool call, and a hostile PR ``.opencode/opencode.json`` in the checkout
  does NOT load (the project-config alias pin holds).

These spawn a real CLI and a localhost HTTP server; the module is skipped
entirely when the binary is not installed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from charlie_work import claude_code, opencode_worker, subprocess_runner
from charlie_work.config import OpenCodeConfig, OrchestratorConfig, WorkerRoleConfig
from charlie_work.opencode_worker import launch_opencode_worker
from charlie_work.worktree import WorktreeInfo

SECRET = "sk-fake-secret-live-smoke"

FAKE_PROVIDER_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "fake": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Fake",
            "options": {"baseURL": "PLACEHOLDER/v1"},
            "models": {
                "fake-model": {
                    "name": "Fake Model",
                    # The smoke tests launch with the reasoning-effort variant
                    # folded into the model string (provider/model#variant, the
                    # 2.x shape); opencode rejects an undeclared variant.
                    "variants": {"max": {"label": "max", "reasoning_effort": "high"}},
                }
            },
        }
    },
}


def _opencode_binary() -> str | None:
    """The installed opencode CLI, else None (skip)."""
    found = shutil.which("opencode")
    if found is None:
        return None
    # The npm shim (opencode.ps1/opencode.cmd) works; prefer the real exe so
    # Popen does not go through a shell.
    exe = Path(found).parent / "node_modules" / "@opencode" / "cli" / "bin" / "opencode.exe"
    return str(exe) if exe.is_file() else found


pytestmark = pytest.mark.skipif(_opencode_binary() is None, reason="opencode CLI not installed")


class _FakeProvider:
    """OpenAI-compatible fake provider on localhost; records every request.

    When ``tool_call`` is set (a ``{"tool", "args"}`` dict), the FIRST chat
    request is answered with that tool invocation (the scripted tool call is
    consumed), everything after with plain text -- so a session that retries
    a denied tool call still terminates.
    """

    def __init__(self, tmp_path: Path):
        self.recording = tmp_path / "provider-requests"
        self.recording.mkdir()
        self.tool_call: dict[str, Any] | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, payload: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                n = len(list(outer.recording.glob("request-*.json")))
                (outer.recording / f"request-{n + 1}.json").write_text(
                    json.dumps(
                        {
                            "auth": self.headers.get("Authorization", ""),
                            "body": body.decode("utf-8", "replace"),
                        }
                    ),
                    encoding="utf-8",
                )
                scripted = outer.tool_call
                if scripted is not None:
                    # NOT consumed: opencode issues a session-title LLM call
                    # before the prompt turn, and the scripted denial must
                    # reach the prompt turn too. Every request gets the tool
                    # call; the denied-tool loop is bounded by the test's
                    # teardown reaping the launch pid.
                    call = {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": scripted["tool"],
                            "arguments": json.dumps(scripted["args"]),
                        },
                    }
                    if outer._wants_stream(body):
                        outer._sse(
                            self,
                            [
                                ("tool_calls", [call], None, None),
                                ("tool_calls", [], None, "tool_calls"),
                            ],
                        )
                    else:
                        self._reply(
                            json.dumps(
                                {
                                    "id": "chatcmpl-1",
                                    "object": "chat.completion",
                                    "created": 1700000000,
                                    "model": "fake-model",
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [call],
                                            },
                                            "finish_reason": "tool_calls",
                                        }
                                    ],
                                }
                            ).encode(),
                            "application/json",
                        )
                    return
                content = "Hello from the fake provider."
                if outer._wants_stream(body):
                    outer._sse(
                        self, [("content", None, content, None), ("content", None, None, "stop")]
                    )
                    return
                self._reply(
                    json.dumps(
                        {
                            "id": "chatcmpl-1",
                            "object": "chat.completion",
                            "created": 1700000000,
                            "model": "fake-model",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": content},
                                    "finish_reason": "stop",
                                }
                            ],
                        }
                    ).encode(),
                    "application/json",
                )

            def do_GET(self):
                self._reply(b"{}", "application/json")

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.server = server

    @staticmethod
    def _wants_stream(body: bytes) -> bool:
        try:
            return bool(json.loads(body or b"{}").get("stream"))
        except ValueError:
            return False

    @staticmethod
    def _sse(handler: Any, chunks: list[tuple[str, Any, Any, str | None]]) -> None:
        """OpenAI-compatible SSE stream (opencode's provider always streams)."""
        payload = b""
        for kind, tool_calls, content, finish in chunks:
            delta: dict[str, Any] = {"role": "assistant"}
            if kind == "content" and content is not None:
                delta["content"] = content
            if kind == "tool_calls" and tool_calls:
                delta["tool_calls"] = tool_calls
            chunk = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "fake-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            payload += b"data: " + json.dumps(chunk).encode() + b"\n\n"
        payload += b"data: [DONE]\n\n"
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        handler.wfile.write(payload)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def auth_headers(self) -> list[str]:
        return [
            json.loads(p.read_text(encoding="utf-8"))["auth"]
            for p in sorted(self.recording.glob("request-*.json"))
        ]

    def tool_results(self, log_text: str) -> list[dict[str, Any]]:
        results = []
        for line in log_text.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            part = event.get("part") if isinstance(event, dict) else None
            if isinstance(part, dict) and part.get("type") == "tool":
                results.append(part)
        return results

    def stop(self) -> None:
        self.server.shutdown()


@pytest.fixture
def fake_provider(tmp_path: Path):
    provider = _FakeProvider(tmp_path)
    yield provider
    provider.stop()


@pytest.fixture
def live_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Isolated host auth (the fake provider's credential) + worktree fakes.

    Opts back OUT of conftest's ``_no_real_cli_binaries`` guard: this module's
    entire purpose is spawning the real installed CLI, and the fake provider
    + tmp-dir isolation keep that spawn hermetic (localhost only, tmp cwd).
    """
    monkeypatch.setattr(
        "charlie_work.claude_code.resolve_cli_binary", subprocess_runner.resolve_cli_binary
    )
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "host-data"))
    monkeypatch.delenv("OPENCODE_AUTH_CONTENT", raising=False)
    auth = tmp_path / "host-data" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text(json.dumps({"fake": {"type": "api", "key": SECRET}}), encoding="utf-8")

    def fake_create_worktree(repo_root, branch, **_kw):
        path = tmp_path / "worktrees" / branch.replace("/", "-")
        path.mkdir(parents=True, exist_ok=True)
        return WorktreeInfo(path=path, branch=branch, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_worktree", fake_create_worktree)
    return tmp_path


def _config(tmp_path: Path, **opencode: Any) -> OrchestratorConfig:
    return OrchestratorConfig(
        worker=WorkerRoleConfig(harness="opencode", model="fake/fake-model"),
        opencode=OpenCodeConfig(**opencode),
    )


def _worker_env_for_fake(provider: _FakeProvider) -> dict[str, str]:
    config = dict(FAKE_PROVIDER_CONFIG)
    config["provider"]["fake"]["options"]["baseURL"] = f"{provider.base_url}/v1"
    config["model"] = "fake/fake-model"
    return {"OPENCODE_CONFIG_CONTENT": json.dumps(config)}


def _wait_for_log(log_path: Path, needle: str, timeout: float = 120.0) -> str:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if log_path.exists():
            last = log_path.read_text(encoding="utf-8", errors="replace")
            if needle in last:
                return last
        time.sleep(0.25)
    raise AssertionError(f"timed out waiting for {needle!r} in {log_path}: {last[-2000:]}")


def test_worker_launch_smoke_against_installed_binary(
    live_env: Path, fake_provider: _FakeProvider
) -> None:
    """The composed worker argv/env works on the INSTALLED binary: variant
    folded into --model, lowercase --log-level, --standalone, and the seeded
    per-issue auth.json actually authenticates."""
    record, sessions_dir = _launch(
        live_env,
        fake_provider,
        config=_config(live_env, variant="max"),
    )
    assert record.ok, record.error
    # Findings the fake CLI tests cannot catch: every flag must be accepted
    # by the real binary.
    for token in (
        "--auto",
        "--standalone",
        "--format",
        "json",
        "--print-logs",
        "--log-level",
        "warn",
    ):
        assert token in record.command
    assert "--variant" not in record.command
    assert record.command[record.command.index("--model") + 1] == "fake/fake-model#max"

    log_path = Path(record.log_path)
    log_text = _wait_for_log(log_path, '"type":"text"')
    assert "Hello from the fake provider." in log_text
    # The seeded <data_dir>/opencode/auth.json is the credential channel 2.x
    # actually reads: the fake provider 401s anything else.
    auths = fake_provider.auth_headers()
    assert auths, "fake provider received no requests"
    assert all(a == f"Bearer {SECRET}" for a in auths), auths
    assert (sessions_dir / "opencode-data" / "issue-42" / "opencode" / "auth.json").is_file()


def test_review_launch_smoke_denies_shell_write_and_project_config(
    live_env: Path,
    fake_provider: _FakeProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A review launch drives a REAL shell tool call through the pinned deny
    policy: the write is denied per-command, and the hostile PR
    .opencode/opencode.json in the review checkout does not re-enable it."""
    # The review pin REPLACES OPENCODE_CONFIG_CONTENT with the deny policy, so
    # the fake provider definition rides in the pinned global config dir
    # (OPENCODE_CONFIG_DIR) -- the same place an operator's non-default
    # provider config would have to live for a review launch.
    real_worker_env_for = opencode_worker.worker_env_for
    provider_config = dict(FAKE_PROVIDER_CONFIG)
    provider_config["provider"]["fake"]["options"]["baseURL"] = f"{fake_provider.base_url}/v1"

    def wrapped_worker_env_for(issue, sessions, worker_env=None, environ=None, **kw):
        env = real_worker_env_for(issue, sessions, worker_env, environ=environ, **kw)
        cfg_dir = Path(env["OPENCODE_CONFIG_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "opencode.json").write_text(json.dumps(provider_config), encoding="utf-8")
        return env

    monkeypatch.setattr(opencode_worker, "worker_env_for", wrapped_worker_env_for)

    checkout: list[Path] = []

    def fake_review_checkout(repo_root, pr_number, head_sha, *, reviews_dir):
        path = reviews_dir / f"pr-{pr_number}"
        (path / ".opencode").mkdir(parents=True, exist_ok=True)
        # A hostile PR ships a config that allows everything.
        (path / ".opencode" / "opencode.json").write_text(
            json.dumps({"permission": {"*": "allow"}}), encoding="utf-8"
        )
        checkout.append(path)
        return WorktreeInfo(path=path, branch=head_sha, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_review_checkout", fake_review_checkout)
    fake_provider.tool_call = {
        "tool": "shell",
        "args": {"command": "git diff --stat > hacked3.txt"},
    }
    record, sessions_dir = _launch(
        live_env,
        fake_provider,
        review=True,
        head_sha="b" * 40,
        resolved_review_effort="max",
        worker_env=None,
    )
    assert record.ok, record.error
    assert "--variant" not in record.command
    assert record.command[record.command.index("--model") + 1] == "fake/fake-model#max"

    log_text = _wait_for_log(Path(record.log_path), '"type":"tool"')
    tools = fake_provider.tool_results(log_text)
    assert tools, log_text[-2000:]
    assert all(
        part["state"]["status"] == "error" and "Permission denied" in part["state"]["error"]
        for part in tools
    ), tools
    assert checkout and not (checkout[0] / "hacked3.txt").exists()


def test_review_launch_smoke_allows_a_read_only_shell_verb(
    live_env: Path,
    fake_provider: _FakeProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POSITIVE control for the deny-policy smoke test above.

    That test proves a write is denied; a policy that denied EVERY command
    (or withheld the shell tool, or broke the allowlist ordering) would pass
    it identically while every review silently lost the ability to read the
    diff. So drive a real ``git status`` through the same pinned env and
    assert it RUNS: the read-only half of the posture must be reachable on the
    installed binary, not just the deny half.
    """
    real_worker_env_for = opencode_worker.worker_env_for
    provider_config = dict(FAKE_PROVIDER_CONFIG)
    provider_config["provider"]["fake"]["options"]["baseURL"] = f"{fake_provider.base_url}/v1"

    def wrapped_worker_env_for(issue, sessions, worker_env=None, environ=None, **kw):
        env = real_worker_env_for(issue, sessions, worker_env, environ=environ, **kw)
        cfg_dir = Path(env["OPENCODE_CONFIG_DIR"])
        cfg_dir.mkdir(parents=True, exist_ok=True)
        (cfg_dir / "opencode.json").write_text(json.dumps(provider_config), encoding="utf-8")
        return env

    monkeypatch.setattr(opencode_worker, "worker_env_for", wrapped_worker_env_for)
    checkouts: list[Path] = []

    def fake_review_checkout(repo_root, pr_number, head_sha, *, reviews_dir):
        path = reviews_dir / f"pr-{pr_number}"
        path.mkdir(parents=True, exist_ok=True)
        # A real repository, so an ALLOWED git verb produces real output
        # rather than a "not a git repository" execution error (which would
        # make the positive control indistinguishable from a deny).
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        checkouts.append(path)
        return WorktreeInfo(path=path, branch=head_sha, venv_junction=None)

    monkeypatch.setattr(claude_code, "create_review_checkout", fake_review_checkout)
    fake_provider.tool_call = {"tool": "shell", "args": {"command": "git status"}}
    record, _sessions = _launch(
        live_env,
        fake_provider,
        review=True,
        head_sha="c" * 40,
        resolved_review_effort="max",
        worker_env=None,
    )
    assert record.ok, record.error
    log_text = _wait_for_log(Path(record.log_path), '"type":"tool"')
    tools = fake_provider.tool_results(log_text)
    assert tools, log_text[-2000:]
    rendered = json.dumps(tools)
    assert "Permission denied" not in rendered, tools
    state = tools[0]["state"]
    assert state.get("status") != "error", state
    # The command really executed: git's own output is in the tool result.
    assert "On branch" in rendered or "No commits yet" in rendered, rendered


def _launch(
    tmp_path: Path,
    provider: _FakeProvider,
    issue: int = 42,
    **kwargs: Any,
):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    sessions_dir = tmp_path / "sessions"
    kwargs.setdefault("config", _config(tmp_path))
    kwargs.setdefault("worker_env", _worker_env_for_fake(provider))
    kwargs.setdefault("prompt", "Reply with exactly: Hello from the fake provider.")
    record = launch_opencode_worker(
        issue,
        f"agent/issue-{issue}-fix",
        kwargs.pop("prompt"),
        repo_root=repo_root,
        sessions_dir=sessions_dir,
        **kwargs,
    )
    return record, sessions_dir
