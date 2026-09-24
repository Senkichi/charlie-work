"""Issue #1780: post-hoc ``worker_literal_tmp_path`` signal for literal ``/tmp`` misuse.

#1767 retargeted ``TMP``/``TEMP``/``TMPDIR`` at a worktree-local directory,
but a literal ``/tmp/...`` path typed into a shell command still resolves
through MSYS's install-wide cached mount under Git Bash (and to the single
shared temp dir on POSIX) — shared across every concurrent worker session.
The rendered prompt now forbids the literal path (``session_scratch_dir``
section + the ``assert_session_scratch_dir`` dispatch-boundary guard); this
module's tests pin the other half of the issue — the structural signal that
a dead claude/api session did it anyway:

* ``worker_literal_tmp.literal_tmp_shell_commands`` scans a session's
  stream-json transcript for ``tool_use`` blocks carrying a shell
  ``command`` with a literal ``/tmp`` token, and
* the dead-session lane in ``dead_worker_reap`` emits a warning-level
  ``worker_literal_tmp_path`` event once per offending session, via
  ``worker_literal_tmp.emit_literal_tmp_path_warning``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from _worker_fixtures import _wg
from charlie_work.claude_code import ClaudeWorkerRecord
from charlie_work.claude_code import _sidecar_path as claude_sidecar_path
from charlie_work.config import OrchestratorConfig, PostMortemConfig
from charlie_work.worker_literal_tmp import literal_tmp_shell_commands


def _assistant_tool_use(command: str, *, tool_name: str = "Bash") -> dict:
    """One real-schema ``assistant`` stream-json event carrying a shell tool_use."""
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "name": tool_name,
                    "input": {"command": command},
                }
            ]
        },
    }


def _write_jsonl(path: Path, events: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in events),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# literal_tmp_shell_commands: unit coverage of the scan itself
# ---------------------------------------------------------------------------


def test_flags_the_incident_shape(tmp_path: Path) -> None:
    """The motivating failure — ``gh pr view ... > /tmp/pr-body.md`` — is caught."""
    log = _write_jsonl(
        tmp_path / "issue-1780.claude.log",
        [
            _assistant_tool_use("gh pr view 5 --json body -q .body > /tmp/pr-body.md"),
            _assistant_tool_use('cat "$TMPDIR/pr-body.md"'),
        ],
    )
    assert literal_tmp_shell_commands(log) == [
        "gh pr view 5 --json body -q .body > /tmp/pr-body.md"
    ]


def test_flags_every_literal_tmp_surface(tmp_path: Path) -> None:
    """Redirects, mktemp templates, cd, and plain args all count — a literal
    ``/tmp`` token anywhere in the command is the same shared-dir hazard."""
    commands = [
        "mktemp -d /tmp/work.XXXXXX",
        "cd /tmp && ls",
        "cp out.txt /tmp/",
        'echo done > "/tmp/result"',
    ]
    log = _write_jsonl(
        tmp_path / "issue-1.claude.log",
        [_assistant_tool_use(c) for c in commands],
    )
    assert literal_tmp_shell_commands(log) == commands


def test_flags_tmpdir_default_fallback(tmp_path: Path) -> None:
    """``${TMPDIR:-/tmp}`` still hard-codes the shared path as a fallback — a
    real residual when TMPDIR is unset, so it is flagged like any literal."""
    log = _write_jsonl(
        tmp_path / "issue-1.claude.log",
        [_assistant_tool_use("out=${TMPDIR:-/tmp}/x && echo $out")],
    )
    assert literal_tmp_shell_commands(log) == ["out=${TMPDIR:-/tmp}/x && echo $out"]


def test_ignores_env_and_non_absolute_tmp_shapes(tmp_path: Path) -> None:
    """Compliant/env-mediated and non-``/tmp``-absolute shapes must not flag.

    ``$TMPDIR``/``${TMPDIR}`` never contain the literal substring at all;
    ``foo/tmp``, ``./tmp``, ``~/tmp`` are relative/home paths that stay inside
    the worktree or home dir; ``/var/tmp``/``/tmpdir``/``/tmp-x`` are not the
    shared ``/tmp`` root itself.
    """
    commands = [
        'gh pr view 5 --json body -q .body > "$TMPDIR/pr-body.md"',
        "mktemp -d ${TMPDIR}/work.XXXXXX",
        "scratch=$(mktemp -d)",
        "mkdir -p ./tmp",
        "cp out.txt foo/tmp/",
        "ls ~/tmp",
        "ls /var/tmp",
        "ls /tmpdir /tmp-x",
    ]
    log = _write_jsonl(
        tmp_path / "issue-1.claude.log",
        [_assistant_tool_use(c) for c in commands],
    )
    assert literal_tmp_shell_commands(log) == []


def test_ignores_prose_mentions_of_tmp(tmp_path: Path) -> None:
    """Assistant *text* and ``result`` prose mentioning ``/tmp`` is not misuse —
    a worker explaining why it avoided the shared dir must not trip the signal.
    Only ``tool_use`` shell commands count."""
    log = _write_jsonl(
        tmp_path / "issue-1.claude.log",
        [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": "I verified no literal /tmp path is used anywhere.",
                        },
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/tmp/should-not-count"},
                        },
                    ]
                },
            },
            {"type": "result", "result": "Done — avoided /tmp as instructed."},
        ],
    )
    assert literal_tmp_shell_commands(log) == []


def test_missing_or_plaintext_log_yields_nothing(tmp_path: Path) -> None:
    """A non-tee'd ``.claude.log`` is plain model output — no tool_use evidence —
    and a missing file is a valid absence state; both return empty."""
    assert literal_tmp_shell_commands(tmp_path / "absent.claude.log") == []

    plain = tmp_path / "issue-1.claude.log"
    plain.write_text(
        "I used /tmp for scratch because it seemed convenient.\nDone.\n",
        encoding="utf-8",
    )
    assert literal_tmp_shell_commands(plain) == []


def test_shell_shaped_tool_names_other_than_bash(tmp_path: Path) -> None:
    """The match is on the ``input.command`` shape, not the tool name — a future
    shell tool (or a renamed one) carrying a ``command`` string is still caught."""
    log = _write_jsonl(
        tmp_path / "issue-1.claude.log",
        [_assistant_tool_use("cat /tmp/x", tool_name="PowerShell")],
    )
    assert literal_tmp_shell_commands(log) == ["cat /tmp/x"]


# ---------------------------------------------------------------------------
# The dead-session lane: worker_literal_tmp_path warning event
# ---------------------------------------------------------------------------


def _dead_api_worker(
    tmp_path: Path,
    sessions_dir: Path,
    issue_number: int,
    log_path: Path,
) -> Path:
    sidecar_path = claude_sidecar_path(sessions_dir, issue_number, "api")
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=77777,
        started_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        adapter_kind="api",
        provider="example",
    )
    sidecar_path.write_text(json.dumps(record.to_dict()), encoding="utf-8")
    return sidecar_path


class _FakeGitHub:
    def __init__(self, config: OrchestratorConfig, issue_number: int) -> None:
        self.issues = [
            {
                "number": issue_number,
                "title": "Test issue",
                "url": f"https://example.test/issues/{issue_number}",
                "body": "Test",
                "labels": [{"name": config.labels.in_progress}],
            }
        ]
        self.prs: list[dict] = []

    def issue_list(self, labels=None, state=None):
        return self.issues

    def issue_view(self, number: int):
        for issue in self.issues:
            if issue["number"] == number:
                return issue
        raise ValueError(f"Issue {number} not found")

    def pr_list(self):
        return self.prs

    def add_issue_label(self, number: int, label: str) -> bool:
        return True

    def remove_issue_label(self, number: int, label: str) -> bool:
        return True


def _run_dead_lane(
    tmp_path: Path,
    sessions_dir: Path,
    config: OrchestratorConfig,
    state_file: Path,
    fake_gh: _FakeGitHub,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    from charlie_work import dead_worker_reap
    from charlie_work.workflow import _classify_dead_sessions_and_update_throttle_state

    monkeypatch.setattr(dead_worker_reap, "sweep_orphan_processes", lambda worktree_path: [])
    monkeypatch.setattr("charlie_work.worker.is_worker_alive", lambda record: False)
    monkeypatch.setattr("charlie_work.worker.is_session_alive", lambda record: False)
    monkeypatch.setattr("charlie_work.worker.real_activity_probe_for", lambda *args: None)

    _classify_dead_sessions_and_update_throttle_state(
        sessions_dir,
        state_file,
        fake_gh,
        config,
        write_gate=_wg(state_file),
    )
    return json.loads(state_file.read_text(encoding="utf-8"))


def test_dead_api_worker_literal_tmp_emits_warning_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead api session whose tee'd log shows a literal ``/tmp`` shell command
    emits exactly one warning-level ``worker_literal_tmp_path`` event carrying
    the offending commands — a diagnosable record, not a silent integrity bug."""
    issue_number = 1780
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Tee'd stream-json .claude.log — the shape api sessions always write.
    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    _write_jsonl(
        log_path,
        [
            _assistant_tool_use("gh pr view 5 --json body -q .body > /tmp/pr-body.md"),
            _assistant_tool_use("mktemp -d /tmp/work.XXXXXX"),
        ],
    )
    sidecar_path = _dead_api_worker(tmp_path, sessions_dir, issue_number, log_path)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": [], "issues": {}}), encoding="utf-8")
    fake_gh = _FakeGitHub(config, issue_number)

    state = _run_dead_lane(tmp_path, sessions_dir, config, state_file, fake_gh, monkeypatch)

    assert not sidecar_path.exists(), "sidecar must be reaped after dead-session classification"
    tmp_events = [e for e in state.get("events", []) if e.get("kind") == "worker_literal_tmp_path"]
    assert len(tmp_events) == 1, "the event fires once per session, not once per command"
    payload = tmp_events[0]["payload"]
    assert payload["issue_number"] == issue_number
    assert payload["adapter_kind"] == "api"
    assert payload["command_count"] == 2
    assert "gh pr view 5 --json body -q .body > /tmp/pr-body.md" in payload["commands"]


def test_dead_api_worker_compliant_tmpdir_emits_no_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead session whose commands only use ``$TMPDIR``/``mktemp -d`` —
    the instructed forms — emits nothing."""
    issue_number = 1780
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    _write_jsonl(
        log_path,
        [
            _assistant_tool_use('gh pr view 5 --json body -q .body > "$TMPDIR/pr-body.md"'),
            _assistant_tool_use("scratch=$(mktemp -d)"),
        ],
    )
    sidecar_path = _dead_api_worker(tmp_path, sessions_dir, issue_number, log_path)

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": [], "issues": {}}), encoding="utf-8")
    fake_gh = _FakeGitHub(config, issue_number)

    state = _run_dead_lane(tmp_path, sessions_dir, config, state_file, fake_gh, monkeypatch)

    assert not sidecar_path.exists()
    kinds = [e.get("kind") for e in state.get("events", [])]
    assert "worker_literal_tmp_path" not in kinds


def test_dead_claude_code_worker_plain_log_emits_no_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented residual gap: a claude-code session launched without
    ``tee_stream_json`` leaves a plain-text log with no tool_use evidence —
    the scan finds nothing rather than false-flagging prose."""
    issue_number = 1780
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    log_path.write_text(
        "Final summary: wrote scratch to /tmp during the run.\n",
        encoding="utf-8",
    )
    sidecar = claude_sidecar_path(sessions_dir, issue_number, "claude-code")
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=77777,
        started_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        adapter_kind="claude-code",
    )
    sidecar.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": [], "issues": {}}), encoding="utf-8")
    fake_gh = _FakeGitHub(config, issue_number)

    state = _run_dead_lane(tmp_path, sessions_dir, config, state_file, fake_gh, monkeypatch)

    assert not sidecar.exists()
    kinds = [e.get("kind") for e in state.get("events", [])]
    assert "worker_literal_tmp_path" not in kinds


def test_dead_claude_code_worker_stream_json_log_emits_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claude-code session launched WITH ``tee_stream_json`` writes the same
    stream-json log api sessions write — the detector covers it identically."""
    issue_number = 1780
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    log_path = sessions_dir / f"issue-{issue_number}.claude.log"
    _write_jsonl(
        log_path,
        [_assistant_tool_use("cp out.txt /tmp/leak.txt")],
    )
    sidecar = claude_sidecar_path(sessions_dir, issue_number, "claude-code")
    record = ClaudeWorkerRecord(
        issue_number=issue_number,
        branch=f"agent/issue-{issue_number}",
        worktree_path=str(tmp_path / "worktree"),
        prompt_path=str(tmp_path / "prompt.md"),
        command=("claude", "prompt.md"),
        pid=77777,
        started_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"),
        log_path=str(log_path),
        error=None,
        failure_kind=None,
        process_start_time=1710000000.0,
        reclaimed=None,
        adapter_kind="claude-code",
    )
    sidecar.write_text(json.dumps(record.to_dict()), encoding="utf-8")

    config = OrchestratorConfig(
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db"))
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"events": [], "issues": {}}), encoding="utf-8")
    fake_gh = _FakeGitHub(config, issue_number)

    state = _run_dead_lane(tmp_path, sessions_dir, config, state_file, fake_gh, monkeypatch)

    tmp_events = [e for e in state.get("events", []) if e.get("kind") == "worker_literal_tmp_path"]
    assert len(tmp_events) == 1
    assert tmp_events[0]["payload"]["adapter_kind"] == "claude-code"
