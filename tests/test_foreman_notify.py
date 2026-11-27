"""``NotifyMessage`` sinks and the foreman's item notify pass (issue #3121).

``emit_message`` delivers the generic payload (title/body/command) through the
same four sinks as ``emit_digest``; ``notify_unnotified_items`` sends each open
item once, stamping ``notified_at`` on success and leaving the item unstamped —
so the next pass retries — on failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from charlie_work.config import NotifyConfig
from charlie_work.foreman.items import create_item, load_items
from charlie_work.foreman.notify import notify_unnotified_items
from charlie_work.foreman.store import scaffold_foreman
from charlie_work.notify import NotifyResult, NotifyMessage, emit_message


def _message(command: str | None = "charlie gang intervene g1 it-1 --choose 0") -> NotifyMessage:
    return NotifyMessage(title="foreman g1: decision it-1", body="Retry?", command=command)


def _foreman_root(fleet: Path, name: str = "g1") -> Path:
    created = scaffold_foreman(fleet, name)
    assert created.ok, created.error
    return created.value


def test_emit_message_file_sink_appends_message_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "sent.jsonl"
    config = NotifyConfig(enabled=True, sink="file", file_path=str(out))
    result = emit_message(config, _message())
    assert result.ok and result.error is None
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {
        "title": "foreman g1: decision it-1",
        "body": "Retry?",
        "command": "charlie gang intervene g1 it-1 --choose 0",
    }


def test_emit_message_webhook_posts_message_fields() -> None:
    config = NotifyConfig(enabled=True, sink="webhook", webhook_url="http://example.com/hook")
    with patch("charlie_work.notify.urllib.request.urlopen") as mock_urlopen:
        response = MagicMock()
        response.status = 200
        response.__enter__ = MagicMock(return_value=response)
        response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = response
        result = emit_message(config, _message())
    assert result.ok
    request = mock_urlopen.call_args[0][0]
    body = json.loads(request.data.decode("utf-8"))
    assert body["title"] == "foreman g1: decision it-1"
    assert body["command"] == "charlie gang intervene g1 it-1 --choose 0"


def test_emit_message_shell_sink_appends_message_json() -> None:
    config = NotifyConfig(enabled=True, sink="shell", shell_command=("echo",))
    with patch("charlie_work.notify.subprocess.run") as mock_run:
        completed = MagicMock()
        completed.returncode = 0
        mock_run.return_value = completed
        result = emit_message(config, _message())
    assert result.ok
    argv = mock_run.call_args[0][0]
    assert argv[:-1] == ("echo",)
    payload = json.loads(argv[-1])
    assert payload["title"] == "foreman g1: decision it-1"
    assert payload["command"] == "charlie gang intervene g1 it-1 --choose 0"


def test_emit_message_desktop_toast_carries_command(monkeypatch) -> None:
    sent: dict[str, str] = {}

    def fake_toast(title: str, message: str) -> NotifyResult:
        sent.update(title=title, message=message)
        return NotifyResult(ok=True)

    monkeypatch.setattr("charlie_work.notify._send_desktop_toast", fake_toast)
    result = emit_message(NotifyConfig(enabled=True, sink="desktop"), _message())
    assert result.ok
    assert "foreman g1: decision it-1" in sent["title"]
    assert "Retry?" in sent["message"]
    assert "charlie gang intervene g1 it-1 --choose 0" in sent["message"]


def test_emit_message_failing_disabled_and_unknown_sinks() -> None:
    # A failing sink is an error value, never a raise.
    result = emit_message(NotifyConfig(enabled=True, sink="file", file_path=""), _message())
    assert not result.ok and result.error
    # Disabled is a quiet no-op.
    result = emit_message(NotifyConfig(enabled=False, sink="file"), _message())
    assert result.ok and result.error == "notify disabled"
    # An unknown sink is a value too.
    result = emit_message(NotifyConfig(enabled=True, sink="pager"), _message())
    assert not result.ok and "unknown sink" in result.error


def test_notify_unnotified_items_sends_each_item_once(tmp_path: Path) -> None:
    root = _foreman_root(tmp_path)
    assert create_item(
        root, kind="decision", question="Retry?", options=("yes", "no"), recommended=1
    ).ok
    assert create_item(root, kind="report", text="all quiet").ok
    out = tmp_path / "sent.jsonl"
    config = NotifyConfig(enabled=True, sink="file", file_path=str(out))
    outcome = notify_unnotified_items(root, config)
    assert outcome.ok, outcome.error
    assert outcome.value.sent == ("it-1", "it-2")
    assert outcome.value.failed == ()
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {line["title"] for line in lines} == {
        "foreman g1: decision it-1",
        "foreman g1: report it-2",
    }
    # A decision without its own command tells the operator how to answer.
    assert lines[0]["command"] == "charlie gang intervene g1 it-1 --choose 1"
    # Stamped items never send twice.
    assert all(item.notified_at for item in load_items(root))
    outcome = notify_unnotified_items(root, config)
    assert outcome.value.sent == () and outcome.value.failed == ()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_notify_unnotified_items_sink_failure_retries_next_pass(tmp_path: Path) -> None:
    root = _foreman_root(tmp_path)
    assert create_item(root, kind="decision", question="Retry?", options=("yes", "no")).ok
    broken = NotifyConfig(
        enabled=True, sink="shell", shell_command=("definitely-missing-binary-xyz",)
    )
    outcome = notify_unnotified_items(root, broken)
    assert outcome.ok
    assert outcome.value.sent == ()
    assert outcome.value.failed and outcome.value.failed[0][0] == "it-1"
    assert load_items(root)[0].notified_at is None
    # The next pass with a working sink delivers the backlog.
    out = tmp_path / "sent.jsonl"
    outcome = notify_unnotified_items(
        root, NotifyConfig(enabled=True, sink="file", file_path=str(out))
    )
    assert outcome.value.sent == ("it-1",)
    assert load_items(root)[0].notified_at


def test_notify_unnotified_items_disabled_sends_and_stamps_nothing(tmp_path: Path) -> None:
    root = _foreman_root(tmp_path)
    assert create_item(root, kind="decision", question="Retry?", options=("yes", "no")).ok
    outcome = notify_unnotified_items(root, NotifyConfig(enabled=False, sink="file"))
    assert outcome.ok and outcome.value.sent == () and outcome.value.failed == ()
    assert load_items(root)[0].notified_at is None
