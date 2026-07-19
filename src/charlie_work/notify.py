from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .subprocess_runner import no_console_window_kwargs


@dataclass(frozen=True)
class NotifyResult:
    """Result of a notification sink operation.

    Errors-as-values: never raises. Callers check ``ok`` or ``error``.
    """

    ok: bool
    error: str | None = None


@dataclass(frozen=True)
class AttentionEntry:
    """Single issue's health transition in a digest."""

    issue_number: int
    adapter_kind: str
    health: str  # WorkerHealth value, e.g. "STALLED" | "RUNAWAY" | "DEAD"
    previous_health: str | None
    last_log_line: str | None
    pid: int | None
    # Issue #261: terminal tool call + one-line reason recovered from the
    # Devin CLI session store post-mortem, when extraction succeeded (e.g.
    # "bash" / "blocked by push-gate hook"). None when unset — for a live
    # (non-DEAD) transition, or when post-mortem extraction found nothing.
    terminal_tool: str | None = None
    terminal_reason: str | None = None


@dataclass(frozen=True)
class AttentionDigest:
    """Digest of needs-attention health transitions in a pass."""

    generated_at: str
    repo: str
    transitions: tuple[AttentionEntry, ...]  # one per issue whose health changed this pass


def _webhook_sink(config: Any, digest: AttentionDigest) -> NotifyResult:
    """POST digest as JSON to webhook URL.

    Uses stdlib urllib.request.urlopen (no new dependency). Timeout-bounded.
    Non-2xx or exception returns NotifyResult(ok=False, error=...), never raises.
    """
    if not config.webhook_url:
        return NotifyResult(ok=False, error="webhook_url is empty")

    try:
        payload = json.dumps(
            {
                "generated_at": digest.generated_at,
                "repo": digest.repo,
                "transitions": [
                    {
                        "issue_number": e.issue_number,
                        "adapter_kind": e.adapter_kind,
                        "health": e.health,
                        "previous_health": e.previous_health,
                        "last_log_line": e.last_log_line,
                        "pid": e.pid,
                        "terminal_tool": e.terminal_tool,
                        "terminal_reason": e.terminal_reason,
                    }
                    for e in digest.transitions
                ],
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            config.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            if 200 <= response.status < 300:
                return NotifyResult(ok=True)
            else:
                return NotifyResult(ok=False, error=f"webhook returned status {response.status}")
    except urllib.error.URLError as e:
        return NotifyResult(ok=False, error=f"webhook URL error: {e}")
    except (OSError, TimeoutError) as e:
        return NotifyResult(ok=False, error=f"webhook request failed: {e}")
    except Exception as e:
        return NotifyResult(ok=False, error=f"webhook unexpected error: {e}")


_DESKTOP_SEVERITIES = frozenset({"STALLED", "RUNAWAY", "DEAD", "ERROR"})
_MAX_DESKTOP_REASON_LENGTH = 80


def _truncate_desktop_reason(text: str | None) -> str | None:
    if not text:
        return text
    if len(text) <= _MAX_DESKTOP_REASON_LENGTH:
        return text
    return text[: _MAX_DESKTOP_REASON_LENGTH - 3] + "..."


def _desktop_context_line(entry: AttentionEntry) -> str:
    if entry.issue_number > 0:
        return f"Issue #{entry.issue_number}"
    return entry.adapter_kind or "fleet"


def _desktop_message(digest: AttentionDigest) -> tuple[str, str] | None:
    """Build the title and message for a desktop toast.

    Filters out benign flow-control transitions (e.g. SKIPPED) and uses repo
    context for pass-level entries that have no real issue number.
    """
    filtered = tuple(e for e in digest.transitions if e.health in _DESKTOP_SEVERITIES)
    if not filtered:
        return None

    if len(filtered) == 1:
        entry = filtered[0]
        context = _desktop_context_line(entry)
        message = f"{context}: {entry.health}"
        if entry.previous_health is not None:
            message += f" (was {entry.previous_health})"

        reason_parts: list[str] = []
        if entry.terminal_reason:
            reason_parts.append(entry.terminal_reason)
        last_line = _truncate_desktop_reason(entry.last_log_line)
        if last_line and last_line != entry.terminal_reason:
            reason_parts.append(last_line)
        if reason_parts:
            message += " — " + " — ".join(reason_parts)
    else:
        contexts = [
            str(e.issue_number) if e.issue_number > 0 else (e.adapter_kind or "fleet")
            for e in filtered
        ]
        message = f"{len(filtered)} issues need attention: {', '.join(contexts)}"

    return f"charlie-work: {digest.repo}", message


def _ps_single_quote(value: str) -> str:
    """Escape a value for insertion into a PowerShell single-quoted string."""
    value = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return value.replace("'", "''")


def _desktop_sink(config: Any, digest: AttentionDigest) -> NotifyResult:
    """OS-native toast notification.

    Severity-gates to genuine attention transitions (STALLED / RUNAWAY / DEAD / ERROR).
    Benign flow-control events (e.g. SKIPPED) are silently dropped so they do not
    spam the operator; they are still written to file/webhook sinks by emit_digest.

    Windows: PowerShell with WinRT Windows.UI.Notifications projection, falling back to msg.exe.
    POSIX: notify-send if present, else ok=False.
    Never raises; all failures return NotifyResult(ok=False, error=...).
    """
    rendered = _desktop_message(digest)
    if rendered is None:
        return NotifyResult(ok=True, error="no desktop-severity transitions")
    title, message = rendered

    if os.name == "nt":
        # Windows: try PowerShell toast first, fall back to msg.exe
        try:
            ps_command = f"""
$ErrorActionPreference = 'Stop'
[void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]
[void][Windows.UI.Notifications.ToastNotification, Windows.UI.Notifications, ContentType=WindowsRuntime]
[void][Windows.UI.Notifications.ToastTemplateType, Windows.UI.Notifications, ContentType=WindowsRuntime]
[void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType=WindowsRuntime]

$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$textNodes = $template.GetElementsByTagName('text')
$textNodes.Item(0).InnerText = '{_ps_single_quote(title)}'
$textNodes.Item(1).InnerText = '{_ps_single_quote(message)}'
$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
$notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('charlie-work')
$notifier.Show($toast)
"""
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=10,
                **no_console_window_kwargs(),
            )
            if result.returncode == 0:
                return NotifyResult(ok=True)
        except (subprocess.SubprocessError, TimeoutError, OSError):
            pass

        try:
            result = subprocess.run(
                ["msg", "*", title + " - " + message],
                capture_output=True,
                text=True,
                timeout=5,
                **no_console_window_kwargs(),
            )
            # msg.exe returns 0 on success even if no active session, treat as ok
            return NotifyResult(ok=True)
        except (subprocess.SubprocessError, TimeoutError, OSError) as e:
            return NotifyResult(ok=False, error=f"msg.exe failed: {e}")
    else:
        # POSIX: try notify-send
        try:
            result = subprocess.run(
                ["notify-send", title, message],
                capture_output=True,
                text=True,
                timeout=5,
                **no_console_window_kwargs(),
            )
            if result.returncode == 0:
                return NotifyResult(ok=True)
            else:
                return NotifyResult(
                    ok=False, error=f"notify-send failed with exit code {result.returncode}"
                )
        except (subprocess.SubprocessError, TimeoutError, OSError) as e:
            return NotifyResult(ok=False, error=f"notify-send not found or failed: {e}")

    return NotifyResult(ok=False, error="desktop notification failed")


def _shell_sink(config: Any, digest: AttentionDigest) -> NotifyResult:
    """Run shell command with digest JSON as argument.

    Command is a tuple (no shell string injection surface). Digest JSON is appended
    as the final argument. Timeout-bounded. Non-zero exit returns ok=False, never raises.
    """
    if not config.shell_command:
        return NotifyResult(ok=False, error="shell_command is empty")

    try:
        digest_json = json.dumps(
            {
                "generated_at": digest.generated_at,
                "repo": digest.repo,
                "transitions": [
                    {
                        "issue_number": e.issue_number,
                        "adapter_kind": e.adapter_kind,
                        "health": e.health,
                        "previous_health": e.previous_health,
                        "last_log_line": e.last_log_line,
                        "pid": e.pid,
                        "terminal_tool": e.terminal_tool,
                        "terminal_reason": e.terminal_reason,
                    }
                    for e in digest.transitions
                ],
            }
        )

        command = config.shell_command + (digest_json,)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            **no_console_window_kwargs(),
        )

        if result.returncode == 0:
            return NotifyResult(ok=True)
        else:
            return NotifyResult(
                ok=False,
                error=f"shell command exited with code {result.returncode}: {result.stderr[:200] if result.stderr else ''}",
            )
    except (subprocess.SubprocessError, TimeoutError, OSError) as e:
        return NotifyResult(ok=False, error=f"shell command failed: {e}")
    except Exception as e:
        return NotifyResult(ok=False, error=f"shell unexpected error: {e}")


def _file_sink(config: Any, digest: AttentionDigest) -> NotifyResult:
    """Append digest as JSONL to file path.

    Append-mode writes are inherently safe for single-writer JSONL (no read-modify-write,
    so state.state_lock is not required for this sink specifically). Each line is
    independently json.loads-able. Never raises; failures return NotifyResult(ok=False).
    """
    if not config.file_path:
        return NotifyResult(ok=False, error="file_path is empty")

    try:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        digest_dict = {
            "generated_at": digest.generated_at,
            "repo": digest.repo,
            "transitions": [
                {
                    "issue_number": e.issue_number,
                    "adapter_kind": e.adapter_kind,
                    "health": e.health,
                    "previous_health": e.previous_health,
                    "last_log_line": e.last_log_line,
                    "pid": e.pid,
                    "terminal_tool": e.terminal_tool,
                    "terminal_reason": e.terminal_reason,
                }
                for e in digest.transitions
            ],
        }

        with file_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(digest_dict) + "\n")

        return NotifyResult(ok=True)
    except (OSError, IOError) as e:
        return NotifyResult(ok=False, error=f"file write failed: {e}")
    except Exception as e:
        return NotifyResult(ok=False, error=f"file unexpected error: {e}")


def emit_digest(config: Any, digest: AttentionDigest) -> NotifyResult:
    """Emit a needs-attention digest to the configured sink.

    Called once at the end of any pass that observed at least one needs-attention
    transition. Never mutates labels or state — read-only consumer of the pass's
    own result. Sink failures never fail the pass; errors are returned as values.
    """
    if not config.enabled:
        return NotifyResult(ok=True, error="notify disabled")

    sink = config.sink.lower()

    if sink == "webhook":
        return _webhook_sink(config, digest)
    elif sink == "desktop":
        return _desktop_sink(config, digest)
    elif sink == "shell":
        return _shell_sink(config, digest)
    elif sink == "file":
        return _file_sink(config, digest)
    else:
        return NotifyResult(ok=False, error=f"unknown sink: {sink}")
