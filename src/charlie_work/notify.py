from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
                return NotifyResult(
                    ok=False, error=f"webhook returned status {response.status}"
                )
    except urllib.error.URLError as e:
        return NotifyResult(ok=False, error=f"webhook URL error: {e}")
    except (OSError, TimeoutError) as e:
        return NotifyResult(ok=False, error=f"webhook request failed: {e}")
    except Exception as e:
        return NotifyResult(ok=False, error=f"webhook unexpected error: {e}")


def _desktop_sink(config: Any, digest: AttentionDigest) -> NotifyResult:
    """OS-native toast notification.

    Windows: powershell -Command with Windows.UI.Notifications or msg.exe fallback.
    POSIX: notify-send if present, else ok=False.
    Never raises; all failures return NotifyResult(ok=False, error=...).
    """
    # Build a human-readable message
    transition_count = len(digest.transitions)
    if transition_count == 1:
        entry = digest.transitions[0]
        message = f"Issue #{entry.issue_number}: {entry.health} (was {entry.previous_health or 'unknown'})"
    else:
        message = f"{transition_count} issues need attention: {', '.join(str(e.issue_number) for e in digest.transitions)}"

    title = f"charlie-work: {digest.repo}"

    if os.name == "nt":
        # Windows: try PowerShell toast first, fall back to msg.exe
        try:
            # Try Windows.UI.Notifications toast (Windows 8+)
            ps_command = f"""
            Add-Type -AssemblyName Windows.UI.Notifications;
            $template = [Windows.UI.Notifications.ToastTemplateManager]::GetTemplateContent('ToastText02');
            $textNodes = $template.GetElementsByTagName('text');
            $textNodes[0].InnerText = '{title}';
            $textNodes[1].InnerText = '{message}';
            $toast = [Windows.UI.Notifications.ToastNotification]::new($template);
            $notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('charlie-work');
            $notifier.Show($toast);
            """
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return NotifyResult(ok=True)
        except (subprocess.SubprocessError, TimeoutError, OSError):
            # Fall back to msg.exe
            try:
                result = subprocess.run(
                    ["msg", "*", title + " - " + message],
                    capture_output=True,
                    text=True,
                    timeout=5,
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
                    }
                    for e in digest.transitions
                ]
            }
        )

        command = config.shell_command + (digest_json,)
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
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
