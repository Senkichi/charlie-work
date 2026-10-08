"""Attention-digest build/emit pipeline for the dispatch lanes (issue #2600).

``_build_attention_digest`` is a verbatim move of the stateful digest builder
that used to live in ``workflow.py`` -- it diffs each issue's current health
against the persisted ``state_field`` baseline under ``state_lock`` and writes
the new baselines back. The ``_detect_*``/``_emit_*`` helpers are the
notify-gated probe + emit tail the fresh-dispatch
(``orchestration/dispatch_state.py``) and rework
(``orchestration/state_dispatch_rework.py``) lanes shared verbatim; moving the
tail here is also what keeps the rework delegate under its file-size ratchet
mark after the #2600 probe gate landed.

``workflow.py`` re-exports every symbol here via a facade import block (the
``dispatch_selection`` / ``escalation`` lineage of issue #1283 Phase A), so
every ``charlie_work.workflow.<name>`` import path and monkeypatch target
keeps resolving unchanged. Like the rest of that lineage this module must
never import ``charlie_work.workflow`` -- the facade already imports from
here, so the reverse edge would be the exact import cycle the pattern exists
to avoid; the names below bind their real defining modules (``.notify``,
``.dead_worker_sweep.effects_sessions``, ``.state``) directly instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .dead_worker_sweep.effects_sessions import _detect_stalled_sessions
from .notify import AttentionDigest, AttentionEntry, emit_digest


def _build_attention_digest(
    state_file: Path,
    health_transitions: dict[int, dict[str, Any]],
    repo: str,
    state_field: str = "health",
) -> AttentionDigest | None:
    """Build an AttentionDigest from health transitions observed in a pass.

    Args:
        state_file: Path to state.json for reading/writing per-issue health baseline
        health_transitions: Dict mapping issue_number to transition data:
            {
                issue_number: {
                    "adapter_kind": str,
                    "health": str,  # current health (e.g., "STALLED", "RUNAWAY", "DEAD")
                    "last_log_line": str | None,
                    "pid": int | None,
                    "terminal_tool": str | None,  # issue #261: post-mortem terminal tool (DEAD only)
                    "terminal_reason": str | None,  # issue #261: one-line terminal cause
                }
            }
        repo: Repository name for the digest
        state_field: The state["issues"][n] field to read/write for transition
            comparison. Defaults to "health"; callers tracking a separate alert
            dimension (e.g. merge_alert) can pass their own field name.

    Returns:
        AttentionDigest if there are transitions, None otherwise. Updates per-issue
        health field in state.json to the current health for transition comparison
        on the next pass.
    """
    if not health_transitions:
        return None

    from .state import load_state, save_state, state_lock
    from .state import utc_now

    entries: list[AttentionEntry] = []

    with state_lock(state_file):
        state = load_state(state_file)

        for issue_number, transition in health_transitions.items():
            current_health = transition["health"]

            # Read the last persisted health for this issue
            issue_key = str(issue_number)
            issue_entry = state.get("issues", {}).get(issue_key, {})
            last_health = issue_entry.get(state_field)

            # Only include if health changed (or no previous health persisted)
            if last_health != current_health:
                entries.append(
                    AttentionEntry(
                        issue_number=issue_number,
                        adapter_kind=transition["adapter_kind"],
                        health=current_health,
                        previous_health=last_health,
                        last_log_line=transition.get("last_log_line"),
                        pid=transition.get("pid"),
                        terminal_tool=transition.get("terminal_tool"),
                        terminal_reason=transition.get("terminal_reason"),
                    )
                )

                # Update the persisted health for this issue
                state["issues"][issue_key] = {
                    **issue_entry,
                    state_field: current_health,
                }

        # Save the updated health baselines
        if entries:
            save_state(state_file, state)

    if not entries:
        return None

    return AttentionDigest(
        generated_at=utc_now(),
        repo=repo,
        transitions=tuple(entries),
    )


def _detect_stalled_sessions_for_notify(app: Any) -> list[dict[str, Any]]:
    """Run the stalled-session probe only when the notify digest can consume it.

    Issue #2600: the probe is a per-worker real-activity scan (sessions.db
    open, per-PID log glob, worktree walk). In the rework lane its only
    consumer is the notify digest, so when ``notify.enabled`` is off (the
    default) the result would be discarded -- skip the scan rather than pay
    host-store I/O for it.
    """
    if not app.config.notify.enabled:
        return []
    return _detect_stalled_sessions(app._layout.sessions_dir, app.config)


def _emit_attention_digest(
    app: Any,
    transitions: dict[int, dict[str, Any]],
    *,
    state_field: str = "health",
) -> None:
    """Emit the attention digest for ``transitions`` when there is anything to send.

    Shared emit tail of the dispatch lanes: no-ops unless ``notify.enabled``
    is on and at least one transition was observed, then builds the stateful
    digest (which also persists the new per-issue baselines) and hands it to
    ``emit_digest``.
    """
    if not transitions or not app.config.notify.enabled:
        return
    digest = _build_attention_digest(
        app.paths.state_file,
        transitions,
        repo=app.repo_root.name,
        state_field=state_field,
    )
    if digest:
        emit_digest(app._layout.notify, digest)


def _emit_stalled_session_digest(app: Any, stalled_entries: list[dict[str, Any]]) -> None:
    """Emit the stalled-session health-transition digest, notify-gated.

    The dispatch lanes' "Emit notification digest" tail: converts the
    stalled-session probe entries into health transitions and emits through
    the shared attention-digest pipeline. ``stalled_entries`` may be empty --
    either because no worker stalled or because the caller skipped the probe
    under ``_detect_stalled_sessions_for_notify`` (issue #2600).
    """
    if not stalled_entries or not app.config.notify.enabled:
        return
    health_transitions: dict[int, dict[str, Any]] = {}
    for entry in stalled_entries:
        health_transitions[entry["issue"]] = {
            "adapter_kind": "unknown",  # Will be filled by #165's full supervisor
            "health": entry.get("health", "STALLED"),
            "last_log_line": None,
            "pid": entry.get("pid"),
            "terminal_tool": entry.get("terminal_tool"),
            "terminal_reason": entry.get("terminal_reason"),
        }
    _emit_attention_digest(app, health_transitions)
