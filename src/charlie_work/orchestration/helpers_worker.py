"""Worker-summary and branch-naming delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 1 (issue #1652, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work.github import GitHubError
import charlie_work.workflow as _wf


def _branch_name(self, issue: dict[str, Any]) -> str:
    return f"{self.config.dispatch.branch_prefix}-{int(issue['number'])}-{_wf.slugify(str(issue.get('title') or 'work'))}"


def _summarize_worker(self, view, health, probe=None) -> dict[str, Any]:
    """Summarize a worker's state for the status() workers list.

    Args:
        view: WorkerView with worker state
        health: WorkerHealth enum value from classify_worker_health
        probe: Optional ``RealActivityProbe`` from
            ``real_activity_probe_for`` (the same corroboration probe the
            watchdog consults). When supplied, the summary surfaces the
            freshest corroborated activity timestamp/source and whether
            that signal is fresh within ``watchdog.stall_minutes`` -- so an
            alive-but-polling worker (stale sidecar log, fresh sessions.db
            / per-PID log) is visibly distinguishable from a genuinely
            stalled one (issue #1346). ``None`` keeps the legacy shape
            (no corroboration fields) for callers that did not compute a
            probe.

    Returns:
        Dict with worker summary fields for status() JSON output
    """
    from charlie_work.claude_code import parse_claude_events
    from charlie_work.post_mortem import RealActivityProbe, _events_path_from_log

    # Resolve repo_key: use view.repo_key if present, otherwise fall back to gh.name_with_owner()
    # This handles both fleet mode (repo_key populated by iter_workers) and single-repo mode
    repo = view.repo_key
    if not repo:
        try:
            repo = self.gh.name_with_owner()
        except GitHubError:
            # If gh fails, use a fallback to avoid breaking status()
            repo = "unknown"

    # Parse tool calls and usage for Claude Code sessions
    tool_calls = None
    tokens = None
    cost_usd = None

    if view.adapter_kind == "claude-code":
        # Canonical derivation (issue #329): supports both plain
        # issue-<n>.claude.log and rework-layout issue-<n>-rework.claude.log,
        # matching the events.jsonl sibling that claude_code actually writes.
        events_path = _events_path_from_log(Path(view.log_path))
        progress = parse_claude_events(events_path)
        if progress is not None:
            tool_calls = progress.tool_call_count
            tokens = progress.tokens
            cost_usd = progress.cost_usd
    # For devin sessions, these fields remain None (no structured stream)

    # Calculate budget remaining (if configured)
    budget_remaining = None
    if tokens is not None and self.config.watchdog.token_budget is not None:
        budget_remaining = max(0, self.config.watchdog.token_budget - tokens)
    elif cost_usd is not None and self.config.watchdog.cost_budget_usd is not None:
        budget_remaining = max(0, self.config.watchdog.cost_budget_usd - cost_usd)

    # Issue #1346: surface the watchdog's corroboration probe so an
    # alive-but-polling worker (stale sidecar log mtime, fresh
    # sessions.db / per-PID log / events.jsonl) is visibly distinguishable
    # from a genuinely stalled one. The probe is the same object
    # ``classify_worker_health`` consulted for ``health`` -- reusing it
    # here means the displayed classification and the displayed
    # corroboration come from one code path, never two. ``probe is None``
    # keeps the legacy shape for callers that did not compute one.
    corroboration_latest_at: str | None = None
    corroboration_latest_source: str | None = None
    corroboration_fresh: bool | None = None
    if isinstance(probe, RealActivityProbe):
        latest_ts = probe.latest_timestamp
        corroboration_latest_at = latest_ts.isoformat() if latest_ts is not None else None
        corroboration_latest_source = probe.latest_source
        corroboration_fresh = probe.is_fresh(self.config.watchdog.stall_minutes)

    return {
        "repo": repo,
        "issue": view.issue_number,
        "adapter": view.adapter_kind,
        "health": health.value,
        "runtime_seconds": view.runtime_seconds(),
        "last_activity_at": view.last_activity_at,
        "last_corroborated_activity_at": corroboration_latest_at,
        "last_corroborated_activity_source": corroboration_latest_source,
        "corroboration_fresh": corroboration_fresh,
        "tool_calls": tool_calls,
        "tokens": tokens,
        "cost_usd": cost_usd,
        "budget_remaining": budget_remaining,
    }
