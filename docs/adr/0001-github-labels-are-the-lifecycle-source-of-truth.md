# GitHub labels are the lifecycle source of truth

An issue's lifecycle state is recorded in its `agent:*` GitHub labels, and those labels are the only authoritative record. `.var/charlie-work/state.json` (the state cache) is a derived mirror of lifecycle and PR state, so the true state survives an orchestrator crash, a reboot, or a different operator's machine, and `charlie roll-call` can rebuild it by re-querying GitHub (`docs/ARCHITECTURE.md`, "Hub-and-spoke model").

## Consequences

- Label writes go through one enforcement point, `labels.transition()`; workflow code names a lifecycle event and never edits individual labels. Label strings are read from `LabelConfig`, never hard-coded (`CLAUDE.md`).
- Dispatchability is decided from labels: `_is_dispatchable()` requires the ready label and no terminal or active label.
- The state cache is written with merge-update, never dict-replace. A wholesale replace erased a recorded review `decision` in production (PR #497, `docs/ARCHITECTURE.md`, "Invariants").
- Drift between labels and reality is repaired, not tolerated. `reconcile.py` detects it (for example a PR a human merged by hand, leaving `agent:in-progress` in place). The escalated-label self-heal sweep (`_repair_escalated_labels`, #586, made reachable with review dispatch off by #1088) re-applies an escalation label whose original `transition()` failed.
- That sweep is the one place the cache leads: an escalated issue is parked in `state.json` first and the sweep converges the label toward it. Labels stay authoritative for dispatch, but a failed label write can leave an issue parked in the cache and invisible on GitHub until the sweep runs.
- A backend with no remote (`local_issues.enabled`) keeps the same model, but its "labels" live in issue-file frontmatter behind `LocalFileGitHub`.
