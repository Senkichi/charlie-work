"""Repository label bootstrap delegates for ``OrchestratorApp``.

Track 2 Phase B leaf L02 batch 4 (issue #1655, parent #1633, umbrella #1582).
Method bodies moved verbatim from ``OrchestratorApp`` in ``charlie_work.workflow``;
the ``workflow_delegation`` installer re-attaches each ``def`` onto the class.
"""

from __future__ import annotations

from charlie_work.github import GitHubError

import charlie_work.workflow as _wf


def _label_color(self, label: str) -> str:
    """Color for a label, derived from LabelConfig fields (no hard-coded names)."""
    labels = self.config.labels
    # The ready marker is green; every other workflow label shares the
    # default purple.
    if label == labels.ready:
        return "0E8A16"
    return "5319E7"


def _label_descriptions(self) -> dict[str, str]:
    """Human-readable description for every LabelConfig-derived label.

    Single source for the descriptions used by both the explicit
    ``charlie bootstrap-labels`` command and the automatic startup
    ensure (issue #1339). Keyed by the resolved label string from
    ``self.config.labels``, never a hard-coded literal.
    """
    labels = self.config.labels
    return {
        labels.ready: "Issue is ready for deterministic agentic automation.",
        labels.queued: "Issue is queued by the orchestrator.",
        labels.in_progress: "A worker is implementing this issue.",
        labels.pr_open: "A worker PR exists for this issue.",
        labels.reviewing: "The orchestrator is adversarially reviewing the worker PR.",
        labels.needs_rework: "The worker PR needs another implementation cycle.",
        labels.blocked: "Automation is blocked and needs intervention.",
        labels.done: "Automation completed and the issue was merged or resolved.",
        labels.human_needed: "A human product or security decision is needed.",
        labels.operator_queue: "A mechanical failure exhausted its automated retries; needs operator triage.",
        labels.prose_only_deps: "Issue has prose-only dependencies that need structured blocker declarations.",
        labels.merge_hold: "Approved PR is held out of the merge queue by operator request.",
    }


def _ensure_labels_core(self) -> _wf.CommandResult:
    """Idempotent ensure of every LabelConfig-derived label on the repo.

    Creates/updates each label in ``self.config.labels.all`` via
    ``gh.label_create`` (``--force`` → update-or-create, so colour and
    description drift is repaired on existing labels too), then verifies
    via ``gh.label_list``. The ensure set is derived entirely from
    ``LabelConfig`` fields — never a hard-coded list — so a new field
    ships its label with no extra wiring (issue #1339).

    Returns a ``CommandResult`` and never raises: a ``GitHubError`` from
    verification is reported in the result, not propagated. The CLI
    ``bootstrap-labels`` command and the automatic startup ensure both
    route through here.
    """
    labels = self.config.labels
    desired = list(labels.all)
    descriptions = self._label_descriptions()
    for label in desired:
        # A brand-new LabelConfig field may ship before its entry in
        # ``_label_descriptions`` is added; fall back to a generic
        # description so the label is still created (issue #1339 AC #3).
        # Never block the ensure on a missing description.
        description = descriptions.get(label, "Orchestrator-managed label.")
        self.gh.label_create(label, self._label_color(label), description)
    # Verify: check which labels actually exist after creation attempts.
    # label_create uses allow_failure=True, so silent failures are possible
    # (e.g. no auth, wrong repo). Don't report success we can't vouch for.
    try:
        live = {
            str(item.get("name") or "") for item in self.gh.label_list() if isinstance(item, dict)
        }
        missing = [name for name in desired if name not in live]
    except GitHubError as exc:
        return _wf.CommandResult(
            False,
            f"labels created but verification failed: {exc}",
            {"labels": desired, "missing": None},
        )
    if missing:
        return _wf.CommandResult(
            False,
            f"bootstrap incomplete — {len(missing)} label(s) still missing: {missing}",
            {"labels": desired, "missing": missing},
        )
    return _wf.CommandResult(True, "labels ensured", {"labels": desired, "missing": []})


def bootstrap_labels(self) -> _wf.CommandResult:
    return self._ensure_labels_core()
