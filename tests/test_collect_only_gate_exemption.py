"""Tests for issue #1686: operator-applied collect-gate exemption.

Split out of ``test_collect_only_gate.py`` (that file is over the
file-size ratchet's cap and cannot grow): everything here covers the
exemption surface added by #1686 -- the pure exemption model
(``resolve_collect_gate_exemption``, the log marker emit/parse pair,
``render_gate_report``'s exemption variants), the CLI command layer
(``--pr`` + the live-labels query, fail-closed semantics, waived-findings
output), and the ci.yml workflow shape the exemption depends on.

Shared CLI-command fixtures (``_make_cli_args``/``_apply_cli_mocks``/
``_CI_YML``) are imported from ``_collect_gate_helpers`` -- the
``tests/_*.py`` shared-fixture seam ``_fakes_github`` already uses.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from charlie_work.collect_only_gate import (
    COLLECT_ONLY_GATE_CHECK_NAME,
    CollectGateExemption,
    CollectOnlyFinding,
    CollectOnlyResult,
    exemption_log_marker,
    parse_exemption_log_marker,
    render_gate_report,
    resolve_collect_gate_exemption,
)
from charlie_work.collect_only_gate_command import run_collect_only_check_command
from _collect_gate_helpers import _apply_cli_mocks, _CI_YML, _make_cli_args


# ---------------------------------------------------------------------------
# CLI command exemption (issue #1686): --pr + live labels query
# ---------------------------------------------------------------------------

_DROPPED_LEAF_BASE = "tests/test_foo.py::test_a\ntests/test_foo.py::test_b\n"
_DROPPED_LEAF_HEAD = "tests/test_foo.py::test_a\n"


def _gh_returning_pr(pr: dict | None):
    """A ``gh`` stub whose ``pr_view`` returns *pr* (the live API answer)."""
    from types import SimpleNamespace

    return SimpleNamespace(pr_view=lambda number, fields=None: pr)


def _write_dropped_leaf(tmp_path: Path) -> None:
    (tmp_path / "base_collect.txt").write_text(_DROPPED_LEAF_BASE, encoding="utf-8")
    (tmp_path / "head_collect.txt").write_text(_DROPPED_LEAF_HEAD, encoding="utf-8")


def test_cli_exempt_label_waives_failing_findings(monkeypatch, tmp_path: Path) -> None:
    """Label present on the live query + a dropped leaf -> exit 0, every
    waived finding printed (kind + leaf), the label named as the waiving
    authority, and the log marker emitted."""
    _write_dropped_leaf(tmp_path)
    _apply_cli_mocks(
        monkeypatch,
        tmp_path,
        gh=_gh_returning_pr({"labels": [{"name": "collect-gate-exempt"}]}),
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is True
    assert "waiv" in result.message.lower()
    assert "collect-gate-exempt" in result.message
    assert "removed" in result.message and "test_b" in result.message
    assert "COLLECT-GATE-EXEMPTION v1 " in result.message
    marker_line = next(
        line for line in result.message.splitlines() if "COLLECT-GATE-EXEMPTION" in line
    )
    payload = parse_exemption_log_marker(marker_line)
    assert payload is not None and payload["active"] is True
    # A vanished leaf produces TWO enforced findings: ``removed`` (clause 1)
    # and ``missing_sibling`` (clause 2) -- both are waived.
    assert [f["leaf_name"] for f in payload["waived"]] == ["test_b", "test_b"]
    assert {f["kind"] for f in payload["waived"]} == {"removed", "missing_sibling"}
    assert result.data["exemption"]["active"] is True


def test_cli_no_exempt_label_preserves_failure(monkeypatch, tmp_path: Path) -> None:
    """--pr given but the label is absent -> identical failure behavior."""
    _write_dropped_leaf(tmp_path)
    _apply_cli_mocks(
        monkeypatch,
        tmp_path,
        gh=_gh_returning_pr({"labels": [{"name": "unrelated"}]}),
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is False
    assert "removed" in result.message
    assert result.data["exemption"]["active"] is False


def test_cli_labels_query_failure_fails_closed(monkeypatch, tmp_path: Path) -> None:
    """The labels query erroring/emptying out resolves to NOT exempt, the
    gate still fails on the dropped leaf, and the output says why."""
    _write_dropped_leaf(tmp_path)
    _apply_cli_mocks(monkeypatch, tmp_path, gh=_gh_returning_pr(None))
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is False
    assert "exemption" in result.message.lower()
    assert "could not fetch" in result.message or "query failed" in result.message
    assert result.data["exemption"]["active"] is False


def test_cli_labels_query_exception_fails_closed(monkeypatch, tmp_path: Path) -> None:
    """``pr_view`` raising (``GitHubError`` -- the real client's contract on
    ``gh`` failure, since it calls ``run`` without ``allow_failure``) must
    NOT propagate: the exemption resolves to NOT exempt, the gate still
    reports the dropped leaf, and the output names the query failure."""
    from types import SimpleNamespace

    def _raising_pr_view(number, fields=None):
        raise RuntimeError("GraphQL: Resource not accessible by integration")

    _write_dropped_leaf(tmp_path)
    _apply_cli_mocks(monkeypatch, tmp_path, gh=SimpleNamespace(pr_view=_raising_pr_view))
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is False
    assert "exemption" in result.message.lower()
    assert "query failed" in result.message or "fail closed" in result.message
    assert "removed" in result.message and "test_b" in result.message
    assert result.data["exemption"]["active"] is False


def test_cli_exempt_label_with_no_findings_reports_stale(monkeypatch, tmp_path: Path) -> None:
    """Label present but nothing to waive -> pass + an explicit "nothing to
    waive" note so a stale label is visible."""
    (tmp_path / "base_collect.txt").write_text("tests/test_foo.py::test_a\n", encoding="utf-8")
    (tmp_path / "head_collect.txt").write_text("tests/test_foo.py::test_a\n", encoding="utf-8")
    _apply_cli_mocks(
        monkeypatch,
        tmp_path,
        gh=_gh_returning_pr({"labels": [{"name": "collect-gate-exempt"}]}),
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is True
    assert "nothing to waive" in result.message.lower() or "stale" in result.message.lower()


def test_cli_no_pr_means_exemption_not_evaluated(monkeypatch, tmp_path: Path) -> None:
    """Without --pr the command does not evaluate the exemption at all:
    identical to pre-#1686 behavior (still fails on the dropped leaf)."""
    _write_dropped_leaf(tmp_path)
    _apply_cli_mocks(monkeypatch, tmp_path)
    result = run_collect_only_check_command(_make_cli_args(tmp_path))
    assert result.ok is False
    assert "exemption" not in result.data or result.data["exemption"] is None


def test_cli_exemption_uses_configured_label_name(monkeypatch, tmp_path: Path) -> None:
    """The label name comes from LabelConfig, not a hardcoded literal: a
    configured rename is honored."""
    from charlie_work.config import LabelConfig, OrchestratorConfig

    _write_dropped_leaf(tmp_path)
    config = OrchestratorConfig(labels=LabelConfig(collect_gate_exempt="custom-exempt"))
    _apply_cli_mocks(
        monkeypatch,
        tmp_path,
        gh=_gh_returning_pr({"labels": [{"name": "custom-exempt"}]}),
        config=config,
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is True
    assert "custom-exempt" in result.message


def test_cli_worker_authored_content_cannot_self_grant(monkeypatch, tmp_path: Path) -> None:
    """The exemption cannot ride a channel the PR author controls: a PR body
    (or title/commit/tree file) containing the label's name is NOT a grant --
    only the live labels query resolves the exemption."""
    _write_dropped_leaf(tmp_path)
    _apply_cli_mocks(
        monkeypatch,
        tmp_path,
        gh=_gh_returning_pr(
            {
                "labels": [],
                "body": "Collect-exempt: this PR renames tests\ncollect-gate-exempt",
                "title": "collect-gate-exempt",
            }
        ),
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is False
    assert result.data["exemption"]["active"] is False


def test_cli_exemption_resolved_from_query_not_event_payload(monkeypatch, tmp_path: Path) -> None:
    """Live-read control: the verdict comes from the labels query's answer.
    A payload snapshot that did not list the label cannot mask a live grant,
    and a stale payload listing it cannot forge one."""
    _write_dropped_leaf(tmp_path)
    # The live query says the label IS present -> exempt, regardless of any
    # event-payload state (which this command never reads at all).
    _apply_cli_mocks(
        monkeypatch,
        tmp_path,
        gh=_gh_returning_pr({"labels": [{"name": "collect-gate-exempt"}]}),
    )
    result = run_collect_only_check_command(_make_cli_args(tmp_path, pr=1697))
    assert result.ok is True


# ---------------------------------------------------------------------------
# Operator exemption model (issue #1686) — pure resolution + marker + rendering
# ---------------------------------------------------------------------------

_EXEMPT_LABEL = "collect-gate-exempt"


def _failing_result() -> CollectOnlyResult:
    """A result with one enforced finding and one reported-only finding."""
    return CollectOnlyResult(
        base_leaf_counts=Counter({"test_a": 1, "test_b": 1}),
        head_leaf_counts=Counter({"test_a": 1, "test_new": 1}),
        findings=(
            CollectOnlyFinding(
                kind="removed",
                leaf_name="test_b",
                detail="leaf name present at base but not at head (base count=1, head count=0)",
                base_count=1,
                head_count=0,
            ),
            CollectOnlyFinding(
                kind="added",
                leaf_name="test_new",
                detail="leaf name present at head but not at base (base count=0, head count=1)",
                base_count=0,
                head_count=1,
            ),
        ),
    )


def test_resolve_exemption_returns_none_without_pr() -> None:
    """No PR number means the exemption was never evaluated -- the gate
    behaves exactly as it did before #1686."""
    exemption = resolve_collect_gate_exemption(
        exemption_label=_EXEMPT_LABEL,
        pr_number=None,
        pr_labels=None,
    )
    assert exemption is None


def test_resolve_exemption_active_when_label_present() -> None:
    exemption = resolve_collect_gate_exemption(
        exemption_label=_EXEMPT_LABEL,
        pr_number=1697,
        pr_labels={"collect-gate-exempt", "other-label"},
    )
    assert exemption is not None
    assert exemption.active is True
    assert _EXEMPT_LABEL in exemption.detail
    assert "1697" in exemption.detail


def test_resolve_exemption_inactive_when_label_absent() -> None:
    exemption = resolve_collect_gate_exemption(
        exemption_label=_EXEMPT_LABEL,
        pr_number=1697,
        pr_labels={"other-label"},
    )
    assert exemption is not None
    assert exemption.active is False
    assert "not present" in exemption.detail


def test_resolve_exemption_fails_closed_on_query_failure() -> None:
    """A failed/empty labels query resolves to NOT exempt and says why --
    an API error must never be read as an exemption."""
    exemption = resolve_collect_gate_exemption(
        exemption_label=_EXEMPT_LABEL,
        pr_number=1697,
        pr_labels=None,
        query_error="could not fetch PR labels",
    )
    assert exemption is not None
    assert exemption.active is False
    assert "could not fetch PR labels" in exemption.detail
    assert "fail" in exemption.detail.lower() or "not granted" in exemption.detail


def test_exemption_log_marker_round_trip() -> None:
    """The marker is a single line the review packet can parse back out of
    the Actions job log."""
    exemption = CollectGateExemption(
        label=_EXEMPT_LABEL,
        active=True,
        detail="label `collect-gate-exempt` present on PR #1697",
    )
    waived = (
        CollectOnlyFinding(kind="removed", leaf_name="test_b", source_module="tests/test_foo.py"),
        CollectOnlyFinding(kind="missing_sibling", leaf_name="test_c"),
    )
    line = exemption_log_marker(exemption, waived)
    assert "\n" not in line
    payload = parse_exemption_log_marker(line)
    assert payload is not None
    assert payload["label"] == _EXEMPT_LABEL
    assert payload["active"] is True
    assert [f["leaf_name"] for f in payload["waived"]] == ["test_b", "test_c"]
    assert payload["waived"][0]["kind"] == "removed"
    assert payload["waived"][0]["source_module"] == "tests/test_foo.py"


def test_parse_exemption_log_marker_ignores_log_noise() -> None:
    """GitHub Actions prefixes every job-log line with a timestamp; the marker
    must still be found on the line and only the last marker wins."""
    import json

    inner = json.dumps(
        {"v": 1, "label": _EXEMPT_LABEL, "active": True, "detail": "d", "waived": []}
    )
    log = (
        "2026-09-17T20:00:00.1Z some other output\n"
        f"2026-09-17T20:00:01.2Z COLLECT-GATE-EXEMPTION v1 {inner}\n"
        "2026-09-17T20:00:02.3Z trailing noise\n"
    )
    payload = parse_exemption_log_marker(log)
    assert payload is not None
    assert payload["active"] is True


def test_parse_exemption_log_marker_absent_or_malformed() -> None:
    assert parse_exemption_log_marker("no marker here\n") is None
    assert parse_exemption_log_marker("COLLECT-GATE-EXEMPTION v1 {not json}\n") is None
    assert parse_exemption_log_marker('COLLECT-GATE-EXEMPTION v1 {"v": 2}\n') is None


def test_render_gate_report_with_active_exemption_waives_failures() -> None:
    """With the exemption active the report still lists every failing finding
    (kind + leaf) but marks them waived by the operator label."""
    result = _failing_result()
    exemption = CollectGateExemption(
        label=_EXEMPT_LABEL, active=True, detail="present on PR #1697"
    )
    report = render_gate_report(result, exemption=exemption)
    assert "waiv" in report.lower()
    assert _EXEMPT_LABEL in report
    assert "removed" in report
    assert "test_b" in report
    # The failing findings are listed as waived, not as gate-fatal.
    assert "failing finding(s)" not in report or "waived" in report.lower()


def test_render_gate_report_with_active_exemption_no_findings_is_stale_note() -> None:
    """Label present but nothing to waive: say so, making a stale label
    visible to the reviewer."""
    result = CollectOnlyResult(
        base_leaf_counts=Counter({"test_a": 1}),
        head_leaf_counts=Counter({"test_a": 1}),
        findings=(),
    )
    exemption = CollectGateExemption(
        label=_EXEMPT_LABEL, active=True, detail="present on PR #1697"
    )
    report = render_gate_report(result, exemption=exemption)
    assert _EXEMPT_LABEL in report
    assert "nothing to waive" in report.lower() or "stale" in report.lower()


def test_render_gate_report_with_inactive_exemption_states_reason() -> None:
    """Evaluated but not granted: the report still fails normally AND states
    why the exemption was not applied."""
    result = _failing_result()
    exemption = CollectGateExemption(
        label=_EXEMPT_LABEL,
        active=False,
        detail="labels query failed for PR #1697: could not fetch PR labels (fail closed)",
    )
    report = render_gate_report(result, exemption=exemption)
    assert "failing finding(s)" in report  # normal failure output preserved
    assert "could not fetch PR labels" in report


def test_render_gate_report_without_exemption_is_unchanged() -> None:
    """exemption=None (no --pr) renders exactly the pre-#1686 report."""
    result = _failing_result()
    report = render_gate_report(result)
    assert "exemption" not in report.lower()
    assert "failing finding(s)" in report


# Issue #1686 -- operator exemption workflow shape
# ---------------------------------------------------------------------------


def test_exempt_label_literal_lives_only_in_labelconfig() -> None:
    """The default label string is declared exactly once (issue #1686).

    ``LabelConfig.collect_gate_exempt`` is the single source of truth --
    every consumer reads ``config.labels.collect_gate_exempt`` so a renamed
    label works via config alone. A second literal in ``src/`` would be a
    silently divergent re-declaration: the CLAUDE.md invariant ("label
    state-machine names come from ``LabelConfig``") applied to the
    exemption label too.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "charlie_work"
    offenders: list[str] = []
    for py_file in src_root.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        if "collect-gate-exempt" in py_file.read_text(encoding="utf-8"):
            offenders.append(str(py_file.relative_to(src_root)))
    assert offenders == ["config.py"], (
        f"the 'collect-gate-exempt' literal must live only in config.py's "
        f"LabelConfig default -- found also in {offenders}; read "
        f"config.labels.collect_gate_exempt instead of re-declaring it"
    )


def _gate_job_and_step() -> tuple[dict, dict, dict]:
    workflow = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    job = workflow["jobs"]["collect-only-gate"]
    by_name = {s["name"]: s for s in job["steps"] if "name" in s}
    return workflow, job, by_name["Run collect-only gate"]


def test_gate_check_name_matches_wire_constant() -> None:
    """The job's ``name:`` IS the check-run name the review packet searches
    for on the PR's head-pinned check list (issue #1686).  Renaming the job
    without updating ``COLLECT_ONLY_GATE_CHECK_NAME`` would silently blind
    the packet's exemption-evidence lookup; this pins both ends of the
    wire contract together.
    """
    _, job, _ = _gate_job_and_step()
    assert job["name"] == COLLECT_ONLY_GATE_CHECK_NAME


def test_exemption_gate_step_passes_pr_and_token() -> None:
    """``--pr`` + ``GH_TOKEN`` wire the live-labels query into the gate step.

    The gate resolves the operator exemption by querying the PR's labels
    through ``gh pr view`` inside the command (issue #1686).  That needs
    two things in the workflow: the PR number on the command line, and
    ``GH_TOKEN`` in the step's environment so the preinstalled ``gh``
    binary authenticates -- the same env-wiring the closing-keyword gate
    step uses.
    """
    _, _, gate_step = _gate_job_and_step()

    run_block = gate_step["run"]
    assert "--pr ${{ github.event.pull_request.number }}" in run_block, (
        "ci.yml 'Run collect-only gate' must pass --pr "
        "${{ github.event.pull_request.number }} so the gate can query the "
        "PR's live labels for the operator exemption (issue #1686)"
    )
    env = gate_step.get("env") or {}
    assert env.get("GH_TOKEN") == "${{ github.token }}", (
        "ci.yml 'Run collect-only gate' must set env GH_TOKEN: "
        "${{ github.token }} -- the gate's `gh pr view` labels query needs "
        "an authenticated gh CLI (same pattern as the closing-keyword "
        "gate step)"
    )


def test_exemption_job_has_pull_requests_read_permission() -> None:
    """The job needs ``pull-requests: read`` for the live-labels query.

    ``gh pr view`` issues a GraphQL pullRequest query, which the default
    ``contents: read`` token cannot run ("Resource not accessible by
    integration" -- the closing-keyword gate's documented failure).  The
    job-level permissions block must keep ``contents: read`` (checkout)
    alongside the new grant because an explicit block makes every unlisted
    scope ``none``.
    """
    _, job, _ = _gate_job_and_step()

    perms = job.get("permissions") or {}
    assert perms.get("pull-requests") == "read", (
        "ci.yml collect-only-gate job must grant pull-requests: read -- "
        "the exemption's live-labels query is a GraphQL pullRequest call "
        "(issue #1686)"
    )
    assert perms.get("contents") == "read", (
        "ci.yml collect-only-gate job must keep contents: read in its "
        "permissions block -- an explicit permissions key zeroes every "
        "unlisted scope, and actions/checkout@v5 needs contents: read"
    )


def test_exemption_does_not_skip_job_or_add_labeled_trigger() -> None:
    """No label-driven skip, no ``labeled`` activity type (issue #1686).

    The issue's contract: the gate must still run to completion when the
    label is present -- it prints every waived finding rather than being
    skipped -- and the workflow must NOT add ``labeled`` to the
    pull_request trigger types, because labels are read LIVE at run time
    precisely so an operator can apply the label and RERUN the failed
    job.  A ``labeled`` trigger would also fire CI on every unrelated
    label change.
    """
    workflow, job, gate_step = _gate_job_and_step()

    job_if = str(job.get("if", ""))
    assert "label" not in job_if.lower(), (
        "collect-only-gate job `if:` must not inspect labels -- the gate "
        "runs to completion and the label waives findings inside the "
        "command (issue #1686), never skips the job"
    )
    step_if = str(gate_step.get("if", ""))
    assert "label" not in step_if.lower(), (
        "Run collect-only gate step `if:` must not inspect labels -- "
        "with the label present the gate still runs the full comparison "
        "and prints every waived finding"
    )

    # `on` may parse as the string 'on' or boolean True depending on the
    # YAML loader version.
    on_section = workflow.get("on", workflow.get(True, {}))
    pr_trigger = on_section.get("pull_request")
    if isinstance(pr_trigger, dict):
        types = pr_trigger.get("types") or []
    else:
        types = []  # bare `pull_request:` -> GitHub's default types
    assert "labeled" not in types, (
        "ci.yml pull_request trigger must not add the 'labeled' activity "
        "type -- the exemption reads LIVE labels at run time so an "
        "operator applies the label and RERUNS the job (issue #1686); a "
        "labeled trigger is both unnecessary and noisy"
    )
