"""Tests for issue #1686: collect-gate exemption evidence in the review packet.

The gate command emits a ``COLLECT-GATE-EXEMPTION v1 {...}`` line into its
Actions job log whenever an exemption is evaluated (``--pr`` given). The
review packet must read that marker back out of the gate job's log for the
REVIEWED head and render: whether the configured label was applied, exactly
which findings were waived, and stale/unverifiable-evidence caveats -- never
presenting missing or head-mismatched evidence as a waiver.

Pattern mirrors ``tests/test_attachment_budget_packet.py``: a FakeGitHub
variant, ``app.review(N)``, read the rendered ``review-prompt.md``.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHubWithChecks
from charlie_work.collect_only_gate import (
    COLLECT_ONLY_GATE_CHECK_NAME,
    CollectGateExemption,
    CollectOnlyFinding,
    exemption_log_marker,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

_HEAD = "sha-abc123"  # FakeGitHub's default PR headRefOid
_JOB_ID = 90210
_GATE_CHECK = {
    "name": COLLECT_ONLY_GATE_CHECK_NAME,
    "state": "SUCCESS",
    "link": f"https://github.com/test-owner/test-repo/actions/runs/777/job/{_JOB_ID}",
}

_WAIVED = [
    CollectOnlyFinding(
        kind="removed",
        leaf_name="test_b",
        source_module="tests/test_foo.py",
        detail="present at base, absent at head",
        base_count=1,
        head_count=0,
    ),
    CollectOnlyFinding(
        kind="missing_sibling",
        leaf_name="test_b",
        source_module="tests/test_foo.py",
        detail="did not reappear under tests/",
    ),
]


def _marker_line(
    *,
    active: bool,
    waived=(),
    head_sha: str | None = _HEAD,
    label: str = "collect-gate-exempt",
) -> str:
    return exemption_log_marker(
        CollectGateExemption(label=label, active=active, detail="test detail"),
        waived,
        head_sha=head_sha,
    )


def _build_packet(
    tmp_path: Path,
    *,
    pr_labels: tuple[str, ...] = (),
    checks: list[dict] | None = None,
    job_logs: dict[int, str] | None = None,
) -> str:
    fake_gh = FakeGitHubWithChecks(checks=list(checks) if checks is not None else [])
    fake_gh.prs[0]["labels"] = [{"name": name} for name in pr_labels]
    fake_gh.job_logs.update(job_logs or {})
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)
    assert result.ok is True, result.message
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    return packet.read_text(encoding="utf-8")


def test_active_exemption_lists_every_waived_finding(tmp_path: Path) -> None:
    """Label present + marker active: label named, every waived finding shown."""
    log = "some earlier log line\n" + _marker_line(active=True, waived=_WAIVED) + "\n"
    packet = _build_packet(
        tmp_path,
        pr_labels=("collect-gate-exempt",),
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: log},
    )
    assert "collect-gate-exempt" in packet
    assert "waiv" in packet.lower()
    assert "removed" in packet
    assert "missing_sibling" in packet
    assert "test_b" in packet


def test_active_exemption_nothing_to_waive_marks_label_stale(tmp_path: Path) -> None:
    """Label present + marker active with zero waived findings: the packet
    says there was nothing to waive so a stale label is visible."""
    log = _marker_line(active=True, waived=()) + "\n"
    packet = _build_packet(
        tmp_path,
        pr_labels=("collect-gate-exempt",),
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: log},
    )
    assert "collect-gate-exempt" in packet
    lowered = packet.lower()
    assert "nothing" in lowered or "no enforced findings" in lowered or "stale" in lowered


def test_label_present_but_no_marker_fails_safe(tmp_path: Path) -> None:
    """Label present but the gate log carries no marker: the packet must NOT
    present the label as a waiver -- it warns that no evidence was found."""
    packet = _build_packet(
        tmp_path,
        pr_labels=("collect-gate-exempt",),
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: "gate ran but no marker in this log\n"},
    )
    assert "collect-gate-exempt" in packet
    lowered = packet.lower()
    assert "waiv" in lowered  # mentions the waiver question explicitly
    assert "no exemption evidence" in lowered or "not" in lowered


def test_label_present_but_log_unavailable_fails_safe(tmp_path: Path) -> None:
    """Label present but the job log cannot be fetched: same fail-safe
    wording -- unavailable evidence is never rendered as a waiver."""
    packet = _build_packet(
        tmp_path,
        pr_labels=("collect-gate-exempt",),
        checks=[_GATE_CHECK],
        job_logs={},  # fetch fails
    )
    assert "collect-gate-exempt" in packet
    assert "no exemption evidence" in packet.lower() or "unavailable" in packet.lower()


def test_marker_for_other_head_is_rejected_as_stale(tmp_path: Path) -> None:
    """A marker stamped for a DIFFERENT head sha must not be presented as
    evidence for the reviewed head -- the label survives ``synchronize``,
    so only head-pinned evidence counts."""
    log = _marker_line(active=True, waived=_WAIVED, head_sha="sha-OLD-HEAD") + "\n"
    packet = _build_packet(
        tmp_path,
        pr_labels=("collect-gate-exempt",),
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: log},
    )
    assert "collect-gate-exempt" in packet
    lowered = packet.lower()
    assert "stale" in lowered or "different head" in lowered or "could not be verified" in lowered
    # The waived findings must not be presented as waived FOR THIS head.
    assert "waived 2" not in lowered


def test_waiver_evidence_survives_label_removal(tmp_path: Path) -> None:
    """Label no longer applied but the gate run on this head DID waive
    findings: the packet still reports the waived findings (a green check
    that passed via waiver stays auditable)."""
    log = _marker_line(active=True, waived=_WAIVED) + "\n"
    packet = _build_packet(
        tmp_path,
        pr_labels=(),  # label removed after the run
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: log},
    )
    assert "test_b" in packet
    assert "waiv" in packet.lower()


def test_label_applied_after_gate_run_is_not_a_waiver(tmp_path: Path) -> None:
    """Label present NOW but the gate run resolved it absent (applied after
    the run): the packet must say nothing was waived -- the operator needs
    to rerun the gate job."""
    log = _marker_line(active=False, waived=()) + "\n"
    packet = _build_packet(
        tmp_path,
        pr_labels=("collect-gate-exempt",),
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: log},
    )
    assert "collect-gate-exempt" in packet
    lowered = packet.lower()
    assert "rerun" in lowered or "not" in lowered


def test_no_label_no_marker_no_section(tmp_path: Path) -> None:
    """The common case -- no label, no exemption evaluation -- renders no
    exemption content at all."""
    packet = _build_packet(
        tmp_path,
        pr_labels=(),
        checks=[_GATE_CHECK],
        job_logs={_JOB_ID: "ordinary gate log\n"},
    )
    assert "collect-gate-exempt" not in packet
    assert "exemption" not in packet.lower()


def test_configured_label_name_is_honored(tmp_path: Path) -> None:
    """A non-default configured label name drives the section, not the
    default -- the label string is never re-declared in code."""
    from charlie_work.config import LabelConfig

    fake_gh = FakeGitHubWithChecks(checks=[_GATE_CHECK])
    fake_gh.prs[0]["labels"] = [{"name": "custom-exempt"}]
    fake_gh.job_logs[_JOB_ID] = _marker_line(active=True, waived=_WAIVED, label="custom-exempt")
    config = OrchestratorConfig(labels=LabelConfig(collect_gate_exempt="custom-exempt"))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)
    assert result.ok is True, result.message
    packet = (
        tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    ).read_text(encoding="utf-8")
    assert "custom-exempt" in packet
    assert "test_b" in packet
