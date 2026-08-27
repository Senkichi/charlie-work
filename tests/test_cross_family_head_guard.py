"""Gate 3: close a fail-open in ``_record_cross_family_verdicts``'s staleness
guard.

The guard exists so a review verdict written against an OLD commit cannot be
recorded against a NEW one:

    report_head = extract_head_ref_oid(report_text)
    packet_head = candidate.get("packet_head_sha")
    if report_head is not None and packet_head is not None and report_head != packet_head:
        continue

When EITHER sha is unknown (``None``), that ``and``-chain short-circuits to
False, so the guard does NOT skip -- it falls through and records the verdict
anyway. An indeterminate comparison collapsed into the permissive branch: an
unverifiable head could authorize a merge.

The fix requires BOTH shas to be known and equal before recording; any other
combination (including either side being ``None``) skips.

The load-bearing fixture below is a cross-family report that PARSES
SUCCESSFULLY (``parse_cross_family_verdict`` returns a real
``CrossFamilyVerdict``, not ``None``) but carries no ``<!-- PR head SHA: -->``
comment, so ``extract_head_ref_oid`` returns ``None``. A report that fails to
parse would never reach the guard at all (it dies at the earlier
``if parsed is None: continue``), so it would test nothing here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from charlie_work.config import CrossFamilyConfig, OrchestratorConfig
from charlie_work.cross_family import _CAVEAT, extract_head_ref_oid, parse_cross_family_verdict
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp

# Reuse the shared FakeGitHub rather than redefining it.
from _fakes_github import FakeGitHub

# A report body with a real MINOR finding and a real Verdict line -- parses to
# an "approved" CrossFamilyVerdict -- but with NO
# "<!-- PR head SHA: ... -->" comment, so extract_head_ref_oid(...) is None.
# Verified directly against parse_cross_family_verdict / extract_head_ref_oid
# before use (see task report): parsed.decision == "approved",
# extract_head_ref_oid(...) is None.
_NO_HEAD_SHA_BODY = "**MINOR**\nsmall issue\n\nVerdict: No BLOCKERs or MAJORs — fix is correct"
_NO_HEAD_SHA_REPORT = (
    f"# Cross-family adversarial review — `glm-5.2`\n\n{_CAVEAT}\n\n---\n\n{_NO_HEAD_SHA_BODY}\n"
)


def _report_with_head_sha(head_sha: str) -> str:
    """Same body, but WITH a head-SHA comment -- the ordinary stale-report shape."""
    return (
        f"# Cross-family adversarial review — `glm-5.2`\n\n"
        f"<!-- PR head SHA: {head_sha} -->\n\n"
        f"{_CAVEAT}\n\n---\n\n{_NO_HEAD_SHA_BODY}\n"
    )


def _cross_family_auto_verdict_app(
    tmp_path: Path, *, prs: list[dict[str, Any]] | None = None
) -> OrchestratorApp:
    config = OrchestratorConfig(cross_family=CrossFamilyConfig(auto_verdict=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "state.json").write_text(
        json.dumps({"version": 1, "issues": {}, "prs": {}, "events": []}),
        encoding="utf-8",
    )
    fake_gh = FakeGitHub()
    if prs is not None:
        fake_gh.prs = prs
    return OrchestratorApp(tmp_path, paths, config, fake_gh)


def _pr(number: int, head_sha: str) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"Fix #{number}: no-head-sha cross-family report",
        "url": f"https://example.test/pull/{number}",
        "headRefName": f"agent/issue-{number}-fix",
        "baseRefName": "main",
        "headRefOid": head_sha,
        "mergeStateStatus": "CLEAN",
        "body": f"Closes #{number}",
        "labels": [],
        "isCrossRepository": False,
        "state": "OPEN",
    }


def _write_review_packet(
    tmp_path: Path, pr_number: int, packet_head_sha: str, decision: dict[str, Any]
) -> Path:
    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr.json").write_text(
        json.dumps({"number": pr_number, "headRefOid": packet_head_sha}),
        encoding="utf-8",
    )
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}", encoding="utf-8"
    )
    (pr_dir / "review-decision.json").write_text(json.dumps(decision), encoding="utf-8")
    return pr_dir


def test_fixture_parses_but_has_no_head_sha() -> None:
    """Sanity check on the fixture itself, independent of the orchestrator:
    it must be a real, successfully-parsed verdict with no extractable head
    SHA -- the exact shape the fail-open exploited."""
    parsed = parse_cross_family_verdict(_NO_HEAD_SHA_REPORT)
    assert parsed is not None
    assert parsed.decision == "approved"
    assert extract_head_ref_oid(_NO_HEAD_SHA_REPORT) is None


def test_record_cross_family_verdicts_skips_report_with_no_head_sha(
    tmp_path: Path,
) -> None:
    """A report that parses cleanly but carries no head SHA must NOT be
    recorded -- the guard cannot positively confirm it matches the live
    packet, so it must not authorize a merge on an unverifiable head."""
    pr_number = 900
    prs = [_pr(pr_number, "sha-900")]
    app = _cross_family_auto_verdict_app(tmp_path, prs=prs)
    pr_dir = _write_review_packet(
        tmp_path, pr_number, "sha-900", {"decision": "pending", "reviewed_head_sha": None}
    )
    (pr_dir / "cross-family-review.md").write_text(_NO_HEAD_SHA_REPORT, encoding="utf-8")

    recorded: list[Any] = []
    original_record_review = app.record_review

    def _tracking_record_review(*args: Any, **kwargs: Any) -> Any:
        recorded.append((args, kwargs))
        return original_record_review(*args, **kwargs)

    app.record_review = _tracking_record_review  # type: ignore[method-assign]

    results = app._record_cross_family_verdicts()

    assert recorded == []
    assert results == []
    # No decision file mutation either -- record_review was never reached.
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "pending"
    # No cross-family bookkeeping was touched -- this is a distinct skip path
    # from the malformed-verdict / max_parse_failures cycle.
    state = load_state(app.paths.state_file)
    assert "cross_family_parse_failure_count" not in state["prs"].get(str(pr_number), {})


def test_mismatched_head_still_skips_silently(tmp_path: Path) -> None:
    """The pre-existing skip -- both shas known and different -- must keep
    skipping, and must NOT take the new indeterminate path. A mismatch is
    self-healing (the next review regenerates the report against the live
    head), so emitting the indeterminate event here would fire on every
    routine stale report. This pins the two branches apart: the split into
    ``report_head is None or packet_head is None`` and ``report_head !=
    packet_head`` is otherwise only half covered."""
    pr_number = 901
    prs = [_pr(pr_number, "sha-901")]
    app = _cross_family_auto_verdict_app(tmp_path, prs=prs)
    pr_dir = _write_review_packet(
        tmp_path, pr_number, "sha-901", {"decision": "pending", "reviewed_head_sha": None}
    )
    # Report written against an older commit than the live/packet head.
    (pr_dir / "cross-family-review.md").write_text(
        _report_with_head_sha("sha-901-old"), encoding="utf-8"
    )

    recorded: list[Any] = []
    original_record_review = app.record_review

    def _tracking_record_review(*args: Any, **kwargs: Any) -> Any:
        recorded.append((args, kwargs))
        return original_record_review(*args, **kwargs)

    app.record_review = _tracking_record_review  # type: ignore[method-assign]

    results = app._record_cross_family_verdicts()

    assert recorded == []
    assert results == []
    decision = json.loads((pr_dir / "review-decision.json").read_text(encoding="utf-8"))
    assert decision["decision"] == "pending"
    state = load_state(app.paths.state_file)
    assert [
        event
        for event in state["events"]
        if event["kind"] == "cross_family_verdict_head_indeterminate"
    ] == []
    assert "cross_family_head_indeterminate" not in state["prs"].get(str(pr_number), {})


def test_indeterminate_head_skip_emits_one_event_not_one_per_pass(tmp_path: Path) -> None:
    """The skip must not be silent -- nothing regenerates a report with no
    head SHA, so the PR would otherwise stall in "reviewing" forever with no
    trace. It must also not re-emit every pass: this runs once per loop pass
    with identical inputs, and an unguarded event would flush the capped
    ``events`` ring in state.json."""
    pr_number = 900
    prs = [_pr(pr_number, "sha-900")]
    app = _cross_family_auto_verdict_app(tmp_path, prs=prs)
    pr_dir = _write_review_packet(
        tmp_path, pr_number, "sha-900", {"decision": "pending", "reviewed_head_sha": None}
    )
    (pr_dir / "cross-family-review.md").write_text(_NO_HEAD_SHA_REPORT, encoding="utf-8")

    app._record_cross_family_verdicts()

    state = load_state(app.paths.state_file)
    events = [
        event
        for event in state["events"]
        if event["kind"] == "cross_family_verdict_head_indeterminate"
    ]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == pr_number
    assert payload["report_head_sha"] is None
    assert payload["packet_head_sha"] == "sha-900"
    assert state["prs"][str(pr_number)]["cross_family_head_indeterminate"] is True

    # Second pass, same inputs: marker suppresses a duplicate event.
    app._record_cross_family_verdicts()

    state = load_state(app.paths.state_file)
    assert (
        len(
            [
                event
                for event in state["events"]
                if event["kind"] == "cross_family_verdict_head_indeterminate"
            ]
        )
        == 1
    )
