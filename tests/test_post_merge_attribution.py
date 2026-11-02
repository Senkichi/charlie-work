"""Tests for the #1717 post-merge outcome attribution scans and their
delegate lane. The scans (``post_merge_attribution_scan``) are plain
functions driven by a minimal app double that carries only the attributes
they reach (gh probes, config's required checks, state paths, the host
clock seam, a real WriteGate and the real ``_record_event`` forwarding
shape) -- the same surface the other per-pass lanes touch. The delegate
(``orchestration.post_merge_attribution._maybe_post_merge_attribution``)
is exercised for its gates (dry-run, no-PR backend) and its per-scan
exception containment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from charlie_work import post_merge_attribution_scan as scans
from charlie_work.orchestration import post_merge_attribution as lane
from charlie_work.state import load_state, save_state
from charlie_work.write_gate import WriteGate

FROZEN = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
FROZEN_ISO = FROZEN.isoformat().replace("+00:00", "Z")

CHECK_FAILED = {"name": "ci-build", "status": "completed", "conclusion": "failure"}
CHECK_OK = {"name": "ci-build", "status": "completed", "conclusion": "success"}
CHECK_OK_LINT = {"name": "lint", "status": "completed", "conclusion": "success"}
RUN_PENDING = {"name": "ci-build", "status": "in_progress", "conclusion": None}


class FakeProbeGitHub:
    """gh double: canned probe payloads and PR lists, with a call log."""

    def __init__(
        self,
        *,
        checks: list[dict[str, Any]] | None = None,
        runs: list[dict[str, Any]] | None = None,
        open_prs: list[dict[str, Any]] | None = None,
        merged_rows: list[dict[str, Any]] | None = None,
        fail_check_runs: bool = False,
    ) -> None:
        self.publishes_pull_requests = True
        self.checks = checks or []
        self.runs = runs or []
        self.open_prs = open_prs or []
        self.merged_rows = merged_rows or []
        self.fail_check_runs = fail_check_runs
        self.call_log: list[tuple[str, str]] = []

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        self.call_log.append(("commit_check_runs", sha))
        if self.fail_check_runs:
            return None
        return list(self.checks)

    def workflow_runs_for_head(self, sha: str) -> list[dict[str, Any]] | None:
        self.call_log.append(("workflow_runs_for_head", sha))
        return list(self.runs)

    def pr_list(self) -> list[dict[str, Any]]:
        return list(self.open_prs)

    def merged_pr_list(self) -> list[dict[str, Any]]:
        return list(self.merged_rows)


def merged_entry(number: int, issue: int = 77, *, sha: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "number": number,
        "issue_number": issue,
        "status": "merged",
        "merged_at": "2026-10-19T00:00:00Z",
    }
    if sha is not None:
        entry["merged_sha"] = sha
    return entry


def make_app(
    tmp_path: Path,
    gh: Any,
    *,
    required: tuple[str, ...] = (),
    dry_run: bool = False,
    merged: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """A minimal app double carrying exactly what the scans reach."""
    state_file = tmp_path / "state.json"
    state = load_state(state_file)
    if merged is not None:
        state["prs"].update(merged)
    save_state(state_file, state)
    app = SimpleNamespace(
        gh=gh,
        dry_run=dry_run,
        config=SimpleNamespace(auto_merge=SimpleNamespace(required_checks=tuple(required))),
        paths=SimpleNamespace(state_file=state_file),
        host=SimpleNamespace(clock=SimpleNamespace(now=lambda: FROZEN)),
        write_gate=WriteGate(dry_run=dry_run, state_path=state_file, repo="testrepo"),
    )
    app._record_event = _record_event_of(app)
    return app


def _record_event_of(app: SimpleNamespace):
    def _record_event(
        state: dict[str, Any], kind: str, payload: dict[str, Any], *, level: str | None = None
    ) -> dict[str, Any]:
        return app.write_gate.record_event(state, kind, payload, level=level)

    return _record_event


def run_probe(app: SimpleNamespace) -> None:
    scans.probe_merged_main_ci(app, state_file=app.paths.state_file, now=FROZEN)


def run_followup(app: SimpleNamespace) -> None:
    scans.scan_followup_pr_attribution(app, state_file=app.paths.state_file)


def events_of(app: SimpleNamespace) -> list[dict[str, Any]]:
    return load_state(app.paths.state_file).get("events", [])


# ---------------------------------------------------------------------------
# Part 1: post-merge main-CI probe
# ---------------------------------------------------------------------------


def test_required_check_failure_is_recorded_against_the_merged_pr(tmp_path: Path) -> None:
    """A required check failing on the merged PR's merge-commit SHA records
    ``post_merge_main_ci_failure`` against the PR and its issue, and stamps
    the entry's terminal probe outcome so it is never probed again."""
    gh = FakeProbeGitHub(checks=[CHECK_FAILED], runs=[CHECK_OK])
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={"41": merged_entry(41, sha="abc123")},
    )

    run_probe(app)

    events = [e for e in events_of(app) if e["kind"] == "post_merge_main_ci_failure"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 41
    assert payload["issue_number"] == 77
    assert payload["merged_sha"] == "abc123"
    assert payload["failed_checks"] == ["ci-build"]
    stamp = load_state(app.paths.state_file)["prs"]["41"]["post_merge_ci_probe"]
    assert stamp["outcome"] == "failure"
    assert stamp["failed_checks"] == ["ci-build"]
    assert stamp["probed_at"] == FROZEN_ISO


def test_clean_merge_records_no_event_and_stamps_clean(tmp_path: Path) -> None:
    """Every run past pending with zero failures stamps ``clean`` and emits
    nothing -- silence stays the no-op contract."""
    gh = FakeProbeGitHub(checks=[CHECK_OK, CHECK_OK_LINT], runs=[CHECK_OK])
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build", "lint"),
        merged={"41": merged_entry(41, sha="abc123")},
    )

    run_probe(app)

    assert [e["kind"] for e in events_of(app) if "post_merge" in e["kind"]] == []
    stamp = load_state(app.paths.state_file)["prs"]["41"]["post_merge_ci_probe"]
    assert stamp["outcome"] == "clean"


def test_pending_runs_record_nothing_and_burn_no_attempt(tmp_path: Path) -> None:
    """A SHA whose runs are still running carries no outcome yet: nothing is
    recorded, the attempt counter does not advance, and the entry is simply
    re-probed on the next pass."""
    gh = FakeProbeGitHub(checks=[CHECK_OK], runs=[RUN_PENDING])
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={"41": merged_entry(41, sha="abc123")},
    )

    run_probe(app)

    entry = load_state(app.paths.state_file)["prs"]["41"]
    assert "post_merge_ci_probe" not in entry
    assert [e["kind"] for e in events_of(app) if "post_merge" in e["kind"]] == []


def test_non_required_failure_is_clean_but_any_failure_falls_back(tmp_path: Path) -> None:
    """With a required list configured, only a failure in it is a defect; with
    an empty list any failure counts (the merge gate cannot have been gating
    on checks at all, so the report takes the whole failure)."""
    gh = FakeProbeGitHub(checks=[dict(CHECK_FAILED, name="flaky")], runs=[CHECK_OK])
    app = make_app(tmp_path, gh, required=("ci-build",), merged={"41": merged_entry(41, sha="s")})
    run_probe(app)
    assert load_state(app.paths.state_file)["prs"]["41"]["post_merge_ci_probe"]["outcome"] == (
        "clean"
    )
    assert [e["kind"] for e in events_of(app) if "post_merge" in e["kind"]] == []

    gh2 = FakeProbeGitHub(checks=[dict(CHECK_FAILED, name="lint-only")], runs=[CHECK_OK])
    app2 = make_app(tmp_path, gh2, required=(), merged={"41": merged_entry(41, sha="s")})
    run_probe(app2)
    events = [e for e in events_of(app2) if e["kind"] == "post_merge_main_ci_failure"]
    assert len(events) == 1
    assert events[0]["payload"]["failed_checks"] == ["lint-only"]


def test_merge_commit_sha_falls_back_to_the_merged_rest_list(tmp_path: Path) -> None:
    """A remote-lane merge records no ``merged_sha`` on the entry; the probe
    resolves the SHA from ``merged_pr_list``'s REST ``mergeCommitOid``."""
    gh = FakeProbeGitHub(
        checks=[CHECK_FAILED],
        runs=[CHECK_OK],
        merged_rows=[{"number": 41, "mergeCommitOid": "deadbeef"}],
    )
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={"41": merged_entry(41)},
    )

    run_probe(app)

    assert ("commit_check_runs", "deadbeef") in gh.call_log
    events = [e for e in events_of(app) if e["kind"] == "post_merge_main_ci_failure"]
    assert events[0]["payload"]["merged_sha"] == "deadbeef"


def test_unreadable_probes_spend_an_attempt_until_the_cap(tmp_path: Path) -> None:
    """Both probe payloads unreadable (None) is inconclusive: the attempt
    counter advances, and after the cap the entry stops being probed."""
    gh = FakeProbeGitHub(fail_check_runs=True, runs=[])
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={"41": merged_entry(41, sha="abc123")},
    )

    for _ in range(scans.PROBE_ATTEMPT_CAP + 2):
        run_probe(app)

    assert len(gh.call_log) == scans.PROBE_ATTEMPT_CAP * 2
    stamp = load_state(app.paths.state_file)["prs"]["41"]["post_merge_ci_probe"]
    assert stamp["outcome"] is None
    assert stamp["attempts"] == scans.PROBE_ATTEMPT_CAP


def test_per_pass_probe_cap_bounds_the_remote_calls(tmp_path: Path) -> None:
    """Only PER_PASS_PROBE_CAP merged entries are probed per pass; the rest
    queue up for the next pass rather than turning one pass into a storm."""
    gh = FakeProbeGitHub(checks=[CHECK_OK], runs=[CHECK_OK])
    merged = {str(number): merged_entry(number, sha=f"sha-{number}") for number in range(41, 48)}
    app = make_app(tmp_path, gh, required=("ci-build",), merged=merged)

    run_probe(app)

    assert len(gh.call_log) == scans.PER_PASS_PROBE_CAP * 2
    state = load_state(app.paths.state_file)
    stamped = [k for k, v in state["prs"].items() if "post_merge_ci_probe" in v]
    assert len(stamped) == scans.PER_PASS_PROBE_CAP


def test_terminal_entries_are_never_probed_again(tmp_path: Path) -> None:
    """A stamped terminal outcome (failure) takes the entry out of the
    rotation: a later pass makes no remote calls and emits nothing."""
    gh = FakeProbeGitHub(checks=[CHECK_FAILED], runs=[CHECK_OK])
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={
            "41": {
                **merged_entry(41, sha="abc123"),
                "post_merge_ci_probe": {
                    "outcome": "failure",
                    "attempts": 1,
                    "failed_checks": ["ci-build"],
                },
            }
        },
    )

    run_probe(app)

    assert gh.call_log == []
    events = [e for e in events_of(app) if e["kind"] == "post_merge_main_ci_failure"]
    assert events == []


# ---------------------------------------------------------------------------
# Part 2: follow-up attribution
# ---------------------------------------------------------------------------


def test_followup_revert_mention_is_attributed_once(tmp_path: Path) -> None:
    """An open PR whose title names a merged PR ('Reverts #41') records
    ``post_merge_followup_attribution`` with the matched text, stamps the
    merged entry's ledger, and never repeats the pair on later passes."""
    gh = FakeProbeGitHub(
        open_prs=[
            {"number": 90, "title": "Reverts #41", "body": "undoes the merge from earlier"},
        ],
    )
    app = make_app(tmp_path, gh, merged={"41": merged_entry(41)})

    run_followup(app)
    run_followup(app)

    events = [e for e in events_of(app) if e["kind"] == "post_merge_followup_attribution"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 41
    assert payload["issue_number"] == 77
    assert payload["followup_pr_number"] == 90
    assert payload["method"] == "keyword_mention"
    assert payload["trigger"] == "reverts"
    assert "#41" in payload["detail"]
    ledger = load_state(app.paths.state_file)["prs"]["41"]["post_merge_followups"]
    assert ledger == [90]


def test_fix_and_followup_references_in_the_body_are_attributed(tmp_path: Path) -> None:
    """The trigger class covers fix/follow-up mentions in the body, and a
    reference to a number with no merged entry is ignored."""
    gh = FakeProbeGitHub(
        open_prs=[
            {
                "number": 91,
                "title": "Add a test",
                "body": "follow-up for #41: missing edge case\nSee also #999 for background",
            },
        ],
    )
    app = make_app(tmp_path, gh, merged={"41": merged_entry(41)})

    run_followup(app)

    events = [e for e in events_of(app) if e["kind"] == "post_merge_followup_attribution"]
    assert len(events) == 1
    assert events[0]["payload"]["pr_number"] == 41
    ledge = load_state(app.paths.state_file)["prs"]["41"]["post_merge_followups"]
    assert ledge == [91]


def test_reference_to_an_unmerged_live_issue_is_not_attributed(tmp_path: Path) -> None:
    """'fixes #42' pointing at a PR still in flight (or any number without a
    merged tracked entry) carries no attribution."""
    gh = FakeProbeGitHub(
        open_prs=[{"number": 92, "title": "fixes #42", "body": None}],
    )
    app = make_app(
        tmp_path,
        gh,
        merged={
            "41": merged_entry(41),
            "42": {"number": 42, "status": "in_progress"},
        },
    )

    run_followup(app)

    events = [e for e in events_of(app) if e["kind"] == "post_merge_followup_attribution"]
    assert events == []
    assert "post_merge_followups" not in load_state(app.paths.state_file)["prs"]["41"]


# ---------------------------------------------------------------------------
# The delegate lane's gates and containment
# ---------------------------------------------------------------------------


def test_lane_skips_dry_run_and_non_pr_backends(tmp_path: Path) -> None:
    """dry-run (the reports the writes feed are not consumed there) and a
    no-PR backend must leave the gh surface untouched."""
    gh = FakeProbeGitHub(
        checks=[CHECK_FAILED],
        runs=[CHECK_OK],
        open_prs=[{"number": 90, "title": "Reverts #41", "body": None}],
    )
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        dry_run=True,
        merged={"41": merged_entry(41, sha="abc123")},
    )
    lane._maybe_post_merge_attribution(app, now=FROZEN)
    assert gh.call_log == []
    assert [e["kind"] for e in events_of(app) if "post_merge" in e["kind"]] == []

    gh.publishes_pull_requests = False
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={"41": merged_entry(41, sha="abc123")},
    )
    lane._maybe_post_merge_attribution(app, now=FROZEN)
    assert gh.call_log == []


def test_lane_contains_one_scan_and_still_runs_the_other(tmp_path: Path) -> None:
    """Each scan is independently contained: a raising CI probe must neither
    kill the call nor block the follow-up scan's own event."""
    gh = FakeProbeGitHub(
        checks=[CHECK_FAILED],
        runs=[CHECK_OK],
        open_prs=[{"number": 90, "title": "Reverts #41", "body": None}],
    )
    app = make_app(
        tmp_path,
        gh,
        required=("ci-build",),
        merged={"41": merged_entry(41, sha="abc123")},
    )

    def _explode(sha: str) -> None:
        raise RuntimeError("probe transport outage")

    gh.commit_check_runs = _explode

    lane._maybe_post_merge_attribution(app, now=FROZEN)

    events: list[dict[str, Any]] = [
        e for e in events_of(app) if e["kind"] == "post_merge_followup_attribution"
    ]
    assert len(events) == 1
    assert events[0]["payload"]["followup_pr_number"] == 90
