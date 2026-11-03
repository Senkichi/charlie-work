"""Issue #2666: re-run a required aggregate job that a wedged runner timed out.

The payload shapes here are vendored from the live public API (captured 2026-11-02, see
``tests/fixtures/gh_commit_check_runs_job_timeout.json`` and the three
``gh_check_run_annotations_*`` files):

* a job-level ``timeout-minutes`` kill is ``cancelled`` and carries the annotation
  "The job has exceeded the maximum execution time of 15m0s" -- and its *run* is
  ``cancelled`` too, which is why the run's conclusion cannot be the discriminator;
* a job cancelled by a newer run's concurrency group carries "Canceling since a higher
  priority waiting request for ..." and no timeout annotation (the negative control).
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _fakes_github import FakeGitHub

from charlie_work import aggregate_job_heal as heal
from charlie_work.aggregate_job_heal import (
    MAX_RERUN_CALLS,
    SOURCE_MERGEQUEUE,
    SOURCE_QUEUE_BOT_DRAFT,
    QueuePr,
    cancelled_required_checks,
    heal_best_effort,
    heal_wedged_aggregates,
    is_timeout_cancel,
    queue_prs,
    rerun_allowed,
)
from charlie_work.command_result import CommandResult
from charlie_work.config import build_config_from_data
from charlie_work.github import GitHubError, GitHubRunResult
from charlie_work.instrumentation import query_events
from charlie_work.pass_deadline import PassDeadline, run_deadline_guarded_maintenance
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state, save_state
from charlie_work.workflow import OrchestratorApp

FIXTURES = Path(__file__).parent / "fixtures"
BOT = "aviator-app[bot]"
RUN = 41505196225  # the run id of the vendored timeout payload
HEAD = "cc498f61a117c77955ffa97624ec4ea3ce1a0cb9"


def _json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


REAL_CHECK_RUNS: list[dict[str, Any]] = _json("gh_commit_check_runs_job_timeout.json")[
    "check_runs"
]
TIMEOUT_ANNOTATIONS = _json("gh_check_run_annotations_job_timeout.json")
SUPERSEDED_ANNOTATIONS = _json("gh_check_run_annotations_superseded_cancel.json")


def _real(name: str) -> dict[str, Any]:
    return copy.deepcopy(next(run for run in REAL_CHECK_RUNS if run["name"] == name))


def _check(
    name: str,
    *,
    check_run_id: int,
    conclusion: str | None = "cancelled",
    run_id: int = RUN,
    status: str = "completed",
    details_url: str | None = None,
) -> dict[str, Any]:
    """A check run in the real payload's shape (cloned from the live timeout-cancelled job)."""
    run = _real("Tests shard 4/6")
    run.update(
        name=name,
        id=check_run_id,
        conclusion=conclusion,
        status=status,
        details_url=details_url
        or f"https://github.com/Senkichi/charlie-work/actions/runs/{run_id}/job/{check_run_id}",
    )
    return run


def _pull(
    number: int,
    *,
    sha: str = HEAD,
    labels: tuple[str, ...] = (),
    login: str = "someone",
    state: str = "open",
    draft: bool = False,
) -> dict[str, Any]:
    """A REST ``pulls`` list entry (the shape ``_normalize_reconcile_pr`` consumes)."""
    return {
        "number": number,
        "state": state,
        "draft": draft,
        "title": f"PR {number}",
        "html_url": f"https://github.com/Senkichi/charlie-work/pull/{number}",
        "labels": [{"name": label} for label in labels],
        "user": {"login": login},
        "head": {"ref": f"branch-{number}", "sha": sha, "repo": {"full_name": "o/r"}},
        "base": {"ref": "main", "repo": {"full_name": "o/r"}},
    }


class _Gh(FakeGitHub):
    """FakeGitHub with the REST surface the lane reads and the one mutation it makes."""

    def __init__(self) -> None:
        super().__init__()
        self.open_pulls: list[dict[str, Any]] = []
        self.check_runs_by_sha: dict[str, list[dict[str, Any]] | None] = {}
        self.annotations_by_check_run: dict[int, list[dict[str, Any]]] = {}
        self.rerun_error: str | None = None
        self.pulls_error: Exception | None = None
        self.pull_list_calls = 0
        self.check_run_reads: list[str] = []
        self.annotation_reads: list[int] = []
        self.rerun_calls: list[list[str]] = []

    def run(
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
        long_call: bool = False,  # accepted for GitHubLike parity; the base fake has no such knob
    ):
        joined = " ".join(args)
        if args[:1] == ["api"] and "pulls?state=open" in joined:
            self.pull_list_calls += 1
            if self.pulls_error is not None:
                raise self.pulls_error
            return list(self.open_pulls)
        if args[:2] == ["run", "rerun"]:
            self.rerun_calls.append(list(args))
            if self.rerun_error is not None:
                return GitHubRunResult(
                    ok=False, returncode=1, stdout="", stderr="", error=self.rerun_error
                )
            return GitHubRunResult(ok=True, returncode=0, stdout="", stderr="")
        return super().run(args, json_output=json_output, allow_failure=allow_failure)

    def commit_check_runs(self, sha: str):
        self.check_run_reads.append(sha)
        return self.check_runs_by_sha.get(sha, [])

    def check_run_annotations(self, check_run_id: int):
        self.annotation_reads.append(check_run_id)
        return self.annotations_by_check_run.get(check_run_id, [])


def _config(**auto_merge: Any):
    merged = {
        "required_checks": ["Tests", "Lint"],
        "mergequeue_label": "mergequeue",
        "queue_bot_login": BOT,
        **auto_merge,
    }
    return build_config_from_data({"auto_merge": merged})


def _app(tmp_path: Path, gh: _Gh | None = None, **auto_merge: Any) -> tuple[OrchestratorApp, _Gh]:
    gh = gh or _Gh()
    config = _config(**auto_merge)
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    save_state(paths.state_file, load_state(paths.state_file))
    return OrchestratorApp(tmp_path, paths, config, gh), gh


def _wedge(gh: _Gh, *, pr: int = 545, labels=("mergequeue",), login="someone") -> None:
    """Seed the issue's incident: a queued PR whose ``Tests`` aggregate job timed out."""
    gh.open_pulls = [_pull(pr, labels=labels, login=login)]
    gh.check_runs_by_sha[HEAD] = [
        _check("Tests", check_run_id=9001),
        _check("Lint", check_run_id=9002, conclusion="success"),
    ]
    gh.annotations_by_check_run[9001] = TIMEOUT_ANNOTATIONS


NOW = datetime.now(UTC)


# ------------------------------------------------------------- real payload shape


def test_vendored_timeout_payload_is_a_cancelled_job_inside_a_cancelled_run() -> None:
    # Pins the evidence behind the discriminator: the real timeout kill is `cancelled`
    # and its annotation carries the timeout message.
    job = _real("Tests shard 4/6")
    assert job["status"] == "completed"
    assert job["conclusion"] == "cancelled"
    assert is_timeout_cancel(TIMEOUT_ANNOTATIONS)


def test_cancelled_required_checks_reads_the_real_payload() -> None:
    found = cancelled_required_checks(REAL_CHECK_RUNS, ["Tests shard 4/6", "Lint", "Tests"])
    assert [(c.check_name, c.check_run_id, c.run_id) for c in found] == [
        ("Tests shard 4/6", 125482666168, RUN)
    ]  # Lint succeeded and `Tests` failed (its needed shard was the cancelled one)


def test_timeout_annotation_is_the_discriminator_not_the_generic_cancel_text() -> None:
    assert is_timeout_cancel(TIMEOUT_ANNOTATIONS)
    # Superseded-by-concurrency carries "The operation was canceled." too, but no timeout.
    assert any("canceled" in a["message"] for a in SUPERSEDED_ANNOTATIONS)
    assert not is_timeout_cancel(SUPERSEDED_ANNOTATIONS)
    assert not is_timeout_cancel([])
    assert not is_timeout_cancel([None, "x", {"message": None}, {}])


# --------------------------------------------------------------- pure decisions


def test_only_the_latest_check_run_per_name_is_judged() -> None:
    older_cancelled_newer_ok = [
        _check("Tests", check_run_id=10),
        _check("Tests", check_run_id=11, conclusion="success"),
    ]
    assert cancelled_required_checks(older_cancelled_newer_ok, ["Tests"]) == ()
    older_ok_newer_cancelled = [
        _check("Tests", check_run_id=10, conclusion="success"),
        _check("Tests", check_run_id=11),
    ]
    assert [
        c.check_run_id for c in cancelled_required_checks(older_ok_newer_cancelled, ["Tests"])
    ] == [11]
    # A newer run that is still queued/in progress also supersedes the cancelled one.
    rerun_in_flight = [
        _check("Tests", check_run_id=10),
        _check("Tests", check_run_id=11, status="in_progress", conclusion=None),
    ]
    assert cancelled_required_checks(rerun_in_flight, ["Tests"]) == ()


@pytest.mark.parametrize(
    "run",
    [
        _check("Tests", check_run_id=1, conclusion="failure"),
        _check("Tests", check_run_id=1, conclusion="success"),
        _check("Tests", check_run_id=1, status="in_progress", conclusion=None),
        # not an Actions job link: no workflow run to re-run
        _check("Tests", check_run_id=1, details_url="https://github.com/o/r/runs/1"),
        _check("Tests", check_run_id=1, details_url=None) | {"details_url": None},
    ],
)
def test_other_check_states_are_never_reported(run: dict[str, Any]) -> None:
    assert cancelled_required_checks([run], ["Tests"]) == ()


def test_a_cancelled_check_that_is_not_required_is_ignored() -> None:
    assert cancelled_required_checks([_check("CodeQL", check_run_id=1)], ["Tests"]) == ()


def test_queue_prs_selects_labelled_prs_and_bot_drafts_only() -> None:
    config = _config()
    prs = [
        heal._normalize_reconcile_pr(p)
        for p in (
            _pull(1, labels=("mergequeue",)),
            _pull(2, login=BOT, draft=True),
            _pull(3),  # neither labelled nor bot-authored
            _pull(4, labels=("mergequeue",), state="closed"),
            _pull(5, labels=("mergequeue",), sha=""),  # no head sha
            _pull(6, labels=("mergequeue",), login=BOT),  # a bot PR that also has the label
        )
    ]
    assert queue_prs(prs, config) == (
        QueuePr(1, HEAD, SOURCE_MERGEQUEUE),
        QueuePr(2, HEAD, SOURCE_QUEUE_BOT_DRAFT),
        QueuePr(6, HEAD, SOURCE_QUEUE_BOT_DRAFT),
    )


def test_queue_prs_with_neither_label_nor_bot_configured_selects_nothing() -> None:
    config = build_config_from_data({"auto_merge": {"required_checks": ["Tests"]}})
    prs = [heal._normalize_reconcile_pr(_pull(1, labels=("mergequeue",), login=BOT))]
    assert queue_prs(prs, config) == ()


@pytest.mark.parametrize(
    ("entry", "allowed"),
    [
        (None, True),
        ({}, True),
        ({"triggered": True, "calls": 1}, False),
        ({"triggered": False, "calls": MAX_RERUN_CALLS - 1}, True),
        ({"triggered": False, "calls": MAX_RERUN_CALLS}, False),
        ({"calls": "garbage"}, True),
        ("garbage", True),
    ],
)
def test_rerun_allowed(entry: Any, allowed: bool) -> None:
    assert rerun_allowed(entry) is allowed


def test_state_sections_are_pure_and_prune_to_the_newest_runs() -> None:
    state: dict[str, Any] = {"prs": {}, "events": []}
    assert heal.is_due(state, NOW)
    armed = heal.armed(state, NOW + timedelta(minutes=3))
    assert "aggregate_job_heal" not in state  # the input is never mutated
    assert not heal.is_due(armed, NOW)
    assert heal.is_due(armed, NOW + timedelta(minutes=3))
    assert heal.is_due({"aggregate_job_heal": {"next_due_at": "not-a-date"}}, NOW)

    grown = state
    for run_id in range(1, heal._KEEP_RUNS + 6):
        grown = heal.with_attempt(grown, run_id, pr_number=1, triggered=False, at=NOW)
    kept = grown["aggregate_job_heal"]["reruns"]
    assert len(kept) == heal._KEEP_RUNS
    assert "1" not in kept and str(heal._KEEP_RUNS + 5) in kept

    again = heal.with_attempt(grown, heal._KEEP_RUNS + 5, pr_number=1, triggered=True, at=NOW)
    entry = again["aggregate_job_heal"]["reruns"][str(heal._KEEP_RUNS + 5)]
    assert entry["calls"] == 2 and entry["triggered"] is True


# ------------------------------------------------- the lane: the issue's verification


def test_wedged_aggregate_on_a_queued_pr_is_rerun_once_with_one_event(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)

    assert heal_wedged_aggregates(app, now=NOW) == 1

    assert gh.rerun_calls == [["run", "rerun", str(RUN), "--failed"]]
    events = query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered")
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["pr_number"] == 545
    assert payload["head_sha"] == HEAD
    assert payload["run_id"] == RUN
    assert payload["checks"] == ["Tests"]
    assert payload["check_run_ids"] == [9001]
    assert payload["source"] == SOURCE_MERGEQUEUE
    recorded = load_state(app.paths.state_file)["aggregate_job_heal"]["reruns"][str(RUN)]
    assert recorded["triggered"] is True and recorded["pr_number"] == 545


def test_a_cancelled_run_that_was_not_a_timeout_triggers_no_rerun(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    gh.annotations_by_check_run[9001] = SUPERSEDED_ANNOTATIONS  # superseded, not wedged

    assert heal_wedged_aggregates(app, now=NOW) == 0

    assert gh.rerun_calls == []
    assert gh.annotation_reads == [9001]  # the cancelled check was inspected, then ruled out
    assert query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered") == []
    assert "reruns" not in load_state(app.paths.state_file).get("aggregate_job_heal", {})


def test_a_wedge_that_repeats_on_the_rerun_is_not_rerun_again(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    assert heal_wedged_aggregates(app, now=NOW) == 1

    # The rerun wedges again: same run id, a new cancelled job -- a real hang must surface.
    gh.check_runs_by_sha[HEAD] = [_check("Tests", check_run_id=9003)]
    gh.annotations_by_check_run[9003] = TIMEOUT_ANNOTATIONS
    later = NOW + timedelta(minutes=app.config.auto_merge.aggregate_heal_interval_minutes + 1)
    assert heal_wedged_aggregates(app, now=later) == 0

    assert len(gh.rerun_calls) == 1
    assert len(query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered")) == 1


def test_queue_bot_draft_is_healed_without_a_state_pr_entry(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh, pr=552, labels=(), login=BOT)

    assert heal_wedged_aggregates(app, now=NOW) == 1

    (event,) = query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered")
    assert event["payload"]["source"] == SOURCE_QUEUE_BOT_DRAFT
    assert event["payload"]["pr_number"] == 552
    assert load_state(app.paths.state_file)["prs"] == {}  # not fleet-owned: no entry created


def test_two_timed_out_checks_of_one_run_cost_one_rerun(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    gh.check_runs_by_sha[HEAD] = [
        _check("Tests", check_run_id=9001),
        _check("Lint", check_run_id=9002),
    ]
    gh.annotations_by_check_run[9002] = TIMEOUT_ANNOTATIONS

    assert heal_wedged_aggregates(app, now=NOW) == 1

    assert len(gh.rerun_calls) == 1
    (event,) = query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered")
    assert event["payload"]["checks"] == ["Tests", "Lint"]


def test_a_pr_the_queue_does_not_own_costs_no_check_run_read(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    gh.open_pulls = [_pull(7)]  # neither labelled nor bot-authored

    assert heal_wedged_aggregates(app, now=NOW) == 0

    assert gh.pull_list_calls == 1
    assert gh.check_run_reads == [] and gh.annotation_reads == [] and gh.rerun_calls == []


def test_unreadable_check_runs_skip_the_pr_without_a_rerun(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    gh.check_runs_by_sha[HEAD] = None

    assert heal_wedged_aggregates(app, now=NOW) == 0
    assert gh.rerun_calls == []


# ----------------------------------------------------- cadence, kill switch, dormancy


def test_cadence_gates_the_whole_tick(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    gh.open_pulls = []

    heal_wedged_aggregates(app, now=NOW)
    heal_wedged_aggregates(app, now=NOW + timedelta(minutes=1))
    assert gh.pull_list_calls == 1  # second call was inside the 3-minute cadence
    heal_wedged_aggregates(app, now=NOW + timedelta(minutes=4))
    assert gh.pull_list_calls == 2


@pytest.mark.parametrize(
    "auto_merge",
    [
        {"aggregate_heal_interval_minutes": 0},  # the kill switch
        {"required_checks": []},
        {"mergequeue_label": None, "queue_bot_login": None},  # no merge queue on this repo
    ],
)
def test_dormant_configs_make_no_api_call_and_write_no_state(
    tmp_path: Path, auto_merge: dict[str, Any]
) -> None:
    app, gh = _app(tmp_path, **auto_merge)
    _wedge(gh)
    before = load_state(app.paths.state_file)

    assert heal_wedged_aggregates(app, now=NOW) == 0

    assert gh.pull_list_calls == 0 and gh.rerun_calls == []
    assert load_state(app.paths.state_file) == before


def test_dry_run_does_nothing(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    app.dry_run = True

    assert heal_wedged_aggregates(app, now=NOW) == 0
    assert gh.pull_list_calls == 0 and gh.rerun_calls == []


def test_the_lane_ships_on_by_default() -> None:
    assert build_config_from_data({}).auto_merge.aggregate_heal_interval_minutes == 3


# ------------------------------------------------------------------- failure paths


def test_a_refused_rerun_is_retried_a_bounded_number_of_times(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    gh.rerun_error = "HTTP 403: workflow run cannot be re-run"
    tick = NOW
    for _ in range(MAX_RERUN_CALLS + 2):
        assert heal_wedged_aggregates(app, now=tick) == 0
        tick += timedelta(minutes=app.config.auto_merge.aggregate_heal_interval_minutes + 1)

    assert len(gh.rerun_calls) == MAX_RERUN_CALLS
    failed = query_events(app.paths.state_file, kind="aggregate_job_rerun_failed")
    assert len(failed) == MAX_RERUN_CALLS
    assert failed[0]["payload"]["stage"] == "rerun" and "403" in failed[0]["payload"]["error"]
    assert query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered") == []


def test_a_refused_rerun_does_not_burn_the_one_rerun_budget(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    _wedge(gh)
    gh.rerun_error = "HTTP 502"
    assert heal_wedged_aggregates(app, now=NOW) == 0
    gh.rerun_error = None
    later = NOW + timedelta(minutes=app.config.auto_merge.aggregate_heal_interval_minutes + 1)

    assert heal_wedged_aggregates(app, now=later) == 1
    assert len(gh.rerun_calls) == 2


def test_an_unreadable_open_pr_list_never_escapes_the_loop(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    gh.pulls_error = GitHubError("api.github.com unreachable")

    assert heal_best_effort(app, now=NOW) == 0

    (event,) = query_events(app.paths.state_file, kind="aggregate_job_rerun_failed")
    assert event["payload"]["stage"] == "pass"
    assert "unreachable" in event["payload"]["error"]
    # The cadence was armed before the work, so a persistent outage is one error per interval.
    assert not heal.is_due(load_state(app.paths.state_file), NOW + timedelta(minutes=1))


def test_a_non_list_open_pr_response_is_an_error_not_an_empty_queue(tmp_path: Path) -> None:
    app, gh = _app(tmp_path)
    gh.run = lambda *_a, **_kw: {"message": "Bad credentials"}  # type: ignore[method-assign]

    with pytest.raises(GitHubError):
        heal_wedged_aggregates(app, now=NOW)


# ------------------------------------------------------------------------- wiring


def test_the_loop_maintenance_pass_runs_the_lane(tmp_path: Path) -> None:
    """L3: ``loop()``'s maintenance step actually calls the lane (not just unit-tested)."""
    app, gh = _app(tmp_path)
    _wedge(gh)

    run_deadline_guarded_maintenance(PassDeadline(None, CommandResult), app)

    assert gh.rerun_calls == [["run", "rerun", str(RUN), "--failed"]]
    assert len(query_events(app.paths.state_file, kind="aggregate_job_rerun_triggered")) == 1
