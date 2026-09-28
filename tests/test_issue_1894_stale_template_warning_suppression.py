"""Tests for issue #1894: reap_loop suppresses the
``review_packet_template_stale`` WARNING for PR statuses beyond "escalated".

The #1338 suppression names only "escalated", but the WARNING's
non-convergence is also reachable through statuses that predicate
deliberately does not cover (state.py documents why ``_escalation_flags``
stays narrow -- a real policy boundary, not an oversight). The predicate
itself lives in ``charlie_work.escalation._stale_template_warning_suppressed``
(extracted out of ``reap_loop._loop_body`` to keep reap_loop.py under the
800-line module cap); these tests exercise it end to end through
``app.loop()``.

Helpers shared with ``test_fix_review_packet_template_stale.py`` are
INLINED here -- tests/test_zero_cross_test_import_guard.py (issue #1284)
bans test_* -> test_* imports, so each test module is self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

from _fakes_github import FakeGitHub
from _review_fixtures import _make_loop_app


# ---------------------------------------------------------------------------
# helpers duplicated from test_fix_review_packet_template_stale.py
# (issue #1284 self-containment -- see module docstring)
# ---------------------------------------------------------------------------


def _pr_dir(tmp_path: Path, pr_number: int) -> Path:
    return tmp_path / ".var" / "charlie-work" / "prs" / f"pr-{pr_number}"


def _plant_packet(
    tmp_path: Path,
    pr_number: int,
    *,
    head_sha: str,
    template_sha: str | None,
) -> Path:
    """Plant a review packet fixture with a stamped template digest."""
    pr_dir = _pr_dir(tmp_path, pr_number)
    pr_dir.mkdir(parents=True, exist_ok=True)
    pr_json: dict = {"number": pr_number, "headRefOid": head_sha}
    if template_sha is not None:
        pr_json["prompt_template_sha"] = template_sha
    (pr_dir / "pr.json").write_text(json.dumps(pr_json), encoding="utf-8")
    (pr_dir / "review-prompt.md").write_text(
        f"review prompt for PR #{pr_number}", encoding="utf-8"
    )
    return pr_dir


def _pr456(head_sha: str) -> dict:
    return {
        "number": 456,
        "title": "Fix #123: search",
        "url": "https://example.test/pull/456",
        "headRefName": "agent/issue-123-fix-search",
        "headRefOid": head_sha,
        "body": "Closes #123",
        "labels": [],
        "isCrossRepository": False,
    }


def _make_loop_app_with_required_checks(
    tmp_path: Path, *, prs: list[dict], required_checks: tuple[str, ...]
) -> tuple[OrchestratorApp, FakeGitHub]:
    """Build a loop() app whose janitor gate enforces ``required_checks``.

    Mirrors ``_make_loop_app`` but configures required checks so the janitor
    can actually fail (and then heal) on them -- the default
    ``_approved_automerge`` leaves ``required_checks=()`` and the janitor is
    vacuously green, which cannot exercise the janitor-diagnostics refresh
    path the suppression predicate reads.
    """
    from charlie_work.config import AutoMergeConfig, ReviewConfig

    config = OrchestratorConfig(
        review=ReviewConfig(require_tests_or_rationale=False),
        auto_merge=AutoMergeConfig(required_checks=required_checks, require_approved_review=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    for pr in prs:
        pr.setdefault("state", "OPEN")
    fake_gh.prs = prs
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    return app, fake_gh


def paths_from_app(app: OrchestratorApp) -> Path:
    return app.paths.state_file


def _events_of_kind(state_path: Path, kind: str) -> list[dict]:
    from charlie_work.instrumentation import query_events

    return query_events(state_path, kind=kind)


# ---------------------------------------------------------------------------
# Issue #1894: the #1338 stale-template suppression names only "escalated".
# This PR extends it to two more shapes, for different reasons:
#
# * janitor_blocked + is_missing_checks_only_block: the flag marks the
#   TRANSIENT "required checks not yet reported" population (janitor.py --
#   "Required check(s) missing" is the sole failure), durable only alongside
#   ci_run_never_created_head (adapters.py). While the flag reads true on a
#   pass, review()'s deterministic janitor gate short-circuits before packet
#   regen, so the WARNING cannot converge while it holds.
# * a "blocked" record on the PR or its linked issue (SINK_STATUSES' other
#   member). Covered per the owner's #1894 amendment / #1897 proposal even
#   though regen is NOT categorically unreachable: review()'s entry gate
#   excludes only "escalated", so a blocked record still flows through
#   review()'s main path -- janitor green regenerates the packet and flips
#   status to "reviewing" in the same pass; janitor red rewrites status to
#   "janitor_blocked". The suppression is deliberate one-shot silencing of
#   a WARNING that carries no automated remediation while a human-owned
#   record stands.
#
# review() is still called for these statuses: the janitor-diagnostics
# refresh, ci_run_never_created detection, and the stale-checks-retrigger
# self-heal lane all live inside review()'s main janitor-gate path, so
# skipping the call would freeze the predicate's own inputs -- the same
# #1397/#1443 frozen-diagnostics failure mode.
# ---------------------------------------------------------------------------


def _seed_pr_janitor_blocked_missing_checks(
    app: OrchestratorApp, pr_number: int, issue_number: int
) -> None:
    """Mark a PR janitor_blocked on a missing-checks-only janitor failure --
    the shape review()'s janitor-gate bookkeeping persists each pass."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["Required check(s) missing: Tests passed"],
            "is_missing_checks_only_block": True,
        }
        save_state(app.paths.state_file, state)


def _seed_pr_blocked(app: OrchestratorApp, pr_number: int, issue_number: int) -> None:
    """Mark only the PR record as "blocked" in state.json -- the shape
    record_review's blocked path persists (the other SINK_STATUSES member, a
    judgment verdict parked on agent:human-needed). No issue record is
    seeded, so a suppression here can only come from the PR-record sink
    arm -- co-seeding the issue would mask which clause fired."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)] = {
            "number": pr_number,
            "issue_number": issue_number,
            "status": "blocked",
        }
        save_state(app.paths.state_file, state)


def _seed_issue_blocked(app: OrchestratorApp, issue_number: int) -> None:
    """Mark only the linked issue as "blocked" (PR record left absent or
    seeded separately) -- isolates the issue-record sink arm."""
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["issues"][str(issue_number)] = {
            "number": issue_number,
            "status": "blocked",
        }
        save_state(app.paths.state_file, state)


def test_loop_skips_template_stale_warning_for_janitor_blocked_missing_checks(
    tmp_path: Path,
) -> None:
    """A janitor_blocked PR whose sole janitor failure is required checks
    not yet reported (is_missing_checks_only_block=True) cannot reach
    packet regen while the flag holds -- review() short-circuits in the
    deterministic janitor gate -- so the stale-template WARNING must not
    re-fire every pass (issue #1894; the swole PR #298 shape). The flag
    marks the TRANSIENT population (janitor.py): it is durable only
    alongside ci_run_never_created_head, and review()'s per-pass refresh
    lifts the suppression as soon as the checks report (proven by the
    heal test below). review() is still called each pass -- the
    janitor-diagnostics refresh and the ci_run_never_created /
    stale-checks-retrigger lanes live inside it."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    # The required check never reports -> the janitor keeps failing
    # missing-checks-only and refresh-stamps is_missing_checks_only_block.
    fake_gh.pr_checks = lambda _number: []  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_janitor_blocked_missing_checks(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    # Run two passes -- the bug fired the WARNING every pass.
    app.loop(limit=0)
    app.loop(limit=0)

    # review() WAS invoked each pass -- it re-runs the janitor gate (which
    # refreshes janitor_ok/janitor_failures/is_missing_checks_only_block
    # and owns the stale-checks retrigger lane), then short-circuits before
    # packet regen.
    assert review_calls.count(456) == 2
    # No staleness WARNING -- the regen is unreachable while the check stays
    # missing, so the WARNING would spam identically without converging.
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_skips_template_stale_warning_for_blocked_pr_record(
    tmp_path: Path,
) -> None:
    """Isolates the PR-record sink arm: status="blocked" on the PR record
    ALONE (no issue record seeded, so neither the issue-sink nor the
    escalated clause can fire, and the record is not janitor_blocked) must
    suppress the WARNING. Single pass -- review() still runs and is
    expected to move the PR record off "blocked" itself (this fixture's
    body is janitor-red, so the janitor gate rewrites the status to
    janitor_blocked), which is exactly why the blocked arm is one-shot
    silencing rather than a claim that regen is unreachable."""
    from charlie_work.state import load_state

    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_blocked(app, pr_number=456, issue_number=123)
    assert "stale-digest" != current_sha

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    app.loop(limit=0)

    assert review_calls.count(456) == 1
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)
    # review()'s janitor gate moved the record off "blocked" within the
    # same pass (janitor red -> janitor_blocked): the suppressed state is
    # not parked-forever, and the predicate must be re-evaluated against
    # whatever status the record carries next pass.
    assert load_state(app.paths.state_file)["prs"]["456"]["status"] != "blocked"


def test_loop_skips_template_stale_warning_for_blocked_issue_no_pr_record(
    tmp_path: Path,
) -> None:
    """Isolates the issue-record sink arm: status="blocked" on the linked
    ISSUE alone, with NO PR record at all (so neither the PR-sink nor the
    janitor_blocked+flag clause can fire), suppresses the WARNING on the
    pass where the issue record reads blocked."""
    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_issue_blocked(app, issue_number=123)

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    app.loop(limit=0)

    assert review_calls.count(456) == 1
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_skips_template_stale_warning_for_blocked_issue_nonsink_pr(
    tmp_path: Path,
) -> None:
    """Same isolation as the no-PR-record case but with a live PR record
    in a non-sink, non-janitor_blocked status ("reviewing", no
    is_missing_checks_only_block): the issue record's "blocked" is the
    only clause that can suppress."""
    from charlie_work.state import load_state, save_state, state_lock

    pr = _pr456("sha-same")
    app, _ = _make_loop_app(tmp_path, prs=[pr])

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_issue_blocked(app, issue_number=123)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "reviewing",
        }
        save_state(app.paths.state_file, state)

    app.loop(limit=0)

    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_blocked_pr_green_janitor_converges_without_warning(
    tmp_path: Path,
) -> None:
    """Pins the blocked arm's actual semantics for a janitor-GREEN PR
    (body carries a tests/verification mention, so the gate passes):
    review() is NOT gated by the suppression -- it regenerates the packet
    with the current template in the same pass and stamps janitor_ok=True
    (the status->"reviewing" flip lives in the same write, gated on
    review_dispatch.enabled which defaults off here -- issue #868). The
    WARNING is silenced for that one pass and the packet still converges:
    deliberate one-shot silencing, not a regen-unreachable claim. The
    pre-existing blocked tests use _pr456's bare 'Closes #123' body,
    which is janitor-red, so only a green fixture proves this."""
    from charlie_work.state import load_state

    pr = _pr456("sha-same")
    # Janitor-green body: linked issue via "Closes #" plus a tests
    # mention satisfies _check_body's require_tests_or_rationale gate.
    pr["body"] = "Closes #123\n\nTests: regression coverage added."
    app, _ = _make_loop_app(tmp_path, prs=[pr])
    current_sha = app._review_template_sha()

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_blocked(app, pr_number=456, issue_number=123)

    app.loop(limit=0)
    app.loop(limit=0)

    # The WARNING never fired -- the one shot it would have produced on
    # pass 1 was suppressed.
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)
    # ...and the packet still converged: review() regenerated it with the
    # current template digest and re-stamped the janitor verdict green.
    pr_json = json.loads((_pr_dir(tmp_path, 456) / "pr.json").read_text(encoding="utf-8"))
    assert pr_json["prompt_template_sha"] == current_sha
    pr_state = load_state(app.paths.state_file)["prs"]["456"]
    assert pr_state["janitor_ok"] is True
    assert pr_state["janitor_failures"] == []


def test_loop_skips_template_stale_warning_for_swole_blocked_shape(
    tmp_path: Path,
) -> None:
    """The production evidence shape (swole #196/PR #298): the linked
    ISSUE carries status="blocked" while the PR record is janitor_blocked
    on unreported required checks. Deliberately NOT an isolation test --
    both the issue-sink and the janitor_blocked+flag clauses can fire here
    (the per-clause isolation lives in the tests above); this pins the
    combined shape the issue was filed on, across two passes."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )

    fake_gh.pr_checks = lambda _number: []  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_janitor_blocked_missing_checks(app, pr_number=456, issue_number=123)
    _seed_issue_blocked(app, issue_number=123)

    review_calls: list[int] = []
    original_review = app.review

    def tracking_review(pr_number: int) -> object:
        review_calls.append(pr_number)
        return original_review(pr_number)

    app.review = tracking_review  # type: ignore[method-assign]

    app.loop(limit=0)
    app.loop(limit=0)

    assert review_calls.count(456) == 2
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)


def test_loop_fires_template_stale_warning_for_janitor_blocked_without_flag(
    tmp_path: Path,
) -> None:
    """janitor_blocked alone is NOT enough for suppression -- a record that
    lacks is_missing_checks_only_block (a different janitor failure class,
    or a record persisted before the flag existed) still has live
    remediation lanes, so the WARNING keeps firing. The suppression is
    scoped to the non-convergent missing-checks-only shape only."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    fake_gh.pr_checks = lambda _number: []  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")

    # Seed janitor_blocked WITHOUT the flag (e.g. a record persisted by an
    # older version): pass 1 must still fire the WARNING. review() then
    # refreshes the real missing-checks shape, so pass 2 is suppressed --
    # the flag flip across passes is itself the refresh staying live.
    from charlie_work.state import load_state, save_state, state_lock

    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["456"] = {
            "number": 456,
            "issue_number": 123,
            "status": "janitor_blocked",
            "janitor_ok": False,
            "janitor_failures": ["PR is marked as draft"],
        }
        save_state(app.paths.state_file, state)
    assert "stale-digest" != current_sha

    app.loop(limit=0)
    app.loop(limit=0)

    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert sum(1 for e in events if e.get("pr_number") == 456) == 1


def test_loop_janitor_blocked_missing_checks_heals_when_check_appears(
    tmp_path: Path,
) -> None:
    """Keeping review() on the call path is what lets the suppression lift:
    once the missing check reports, the janitor gate passes and the packet
    regenerates with the current template -- the PR converges instead of
    wedging inside its own suppressed state. This is the property the
    issue's "keep the janitor-diagnostics-refresh behavior" clause pins."""
    pr = _pr456("sha-same")
    app, fake_gh = _make_loop_app_with_required_checks(
        tmp_path, prs=[pr], required_checks=("Tests passed",)
    )
    current_sha = app._review_template_sha()

    checks_sequence: list[list[dict]] = [
        [],  # pass 1: required check still missing
        [{"name": "Tests passed", "state": "SUCCESS"}],  # pass 2: it reports
    ]
    checks_calls: list[int] = []

    def fake_pr_checks(number: int) -> list[dict]:
        checks_calls.append(number)
        return checks_sequence[min(len(checks_calls) - 1, len(checks_sequence) - 1)]

    fake_gh.pr_checks = fake_pr_checks  # type: ignore[method-assign]

    _plant_packet(tmp_path, 456, head_sha="sha-same", template_sha="stale-digest")
    _seed_pr_janitor_blocked_missing_checks(app, pr_number=456, issue_number=123)

    app.loop(limit=0)
    app.loop(limit=0)

    # review() regenerated the packet with the current template digest --
    # the suppressed state converged instead of spamming.
    pr_json_path = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "pr.json"
    pr_json = json.loads(pr_json_path.read_text(encoding="utf-8"))
    assert pr_json["prompt_template_sha"] == current_sha
    events = _events_of_kind(paths_from_app(app), "review_packet_template_stale")
    assert not any(e.get("pr_number") == 456 for e in events)
