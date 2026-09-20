"""Unauthorized-merge tripwire: baseline arming, post-arm detection, dry-run safety, and ack suppression.

Split out of ``tests/test_charlie_work.py`` (issue #1553,
Track-1 wave 7/8).
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import (
    _ack_unauthorized_merge,
    _arm_unauthorized_merge_tripwire,
    _merged_worker_pr,
)
from charlie_work import github as github_module
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401
import json
import subprocess
from charlie_work.state import load_state


def test_unauthorized_merge_tripwire_arms_instead_of_flagging_history(tmp_path: Path) -> None:
    """The first pass records pre-existing uncovered merges as a baseline and reports nothing.

    Without this bound the tripwire asserts its policy retroactively over the
    whole 500-PR ``merged_pr_list()`` window. Measured against the live repo
    before this landed, that was 48 findings appended to ``loop()``'s ``errors``
    bucket on EVERY pass — there is no dedupe — which pins ok=False forever and
    buries a real self-merge in constant noise. A control that can never go quiet
    is not a control.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    history = [
        _merged_worker_pr(101, 91, "sha-101"),
        _merged_worker_pr(102, 92, "sha-102"),
    ]

    # No decision files at all: both would be flagged by an unbounded tripwire.
    assert app._detect_unauthorized_merges(history) == [], (
        "the arming pass must report nothing, not the whole backlog"
    )

    state = load_state(paths.state_file)
    baseline = state.get(UNAUTHORIZED_MERGE_BASELINE_KEY)
    assert isinstance(baseline, dict), "arming must persist a baseline to state.json"
    assert baseline["pre_existing_prs"] == [101, 102]
    assert baseline["armed_at"]

    # The backlog must remain auditable rather than being silently dropped: the
    # event carries the full PR list, not just a count.
    armed = [e for e in state["events"] if e["kind"] == "unauthorized_merge_baseline_armed"]
    assert len(armed) == 1
    assert armed[0]["payload"]["pre_existing_count"] == 2
    assert armed[0]["payload"]["pre_existing_prs"] == [101, 102]


def test_unauthorized_merge_tripwire_flags_merges_after_arming(tmp_path: Path) -> None:
    """A merge that lands after arming is still flagged — the baseline suppresses history only."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    history = [_merged_worker_pr(101, 91, "sha-101")]
    assert app._detect_unauthorized_merges(history) == []

    # A NEW uncovered merge appears. Note its number (99) is BELOW the baselined
    # PR's (101): a high-water-mark watermark would wrongly exempt it, which is
    # why the baseline is an explicit set. Three worker PRs were open and
    # below the highest merged PR when this armed on the live repo, so this is
    # the real case, not a contrived one.
    later = [*history, _merged_worker_pr(99, 89, "sha-99")]
    detected = app._detect_unauthorized_merges(later)

    assert [d["pr"] for d in detected] == [99], (
        f"only the post-arming merge may be flagged, got {detected}"
    )
    assert detected[0]["issue"] == 89
    assert detected[0]["live_head_sha"] == "sha-99"


def test_unauthorized_merge_baseline_arms_once_and_does_not_widen(tmp_path: Path) -> None:
    """Re-running the tripwire must not re-arm and swallow merges that landed after the first pass."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    first = load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]

    grown = [_merged_worker_pr(101, 91, "sha-101"), _merged_worker_pr(102, 92, "sha-102")]
    for _ in range(3):
        assert [d["pr"] for d in app._detect_unauthorized_merges(grown)] == [102]

    after = load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]
    assert after == first, "the baseline must be written once and never widened"


def test_unauthorized_merge_tripwire_does_not_arm_when_pr_fetch_fails(tmp_path: Path) -> None:
    """A gh failure must not bake an empty baseline that permanently exempts real history.

    This is the sharpest failure mode of the whole mechanism: if the very first
    pass after deployment cannot fetch the merged PR list and arms anyway, the
    baseline records "nothing pre-existed" and the tripwire is then permanently
    blind to every merge it never saw.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class FakeGitHubFailingMergedList(FakeGitHub):
        def merged_pr_list(self):
            raise github_module.GitHubError("gh unavailable")

    app = OrchestratorApp(tmp_path, paths, config, FakeGitHubFailingMergedList())

    assert app._detect_unauthorized_merges() == []
    assert UNAUTHORIZED_MERGE_BASELINE_KEY not in load_state(paths.state_file), (
        "a failed fetch must leave the tripwire unarmed so it can arm from real data later"
    )

    # Proof the unarmed state is recoverable: once gh works, arming sees the real
    # history and the tripwire still reports post-arming merges.
    app.gh = FakeGitHub()  # type: ignore[assignment]
    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    assert load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]["pre_existing_prs"] == [
        101
    ]


def test_unauthorized_merge_baseline_arming_writes_nothing_in_dry_run(tmp_path: Path) -> None:
    """--dry-run must not persist the baseline (issues #609/#613/#621).

    A preview that arms the tripwire would silently consume the one-time arming
    opportunity, permanently baselining whatever happened to be merged at preview
    time. The preview still reports nothing, which is what an armed pass reports.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.state import load_state
    from charlie_work.workflow import UNAUTHORIZED_MERGE_BASELINE_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub(), dry_run=True)

    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    assert UNAUTHORIZED_MERGE_BASELINE_KEY not in load_state(paths.state_file), (
        "dry-run must leave no baseline behind"
    )


def test_unauthorized_merge_ack_suppresses_acknowledged_finding(tmp_path: Path) -> None:
    """An acknowledged post-arming finding must stop polluting every pass (issue #673).

    The tripwire keeps its bite until a finding is explicitly acknowledged; once
    acked it is filtered the same way the pre-arming baseline filters history, so
    ``ok=False`` / ``errors`` go back to meaning "there is something new to look
    at" instead of "the mechanism cannot ever clear this".
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    fake_gh = FakeGitHub()
    fake_gh.issues = []
    fake_gh.prs = [
        _merged_worker_pr(1408, 1404, "sha-1408"),
        _merged_worker_pr(1392, 1268, "sha-1392"),
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Pre-ack: both post-arming findings are flagged.
    detected = app._detect_unauthorized_merges(fake_gh.prs)
    assert sorted(d["pr"] for d in detected) == [1392, 1408]

    # Acknowledge both (e.g. root cause fixed in #672, confirmed benign per #634).
    _ack_unauthorized_merge(paths, 1408, "root cause fixed in #672")
    _ack_unauthorized_merge(paths, 1392, "root cause fixed in #672")

    # Post-ack: both are suppressed — the tripwire can go quiet.
    assert app._detect_unauthorized_merges(fake_gh.prs) == [], (
        "an acknowledged finding must not be re-reported on every pass"
    )

    # A NEW post-arming finding is still flagged — ack only suppresses what was
    # explicitly acked, it does not auto-acknowledge anything else.
    new_prs = [*fake_gh.prs, _merged_worker_pr(1500, 1501, "sha-1500")]
    detected_after = app._detect_unauthorized_merges(new_prs)
    assert [d["pr"] for d in detected_after] == [1500]


def test_unauthorized_merge_ack_does_not_suppress_unacked(tmp_path: Path) -> None:
    """The ack set suppresses only the acked PR, never a sibling finding (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    _ack_unauthorized_merge(paths, 1408, "fixed")

    fake_gh = FakeGitHub()
    prs = [
        _merged_worker_pr(1408, 1404, "sha-1408"),
        _merged_worker_pr(1392, 1268, "sha-1392"),
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    detected = app._detect_unauthorized_merges(prs)
    assert [d["pr"] for d in detected] == [1392], (
        "acking #1408 must not also suppress the unrelated #1392 finding"
    )


def test_detect_unauthorized_merges_flags_worker_self_merge(tmp_path: Path) -> None:
    """A merged worker branch without an approved review decision is flagged as a possible self-merge (issue #502)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class FakeGitHubWithMergedWorkerPR(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.prs = [
                {
                    "number": 501,
                    "title": "fix: worker self-merge",
                    "url": "https://example.test/pull/501",
                    "headRefName": "agent/issue-494-fix",
                    "baseRefName": "main",
                    "headRefOid": "sha-501",
                    "state": "MERGED",
                    "isCrossRepository": False,
                    "body": "Closes #494",
                    "labels": [],
                },
                {
                    "number": 502,
                    "title": "fix: approved merge",
                    "url": "https://example.test/pull/502",
                    "headRefName": "agent/issue-495-fix",
                    "baseRefName": "main",
                    "headRefOid": "sha-502",
                    "state": "MERGED",
                    "isCrossRepository": False,
                    "body": "Closes #495",
                    "labels": [],
                },
            ]

    fake_gh = FakeGitHubWithMergedWorkerPR()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # PR 502 has an approved review decision on the merged head
    pr_dir = paths.prs / "pr-502"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": "sha-502",
            }
        ),
        encoding="utf-8",
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 501
    assert detected[0]["issue"] == 494
    assert detected[0]["decision"] == "missing"


def test_detect_unauthorized_merges_flags_approved_sha_mismatch(tmp_path: Path) -> None:
    """A merged worker branch with an approved decision for a different head SHA is flagged (issue #502 / cw #467)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)

    class FakeGitHubWithMergedWorkerPR(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.prs = [
                {
                    "number": 503,
                    "title": "fix: approved but then amended",
                    "url": "https://example.test/pull/503",
                    "headRefName": "agent/issue-496-fix",
                    "baseRefName": "main",
                    "headRefOid": "sha-503-final",
                    "state": "MERGED",
                    "isCrossRepository": False,
                    "body": "Closes #496",
                    "labels": [],
                },
            ]

    fake_gh = FakeGitHubWithMergedWorkerPR()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # The approval was recorded for an earlier head; the merged head differs.
    pr_dir = paths.prs / "pr-503"
    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "approved",
                "reviewed_head_sha": "sha-503-reviewed",
            }
        ),
        encoding="utf-8",
    )

    detected = app._detect_unauthorized_merges()

    assert len(detected) == 1
    assert detected[0]["pr"] == 503
    assert detected[0]["issue"] == 496
    assert detected[0]["decision"] == "approved"
    assert detected[0]["reviewed_head_sha"] == "sha-503-reviewed"
    assert detected[0]["live_head_sha"] == "sha-503-final"


def test_detect_unauthorized_merges_reuses_dispatch_merged_prs(tmp_path: Path) -> None:
    """loop() should reuse the merged PR list from dispatch() instead of calling merged_pr_list() again (issue #502 Finding 3)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    class CountingFakeGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.merged_pr_list_calls = 0

        def merged_pr_list(self):
            self.merged_pr_list_calls += 1
            return super().merged_pr_list()

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    _arm_unauthorized_merge_tripwire(paths)
    fake_gh = CountingFakeGitHub()
    fake_gh.prs = [
        {
            "number": 501,
            "title": "fix: worker self-merge",
            "url": "https://example.test/pull/501",
            "headRefName": "agent/issue-494-fix",
            "baseRefName": "main",
            "headRefOid": "sha-501",
            "state": "MERGED",
            "isCrossRepository": False,
            "body": "Closes #494",
            "labels": [],
        },
    ]

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # dispatch() fetches merged PRs as part of its normal pass.
    result = app.dispatch(limit=1)
    assert result.data.get("merged_prs") == fake_gh.prs
    assert fake_gh.merged_pr_list_calls == 1

    # The tripwire, when handed that list, must not make a second API call.
    detected = app._detect_unauthorized_merges(result.data["merged_prs"])
    assert fake_gh.merged_pr_list_calls == 1
    assert len(detected) == 1
    assert detected[0]["pr"] == 501


def test_detect_unauthorized_merges_against_real_rest_merged_pr_list(
    monkeypatch, tmp_path: Path
) -> None:
    """The tripwire must fire against the shape merged_pr_list() ACTUALLY returns.

    Every other tripwire test above builds its merged-PR fixture by hand and
    spells ``headRefOid`` explicitly. But ``merged_pr_list()`` is REST-only by
    construction (issue #361) and the REST payload spells that field
    ``head.sha``, so those fixtures asserted a key the production path did not
    emit: ``live_head_sha`` was always ``None``, ``head_matches`` was always
    False, and the SHA half of this control could never distinguish an
    authorized merge from a bypass. Fixed in #631; this test is the regression
    guard, and it exercises the real producer so the *contract* between
    ``merged_pr_list()`` and the tripwire is what is under test rather than a
    hand-written dict.
    """
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths

    merged_head_sha = "27a20fbd9c1e4d3a8f5b6c7d8e9f0a1b2c3d4e5f"
    rest_page = [
        {
            "number": 501,
            "title": "fix: a worker branch that got merged",
            "body": "Closes #494",
            "merged_at": "2026-07-20T20:19:07Z",
            "head": {
                "ref": "agent/issue-494-fix",
                "sha": merged_head_sha,
                "repo": {"full_name": "o/r"},
            },
            "base": {"repo": {"full_name": "o/r"}},
        }
    ]
    # merged_pr_list() paginates until it sees an empty page.
    responses = [json.dumps(rest_page), "[]"]

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout=responses.pop(0), stderr=""
        )

    monkeypatch.setattr(github_module.subprocess, "run", fake_run)

    # The real producer, driven off a real REST payload.
    merged_prs = github_module.GitHub(tmp_path).merged_pr_list()
    assert len(merged_prs) == 1
    assert merged_prs[0]["headRefOid"] == merged_head_sha, (
        "merged_pr_list() must map REST head.sha onto headRefOid (#631) — "
        "without it the tripwire below cannot compare SHAs at all"
    )

    # The real consumer. loop() hands the fetched list in as a parameter on the
    # hot path, so passing it explicitly is the production shape; FakeGitHub is
    # here only to satisfy the app's constructor, not to supply the fixture.
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    # Steady state, not the arming pass: this test is about the producer/consumer
    # SHA contract, so the baseline must not be what silences it.
    _arm_unauthorized_merge_tripwire(paths)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    pr_dir = paths.prs / "pr-501"
    pr_dir.mkdir(parents=True, exist_ok=True)
    decision_path = pr_dir / "review-decision.json"

    # 1. Approved for exactly the merged head -> authorized, tripwire silent.
    #    This is the assertion that fails when the REST normalizer omits
    #    headRefOid: head_matches degrades to False and a properly reviewed
    #    merge gets reported as a bypass.
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": merged_head_sha}),
        encoding="utf-8",
    )
    assert app._detect_unauthorized_merges(merged_prs) == []

    # 2. Approved for a DIFFERENT head -> flagged, and live_head_sha must carry
    #    the real REST head.sha rather than None.
    decision_path.write_text(
        json.dumps({"decision": "approved", "reviewed_head_sha": "stale-review-sha"}),
        encoding="utf-8",
    )
    detected = app._detect_unauthorized_merges(merged_prs)

    assert len(detected) == 1
    assert detected[0]["pr"] == 501
    assert detected[0]["issue"] == 494
    assert detected[0]["decision"] == "approved"
    assert detected[0]["reviewed_head_sha"] == "stale-review-sha"
    assert detected[0]["live_head_sha"] == merged_head_sha


def test_ack_unauthorized_merge_records_ack_and_event(tmp_path: Path) -> None:
    """``ack_unauthorized_merge`` persists the ack set and an audit event (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.ack_unauthorized_merge(1408, "root cause fixed in #672", by="operator")

    assert result.ok is True
    state = load_state(paths.state_file)
    acks = state.get(UNAUTHORIZED_MERGE_ACK_KEY)
    assert isinstance(acks, dict), "ack must persist an ack set to state.json"
    entry = acks["1408"]
    assert entry["reason"] == "root cause fixed in #672"
    assert entry["acknowledged_at"]
    assert entry["by"] == "operator"

    # The ack must be auditable: an event carries who/why/when.
    acked = [e for e in state["events"] if e["kind"] == "unauthorized_merge_acknowledged"]
    assert len(acked) == 1
    assert acked[0]["payload"]["pr"] == 1408
    assert acked[0]["payload"]["reason"] == "root cause fixed in #672"
    assert acked[0]["payload"]["by"] == "operator"


def test_ack_unauthorized_merge_requires_reason(tmp_path: Path) -> None:
    """An ack without a reason is rejected — a tripwire that can be silenced silently is no control (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.ack_unauthorized_merge(1408, "   ")
    assert result.ok is False
    assert "reason" in result.message.lower()
    assert UNAUTHORIZED_MERGE_ACK_KEY not in load_state(paths.state_file), (
        "a rejected ack must not have written anything to state"
    )


def test_ack_unauthorized_merge_updates_existing_ack(tmp_path: Path) -> None:
    """Re-acking a PR updates the record rather than refusing or duplicating (issue #673)."""
    from charlie_work.config import OrchestratorConfig
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import UNAUTHORIZED_MERGE_ACK_KEY

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    app.ack_unauthorized_merge(1408, "initial triage", by="alice")
    first = load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]["1408"]

    app.ack_unauthorized_merge(1408, "root cause fixed in #672", by="bob")
    second = load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]["1408"]

    assert len(load_state(paths.state_file)[UNAUTHORIZED_MERGE_ACK_KEY]) == 1, (
        "re-acking must not duplicate the entry"
    )
    assert second["reason"] == "root cause fixed in #672"
    assert second["by"] == "bob"
    assert second["acknowledged_at"] >= first["acknowledged_at"]
