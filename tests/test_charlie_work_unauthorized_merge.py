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
