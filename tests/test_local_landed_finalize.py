"""Issue #2739: a zero-net-diff local record reaches the terminal "already
landed" edge instead of re-entering review forever.

The observed livelock: an approved record whose content already sits on the
base -- ``git diff <base>...<branch>`` is the empty string, the legitimate
"already landed" representation, not a failure -- lost its approval every
pass because ``approval_survives_head_move`` rejected the empty live diff.
The packet phase then rebuilt, dispatched a fresh reviewer, got a fresh
approval, and the merge gate's own base-sync merge moved the head again --
repeat. A closed tracker issue with stale ``agent:pr-open`` /
``agent:reviewing`` labels converged the same way: the drift reconciler
stripped them, the packet phase re-applied ``review_started``, repeat.

The fix gives an empty *successful* ``branch_diff`` terminal semantics at
three seams: the shared carry-forward predicate, the packet phase
(``_local_review_packets``), and the gate's launch site
(``_local_gate_launch``). A bounded "record left approved" abort streak
(``LOCAL_GATE_ABORT_STREAK_LIMIT``) escalates the cycle mechanically if it
ever re-forms.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from charlie_work.labels import transition
from charlie_work.local_lane import branch_diff, worktree_for_branch
from charlie_work.process_utils import is_pid_alive
from charlie_work.state import load_state, load_state_locked, save_state, state_lock
from charlie_work.workflow import OrchestratorApp

from _local_gate_async_fixtures import (  # noqa: E402
    SLEEP_SUITE,
    _adopt_and_approve,
    _commit_file,
    _event_kinds,
    _git,
    _init_repo,
    _kill_claimed_gate,
    _lane_app,
    _lane_config,
    _make_branch,
    _no_windows_child_enumeration,  # noqa: F401 -- registers the kill-stub autouse fixture
    _seed_dead_claim,
    _wait_pid_dead,
    _write_issue,
)

# Same late-import rule as test_local_merge_gate_async.py: the delegate
# module does ``import charlie_work.workflow as _wf`` at module level.
from charlie_work.orchestration.local_merge_gate import (  # noqa: E402
    LOCAL_GATE_ABORT_STREAK_LIMIT,
)

BRANCH = "agent/issue-7-x"
BRANCH_8 = "agent/issue-8-y"


@pytest.fixture
def lane_repo() -> Path:
    """Real git repo under the system temp dir ($GIT_DIR headroom)."""
    return Path(tempfile.mkdtemp(prefix="cw-landed-"))


def _landed_sync_merge(repo: Path, branch: str) -> str:
    """What the gate's base-sync leaves once the content is already on the
    base: a moved head whose ``git diff main...<branch>`` is empty."""
    _git(repo, "checkout", branch)
    _git(repo, "merge", "--no-ff", "main", "-m", "merge: sync main")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "main")
    return head


def _set_record_status(app: OrchestratorApp, pr_number: int, status: str) -> None:
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"][str(pr_number)]["status"] = status
        save_state(app.paths.state_file, state)


def test_landed_approved_record_finalizes_and_kills_inflight_suite(
    lane_repo: Path,
) -> None:
    """The observed interleave: approved record + claimed suite in flight,
    then the content lands on the base by another path and a sync merge
    empties the diff. The next packet pass must retire the record -- kill
    the stray suite, take the merged bookkeeping, close the issue -- not
    rebuild the packet and dispatch another reviewer."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(
        lane_repo,
        issues_dir,
        config=_lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE}),
    )
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)

    launched = app._local_merge_approved()
    assert launched[0]["outcome"] == "suite_launched"
    claimed_pid = int(load_state_locked(app.paths.state_file)["prs"]["7"]["local_suite_pid"])
    assert is_pid_alive(claimed_pid)

    # The work lands on the base by another path while the suite runs; a
    # second sync merge (in the gate worktree, where the branch is checked
    # out) is what empties the three-dot diff.
    _commit_file(lane_repo, "a.py", "a = 1\n", "feat: same content landed directly")
    gate_wt = worktree_for_branch(lane_repo, BRANCH)
    assert gate_wt is not None
    _git(gate_wt, "merge", "--no-ff", "main", "-m", "merge: sync main")
    assert branch_diff(lane_repo, "main", BRANCH) == ""

    marker = len(_event_kinds(app))
    outcome = app._local_review_packets()

    assert outcome["packets"] == []
    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert record["status"] == "merged"
    assert not record.get("local_suite_pid")
    _wait_pid_dead(claimed_pid)
    new_kinds = _event_kinds(app)[marker:]
    assert "local_review_skipped_landed" in new_kinds
    assert "merge_succeeded" in new_kinds
    assert "review_packet" not in new_kinds
    issue = app.gh.issue_view(7)
    assert issue["state"] == "CLOSED"
    assert {label["name"] for label in issue["labels"]} == {app.config.labels.done}


def test_landed_pending_record_finalizes_without_review(lane_repo: Path) -> None:
    """A parked branch adopted while already zero-diff goes straight to the
    terminal bookkeeping -- no packet, no ``review_started`` edge, no
    reviewer dispatch for a diff that does not exist."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    _commit_file(lane_repo, "a.py", "a = 1\n", "feat: same content landed directly")
    _landed_sync_merge(lane_repo, BRANCH)
    assert branch_diff(lane_repo, "main", BRANCH) == ""
    app = _lane_app(lane_repo, issues_dir)
    labels = app.config.labels
    _write_issue(issues_dir, 7, labels=(labels.ready, labels.in_progress))
    transition(app.gh, labels, 7, "local_work_ready", state_path=app.paths.state_file)
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state.setdefault("issues", {})["7"] = {
            "number": 7,
            "title": "Test issue",
            "status": "dispatched",
            "branch_name": BRANCH,
        }
        save_state(app.paths.state_file, state)

    outcome = app._local_review_packets()

    assert outcome["packets"] == []
    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert record["status"] == "merged"
    kinds = _event_kinds(app)
    assert "local_review_skipped_landed" in kinds
    assert "merge_succeeded" in kinds
    assert "review_packet" not in kinds
    issue = app.gh.issue_view(7)
    assert issue["state"] == "CLOSED"
    assert {label["name"] for label in issue["labels"]} == {labels.done}


def test_closed_issue_landed_branch_converges_once(lane_repo: Path) -> None:
    """Closed tracker issue + approved record + already-landed branch +
    stale lifecycle labels: one pass retires the record and collapses the
    labels to done; the NEXT pass emits nothing -- no label fight with the
    drift reconciler."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)
    _commit_file(lane_repo, "a.py", "a = 1\n", "feat: same content landed directly")
    _landed_sync_merge(lane_repo, BRANCH)
    # The stale-label shape the reconciler keeps stripping and the lane
    # kept re-applying: closed issue still wearing pr-open + reviewing.
    labels = app.config.labels
    _write_issue(issues_dir, 7, state="closed", labels=(labels.pr_open, labels.reviewing))

    app._local_review_packets()

    issue = app.gh.issue_view(7)
    assert issue["state"] == "CLOSED"
    assert {label["name"] for label in issue["labels"]} == {labels.done}
    assert load_state_locked(app.paths.state_file)["prs"]["7"]["status"] == "merged"

    marker = len(_event_kinds(app))
    second = app._local_review_packets()
    assert second["packets"] == []
    assert _event_kinds(app)[marker:] == []


def test_closed_issue_unprovable_diff_is_skipped_not_rebuilt(lane_repo: Path) -> None:
    """A closed issue whose diff cannot be computed at all gets neither a
    packet rebuild nor the ``review_started`` edge -- only the audit event.
    The label writes are left to the drift reconciler, so the lane and the
    reconciler stop fighting."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)
    labels = app.config.labels
    _write_issue(issues_dir, 7, state="closed", labels=(labels.pr_open, labels.reviewing))
    _set_record_status(app, 7, "reviewing")
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["7"]["baseRefName"] = "no-such-base-ref"
        save_state(app.paths.state_file, state)
    # Move the head so the packet would be rebuilt were the record allowed
    # past the closed-issue check.
    _git(lane_repo, "checkout", BRANCH)
    _commit_file(lane_repo, "b.py", "b = 1\n", "feat: more work")
    _git(lane_repo, "checkout", "main")

    marker = len(_event_kinds(app))
    outcome = app._local_review_packets()

    assert outcome["packets"] == []
    assert {e["reason"] for e in outcome["skipped"]} == {"issue_closed"}
    new_kinds = _event_kinds(app)[marker:]
    assert "local_review_skipped_issue_closed" in new_kinds
    assert "review_packet" not in new_kinds
    issue = app.gh.issue_view(7)
    assert {label["name"] for label in issue["labels"]} == {
        labels.pr_open,
        labels.reviewing,
    }
    assert load_state_locked(app.paths.state_file)["prs"]["7"]["status"] == "reviewing"


def test_closed_issue_with_real_diff_still_surfaces(lane_repo: Path) -> None:
    """``closed`` only suppresses the cycle when the branch cannot prove
    unmerged content: a real diff on a closed issue still gets its packet
    (``are_issues_open`` treats the same shape as still-open)."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)
    _set_record_status(app, 7, "reviewing")
    labels = app.config.labels
    _write_issue(issues_dir, 7, state="closed", labels=(labels.pr_open, labels.reviewing))
    # A head move is what makes the packet stale; the diff stays non-empty.
    _git(lane_repo, "checkout", BRANCH)
    _commit_file(lane_repo, "b.py", "b = 1\n", "feat: more work")
    _git(lane_repo, "checkout", "main")
    assert branch_diff(lane_repo, "main", BRANCH)

    outcome = app._local_review_packets()

    assert [p["issue"] for p in outcome["packets"]] == [7]


def test_gate_launch_on_landed_branch_merges_without_suite(lane_repo: Path) -> None:
    """The gate's own seam: an approved record whose diff is already empty
    takes the terminal bookkeeping at launch time instead of spawning a
    suite that can prove nothing -- and does not hold the one-gate slot."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)
    _commit_file(lane_repo, "a.py", "a = 1\n", "feat: same content landed directly")
    _landed_sync_merge(lane_repo, BRANCH)
    assert branch_diff(lane_repo, "main", BRANCH) == ""

    results = app._local_merge_approved()

    assert results[0]["outcome"] in ("merged", "already_merged")
    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert record["status"] == "merged"
    assert "local_suite_launched" not in _event_kinds(app)
    assert "local_review_skipped_landed" in _event_kinds(app)


def test_landed_record_does_not_starve_a_real_record(lane_repo: Path) -> None:
    """A zero-diff record sorted ahead of a real one must not hold the
    one-gate slot: 7 finalizes without a suite, 8 still launches."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head7 = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    head8 = _make_branch(lane_repo, BRANCH_8, "b.py", "b = 1\n")
    app = _lane_app(
        lane_repo,
        issues_dir,
        config=_lane_config(lane_repo, issues_dir, dispatch={"test_command": SLEEP_SUITE}),
    )
    try:
        _adopt_and_approve(app, issues_dir, 7, BRANCH, head7)
        _adopt_and_approve(app, issues_dir, 8, BRANCH_8, head8)
        _commit_file(lane_repo, "a.py", "a = 1\n", "feat: same content landed directly")
        _landed_sync_merge(lane_repo, BRANCH)

        results = app._local_merge_approved()

        by_pr = {entry["pr"]: entry for entry in results}
        assert by_pr[7]["outcome"] in ("merged", "already_merged")
        assert by_pr[8]["outcome"] == "suite_launched"
        record8 = load_state_locked(app.paths.state_file)["prs"]["8"]
        assert is_pid_alive(int(record8["local_suite_pid"]))
    finally:
        _kill_claimed_gate(app, 8)


def test_repeated_abort_with_unchanged_patch_id_escalates(lane_repo: Path) -> None:
    """Backstop for the loop shape: every "record left approved" gate abort
    under an unchanged patch-id counts one streak; at the bound the record
    escalates mechanically instead of burning another reviewer cycle."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)

    for _ in range(LOCAL_GATE_ABORT_STREAK_LIMIT):
        _seed_dead_claim(app, lane_repo, 7, head)
        _set_record_status(app, 7, "reviewing")
        entry = app._local_merge_approved()[0]

    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert entry["outcome"] == "error"
    assert record["status"] == "escalated"
    assert record["escalation_reason"] == "local_gate_abort_streak_exceeded"
    issue = app.gh.issue_view(7)
    assert {label["name"] for label in issue["labels"]} == {
        app.config.labels.operator_queue,
        # ``automated-ready`` is not a workflow label, so the mechanical
        # escalation edge leaves it alone.
        app.config.labels.ready,
    }


def test_abort_streak_resets_when_patch_id_changes(lane_repo: Path) -> None:
    """The streak counts aborts against ONE patch-id: genuinely new content
    re-entering review re-baselines it instead of inheriting the last
    episode's count."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)

    for _ in range(LOCAL_GATE_ABORT_STREAK_LIMIT - 1):
        _seed_dead_claim(app, lane_repo, 7, head)
        _set_record_status(app, 7, "reviewing")
        assert app._local_merge_approved()[0]["outcome"] == "gate_aborted"

    # New content arrived between aborts: the recorded patch-id changed, so
    # the next abort restarts the streak instead of tripping the bound.
    with state_lock(app.paths.state_file):
        state = load_state(app.paths.state_file)
        state["prs"]["7"]["reviewed_patch_id"] = "patch-id-of-new-content"
        save_state(app.paths.state_file, state)
    _seed_dead_claim(app, lane_repo, 7, head)
    _set_record_status(app, 7, "reviewing")
    entry = app._local_merge_approved()[0]

    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert entry["outcome"] == "gate_aborted"
    assert record["local_gate_abort_streak"] == 1
    assert record["status"] == "reviewing"


def test_finalize_resets_abort_streak(lane_repo: Path) -> None:
    """A record carrying an abort streak that reaches the terminal edge
    restarts clean -- the counter is per-episode, not per-record-lifetime."""
    issues_dir = lane_repo / "docs" / "issues"
    _init_repo(lane_repo)
    head = _make_branch(lane_repo, BRANCH, "a.py", "a = 1\n")
    app = _lane_app(lane_repo, issues_dir)
    _adopt_and_approve(app, issues_dir, 7, BRANCH, head)
    _seed_dead_claim(app, lane_repo, 7, head)
    _set_record_status(app, 7, "reviewing")
    app._local_merge_approved()
    _set_record_status(app, 7, "approved")
    _commit_file(lane_repo, "a.py", "a = 1\n", "feat: same content landed directly")
    _landed_sync_merge(lane_repo, BRANCH)

    results = app._local_merge_approved()

    assert results[0]["outcome"] in ("merged", "already_merged")
    record = load_state_locked(app.paths.state_file)["prs"]["7"]
    assert record["local_gate_abort_streak"] == 0
    assert record["local_gate_abort_streak_patch_id"] is None
