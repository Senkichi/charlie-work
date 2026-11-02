"""Issue #2279: regression tests for the window-arming sites the review found
untested, plus direct unit tests of the sidecar read itself.

Every quota-window writer stamps the window (or the ``reviewer_quota``
record) with the dead session's ``role_entry`` ``(harness, model)`` so
``role_selection.window_covered`` can tell a fallback's quota window from
the selected entry's own death. A dropped stamp silently reverts a site to
adapter-wide blocking, so each site gets a test that fails if the stamp
never reaches the record:

* ``dead_worker_sweep.dead_sessions._persist_failure`` -- the
  ``dead_sessions_reap`` and ``dead_sessions_launch_failure`` sources;
* ``dead_worker_sweep.apply_stalled._arm_throttle`` -- the
  ``stalled_sessions_reap`` and ``stalled_sessions_rate_limit_defer``
  sources;
* ``role_quota_ledger.role_key_for_session`` /
  ``role_key_for_view`` -- the sidecar read every site shares.

The reconcile ``provider_throttle_detected`` and ``stalled_review_reap``
provider-API-error arms are covered in ``test_reconcile_drift_sessions.py``
and ``test_issue_1808_provider_api_error.py`` respectively, where their
fixtures already live.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from charlie_work import role_quota_ledger
from charlie_work.config import OrchestratorConfig
from charlie_work.dead_worker_sweep.apply_stalled import _arm_throttle
from charlie_work.dead_worker_sweep.dead_sessions import SessionPass, _persist_failure
from charlie_work.dead_worker_sweep.decide_stalled import DEFER_SOURCE, REAP_SOURCE
from charlie_work.dead_worker_sweep.stalled_model import ThrottleArm
from charlie_work.state import empty_state, load_state, save_state
from charlie_work.worker import WorkerView
from charlie_work.write_gate import WriteGate

from _fakes_github import FakeGitHub

ISSUE = 42
STAMP = ("devin-shell", "swe-2-high")


def _until() -> str:
    # Derived from the clock, never a literal: ``set_throttled_until`` is
    # monotonic against the live window, so a hardcoded date rots.
    return (datetime.now(UTC) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")


def _worker(**kw: object) -> WorkerView:
    base = {
        "adapter_kind": "devin",
        "issue_number": ISSUE,
        "repo_key": "",
        "pid": None,
        "started_at": "2026-10-02T04:30:00Z",
        "process_start_time": None,
        "log_path": "x.log",
        "worktree_path": "wt",
        "error": None,
        "failure_kind": None,
        "reclaimed": None,
    }
    return WorkerView(**{**base, **kw})


_UNSTAMPED = object()


def _stamp_sidecar(sessions_dir: Path, stamp: object = _UNSTAMPED) -> None:
    """Write a devin sidecar for ISSUE carrying ``stamp`` as ``role_entry``.

    The default stamps the full STAMP entry; pass ``None`` explicitly for a
    pre-#2086 (unstamped) sidecar.
    """
    payload: dict = {"issue_number": ISSUE}
    if stamp is not None:
        payload[role_quota_ledger.SESSION_ROLE_KEY] = (
            role_quota_ledger.session_stamp("worker", *STAMP, 0) if stamp is _UNSTAMPED else stamp
        )
    (sessions_dir / f"issue-{ISSUE}.json").write_text(json.dumps(payload), encoding="utf-8")


def _wg(state_file: Path) -> WriteGate:
    return WriteGate(dry_run=False, state_path=state_file, repo="charlie-work")


def _ctx(tmp_path: Path) -> tuple[SessionPass, Path]:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    state_file = tmp_path / "state.json"
    save_state(state_file, empty_state())
    return (
        SessionPass(
            sessions_dir=sessions_dir,
            state_file=state_file,
            gh=FakeGitHub(),
            config=OrchestratorConfig(),
            write_gate=_wg(state_file),
            repo_root=None,
            open_prs_by_issue={},
            now_for_health=datetime.now(UTC),
            fleet_repos=(),
            dispatching_repo_name="",
        ),
        state_file,
    )


def _window_events(state: dict) -> list[dict]:
    return [e for e in state["events"] if e["kind"] == "throttle_window_set"]


# --- role_key_for_session: the sidecar read -----------------------------------


def _write_devin_sidecar(sessions_dir: Path, payload: object) -> Path:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"issue-{ISSUE}.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return path


def test_role_key_for_session_full_stamp(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    _write_devin_sidecar(
        sessions_dir,
        {role_quota_ledger.SESSION_ROLE_KEY: role_quota_ledger.session_stamp("worker", *STAMP, 0)},
    )
    assert role_quota_ledger.role_key_for_session(sessions_dir, "devin", ISSUE) == STAMP


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(None, id="no-sidecar"),
        pytest.param("[1, 2]", id="non-dict-payload"),
        pytest.param("{not json", id="unreadable-json"),
        pytest.param({}, id="no-stamp-key"),
        pytest.param({role_quota_ledger.SESSION_ROLE_KEY: {"harness": "h"}}, id="harness-only"),
        pytest.param({role_quota_ledger.SESSION_ROLE_KEY: {"model": "m"}}, id="model-only"),
        pytest.param({role_quota_ledger.SESSION_ROLE_KEY: "a-string"}, id="non-mapping-stamp"),
    ],
)
def test_role_key_for_session_yields_none(tmp_path: Path, payload: object) -> None:
    sessions_dir = tmp_path / "sessions"
    if payload is not None:
        _write_devin_sidecar(sessions_dir, payload)
    assert role_quota_ledger.role_key_for_session(sessions_dir, "devin", ISSUE) is None


def test_role_key_for_session_unknown_adapter_kind(tmp_path: Path) -> None:
    """An adapter_kind with no harness mapping can never name a sidecar."""
    sessions_dir = tmp_path / "sessions"
    _write_devin_sidecar(
        sessions_dir,
        {role_quota_ledger.SESSION_ROLE_KEY: role_quota_ledger.session_stamp("worker", *STAMP, 0)},
    )
    assert role_quota_ledger.role_key_for_session(sessions_dir, "bogus", ISSUE) is None


def test_role_key_for_view_unpacks_to_a_pair_of_nones(tmp_path: Path) -> None:
    """The one unwrapping site: every caller gets ``(None, None)`` unstamped,
    never ``None`` itself."""
    sessions_dir = tmp_path / "sessions"
    w = _worker()
    assert role_quota_ledger.role_key_for_view(sessions_dir, w) == (None, None)
    _write_devin_sidecar(
        sessions_dir,
        {role_quota_ledger.SESSION_ROLE_KEY: role_quota_ledger.session_stamp("worker", *STAMP, 0)},
    )
    assert role_quota_ledger.role_key_for_view(sessions_dir, w) == STAMP


# --- dead_sessions._persist_failure --------------------------------------------


@pytest.mark.parametrize(
    ("launch_failure", "source"),
    [
        pytest.param(False, "dead_sessions_reap", id="dead_sessions_reap"),
        pytest.param(True, "dead_sessions_launch_failure", id="dead_sessions_launch_failure"),
    ],
)
def test_persist_failure_stamps_the_window_with_the_dead_sessions_role_key(
    tmp_path: Path, launch_failure: bool, source: str
) -> None:
    """Both ``_persist_failure`` sources stamp the window with the dying
    session's role entry; dropping the stamp reverts to adapter-wide
    blocking for a fallback's window."""
    ctx, state_file = _ctx(tmp_path)
    _stamp_sidecar(ctx.sessions_dir)

    _persist_failure(ctx, _worker(), "rate_limited", _until(), launch_failure=launch_failure)

    state = load_state(state_file)
    assert state["throttle_harness"] == STAMP[0]
    assert state["throttle_model"] == STAMP[1]
    windows = _window_events(state)
    assert len(windows) == 1
    assert windows[0]["payload"]["source"] == source
    assert (windows[0]["payload"]["harness"], windows[0]["payload"]["model"]) == STAMP


def test_persist_failure_without_a_stamp_leaves_the_window_untagged(tmp_path: Path) -> None:
    """A pre-#2086 sidecar (no ``role_entry``) writes the untagged window
    shape -- it must keep blocking."""
    ctx, state_file = _ctx(tmp_path)
    _stamp_sidecar(ctx.sessions_dir, stamp=None)

    _persist_failure(ctx, _worker(), "rate_limited", _until(), launch_failure=False)

    state = load_state(state_file)
    assert state["throttle_harness"] is None
    assert state["throttle_model"] is None


# --- apply_stalled._arm_throttle ------------------------------------------------


@pytest.mark.parametrize(
    ("source", "event_source"),
    [
        pytest.param(REAP_SOURCE, "stalled_sessions_reap", id="stalled_sessions_reap"),
        pytest.param(
            DEFER_SOURCE,
            "stalled_sessions_rate_limit_defer",
            id="stalled_sessions_rate_limit_defer",
        ),
    ],
)
def test_arm_throttle_stamps_the_window_with_the_sessions_role_key(
    tmp_path: Path, source: str, event_source: str
) -> None:
    """Both ``_arm_throttle`` branches stamp the window from the live
    sidecar -- this lane never reaps it, so the stamp is always readable."""
    sessions_dir = tmp_path / "sessions"
    _write_devin_sidecar(
        sessions_dir,
        {role_quota_ledger.SESSION_ROLE_KEY: role_quota_ledger.session_stamp("worker", *STAMP, 0)},
    )
    state_file = tmp_path / "state.json"
    save_state(state_file, empty_state())
    arm = ThrottleArm(until=_until(), source=source, reason="rate_limited", adapter_kind="devin")

    state = _arm_throttle(
        empty_state(), arm, _worker(), sessions_dir=sessions_dir, write_gate=_wg(state_file)
    )

    assert state["throttle_harness"] == STAMP[0]
    assert state["throttle_model"] == STAMP[1]
    windows = _window_events(state)
    assert len(windows) == 1
    assert windows[0]["payload"]["source"] == event_source


def test_arm_throttle_without_a_stamp_leaves_the_window_untagged(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    state_file = tmp_path / "state.json"
    save_state(state_file, empty_state())
    arm = ThrottleArm(
        until=_until(), source=REAP_SOURCE, reason="rate_limited", adapter_kind="devin"
    )

    state = _arm_throttle(
        empty_state(), arm, _worker(), sessions_dir=sessions_dir, write_gate=_wg(state_file)
    )

    assert state["throttle_harness"] is None
    assert state["throttle_model"] is None
