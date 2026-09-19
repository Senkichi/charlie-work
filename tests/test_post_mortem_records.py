"""Post-mortem sidecar record tests.

Split out of ``tests/test_post_mortem.py`` (issue #1567, Track 1):
``merge_attempt_snapshot`` ref folding/appending/no-op paths and
``read_post_mortem`` corrupt/absent sidecar degradation.
"""

from __future__ import annotations

import json
from pathlib import Path

from _post_mortem_fixtures import (
    _NOW,
    _build_sessions_db,
    _config_with_db,
    _make_worker,
)

from charlie_work.attempt_refs import AttemptSnapshot
from charlie_work.post_mortem import (
    MessageNode,
    classify_and_record,
    merge_attempt_snapshot,
    read_post_mortem,
)


# ---------------------------------------------------------------------------
# merge_attempt_snapshot
# ---------------------------------------------------------------------------


def test_merge_attempt_snapshot_folds_ref_into_existing_sidecar(tmp_path: Path) -> None:
    """merge_attempt_snapshot must attach the ref name/ahead-count to an
    existing post-mortem sidecar without corrupting message_nodes (a prior
    draft used asdict()+reconstruct, which silently turned MessageNode
    instances into plain dicts - see attempt_refs merge bug fix)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(
        db_path,
        nodes=[("tool", json.dumps({"tool": "bash"}), "2026-07-11T11:57:00")],
    )
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"

    classify_and_record(sessions_dir, config, worker, now=_NOW)
    before = read_post_mortem(sessions_dir, worker.issue_number)
    assert before is not None
    assert before.attempts == ()

    snapshot = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-42/attempt-1",
        old_tip="deadbeef" * 5,
        ahead_of_main_count=3,
    )
    merge_attempt_snapshot(sessions_dir, worker.issue_number, snapshot, now=_NOW)

    after = read_post_mortem(sessions_dir, worker.issue_number)
    assert after is not None
    assert len(after.attempts) == 1
    assert after.attempts[0].ref == "refs/charlie/attempts/issue-42/attempt-1"
    assert after.attempts[0].ahead_of_main == 3
    # message_nodes must survive untouched, still typed as MessageNode.
    assert after.message_nodes == before.message_nodes
    assert all(isinstance(n, MessageNode) for n in after.message_nodes)


def test_merge_attempt_snapshot_appends_without_overwriting_prior_attempts(
    tmp_path: Path,
) -> None:
    """A sidecar can outlive more than one redispatch attempt before it is
    next read/rotated — a second merge_attempt_snapshot call must append a
    second AttemptAttachment, not overwrite the first one (issue #261 F4:
    the prior implementation used dataclasses.replace on singular
    attempt_ref/attempt_ahead_of_main fields, silently losing the first
    attempt's ref)."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(db_path, nodes=[])
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"
    classify_and_record(sessions_dir, config, worker, now=_NOW)

    first = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-42/attempt-1",
        old_tip="a" * 40,
        ahead_of_main_count=1,
    )
    second = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-42/attempt-2",
        old_tip="b" * 40,
        ahead_of_main_count=2,
    )
    merge_attempt_snapshot(sessions_dir, worker.issue_number, first, now=_NOW)
    merge_attempt_snapshot(sessions_dir, worker.issue_number, second, now=_NOW)

    after = read_post_mortem(sessions_dir, worker.issue_number)
    assert after is not None
    assert [a.ref for a in after.attempts] == [
        "refs/charlie/attempts/issue-42/attempt-1",
        "refs/charlie/attempts/issue-42/attempt-2",
    ]
    assert [a.ahead_of_main for a in after.attempts] == [1, 2]


def test_merge_attempt_snapshot_noop_when_no_sidecar_exists(tmp_path: Path) -> None:
    """No existing post-mortem sidecar -> nothing to attach to -> no-op,
    never creates a sidecar out of thin air."""
    sessions_dir = tmp_path / "sessions"
    snapshot = AttemptSnapshot(
        ref_name="refs/charlie/attempts/issue-1/attempt-1",
        old_tip="abc123",
        ahead_of_main_count=1,
    )

    merge_attempt_snapshot(sessions_dir, 1, snapshot)

    assert read_post_mortem(sessions_dir, 1) is None


def test_merge_attempt_snapshot_noop_when_ref_name_is_none(tmp_path: Path) -> None:
    """A snapshot with no ref (nothing was preserved) must not touch an
    existing sidecar."""
    db_path = tmp_path / "sessions.db"
    _build_sessions_db(db_path, nodes=[])
    config = _config_with_db(db_path)
    worker = _make_worker()
    sessions_dir = tmp_path / "sessions"
    classify_and_record(sessions_dir, config, worker, now=_NOW)
    before = read_post_mortem(sessions_dir, worker.issue_number)
    assert before is not None

    snapshot = AttemptSnapshot(ref_name=None, old_tip=None, ahead_of_main_count=None)
    merge_attempt_snapshot(sessions_dir, worker.issue_number, snapshot)

    after = read_post_mortem(sessions_dir, worker.issue_number)
    assert after == before


# ---------------------------------------------------------------------------
# read_post_mortem: corrupt sidecar never raises
# ---------------------------------------------------------------------------


def test_read_post_mortem_corrupt_json_returns_none(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True)
    (sessions_dir / "issue-5.post-mortem.json").write_text("{not json", encoding="utf-8")

    assert read_post_mortem(sessions_dir, 5) is None


def test_read_post_mortem_absent_returns_none(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    assert read_post_mortem(sessions_dir, 5) is None
