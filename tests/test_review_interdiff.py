"""Unit tests for ``charlie_work.review_interdiff`` (issue #2837).

The packet-level behaviour (what a re-review packet says and stamps) is covered
end to end in ``test_issue_2837_incremental_review`` and
``test_charlie_work_review_round_history``; this module pins the module's own
failure posture and the single-source decisions those tests sit on top of:
errors come back as values, the packet builder and the round-kind classifier
agree on whether a round is incremental, and the prior head is the latest
*archived* round's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import charlie_work.review_interdiff as ri
from _incremental_review_fixtures import (
    ROUND1_HEAD,
    ROUND1_PR_DIFF,
    ROUND2_PR_DIFF,
    plant_round1,
)
from charlie_work.review_decision import _round_history_entries

_NOT_UTF8 = b"\xff\xfe\x80 not utf-8 \xc3"


# ---------------------------------------------------------------------------
# Errors come back as values.
# ---------------------------------------------------------------------------


def test_read_optional_returns_none_for_missing_undecodable_and_unreadable(
    tmp_path: Path,
) -> None:
    readable = tmp_path / "ok.patch"
    readable.write_text("diff\n", encoding="utf-8")
    undecodable = tmp_path / "bad.patch"
    undecodable.write_bytes(_NOT_UTF8)

    assert ri.read_optional(readable) == "diff\n"  # positive control
    assert ri.read_optional(tmp_path / "missing.patch") is None
    assert ri.read_optional(undecodable) is None
    # A directory exists but cannot be read as text: an OSError, still a value.
    assert ri.read_optional(tmp_path) is None


def test_archive_round_diff_copies_the_live_diff(tmp_path: Path) -> None:
    (tmp_path / "diff.patch").write_text(ROUND1_PR_DIFF, encoding="utf-8")
    round_dir = tmp_path / "rounds" / "round-1"

    archived = ri.archive_round_diff(tmp_path, round_dir)

    assert archived == round_dir / "diff.patch"
    assert archived is not None
    assert archived.read_text(encoding="utf-8") == ROUND1_PR_DIFF


def test_archive_round_diff_degrades_to_none_instead_of_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``record_review`` calls this AFTER the verdict is recorded: no failure
    mode here may abort it. Each degrades the next packet to the legacy
    ``compare_diff`` fallback instead."""
    round_dir = tmp_path / "rounds" / "round-1"

    # No live diff (a legacy packet).
    assert ri.archive_round_diff(tmp_path, round_dir) is None

    # A live diff that is not valid UTF-8.
    (tmp_path / "diff.patch").write_bytes(_NOT_UTF8)
    assert ri.archive_round_diff(tmp_path, round_dir) is None

    # A readable live diff whose atomic write fails.
    (tmp_path / "diff.patch").write_text(ROUND1_PR_DIFF, encoding="utf-8")

    def _boom(_path: Path, _text: str) -> None:
        raise PermissionError("rename refused")

    monkeypatch.setattr("charlie_work.rework_prompts._write_text_atomic", _boom)
    assert ri.archive_round_diff(tmp_path, round_dir) is None
    assert not (round_dir / "diff.patch").exists()


# ---------------------------------------------------------------------------
# The shared round-2 predicate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("decision", "terminal", "round2"),
    [
        pytest.param(None, False, False, id="absent"),
        pytest.param({}, False, False, id="empty"),
        pytest.param({"decision": "missing"}, False, False, id="missing"),
        pytest.param(
            {"decision": "pending", "reviewed_head_sha": "sha-x"}, False, False, id="pending"
        ),
        pytest.param({"decision": "request_changes"}, True, False, id="terminal-no-head"),
        pytest.param(
            {"decision": "request_changes", "reviewed_head_sha": "sha-x"}, True, True, id="rc"
        ),
        pytest.param({"decision": "approved", "reviewed_head_sha": "sha-x"}, True, True, id="ok"),
    ],
)
def test_round2_predicates(decision: Any, terminal: bool, round2: bool) -> None:
    assert ri.has_terminal_decision(decision) is terminal
    assert ri.is_round2_review(decision) is round2


# ---------------------------------------------------------------------------
# One loader feeds both the packet builder and the classifier.
# ---------------------------------------------------------------------------


def _entries(pr_dir: Path) -> list[tuple[int, dict[str, Any]]]:
    return _round_history_entries(pr_dir / "rounds")


def test_load_incremental_inputs_reports_why_a_round_is_not_incremental(
    tmp_path: Path,
) -> None:
    assert ri.load_incremental_inputs(tmp_path, []).fallback_reason == (
        ri.FALLBACK_NO_ARCHIVED_DIFF
    )

    # Legacy round: decision archived, no diff.patch beside it.
    legacy = tmp_path / "legacy"
    plant_round1(legacy, diff=None)
    (legacy / "diff.patch").write_text(ROUND2_PR_DIFF, encoding="utf-8")
    assert ri.load_incremental_inputs(legacy, _entries(legacy)).fallback_reason == (
        ri.FALLBACK_NO_ARCHIVED_DIFF
    )

    # Archived diff present, live diff.patch missing.
    no_live = tmp_path / "no_live"
    plant_round1(no_live)
    assert ri.load_incremental_inputs(no_live, _entries(no_live)).fallback_reason == (
        ri.FALLBACK_ARCHIVE_READ_FAILED
    )

    # Archived diff present but not decodable.
    undecodable = tmp_path / "undecodable"
    plant_round1(undecodable)
    (undecodable / "diff.patch").write_text(ROUND2_PR_DIFF, encoding="utf-8")
    (undecodable / "rounds" / "round-1" / "diff.patch").write_bytes(_NOT_UTF8)
    assert ri.load_incremental_inputs(undecodable, _entries(undecodable)).fallback_reason == (
        ri.FALLBACK_ARCHIVE_READ_FAILED
    )

    # Both readable: usable, carrying the two texts.
    usable = tmp_path / "usable"
    plant_round1(usable)
    (usable / "diff.patch").write_text(ROUND2_PR_DIFF, encoding="utf-8")
    loaded = ri.load_incremental_inputs(usable, _entries(usable))
    assert loaded.usable
    assert (loaded.prior_diff, loaded.new_diff) == (ROUND1_PR_DIFF, ROUND2_PR_DIFF)


def _classify(
    pr_dir: Path,
    *,
    prior_decision: dict[str, Any] | None,
    new_head: str = "sha-new",
    round2: bool = True,
    enabled: bool = True,
) -> str | None:
    return ri.classify_review_round_kind(
        pr_dir,
        prior_decision,
        new_head,
        incremental_enabled=enabled,
        is_round2_review=round2,
    )


def _round2_dir(tmp_path: Path, *, archived: str | bytes | None = ROUND1_PR_DIFF) -> Path:
    """A moved-head round-2 packet dir: round 1 archived, live diff written."""
    pr_dir = tmp_path / "pr-1"
    plant_round1(pr_dir, diff=None)
    if isinstance(archived, bytes):
        (pr_dir / "rounds" / "round-1" / "diff.patch").write_bytes(archived)
    elif archived is not None:
        (pr_dir / "rounds" / "round-1" / "diff.patch").write_text(archived, encoding="utf-8")
    (pr_dir / "diff.patch").write_text(ROUND2_PR_DIFF, encoding="utf-8")
    return pr_dir


def test_classify_kill_switch_and_first_round(tmp_path: Path) -> None:
    pr_dir = _round2_dir(tmp_path)
    prior = {"decision": "request_changes", "reviewed_head_sha": ROUND1_HEAD}

    assert _classify(pr_dir, prior_decision=prior, enabled=False) is None
    assert _classify(pr_dir, prior_decision=None, round2=False) == "first"


def test_classify_incremental_when_both_diffs_are_readable(tmp_path: Path) -> None:
    pr_dir = _round2_dir(tmp_path)
    prior = {"decision": "request_changes", "reviewed_head_sha": ROUND1_HEAD}

    assert _classify(pr_dir, prior_decision=prior) == "incremental"


@pytest.mark.parametrize(
    "archived",
    [
        pytest.param(None, id="legacy-no-archived-diff"),
        pytest.param(_NOT_UTF8, id="archived-diff-undecodable"),
    ],
)
def test_classify_fallback_when_the_archive_cannot_be_used(
    tmp_path: Path, archived: bytes | None
) -> None:
    pr_dir = _round2_dir(tmp_path, archived=archived)
    prior = {"decision": "request_changes", "reviewed_head_sha": ROUND1_HEAD}

    assert _classify(pr_dir, prior_decision=prior) == "fallback"


def test_classify_fallback_when_the_live_diff_cannot_be_read(tmp_path: Path) -> None:
    """The ``archive_read_failed`` the builder reports for an unreadable LIVE
    diff.patch must classify as fallback too -- never ``incremental``."""
    pr_dir = _round2_dir(tmp_path)
    (pr_dir / "diff.patch").unlink()
    prior = {"decision": "request_changes", "reviewed_head_sha": ROUND1_HEAD}

    assert _classify(pr_dir, prior_decision=prior) == "fallback"


def test_classify_same_head_is_fallback(tmp_path: Path) -> None:
    pr_dir = _round2_dir(tmp_path)
    prior = {"decision": "request_changes", "reviewed_head_sha": ROUND1_HEAD}

    assert _classify(pr_dir, prior_decision=prior, new_head=ROUND1_HEAD) == "fallback"


def test_classify_derives_the_prior_head_from_the_archive_not_the_flat_mirror(
    tmp_path: Path,
) -> None:
    """The builder compares the live head against the latest ARCHIVED round's
    head, so the classifier must too: a flat mirror that disagrees with the
    archive cannot make the two classify one packet differently."""
    pr_dir = _round2_dir(tmp_path)  # latest archived round: head ROUND1_HEAD

    # Flat mirror claims it already reviewed the live head; the archive did
    # not -> the head moved, so the round is incremental.
    mirror_says_same = {"decision": "request_changes", "reviewed_head_sha": "sha-new"}
    assert _classify(pr_dir, prior_decision=mirror_says_same) == "incremental"

    # Flat mirror claims an older head; the archive's latest IS the live head
    # -> same head, so there is no interdiff and the round is a fallback.
    mirror_says_moved = {"decision": "request_changes", "reviewed_head_sha": "sha-other"}
    assert _classify(pr_dir, prior_decision=mirror_says_moved, new_head=ROUND1_HEAD) == "fallback"


# ---------------------------------------------------------------------------
# Interdiff content and measurement.
# ---------------------------------------------------------------------------


def test_build_pr_interdiff_partitions_changed_unchanged_and_removed() -> None:
    gone = "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ b/gone.py\n@@ -1 +1 @@\n-x\n+y\n"

    built = ri.build_pr_interdiff(ROUND1_PR_DIFF + gone, ROUND2_PR_DIFF)

    assert built.changed == ("three.py",)
    assert built.unchanged == ("one.py", "two.py")
    assert built.removed == ("gone.py",)
    assert built.file_count == 1
    assert built.line_count == 5
    assert built.text.startswith("diff --git a/three.py b/three.py")


def test_count_interdiff_reads_the_artifact_and_is_zero_without_one(tmp_path: Path) -> None:
    assert ri.count_interdiff(tmp_path) == (0, 0)

    (tmp_path / "interdiff.patch").write_text(
        "diff --git a/three.py b/three.py\n--- /dev/null\n+++ b/three.py\n@@ -0,0 +1 @@\n+hello\n",
        encoding="utf-8",
    )
    assert ri.count_interdiff(tmp_path) == (1, 5)
