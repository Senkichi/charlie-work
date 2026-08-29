"""Tests for issue #1460: attachment-budget review-packet section.

Pattern mirrors ``tests/test_issue_1445_over_cap_rubric.py``: FakeGitHub with
a synthetic diff, ``app.review(N)``, read the rendered ``review-prompt.md``
packet.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, dumps
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def _doc(entries: list[dict]) -> dict:
    return {
        "version": 1,
        "generated_by": "test",
        "generated_at": "2026-08-25T00:00:00Z",
        "floor": 1,
        "entries": entries,
    }


def _entry(identity: str, file: str, member_count: int, bumps: list[dict] | None = None) -> dict:
    return {
        "kind": "class",
        "identity": identity,
        "file": file,
        "member_count": member_count,
        "boundary": 4.0,
        "bumps": bumps or [],
    }


def _bump(to: int, actor: str, ack: str, reason: str = "growth") -> dict:
    return {"to": to, "reason": reason, "actor": actor, "ack": ack}


def _git_diff_for_baseline(base_text: str, head_text: str) -> str:
    """Build a real ``diff --git`` section for `.attachment-budgets.json`
    from base/head text using ``difflib``, so the diff is guaranteed
    consistent with what ``reconstruct_baseline_head_text`` can re-derive."""
    diff_lines = list(
        difflib.unified_diff(
            base_text.splitlines(keepends=True),
            head_text.splitlines(keepends=True),
            fromfile=f"a/{BASELINE_FILENAME}",
            tofile=f"b/{BASELINE_FILENAME}",
            n=3,
        )
    )
    body = "".join(diff_lines)
    return (
        f"diff --git a/{BASELINE_FILENAME} b/{BASELINE_FILENAME}\nindex 111..222 100644\n" + body
    )


def _src_file_diff(path: str, extra_content: str = "line 200\n") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        "index 111..222 100644\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1,1 +1,2 @@\n"
        " existing line\n"
        f"+{extra_content}"
    )


def _build_packet(tmp_path: Path, diff: str) -> str:
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = diff
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)
    assert result.ok is True, result.message
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    return packet.read_text(encoding="utf-8")


def _build_packet_with_comments(tmp_path: Path, diff: str, comments: list[dict]) -> str:
    """Like ``_build_packet`` but seeds PR issue-level comments (issue #1466)."""
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = diff
    fake_gh.pr_external_issue_comments[456] = comments
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)
    assert result.ok is True, result.message
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    return packet.read_text(encoding="utf-8")


def _advisories_comment(records: list[dict]) -> dict:
    """Build a PR comment dict whose body is the worker-published advisories
    marker + fenced JSON array (issue #1466)."""
    from charlie_work.attachment_contracts.hook_entry import ADVISORY_COMMENT_MARKER

    body = ADVISORY_COMMENT_MARKER + "\n```json\n" + json.dumps(records) + "\n```\n"
    return {"body": body}


def test_no_baselined_touch_section_absent(tmp_path: Path) -> None:
    """No `.attachment-budgets.json` at all -> cheap gate fires immediately,
    section renders empty."""
    diff = _src_file_diff("src/unrelated.py")
    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" not in packet
    assert "$attachment_budget_section" not in packet


def test_touches_baselined_host_file_yields_saturated_row(tmp_path: Path) -> None:
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    diff = _src_file_diff("src/foo.py")
    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" in packet
    assert "Foo" in packet
    assert "frozen at 10 members" in packet
    assert "$attachment_budget_section" not in packet


def test_worker_bump_with_source_id_ack_is_blocking(tmp_path: Path) -> None:
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    head = _doc([_entry("Foo", "src/foo.py", 10, [_bump(12, "worker", "dispatch:abc123")])])
    base_text = dumps(base)
    (tmp_path / BASELINE_FILENAME).write_text(base_text, encoding="utf-8")

    diff = _git_diff_for_baseline(base_text, dumps(head))
    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" in packet
    assert "BLOCKING" in packet
    assert "worker may not justify its own bump" in packet.lower()
    assert "$attachment_budget_section" not in packet


def test_worker_bump_with_issue_ack_no_blocking_and_suppressed(tmp_path: Path) -> None:
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    head = _doc([_entry("Foo", "src/foo.py", 10, [_bump(12, "worker", "#123")])])
    base_text = dumps(base)
    (tmp_path / BASELINE_FILENAME).write_text(base_text, encoding="utf-8")

    diff = _git_diff_for_baseline(base_text, dumps(head))
    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" in packet
    assert "BLOCKING" not in packet
    # item-1 (saturated-touched) suppressed for Foo since its new bump is
    # G4-valid (externally acked) -- the row would be redundant noise.
    assert "frozen at 10 members" not in packet
    assert "$attachment_budget_section" not in packet


def test_head_reconstruction_failure_yields_could_not_evaluate_note(tmp_path: Path) -> None:
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    base_text = dumps(base)
    (tmp_path / BASELINE_FILENAME).write_text(base_text, encoding="utf-8")

    # A hunk whose context does not match base_text at all -> reconstruction
    # fails structurally.
    bogus_diff = (
        f"diff --git a/{BASELINE_FILENAME} b/{BASELINE_FILENAME}\n"
        "index 111..222 100644\n"
        f"--- a/{BASELINE_FILENAME}\n"
        f"+++ b/{BASELINE_FILENAME}\n"
        "@@ -1,3 +1,3 @@\n"
        " THIS CONTEXT DOES NOT MATCH ANYTHING\n"
        "-old\n"
        "+new\n"
        " more-mismatched-context\n"
    )
    packet = _build_packet(tmp_path, bogus_diff)

    assert "## Attachment-budget diff" in packet
    assert "could not evaluate" in packet
    assert "$attachment_budget_section" not in packet


# ---------------------------------------------------------------------------
# Issue #1466: worker-published advisories PR-comment channel.
# ---------------------------------------------------------------------------


def test_pr_comment_advisories_yield_redirects_not_taken(tmp_path: Path) -> None:
    """A worker-published advisories PR comment with a redirect the diff does
    NOT touch surfaces as a "redirect not taken" row, sourced from the PR-
    comment channel (no local advisories log present)."""
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    diff = _src_file_diff("src/foo.py")
    comment = _advisories_comment(
        [
            {
                "severity": "block",
                "file": "src/foo.py",
                "identity": "Foo",
                "message": "Foo is saturated",
                "redirect": "src/foo_extra.py",
                "timestamp": "2026-08-25T00:00:00+00:00",
            }
        ]
    )
    packet = _build_packet_with_comments(tmp_path, diff, [comment])

    assert "## Attachment-budget diff" in packet
    assert "redirect not taken" in packet
    assert "src/foo_extra.py" in packet
    # The PR-comment channel is present, so the "log not available" NOTE
    # must NOT render even though no local advisories log exists.
    assert "advisories log not available" not in packet
    assert "$attachment_budget_section" not in packet


def test_pr_comment_present_empty_suppresses_log_not_available_note(
    tmp_path: Path,
) -> None:
    """A present marker comment with an empty JSON array is a present channel
    (clean pass) -- the "log not available" NOTE must NOT render, and the
    builder must NOT fall back to the (absent) local log."""
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    diff = _src_file_diff("src/foo.py")
    comment = _advisories_comment([])
    packet = _build_packet_with_comments(tmp_path, diff, [comment])

    assert "## Attachment-budget diff" in packet
    assert "advisories log not available" not in packet
    assert "redirect not taken" not in packet
    assert "$attachment_budget_section" not in packet


def test_no_pr_comment_no_local_log_yields_log_not_available_note(
    tmp_path: Path,
) -> None:
    """Neither channel present -> the "log not available" NOTE renders (the
    pre-#1466 vacuous case, now the fallback of last resort)."""
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    diff = _src_file_diff("src/foo.py")
    # No PR comments seeded, no local advisories log.
    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" in packet
    assert "advisories log not available" in packet
    assert "$attachment_budget_section" not in packet


def test_no_pr_comment_falls_back_to_local_log(tmp_path: Path) -> None:
    """No marker PR comment -> the builder falls back to the local advisories
    log under ``repo_root`` (the pre-#1466 channel, still supported)."""
    from charlie_work.attachment_contracts.hook_entry import _ADVISORY_LOG_REL

    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    # Plant a local advisories log with one redirect-not-taken record.
    log_path = tmp_path / _ADVISORY_LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "severity": "block",
                "file": "src/foo.py",
                "identity": "Foo",
                "message": "Foo is saturated",
                "redirect": "src/foo_extra.py",
                "timestamp": "2026-08-25T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    diff = _src_file_diff("src/foo.py")
    # No PR comments seeded -> local log fallback.
    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" in packet
    assert "redirect not taken" in packet
    assert "src/foo_extra.py" in packet
    assert "advisories log not available" not in packet
    assert "$attachment_budget_section" not in packet


def test_pr_comment_wins_over_local_log(tmp_path: Path) -> None:
    """When BOTH channels are present, the PR-comment channel wins. The local
    log's record (redirect to ``src/local_only.py``) must NOT surface; only
    the PR comment's record (redirect to ``src/comment_only.py``) does."""
    from charlie_work.attachment_contracts.hook_entry import _ADVISORY_LOG_REL

    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    # Local log with a distinct redirect.
    log_path = tmp_path / _ADVISORY_LOG_REL
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "severity": "block",
                "file": "src/foo.py",
                "identity": "Foo",
                "message": "local log record",
                "redirect": "src/local_only.py",
                "timestamp": "2026-08-25T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    diff = _src_file_diff("src/foo.py")
    comment = _advisories_comment(
        [
            {
                "severity": "block",
                "file": "src/foo.py",
                "identity": "Foo",
                "message": "comment record",
                "redirect": "src/comment_only.py",
                "timestamp": "2026-08-25T00:00:00+00:00",
            }
        ]
    )
    packet = _build_packet_with_comments(tmp_path, diff, [comment])

    assert "## Attachment-budget diff" in packet
    assert "src/comment_only.py" in packet
    assert "src/local_only.py" not in packet
    assert "$attachment_budget_section" not in packet


def test_non_marker_comment_does_not_count_as_channel(tmp_path: Path) -> None:
    """A PR comment that does NOT start with the marker is not an advisories
    comment -- the builder falls back to the local-log / not-available path,
    never parsing an unrelated comment as advisories."""
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    diff = _src_file_diff("src/foo.py")
    # A regular review comment, no marker.
    packet = _build_packet_with_comments(tmp_path, diff, [{"body": "looks good to me"}])

    assert "## Attachment-budget diff" in packet
    # No channel present -> "log not available" NOTE.
    assert "advisories log not available" in packet
    assert "$attachment_budget_section" not in packet


def test_most_recent_marker_comment_wins(tmp_path: Path) -> None:
    """When multiple marker comments exist (worker re-posted on a later
    push), the most recent one's records surface -- the builder scans in
    chronological order and keeps the last match."""
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    diff = _src_file_diff("src/foo.py")
    old_comment = _advisories_comment(
        [
            {
                "severity": "block",
                "file": "src/foo.py",
                "identity": "Foo",
                "message": "stale",
                "redirect": "src/stale.py",
                "timestamp": "2026-08-25T00:00:00+00:00",
            }
        ]
    )
    new_comment = _advisories_comment(
        [
            {
                "severity": "block",
                "file": "src/foo.py",
                "identity": "Foo",
                "message": "fresh",
                "redirect": "src/fresh.py",
                "timestamp": "2026-08-25T01:00:00+00:00",
            }
        ]
    )
    packet = _build_packet_with_comments(tmp_path, diff, [old_comment, new_comment])

    assert "## Attachment-budget diff" in packet
    assert "src/fresh.py" in packet
    assert "src/stale.py" not in packet
    assert "$attachment_budget_section" not in packet
