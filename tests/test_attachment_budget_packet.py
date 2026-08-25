"""Tests for issue #1460: attachment-budget review-packet section.

Pattern mirrors ``tests/test_issue_1445_over_cap_rubric.py``: FakeGitHub with
a synthetic diff, ``app.review(N)``, read the rendered ``review-prompt.md``
packet.
"""

from __future__ import annotations

import difflib
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
