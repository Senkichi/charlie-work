"""Tests for issue #1539: attachment-point ratchet remedy text.

Mirrors the #1496/#1528 file-size ratchet remedy pattern
(``test_render_over_cap_section_instructs_refresh_script`` and
``test_rubric_text_present_in_built_packet_default`` in
``test_issue_1445_over_cap_rubric.py``): a keystone test asserts the remedy
text renders when a shrunk attachment point exists, plus an integration test
drives ``OrchestratorApp.review`` end-to-end and reads the rendered
``review-prompt.md`` packet.

The ratchet remedy tells a worker whose PR shrinks a saturated attachment
point to run ``python -m charlie_work.attachment_contracts baseline
--ratchet`` and commit the resulting ``.attachment-budgets.json`` tightening
in the same PR. A lowered count is a ratchet, not a bump -- G4 (workers may
not self-ack bumps) governs raises only; CI re-verifies ``actual <=
baseline`` deterministically, so there is nothing to launder by
self-committing a decrease.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.attachment_budget_prompt import render_attachment_budget_section
from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, dumps
from charlie_work.attachment_contracts.review_delta import (
    BudgetSection,
    RatchetablePoint,
)
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp

# ---------------------------------------------------------------------------
# Keystone test: render_attachment_budget_section emits ratchet remedy text
# when a ratchetable point (live count below baseline) is present.
# Mirrors test_render_over_cap_section_instructs_refresh_script.
# ---------------------------------------------------------------------------


def test_render_section_with_ratchetable_point_emits_remedy() -> None:
    """Issue #1539: when a point's live member count is below its baseline,
    the rendered section must include the ratchet-and-commit instruction."""
    ratchetable = (
        RatchetablePoint(
            kind="class",
            identity="Foo",
            file="src/foo.py",
            baseline_members=10,
            live_count=5,
        ),
    )
    section = BudgetSection(
        bumps=(),
        blocking_bumps=(),
        saturated_touched=(),
        redirects_not_taken=(),
        ratchetable=ratchetable,
    )
    text = render_attachment_budget_section(section)
    # The ratchet command must be named so the worker knows what to run.
    assert "python -m charlie_work.attachment_contracts baseline --ratchet" in text
    # The baseline file that the command writes must be named.
    assert ".attachment-budgets.json" in text
    # The instruction must say to commit in the same PR.
    assert "same PR" in text or "this PR" in text
    # A lowered count is a ratchet, not a bump -- the text must say so.
    assert "ratchet" in text.lower()
    assert "not a bump" in text.lower()
    # The live and baseline counts must appear so the reviewer can see the
    # tightening magnitude.
    assert "5" in text
    assert "10" in text
    # The point identity and file must appear so the row is actionable.
    assert "Foo" in text
    assert "src/foo.py" in text


def test_render_section_without_ratchetable_has_no_remedy() -> None:
    """Issue #1539: when no ratchetable point exists, the ratchet remedy text
    must NOT render -- it is conditional on a shrunk point, not permanent
    prose (unlike the file-size cap rubric which is always present)."""
    section = BudgetSection(
        bumps=(),
        blocking_bumps=(),
        saturated_touched=(),
        redirects_not_taken=(),
        ratchetable=(),
    )
    text = render_attachment_budget_section(section)
    assert "baseline --ratchet" not in text


# ---------------------------------------------------------------------------
# Integration test: a synthetic PR that shrinks a saturated attachment point
# yields the ratchet remedy text in the built review packet.
# Mirrors test_touches_baselined_host_file_yields_saturated_row.
# ---------------------------------------------------------------------------


def _doc(entries: list[dict]) -> dict:
    return {
        "version": 1,
        "generated_by": "test",
        "generated_at": "2026-08-25T00:00:00Z",
        "floor": 1,
        "entries": entries,
    }


def _entry(identity: str, file: str, member_count: int) -> dict:
    return {
        "kind": "class",
        "identity": identity,
        "file": file,
        "member_count": member_count,
        "boundary": 4.0,
        "bumps": [],
    }


def _class_source(name: str, n_methods: int) -> str:
    """Generate Python source for a class with ``n_methods`` methods.

    Method names use letter suffixes (``do_a``, ``do_b``, ...) rather than
    numeric suffixes so the ledger detector (``classify_ledger``) does not
    classify the class as a ``migration_runner`` -- the test needs a plain
    ``class`` archetype to match the baseline entry's kind.
    """
    lines = [f"class {name}:"]
    for i in range(n_methods):
        suffix = chr(ord("a") + i)
        lines.append(f"    def do_{suffix}(self):")
        lines.append(f"        return {i}")
    return "\n".join(lines) + "\n"


def _src_diff(path: str, base_text: str, head_text: str) -> str:
    """Build a real ``diff --git`` section for a source file from base/head
    text using ``difflib``."""
    diff_lines = list(
        difflib.unified_diff(
            base_text.splitlines(keepends=True),
            head_text.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    body = "".join(diff_lines)
    return f"diff --git a/{path} b/{path}\nindex 111..222 100644\n" + body


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


def test_shrunk_point_in_packet_yields_ratchet_remedy(tmp_path: Path) -> None:
    """Issue #1539: a PR that shrinks a saturated attachment point (live
    member count below baseline) yields the ratchet remedy text in the built
    review packet."""
    # Plant the baseline: Foo is saturated at 10 members in src/foo.py.
    base = _doc([_entry("Foo", "src/foo.py", 10)])
    (tmp_path / BASELINE_FILENAME).write_text(dumps(base), encoding="utf-8")

    # Plant the BASE source on disk (10 methods) -- the orchestrator's
    # checkout is at base; the diff transforms it to head.
    base_src = _class_source("Foo", 10)
    foo_path = tmp_path / "src" / "foo.py"
    foo_path.parent.mkdir(parents=True, exist_ok=True)
    foo_path.write_text(base_src, encoding="utf-8")

    # The diff shrinks Foo from 10 to 5 methods.
    head_src = _class_source("Foo", 5)
    diff = _src_diff("src/foo.py", base_src, head_src)

    packet = _build_packet(tmp_path, diff)

    assert "## Attachment-budget diff" in packet
    # The render row's ratchet command (not the review.md rubric, which also
    # names the command) -- "live member count" is unique to the render row.
    assert "live member count" in packet
    assert "is below baseline" in packet
    assert "not a bump" in packet.lower()
    assert "python -m charlie_work.attachment_contracts baseline --ratchet" in packet
    assert ".attachment-budgets.json" in packet
    assert "$attachment_budget_section" not in packet


def test_review_md_states_ratchet_not_tamper() -> None:
    """Issue #1539: review.md must state that a lowered attachment-point
    count in a PR is a ratchet, not tamper, and must not be flagged."""
    from charlie_work.prompts import TEMPLATE_DIR

    review_md = (TEMPLATE_DIR / "review.md").read_text(encoding="utf-8")
    assert "ratchet" in review_md.lower()
    assert "tamper" in review_md.lower()
    # The text must say a lowered count should not be blocked or flagged.
    assert "not" in review_md.lower() and "flag" in review_md.lower()
