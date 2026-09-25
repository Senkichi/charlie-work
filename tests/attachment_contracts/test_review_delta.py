"""Tests for issue #1460: review_delta.py (reconstruct, new_bumps, suppression).

No AST, no scan_tree, no line-count arithmetic is exercised or asserted here
by design -- this module's whole point is proving the review section reasons
about *member-bearing* bumps/entries, never raw diff line volume (the
behavioral no-LOC corollary test at the bottom).
"""

from __future__ import annotations

from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME
from charlie_work.attachment_contracts.model import AdvisoryRecord
from charlie_work.attachment_contracts.review_delta import (
    build_budget_findings,
    reconstruct_baseline_dir_head,
    reconstruct_baseline_head_text,
)
from charlie_work.janitor import iter_diff_files

# ---------------------------------------------------------------------------
# reconstruct_baseline_head_text
# ---------------------------------------------------------------------------

_BASE_TEXT = "line1\nline2\nline3\nline4\nline5\n"


def _hunks_for(diff_text: str, filename: str) -> str:
    for name, _is_new, hunks in iter_diff_files(diff_text):
        if name == filename:
            return "\n".join(hunks)
    raise AssertionError(f"{filename} not found in diff")


def test_reconstruct_clean_apply() -> None:
    diff = (
        "diff --git a/.attachment-budgets.json b/.attachment-budgets.json\n"
        "index 111..222 100644\n"
        "--- a/.attachment-budgets.json\n"
        "+++ b/.attachment-budgets.json\n"
        "@@ -2,3 +2,4 @@\n"
        " line2\n"
        "-line3\n"
        "+line3-changed\n"
        "+line3b\n"
        " line4\n"
    )
    file_diff = _hunks_for(diff, BASELINE_FILENAME)

    result = reconstruct_baseline_head_text(_BASE_TEXT, file_diff)

    assert result == "line1\nline2\nline3-changed\nline3b\nline4\nline5\n"


def test_reconstruct_newly_added_file_returns_added_content() -> None:
    diff = (
        "diff --git a/.attachment-budgets.json b/.attachment-budgets.json\n"
        "new file mode 100644\n"
        "index 0000000..1234567\n"
        "--- /dev/null\n"
        "+++ b/.attachment-budgets.json\n"
        "@@ -0,0 +1,3 @@\n"
        "+a\n"
        "+b\n"
        "+c\n"
    )
    file_diff = _hunks_for(diff, BASELINE_FILENAME)

    result = reconstruct_baseline_head_text(None, file_diff)

    assert result == "a\nb\nc\n"


def test_reconstruct_context_mismatch_returns_none() -> None:
    """A hunk claiming context that doesn't match base_text -> None, never a
    silently-wrong reconstruction."""
    bogus_file_diff = "@@ -2,3 +2,4 @@\n THIS-IS-NOT-LINE2\n-line3\n+line3-changed\n line4\n"

    result = reconstruct_baseline_head_text(_BASE_TEXT, bogus_file_diff)

    assert result is None


def test_reconstruct_deletion_only_hunk() -> None:
    diff = (
        "diff --git a/.attachment-budgets.json b/.attachment-budgets.json\n"
        "index 111..222 100644\n"
        "--- a/.attachment-budgets.json\n"
        "+++ b/.attachment-budgets.json\n"
        "@@ -1,5 +1,4 @@\n"
        " line1\n"
        " line2\n"
        "-line3\n"
        " line4\n"
        " line5\n"
    )
    file_diff = _hunks_for(diff, BASELINE_FILENAME)

    result = reconstruct_baseline_head_text(_BASE_TEXT, file_diff)

    assert result == "line1\nline2\nline4\nline5\n"


# ---------------------------------------------------------------------------
# build_budget_findings: new_bumps + suppression logic
# ---------------------------------------------------------------------------


def _entry_dict(
    identity: str, file: str, member_count: int, bumps: list[dict] | None = None
) -> dict:
    return {
        "kind": "class",
        "identity": identity,
        "file": file,
        "member_count": member_count,
        "boundary": 4.0,
        "bumps": bumps or [],
    }


def _doc(entries: list[dict]) -> dict:
    return {
        "version": 1,
        "generated_by": "test",
        "generated_at": "2026-08-25T00:00:00Z",
        "floor": 1,
        "entries": entries,
    }


def _bump_dict(to: int, actor: str, ack: str, reason: str = "growth") -> dict:
    return {"to": to, "reason": reason, "actor": actor, "ack": ack}


def test_build_budget_findings_no_bump_or_touch_is_empty() -> None:
    section = build_budget_findings(
        base_document=None,
        head_document=None,
        changed_files=frozenset({"src/other.py"}),
        baseline_touched=False,
        advisories=(),
    )
    assert section.bumps == ()
    assert section.blocking_bumps == ()
    assert section.saturated_touched == ()


def test_worker_bump_with_source_id_ack_is_blocking() -> None:
    base = _doc([_entry_dict("Foo", "src/foo.py", 10)])
    head = _doc(
        [_entry_dict("Foo", "src/foo.py", 10, [_bump_dict(12, "worker", "dispatch:abc123")])]
    )
    section = build_budget_findings(
        base_document=base,
        head_document=head,
        changed_files=frozenset(
            {".attachment-budgets/entries/src/foo.py/class--Foo.json", "src/foo.py"}
        ),
        baseline_touched=True,
        advisories=(),
    )
    assert len(section.bumps) == 1
    assert len(section.blocking_bumps) == 1
    blocking_entry, blocking_bump = section.blocking_bumps[0]
    assert blocking_entry.identity == "Foo"
    assert blocking_bump.ack == "dispatch:abc123"
    # A worker-authored, non-external-ack bump does NOT suppress the
    # saturated-touched row -- it's still frozen at its old ceiling as far as
    # G4 is concerned.
    assert any(e.identity == "Foo" for e in section.saturated_touched)


def test_worker_bump_with_issue_ack_suppresses_saturated_row_and_no_blocking() -> None:
    base = _doc([_entry_dict("Foo", "src/foo.py", 10)])
    head = _doc([_entry_dict("Foo", "src/foo.py", 10, [_bump_dict(12, "worker", "#1460")])])

    section = build_budget_findings(
        base_document=base,
        head_document=head,
        changed_files=frozenset(
            {".attachment-budgets/entries/src/foo.py/class--Foo.json", "src/foo.py"}
        ),
        baseline_touched=True,
        advisories=(),
    )
    assert len(section.bumps) == 1
    assert section.blocking_bumps == ()
    assert not any(e.identity == "Foo" for e in section.saturated_touched)


def test_saturated_touched_without_any_bump() -> None:
    base = _doc([_entry_dict("Foo", "src/foo.py", 10)])
    head = _doc([_entry_dict("Foo", "src/foo.py", 10)])  # unchanged

    section = build_budget_findings(
        base_document=base,
        head_document=head,
        changed_files=frozenset({"src/foo.py"}),
        baseline_touched=False,
        advisories=(),
    )
    assert section.bumps == ()
    assert section.blocking_bumps == ()
    assert len(section.saturated_touched) == 1
    assert section.saturated_touched[0].identity == "Foo"


def test_advisories_none_sets_unavailable_and_no_redirects() -> None:
    section = build_budget_findings(
        base_document=None,
        head_document=None,
        changed_files=frozenset(),
        baseline_touched=False,
        advisories=None,
    )
    assert section.advisories_unavailable is True
    assert section.redirects_not_taken == ()


def test_redirects_not_taken_filters_to_untaken_redirects() -> None:
    advisories = (
        AdvisoryRecord(
            severity="block",
            file="src/foo.py",
            identity="Foo",
            message="saturated",
            redirect="src/foo_extra.py",
        ),
        AdvisoryRecord(
            severity="block",
            file="src/bar.py",
            identity="Bar",
            message="saturated",
            redirect="src/bar_extra.py",
        ),
    )
    section = build_budget_findings(
        base_document=None,
        head_document=None,
        changed_files=frozenset({"src/bar_extra.py"}),  # only Bar's redirect taken
        baseline_touched=False,
        advisories=advisories,
    )
    assert len(section.redirects_not_taken) == 1
    assert section.redirects_not_taken[0].identity == "Foo"


# ---------------------------------------------------------------------------
# Behavioral no-LOC corollary: a diff adding many non-member lines (comments/
# docstrings) to a saturated host file with no bump -> no BLOCKING row. This
# is what "no line-count logic anywhere in this codepath" reduces to
# behaviorally: whether the diff added 2 lines or 2000 lines of comments is
# irrelevant to blocking_bumps, which only ever reasons about bumps.
# ---------------------------------------------------------------------------


def test_large_comment_only_addition_to_saturated_file_yields_no_blocking() -> None:
    base = _doc([_entry_dict("Foo", "src/foo.py", 10)])
    head = _doc([_entry_dict("Foo", "src/foo.py", 10)])  # no bump added

    # changed_files simulates a diff that added hundreds of comment lines to
    # src/foo.py -- irrelevant to this function, since it never reads line
    # counts, only the changed_files set and baseline documents.
    changed_files = frozenset({"src/foo.py"})

    section = build_budget_findings(
        base_document=base,
        head_document=head,
        changed_files=changed_files,
        baseline_touched=False,
        advisories=(),
    )
    assert section.blocking_bumps == ()


# ---------------------------------------------------------------------------
# reconstruct_baseline_dir_head (issue #1839): per-file map for the
# .attachment-budgets/ directory layout -- models entry adds, deletes, and
# renames, which the single-file reconstructor never had to.
# ---------------------------------------------------------------------------

_DIR = ".attachment-budgets"
_META = '{\n "floor": 4\n}\n'
_ENTRY_A = '{\n "kind": "class",\n "identity": "A",\n "file": "src/a.py"\n}\n'


def _base_files() -> dict[str, str]:
    return {
        "meta.json": _META,
        "entries/src/a.py/class--A.json": _ENTRY_A,
    }


def test_dir_head_passthrough_when_diff_touches_no_baseline_files() -> None:
    diff = (
        "diff --git a/src/other.py b/src/other.py\n"
        "index 111..222 100644\n"
        "--- a/src/other.py\n"
        "+++ b/src/other.py\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )

    assert reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR) == _base_files()


def test_dir_head_applies_entry_file_edit() -> None:
    new_entry = (
        '{\n "kind": "class",\n "identity": "A",\n "file": "src/a.py",\n "member_count": 9\n}\n'
    )
    diff = (
        "diff --git a/.attachment-budgets/entries/src/a.py/class--A.json "
        "b/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "index 111..222 100644\n"
        "--- a/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "+++ b/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "@@ -1,4 +1,5 @@\n"
        " {\n"
        '  "kind": "class",\n'
        '  "identity": "A",\n'
        '- "file": "src/a.py"\n'
        '+ "file": "src/a.py",\n'
        '+ "member_count": 9\n'
        " }\n"
    )

    result = reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR)

    assert result == {
        "meta.json": _META,
        "entries/src/a.py/class--A.json": new_entry,
    }


def test_dir_head_models_new_entry_file() -> None:
    diff = (
        "diff --git a/.attachment-budgets/entries/src/b.py/class--B.json "
        "b/.attachment-budgets/entries/src/b.py/class--B.json\n"
        "new file mode 100644\n"
        "index 0000000..1234567\n"
        "--- /dev/null\n"
        "+++ b/.attachment-budgets/entries/src/b.py/class--B.json\n"
        "@@ -0,0 +1,2 @@\n"
        "+{\n"
        "+}\n"
    )

    result = reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR)

    assert result is not None
    assert result["entries/src/b.py/class--B.json"] == "{\n}\n"
    assert result["entries/src/a.py/class--A.json"] == _ENTRY_A


def test_dir_head_models_deleted_entry_file() -> None:
    diff = (
        "diff --git a/.attachment-budgets/entries/src/a.py/class--A.json "
        "b/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "deleted file mode 100644\n"
        "index 1234567..0000000\n"
        "--- a/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "+++ /dev/null\n"
        "@@ -1,4 +0,0 @@\n"
        "-{\n"
        '- "kind": "class",\n'
        '- "identity": "A",\n'
        '- "file": "src/a.py"\n'
        "-}\n"
    )

    result = reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR)

    assert result == {"meta.json": _META}


def test_dir_head_models_entry_rename() -> None:
    """A host-file rename re-keys the entry (the filename derives from its
    content key) -- a pure rename section with no hunks moves the file."""
    diff = (
        "diff --git a/.attachment-budgets/entries/src/a.py/class--A.json "
        "b/.attachment-budgets/entries/src/b.py/class--B.json\n"
        "similarity index 100%\n"
        "rename from .attachment-budgets/entries/src/a.py/class--A.json\n"
        "rename to .attachment-budgets/entries/src/b.py/class--B.json\n"
    )

    result = reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR)

    assert result == {
        "meta.json": _META,
        "entries/src/b.py/class--B.json": _ENTRY_A,
    }


def test_dir_head_mismatched_hunk_returns_none() -> None:
    """Context that doesn't match the base file -> None, never a silently
    wrong reconstruction (same contract as the single-file variant)."""
    diff = (
        "diff --git a/.attachment-budgets/entries/src/a.py/class--A.json "
        "b/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "--- a/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "+++ b/.attachment-budgets/entries/src/a.py/class--A.json\n"
        "@@ -1,4 +1,4 @@\n"
        " NOT-THE-REAL-CONTENT\n"
        '- "kind": "class",\n'
        '+ "kind": "forged",\n'
        '  "identity": "A",\n'
        '  "file": "src/a.py"\n'
        " }\n"
    )

    assert reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR) is None


def test_dir_head_modified_member_absent_from_base_returns_none() -> None:
    """A diff section modifying a baseline member the base snapshot doesn't
    contain is a stale/mismatched snapshot -- fail closed."""
    diff = (
        "diff --git a/.attachment-budgets/entries/src/ghost.py/class--G.json "
        "b/.attachment-budgets/entries/src/ghost.py/class--G.json\n"
        "--- a/.attachment-budgets/entries/src/ghost.py/class--G.json\n"
        "+++ b/.attachment-budgets/entries/src/ghost.py/class--G.json\n"
        "@@ -1 +1 @@\n"
        "-x\n"
        "+y\n"
    )

    assert reconstruct_baseline_dir_head(_base_files(), diff, dir_prefix=_DIR) is None


def test_dir_head_none_base_map_returns_none() -> None:
    assert reconstruct_baseline_dir_head(None, "diff --git a/x b/x\n", dir_prefix=_DIR) is None
