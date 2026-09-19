"""Targeting-surface tests for scripts/worker_stop_gate.py.

Split verbatim out of ``tests/test_worker_stop_gate.py`` (issue #1573,
Track 1 shoulder). Shared helpers and the ``gate``/``repo`` fixtures live
in ``tests/_worker_stop_gate_fixtures.py`` -- the ``tests/_*.py``
hoisted-fixture convention is the sanctioned import target for shared
test helpers (see ``tests/test_zero_cross_test_import_guard.py``).

Covers the W4/#1262 emit-site targeting rule and the
``git status --porcelain`` changed-file surface that feeds it.
"""

from __future__ import annotations

import pytest

from _worker_stop_gate_fixtures import _run_git, gate as gate, repo as repo


# ---------------------------------------------------------------------------
# W4/#1262 targeting rule.
# ---------------------------------------------------------------------------


def test_w4_targeting_rule_positive_emit_site_pulls_instrumentation_test(gate, repo):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "new_thing.py").write_text(
        "def do_it():\n    log_event(state_path, 'thing_happened', {})\n", encoding="utf-8"
    )

    changed = gate._changed_files(repo)
    targets = gate._targeted_tests(repo, changed)

    assert gate.INSTRUMENTATION_TEST_PATH in targets


def test_w4_targeting_rule_negative_no_emit_site_no_instrumentation_test(gate, repo):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "new_thing.py").write_text("def do_it():\n    return 1\n", encoding="utf-8")

    changed = gate._changed_files(repo)
    targets = gate._targeted_tests(repo, changed)

    assert gate.INSTRUMENTATION_TEST_PATH not in targets
    assert targets == ()


def test_targeted_tests_includes_changed_test_files_without_w4_trigger(gate, repo):
    (repo / "tests").mkdir()
    (repo / "tests" / "test_something.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8"
    )

    changed = gate._changed_files(repo)
    targets = gate._targeted_tests(repo, changed)

    assert targets == ("tests/test_something.py",)


@pytest.mark.parametrize(
    "call_form",
    [
        "log_event(state_path, 'k', {})",
        "append_event(path, kind='k')",
        "self._record_event('k', {})",
    ],
)
def test_touches_emit_site_matches_all_three_call_forms(gate, repo, call_form):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    (src_dir / "site.py").write_text(f"def f():\n    {call_form}\n", encoding="utf-8")

    changed = gate._changed_files(repo)

    assert gate._touches_emit_site(repo, changed) is True


def test_touches_emit_site_ignores_deleted_files(gate, repo):
    src_dir = repo / "src" / "charlie_work"
    src_dir.mkdir(parents=True)
    target = src_dir / "gone.py"
    target.write_text("log_event(x, 'k', {})\n", encoding="utf-8")
    _run_git(["add", "-A"], cwd=repo)
    _run_git(["commit", "-m", "add gone"], cwd=repo)
    target.unlink()

    changed = gate._changed_files(repo)

    assert gate._touches_emit_site(repo, changed) is False


def test_touches_emit_site_fails_closed_when_file_unreadable(gate, repo):
    changed = (gate.ChangedFile(path="src/does_not_exist_on_disk.py", deleted=False),)

    assert gate._touches_emit_site(repo, changed) is True


# ---------------------------------------------------------------------------
# git status --porcelain parsing.
# ---------------------------------------------------------------------------


def test_changed_files_parses_deleted_and_untracked_entries(gate, repo):
    tracked = repo / "keep.py"
    tracked.write_text("x = 1\n", encoding="utf-8")
    _run_git(["add", "keep.py"], cwd=repo)
    _run_git(["commit", "-m", "add keep"], cwd=repo)

    tracked.unlink()
    (repo / "new_untracked.py").write_text("y = 2\n", encoding="utf-8")

    changed = gate._changed_files(repo)
    by_path = {cf.path: cf for cf in changed}

    assert by_path["keep.py"].deleted is True
    assert by_path["keep.py"].untracked is False  # tracked file, now deleted
    assert by_path["new_untracked.py"].deleted is False
    assert by_path["new_untracked.py"].untracked is True  # ?? entry


def test_changed_files_reports_non_ascii_filename_unquoted(gate, repo):
    # Review round, #1259 follow-up 2: without -c core.quotePath=false, git
    # C-quotes a non-ASCII byte into an octal-escaped, double-quoted string
    # (e.g. "caf\303\251.py"), which does not .endswith(".py") and silently
    # drops the file from every rule downstream.
    (repo / "café.py").write_text("x = 1\n", encoding="utf-8")

    changed = gate._changed_files(repo)

    assert any(cf.path == "café.py" and not cf.deleted for cf in changed)
