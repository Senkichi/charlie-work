"""Wiring coverage for the issue #2281 ``pr_commits`` channel into run_janitor.

The exemption-claim escape's semantics are covered at the ``run_janitor``
level by ``tests/test_janitor_no_op_rework_exempt_escape.py``; this file
pins the three production call sites that feed it --
``workflow.review()``'s normal and escalated-PR paths and
``orchestration/state_mechanical._deescalate_mechanical_issue`` -- plus the
``no_op_escape_needs_pr_commits`` fetch gate in front of each. The kwarg
defaults to the fail-closed ``None``, so a dropped ``pr_commits=`` or a
dropped conditional fetch would silently kill the fix while every
gate-level test stayed green -- exactly the silent-regression shape the
PR #2297 review finding flagged. Each positive test asserts both ends of
the wire: ``gh.pr_commits`` was consulted AND ``run_janitor`` received the
fetched list, with the spy wrapping the real ``run_janitor`` so the
behavioral outcome is verified end-to-end too.

Helpers are inlined per the #1284 self-containment rule enforced by
``tests/test_zero_cross_test_import_guard.py``; ``_fakes_github`` /
``_review_fixtures`` are the sanctioned shared-fixture modules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import charlie_work.workflow as workflow_mod
from _fakes_github import FakeGitHub
from _review_fixtures import _test_adequacy_app

from charlie_work.config import (
    OrchestratorConfig,
    PostMortemConfig,
    TestAdequacyConfig,
)
from charlie_work.janitor import _calculate_patch_id
from charlie_work.paths import runtime_paths
from charlie_work.state import PASSIVE_OPEN_STATUS, load_state, save_state
from charlie_work.workflow import OrchestratorApp


_NO_OP_DIFF = """\
diff --git a/test.txt b/test.txt
index 1234567..abcdef0 100644
--- a/test.txt
+++ b/test.txt
@@ -1,2 +1,2 @@
 line 1
-line 2
+line 2 modified
"""

# FakeGitHub's seeded PR #456 body -- the body baseline must hash exactly
# this text or the #1939 body escape (which runs before the trailer escape)
# would satisfy the gate first and the test would pass without the #2281
# channel ever being consulted.
_FAKE_PR_BODY = "Closes #123\n\nTests: regression coverage added."


def _body_sha256(body: str | None) -> str:
    """The ``reviewed_body_sha256`` wire contract (issue #1939): SHA-256 of
    the body with CRLF/CR normalized to LF, ``None`` as the empty string."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _commit(sha: str, message: str) -> dict:
    """One ``pulls/{n}/commits`` REST entry: top-level ``sha`` + ``commit.message``."""
    return {"sha": sha, "commit": {"message": message}}


def _seed_no_op_verdict(app: OrchestratorApp, pr_number: int = 456) -> None:
    """Pin a request_changes verdict whose baselines match the LIVE
    ``_NO_OP_DIFF`` and PR body -- the exact pre-#2281 no-op shape (patch-id
    unchanged, body baseline unchanged, reviewed head ``aaa111`` behind the
    live head) that only the exemption-claim escape can clear."""
    baseline = {
        "decision": "request_changes",
        "reviewed_head_sha": "aaa111",
        "reviewed_patch_id": _calculate_patch_id(_NO_OP_DIFF),
        "reviewed_body_sha256": _body_sha256(_FAKE_PR_BODY),
    }
    state = load_state(app.paths.state_file)
    state["prs"][str(pr_number)] = {
        "number": pr_number,
        "issue_number": 123,
        **baseline,
    }
    save_state(app.paths.state_file, state)
    pr_dir = app.paths.prs / f"pr-{pr_number}"
    pr_dir.mkdir(parents=True, exist_ok=True)
    # Issue #1362 Stage 1: the decision gate reads the file-first reader, so
    # the verdict must exist on disk, not only in state.json.
    (pr_dir / "review-decision.json").write_text(
        json.dumps(baseline),
        encoding="utf-8",
    )


def _exempt_trailer_commits() -> list[dict[str, Any]]:
    """Reviewed head ``aaa111`` plus a post-verdict empty commit carrying a
    valid ``Test-exempt:`` trailer -- the trailer-only rework shape."""
    return [
        _commit("aaa111", "feat: add feature without tests"),
        _commit(
            "ccc333",
            "chore: claim test-adequacy exemption\n\n"
            "Test-exempt: pure refactor, no behavior change",
        ),
    ]


def _spy_run_janitor(monkeypatch, captured: dict[str, Any]) -> None:
    """Wrap ``charlie_work.workflow.run_janitor`` in a kwarg-capturing spy.

    Both ``review()`` paths resolve ``run_janitor`` as a workflow module
    global and ``state_mechanical`` reaches it through ``_wf.`` (the same
    module object), so one patch point observes all three call sites. The
    spy delegates to the real implementation so the test still asserts the
    end-to-end outcome, not only the wiring.
    """
    real_run_janitor = workflow_mod.run_janitor

    def _spy(*args: Any, **kwargs: Any) -> Any:
        captured["call_count"] = captured.get("call_count", 0) + 1
        captured["pr_commits"] = kwargs.get("pr_commits")
        return real_run_janitor(*args, **kwargs)

    monkeypatch.setattr(workflow_mod, "run_janitor", _spy)


def _counting_pr_commits(gh: FakeGitHub) -> list[int]:
    """Replace ``gh.pr_commits`` with a call-counting wrapper."""
    calls: list[int] = []
    real_pr_commits = gh.pr_commits

    def _spy(number: int) -> Any:
        calls.append(number)
        return real_pr_commits(number)

    gh.pr_commits = _spy
    return calls


def test_review_normal_path_wires_pr_commits_into_no_op_escape(
    tmp_path: Path, monkeypatch
) -> None:
    """Normal review() path: request_changes verdict + adequacy gate on +
    trailer-only rework -> gh.pr_commits consulted, run_janitor receives the
    list, and the gate is satisfied by the exemption claim instead of
    escalating the PR as a no-op."""
    app = _test_adequacy_app(tmp_path, enabled=True)
    app.gh.diffs[456] = _NO_OP_DIFF
    app.gh.pr_head_shas[456] = "ccc333"
    commits = _exempt_trailer_commits()
    app.gh.pr_commits_by_number[456] = commits
    _seed_no_op_verdict(app)
    pr_commits_calls = _counting_pr_commits(app.gh)
    captured: dict[str, Any] = {}
    _spy_run_janitor(monkeypatch, captured)

    result = app.review(456)

    # Exactly one fetch for the whole pass: check_test_adequacy reuses the
    # list the no-op escape already fetched.
    assert pr_commits_calls == [456]
    assert captured["pr_commits"] == commits
    assert result.ok is True
    assert "prompt_path" in result.data
    assert not result.data.get("is_no_op_rework")


def test_review_escalated_path_wires_pr_commits_into_no_op_escape(
    tmp_path: Path, monkeypatch
) -> None:
    """Escalated-PR refresh arm of review(): the diagnostics-only run_janitor
    call gets the same fetched list, so an escalated PR that lands a
    trailer-only rework re-observes janitor_ok=True instead of a frozen
    no-op failure."""
    app = _test_adequacy_app(tmp_path, enabled=True)
    app.gh.diffs[456] = _NO_OP_DIFF
    app.gh.pr_head_shas[456] = "ccc333"
    commits = _exempt_trailer_commits()
    app.gh.pr_commits_by_number[456] = commits
    _seed_no_op_verdict(app)
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["status"] = "escalated"
    save_state(app.paths.state_file, state)
    pr_commits_calls = _counting_pr_commits(app.gh)
    captured: dict[str, Any] = {}
    _spy_run_janitor(monkeypatch, captured)

    result = app.review(456)

    assert pr_commits_calls == [456]
    assert captured["call_count"] == 1
    assert captured["pr_commits"] == commits
    assert result.data.get("pass_skipped") is True
    # End-to-end: the refreshed janitor diagnostics show the escape fired.
    state = load_state(app.paths.state_file)
    assert state["prs"]["456"]["janitor_ok"] is True


def test_deescalate_mechanical_issue_wires_pr_commits_into_no_op_escape(
    tmp_path: Path, monkeypatch
) -> None:
    """``_deescalate_mechanical_issue``'s fresh run_janitor re-check gets the
    fetched commit list too: a mechanically-escalated issue whose PR landed
    the trailer-only rework clears because the escape sees the claim."""
    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(enabled=True, exempt_marker="Test-exempt:"),
        # Same isolation as _unescalate_fixtures._app: point the post-mortem
        # sessions.db probe at a nonexistent path so issue_worker_liveness
        # never picks up a real runner DB for the test PID.
        post_mortem=PostMortemConfig(db_path=str(tmp_path / "missing-sessions.db")),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    app.gh.diffs[456] = _NO_OP_DIFF
    app.gh.pr_head_shas[456] = "ccc333"
    commits = _exempt_trailer_commits()
    app.gh.pr_commits_by_number[456] = commits
    _seed_no_op_verdict(app)
    state = load_state(app.paths.state_file)
    state["prs"]["456"]["status"] = "escalated"
    state["prs"]["456"]["escalation_reason"] = "session_failed_escalated"
    state["issues"]["123"] = {
        "number": 123,
        "status": "escalated",
        "escalation_reason": "session_failed_escalated",
        "reason_class": "mechanical",
    }
    save_state(app.paths.state_file, state)
    pr_commits_calls = _counting_pr_commits(app.gh)
    captured: dict[str, Any] = {}
    _spy_run_janitor(monkeypatch, captured)

    outcome = app._deescalate_mechanical_issue(123)

    assert pr_commits_calls == [456]
    assert captured["call_count"] == 1
    assert captured["pr_commits"] == commits
    assert outcome.get("cleared") is True
    state = load_state(app.paths.state_file)
    assert state["issues"]["123"]["status"] == PASSIVE_OPEN_STATUS


def test_review_no_fetch_without_request_changes_verdict(tmp_path: Path, monkeypatch) -> None:
    """Negative wiring: no terminal verdict on disk -> the escape is dead, so
    review() must not spend the fetch for the no-op gate. The ONE call still
    observed is check_test_adequacy's own channel (the gate is enabled),
    which run_janitor provably never saw."""
    app = _test_adequacy_app(tmp_path, enabled=True)
    app.gh.diffs[456] = _NO_OP_DIFF
    app.gh.pr_commits_by_number[456] = _exempt_trailer_commits()
    pr_commits_calls = _counting_pr_commits(app.gh)
    captured: dict[str, Any] = {}
    _spy_run_janitor(monkeypatch, captured)

    result = app.review(456)

    assert captured["pr_commits"] is None
    assert pr_commits_calls == [456]
    assert result.ok is True
    assert "prompt_path" in result.data


def test_review_no_fetch_when_adequacy_gate_disabled(tmp_path: Path, monkeypatch) -> None:
    """Negative wiring: gate disabled -> the marker channel is unsanctioned,
    so a live request_changes verdict still buys no fetch. Zero pr_commits
    calls at all (check_test_adequacy does not run either), and the no-op
    block still fires fail-closed."""
    app = _test_adequacy_app(tmp_path, enabled=False)
    app.gh.diffs[456] = _NO_OP_DIFF
    app.gh.pr_head_shas[456] = "ccc333"
    # Data is available and would satisfy the escape -- it must still never
    # be read while the gate is off.
    app.gh.pr_commits_by_number[456] = _exempt_trailer_commits()
    _seed_no_op_verdict(app)
    pr_commits_calls = _counting_pr_commits(app.gh)
    captured: dict[str, Any] = {}
    _spy_run_janitor(monkeypatch, captured)

    result = app.review(456)

    assert pr_commits_calls == []
    assert captured["pr_commits"] is None
    assert "prompt_path" not in result.data


def test_review_no_fetch_when_head_not_advanced(tmp_path: Path, monkeypatch) -> None:
    """Negative wiring for the head-not-advanced arm: the live head still
    equals the verdict's ``reviewed_head_sha``, so no post-verdict commit
    exists that could carry a new claim -- the fetch is skipped even with
    gate on + request_changes, and the no-op block still applies."""
    app = _test_adequacy_app(tmp_path, enabled=True)
    app.gh.diffs[456] = _NO_OP_DIFF
    app.gh.pr_head_shas[456] = "aaa111"  # head pinned at the reviewed commit
    app.gh.pr_commits_by_number[456] = _exempt_trailer_commits()
    _seed_no_op_verdict(app)
    pr_commits_calls = _counting_pr_commits(app.gh)
    captured: dict[str, Any] = {}
    _spy_run_janitor(monkeypatch, captured)

    result = app.review(456)

    assert captured["pr_commits"] is None
    # Zero fetches at all: the no-op gate blocks before the packet path's
    # check_test_adequacy fetch is ever reached.
    assert pr_commits_calls == []
    assert "prompt_path" not in result.data
