"""Rework-dispatch brief regeneration: staleness detection and sidecar writes.

Split out of ``tests/test_charlie_work.py`` (issue #1547, Track-1 wave 1/8):
the ``test_dispatch_rework_*`` seam's brief/manifest half -- stale-brief
regeneration after decision edits or renderer changes, untouched-brief
no-ops, unreadable-sidecar tolerance, and combined-manifest adapter-label
writes. Rendered-brief content lives in
``tests/test_charlie_work_rework_brief.py``; shared fakes and helpers in
``tests/_rework_dispatch_fixtures.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import pytest
from _fakes_github import FakeGitHub
from _rework_dispatch_fixtures import (
    _TwoReworkIssuesGitHub,
    _api_worker_config_for_test,
    _fake_dispatch_sessions_writing_manifests,
    _seed_two_rework_issues,
)
from charlie_work.config import (
    DevinConfig,
    OrchestratorConfig,
    WorkerRoleConfig,
)
from charlie_work.instrumentation import query_events
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    load_state,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp


def test_dispatch_rework_regenerates_stale_brief_after_decision_edit(
    tmp_path: Path,
) -> None:
    """Issue #632 defect 4 / #510 case: dispatch_rework reads the brief
    verbatim, so a hand-corrected review-decision.json never reached the
    worker. The brief must be regenerated when the verdict is newer than the
    brief, reflecting the edit."""
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    app = OrchestratorApp(tmp_path, paths, config, ReworkGitHub())

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    # A stale brief predating the fix: churn note, zero findings — exactly
    # the #510 artifact.
    brief_path = pr_dir / "rework-prompt.md"
    brief_path.write_text("# Rework\n\nThe previous cycle was a no-op.\n", encoding="utf-8")
    # The operator then corrects the verdict by hand, adding the findings.
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "corrected verdict",
                "required_changes": ["handle the empty-list case", "guard the index access"],
            }
        ),
        encoding="utf-8",
    )
    # Make the corrected verdict strictly newer than the stale brief.
    now = time.time()
    os.utime(brief_path, (now, now))
    os.utime(decision_path, (now + 10, now + 10))

    result = app.dispatch_rework()

    assert result.ok is True, result.message
    assert result.data["selected_count"] == 1
    regenerated = brief_path.read_text(encoding="utf-8")
    # The regenerated brief reflects the edited verdict.
    assert "handle the empty-list case" in regenerated
    assert "guard the index access" in regenerated
    # The stale churn-only content is gone.
    assert "The previous cycle was a no-op." not in regenerated


def test_dispatch_rework_regenerates_brief_after_renderer_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #800: the brief goes stale along a second axis the mtime gate
    cannot see.

    ``_is_verdict_newer_than_brief`` compares timestamps, so it only detects
    *decision content* drift. The #800 case is the other axis: the decision is
    byte-identical and older than the brief, but the **renderer** changed
    underneath it (commit ``35c072d`` added tier-2/tier-3 fallbacks to
    ``_render_required_changes_section``; #883 changed ``rework.md`` itself).
    Every brief already on disk then keeps rendering through the old code
    forever, because nothing ever makes the verdict look newer.

    Monkeypatching the renderer reproduces that exactly: same decision, same
    mtimes, different output. The brief must pick the change up at dispatch.

    Issue #1283 Phase A: ``_render_required_changes_section`` moved to
    ``charlie_work/rework_prompts.py``. Its sole in-family caller,
    ``_render_rework_prompt``, moved with it and resolves the name from
    ``rework_prompts``'s own module globals, not workflow.py's facade
    re-export -- so the monkeypatch target below must be
    ``rework_prompts_module``, not ``workflow_module``.
    """
    from charlie_work import rework_prompts as rework_prompts_module
    from charlie_work.workflow import _is_verdict_newer_than_brief

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "needs work",
                "required_changes": ["handle the empty-list case"],
            }
        ),
        encoding="utf-8",
    )
    # Write the brief through the real writer, so it carries a dispatch-note
    # sidecar exactly as production briefs do. Seed it from the *same* PR dict
    # dispatch_rework will re-render with, so the renderer swap below is the
    # only thing that differs — otherwise a PR-metadata mismatch would
    # regenerate the brief and the test would pass without proving anything.
    pr = gh.prs[0]
    brief_path = app._write_rework_prompt(pr, 123, "the operational note")
    original = brief_path.read_text(encoding="utf-8")
    assert "handle the empty-list case" in original

    # The decision is *older* than the brief, so the #632 mtime gate is
    # definitively off — this test cannot pass through that path.
    now = time.time()
    os.utime(decision_path, (now, now))
    os.utime(brief_path, (now + 10, now + 10))
    assert not _is_verdict_newer_than_brief(decision_path, brief_path)

    # Now the renderer changes, with the decision untouched.
    monkeypatch.setattr(
        rework_prompts_module,
        "_render_required_changes_section",
        lambda decision: "## Required changes\n\nRENDERED-BY-NEW-CODE\n",
    )

    result = app.dispatch_rework()

    assert result.ok is True, result.message
    assert result.data["selected_count"] == 1
    regenerated = brief_path.read_text(encoding="utf-8")
    assert "RENDERED-BY-NEW-CODE" in regenerated
    # The note survives the regeneration: it is replayed from the sidecar,
    # which is what makes re-rendering input-preserving rather than lossy.
    assert "the operational note" in regenerated
    # A silent rewrite would be unauditable, and the two staleness axes are
    # worth telling apart in the log: this one is renderer drift, not the
    # #632 newer-verdict case.
    events = query_events(paths.state_file, kind="rework_brief_regenerated")
    assert len(events) == 1, events
    assert events[0]["payload"]["reason"] == "renderer_drift"
    assert events[0]["payload"]["pr_number"] == 456


def test_dispatch_rework_leaves_brief_untouched_when_nothing_changed(
    tmp_path: Path,
) -> None:
    """Complement to the #800 regeneration test: re-rendering is unconditional,
    but *writing* is not.

    Without this the fix would rewrite every brief on every dispatch pass,
    churning mtimes that ``_is_verdict_newer_than_brief`` reads. Content is
    compared before writing, so a brief that is already current keeps its
    bytes and its mtime.
    """
    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    (pr_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "needs work",
                "required_changes": ["handle the empty-list case"],
            }
        ),
        encoding="utf-8",
    )
    pr = gh.prs[0]
    brief_path = app._write_rework_prompt(pr, 123, "the operational note")
    before = brief_path.read_text(encoding="utf-8")
    before_mtime = brief_path.stat().st_mtime_ns

    result = app.dispatch_rework()

    assert result.ok is True, result.message
    assert brief_path.read_text(encoding="utf-8") == before
    assert brief_path.stat().st_mtime_ns == before_mtime


def test_dispatch_rework_does_not_regenerate_when_sidecar_is_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-rendering is only safe because the dispatch note can be replayed from
    the sidecar. An unreadable sidecar must be treated exactly like an absent
    one — fall back to the mtime gate — because regenerating with an empty note
    would silently drop the note the brief is carrying.

    Its positive control is ``..._regenerates_brief_after_renderer_change``:
    identical setup and identical mtime ordering, differing only in that the
    sidecar is readable there. That one regenerates and this one must not, so
    "no regeneration" here is attributable to the sidecar rather than to the
    PR never being selected for dispatch at all.

    Issue #1283 Phase A: see the sibling test's docstring above for why the
    monkeypatch target is ``rework_prompts_module``, not ``workflow_module``.
    """
    from charlie_work import rework_prompts as rework_prompts_module

    config = OrchestratorConfig(
        devin=DevinConfig(
            dispatch_command=(
                sys.executable,
                "-c",
                "import sys; print(sys.argv[1])",
                "{issue_number}",
            )
        ),
        worker=WorkerRoleConfig(harness="command"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    class ReworkGitHub(FakeGitHub):
        def __init__(self) -> None:
            super().__init__()
            self.issues[0]["labels"] = [{"name": "agent:needs-rework"}]

    paths.root.mkdir(parents=True, exist_ok=True)
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        state["issues"]["123"] = {
            "number": 123,
            "title": "Fix search",
            "url": "https://example.test/issues/123",
            "status": "rework_requested",
        }
        save_state(paths.state_file, state)

    gh = ReworkGitHub()
    app = OrchestratorApp(tmp_path, paths, config, gh)

    pr_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456"
    pr_dir.mkdir(parents=True)
    decision_path = pr_dir / "review-decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "needs work",
                "required_changes": ["handle the empty-list case"],
            }
        ),
        encoding="utf-8",
    )
    pr = gh.prs[0]
    brief_path = app._write_rework_prompt(pr, 123, "the operational note")
    # Corrupt the sidecar: not valid UTF-8, so reading it raises.
    (pr_dir / "rework-dispatch-note.txt").write_bytes(b"\xff\xfe not utf-8 \xff")
    before = brief_path.read_text(encoding="utf-8")

    now = time.time()
    os.utime(decision_path, (now, now))
    os.utime(brief_path, (now + 10, now + 10))

    monkeypatch.setattr(
        rework_prompts_module,
        "_render_required_changes_section",
        lambda decision: "## Required changes\n\nRENDERED-BY-NEW-CODE\n",
    )

    result = app.dispatch_rework()

    assert result.ok is True, result.message
    # The renderer changed, but without a replayable note the brief is left
    # alone rather than rewritten without its note.
    assert brief_path.read_text(encoding="utf-8") == before
    assert "the operational note" in before


def test_dispatch_rework_combined_manifest_mixed_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #626: when a rework pass dispatches both a normal issue (via the
    default adapter ``devin-shell``) and a rescue-marked issue (via the
    claude-code rescue adapter), the combined manifest's adapter label is
    ``"mixed"`` — derived from the actual partition, not the default adapter
    name. Before #626, the trailing write used ``self.config.devin.adapter``
    unconditionally, mislabeling a mixed batch."""
    from charlie_work.config import RescueConfig

    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
        rescue=RescueConfig(enabled=True, worker_model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_two_rework_issues(paths, config, rescue_issue_numbers={124})
    fake_gh = _TwoReworkIssuesGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    manifest_writes: list[str] = []
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        _fake_dispatch_sessions_writing_manifests(manifest_writes, tmp_path),
    )

    result = app.dispatch_rework(limit=5)

    assert result.ok is True
    # Sub-calls wrote manifests with their own adapter labels; the combined
    # trailing write is the last manifest write and must be "mixed".
    manifest_path = tmp_path / ".var" / "charlie-work" / "dispatches" / "session-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_issue_numbers = {s["issue_number"] for s in manifest["sessions"]}
    assert session_issue_numbers == {123, 124}
    assert manifest["adapter"] == "mixed"
    assert "more than one worker" in " ".join(manifest["instructions"])


def test_dispatch_rework_combined_manifest_homogeneous_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #626: when a rework pass dispatches both a normal issue and a
    rescue-marked issue, but both use the same adapter kind (claude-code),
    the combined manifest's adapter label is ``"claude-code"`` — not
    ``"mixed"`` and not the default adapter name.

    The normal issue uses claude-code because that is the pass's configured
    worker harness (``worker.harness``), while the rescue issue uses
    claude-code via the rescue adapter. Both kinds are ``"claude-code"`` →
    homogeneous → ``"claude-code"``. Before #626, the trailing write used
    ``self.config.devin.adapter`` (``"devin-shell"`` here) unconditionally,
    mislabeling the homogeneous batch."""
    from charlie_work.config import RescueConfig

    # DevinConfig.adapter has been deleted (worker harness now lives on
    # worker.harness). Both issues actually use claude-code: the normal issue
    # via the configured worker harness, the rescue issue via the rescue
    # adapter. Both kinds are "claude-code" -> homogeneous -> "claude-code".
    # (Historically, before #626, the trailing write used
    # self.config.devin.adapter unconditionally and would have mislabeled
    # this "devin-shell".)
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="claude-code"),
        api_worker=_api_worker_config_for_test(enabled=True),
        rescue=RescueConfig(enabled=True, worker_model="claude-opus-4-1"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    _seed_two_rework_issues(paths, config, rescue_issue_numbers={124})
    fake_gh = _TwoReworkIssuesGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    manifest_writes: list[str] = []
    monkeypatch.setattr(
        "charlie_work.workflow.dispatch_sessions",
        _fake_dispatch_sessions_writing_manifests(manifest_writes, tmp_path),
    )

    result = app.dispatch_rework(limit=5)

    assert result.ok is True
    manifest_path = tmp_path / ".var" / "charlie-work" / "dispatches" / "session-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_issue_numbers = {s["issue_number"] for s in manifest["sessions"]}
    assert session_issue_numbers == {123, 124}
    # Both normal and rescue use claude-code → homogeneous → "claude-code".
    assert manifest["adapter"] == "claude-code"
    assert "multiple worker adapters" not in " ".join(manifest["instructions"])


def test_dispatch_rework_no_rescue_skips_redundant_manifest_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #626: when a rework pass has only normal (non-rescue) issues, the
    trailing combined manifest write is skipped — ``dispatch_sessions``
    already wrote the correct manifest. Before #626, the trailing write was
    unconditional, writing the manifest twice per pass with the wrong label.

    This test monkeypatches ``write_session_manifest`` in the workflow module
    (not just ``dispatch_sessions``) so the trailing direct call is counted
    too — that is the call #626 makes conditional."""
    config = OrchestratorConfig(
        devin=DevinConfig(),
        worker=WorkerRoleConfig(harness="devin-shell"),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    # Only issue 123 (normal), no rescue marker on issue 124's PR.
    _seed_two_rework_issues(paths, config, rescue_issue_numbers=set())
    # Remove issue 124's rework state so only 123 is a candidate.
    with state_lock(paths.state_file):
        state = load_state(paths.state_file)
        del state["issues"]["124"]
        save_state(paths.state_file, state)
    fake_gh = _TwoReworkIssuesGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # Count every write_session_manifest call from the workflow module,
    # including the trailing combined write that #626 makes conditional.
    from charlie_work.adapters import write_session_manifest as _real_wsm

    manifest_write_count = 0

    def _counting_write_session_manifest(path, requests, *, adapter="manual"):
        nonlocal manifest_write_count
        manifest_write_count += 1
        return _real_wsm(path, requests, adapter=adapter)

    # Patch write_session_manifest in both modules: workflow.py's trailing
    # write uses the workflow-module binding, and dispatch_sessions uses the
    # adapters-module binding.
    monkeypatch.setattr(
        "charlie_work.workflow.write_session_manifest",
        _counting_write_session_manifest,
    )
    monkeypatch.setattr(
        "charlie_work.adapters.write_session_manifest",
        _counting_write_session_manifest,
    )

    # Also patch dispatch_sessions to avoid real worker launches, having it
    # call the counting write_session_manifest like the real one does.
    def _fake_dispatch_sessions(_repo_root, manifest_path, results_path, settings, requests):
        from charlie_work.adapters import SessionDispatchResult, write_session_results

        _counting_write_session_manifest(manifest_path, requests, adapter=settings.adapter)
        results = [
            SessionDispatchResult(
                issue_number=r.issue_number,
                issue_title=r.issue_title,
                prompt_path=str(r.prompt_path),
                branch_name=r.branch_name,
                adapter=settings.adapter,
                ok=True,
                pid=4242,
                process_start_time=1.0,
            )
            for r in requests
        ]
        write_session_results(results_path, results)
        return results

    monkeypatch.setattr("charlie_work.workflow.dispatch_sessions", _fake_dispatch_sessions)

    result = app.dispatch_rework(limit=5)

    assert result.ok is True
    # With the fix: one write from dispatch_sessions inside
    # _dispatch_rework_impl. The trailing combined write is skipped because
    # rescue_requests is empty. Before #626, this was 2 (the trailing write
    # was unconditional).
    assert manifest_write_count == 1
    manifest_path = tmp_path / ".var" / "charlie-work" / "dispatches" / "session-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["adapter"] == "devin-shell"
