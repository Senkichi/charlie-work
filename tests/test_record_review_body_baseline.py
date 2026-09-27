"""``reviewed_body_sha256`` stamping in ``record_review`` and the body-only
rework escape in the janitor's no-op gate (issue #1939).

Split out of ``tests/test_charlie_work_record_review.py``: that file sits
under the 800-line ratchet cap (issue #1442), so the #1939 coverage lives
here. The unit-level gate surface for the same escape hatch is in
``tests/test_janitor_no_op_rework.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _fakes_github import FakeGitHub
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401

from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp


def _body_sha256(body: str | None) -> str:
    """The ``reviewed_body_sha256`` wire contract (issue #1939): SHA-256 of the
    PR body with CRLF/CR line endings normalized to LF, ``None`` treated as
    the empty string."""
    text = (body or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_record_review_persists_reviewed_body_sha256(tmp_path: Path) -> None:
    """Issue #1939: record_review stamps the hash of the body the reviewer
    saw alongside ``reviewed_patch_id`` -- both in the durable decision
    file and in the ``state["prs"]`` mirror -- so the janitor's no-op
    rework gate can tell a body-only rework (no code diff by construction)
    apart from a genuine no-op."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    # No prior review() call -> the verdict pins to the live head/body.
    result = app.record_review(
        456,
        "request_changes",
        summary="fix the PR description",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    expected = _body_sha256(fake_gh.prs[0]["body"])
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_source"] == "live"
    assert decision["reviewed_body_sha256"] == expected
    assert load_state(paths.state_file)["prs"]["456"]["reviewed_body_sha256"] == expected


def test_record_review_body_baseline_comes_from_packet(tmp_path: Path) -> None:
    """Issue #1939: when the verdict is pinned to the packet, the stamped
    body baseline is the body the reviewer actually read (the packet's
    ``pr.json``), not a live re-fetch -- the same source rule
    ``reviewed_head_sha``/``reviewed_patch_id`` already follow. A body edit
    landing between packet generation and verdict recording is then a real
    change the reviewer has not seen, which the gate must not swallow."""
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 fixed\n"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True

    # The description is edited between packet generation and verdict
    # recording; the packet's pr.json still holds the body the reviewer read.
    packet_body = fake_gh.prs[0]["body"]
    fake_gh.prs[0]["body"] = "Closes #123\n\nTests: corrected after packet gen."

    result = app.record_review(
        456,
        "request_changes",
        summary="fix the PR description",
        verdict_provenance="fresh_llm_review",
    )

    assert result.ok is True
    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_source"] == "packet"
    assert decision["reviewed_body_sha256"] == _body_sha256(packet_body)
    assert decision["reviewed_body_sha256"] != _body_sha256(fake_gh.prs[0]["body"])


def test_body_only_rework_clears_janitor_no_op_gate(tmp_path: Path) -> None:
    """Issue #1939 end-to-end (the swole #198 / PR #348 incident): a
    request_changes verdict, then an orchestrator-applied ``pr_body``
    outcome -- the only channel a credential-less rework worker has to
    edit the description -- must clear the janitor's no-op gate even
    though the patch-id is byte-identical. Before the fix the second
    janitor call below failed with the unchanged-diff failure no matter
    what the worker did to the body."""
    from _janitor_fixtures import _config
    from charlie_work.janitor import run_janitor

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/file b/file\n"
        "index 123..456 100644\n"
        "--- a/file\n"
        "+++ b/file\n"
        "@@ -1,2 +1,2 @@\n"
        " line1\n"
        "-line2\n"
        "+line2 fixed\n"
    )
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    review_result = app.review(456)
    assert review_result.ok is True
    result = app.record_review(
        456,
        "request_changes",
        summary="the PR description misstates the diff; fix the body text",
        verdict_provenance="fresh_llm_review",
    )
    assert result.ok is True

    decision = json.loads(
        (paths.prs / "pr-456" / "review-decision.json").read_text(encoding="utf-8")
    )
    pr_state = load_state(paths.state_file)["prs"]["456"]
    janitor_config = _config()

    # Pre-condition: with the body untouched, the gate blocks exactly as before.
    verdict = run_janitor(
        fake_gh.pr_view(456),
        fake_gh.pr_checks(456),
        janitor_config,
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=fake_gh.pr_diff(456),
        review_decision=decision,
    )
    assert verdict.ok is False
    assert verdict.is_no_op_rework
    assert any("PR diff unchanged since request_changes verdict" in f for f in verdict.failures)

    # The rework outcome the orchestrator applies on the worker's behalf
    # (issue #1853's pr_body channel): only the description changes.
    new_body = tmp_path / "reworked-body.md"
    new_body.write_text(
        "Closes #123\n\nTests: regression coverage added -- description corrected.",
        encoding="utf-8",
    )
    fake_gh.pr_edit(456, new_body)

    verdict = run_janitor(
        fake_gh.pr_view(456),
        fake_gh.pr_checks(456),
        janitor_config,
        pr_state=pr_state,
        repo_root=tmp_path,
        pr_diff=fake_gh.pr_diff(456),
        review_decision=decision,
    )
    assert verdict.ok is True, (
        f"Expected the body-only rework to satisfy the gate, got {verdict.failures}"
    )
    assert not verdict.is_no_op_rework
    assert not any("unchanged since request_changes verdict" in f for f in verdict.failures)
