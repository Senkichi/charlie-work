"""Review-packet optional sections: test-adequacy, static/coverage probes, check carry-forward.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

import json
from pathlib import Path
from _fakes_github import FakeGitHub
from _review_fixtures import (
    _review_queue_carry_forward_app,
    _write_review_packet,
)
from charlie_work.config import (
    OrchestratorConfig,
    TestAdequacyConfig,
)
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp
from _dispatch_fixtures import _stub_real_activity_probe_for_stalled_tests  # noqa: F401


def test_check_carry_forward_tier1_whitespace_collision_bypasses_tier2(
    tmp_path: Path,
) -> None:
    """Issue #1187 (audit #634 rework): ``git patch-id --stable`` strips
    leading whitespace from ``+``/``-`` content lines, so two diffs that
    differ ONLY in indentation depth produce the identical patch-id.  The
    tier-1 fast path in ``_check_carry_forward`` previously matched on that
    hash and returned ``"patch-id"`` immediately — it never consulted the
    tier-2 line-content signature, which DOES preserve whitespace and WOULD
    distinguish the two diffs.

    In Python, an indentation-only change can alter control flow (e.g. moving
    a ``return`` into or out of an ``if`` block).  Carrying forward an
    approved verdict across such a change without review is a review-gate
    bypass.  After the fix, the tier-1 patch-id match is no longer
    sufficient: the tier-2 line-content signature is also validated, and
    when it differs (a whitespace-only change that patch-id collapsed) the
    carry-forward is REFUSED — the verdict is reported stale.
    """
    from charlie_work.janitor import _calculate_patch_id, _diff_content_signature

    # Diff reviewed at approval time: indent ``return True`` to 8 spaces.
    reviewed_diff = (
        "diff --git a/mod.py b/mod.py\n"
        "index 0c54d1a..8e25a7e 100644\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return True\n"
        "+        return True\n"
    )
    # Live diff: same logical change but indented to 12 spaces instead of 8.
    # ``git patch-id --stable`` strips leading whitespace, so both produce
    # the same hash.  The tier-2 signature preserves whitespace verbatim and
    # differs.
    live_diff = (
        "diff --git a/mod.py b/mod.py\n"
        "index 0c54d1a..f80ba40 100644\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def f():\n"
        "-    return True\n"
        "+            return True\n"
    )

    reviewed_patch_id = _calculate_patch_id(reviewed_diff)
    live_patch_id = _calculate_patch_id(live_diff)
    assert reviewed_patch_id == live_patch_id, (
        "two diffs differing only in indentation depth must produce the "
        "same git patch-id --stable (the vulnerability's precondition)"
    )
    assert reviewed_patch_id != "", "sanity: patch-id must be non-empty"

    reviewed_sig = _diff_content_signature(reviewed_diff)
    live_sig = _diff_content_signature(live_diff)
    assert reviewed_sig.changed_lines != live_sig.changed_lines, (
        "tier-2 signatures must differ — tier-2 preserves whitespace and "
        "WOULD catch the indentation change that tier-1 misses"
    )
    assert reviewed_sig.changed_files == live_sig.changed_files

    old_head = "sha-reviewed-head"
    new_head = "sha-reindented-head"
    pr_number = 456
    issue_number = 123

    prs = [
        {
            "number": pr_number,
            "title": f"Fix #{issue_number}",
            "url": f"https://example.test/pull/{pr_number}",
            "headRefName": f"agent/issue-{issue_number}-fix",
            "baseRefName": "main",
            "headRefOid": new_head,
            "mergeStateStatus": "CLEAN",
            "body": f"Closes #{issue_number}",
            "labels": [],
            "isCrossRepository": False,
            "state": "OPEN",
        }
    ]
    app = _review_queue_carry_forward_app(tmp_path, prs=prs)
    fake_gh = app.gh
    fake_gh.diffs[pr_number] = live_diff

    _write_review_packet(
        tmp_path,
        pr_number,
        new_head,
        {
            "decision": "approved",
            "reviewed_head_sha": old_head,
            "reviewed_patch_id": reviewed_patch_id,
            "reviewed_changed_lines": list(reviewed_sig.changed_lines),
            "reviewed_changed_files": sorted(reviewed_sig.changed_files),
            "reviewed_has_binary": reviewed_sig.has_binary,
            "carried_forward_from": [],
        },
    )

    result = app.review_queue()

    assert result.ok is True
    assert result.data["queue"] != [], (
        "the verdict must NOT be carried forward across an "
        "indentation-only change — tier-2 signature differs, so the "
        "verdict is reported stale for re-review"
    )

    decision = json.loads(
        (app.paths.prs / f"pr-{pr_number}" / "review-decision.json").read_text(encoding="utf-8")
    )
    assert decision["reviewed_head_sha"] == old_head, (
        "the approved verdict must NOT be carried forward to the "
        "reindented head — the review-gate bypass is closed"
    )
    assert decision.get("carry_forward_tier") != "patch-id", (
        "carry-forward must not be via tier-1 (patch-id) when the tier-2 "
        "line-content signature differs — the whitespace collision is "
        "detected and the verdict is refused"
    )


def test_test_adequacy_section_in_review_packet_when_enabled(tmp_path: Path) -> None:
    """Integration test: verify test_adequacy_section appears in review packet when gate is enabled and passes (issue #180)."""
    from unittest.mock import patch
    from charlie_work.janitor import TestAdequacyFacts, TestAdequacyVerdict

    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(
            enabled=True,
            exempt_marker="Test-exempt:",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    # Mock check_test_adequacy to return a passing verdict with facts
    mock_facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=(),
        exempt=False,
        exempt_reason="",
    )
    mock_verdict = TestAdequacyVerdict(
        ok=True,
        failures=(),
        warnings=(),
        facts=mock_facts,
    )

    with patch("charlie_work.workflow.check_test_adequacy", return_value=mock_verdict):
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Verify the test-adequacy facts section appears in the packet
    assert "## Test-adequacy facts (Tier 1, deterministic)" in packet_text
    # Verify no unresolved placeholder
    assert "$test_adequacy_section" not in packet_text


def test_test_adequacy_section_not_in_review_packet_when_disabled(tmp_path: Path) -> None:
    """Integration test: verify test_adequacy_section does not appear in review packet when gate is disabled (issue #180)."""
    config = OrchestratorConfig(
        test_adequacy=TestAdequacyConfig(
            enabled=False,  # Gate disabled
            exempt_marker="Test-exempt:",
        )
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Verify the test-adequacy facts section does NOT appear in the packet
    assert "## Test-adequacy facts (Tier 1, deterministic)" not in packet_text
    # Verify no unresolved placeholder
    assert "$test_adequacy_section" not in packet_text


def test_static_probe_section_not_in_review_packet_when_disabled(tmp_path: Path) -> None:
    """When coverage_probe.enabled=False (default), the computed section is
    empty and no dynamic probe content leaks into the packet. The STATIC
    '## Static probe' heading + rubric prose (W20 item 2) are permanent
    template text and remain present regardless -- mirrors the always-
    present '## Test adequacy' template heading precedent."""
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "index 123..456 100644\n"
        "--- a/src/feature.py\n"
        "+++ b/src/feature.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def feature():\n"
        "     pass\n"
        "+def new_feature(x):\n"
        "+    if x:\n"
        "+        return 1\n"
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    # Static template heading + rubric prose (W20 item 2) always present.
    assert "## Static probe" in packet_text
    assert "Name the production caller." in packet_text
    assert "Classify test strength." in packet_text
    assert "existence < type < status < value <" in packet_text
    # No dynamic probe content and no unresolved placeholder.
    assert "Branch-coverage heuristic (W3)" not in packet_text
    assert "Static probe: no findings." not in packet_text
    assert "$static_probe_section" not in packet_text


def test_static_probe_section_in_review_packet_when_enabled_with_findings(
    tmp_path: Path,
) -> None:
    """Integration test: findings from both probe halves land in
    $static_probe_section, adjacent to (not folded into) ## Test adequacy.

    ``repo_root`` (``tmp_path``) is given a real ``src/`` tree containing a
    file that does NOT reference the flagged symbol, so
    ``_collect_repo_referenced_names`` actually walks it and the line-295
    collision filter runs live (issue #1260/#1261 review finding A5) instead
    of short-circuiting on a missing ``src/`` directory. The flagged symbol
    uses a distinctive name (not ``helper``) precisely so it cannot
    accidentally collide with anything incidental in that tree.
    """
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    # Real src/ tree under repo_root, unrelated to the diffed symbol below --
    # exercises the collision filter live rather than short-circuiting it.
    unrelated_src = tmp_path / "src" / "unrelated_module.py"
    unrelated_src.parent.mkdir(parents=True, exist_ok=True)
    unrelated_src.write_text(
        "def totally_unrelated_function(value):\n    return value * 2\n",
        encoding="utf-8",
    )

    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "index 123..456 100644\n"
        "--- a/src/feature.py\n"
        "+++ b/src/feature.py\n"
        "@@ -1,2 +1,5 @@\n"
        " def feature():\n"
        "     pass\n"
        "+def compute_shard_checksum(x):\n"
        "+    if x:\n"
        "+        return 1\n"
        "diff --git a/tests/test_feature.py b/tests/test_feature.py\n"
        "index 123..456 100644\n"
        "--- a/tests/test_feature.py\n"
        "+++ b/tests/test_feature.py\n"
        "@@ -1,2 +1,4 @@\n"
        " def test_existing():\n"
        "     pass\n"
        "+def test_compute_shard_checksum():\n"
        "+    assert compute_shard_checksum(True) == 1\n"
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")

    assert "## Static probe" in packet_text
    # compute_shard_checksum() is defined in src/feature.py, referenced only
    # from the test, and absent from the real (live-walked) src/ tree -- the
    # collision filter runs and correctly does not suppress it.
    assert "Unwired-symbol probe (W20)" in packet_text
    assert "compute_shard_checksum" in packet_text
    assert "$static_probe_section" not in packet_text
    assert packet_text.index("## Test adequacy") < packet_text.index("## Static probe")


def test_static_probe_section_no_findings_renders_visible_clean_line(tmp_path: Path) -> None:
    """Enabled + zero findings still renders visible text, not "" (mirrors
    render_test_adequacy_section's own always-visible-when-enabled shape)."""
    from charlie_work.config import CoverageProbeConfig

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = (
        "diff --git a/README.md b/README.md\n"
        "index 123..456 100644\n"
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1 +1,2 @@\n"
        " Hello\n"
        "+World\n"
    )

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")
    assert "Static probe: no findings." in packet_text


def test_static_probe_section_shows_visible_degradation_on_internal_error(
    tmp_path: Path, monkeypatch
) -> None:
    """An internal exception in the probe must render a visible warning
    line in the packet, never a silently empty section (design item 7)."""
    from charlie_work.config import CoverageProbeConfig

    def _boom(diff, config):
        raise ValueError("synthetic failure")

    monkeypatch.setattr("charlie_work.diff_coverage_probe.check_branch_coverage", _boom)

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=True))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()

    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)

    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    packet_text = packet.read_text(encoding="utf-8")
    assert "static probe degraded" in packet_text
    assert "branch-coverage heuristic failed" in packet_text


def test_coverage_probe_never_called_when_disabled(tmp_path: Path, monkeypatch) -> None:
    """When coverage_probe.enabled=False (default), run_static_probe is
    never invoked -- mirrors test_review_test_adequacy_disabled_is_noop."""
    from charlie_work.config import CoverageProbeConfig

    calls = {"n": 0}

    def _fake_run_static_probe(diff, repo_root, config):
        calls["n"] += 1
        raise AssertionError("run_static_probe should not be called when disabled")

    monkeypatch.setattr("charlie_work.workflow.run_static_probe", _fake_run_static_probe)

    config = OrchestratorConfig(coverage_probe=CoverageProbeConfig(enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())

    result = app.review(456)

    assert calls["n"] == 0
    assert result.ok is True
