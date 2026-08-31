"""Tests for issue #1445: reviewer rubric line -- new code added to an
over-cap file is a reportable finding.

Covers the three acceptance criteria:
1. The rubric text is present in built review packets (always -- permanent
   template prose, like the ``## Static probe`` heading).
2. A synthetic PR that adds code to a god (over-cap) file yields the finding
   in the built packet.
3. A synthetic PR that adds code to a normal (under-cap) module does NOT
   yield the finding.

Unit tests exercise ``_over_cap_file_findings`` / ``render_over_cap_section``
directly; integration tests drive ``OrchestratorApp.review`` end-to-end and
read the rendered ``review-prompt.md`` packet, mirroring the
``test_static_probe_section_in_review_packet_*`` pattern.
"""

from __future__ import annotations

from pathlib import Path

from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig, ReviewDispatchConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import (
    OrchestratorApp,
    OverCapFileFinding,
    _over_cap_file_findings,
    render_over_cap_section,
)

# A small unified diff that adds one line to ``src/god_file.py``.
_GOD_FILE_DIFF = (
    "diff --git a/src/god_file.py b/src/god_file.py\n"
    "index 123..456 100644\n"
    "--- a/src/god_file.py\n"
    "+++ b/src/god_file.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def existing():\n"
    "     pass\n"
    "+def new_helper(x):\n"
    "+    return x\n"
)

# Same shape, but for a normal-sized module.
_NORMAL_FILE_DIFF = (
    "diff --git a/src/normal.py b/src/normal.py\n"
    "index 123..456 100644\n"
    "--- a/src/normal.py\n"
    "+++ b/src/normal.py\n"
    "@@ -1,3 +1,4 @@\n"
    " def existing():\n"
    "     pass\n"
    "+def new_helper(x):\n"
    "+    return x\n"
)


def _write_lines(path: Path, n: int) -> None:
    """Write a file with exactly ``n`` lines to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"line {i}" for i in range(n)) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests: _over_cap_file_findings / render_over_cap_section
# ---------------------------------------------------------------------------


def test_over_cap_disabled_returns_empty(tmp_path: Path) -> None:
    """cap=0 disables the probe -- no findings regardless of file size."""
    god = tmp_path / "src" / "god_file.py"
    _write_lines(god, 1000)
    assert _over_cap_file_findings(_GOD_FILE_DIFF, tmp_path, 0) == ()


def test_over_cap_addition_to_god_file_yields_finding(tmp_path: Path) -> None:
    """A diff that adds code to a file whose post-diff line count exceeds the
    cap produces a finding (acceptance criterion 2, unit-level)."""
    god = tmp_path / "src" / "god_file.py"
    _write_lines(god, 100)  # base 100; diff adds 2 -> 102 > 100
    findings = _over_cap_file_findings(_GOD_FILE_DIFF, tmp_path, 100)
    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, OverCapFileFinding)
    assert finding.filename == "src/god_file.py"
    assert finding.cap == 100
    assert finding.added_lines == 2
    assert finding.line_count == 102


def test_over_cap_addition_to_normal_module_no_finding(tmp_path: Path) -> None:
    """A diff that adds code to a file under the cap produces no finding
    (acceptance criterion 3, unit-level)."""
    normal = tmp_path / "src" / "normal.py"
    _write_lines(normal, 50)  # base 50; diff adds 2 -> 52 < 100
    assert _over_cap_file_findings(_NORMAL_FILE_DIFF, tmp_path, 100) == ()


def test_over_cap_skips_files_with_no_additions(tmp_path: Path) -> None:
    """A pure deletion to an over-cap file is not 'new code added' -- no
    finding. The rubric flags additions, not the file's mere existence."""
    god = tmp_path / "src" / "god_file.py"
    _write_lines(god, 200)
    deletion_only = (
        "diff --git a/src/god_file.py b/src/god_file.py\n"
        "index 123..456 100644\n"
        "--- a/src/god_file.py\n"
        "+++ b/src/god_file.py\n"
        "@@ -1,4 +1,2 @@\n"
        " def existing():\n"
        "     pass\n"
        "-def gone():\n"
        "-    return 0\n"
    )
    assert _over_cap_file_findings(deletion_only, tmp_path, 100) == ()


def test_over_cap_new_file_uses_added_lines_as_size(tmp_path: Path) -> None:
    """A brand-new file (not present at repo_root) is sized by its added
    lines; if that exceeds the cap it is a finding."""
    new_file_diff = (
        "diff --git a/src/brand_new.py b/src/brand_new.py\n"
        "new file mode 100644\n"
        "index 0000000..1234567\n"
        "--- /dev/null\n"
        "+++ b/src/brand_new.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+def a():\n"
        "+    pass\n"
        "+def b():\n"
        "+    pass\n"
        "+def c():\n"
    )
    findings = _over_cap_file_findings(new_file_diff, tmp_path, 4)
    assert len(findings) == 1
    assert findings[0].filename == "src/brand_new.py"
    assert findings[0].line_count == 5
    assert findings[0].added_lines == 5


def test_render_over_cap_section_disabled_is_empty() -> None:
    assert render_over_cap_section(None) == ""


def test_render_over_cap_section_clean_is_visible() -> None:
    """Enabled + zero findings renders a visible clean line, never bare ''."""
    section = render_over_cap_section(())
    assert section == "File-size cap: no over-cap additions in this diff.\n"


def test_render_over_cap_section_with_finding() -> None:
    findings = (OverCapFileFinding("src/god_file.py", 102, 100, 2),)
    section = render_over_cap_section(findings)
    assert "Over-cap file additions (issue #1445)" in section
    assert "src/god_file.py" in section
    assert "102 lines" in section
    assert "cap 100" in section
    assert "+2 added" in section
    assert "REPORTABLE FINDING" in section
    assert "#1283-era extractions" in section


def test_render_over_cap_section_instructs_refresh_script() -> None:
    """Issue #1496: the extraction remedy must also tell the worker to run the
    ratchet refresh script in the shrink PR, so the baseline tightens
    alongside the extraction instead of accumulating stale-high headroom.
    The script's default mode is lower-only, so the instruction is safe to
    run mid-PR (same-bucket shrinks produce no diff; cross-bucket shrinks
    write the deterministic quantized value)."""
    findings = (OverCapFileFinding("src/god_file.py", 102, 100, 2),)
    section = render_over_cap_section(findings)
    assert "refresh_file_size_ratchet.py" in section
    assert "file_size_ratchet_baseline.json" in section
    # The safety rationale must travel with the instruction so a reviewer
    # does not strip it as a risky side effect.
    assert "lower-only" in section


# ---------------------------------------------------------------------------
# Integration tests: built review packet
# ---------------------------------------------------------------------------


def _build_packet(tmp_path: Path, config: OrchestratorConfig, diff: str) -> str:
    fake_gh = FakeGitHub()
    fake_gh.diffs[456] = diff
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    result = app.review(456)
    assert result.ok is True
    packet = tmp_path / ".var" / "charlie-work" / "prs" / "pr-456" / "review-prompt.md"
    return packet.read_text(encoding="utf-8")


def test_rubric_text_present_in_built_packet_default(tmp_path: Path) -> None:
    """Acceptance criterion 1: the rubric text is present in built review
    packets even with the cap disabled (default). Permanent template prose,
    like the ``## Static probe`` heading."""
    config = OrchestratorConfig()  # file_size_cap_lines defaults to 0
    packet = _build_packet(tmp_path, config, _GOD_FILE_DIFF)

    assert "## File-size cap" in packet
    assert "REPORTABLE FINDING" in packet
    # The rubric references the cap source generically, not a hardcoded number.
    # The source is the ``review_dispatch:`` config section (ReviewDispatchConfig),
    # NOT the unrelated ``review:`` section (ReviewConfig) -- issue #1445 review
    # finding: the prose must name the correct namespace so reviewers read the
    # cap from its actual source.
    assert "review_dispatch.file_size_cap_lines" in packet
    assert "review.file_size_cap_lines" not in packet
    assert "#1442" in packet
    assert "#1283-era extractions" in packet
    # Issue #1496: the rubric's extraction remedy must name the refresh
    # script so the reviewer's required_changes propagate the tightening
    # step to the rework worker. Present even with the cap disabled (the
    # rubric is permanent template prose).
    assert "refresh_file_size_ratchet.py" in packet
    # Disabled -> no dynamic section text, and no unresolved placeholder.
    assert "Over-cap file additions" not in packet
    assert "$over_cap_section" not in packet


def test_god_file_addition_yields_finding_in_packet(tmp_path: Path) -> None:
    """Acceptance criterion 2: a synthetic PR that adds code to a god
    (over-cap) file yields the finding in the built packet."""
    god = tmp_path / "src" / "god_file.py"
    _write_lines(god, 100)  # base 100; diff adds 2 -> 102 > 100
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(file_size_cap_lines=100))
    packet = _build_packet(tmp_path, config, _GOD_FILE_DIFF)

    assert "Over-cap file additions (issue #1445)" in packet
    assert "src/god_file.py" in packet
    assert "REPORTABLE FINDING" in packet
    assert "$over_cap_section" not in packet


def test_normal_module_addition_does_not_yield_finding_in_packet(
    tmp_path: Path,
) -> None:
    """Acceptance criterion 3: a synthetic PR that adds code to a normal
    (under-cap) module does NOT yield the finding in the built packet."""
    normal = tmp_path / "src" / "normal.py"
    _write_lines(normal, 50)  # base 50; diff adds 2 -> 52 < 100
    config = OrchestratorConfig(review_dispatch=ReviewDispatchConfig(file_size_cap_lines=100))
    packet = _build_packet(tmp_path, config, _NORMAL_FILE_DIFF)

    # Rubric prose still present (permanent), but no finding.
    assert "## File-size cap" in packet
    assert "Over-cap file additions" not in packet
    # Enabled + clean renders the visible clean line, not bare ''.
    assert "File-size cap: no over-cap additions in this diff." in packet
    assert "$over_cap_section" not in packet
