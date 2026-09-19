"""review.md template-source and test-adequacy-section rendering tests.

Split out of ``tests/test_prompt_sections.py`` (issue #1570, Track 1
shoulder) -- bodies are verbatim relocations; shared helpers live in
``tests/_prompt_sections_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path


def test_review_template_contains_test_adequacy_section_placeholder() -> None:
    """Verify that review.md contains the $test_adequacy_section placeholder (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "$test_adequacy_section" in text, (
        "review.md must contain $test_adequacy_section placeholder"
    )


def test_review_template_contains_hollow_test_heuristics() -> None:
    """Verify that review.md contains the four hollow-test heuristic phrases (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the four hollow-test heuristics are present
    assert "Asserts only that a mock/stub was called, without asserting on real behavior" in text
    assert "Re-asserts a constant the code already hardcodes" in text
    assert "Contains an assertion that cannot fail" in text
    assert "Never imports or exercises the changed symbol" in text


def test_review_template_contains_new_approval_criteria_bullet() -> None:
    """Verify that review.md contains the new test-adequacy approval-criteria bullet (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the new approval-criteria bullet is present
    assert "Every non-exempt changed behavior has a genuine regression test" in text


def test_review_template_retains_existing_test_bullet() -> None:
    """Verify that review.md retains the existing test-related approval-criteria bullet (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the existing bullet is still present (design §9/R4: flagged not removed)
    assert "Tests or a strong no-test rationale are present" in text


def test_review_template_contains_exemption_scrutiny_instruction() -> None:
    """Verify that review.md contains the exemption-scrutiny instruction (issue #180)."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    # Verify the exemption-scrutiny instruction is present
    assert "If a `Test-exempt:` reason is present above, treat it as a claim to verify" in text
    assert "not a fact to accept" in text


def test_review_template_contains_self_report_distrust_rule() -> None:
    """Verify that review.md tells the reviewer not to trust the PR's self-report."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "## Do not trust the PR's self-report" in text
    assert "never by itself downgrades a finding's severity" in text


def test_review_template_contains_investigation_discipline() -> None:
    """Verify that review.md bounds out-of-diff investigation and forbids re-running the full suite."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "do not otherwise crawl the broader codebase" in text
    assert "Do not\nre-run the full test suite to confirm results already reported by CI" in text


def test_review_template_contains_calibration_section() -> None:
    """Verify that review.md has a severity-calibration section distinguishing Critical/Important/Minor."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "## Calibration" in text
    assert "Tag every finding Critical, Important, or Minor" in text
    assert "label it\nplan-mandated" in text
    assert "Acknowledge what was done well before listing issues" in text


def test_review_template_summary_requires_strengths_and_severity_tags() -> None:
    """Verify that review.md's required summary fields include Strengths and severity-tagged Findings."""
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "charlie_work" / "prompts"
    text = (prompts_dir / "review.md").read_text(encoding="utf-8")

    assert "Strengths — what's done well, specifically" in text
    assert "Findings, each tagged Critical / Important / Minor" in text


def test_review_rendered_with_populated_test_adequacy_section() -> None:
    """Verify that review.md renders with populated test-adequacy facts when enabled (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Create a populated TestAdequacyFacts
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=("src/foo.py", "src/bar.py"),
        exempt=False,
        exempt_reason="",
    )
    warnings = ("Zero recognized assertions in added test lines",)

    # Render the section
    section = render_test_adequacy_section(facts, warnings)

    # Verify the facts appear in the rendered output
    assert "## Test-adequacy facts (Tier 1, deterministic)" in section
    assert "Added product LOC: 100" in section
    assert "Added test LOC: 50" in section
    assert "Assertion-bearing added test lines: 10" in section
    assert "Test files changed: 2" in section
    assert "Untested product files: src/foo.py, src/bar.py" in section
    assert "Zero recognized assertions in added test lines" in section


def test_review_rendered_with_disabled_test_adequacy_section() -> None:
    """Verify that review.md renders with empty test-adequacy section when disabled (issue #180)."""
    from charlie_work.workflow import render_test_adequacy_section

    # When facts is None (gate disabled), should return empty string
    section = render_test_adequacy_section(None, ())

    assert section == "", "Disabled gate should render empty string"


def test_review_rendered_with_exempt_claim() -> None:
    """Verify that review.md renders with exempt claim when present (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Create facts with an exempt claim
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=0,
        assertion_count=0,
        test_files_changed=0,
        untested_product_files=(),
        exempt=True,
        exempt_reason="n/a - pure refactoring",
    )

    # Render the section
    section = render_test_adequacy_section(facts, ())

    # Verify the exempt claim appears
    assert 'Test-exempt claim: "n/a - pure refactoring" (verify against the diff)' in section


def test_review_rendered_with_no_untested_files() -> None:
    """Verify that review.md renders without untested files list when none exist (issue #180)."""
    from charlie_work.janitor import TestAdequacyFacts
    from charlie_work.workflow import render_test_adequacy_section

    # Create facts with no untested files
    facts = TestAdequacyFacts(
        added_product_loc=100,
        added_test_loc=50,
        assertion_count=10,
        test_files_changed=2,
        untested_product_files=(),
        exempt=False,
        exempt_reason="",
    )

    # Render the section
    section = render_test_adequacy_section(facts, ())

    # Verify the untested files line does NOT appear
    assert "Untested product files:" not in section
