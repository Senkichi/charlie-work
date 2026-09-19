"""Review-effort experiment helpers: ``_review_effort_arm``
determinism/distribution and ``resolve_review_effort`` treatment split.

Split out of ``tests/test_claude_code_adapter.py`` (issue #1560,
Track 1) -- bodies are verbatim relocations; shared helpers live in
``tests/_claude_adapter_fixtures.py``.
"""

from __future__ import annotations

from charlie_work.config import (
    ClaudeCodeConfig,
    ReviewerRoleConfig,
)
from charlie_work.claude_code import (
    resolve_review_effort,
    _review_effort_arm,
)


def test_review_effort_arm_is_deterministic() -> None:
    """Same inputs must always yield the same arm (stable across re-dispatches)."""
    for pr_number in (1, 42, 999, 123456):
        first = _review_effort_arm(pr_number, 0.5, "salt")
        second = _review_effort_arm(pr_number, 0.5, "salt")
        assert first == second


def test_review_effort_arm_fraction_zero_never_treatment() -> None:
    for pr_number in range(1, 200):
        assert _review_effort_arm(pr_number, 0.0, "") is False


def test_review_effort_arm_fraction_one_always_treatment() -> None:
    for pr_number in range(1, 200):
        assert _review_effort_arm(pr_number, 1.0, "") is True


def test_review_effort_arm_salt_change_flips_some_assignments() -> None:
    """Changing the salt re-randomizes arm assignment for a new epoch."""
    prs = range(1, 500)
    arms_a = {pr: _review_effort_arm(pr, 0.5, "epoch-1") for pr in prs}
    arms_b = {pr: _review_effort_arm(pr, 0.5, "epoch-2") for pr in prs}
    flipped = sum(1 for pr in prs if arms_a[pr] != arms_b[pr])
    assert flipped > 0


def test_review_effort_arm_distribution_sanity() -> None:
    """Over many sequential PR numbers, treatment share should land near
    the configured fraction (loose band to avoid test flakiness)."""
    fraction = 0.5
    prs = range(1, 1001)
    treatment_count = sum(1 for pr in prs if _review_effort_arm(pr, fraction, "sanity-salt"))
    share = treatment_count / len(prs)
    assert 0.4 < share < 0.6


def test_resolve_review_effort_disabled_uses_review_effort_unconditionally() -> None:
    """fraction<=0.0 (default): reviewer.effort, if set, applies to every PR
    --- exactly the pre-experiment behavior. arm is None (experiment not
    running)."""
    reviewer = ReviewerRoleConfig(effort="high")
    claude_code_cfg = ClaudeCodeConfig(effort="low")
    for pr_number in (1, 2, 3, 4, 5):
        effort, arm = resolve_review_effort(pr_number, reviewer, claude_code_cfg)
        assert effort == "high"
        assert arm is None


def test_resolve_review_effort_disabled_falls_back_when_review_effort_unset() -> None:
    reviewer = ReviewerRoleConfig(effort="")
    claude_code_cfg = ClaudeCodeConfig(effort="medium")
    effort, arm = resolve_review_effort(101, reviewer, claude_code_cfg)
    assert effort == "medium"
    assert arm is None


def test_resolve_review_effort_enabled_splits_treatment_and_control() -> None:
    """fraction=1.0: every PR is treatment and gets reviewer.effort.
    fraction=0.0-adjacent control case is exercised via a PR known to hash to
    False for a tiny fraction."""
    reviewer = ReviewerRoleConfig(effort="high", effort_experiment_fraction=1.0)
    claude_code_cfg = ClaudeCodeConfig(effort="low")
    effort, arm = resolve_review_effort(777, reviewer, claude_code_cfg)
    assert (effort, arm) == ("high", "treatment")

    # A vanishingly small fraction (but > 0.0, so the experiment IS enabled)
    # makes control the overwhelmingly likely outcome for an arbitrary PR;
    # assert against the deterministic arm function directly instead of
    # relying on probability for a single PR.
    tiny_fraction = 1e-9
    salt = ""
    is_treatment = _review_effort_arm(777, tiny_fraction, salt)
    reviewer_tiny = ReviewerRoleConfig(effort="high", effort_experiment_fraction=tiny_fraction)
    effort, arm = resolve_review_effort(777, reviewer_tiny, claude_code_cfg)
    if is_treatment:
        assert (effort, arm) == ("high", "treatment")
    else:
        assert (effort, arm) == ("low", "control")
