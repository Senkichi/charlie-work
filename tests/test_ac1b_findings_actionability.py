"""Tests for scripts/ac1b_findings_actionability.py.

Loads the script as a module without adding scripts/ to sys.path, mirroring
tests/test_backfill_stale_rework_briefs.py's pattern for the other
standalone script.

Covers two real bugs found running the harness against the live corpus
(docs/plans/rework-findings-channel.md section 8):

1. ``derive_cross_family_collapse_sentinel`` assumed
   ``cross_family.parse_cross_family_verdict`` returns a ``(decision,
   summary)`` tuple. It actually returns a ``CrossFamilyVerdict`` dataclass
   (or ``None``) -- see cross_family.py:396-474 and
   test_charlie_work.py's ``test_parse_cross_family_verdict_*`` tests, which
   already assert attribute access (``result.decision`` / ``result.summary``).
   The tuple-shaped `isinstance` check always failed, raising RuntimeError
   and collapsing every verdict to ``UNKNOWN_provenance_unavailable``.
2. ``find_concrete_referents`` treated any bare ``identifier(`` token as a
   ``code_symbol`` referent, so ordinary prose/verification-command text
   (e.g. `python -c "import charlie_work; print(charlie_work.__file__)"`)
   scored as "actionable" purely because it called a Python builtin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from _script_loader import load_script_module
from charlie_work import cross_family
from charlie_work.config import OrchestratorConfig, RuntimeConfig
from charlie_work.paths import RuntimePaths


def _load_ac1b_script() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "ac1b_findings_actionability.py"
    return load_script_module(path, "ac1b_findings_actionability")


@pytest.fixture(scope="module")
def ac1b() -> ModuleType:
    return _load_ac1b_script()


# --------------------------------------------------------------------------
# Bug 1: sentinel derivation must handle the REAL CrossFamilyVerdict return
# shape, not a (decision, summary) tuple.
# --------------------------------------------------------------------------


def test_derive_cross_family_collapse_sentinel_matches_real_parser(ac1b: ModuleType) -> None:
    """Issue #784 rewrote the legacy fallback (this script's own
    docstring's "F5"): the live parser can no longer construct a
    content-free ``CrossFamilyVerdict`` for a BLOCKER-only report with no
    ``Verdict:`` marker -- it now returns ``MalformedCrossFamilyVerdict``
    instead (``CrossFamilyVerdict.__post_init__`` raises on that exact
    shape). The prior version of this test asserted the parser returned the
    vacuous placeholder as a real verdict -- that assertion encoded the
    defect #784 fixes, so it is replaced rather than preserved: the sentinel
    must still equal the historical constant (kept for classifying PRE-#784
    on-disk records), but derivation must prove the *live* parser has
    actually adopted the new contract rather than trusting the constant
    blindly.
    """
    sentinel = ac1b.derive_cross_family_collapse_sentinel()
    assert sentinel == "Cross-family review found BLOCKER/MAJOR findings"
    assert sentinel == cross_family.LEGACY_VACUOUS_SUMMARY

    # Cross-check directly against the parser this is meant to track: the
    # exact probe that used to produce a vacuous CrossFamilyVerdict now
    # returns a MalformedCrossFamilyVerdict, proving #784's fix is live in
    # the code under test, not just assumed.
    probe = "## Report\n\n**BLOCKER** unparseable body with no Verdict: marker\n"
    parsed = cross_family.parse_cross_family_verdict(probe)
    assert isinstance(parsed, cross_family.MalformedCrossFamilyVerdict)
    assert parsed.reason == "blocker_or_major_with_no_extractable_summary"


def test_classify_verdict_uses_derived_sentinel_for_cross_family_collapse(
    ac1b: ModuleType,
) -> None:
    """A verdict whose summary is exactly the derived sentinel classifies as
    cross_family_generic_collapse -- the category this whole script exists
    to isolate (docs/plans/rework-findings-channel.md section 8).
    """
    sentinel = ac1b.derive_cross_family_collapse_sentinel()
    assert ac1b.classify_verdict(sentinel, sentinel) == ac1b.CROSS_FAMILY_COLLAPSE
    assert ac1b.classify_verdict("CI failed on Lint; push a fix", sentinel) == (
        ac1b.SYNTHETIC_CI_FAILURE
    )
    assert ac1b.classify_verdict("Some real reviewer prose.", sentinel) == (
        ac1b.REAL_REVIEWER_PROSE
    )


# --------------------------------------------------------------------------
# Bug 2: bare builtin calls (print(, int(, str(, ...) in prose must not
# count as concrete code_symbol referents.
# --------------------------------------------------------------------------


def test_verification_command_prose_is_not_actionable(ac1b: ModuleType) -> None:
    """The exact false-positive shape found in the live corpus (pr-182,
    pr-187, pr-188, pr-190): a verification-command snippet whose only
    identifier-shaped token is a bare `print(` call. This must not count
    as a reviewer naming a symbol to change.
    """
    text = 'python -c "import charlie_work; print(charlie_work.__file__)"'
    referents = ac1b.find_concrete_referents(text)
    assert referents == []
    assert ac1b.is_actionable(text) is False


def test_bare_int_call_is_not_actionable(ac1b: ModuleType) -> None:
    """The pr-500 false-positive shape: an error-message snippet whose only
    identifier-shaped token is a bare `int(` call.
    """
    text = "ValueError invalid literal for int() with base 10: 'two'"
    referents = ac1b.find_concrete_referents(text)
    assert referents == []
    assert ac1b.is_actionable(text) is False


def test_bare_non_builtin_call_still_counts(ac1b: ModuleType) -> None:
    """A bare call to a real project symbol (not a Python builtin) must
    still be flagged -- the fix narrows to builtins specifically, it does
    not disable the bare-call referent shape altogether.
    """
    text = "the fix should call _is_review_dispatchable(pr) before dispatching"
    referents = ac1b.find_concrete_referents(text)
    assert ("code_symbol", "_is_review_dispatchable(") in referents
    assert ac1b.is_actionable(text) is True


def test_backtick_quoted_symbols_unaffected_by_builtin_filter(ac1b: ModuleType) -> None:
    """Backtick-quoted symbols (the legitimate signal the task calls out to
    keep) are a separate regex alternative with no `(` in its character
    class, so they are untouched by the builtin-call filter either way.
    """
    text = "See `_diff_content_signature` at src/charlie_work/workflow.py:3700"
    referents = ac1b.find_concrete_referents(text)
    assert ("code_symbol", "`_diff_content_signature`") in referents
    assert ("file_path", "src/charlie_work/workflow.py") in referents
    assert ("line_number", ":3700") in referents
    assert ac1b.is_actionable(text) is True


def test_mixed_builtin_and_real_referent_keeps_only_the_real_one(ac1b: ModuleType) -> None:
    """A verdict mentioning both a bare builtin call and a real file path
    (the pr-182/pr-188 shape) stays actionable, but the builtin call is
    dropped from the referent list -- the fix removes noise without
    flipping an already-correct classification.
    """
    text = 'python -c "import charlie_work; print(charlie_work.__file__)"\ntests/test_worker.py'
    referents = ac1b.find_concrete_referents(text)
    assert ("code_symbol", "print(") not in referents
    assert ("file_path", "tests/test_worker.py") in referents
    assert ac1b.is_actionable(text) is True


# --------------------------------------------------------------------------
# Review feedback on PR #1076:
# - The baseline doc must use F1's deploy time as the pre/post split.
# - carried_forward_from is a list of head SHAs, never a boolean.
# - The script must not keep emitting a 'proj. post-F1 AC-1b' column on
#   post-F1 runs while the doc claims there is 'no projection left to show'.
# --------------------------------------------------------------------------


def test_renderer_has_f1_summary_fallback_true(ac1b: ModuleType) -> None:
    """The real renderer in this checkout has F1's summary fallback, so the
    diagnostic projection is suppressed.
    """
    assert ac1b._renderer_has_f1_summary_fallback() is True


def test_project_f1_rendering_fences_summary_when_required_changes_empty(
    ac1b: ModuleType,
) -> None:
    """The pre-F1 stand-in fences the reviewer summary when the structured
    list is empty, producing a scoreable body with concrete referents.
    """
    decision = {
        "decision": "request_changes",
        "required_changes": [],
        "summary": "Fix `foo()` in src/bar.py:10.",
    }
    rendered = ac1b.project_f1_rendering(decision)
    assert rendered.startswith("## Required changes (fallback: reviewer summary)")
    body = ac1b.extract_scoreable_body(rendered)
    referents = ac1b.find_concrete_referents(body)
    assert any(kind == "file_path" and "src/bar.py" in value for kind, value in referents)
    assert any(kind == "code_symbol" and "foo" in value for kind, value in referents)


def _make_fake_repo(tmp_path: Path) -> Path:
    prs_dir = tmp_path / ".var" / "charlie-work" / "prs" / "pr-1"
    prs_dir.mkdir(parents=True)
    (prs_dir / "review-decision.json").write_text(
        json.dumps(
            {
                "decision": "request_changes",
                "summary": "See src/charlie_work/workflow.py:1 for details.",
                "required_changes": [],
            }
        ),
        encoding="utf-8",
    )
    return prs_dir.parent


def _fake_runtime_paths(root: Path, state_dir: str) -> RuntimePaths:
    resolved = root / state_dir
    return RuntimePaths(
        root=resolved,
        issues=resolved / "issues",
        prs=resolved / "prs",
        dispatches=resolved / "dispatches",
        logs=resolved / "logs",
        state_file=resolved / "state.json",
        worktrees=resolved / "worktrees",
        cross_family=resolved / "cross_family",
    )


def _fake_repo_root(path: Path, *, explicit: bool) -> Path:  # noqa: ARG001
    return path


def _fake_config(_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(runtime=RuntimeConfig(state_dir=".var/charlie-work"))


def _fake_code_sha() -> str:
    return "1234567890abcdef1234567890abcdef12345678"


def test_main_omits_projection_column_on_post_f1(
    ac1b: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """On a post-F1 checkout, the script detects the real F1 renderer and
    omits the diagnostic 'proj. post-F1 AC-1b' column. The review found this
    column was still being printed despite the baseline claiming there was
    'no projection left to show'.
    """
    _make_fake_repo(tmp_path)
    monkeypatch.setattr(ac1b, "find_repo_root", _fake_repo_root)
    monkeypatch.setattr(ac1b, "load_layered_config", _fake_config)
    monkeypatch.setattr(ac1b, "resolve_code_sha", _fake_code_sha)
    monkeypatch.setattr(ac1b, "runtime_paths", _fake_runtime_paths)
    monkeypatch.setattr(sys, "argv", ["ac1b_findings_actionability", "--repo", str(tmp_path)])

    rc = ac1b.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "proj. post-F1 AC-1b" not in out
    assert "proj_AC1b" not in out
    assert "AC-1b (actionable)" in out
    assert "suppressed (real F1 renderer in use)" in out


def test_main_shows_projection_column_on_pre_f1(
    ac1b: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """On a pre-F1 checkout, the script keeps the diagnostic projection
    column so the pre/post comparison is still visible.
    """
    _make_fake_repo(tmp_path)
    real_renderer = ac1b._render_required_changes_section

    def _pre_f1_renderer(decision: dict[str, object] | None) -> str:
        # Pre-F1: only render when required_changes is non-empty.
        if not isinstance(decision, dict):
            return ""
        changes = decision.get("required_changes")
        if isinstance(changes, list) and changes:
            return real_renderer(decision)
        return ""

    monkeypatch.setattr(ac1b, "_render_required_changes_section", _pre_f1_renderer)
    monkeypatch.setattr(ac1b, "find_repo_root", _fake_repo_root)
    monkeypatch.setattr(ac1b, "load_layered_config", _fake_config)
    monkeypatch.setattr(ac1b, "resolve_code_sha", _fake_code_sha)
    monkeypatch.setattr(ac1b, "runtime_paths", _fake_runtime_paths)
    monkeypatch.setattr(sys, "argv", ["ac1b_findings_actionability", "--repo", str(tmp_path)])

    rc = ac1b.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "proj. post-F1 AC-1b" in out
    assert "shown (pre-F1 renderer; local stand-in)" in out


def test_ac1b_baseline_markdown_split_and_schema() -> None:
    """The baseline doc must use F1's deploy time as the pre/post split, not
    the merge/land time, and must not mis-state ``carried_forward_from`` as a
    boolean.
    """
    baseline_path = Path(__file__).parent.parent / "docs" / "plans" / "ac1b-baseline.md"
    text = baseline_path.read_text(encoding="utf-8")
    pre_f1_section, _, _ = text.partition("## Pre-F1 baseline")

    assert "2026-07-31T02:06" in pre_f1_section
    assert "reviewed_at < 2026-07-31T02:29:42Z" not in text
    assert "carried_forward_from=False" not in text
    assert "carried_forward_from=[]" in text
