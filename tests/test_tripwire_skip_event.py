"""Tests for issue #937: a blinded unauthorized-merge tripwire pass must not be silent.

``_detect_unauthorized_merges`` fails open when ``gh.merged_pr_list()`` raises
``GitHubError``: it returns ``[]`` without arming the baseline. Before this
change, that fail-open path was also fail-silent -- the pass reported
``ok=True`` and was byte-identical in every stored artifact to a pass where
the tripwire actually ran and found nothing, discarding the distinction #633
deliberately created when it made ``merged_pr_list()`` raise instead of
coercing a bad result to ``[]``.

``OrchestratorApp._record_unauthorized_merge_skip`` closes that gap: it logs
a warning and durably records an ``unauthorized_merge_check_skipped`` event
so a blinded pass is distinguishable from a clean one.

Reuses the tripwire fixtures from ``test_charlie_work.py``
(``FakeGitHub``, ``_arm_unauthorized_merge_tripwire``, ``_merged_worker_pr``)
per this suite's existing convention (see ``test_tripwire_detection_record.py``).
"""

from __future__ import annotations

from pathlib import Path

from charlie_work import github as github_module
from charlie_work.config import OrchestratorConfig
from charlie_work.instrumentation import _classify_level
from charlie_work.paths import runtime_paths
from charlie_work.state import load_state
from charlie_work.workflow import OrchestratorApp, UNAUTHORIZED_MERGE_BASELINE_KEY

from _fakes_github import FakeGitHub
from _merge_tripwire_fixtures import _arm_unauthorized_merge_tripwire, _merged_worker_pr


def _make_app(tmp_path: Path, fake_gh: FakeGitHub, **kwargs) -> tuple[OrchestratorApp, object]:
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh, **kwargs)
    return app, paths


class FakeGitHubFailingMergedList(FakeGitHub):
    """A gh that always fails to list merged PRs, e.g. a transient outage."""

    def __init__(self, message: str = "gh unavailable") -> None:
        super().__init__()
        self._message = message

    def merged_pr_list(self):
        raise github_module.GitHubError(self._message)


def test_detect_unauthorized_merges_returns_empty_and_records_one_skip_event(
    tmp_path: Path,
) -> None:
    """A raising ``merged_pr_list`` must fail open (``[]``) AND leave an audit trail.

    Before #937, this path returned ``[]`` silently -- the stored state was
    indistinguishable from a pass where the tripwire ran cleanly and found
    nothing. The event is what makes the blinded pass observable.
    """
    fake_gh = FakeGitHubFailingMergedList()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    result = app._detect_unauthorized_merges(None)

    assert result == []
    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_check_skipped"]
    assert len(events) == 1, (
        f"expected exactly one unauthorized_merge_check_skipped event, got {len(events)}"
    )


def test_skip_event_payload_carries_exception_text_and_class_name(tmp_path: Path) -> None:
    fake_gh = FakeGitHubFailingMergedList("gh: command not found")
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    app._detect_unauthorized_merges(None)

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_check_skipped"]
    payload = events[0]["payload"]
    assert payload["reason"] == "gh: command not found"
    assert payload["error_type"] == "GitHubError"


def test_classify_level_check_skipped_is_warning_contrasted_with_detected_error() -> None:
    """``unauthorized_merge_check_skipped`` is a degraded-but-not-broken control (warning).

    Pinned alongside ``unauthorized_merge_detected`` (error) so the two events'
    distinct severities -- "the control didn't run" vs. "the control found a
    real bypass" -- can't silently collapse to the same level in a refactor.
    """
    assert _classify_level("unauthorized_merge_check_skipped") == "warning"
    assert _classify_level("unauthorized_merge_detected") == "error"


def test_skip_event_fires_once_per_pass_not_deduped_across_passes(tmp_path: Path) -> None:
    """Unlike ``unauthorized_merge_detected``, this event is NOT deduped across calls.

    A finding is one fact that stays true across passes, so
    ``unauthorized_merge_detected`` intentionally fires once per PR (see
    ``UNAUTHORIZED_MERGE_DETECTED_KEY``). A skipped check is different: each
    call is a distinct occurrence -- a distinct window in which an
    unauthorized merge could have landed unseen -- so collapsing repeat
    occurrences into one event would destroy exactly the count that makes the
    blind spot measurable. A refactor that "helpfully" dedupes this event the
    way the detected-event is deduped would break this invariant silently;
    this test exists to catch that.
    """
    fake_gh = FakeGitHubFailingMergedList()
    app, paths = _make_app(tmp_path, fake_gh)
    _arm_unauthorized_merge_tripwire(paths)

    app._detect_unauthorized_merges(None)
    app._detect_unauthorized_merges(None)
    app._detect_unauthorized_merges(None)

    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_check_skipped"]
    assert len(events) == 3, (
        f"expected 3 unauthorized_merge_check_skipped events across 3 raising "
        f"calls (once per pass, not deduped), got {len(events)}"
    )


def test_skip_event_writes_nothing_in_dry_run(tmp_path: Path) -> None:
    fake_gh = FakeGitHubFailingMergedList()
    app, paths = _make_app(tmp_path, fake_gh, dry_run=True)
    _arm_unauthorized_merge_tripwire(paths)

    result = app._detect_unauthorized_merges(None)

    assert result == []
    state = load_state(paths.state_file)
    events = [e for e in state["events"] if e["kind"] == "unauthorized_merge_check_skipped"]
    assert events == [], "dry-run must not emit unauthorized_merge_check_skipped"
    assert "events" not in state or events == []


def test_skip_path_does_not_arm_the_baseline(tmp_path: Path) -> None:
    """A raising fetch must leave the tripwire unarmed, same as before #937.

    ``_record_unauthorized_merge_skip`` only adds an audit event -- it must
    never touch ``UNAUTHORIZED_MERGE_BASELINE_KEY``. If it did (or if some
    future refactor folded the skip-recording into the baseline-arming
    write), a gh outage could bake an empty baseline and permanently exempt
    every merge that happened before gh recovered -- the exact failure
    ``_apply_unauthorized_merge_baseline``'s docstring calls "the sharpest
    failure mode of the whole mechanism".
    """
    fake_gh = FakeGitHubFailingMergedList()
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.ensure()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)
    # Deliberately NOT armed: this test exercises the very-first-pass case,
    # where a baked empty baseline would be most damaging.

    app._detect_unauthorized_merges(None)

    state = load_state(paths.state_file)
    assert UNAUTHORIZED_MERGE_BASELINE_KEY not in state, (
        "a skipped check must never arm the baseline -- that would exempt real "
        "history the control never actually saw"
    )

    # Confirm recovery still works cleanly once gh is healthy again, showing
    # the skip path left the tripwire in its normal pre-arming state.
    app.gh = FakeGitHub()  # type: ignore[assignment]
    assert app._detect_unauthorized_merges([_merged_worker_pr(101, 91, "sha-101")]) == []
    assert load_state(paths.state_file)[UNAUTHORIZED_MERGE_BASELINE_KEY]["pre_existing_prs"] == [
        101
    ]
