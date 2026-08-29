"""Tests for issue #944: unfiltered backlog reachability observation.

``workflow.classify_backlog_reachability`` runs one unfiltered
``gh.issue_list(state="open")`` and bins every open issue by the first
``_is_dispatchable`` arm that would reject it, so a zero-dispatch pass comes
with a reason instead of an indistinguishable-from-healthy silence.
``cli._render_backlog_reachability`` turns that dict into a one-line status
suffix. See both docstrings for the design rationale this file locks in.

All label membership is derived from ``config.labels.*`` (never a literal
string) per CLAUDE.md's "label state-machine names come from LabelConfig"
invariant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from charlie_work import cli
from charlie_work.config import OrchestratorConfig
from charlie_work.workflow import (
    _MergedPRListOutcome,
    classify_backlog_reachability,
    resolve_dispatch_mention_coverage,
)


class FakeGh:
    """Minimal stub for the GitHubLike methods classify_backlog_reachability
    calls: issue_list(state=...) and, since issue #1110, the blocker-check
    surface (are_issues_open + the dependency cache that
    get_github_issue_dependencies reads). Never touches the network."""

    def __init__(
        self,
        issues: list[dict[str, Any]],
        *,
        open_blockers: set[int] | None = None,
        dependencies: dict[int, list[int]] | None = None,
    ) -> None:
        self._issues = issues
        self.calls: list[dict[str, Any]] = []
        # _list_cache is the pass-scoped cache the real GitHubClient uses;
        # get_github_issue_dependencies checks it before calling gh.run, so
        # pre-seeding it simulates GitHub-native dependencies without a network
        # call. Issues not in ``dependencies`` fall through to gh.run -> None
        # -> [] (fail-open, matching production).
        self._list_cache: dict[tuple[str, Any], Any] = {}
        for number, deps in (dependencies or {}).items():
            self._list_cache[("issue_dependencies", number)] = deps
        self._open_blockers = open_blockers or set()

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        self.calls.append({"labels": labels, "state": state})
        return list(self._issues)

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        return {n for n in issue_numbers if n in self._open_blockers}

    def run(
        self, args: list[str], *, json_output: bool = False, allow_failure: bool = False
    ) -> Any:
        # Simulate "GitHub-native dependencies feature not available" —
        # get_github_issue_dependencies treats None as a transient failure and
        # returns [] (fail-open). Per-issue deps are injected via the cache
        # instead, so this is only hit for issues without explicit deps.
        return None


def _issue(number: int, names: set[str], body: str = "") -> dict[str, Any]:
    return {"number": number, "labels": [{"name": n} for n in names], "body": body}


# ---------------------------------------------------------------------------
# 1. The empty-fetch ambiguity -- the most important test.
# ---------------------------------------------------------------------------


def test_empty_fetch_reports_not_observed_not_empty_backlog() -> None:
    # GitHubClient._list_json returns [] both for "genuinely no open issues"
    # and for a failed `gh` call (non-list result coerced to []). If this
    # function reported open_total=0/dispatchable=0 for an empty fetch, that
    # would render identically to "all healthy" and silently reintroduce the
    # exact four-day dispatch-stall bug (#944) this function exists to catch:
    # a failed/ambiguous fetch must never be indistinguishable from "no work".
    gh = FakeGh([])
    config = OrchestratorConfig()

    result = classify_backlog_reachability(gh, config)

    assert result["observed"] is False
    # dispatchable/open_total values in an unobserved result must never be
    # trusted or read as "0 open, healthy" by a caller.
    assert result["dispatchable"] == 0
    assert result["open_total"] == 0
    # `consistent` is tri-state (None/True/False). On this path the
    # cross-check never ran at all -- the fetch itself failed/was empty
    # before ready_open_count could even be compared. It must read as
    # "unverified" (None), never default to True ("verified"): a reassuring
    # default standing in for an unrun check is this exact bug (an unknown
    # state silently rendered as healthy) one field over from `observed`.
    assert result["consistent"] is None

    rendered = cli._render_backlog_reachability(result)
    assert rendered == "  [backlog not observed]"
    # Must not say anything implying a healthy/empty backlog.
    assert "0 open" not in rendered
    assert "healthy" not in rendered.lower()
    assert rendered.isascii()


# ---------------------------------------------------------------------------
# 2 & 3. Each exclusion arm bins independently, in precedence order; terminal
# and active are distinct reason codes even though both exclude.
# ---------------------------------------------------------------------------


def test_exclusion_arms_bin_independently_in_precedence_order() -> None:
    config = OrchestratorConfig()
    labels = config.labels

    # Confirm the config-derived precondition this test relies on: human_needed
    # is terminal, not active. If LabelConfig ever moved it, this test should
    # fail loudly here rather than silently mis-assert reason codes below.
    assert labels.human_needed in labels.terminal
    assert labels.human_needed not in labels.active
    assert labels.in_progress in labels.active
    assert labels.in_progress not in labels.terminal

    issue_missing_ready = _issue(1, set())
    issue_terminal = _issue(2, {labels.ready, labels.human_needed})
    issue_active = _issue(3, {labels.ready, labels.in_progress})
    issue_claimed = _issue(4, {labels.ready})
    issue_dispatchable = _issue(5, {labels.ready})

    gh = FakeGh(
        [
            issue_missing_ready,
            issue_terminal,
            issue_active,
            issue_claimed,
            issue_dispatchable,
        ]
    )

    result = classify_backlog_reachability(gh, config, operator_claimed={4})

    assert result["observed"] is True
    assert result["open_total"] == 5
    assert result["missing_ready"] == 1
    assert result["terminal_label"] == 1
    assert result["active_label"] == 1
    assert result["operator_claimed"] == 1
    assert result["dispatchable"] == 1
    # Every open issue lands in exactly one bin -- the counts partition
    # open_total, they don't just happen to sum to it by coincidence here.
    assert (
        result["missing_ready"]
        + result["terminal_label"]
        + result["active_label"]
        + result["operator_claimed"]
        + result["blocked_by_open_dependency"]
        + result["mention_covered_awaiting_operator"]
        + result["dispatchable"]
        == result["open_total"]
    )

    examples = result["unreachable_examples"]
    assert examples["missing_ready"] == [1]
    assert examples["terminal_label"] == [2]
    assert examples["active_label"] == [3]
    assert examples["operator_claimed"] == [4]
    # dispatchable issues are never recorded as "unreachable" examples.
    assert "dispatchable" not in examples


def test_terminal_label_takes_precedence_over_active_when_both_present() -> None:
    # An issue carrying both a terminal and an active label (shouldn't happen
    # in practice, but _is_dispatchable's precedence order must still be
    # locked in) bins as terminal_label -- the terminal check runs first.
    config = OrchestratorConfig()
    labels = config.labels
    issue = _issue(7, {labels.ready, labels.human_needed, labels.in_progress})
    gh = FakeGh([issue])

    result = classify_backlog_reachability(gh, config)

    assert result["terminal_label"] == 1
    assert result["active_label"] == 0


def test_operator_claimed_only_applies_after_terminal_and_active_checks() -> None:
    # An issue that is both terminal-labelled AND operator-claimed still bins
    # as terminal_label, matching _is_dispatchable's arm order.
    config = OrchestratorConfig()
    labels = config.labels
    issue = _issue(9, {labels.ready, labels.human_needed})
    gh = FakeGh([issue])

    result = classify_backlog_reachability(gh, config, operator_claimed={9})

    assert result["terminal_label"] == 1
    assert result["operator_claimed"] == 0


# ---------------------------------------------------------------------------
# 4. ready_open_count superset cross-check.
# ---------------------------------------------------------------------------


def test_ready_open_count_consistent_when_unfiltered_list_is_a_superset() -> None:
    config = OrchestratorConfig()
    labels = config.labels
    issues = [
        _issue(1, {labels.ready}),
        _issue(2, {labels.ready}),
        _issue(3, set()),  # missing ready, doesn't count toward ready_seen
    ]
    gh = FakeGh(issues)

    # Caller's ready-filtered query saw exactly the 2 ready-labelled issues.
    result = classify_backlog_reachability(gh, config, ready_open_count=2)

    assert result["consistent"] is True


def test_ready_open_count_inconsistent_when_unfiltered_list_is_missing_issues() -> None:
    # The unfiltered fetch found fewer ready-labelled issues than the caller's
    # own ready-filtered query already saw -- the unfiltered list cannot be a
    # superset, so the fetch itself is unreliable and must be flagged rather
    # than silently trusted.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1, {labels.ready})]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, ready_open_count=5)

    assert result["consistent"] is False


def test_ready_open_count_omitted_leaves_consistent_unrun() -> None:
    # No ready_open_count -> the cross-check never runs. `consistent` must
    # stay None (unverified), not fall back to True, even though the fetch
    # itself succeeded (observed=True) and has nothing wrong with it.
    config = OrchestratorConfig()
    labels = config.labels
    gh = FakeGh([_issue(1, {labels.ready})])

    result = classify_backlog_reachability(gh, config)

    assert result["observed"] is True
    assert result["consistent"] is None


# ---------------------------------------------------------------------------
# 5 & 6. Renderer stays silent when work is flowing; fires when it isn't.
# ---------------------------------------------------------------------------


def test_renderer_silent_when_dispatchable_present() -> None:
    # NOTE on scope: the renderer's real contract is narrower than "silent
    # whenever dispatchable > 0" -- it is "silent when observed AND
    # consistent is not False AND dispatchable > 0" (see
    # test_renderer_reports_inconsistent_backlog_fetch below, which has
    # dispatchable == 1 but a non-empty render because consistent is False).
    # A broken/unreliable fetch must still speak even if it happens to show
    # some dispatchable issues. This test covers the no-cross-check path,
    # where consistent is None (not True -- see the tri-state note above).
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1, {labels.ready}), _issue(2, set())]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)
    assert result["dispatchable"] == 1
    assert result["consistent"] is None

    rendered = cli._render_backlog_reachability(result)
    # A healthy repo's status line must be unchanged -- guards against alert
    # fatigue from a line that fires on every pass.
    assert rendered == ""


def test_renderer_silent_when_consistent_true_and_dispatchable_present() -> None:
    # The plain reading of requirement 5: the cross-check RAN, passed
    # (consistent is True, not just None/unrun), and work is flowing. This is
    # the actual "healthy repo" case, distinct from the no-cross-check-ran
    # case covered above.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1, {labels.ready}), _issue(2, set())]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, ready_open_count=1)
    assert result["consistent"] is True
    assert result["dispatchable"] == 1

    assert cli._render_backlog_reachability(result) == ""


def test_renderer_silent_for_consistent_none_even_with_dispatchable() -> None:
    # Constructed dict, independent of classify_backlog_reachability's own
    # wiring: `consistent: None` (unrun cross-check) must never render as
    # "INCONSISTENT" -- the renderer checks `is False` deliberately, not
    # falsiness, so None can't be mistaken for a failed check.
    reachability = {
        "observed": True,
        "consistent": None,
        "open_total": 3,
        "dispatchable": 2,
        "missing_ready": 1,
        "terminal_label": 0,
        "active_label": 0,
        "operator_claimed": 0,
        "unreachable_examples": {},
    }

    assert cli._render_backlog_reachability(reachability) == ""


def test_renderer_fires_when_open_total_positive_and_dispatchable_zero() -> None:
    config = OrchestratorConfig()
    labels = config.labels
    issues = [
        _issue(1, set()),  # missing_ready
        _issue(2, {labels.ready, labels.human_needed}),  # terminal_label
    ]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)
    assert result["open_total"] == 2
    assert result["dispatchable"] == 0

    rendered = cli._render_backlog_reachability(result)
    assert rendered != ""
    assert "2 open" in rendered
    assert "0 dispatchable" in rendered
    assert "missing_ready=1" in rendered
    assert "terminal_label=1" in rendered
    assert rendered.isascii()


def test_renderer_reports_inconsistent_backlog_fetch() -> None:
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1, {labels.ready})]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, ready_open_count=99)
    assert result["consistent"] is False

    rendered = cli._render_backlog_reachability(result)
    assert "INCONSISTENT" in rendered
    assert rendered.isascii()


# ---------------------------------------------------------------------------
# 7. ASCII-only output (console codepage is cp437).
# ---------------------------------------------------------------------------


def test_all_renderer_outputs_are_ascii() -> None:
    config = OrchestratorConfig()
    labels = config.labels

    not_observed = cli._render_backlog_reachability({"observed": False})
    healthy = cli._render_backlog_reachability(
        classify_backlog_reachability(FakeGh([_issue(1, {labels.ready})]), config)
    )
    unreachable = cli._render_backlog_reachability(
        classify_backlog_reachability(FakeGh([_issue(1, set())]), config)
    )
    not_a_dict = cli._render_backlog_reachability(None)

    for rendered in (not_observed, healthy, unreachable, not_a_dict):
        assert rendered.isascii()


# ---------------------------------------------------------------------------
# 8. unreachable_examples is capped (max 5 per reason) and sorted.
# ---------------------------------------------------------------------------


def test_unreachable_examples_capped_at_five_and_sorted() -> None:
    config = OrchestratorConfig()
    # Deliberately out of numeric order, and more than the cap, so a cap bug
    # (e.g. capping post-sort instead of at collection time) would be caught:
    # the first 5 encountered in iteration order are 30, 10, 50, 20, 40 --
    # 60 and 5 arrive after the cap and must be dropped, not swapped in.
    numbers_in_order = [30, 10, 50, 20, 40, 60, 5]
    issues = [_issue(n, set()) for n in numbers_in_order]  # all missing_ready
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)

    assert result["missing_ready"] == 7
    examples = result["unreachable_examples"]["missing_ready"]
    assert len(examples) == 5
    assert examples == sorted(examples)
    assert examples == [10, 20, 30, 40, 50]
    assert 60 not in examples
    assert 5 not in examples


# ---------------------------------------------------------------------------
# 9. A number-less issue is BINNED, not dropped. Without this the bins can sum
#    to less than open_total, and a backlog made entirely of such issues makes
#    the renderer fire with an empty reason list: "N open, 0 dispatchable ()".
#    An alarm that names no cause is the failure this whole module exists to
#    prevent, one layer up.
# ---------------------------------------------------------------------------


def test_issue_without_number_is_binned_as_unidentified_not_dropped() -> None:
    config = OrchestratorConfig()
    gh = FakeGh(
        [
            _issue(1, {config.labels.ready}),
            {"labels": [{"name": config.labels.ready}]},  # no "number" key
            {"number": None, "labels": []},  # explicit null
        ]
    )

    result = classify_backlog_reachability(gh, config)  # type: ignore[arg-type]

    assert result["unidentified"] == 2
    assert result["dispatchable"] == 1
    # The bins must PARTITION the backlog: every fetched issue lands in exactly
    # one of them, so they sum to open_total with nothing unaccounted for.
    binned = sum(
        int(result[reason])
        for reason in (
            "missing_ready",
            "terminal_label",
            "active_label",
            "operator_claimed",
            "blocked_by_open_dependency",
            "mention_covered_awaiting_operator",
            "unidentified",
            "dispatchable",
        )
    )
    assert binned == result["open_total"] == 3
    # A number-less issue cannot be named as an example -- it has no identity.
    assert all(n is not None for names in result["unreachable_examples"].values() for n in names)


def test_renderer_names_unidentified_and_never_fires_with_empty_reasons() -> None:
    config = OrchestratorConfig()
    gh = FakeGh([{"labels": []}, {"number": None, "labels": []}])

    result = classify_backlog_reachability(gh, config)  # type: ignore[arg-type]
    rendered = cli._render_backlog_reachability(result)

    assert result["open_total"] == 2
    assert result["unidentified"] == 2
    assert "0 dispatchable" in rendered
    assert "unidentified=2" in rendered
    # The specific regression: an alarm that fires naming no cause at all.
    assert "()" not in rendered
    assert "no reason recorded" not in rendered
    assert rendered.isascii()


# ---------------------------------------------------------------------------
# 10. Issue #1110: dependency-blocked issues must bin as
#     ``blocked_by_open_dependency``, not ``dispatchable``. The dispatch path
#     applies a dependency gate (_filter_blocked_issues) that the label-only
#     _is_dispatchable check does not model. Without this bin, a deliberately
#     sequenced cohort tail (e.g. #887/#888 blocked by an open #886) was counted
#     ``dispatchable`` by reachability while being permanently unselectable by
#     dispatch, producing false dispatch_stale alarms.
# ---------------------------------------------------------------------------


def test_dependency_blocked_issue_bins_as_blocked_not_dispatchable() -> None:
    # The exact observed scenario from issue #1110: #887 and #888 are
    # automated-ready, no agent: label, but both bodies declare "Blocked by
    # #886" and #886 is open. Dispatch correctly skips them; reachability
    # must not count them as dispatchable.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [
        _issue(887, {labels.ready}, body="Blocked by #886"),
        _issue(888, {labels.ready}, body="Blocked by #886"),
    ]
    gh = FakeGh(issues, open_blockers={886})

    result = classify_backlog_reachability(gh, config)

    assert result["observed"] is True
    assert result["open_total"] == 2
    assert result["dispatchable"] == 0
    assert result["blocked_by_open_dependency"] == 2
    assert result["unreachable_examples"]["blocked_by_open_dependency"] == [887, 888]
    # The bins still partition the backlog.
    assert (
        result["missing_ready"]
        + result["terminal_label"]
        + result["active_label"]
        + result["operator_claimed"]
        + result["blocked_by_open_dependency"]
        + result["mention_covered_awaiting_operator"]
        + result["dispatchable"]
        == result["open_total"]
    )


def test_closed_blocker_issue_still_dispatchable() -> None:
    # An issue whose declared blocker is CLOSED is not blocked -- the
    # dependency gate passes it. It must bin as ``dispatchable``, not
    # ``blocked_by_open_dependency``. This is the counterpart to the test
    # above: the blocker check must consult open state, not just presence of
    # a blocker declaration.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(900, {labels.ready}, body="Blocked by #886")]
    # 886 is NOT in open_blockers -> are_issues_open returns empty set.
    gh = FakeGh(issues, open_blockers=set())

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 1
    assert result["blocked_by_open_dependency"] == 0


def test_mixed_blocked_and_unblocked_ready_issues() -> None:
    # A backlog with one blocked and one unblocked ready issue: the unblocked
    # one bins as ``dispatchable``, the blocked one as
    # ``blocked_by_open_dependency``. This confirms the blocker check runs
    # per-issue, not as a blanket flag.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [
        _issue(887, {labels.ready}, body="Blocked by #886"),
        _issue(950, {labels.ready}),  # no blockers
    ]
    gh = FakeGh(issues, open_blockers={886})

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 1
    assert result["blocked_by_open_dependency"] == 1
    assert result["unreachable_examples"]["blocked_by_open_dependency"] == [887]


def test_github_native_dependency_also_blocks() -> None:
    # The blocker check unions body-declared blockers with GitHub-native
    # dependencies (get_github_issue_dependencies). An issue with no body
    # blocker but a GitHub-native dependency on an open issue must also bin
    # as ``blocked_by_open_dependency``.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(960, {labels.ready})]  # no body blocker
    gh = FakeGh(issues, open_blockers={886}, dependencies={960: [886]})

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 0
    assert result["blocked_by_open_dependency"] == 1


def test_renderer_names_blocked_by_open_dependency() -> None:
    # When all ready issues are dependency-blocked, dispatchable == 0 and the
    # renderer fires -- naming ``blocked_by_open_dependency`` as the cause
    # so an operator sees the real reason rather than a label-only mystery.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [
        _issue(887, {labels.ready}, body="Blocked by #886"),
        _issue(888, {labels.ready}, body="Blocked by #886"),
    ]
    gh = FakeGh(issues, open_blockers={886})

    result = classify_backlog_reachability(gh, config)
    rendered = cli._render_backlog_reachability(result)

    assert "0 dispatchable" in rendered
    assert "blocked_by_open_dependency=2" in rendered
    assert rendered.isascii()


# ---------------------------------------------------------------------------
# 11. Issue #1337: a merged-PR mention-only exclusion must bin as
#     ``mention_covered_awaiting_operator``, not ``dispatchable``. Without
#     this the classifier reported such an issue as dispatchable on every
#     heartbeat check forever while dispatch silently dropped it each pass --
#     the exact contradiction that triggered a manual investigation for #1059.
#     The ``mention_covered`` map is derived from the same
#     _merged_pr_referenced_issue_numbers + _mention_rearmed_issue_numbers
#     helpers dispatch uses (computed by the caller, not re-implemented here).
# ---------------------------------------------------------------------------


def test_mention_covered_issue_bins_as_covered_not_dispatchable() -> None:
    # The exact observed scenario from issue #1337: #1059 is automated-ready,
    # no agent: label, no open blockers, but a merged PR mentions it in free
    # text. Dispatch correctly excludes it; reachability must NOT count it as
    # dispatchable.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1059, {labels.ready})]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, mention_covered={1059: [2043]})

    assert result["observed"] is True
    assert result["open_total"] == 1
    assert result["dispatchable"] == 0
    assert result["mention_covered_awaiting_operator"] == 1
    assert result["unreachable_examples"]["mention_covered_awaiting_operator"] == [1059]
    # The reason names the mentioning PR(s).
    assert result["mention_covered_prs"] == {1059: [2043]}
    # The bins still partition the backlog.
    assert (
        result["missing_ready"]
        + result["terminal_label"]
        + result["active_label"]
        + result["operator_claimed"]
        + result["blocked_by_open_dependency"]
        + result["mention_covered_awaiting_operator"]
        + result["dispatchable"]
        == result["open_total"]
    )


def test_mention_covered_issue_with_exclusion_lifted_classifies_dispatchable() -> None:
    # Acceptance criterion 3: the same issue with the exclusion lifted (the
    # operator re-armed it by removing agent:human-needed) classifies as
    # dispatchable. The caller's _mention_coverage_map excludes re-armed
    # issues from the map, so the classifier never sees them as covered.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1059, {labels.ready})]
    gh = FakeGh(issues)

    # Empty map -- the exclusion was lifted, so the issue is NOT in the map.
    result = classify_backlog_reachability(gh, config, mention_covered={})

    assert result["dispatchable"] == 1
    assert result["mention_covered_awaiting_operator"] == 0
    assert result["mention_covered_prs"] == {}


def test_mention_covered_takes_precedence_over_dependency_gate() -> None:
    # The mention exclusion is checked BEFORE the dependency gate, mirroring
    # dispatch's filter order (label gate -> merged-PR exclusion -> dependency
    # gate). Both the dry-run and real _dispatch_impl paths apply the
    # merged_pr_issue_numbers exclusion in the candidate list comprehension
    # and only run _filter_blocked_issues on the already-filtered list, so an
    # issue that is BOTH blocked by an open dependency AND mention-covered is
    # dropped by the mention exclusion and never reaches the dependency gate.
    # The classifier bins it the same way -- ``mention_covered_awaiting_operator``
    # -- so the reason names the still-active exclusion rather than the
    # dependency gate the issue never reached.
    #
    # The planted open blocker (#886) is the mutation sentinel: under the
    # pre-fix order (dependency gate first) this issue binned as
    # ``blocked_by_open_dependency``; under the corrected order it bins as
    # ``mention_covered_awaiting_operator`` because the mention exclusion
    # fires first and the dependency-gate check never runs.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1059, {labels.ready}, body="Blocked by #886")]
    gh = FakeGh(issues, open_blockers={886})

    result = classify_backlog_reachability(gh, config, mention_covered={1059: [2043]})

    assert result["mention_covered_awaiting_operator"] == 1
    assert result["blocked_by_open_dependency"] == 0
    assert result["dispatchable"] == 0
    # The reason names the mentioning PR(s).
    assert result["mention_covered_prs"] == {1059: [2043]}
    # The bins still partition the backlog.
    assert (
        result["missing_ready"]
        + result["terminal_label"]
        + result["active_label"]
        + result["operator_claimed"]
        + result["blocked_by_open_dependency"]
        + result["mention_covered_awaiting_operator"]
        + result["dispatchable"]
        == result["open_total"]
    )


def test_mention_covered_does_not_apply_to_non_ready_issue() -> None:
    # The mention exclusion only matters for ready-labelled issues that pass
    # all the label checks and reach the dispatchable else-branch. A
    # non-ready issue in the map is still binned as ``missing_ready`` (the
    # label gate runs first and never reaches the mention check).
    config = OrchestratorConfig()
    issues = [_issue(1059, set())]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, mention_covered={1059: [2043]})

    assert result["missing_ready"] == 1
    assert result["mention_covered_awaiting_operator"] == 0
    assert result["dispatchable"] == 0


def test_mention_covered_prs_names_multiple_prs() -> None:
    # A single issue can be mentioned by multiple merged PRs. The
    # ``mention_covered_prs`` detail carries all of them, sorted.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1059, {labels.ready})]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, mention_covered={1059: [2043, 2040]})

    assert result["mention_covered_awaiting_operator"] == 1
    assert result["mention_covered_prs"] == {1059: [2040, 2043]}


def test_renderer_names_mention_covered_awaiting_operator() -> None:
    # When all ready issues are mention-covered, dispatchable == 0 and the
    # renderer fires -- naming ``mention_covered_awaiting_operator`` as the
    # cause so an operator sees the real reason rather than a
    # "dispatchable but never dispatched" mystery.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1059, {labels.ready})]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config, mention_covered={1059: [2043]})
    rendered = cli._render_backlog_reachability(result)

    assert "0 dispatchable" in rendered
    assert "mention_covered_awaiting_operator=1" in rendered
    assert rendered.isascii()


def test_mention_covered_none_defaults_to_empty_no_coverage() -> None:
    # Fail-open: a None map (the default when the caller could not compute
    # coverage, e.g. merged_pr_list failed) leaves the bin at zero and issues
    # classify as ``dispatchable`` -- matching the blocker check's fail-open
    # behaviour. The classifier is advisory and must not raise.
    config = OrchestratorConfig()
    labels = config.labels
    issues = [_issue(1059, {labels.ready})]
    gh = FakeGh(issues)

    result = classify_backlog_reachability(gh, config)

    assert result["dispatchable"] == 1
    assert result["mention_covered_awaiting_operator"] == 0
    assert result["mention_covered_prs"] == {}


# ---------------------------------------------------------------------------
# 12. Issue #1337 caller-side wiring: status() real output and
#     resolve_dispatch_mention_coverage fetch/reuse branches. The classifier
#     tests above exercise classify_backlog_reachability in isolation; these
#     exercise the actual call sites that compute the mention_covered map and
#     pass it through, so a wiring regression (wrong kwarg, missing fetch,
#     broken reuse) is caught here, not just the classifier logic.
# ---------------------------------------------------------------------------


def test_status_reports_mention_covered_issue_in_backlog_reachability(
    tmp_path: Path,
) -> None:
    """Issue #1337: ``status()`` must populate
    ``backlog_reachability['mention_covered_awaiting_operator']`` and
    ``['mention_covered_prs']`` on the real ``status()`` call -- not just the
    classifier function in isolation.

    A ready issue mentioned by a merged PR (but not bound by branch-prefix or
    closing keyword) is excluded from dispatch by ``_dispatch_impl``'s
    merged_pr_issue_numbers filter. Without the mention-coverage wiring in
    ``status()``, the reachability classifier reported it as ``dispatchable``
    on every heartbeat check forever while dispatch silently dropped it --
    the exact contradiction (#1059) this test locks down.
    """
    from _fakes_github import FakeGitHub
    from charlie_work.paths import runtime_paths
    from charlie_work.workflow import OrchestratorApp

    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # Issue #1059: automated-ready, OPEN, no blockers -- would be
    # dispatchable if not for the merged-PR mention exclusion.
    fake_gh.issues = [
        {
            "number": 1059,
            "title": "Mentioned by a merged PR",
            "url": "https://example.test/issues/1059",
            "body": "",
            "labels": [{"name": config.labels.ready}],
            "state": "OPEN",
        }
    ]
    # PR #2043: MERGED, same-repo, mentions issue #1059 in free text but
    # does NOT bind (branch name doesn't match agent/issue- prefix, no
    # closing keyword in title/body).
    fake_gh.prs = [
        {
            "number": 2043,
            "title": "Refactor search index",
            "url": "https://example.test/pull/2043",
            "headRefName": "feature/search-refactor",
            "baseRefName": "main",
            "headRefOid": "sha-2043",
            "body": "See issue #1059 for the original bug report.",
            "labels": [],
            "isCrossRepository": False,
            "state": "MERGED",
        }
    ]
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    result = app.status()

    assert result.ok is True
    reachability = result.data["backlog_reachability"]
    # The issue bins as mention_covered, NOT dispatchable.
    assert reachability["mention_covered_awaiting_operator"] == 1
    assert reachability["dispatchable"] == 0
    # The reason names the mentioning PR(s).
    assert reachability["mention_covered_prs"] == {1059: [2043]}


# ---------------------------------------------------------------------------
# 13. Issue #1337: resolve_dispatch_mention_coverage fetch/reuse branches.
#     Three branches mirror the original inline wiring in _dispatch_impl:
#       1. Reuse of an already-fetched (called=True, no error) outcome.
#       2. Fresh-fetch path when the outcome was not called (called=False).
#       3. Fail-open GitHubError path (fresh fetch raises).
#     Each test asserts both the returned mention_covered map and the
#     fetched_merged_prs sentinel (None = no fresh fetch / caller must not
#     rebind; non-None = caller rebinds merged_prs to a called=True outcome).
# ---------------------------------------------------------------------------


class _StubApp:
    """Minimal stub for the app methods compute_mention_coverage_map calls:
    ``_merged_pr_referenced_issue_numbers`` and
    ``_mention_rearmed_issue_numbers``, plus ``paths.state_file``. Avoids
    constructing a full OrchestratorApp so the fetch/reuse branches are
    tested in isolation, not through the entire dispatch path."""

    def __init__(self, state_file: Path, mention_only: set[int]) -> None:
        self.paths = type("Paths", (), {"state_file": state_file})()
        self._mention_only = mention_only

    def _merged_pr_referenced_issue_numbers(
        self, issues: list[dict[str, Any]], merged_prs: list[dict[str, Any]]
    ) -> tuple[set[int], set[int], set[int], dict[int, list[int]]]:
        # Return the stubbed mention_only set with PR numbers derived from
        # the merged_prs list. bound is empty so mention_only is not
        # reduced.
        mention_pr_numbers_by_issue: dict[int, list[int]] = {}
        for pr in merged_prs:
            if str(pr.get("state") or "").upper() != "MERGED":
                continue
            pr_number = int(pr["number"])
            for issue_number in self._mention_only:
                mention_pr_numbers_by_issue.setdefault(issue_number, []).append(pr_number)
        mention_pr_numbers_by_issue = {
            k: sorted(v) for k, v in sorted(mention_pr_numbers_by_issue.items())
        }
        return set(), set(self._mention_only), set(), mention_pr_numbers_by_issue

    def _mention_rearmed_issue_numbers(
        self,
        mention_only: set[int],
        issues: list[dict[str, Any]],
        state: dict[str, Any],
        already_flagged: set[int],
    ) -> tuple[set[int], list[int]]:
        # No re-arming in these tests -- all mention-only issues stay
        # excluded, matching the common case (issue never flagged).
        return set(), []


def _merged_pr(number: int) -> dict[str, Any]:
    return {
        "number": number,
        "state": "MERGED",
        "isCrossRepository": False,
        "headRefName": f"feature/pr-{number}",
        "title": f"PR {number}",
        "body": "",
    }


class _FakeGhWithMergedPrs:
    """Minimal gh stub for resolve_dispatch_mention_coverage: only
    merged_pr_list() is called (on the fresh-fetch branch)."""

    def __init__(self, merged_prs: list[dict[str, Any]]) -> None:
        self._merged_prs = merged_prs

    def merged_pr_list(self) -> list[dict[str, Any]]:
        return list(self._merged_prs)


class _FakeGhRaisingOnMergedPrList:
    """gh stub that raises GitHubError from merged_pr_list() -- the
    fail-open branch."""

    def merged_pr_list(self) -> list[dict[str, Any]]:
        from charlie_work.github import GitHubError

        raise GitHubError("simulated transient error")


def test_resolve_dispatch_mention_coverage_reuses_already_fetched_outcome(
    tmp_path: Path,
) -> None:
    """Branch 1: ``merged_prs_outcome`` was already fetched (called=True,
    no error) by ``_finalize_externally_merged_issues``. The function must
    reuse its items and NOT trigger a fresh fetch (fetched_merged_prs is
    None)."""
    issues = [{"number": 1059, "labels": [{"name": "automated-ready"}], "body": ""}]
    merged_prs = [_merged_pr(2043)]
    app = _StubApp(tmp_path / "state.json", mention_only={1059})
    outcome = _MergedPRListOutcome(merged_prs, called=True)

    mention_covered, fetched = resolve_dispatch_mention_coverage(
        issues, outcome, _FakeGhWithMergedPrs([]), app
    )

    # Reused the outcome's items -- coverage map is populated.
    assert mention_covered == {1059: [2043]}
    # No fresh fetch -- caller must not rebind merged_prs.
    assert fetched is None


def test_resolve_dispatch_mention_coverage_fresh_fetch_when_not_called(
    tmp_path: Path,
) -> None:
    """Branch 2: ``merged_prs_outcome`` was NOT fetched (called=False -- the
    common case when there are no closed-ready issues). The function must
    fetch merged_pr_list() itself and return the fetched list so the caller
    rebinds ``merged_prs`` to a called=True outcome (no second API call
    later)."""
    issues = [{"number": 1059, "labels": [{"name": "automated-ready"}], "body": ""}]
    merged_prs = [_merged_pr(2043)]
    app = _StubApp(tmp_path / "state.json", mention_only={1059})
    # called=False -- _finalize_externally_merged_issues did not fetch.
    outcome = _MergedPRListOutcome(called=False)
    gh = _FakeGhWithMergedPrs(merged_prs)

    mention_covered, fetched = resolve_dispatch_mention_coverage(issues, outcome, gh, app)

    # Fresh fetch populated the coverage map.
    assert mention_covered == {1059: [2043]}
    # The fetched list is returned so the caller can rebind merged_prs.
    assert fetched == merged_prs


def test_resolve_dispatch_mention_coverage_fail_open_on_github_error(
    tmp_path: Path,
) -> None:
    """Branch 3: the fresh fetch raises ``GitHubError``. The function must
    fail open -- return an empty coverage map (issues classify as
    ``dispatchable``) and None for the fetched list. The classifier is
    advisory and must not raise; the later ``_resolve_merged_prs`` re-fetches
    and RAISES so dispatch defers rather than re-dispatching covered issues."""
    issues = [{"number": 1059, "labels": [{"name": "automated-ready"}], "body": ""}]
    app = _StubApp(tmp_path / "state.json", mention_only={1059})
    outcome = _MergedPRListOutcome(called=False)

    mention_covered, fetched = resolve_dispatch_mention_coverage(
        issues, outcome, _FakeGhRaisingOnMergedPrList(), app
    )

    # Fail-open: empty coverage, issues classify as dispatchable.
    assert mention_covered == {}
    # No fetched list -- caller must not rebind.
    assert fetched is None


def test_resolve_dispatch_mention_coverage_empty_issues_skips_fetch(
    tmp_path: Path,
) -> None:
    """Edge case: when ``issues`` is empty, the function must skip the fetch
    entirely (matching ``_resolve_merged_prs``'s issue #361 guard) and return
    an empty map with no fetched list."""
    app = _StubApp(tmp_path / "state.json", mention_only=set())
    outcome = _MergedPRListOutcome(called=False)

    mention_covered, fetched = resolve_dispatch_mention_coverage(
        [], outcome, _FakeGhWithMergedPrs([_merged_pr(2043)]), app
    )

    assert mention_covered == {}
    assert fetched is None
