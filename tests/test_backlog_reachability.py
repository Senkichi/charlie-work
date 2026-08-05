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

from typing import Any

from charlie_work import cli
from charlie_work.config import OrchestratorConfig
from charlie_work.workflow import classify_backlog_reachability


class FakeGh:
    """Minimal stub for the one GitHubLike method classify_backlog_reachability
    calls: issue_list(state=...). Never touches the network."""

    def __init__(self, issues: list[dict[str, Any]]) -> None:
        self._issues = issues
        self.calls: list[dict[str, Any]] = []

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        self.calls.append({"labels": labels, "state": state})
        return list(self._issues)


def _issue(number: int, names: set[str]) -> dict[str, Any]:
    return {"number": number, "labels": [{"name": n} for n in names]}


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
