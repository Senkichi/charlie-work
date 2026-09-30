"""GitHub rate-limit budget: pure values fed by response headers (ADR-0006)."""

from __future__ import annotations

import dataclasses

import pytest

from charlie_work.api_budget import (
    GitHubRateBudget,
    GitHubRateWindow,
    github_headroom,
    observe_github_rate,
)
from charlie_work.github_transport import RateBudgetHolder, Response

HEADERS = (
    ("x-ratelimit-limit", "5000"),
    ("x-ratelimit-remaining", "4000"),
    ("x-ratelimit-reset", "2000"),
)


def test_observing_headers_adds_a_core_window_by_default() -> None:
    budget = observe_github_rate(GitHubRateBudget(), HEADERS, 100.0)
    assert budget.windows == (GitHubRateWindow("core", 5000, 4000, 2000, 100.0),)


def test_the_window_for_a_resource_is_replaced_not_appended() -> None:
    first = observe_github_rate(GitHubRateBudget(), HEADERS, 100.0)
    newer = (
        ("X-RateLimit-Limit", "5000"),
        ("X-RateLimit-Remaining", "3999"),
        ("X-RateLimit-Reset", "2000"),
    )
    second = observe_github_rate(first, newer, 101.0)  # names are matched case-insensitively
    assert [w.remaining for w in second.windows] == [3999]


def test_resources_are_tracked_independently() -> None:
    budget = observe_github_rate(GitHubRateBudget(), HEADERS, 100.0)
    graphql = (*HEADERS, ("x-ratelimit-resource", "graphql"))
    budget = observe_github_rate(budget, graphql, 101.0)
    assert {w.resource for w in budget.windows} == {"core", "graphql"}


@pytest.mark.parametrize(
    "headers",
    [
        (),
        (("x-ratelimit-limit", "5000"),),
        (("x-ratelimit-limit", "x"), ("x-ratelimit-remaining", "1"), ("x-ratelimit-reset", "2")),
    ],
)
def test_missing_or_malformed_headers_leave_the_budget_unchanged(headers) -> None:
    before = observe_github_rate(GitHubRateBudget(), HEADERS, 100.0)
    assert observe_github_rate(before, headers, 101.0) is before


def test_headroom_is_none_for_an_unseen_or_expired_window() -> None:
    budget = observe_github_rate(GitHubRateBudget(), HEADERS, 100.0)
    assert github_headroom(budget, "search", 100.0) is None
    live = github_headroom(budget, "core", 1999.0)
    assert live is not None and live.remaining == 4000
    assert github_headroom(budget, "core", 2000.0) is None  # reset passed: says nothing about now


def test_the_values_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        GitHubRateWindow("core", 1, 1, 1, 1.0).remaining = 0  # type: ignore[misc]


def test_the_holder_replaces_one_value_under_a_lock() -> None:
    holder = RateBudgetHolder()
    assert holder.value == GitHubRateBudget()
    holder.observe(Response(200, HEADERS, "", "http"), 100.0)
    assert holder.value.windows[0].remaining == 4000
