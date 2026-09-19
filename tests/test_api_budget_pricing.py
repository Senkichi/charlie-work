"""cost_usd pricing math for the api worker spend ledger (issue #480).

Split out of ``tests/test_api_budget.py`` (issue #1571, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_api_budget_unit_fixtures.py``.
"""

from __future__ import annotations

import pytest

from charlie_work.api_budget import Usage, cost_usd
from charlie_work.config import ApiProviderConfig

from _api_budget_unit_fixtures import _provider


def test_cost_usd_hand_computed_with_cached_pricing() -> None:
    """1M input @ $3, 0.2M output @ $15, 0.5M cached @ $0.30 = 3 + 3 + 0.15 = 6.15."""
    usage = Usage(input_tokens=1_000_000, output_tokens=200_000, cached_tokens=500_000)
    assert cost_usd(usage, _provider()) == pytest.approx(6.15)


def test_cost_usd_zero_usage() -> None:
    assert cost_usd(Usage(), _provider()) == 0.0


def test_cost_usd_cached_default_zero_rate() -> None:
    """cached_input_usd_per_mtok defaults to 0.0 → cached tokens are free."""
    provider = ApiProviderConfig(
        base_url="https://api.example.com/anthropic",
        api_key_env="EXAMPLE_API_KEY",
        model="example-model",
        input_usd_per_mtok=3.0,
        output_usd_per_mtok=15.0,
        cached_input_usd_per_mtok=0.0,
    )
    usage = Usage(input_tokens=1_000_000, output_tokens=0, cached_tokens=2_000_000)
    # Only input billed: 1M * 3 = 3.0; cached 2M * 0 = 0.
    assert cost_usd(usage, provider) == pytest.approx(3.0)


def test_cost_usd_small_token_volume_precision() -> None:
    """10k input, 1k output, 5k cached at default pricing."""
    usage = Usage(input_tokens=10_000, output_tokens=1_000, cached_tokens=5_000)
    expected = (10_000 / 1e6) * 3.0 + (1_000 / 1e6) * 15.0 + (5_000 / 1e6) * 0.30
    assert cost_usd(usage, _provider()) == pytest.approx(expected)
