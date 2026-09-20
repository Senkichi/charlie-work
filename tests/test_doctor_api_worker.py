"""API-worker configuration and budget-ledger checks for ``run_doctor``.

Split out of ``tests/test_doctor.py`` (issue #1563, Track 1 shoulder) --
bodies are verbatim relocations; shared helpers live in
``tests/_doctor_fixtures.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from charlie_work.config import (
    ApiBudgetConfig,
    AutoMergeConfig,
)
from charlie_work.doctor import run_doctor
from charlie_work.paths import runtime_paths
from _doctor_fixtures import (
    FakeDoctorGitHub,
    _api_provider,
    _api_worker_config,
    _config,
)


def test_doctor_api_worker_not_configured_emits_nothing(tmp_path: Path) -> None:
    """Default (absent) api_worker section → no api_worker checks at all."""
    config = _config(auto_merge=AutoMergeConfig(required_checks=(), enabled=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    names = {check.name for check in checks}
    assert not any(name.startswith("api_worker") for name in names)
    assert ok is True


def test_doctor_api_worker_disabled_emits_notice(tmp_path: Path) -> None:
    """Configured but disabled → single notice line, warning severity, ok=True."""
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=False),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    notice = by_name["api_worker configured but disabled"]
    assert notice.ok is True
    assert notice.severity == "warning"
    assert "disabled" in notice.detail
    # No enabled-mode checks should appear.
    assert "api_worker api key" not in by_name
    assert "api_worker base url" not in by_name
    assert ok is True


def test_doctor_api_worker_enabled_all_checks_ok(tmp_path: Path, monkeypatch) -> None:
    """Enabled with key present, valid URL, no ledger → all four checks pass."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["api_worker api key"].ok is True
    assert "MOONSHOT_API_KEY" in by_name["api_worker api key"].detail
    # No secret value leaks into the detail.
    assert "sk-test-value" not in by_name["api_worker api key"].detail

    assert by_name["api_worker base url"].ok is True
    assert "https" in by_name["api_worker base url"].detail

    assert by_name["api_worker budget ledger"].ok is True
    assert "not yet created" in by_name["api_worker budget ledger"].detail

    headroom = by_name["api_worker budget headroom"]
    assert headroom.ok is True
    assert headroom.severity == "warning"
    assert "$0.00 spent today" in headroom.detail
    assert "$15.00" in headroom.detail  # lifetime cap default
    assert ok is True


def test_doctor_api_worker_missing_env_var(tmp_path: Path, monkeypatch) -> None:
    """Enabled but env var absent → api key check fails (error severity)."""
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["api_worker api key"].ok is False
    assert "NOT set" in by_name["api_worker api key"].detail
    assert "MOONSHOT_API_KEY" in by_name["api_worker api key"].detail
    assert ok is False  # error-severity failure blocks


def test_doctor_api_worker_bad_url(tmp_path: Path, monkeypatch) -> None:
    """Enabled but base_url is not https → base url check fails."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(
            enabled=True,
            provider=_api_provider(base_url="http://insecure.example.com"),
        ),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    assert by_name["api_worker base url"].ok is False
    assert "not a valid https URL" in by_name["api_worker base url"].detail
    assert ok is False


def test_doctor_api_worker_corrupt_ledger(tmp_path: Path, monkeypatch) -> None:
    """Enabled with a corrupt ledger file → ledger check fails (corrupt detected)."""
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    # Write a corrupt ledger file.
    ledger_file = paths.root / "api-budget.json"
    ledger_file.write_text("{ this is not valid json", encoding="utf-8")
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    by_name = {check.name: check for check in checks}
    ledger_check = by_name["api_worker budget ledger"]
    assert ledger_check.ok is False
    assert "corrupt" in ledger_check.detail.lower()
    assert "quarantined" in ledger_check.detail.lower()


def test_doctor_api_worker_headroom_math(tmp_path: Path, monkeypatch) -> None:
    """Enabled with a ledger showing spend -> headroom check reports correct math.

    Regression for issue #828 (originally #822's class): production derives
    its own `today = now.strftime("%Y-%m-%d")` ledger key independently of
    this test's fixture write. If the wall clock crosses UTC midnight between
    the write and `run_doctor`'s read, the lookup misses and the report shows
    $0.00 instead of the expected spend -- a real (if rare) production defect,
    not just a test flake. `now` is frozen and passed to both the fixture and
    `run_doctor` so the ledger key always matches regardless of any stall or
    midnight boundary in between.
    """
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True, budget=budget),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    # Write a ledger with today's spend and lifetime spend.
    from datetime import UTC, datetime

    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    today = frozen_now.strftime("%Y-%m-%d")
    ledger_data = {
        "days": {
            today: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "usd": 3.50,
            }
        },
        "lifetime_usd": 10.00,
        "sessions": [],
    }
    ledger_file = paths.root / "api-budget.json"
    ledger_file.write_text(json.dumps(ledger_data), encoding="utf-8")
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, now=frozen_now)

    by_name = {check.name: check for check in checks}
    headroom = by_name["api_worker budget headroom"]
    # 3.50 spent today + 1.0 reserve = 4.50 <= 5.0 cap → daily ok
    # 10.00 < 15.0 → lifetime ok
    assert headroom.ok is True
    assert "$3.50 spent today" in headroom.detail
    assert "$5.00 cap" in headroom.detail
    assert "$1.50 remaining" in headroom.detail  # 5.0 - 3.50
    assert "$10.00 spent lifetime" in headroom.detail
    assert "$5.00 remaining" in headroom.detail  # 15.0 - 10.00
    assert ok is True


def test_doctor_api_worker_headroom_exhausted(tmp_path: Path, monkeypatch) -> None:
    """Enabled with spend at caps -> headroom check fails (exhausted).

    Regression for issue #828 (originally #822's class): same UTC-midnight
    ledger-key race as ``test_doctor_api_worker_headroom_math`` above -- see
    that test's docstring. ``now`` is frozen and passed to both the fixture
    and ``run_doctor``.
    """
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test-value")
    budget = ApiBudgetConfig(
        max_usd_per_session=0.0,
        preflight_reserve_usd=1.0,
        max_usd_per_day=5.0,
        lifetime_usd=15.0,
    )
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True, budget=budget),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)

    from datetime import UTC, datetime

    frozen_now = datetime(2026, 7, 29, 12, 0, 0, tzinfo=UTC)
    today = frozen_now.strftime("%Y-%m-%d")
    # Daily: 4.50 spent + 1.0 reserve = 5.50 > 5.0 -> daily exhausted
    # Lifetime: 15.00 >= 15.0 -> lifetime exhausted
    ledger_data = {
        "days": {
            today: {
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "usd": 4.50,
            }
        },
        "lifetime_usd": 15.00,
        "sessions": [],
    }
    ledger_file = paths.root / "api-budget.json"
    ledger_file.write_text(json.dumps(ledger_data), encoding="utf-8")
    gh = FakeDoctorGitHub(labels=config.labels.all)

    ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh, now=frozen_now)

    by_name = {check.name: check for check in checks}
    headroom = by_name["api_worker budget headroom"]
    assert headroom.ok is False
    # Both daily and lifetime are exhausted in this fixture; the detail must
    # report each leg's exhaustion, not just one occurrence of the word.
    assert headroom.detail.count("exhausted") >= 2
    assert "$4.50 spent today" in headroom.detail
    assert "$0.50 remaining" in headroom.detail  # max(0, 5.0 - 4.50)
    assert "$15.00 spent lifetime" in headroom.detail
    assert "$0.00 remaining" in headroom.detail  # max(0, 15.0 - 15.00)
    # Headroom is warning severity, so it does not block the overall ok —
    # the api key / base url / ledger checks all pass, so ok stays True.
    assert headroom.severity == "warning"
    assert ok is True


def test_doctor_api_worker_no_secret_in_output(tmp_path: Path, monkeypatch) -> None:
    """The API key VALUE must never appear in any doctor check detail."""
    secret = "sk-super-secret-key-value-9999"
    monkeypatch.setenv("MOONSHOT_API_KEY", secret)
    config = _config(
        auto_merge=AutoMergeConfig(required_checks=(), enabled=False),
        api_worker=_api_worker_config(enabled=True),
    )
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    gh = FakeDoctorGitHub(labels=config.labels.all)

    _ok, checks = run_doctor(tmp_path, paths, config, tmp_path / "c.yaml", gh)

    for check in checks:
        if check.name.startswith("api_worker"):
            assert secret not in check.detail, (
                f"Secret leaked in check {check.name!r}: {check.detail}"
            )
