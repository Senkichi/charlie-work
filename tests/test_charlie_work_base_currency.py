"""Base-currency gating and base-freshness fail-closed predicates.

Split out of ``tests/test_charlie_work.py`` (issue #1554,
Track-1 wave 8/8).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from _fakes_github import FakeGitHub
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.workflow import OrchestratorApp


def test_is_base_currency_gated_protection_raises_never_lowers_config(tmp_path: Path) -> None:
    """Issue #875 truth table. The merge gate is ``require_current_base OR
    protection.strict`` -- protection can only ever raise the requirement.

    The asymmetry versus ``_is_base_freshness_required`` (which protection
    fully overrides) is deliberate: that one governs the broadcast sweep, whose
    write costs N CI cycles across every open PR, where this one costs a single
    cycle for the PR actually merging. Cheap enough to always pay; the broadcast
    is not, which is what issue #812 correctly established.
    """
    from charlie_work.config import AutoMergeConfig

    strict_true = {"required_status_checks": {"strict": True}}
    strict_false = {"required_status_checks": {"strict": False}}

    # (require_current_base, protection payload or None, expected gate)
    cases: list[tuple[bool, dict[str, Any] | None, bool]] = [
        # THE #875 FIX: config asks for the gate, protection must not veto it.
        (True, strict_false, True),
        # #812's direction still holds: protection alone can turn the gate on.
        (False, strict_true, True),
        # Both agree.
        (True, strict_true, True),
        (False, strict_false, False),
        # Unreadable protection falls back to config, in both directions.
        (True, None, True),
        (False, None, False),
    ]

    for require_current_base, payload, expected in cases:
        config = OrchestratorConfig(
            auto_merge=AutoMergeConfig(
                require_current_base=require_current_base,
                # strategy must stay on: require_current_base=True + "off" is
                # blocked by __post_init__ as an inescapable deferral loop.
                update_open_prs=True,
            )
        )
        paths = runtime_paths(tmp_path, config.runtime.state_dir)
        fake_gh = FakeGitHub()
        if payload is not None:
            fake_gh.branch_protection_overrides["main"] = payload
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        assert app._is_base_currency_gated("main") is expected, (
            f"require_current_base={require_current_base} payload={payload!r}"
        )


def test_is_base_currency_gated_skips_protection_read_when_config_requires(
    tmp_path: Path,
) -> None:
    """``require_current_base=True`` short-circuits before the protection read.

    Not merely an optimization: it means the merge gate cannot be disabled by a
    protection API outage, rate limit, or a repo whose protection is readable
    but shaped unexpectedly. The strongest form of failing closed is not
    depending on the remote read at all.

    Deliberately asserts on a recorded call rather than raising from
    ``branch_protection``: ``_is_base_freshness_required`` catches broad
    ``Exception``, so a raise would be swallowed and converted into the same
    ``True`` this test expects -- the assertion would hold whether or not the
    short-circuit existed, making it blind to the very regression it exists to
    catch.
    """
    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()
    # strict:false would DISABLE the gate if the protection read were consulted,
    # so this payload makes the read's influence observable in the return value
    # as well as in the call log.
    fake_gh.branch_protection_overrides["main"] = {"required_status_checks": {"strict": False}}
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_currency_gated("main") is True
    assert fake_gh.branch_protection_calls == []


def test_is_base_freshness_required_fails_closed_when_protection_raises(tmp_path: Path) -> None:
    """Fail-closed shape 1/3: the protection read raising an exception must not
    propagate as an unhandled error, and must not be treated as "no freshness
    required" -- it falls back to the require_current_base config value.
    """

    class ExplodingProtectionGitHub(FakeGitHub):
        def branch_protection(self, base: str) -> dict[str, Any] | None:
            raise RuntimeError("simulated network failure")

    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = ExplodingProtectionGitHub()
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_freshness_required("main") is True


def test_is_base_freshness_required_fails_closed_when_protection_read_errors(
    tmp_path: Path,
) -> None:
    """Fail-closed shape 2/3: an error-value read (404 / rate-limited / gh
    unavailable) surfaces as branch_protection() returning None -- the fake's
    default for a base with no configured override, mirroring the real
    GitHub.branch_protection()'s contract. This must fall back to
    require_current_base, not be treated as "no freshness required".
    """
    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # no override for "main" -> branch_protection("main") is None
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_freshness_required("main") is True


def test_is_base_freshness_required_fails_closed_on_malformed_protection_payload(
    tmp_path: Path,
) -> None:
    """Fail-closed shape 3/3: the protection payload is readable (200 OK) but
    `required_status_checks.strict` is absent or the wrong type -- e.g. a repo
    with protection configured for something other than status checks, or a
    GitHub API response shape this code doesn't anticipate. Every malformed
    shape must fall back to require_current_base, never silently disable the
    gate.
    """
    config = OrchestratorConfig()  # require_current_base defaults to True
    paths = runtime_paths(tmp_path, config.runtime.state_dir)

    malformed_payloads: list[dict[str, Any]] = [
        {},  # no required_status_checks key at all
        {"required_status_checks": {}},  # present but no "strict" key
        {"required_status_checks": {"strict": "yes"}},  # wrong type (str, not bool)
        {"required_status_checks": None},  # wrong type (None, not dict)
        {"enforce_admins": {"enabled": True}},  # unrelated protection field only
    ]
    for payload in malformed_payloads:
        fake_gh = FakeGitHub()
        fake_gh.branch_protection_overrides["main"] = payload
        app = OrchestratorApp(tmp_path, paths, config, fake_gh)
        assert app._is_base_freshness_required("main") is True, f"payload={payload!r}"


def test_is_base_freshness_required_bidirectional_config_fallback(tmp_path: Path) -> None:
    """The fallback target on failure is require_current_base itself (per issue
    #812's non-goal: preserve it as the fallback), not a hardcoded True -- an
    operator who has explicitly opted out via config keeps that behavior when
    protection can't be read, matching pre-#812 semantics exactly.
    """
    from charlie_work.config import AutoMergeConfig

    config = OrchestratorConfig(auto_merge=AutoMergeConfig(require_current_base=False))
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    fake_gh = FakeGitHub()  # unreadable protection (no override -> None)
    app = OrchestratorApp(tmp_path, paths, config, fake_gh)

    assert app._is_base_freshness_required("main") is False
