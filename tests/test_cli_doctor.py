"""``charlie doctor`` structured-finding reporting.

Split out of ``tests/test_cli.py`` (issue #1561, Track 1) -- bodies are
verbatim relocations; shared fakes live in ``tests/_cli_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from charlie_work import cli


def test_run_doctor_command_reports_structured_finding_on_unparseable_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6-G / G-AC3: doctor must not itself crash on the exact condition it
    exists to diagnose.

    Before this fix, ``run_doctor_command`` called ``load_layered_config``
    unguarded, so a config parse failure (e.g. the 2026-07-29 incident's
    unknown ``cross_family: auto_verdict`` key) propagated past this function
    to ``main()``'s generic ``except (ConfigError, ValueError)`` handler,
    which prints to stderr and exits 2 with no machine-readable finding --
    the operator gets nothing to act on. Now the failure is caught locally
    and rendered as a structured, blocking ``DoctorCheck`` finding instead.

    The role-config Phase 2 cleanup deleted the dual-accept section
    tolerance mechanism entirely, so a bare ``cross_family:`` section is no
    longer specially tolerated -- it is rejected as an unknown top-level
    section like any other bogus key, the same as the original incident's
    exact reproduction shape would raise again today. This test instead
    drives the REAL ``load_layered_config`` (not mocked) against an unknown
    key inside a section that has always validated its own keys
    (``labels``), which raises the identical ``ConfigError`` shape the
    original incident hit -- proving the exception-handling path this test
    exists for, independent of which section happens to raise.
    """
    monkeypatch.setattr(cli, "find_repo_root", lambda repo, explicit=False, **kw: tmp_path)
    # Deliberately NOT mocking cli.load_layered_config.
    (tmp_path / "orchestrator.config.yaml").write_text(
        "labels:\n  ready: automated-ready\n  totally_unknown_key: true\n",
        encoding="utf-8",
    )

    args = cli.build_parser().parse_args(["--fleet-dir", str(tmp_path), "doctor"])
    result = cli.run_doctor_command(args)

    assert result.ok is False, "an unparseable config must fail the command, not crash it"
    assert "1 finding" in result.message
    checks = result.data["checks"]
    assert len(checks) == 1, f"expected exactly one synthetic finding, got: {checks!r}"
    check = checks[0]
    assert check["name"] == "config file"
    assert check["ok"] is False
    assert check["severity"] == "error", "a config parse failure must be blocking, not a warning"
    assert "totally_unknown_key" in check["detail"]
    assert "labels" in check["detail"]
