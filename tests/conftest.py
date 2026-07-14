from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_fleet_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect the fleet registry to a per-test directory.

    This prevents any test that drives ``cli.main`` / ``build_app`` without an
    explicit ``--fleet-dir`` from writing to the operator's real
    ``%LOCALAPPDATA%\\charlie-work\\fleet.json`` (or the platform equivalent).
    ``fleet_dir()`` already honors ``CHARLIE_WORK_FLEET_DIR``; the fixture uses
    that single knob for suite-wide isolation.
    """
    monkeypatch.setenv("CHARLIE_WORK_FLEET_DIR", str(tmp_path / "fleet"))
