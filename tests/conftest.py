from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import create_autospec

import pytest


_UNSET = object()


def autospec_patch(
    monkeypatch: pytest.MonkeyPatch,
    target: Any,
    name: str,
    side_effect: Callable[..., Any] | None = None,
    return_value: Any = _UNSET,
) -> Any:
    """Replace ``target.name`` with a ``create_autospec``-wrapped double.

    The mock's call signature is derived from the real object, so a double that
    omits parameters or uses different names fails immediately in the test that
    installed it, rather than in a later unrelated test.  ``side_effect`` and/or
    ``return_value`` configure the replacement as in ``unittest.mock``.
    """
    real = getattr(target, name)
    if not callable(real):
        raise TypeError(f"cannot autospec non-callable {name!r} on {target!r}")
    mock = create_autospec(real)
    if side_effect is not None:
        mock.side_effect = side_effect
    if return_value is not _UNSET:
        mock.return_value = return_value
    monkeypatch.setattr(target, name, mock)
    return mock


@pytest.fixture
def autospec() -> Callable[..., Any]:
    """Provide the autospec_patch helper to tests."""
    return autospec_patch


@pytest.fixture(autouse=True)
def _no_real_cli_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a test resolve a bare CLI name to a real installed binary.

    Issue #569: launch-path tests that omit ``command_template`` fall through
    to the default template, whose bare ``"claude"`` argv[0] is resolved by
    ``resolve_cli_binary`` via PATH — on any machine with the CLI installed
    (dev boxes, the self-hosted CI runner) that spawned REAL authenticated
    Claude sessions: API/quota burn, OS toast spam, and console-window
    flashes from the orphan sessions' hook children. On runners without the
    CLI the same tests silently degraded to error records, so the suite was
    green everywhere while misbehaving only where it hurt.

    The dynamic rule (no hardcoded binary denylist): an explicit absolute
    argv[0] (``sys.executable`` fakes, tmp-dir scripts) resolves normally; a
    bare name that PATH resolution would turn into a real installed CLI —
    the only route to one — resolves to ``sys.executable`` instead, which
    exits immediately on the unrecognized worker flags with no network side
    effects; a bare name that is genuinely not installed passes through
    unchanged so missing-binary error handling stays testable.
    ``devin_shell`` needs no guard: it never resolves argv[0], so bare
    names already fail spawn (WinError 2) into error records.

    Guarded at the adapter namespace, not ``subprocess_runner`` itself, so
    git/gh plumbing and the direct ``resolve_cli_binary`` unit tests are
    unaffected. Tests that monkeypatch the resolver themselves override this
    fixture's patch as before.
    """
    import charlie_work.claude_code as claude_code_module

    real_resolve = claude_code_module.resolve_cli_binary

    def _guarded(name: str) -> str:
        if Path(name).is_absolute():
            return real_resolve(name)
        resolved = real_resolve(name)
        if resolved == name and shutil.which(name) is None:
            return name
        return sys.executable

    monkeypatch.setattr(claude_code_module, "resolve_cli_binary", _guarded)


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
