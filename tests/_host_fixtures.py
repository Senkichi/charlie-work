"""Helpers for tests that script process liveness through the host probe.

``_no_real_foreman_pass_child`` joined under the #3122 rework: the autouse
port guard moved out of ``conftest.py`` when the file crossed the 800-line
cap (file-size ratchet, issue #1442) — a host-port fake belongs with the
host helpers, and this module's light import set keeps
``charlie_work.foreman`` out of the conftest path (ADR-0010's seam
notwithstanding, the lazy import inside the fixture body is what scopes the
cost to the tests that install a launcher).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from charlie_work import host as host_pkg
from charlie_work.host.fakes import FakeProcessProbe


@contextmanager
def host_probe(
    monkeypatch: pytest.MonkeyPatch, alive: bool, pid: int = 99999
) -> Iterator[FakeProcessProbe]:
    """Install a fake probe where ``pid`` is alive (or dead) for the block.

    The swap goes through the test's own ``monkeypatch`` so teardown ordering
    composes with ``fake_host`` (issue #2237): when ``fake_host(...)`` runs
    inside the block its undo entry records the probe ports, and monkeypatch's
    LIFO teardown then unwinds past it to this block's pre-value — with a
    private save/restore, teardown would instead resurrect the probe fake and
    leak it to the next test on the same xdist worker.  Block exit still
    restores the pre-block value immediately so the probe does not leak past
    the ``with`` within the test itself.
    """
    probe = FakeProcessProbe({pid: None} if alive else {})
    prev = host_pkg.current()
    monkeypatch.setattr(host_pkg, "_ACTIVE", dataclasses.replace(prev, probe=probe))
    try:
        yield probe
    finally:
        host_pkg._ACTIVE = prev


@pytest.fixture(autouse=True)
def _no_real_foreman_pass_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let a test spawn a real ``charlie scheme`` child (issue #3122).

    The fleet supervise loop spawns the periodic foreman pass through
    ``host.current().foreman_pass_launch`` — ``Popen(sys.executable -m
    charlie_work … scheme)``, a real detached process. The REAL launcher's
    ``launch`` is stubbed with a fake (the port object itself stays REAL —
    ``fake_host(foreman_pass_launch=...)`` still replaces it wholesale for
    scripted outcomes).
    """
    from charlie_work.host.fakes import FakeForemanPassLauncher

    monkeypatch.setattr(
        host_pkg.REAL.foreman_pass_launch,
        "launch",
        FakeForemanPassLauncher().launch,
    )
