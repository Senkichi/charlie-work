"""Helpers for tests that script process liveness through the host probe."""

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
