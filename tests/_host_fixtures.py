"""Helpers for tests that script process liveness through the host probe."""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from collections.abc import Iterator

from charlie_work import host as host_pkg
from charlie_work.host.fakes import FakeProcessProbe


@contextmanager
def host_probe(alive: bool, pid: int = 99999) -> Iterator[FakeProcessProbe]:
    """Install a fake probe where ``pid`` is alive (or dead) for the block."""
    probe = FakeProcessProbe({pid: None} if alive else {})
    prev = host_pkg._ACTIVE
    host_pkg._ACTIVE = dataclasses.replace(prev, probe=probe)
    try:
        yield probe
    finally:
        host_pkg._ACTIVE = prev
