"""``os.name`` must read the host's value between pytest phases (issue #3010).

Incident (ci-fleet nightly break 16, "unbisectable", no failing test): four tests
set the process-global ``os.name`` to ``"posix"`` through ``monkeypatch`` on a
Windows host. A ``monkeypatch`` patch lives until fixture teardown, so it was
still in force when the next pytest phase began. On Windows ``pathlib`` builds a
``PosixPath`` once ``os.name`` is ``"posix"``, and a drive-letter path is
"relative" to it -- ``PosixPath.as_uri()`` raises, and ``.parents`` raises
``UnsupportedOperation``. Anything that touches a path at a phase boundary then
breaks:

* coverage 7.17.1 (released the morning of the failing nightly) converts the data
  file path with ``Path(...).as_uri()`` on every connect. pytest-cov switches the
  coverage context at the start of each phase, so the nightly's map pass raised
  ``ValueError`` in ``pytest_runtest_teardown``;
* pytest's own traceback filter (``Path(entry.path).parents``) raises while
  formatting *any* failure reported inside the window.

Either way pytest reports an ``INTERNALERROR`` (exit 3): the run is red with no
failed test, which the ci-fleet bisect can only call ``unbisectable``.

The structural fix is this guard, not a list of offenders: after the ``setup``
and ``call`` phases of every test, ``os.name`` must equal the value it had when
the session started. A violation restores the host's value (so the failure can
be reported at all) and fails the offending test with the remedy. Bound the
patch to the code under test instead::

    with monkeypatch.context() as patched:
        patched.setattr(os, "name", "posix")
        result = function_under_test()
    assert result == ...

The guard is enforced on a Windows host only: that is where the nightly and every
pull request run, and where a ``"posix"`` value breaks ``pathlib``. Elsewhere the
suite has many untriaged ``"nt"`` patches; enforcing there would be a separate,
larger change.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import pytest

# Captured at import, before any test can patch it. Also the plugin's own name, so
# ``-p _os_name_guard`` and the conftest registration are one and the same plugin.
PLUGIN_NAME = "_os_name_guard"
HOST_OS_NAME = os.name
ENFORCED = HOST_OS_NAME == "nt"


def restore_host_os_name() -> str | None:
    """Put the host's ``os.name`` back; return the foreign value that was live, if any."""
    live = os.name
    if live == HOST_OS_NAME:
        return None
    os.name = HOST_OS_NAME
    return live


def foreign_name_message(nodeid: str, phase: str, live: str) -> str:
    return (
        f"{nodeid} left os.name == {live!r} (host: {HOST_OS_NAME!r}) in force at the end of "
        f"its {phase} phase. A patch of the process-global os.name that outlives the test "
        "body breaks pathlib and coverage's context switch (issue #3010). Bound it to the "
        "code under test with `with monkeypatch.context() as patched: "
        "patched.setattr(os, 'name', ...)`."
    )


def _fail_if_foreign(nodeid: str, phase: str) -> None:
    live = restore_host_os_name()
    if live is not None:
        pytest.fail(foreign_name_message(nodeid, phase, live), pytrace=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, Any, Any]:
    if not ENFORCED:
        return (yield)
    try:
        result = yield
    except BaseException:
        # The setup already failed; restoring keeps THAT failure formattable.
        restore_host_os_name()
        raise
    _fail_if_foreign(item.nodeid, "setup")
    return result


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, Any, Any]:
    if not ENFORCED:
        return (yield)
    try:
        result = yield
    except BaseException:
        # A failing assertion inside the window would otherwise be reported while
        # pathlib is broken: restore first so the real failure reaches the report.
        restore_host_os_name()
        raise
    _fail_if_foreign(item.nodeid, "call")
    return result
