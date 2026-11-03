"""The ``os.name`` phase-boundary guard (issue #3010, ci-fleet nightly break 16).

Each end-to-end case runs a real inner pytest session, because the failure mode
is how pytest itself behaves with a foreign ``os.name`` between phases: an
in-process fake could not tell the guard from a stub.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import _os_name_guard as guard

# Applied per test, not module-wide: the wiring test below must run on every host, because
# the conftest registration it pins is unconditional (only the guard's *effect* is Windows-only).
_windows_only = pytest.mark.skipif(
    not guard.ENFORCED, reason="the guard is enforced on a Windows host only"
)

_TESTS_DIR = Path(__file__).resolve().parent
_FOREIGN = "posix"

_LEAKS = textwrap.dedent(
    f"""
    import os
    import pytest

    FOREIGN = {_FOREIGN!r}

    @pytest.fixture
    def leaky_fixture(monkeypatch):
        monkeypatch.setattr(os, "name", FOREIGN)

    def test_leaks_until_teardown(monkeypatch):
        monkeypatch.setattr(os, "name", FOREIGN)

    def test_bounded_to_the_call(monkeypatch):
        with monkeypatch.context() as patched:
            patched.setattr(os, "name", FOREIGN)
        assert os.name != FOREIGN

    def test_leaks_from_a_fixture(leaky_fixture):
        pass
    """
)

# The nightly's actual shape. pytest-cov's ``pytest_runtest_teardown`` runs BEFORE the
# core teardown (the one that undoes ``monkeypatch``) and, on coverage 7.17.1, raised
# while ``os.name`` was foreign. The exception skips the core teardown, so the failure is
# reported with the patch still live: the error's traceback holds no frame of the test
# file (it is all plugin frames), so pytest runs it through its traceback filter, which
# walks ``Path(...).parents`` -- impossible while ``os.name`` is foreign -- and the
# session aborts (INTERNALERROR) instead of reporting a failed test. This hook models
# that without needing coverage in the session.
_TEARDOWN_HOOK_RAISES = textwrap.dedent(
    f"""
    import os
    import pytest

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_teardown(item):
        if os.name != {guard.HOST_OS_NAME!r}:
            raise ValueError("a plugin hook failed while os.name was foreign")
    """
)

_LEAKS_INTO_A_FAILING_HOOK = textwrap.dedent(
    f"""
    import os

    def test_leaks_into_a_failing_teardown_hook(monkeypatch):
        monkeypatch.setattr(os, "name", {_FOREIGN!r})
    """
)


# One failure in each wrapped phase, each raised while a foreign ``os.name`` is live:
# the call phase (the test body) and the setup phase (a fixture after the leaking one).
_LEAKS_THEN_FAILS = textwrap.dedent(
    f"""
    import os
    import pytest

    @pytest.fixture
    def leaky_fixture(monkeypatch):
        monkeypatch.setattr(os, "name", {_FOREIGN!r})

    @pytest.fixture
    def failing_fixture(leaky_fixture):
        raise RuntimeError("the fixture's own failure")

    def test_leaks_then_fails_its_own_assertion(monkeypatch):
        monkeypatch.setattr(os, "name", {_FOREIGN!r})
        raise AssertionError("the test's own failure")

    def test_leaks_then_a_later_fixture_fails(failing_fixture):
        pass
    """
)


def _run_inner(tmp_path: Path, body: str, *, guarded: bool, conftest: str = "") -> tuple[int, str]:
    proj = tmp_path / "proj"
    proj.mkdir()
    # An ini file pins the inner rootdir: a basetemp inside a checkout would
    # otherwise inherit that repo's addopts (e.g. "-n auto").
    (proj / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (proj / "test_inner.py").write_text(body, encoding="utf-8")
    if conftest:
        (proj / "conftest.py").write_text(conftest, encoding="utf-8")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PYTEST_", "CI_FLEET_", "COVERAGE", "CHARLIE_TEST_SLOT"))
    }
    env["PYTHONPATH"] = str(_TESTS_DIR)
    # No entry-point plugins (xdist, ci-fleet's recorder): the session is the guard's alone.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rA"]
    if guarded:
        argv += ["-p", guard.PLUGIN_NAME]
    done = subprocess.run(
        [*argv, "test_inner.py"],
        cwd=proj,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def test_the_guard_is_registered_in_the_live_pytest_session(
    request: pytest.FixtureRequest,
) -> None:
    """Wiring: every other test here runs a *separate* inner session with the guard passed
    explicitly (``-p``), so none of them notices if ``tests/conftest.py::pytest_configure``
    stops registering it. This asks the session that is running the suite right now."""
    pluginmanager = request.config.pluginmanager

    assert pluginmanager.has_plugin(guard.PLUGIN_NAME)
    assert pluginmanager.get_plugin(guard.PLUGIN_NAME) is guard
    for hook in (pluginmanager.hook.pytest_runtest_setup, pluginmanager.hook.pytest_runtest_call):
        owners = {impl.plugin_name for impl in hook.get_hookimpls()}
        assert guard.PLUGIN_NAME in owners, f"{guard.PLUGIN_NAME} has no live {hook.name} impl"


def test_restore_host_os_name_is_a_no_op_on_the_host_value() -> None:
    assert os.name == guard.HOST_OS_NAME
    assert guard.restore_host_os_name() is None
    assert os.name == guard.HOST_OS_NAME


def test_restore_host_os_name_returns_the_foreign_value_and_puts_the_host_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as patched:
        patched.setattr(os, "name", _FOREIGN)
        live = guard.restore_host_os_name()
        restored = os.name
    assert (live, restored) == (_FOREIGN, guard.HOST_OS_NAME)


@_windows_only
def test_a_patch_that_outlives_the_test_body_fails_that_test_with_the_remedy(
    tmp_path: Path,
) -> None:
    code, out = _run_inner(tmp_path, _LEAKS, guarded=True)

    assert code == 1, out
    assert "INTERNALERROR" not in out, out
    # Only the two leaking tests are flagged; the bounded one is clean.
    assert "PASSED test_inner.py::test_bounded_to_the_call" in out, out
    assert "FAILED test_inner.py::test_leaks_until_teardown" in out, out
    assert "ERROR test_inner.py::test_leaks_from_a_fixture" in out, out
    assert "test_inner.py::test_leaks_until_teardown left os.name == 'posix'" in out, out
    assert "end of its call phase" in out and "end of its setup phase" in out, out
    assert "monkeypatch.context()" in out, out


# The two "without the guard" controls pin pytest/CPython internals (what pytest tolerates and
# how it fails with a foreign os.name). If one goes red after an upgrade, delete or rewrite the
# control -- it does not mean the guard broke.
@_windows_only
def test_without_the_guard_the_same_leak_passes_silently(tmp_path: Path) -> None:
    """Control: the guard is what flags the leak -- nothing else in pytest does."""
    code, out = _run_inner(tmp_path, _LEAKS, guarded=False)

    assert code == 0, out
    assert "3 passed" in out, out


@_windows_only
def test_without_the_guard_a_teardown_hook_failure_in_the_window_is_an_internal_error(
    tmp_path: Path,
) -> None:
    """The nightly's crash mode with no coverage in the session: the run is red with
    exit 3 (``INTERNALERROR``) and no failed test, because pytest cannot even format
    the hook's error while ``os.name`` is foreign. (Control: pins pytest/CPython
    internals -- red means delete this control, not that the guard broke.)"""
    code, out = _run_inner(
        tmp_path, _LEAKS_INTO_A_FAILING_HOOK, guarded=False, conftest=_TEARDOWN_HOOK_RAISES
    )

    assert code == 3, out
    assert "INTERNALERROR" in out, out


@_windows_only
def test_with_the_guard_the_same_teardown_hook_never_sees_a_foreign_os_name(
    tmp_path: Path,
) -> None:
    code, out = _run_inner(
        tmp_path, _LEAKS_INTO_A_FAILING_HOOK, guarded=True, conftest=_TEARDOWN_HOOK_RAISES
    )

    assert code == 1, out
    assert "INTERNALERROR" not in out, out
    # One failed test, named, with the remedy -- and the hook found the host's value.
    assert "FAILED test_inner.py::test_leaks_into_a_failing_teardown_hook" in out, out
    assert "left os.name == 'posix'" in out, out
    assert "a plugin hook failed" not in out, out


@_windows_only
def test_with_the_guard_a_test_that_leaks_then_fails_reports_its_own_failure(
    tmp_path: Path,
) -> None:
    """The exception path, in BOTH wrapped phases: a test body (call) or a later fixture
    (setup) that fails inside the window must not leave the foreign name live for the
    teardown hooks (here the hook that aborts the run when it sees it)."""
    code, out = _run_inner(
        tmp_path, _LEAKS_THEN_FAILS, guarded=True, conftest=_TEARDOWN_HOOK_RAISES
    )

    assert code == 1, out
    assert "INTERNALERROR" not in out, out
    # Call phase: the test body's own failure.
    assert "FAILED test_inner.py::test_leaks_then_fails_its_own_assertion" in out, out
    assert "the test's own failure" in out, out
    # Setup phase: the later fixture's own error, reported as an ERROR of that test.
    assert "ERROR test_inner.py::test_leaks_then_a_later_fixture_fails" in out, out
    assert "the fixture's own failure" in out, out
    # Each failure is the report; the guard does not replace it with its message.
    assert "left os.name" not in out, out
    assert "a plugin hook failed" not in out, out
