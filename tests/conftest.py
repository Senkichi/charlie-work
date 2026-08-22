from __future__ import annotations

# --- Clock-skew shim (issue #1369) -------------------------------------------
# Must run BEFORE any other import that could bind ``datetime.datetime`` at
# module level.  When ``CHARLIE_TEST_CLOCK_SKEW_DAYS`` is set, every
# ``datetime.now()`` call in production code (imported after this conftest) sees
# a clock shifted forward by that many days.  This lets the weekly canary
# workflow (and local runs) detonate age-sensitive fixtures that compare
# hardcoded absolute timestamps against wall-clock now — before they block
# merges on a random Tuesday.
#
# The shim replaces ``datetime.datetime`` in the ``datetime`` module with a
# subclass whose ``now()`` adds the skew.  Modules that do
# ``from datetime import datetime`` *after* this conftest loads (i.e. all
# production code imported during test collection) get the skewed class.
# ``fromisoformat`` and other parsers are inherited unchanged and return
# original ``datetime`` instances, so only ``now()`` is affected.
import datetime as _datetime_module
import os as _os

_SKEW_DAYS_ENV = _os.environ.get("CHARLIE_TEST_CLOCK_SKEW_DAYS", "")
if _SKEW_DAYS_ENV:
    _skew = _datetime_module.timedelta(days=int(_SKEW_DAYS_ENV))
    _RealDateTime = _datetime_module.datetime

    class _SkewedDateTime(_RealDateTime):
        @classmethod
        def now(cls, tz=None):
            return _RealDateTime.now(tz) + _skew

    _datetime_module.datetime = _SkewedDateTime
# --- End clock-skew shim -----------------------------------------------------

import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile as _tempfile_module  # noqa: E402
from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402
from unittest.mock import create_autospec  # noqa: E402

import pytest  # noqa: E402

from charlie_work.preflight import PreflightResult  # noqa: E402

_UNSET = object()

# Issue #1372: save the real tempfile.gettempdir before any autouse fixture
# patches it. Tests that specifically exercise touch_repo's temp-dir backstop
# restore this via ``monkeypatch.setattr(tempfile, "gettempdir", _real_gettempdir)``.
_real_gettempdir = _tempfile_module.gettempdir


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


@pytest.fixture(autouse=True)
def _redirect_temp_dir_for_touch_repo_backstop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redirect ``fleet_registry._get_temp_dir`` so ``touch_repo``'s temp-dir
    backstop (issue #1372) does not reject ``tmp_path``-based repo_roots in
    tests that exercise ``touch_repo``'s core functionality (first_seen/
    last_seen, moved repo, gh error, etc.).

    pytest's ``tmp_path`` is under the real ``%TEMP%``, so without this
    redirect every existing ``touch_repo`` test would hit the backstop and
    silently skip the write — the test would pass vacuously instead of
    exercising the registration logic. The backstop itself is tested directly
    in ``test_issue_1372_fleet_registry_stale.py`` with the real
    ``tempfile.gettempdir`` (restored by monkeypatching
    ``fleet_registry._get_temp_dir`` back to ``_real_gettempdir``).

    The redirect is scoped to ``fleet_registry._get_temp_dir`` only — the
    helper exists specifically so tests can redirect the backstop's view of
    the temp dir without patching the global ``tempfile.gettempdir``, which
    would break every test that calls ``tempfile.mkdtemp()`` /
    ``NamedTemporaryFile`` (they resolve the temp dir via ``gettempdir()``
    and the redirect target does not exist on disk).

    The redirect points to ``tmp_path / "__system_temp__"`` — a sibling of
    typical ``tmp_path / "repo"`` repo_roots, not a parent — so
    ``contains(temp_root, repo_root)`` correctly returns False and the
    backstop does not fire.
    """
    from charlie_work import fleet_registry

    monkeypatch.setattr(fleet_registry, "_get_temp_dir", lambda: str(tmp_path / "__system_temp__"))


@pytest.fixture(autouse=True)
def _isolate_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip ambient credential env vars so no test's premise depends on what
    happens to be set on the host.

    The 2026-08-15 host reboot made a User-scope ``MOONSHOT_API_KEY`` visible
    to the self-hosted CI runner and every fresh shell. Any test that builds
    an api_worker config without explicitly setting the key (relying on
    "the key is absent, so preflight falls back to devin-shell") silently
    changed routing: dispatch went to the api adapter instead of the
    monkeypatched devin-shell path and hit real ``git worktree add`` calls
    (``test_dispatch_partitioned_homogeneous_batch_labels_with_single_kind``
    turned main red).

    The deny-set is derived from the naming convention rather than
    enumerated: ``api_worker``'s ``api_key_env`` contract names credential
    variables ``*_API_KEY`` (api_worker.py), so every ambient variable with
    that suffix is removed. A test that needs a credential sets it explicitly
    with ``monkeypatch.setenv`` in its own body — autouse fixtures run first,
    so the per-test value wins.
    """
    for name in [key for key in os.environ if key.endswith("_API_KEY")]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_real_pr_create_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let ``pr_create_retry.create_pr_with_retry`` really sleep.

    cw#1273: its default backoff is 10s/30s/90s, spanning the ~45s TLS blips
    observed on this host -- exactly the point of the feature. But that
    means any *existing* test that drives a failing ``gh pr create`` through
    ``_open_salvage_pr``/``apply_fixes`` (written before this module
    existed, with no reason to know about a ``sleep_fn`` parameter) now
    silently costs 130s of real wall-clock time instead of failing fast,
    the same "misbehaves everywhere without going red" hazard the sibling
    ``_no_real_cli_binaries`` fixture above guards against for a different
    real side effect. A test that specifically wants to assert on backoff
    timing (e.g. "sleep_fn saw 10 then 30") passes its own ``sleep_fn``
    directly to ``create_pr_with_retry``, or overrides this patch with its
    own ``monkeypatch.setattr`` -- either wins over this default no-op.

    Patches ``pr_create_retry._default_sleep`` -- a module-local name that
    function's body looks up fresh on every call -- rather than the shared
    stdlib ``time`` module's ``sleep`` attribute. An earlier version of this
    fixture patched ``pr_create_retry.time.sleep``, which (a) did nothing
    for its stated purpose, since ``create_pr_with_retry``'s old
    ``sleep_fn: ... = time.sleep`` default parameter was already bound to
    the original function object at module-import time and is insensitive
    to a later attribute reassignment, and (b) being global-module
    reassignment via ``monkeypatch.setattr(module.time, "sleep", ...)``
    actually mutates the *shared* ``time`` module object every other test
    file also imports, so it silently zeroed out real ``time.sleep(...)``
    calls in unrelated tests for the fixture's entire autouse scope --
    caught when it made ``test_fleet_registry_touch_repo_second_call``'s
    real ``time.sleep(2.0)`` a no-op, collapsing two timestamps that must
    differ. See ``pr_create_retry._default_sleep``'s docstring for why the
    indirection is required for a fixture to reach this at all.
    """
    import charlie_work.pr_create_retry as pr_create_retry_module

    monkeypatch.setattr(pr_create_retry_module, "_default_sleep", lambda seconds: None)


def _healthy_preflight(*args: object, **kwargs: object) -> PreflightResult:
    return PreflightResult(checks=())


@pytest.fixture(autouse=True)
def _default_healthy_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Issue #1363: default every test to a healthy ``run_preflight`` result.

    The preflight gate wired into ``OrchestratorApp._loop_impl`` and
    ``fleet_dispatch.run_fleet_supervise`` (``charlie_work.workflow.run_preflight``
    and ``charlie_work.fleet_dispatch.run_preflight`` respectively -- each module
    imported its own reference, so both must be patched independently) inspects
    the REAL host: free disk space, ``sys.executable``'s location relative to a
    conventional ``.venv``, ``charlie_work.__file__``'s location, and
    ``state.json``'s mtime versus wall clock. That is exactly the point of the
    check in production, but it means every pre-existing test that drives
    ``OrchestratorApp(...).loop()`` or ``run_fleet_supervise(...)`` -- none of
    which exist to test host preconditions -- would otherwise pass or fail
    depending on incidental facts about whichever machine/venv/checkout layout
    pytest happens to run under. Worktree-based local test runs in particular
    put ``sys.executable`` under a DIFFERENT checkout's venv than
    ``charlie_work.__file__`` resolves to, by deliberate project convention
    (see CLAUDE.md's Worktree Discipline section) -- which is indistinguishable,
    to this check, from the real wrong-venv bug class it exists to catch.
    Confirmed by measurement: leaving ``fleet_dispatch``'s reference unpatched
    made every ``test_run_fleet_supervise_*``/``test_supervisor_lifecycle_*``
    test fail, and that failure's early return without the normal lock
    teardown cascaded into unrelated ``test_cli.py`` fleet-registry pollution.

    Tests that target the preflight gate's OWN wiring re-monkeypatch
    ``run_preflight`` inside their own test body via the same ``monkeypatch``
    fixture instance, which cleanly overrides this default for the duration of
    that one test (`test_charlie_work.py`'s two dedicated wiring tests;
    `test_supervise_loop.py`'s tests script the subprocess boundary directly
    and never import either module).
    """
    monkeypatch.setattr("charlie_work.workflow.run_preflight", _healthy_preflight)
    monkeypatch.setattr("charlie_work.fleet_dispatch.run_preflight", _healthy_preflight)
