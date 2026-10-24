"""Issue #2808: the CLI entry must check declared requirements before importing cli.

A ``uv tool install --editable`` tool environment tracks the checkout's source
but never re-resolves its dependencies. After a dependency floor is raised in
``pyproject.toml``, the next ``charlie`` invocation used to die at *import*
time -- ``charlie_work.cli`` imports ``doctor`` -> ``paths`` ->
``host_io_worker`` -> ``ci_fleet.host_io`` -- with a bare
``ModuleNotFoundError`` and no hint that the tool environment was stale.

``charlie_work._entry`` is the fix: the console scripts and ``python -m
charlie_work`` now enter through it, and it compares the installed
distributions against the declared requirements before importing anything
heavy. The tests model the observed 2026-10-23 incident (``ci-fleet`` 0.2.0
installed against a declared ``>=0.7.0``-class floor) by patching
``importlib.metadata`` -- the seam the entry reads -- so the tests are
hermetic against whatever the running environment actually has installed.

Two parsers can serve the check: ``packaging`` (present in dev/CI
environments) and the stdlib fallback (``packaging`` is *not* guaranteed
inside a uv tool environment -- the live one on the operator host does not
have it). The fallback path is therefore exercised explicitly, not only
incidentally.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import pathlib
import subprocess
import sys
import types

import pytest

_ENTRY = importlib.import_module("charlie_work._entry")

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# A fake ``uv tool`` prefix: ends with uv/tools/<tool name>, the shape the
# entry's detection looks for (verified live: `uv tool dir` is
# %APPDATA%/uv/tools on Windows, ~/.local/share/uv/tools on POSIX).
_FAKE_TOOL_PREFIX = str(pathlib.Path.home() / "uvshare" / "uv" / "tools" / "charlie-work")

_DECLARED = [
    "pyyaml>=6.0",
    "psutil>=5.9",
    "tzdata>=2026.3",
    "ci-fleet>=0.7.6",
]
_SATISFYING = {"pyyaml": "6.0", "psutil": "5.9", "tzdata": "2026.4", "ci-fleet": "0.7.7"}
_STALE = {"pyyaml": "6.0", "psutil": "5.9", "tzdata": "2026.4", "ci-fleet": "0.2.0"}


def _install_metadata(
    monkeypatch: pytest.MonkeyPatch,
    installed: dict[str, str],
    declared: list[str] | None = None,
) -> list[str]:
    """Serve ``declared`` from ``importlib.metadata`` and record version lookups.

    Returns the list of distribution names whose installed version was
    actually looked up -- assertions use it to prove marker-skipped
    requirements were never checked at all.
    """
    declared = _DECLARED if declared is None else declared
    looked_up: list[str] = []

    def fake_requires(name: str) -> list[str]:
        if name == "charlie-work":
            return list(declared)
        raise importlib.metadata.PackageNotFoundError(name)

    def fake_version(name: str) -> str:
        if name in installed:
            looked_up.append(name)
            return installed[name]
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "requires", fake_requires)
    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    return looked_up


def _stub_cli(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Plant ``charlie_work.cli`` in sys.modules without importing the real one."""
    calls: list[int] = []
    stub = types.ModuleType("charlie_work.cli")
    stub.main = lambda: calls.append(1) or 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "charlie_work.cli", stub)
    return calls


def test_stale_dependency_exits_two_and_names_the_recovery(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The modeled incident: installed 0.2.0 against a newer declared floor.

    Must exit 2 with stderr naming the distribution, its installed version,
    the declared requirement, and the exact ``uv tool install --editable``
    recovery command. Whether anything heavy was imported on the way is made
    observable by the subprocess test below.
    """
    _install_metadata(monkeypatch, _STALE)
    monkeypatch.setattr(sys, "prefix", _FAKE_TOOL_PREFIX)

    with pytest.raises(SystemExit) as stop:
        _ENTRY.main()

    assert stop.value.code == 2
    stderr = capsys.readouterr().err
    assert "ci-fleet" in stderr
    assert "0.2.0" in stderr
    assert "ci-fleet>=0.7.6" in stderr
    assert "uv tool install --editable" in stderr
    assert "--force" in stderr
    assert "--refresh-package ci-fleet" in stderr


def test_stale_dependency_leaves_cli_unimported_in_a_fresh_interpreter() -> None:
    """Subprocess proof: the failure path must not import ``charlie_work.cli``.

    In-process, ``charlie_work.cli`` is often already in ``sys.modules`` from
    other tests, so a fresh interpreter is the only honest observation.
    PYTHONPATH pins this checkout's ``src`` (the venv's editable install may
    point elsewhere in a worktree) -- the same recipe
    ``test_cli_module_entrypoint`` uses. The driver re-raises the exit code so
    the parent can assert the process exits 2 while also reading its report.
    """
    driver = (
        "import importlib.metadata as metadata\n"
        "import sys\n"
        "sys.prefix = sys.argv[1]\n"
        "def fake_requires(name):\n"
        f"    if name == 'charlie-work':\n"
        f"        return {_DECLARED!r}\n"
        "    raise metadata.PackageNotFoundError(name)\n"
        "def fake_version(name):\n"
        f"    versions = {_STALE!r}\n"
        "    if name in versions:\n"
        "        return versions[name]\n"
        "    raise metadata.PackageNotFoundError(name)\n"
        "metadata.requires = fake_requires\n"
        "metadata.version = fake_version\n"
        "code = 0\n"
        "try:\n"
        "    import charlie_work._entry as entry\n"
        "    entry.main()\n"
        "except SystemExit as stop:\n"
        "    code = stop.code\n"
        "print('CLI_IMPORTED' if 'charlie_work.cli' in sys.modules else 'CLI_NOT_IMPORTED')\n"
        "raise SystemExit(code)\n"
    )
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(_SRC) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-c", driver, _FAKE_TOOL_PREFIX],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(_SRC.parent),
    )

    assert result.returncode == 2, f"stderr={result.stderr[:500]}"
    assert "CLI_NOT_IMPORTED" in result.stdout, (
        f"charlie_work.cli was imported before the dependency check (stdout={result.stdout!r})"
    )
    for expected in ("ci-fleet", "0.2.0", "ci-fleet>=0.7.6", "uv tool install --editable"):
        assert expected in result.stderr, f"missing {expected!r} in: {result.stderr[:500]}"


def test_satisfied_requirements_reach_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: every requirement met -> main() reaches cli.main, returns 0.

    A guard can grow a bug that never passes; this plants the counter-case.
    The success path must land on ``charlie_work.cli.main`` -- here observed
    through the stub, proving the real (heavy) module was never imported.
    """
    _install_metadata(monkeypatch, _SATISFYING)
    calls = _stub_cli(monkeypatch)

    assert _ENTRY.main() == 0
    assert calls == [1], "cli.main was never reached despite a satisfied environment"


def test_non_matching_marker_requirement_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker that does not match this environment is ignored, not reported.

    ``haiku-only`` would read unmet (``0.0.1`` installed against ``>=99.0``)
    if its marker were evaluated anyway. ``main`` must still reach the CLI,
    and the recorded lookups must show the version was never even probed.
    """
    declared = ["ci-fleet>=0.7.6", "haiku-only>=99.0; sys_platform == 'haiku'"]
    looked_up = _install_metadata(
        monkeypatch,
        {"ci-fleet": "0.7.7", "haiku-only": "0.0.1"},
        declared=declared,
    )
    calls = _stub_cli(monkeypatch)

    assert _ENTRY.main() == 0
    assert calls == [1]
    assert "haiku-only" not in looked_up, (
        "the non-matching marker requirement was version-checked, not skipped"
    )


def test_non_tool_environment_prints_uv_sync_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Outside a uv tool environment the recovery is a plain ``uv sync`` hint."""
    _install_metadata(monkeypatch, _STALE)

    with pytest.raises(SystemExit) as stop:
        _ENTRY.main()

    assert stop.value.code == 2
    stderr = capsys.readouterr().err
    assert "uv sync" in stderr
    assert "uv tool install" not in stderr


def test_uncheckable_metadata_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A check that cannot run at all must not block the start.

    The guard only arms a broken start; it must never become a second way for
    a healthy environment to fail to open.
    """

    def explode() -> list:
        raise RuntimeError("corrupt dist-info")

    monkeypatch.setattr(_ENTRY, "_unmet_requirements", explode)
    calls = _stub_cli(monkeypatch)

    assert _ENTRY.main() == 0
    assert calls == [1]


# --- the stdlib fallback, end to end (production path inside uv tool envs) ---
# The live tool environment has no `packaging`, so the fallback parser is what
# will actually run there; force it for the behavior assertions below instead
# of only exercising it incidentally.


def test_fallback_comparator_rejects_the_modeled_incident() -> None:
    """The stdlib comparator alone sees ci-fleet 0.2.0 as unmet for >=0.7.6."""
    assert _ENTRY._stdlib_spec_satisfied("0.2.0", ">=", "0.7.6") is False
    assert _ENTRY._stdlib_spec_satisfied("0.7.6", ">=", "0.7.6") is True
    assert _ENTRY._stdlib_spec_satisfied("0.7.7", ">=", "0.7.6") is True
    assert _ENTRY._stdlib_spec_satisfied("2026.4", ">=", "2026.3") is True
    assert _ENTRY._stdlib_spec_satisfied("0.7.6", "==", "0.7.6") is True
    assert _ENTRY._stdlib_spec_satisfied("0.7", "==", "0.7.0") is True
    assert _ENTRY._stdlib_spec_satisfied("0.7.6", "!=", "0.7.6") is False
    assert _ENTRY._stdlib_spec_satisfied("0.8.0", "<=", "0.7.6") is False
    assert _ENTRY._stdlib_spec_satisfied("0.7.7", "~=", "0.7.6") is True
    assert _ENTRY._stdlib_spec_satisfied("0.8.0", "~=", "0.7.6") is False
    assert _ENTRY._stdlib_spec_satisfied("0.8.0", "~=", "0.7") is True
    assert _ENTRY._stdlib_spec_satisfied("0.6.9", "~=", "0.7") is False


def test_fallback_marker_applies_and_skips() -> None:
    assert _ENTRY._stdlib_marker_applies("sys_platform == 'haiku'") is False
    assert _ENTRY._stdlib_marker_applies("extra == 'dev'") is False
    assert _ENTRY._stdlib_marker_applies("python_version >= '3.9'") is True
    assert _ENTRY._stdlib_marker_applies("") is True
    assert _ENTRY._stdlib_marker_applies("(sys_platform == 'haiku')") is None
    assert _ENTRY._stdlib_marker_applies("unknown_variable == 'x'") is None


def test_stale_dependency_exits_two_without_packaging(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Same modeled incident, forced through the stdlib fallback parser."""
    monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
    _install_metadata(monkeypatch, _STALE)
    monkeypatch.setattr(sys, "prefix", _FAKE_TOOL_PREFIX)

    with pytest.raises(SystemExit) as stop:
        _ENTRY.main()

    assert stop.value.code == 2
    stderr = capsys.readouterr().err
    for expected in ("ci-fleet", "0.2.0", "ci-fleet>=0.7.6", "uv tool install --editable"):
        assert expected in stderr, f"missing {expected!r} in: {stderr[:500]}"


def test_declared_suite_passes_without_packaging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback parser must satisfy the whole declared set when versions match."""
    monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
    _install_metadata(
        monkeypatch,
        {"pyyaml": "6.0.3", "psutil": "7.2.2", "tzdata": "2026.5", "ci-fleet": "0.7.7"},
    )
    calls = _stub_cli(monkeypatch)

    assert _ENTRY.main() == 0
    assert calls == [1]
