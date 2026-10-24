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
import tomllib
import types

import pytest

from charlie_work.supervise_loop import EXIT_RESTART_REQUESTED, PREFLIGHT_REFUSAL_EXIT_CODE

_ENTRY = importlib.import_module("charlie_work._entry")
_MODULE_MAIN = importlib.import_module("charlie_work.__main__")

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

# A fake ``uv tool`` prefix: ends with uv/tools/<tool name>, the shape the
# entry's detection looks for (verified live: `uv tool dir` is
# %APPDATA%/uv/tools on Windows, ~/.local/share/uv/tools on POSIX).
_FAKE_TOOL_PREFIX = str(pathlib.Path.home() / "uvshare" / "uv" / "tools" / "charlie-work")

# A fake non-tool prefix: pinned explicitly (review round 1) so the uv-sync
# branch is asserted against a decisive shape, not whatever prefix the test
# process happens to run under.
_NON_TOOL_PREFIX = str(pathlib.Path.home() / "venvs" / "charlie-dev")

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


def _stub_cli(monkeypatch: pytest.MonkeyPatch, return_value: int = 0) -> list[int]:
    """Plant ``charlie_work.cli`` in sys.modules without importing the real one."""
    calls: list[int] = []
    stub = types.ModuleType("charlie_work.cli")
    stub.main = lambda: calls.append(1) or return_value  # type: ignore[attr-defined]
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


@pytest.mark.parametrize(
    "cli_exit_code",
    [0, EXIT_RESTART_REQUESTED, PREFLIGHT_REFUSAL_EXIT_CODE],
    ids=["clean-exit-0", "restart-requested", "preflight-refusal"],
)
def test_main_returns_cli_exit_code_unchanged(
    monkeypatch: pytest.MonkeyPatch, cli_exit_code: int
) -> None:
    """``_entry.main()`` returns whatever ``cli.main()`` returned, unchanged.

    ``cli.main`` returns more than 0 and success: ``EXIT_RESTART_REQUESTED``
    is a cross-version wire contract (ADR-0004, #862) the supervise-loop
    wrapper reads to decide to relaunch, and ``PREFLIGHT_REFUSAL_EXIT_CODE``
    (#1363) is a distinct refusal the operator must be able to tell apart
    from a clean exit. ``_entry.main`` is the new console entry point, so a
    ``return 0`` there -- or any code between cli and the caller -- would
    make a restart request read as a clean exit: #862, reintroduced through
    the very seam this PR adds. The stubbed return is parametrized over both
    constants (imported, never hard-coded, so a renumber turns this loud)
    plus plain 0, and each must come back through ``_entry.main`` verbatim.
    """
    _install_metadata(monkeypatch, _SATISFYING)
    calls = _stub_cli(monkeypatch, return_value=cli_exit_code)

    assert _ENTRY.main() == cli_exit_code
    assert calls == [1], "cli.main was never reached despite a satisfied environment"


@pytest.mark.parametrize("force_fallback", [False, True], ids=["packaging", "stdlib-fallback"])
def test_non_matching_marker_requirement_is_ignored(
    monkeypatch: pytest.MonkeyPatch, force_fallback: bool
) -> None:
    """A marker that does not match this environment is ignored, not reported.

    Run through BOTH parsers (review round 1): ``packaging`` is the parser in
    dev/CI environments, the stdlib fallback is the one inside uv tool
    environments, and the skip must not depend on which one runs.

    ``haiku-only`` would read unmet (``0.0.1`` installed against ``>=99.0``)
    if its marker were evaluated anyway. ``main`` must still reach the CLI,
    and the recorded lookups must show the version was never even probed.
    """
    if force_fallback:
        monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
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


def test_fallback_skips_non_matching_and_extra_markers_without_version_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the fallback: markers that do not apply are skipped.

    ``packaging`` is absent in the production tool environment, so the
    skip-without-lookup behavior is asserted there too (review round 1): a
    ``sys_platform == 'haiku'`` entry AND an ``extra == 'dev'`` entry are
    ignored, and the record of version lookups proves neither distribution
    was probed at all -- even though both would read unmet if checked.
    """
    monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
    declared = [
        "ci-fleet>=0.7.6",
        "haiku-only>=99.0; sys_platform == 'haiku'",
        "dev-tool>=99.0; extra == 'dev'",
    ]
    looked_up = _install_metadata(
        monkeypatch,
        {"ci-fleet": "0.7.7", "haiku-only": "0.0.1", "dev-tool": "0.0.1"},
        declared=declared,
    )
    calls = _stub_cli(monkeypatch)

    assert _ENTRY.main() == 0
    assert calls == [1]
    assert looked_up == ["ci-fleet"], (
        f"non-matching/extra-marker requirements were version-probed: {looked_up}"
    )


@pytest.mark.parametrize("force_fallback", [False, True], ids=["packaging", "stdlib-fallback"])
def test_non_tool_environment_prints_uv_sync_hint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    force_fallback: bool,
) -> None:
    """Outside a uv tool environment the recovery is a plain ``uv sync`` hint.

    Parametrized over both parsers (review round 1), with ``sys.prefix``
    pinned to a decisive non-tool shape: the real process prefix could in
    principle end in uv/tools/<name>, in which case the unparametrized
    original would assert the opposite branch.
    """
    if force_fallback:
        monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
    _install_metadata(monkeypatch, _STALE)
    monkeypatch.setattr(sys, "prefix", _NON_TOOL_PREFIX)

    with pytest.raises(SystemExit) as stop:
        _ENTRY.main()

    assert stop.value.code == 2
    stderr = capsys.readouterr().err
    assert "uv sync" in stderr
    assert "uv tool install" not in stderr
    assert _NON_TOOL_PREFIX in stderr


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


# --- the fallback fails OPEN on specifiers it cannot judge (review round 1) ---
# The old comparator silently dropped parse DecorATIONS: `0.7.*` became the
# release (0, 7), so `==0.7.*` against an installed 0.7.3 read UNSATISFIED
# (blocking a healthy start) and `!=0.7.*` (wrongly) read SATISFIED. Every
# comparison it cannot fully judge is now `None`, which callers treat as
# satisfied -- the fail-open invariant the module promises.


@pytest.mark.parametrize(
    ("installed", "operator", "spec_version", "why"),
    [
        ("0.7.3", "==", "0.7.*", "wildcard spec: impossible without wildcard semantics"),
        ("0.7.3", "!=", "0.7.*", "wildcard spec: old comparator dropped the * and judged"),
        ("0.7.3", ">=", "0.7.*", "wildcard spec, even for an op the answer seems obvious"),
        ("0.7.3", "===", "0.7.3", "arbitrary-precision equality: packaging territory"),
        ("0.7.3", "==", "0.7.3", "== stays judgeable and the pair genuinely matches"),
    ],
)
def test_fallback_fails_open_on_unjudgeable_specifiers(
    installed: str, operator: str, spec_version: str, why: str
) -> None:
    """Wildcard specs and ``===`` return None -- never a judged verdict.

    The wildcard cases fail against the pre-hardening comparator, which
    reduced `0.7.*` to `(0, 7)` and judged the comparison anyway.
    """
    result = _ENTRY._stdlib_spec_satisfied(installed, operator, spec_version)
    if operator == "==" and spec_version == "0.7.3":
        assert result is True, why
    else:
        assert result is None, why


@pytest.mark.parametrize(
    ("installed", "plen_check"),
    [
        ("1.0.0rc1", "pre-release decoration: N.Y.ZrcM is not a parseable release tail"),
        ("1.0.post2", "post-release decoration"),
        ("2026.4.dev1", "dev decoration"),
        ("", "empty version: nothing to compare"),
        ("0.7.*", "wildcard installed side, defensive"),
    ],
)
def test_fallback_fails_open_on_fully_unparseable_versions(
    installed: str, plen_check: str
) -> None:
    """A version _stdlib_version_key cannot fully parse makes the comparison None.

    The pre-hardening comparator truncated at the first non-digit chunk
    (0.1.0rc1 -> (0, 1, 0)) and judged; it now abstains.
    """
    assert _ENTRY._stdlib_spec_satisfied(installed, ">=", "0.1") is None, plen_check
    assert _ENTRY._stdlib_version_key("6.0") == (6, 0)  # fully parseable versions still work


def test_fallback_wildcard_and_exact_precision_requirements_are_not_unmet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement-level consequence of the fail-open abstentions.

    Against the hardening's own comparator: `ci-fleet==0.7.*` with installed
    0.7.3 is satisfied, and `ci-fleet !=0.7.*` / `ci-fleet===0.7.3` are NOT
    reported unmet (they abstain, and an abstention reads as satisfied). The
    pre-hardening comparator failed both directions: `==0.7.*` read unmet,
    and `!=0.7.*` was *wrongly* judged satisfied by a truncated comparison.
    """
    monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
    declared = [
        "wildcard-eq==0.7.*",
        "wildcard-ne!=0.7.*",
        "exact===0.7.3",
    ]
    _install_metadata(
        monkeypatch,
        {"wildcard-eq": "0.7.3", "wildcard-ne": "0.7.3", "exact": "0.7.3"},
        declared=declared,
    )
    calls = _stub_cli(monkeypatch)

    assert _ENTRY.main() == 0
    assert calls == [1]


# --- the recovery command hardening (review round 1): verified root or placeholder ---


def test_project_root_requires_the_charlie_work_project_name(tmp_path: pathlib.Path) -> None:
    """A stray pyproject.toml above the package must not be handed to uv.

    The old walk accepted the FIRST pyproject.toml found while ascending, so
    a host tool's config file or an unrelated project above the checkout
    could be named in the recovery command -- uv would then install *that*
    project. The walk now verifies [project].name.
    """
    foreign = tmp_path / "some-operator-app"
    nested = foreign / "src" / "charlie_work"
    nested.mkdir(parents=True)
    (foreign / "pyproject.toml").write_text(
        '[project]\nname = "some-operator-app"\n', encoding="utf-8"
    )

    assert _ENTRY._project_root_from(nested) is None


def test_project_root_accepts_the_charlie_work_checkout(tmp_path: pathlib.Path) -> None:
    """Positive control: the declared checkout is found and returned."""
    checkout = tmp_path / "charlie-work"
    nested = checkout / "src" / "charlie_work"
    nested.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "charlie-work"\n', encoding="utf-8"
    )

    assert _ENTRY._project_root_from(nested) == checkout


def test_project_root_ignores_an_unparseable_pyproject_and_walks_on(
    tmp_path: pathlib.Path,
) -> None:
    """A corrupt TOML above the package is skipped, not fatal."""
    broken = tmp_path / "broken"
    nested = broken / "src" / "charlie_work"
    nested.mkdir(parents=True)
    (broken / "pyproject.toml").write_text("not [valid toml", encoding="utf-8")

    assert _ENTRY._project_root_from(nested) is None


def test_recovery_command_names_the_verified_checkout_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """With a verified root, the command installs THAT checkout, editable."""
    monkeypatch.setattr(_ENTRY, "_project_root", lambda: tmp_path)
    unmet = [("ci-fleet", "ci-fleet>=0.7.6", "0.2.0"), ("pyyaml", "pyyaml>=6.0", "5.0")]

    command = _ENTRY._recovery_command(unmet)

    assert command == (
        f"uv tool install --editable {tmp_path} --force "
        "--refresh-package ci-fleet --refresh-package pyyaml"
    )


def test_recovery_command_without_a_root_prints_an_editable_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No root: an explicit editable placeholder, never a registry-name install.

    The old fallback printed `uv tool install <tool env dirname>` -- an
    index-resolved install of the published package, never the situation an
    editable tool env needs repairing.
    """
    monkeypatch.setattr(_ENTRY, "_project_root", lambda: None)

    command = _ENTRY._recovery_command([("ci-fleet", "ci-fleet>=0.7.6", "0.2.0")])

    assert command == (
        "uv tool install --editable <path to your charlie-work checkout> "
        "--force --refresh-package ci-fleet"
    )


def test_no_root_failure_message_prints_the_placeholder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end: the message's recovery is the placeholder command, not the env name."""
    monkeypatch.setattr(_ENTRY, "_PackagingRequirement", None)
    monkeypatch.setattr(_ENTRY, "_project_root", lambda: None)
    _install_metadata(monkeypatch, _STALE)
    monkeypatch.setattr(sys, "prefix", _FAKE_TOOL_PREFIX)

    with pytest.raises(SystemExit) as stop:
        _ENTRY.main()

    assert stop.value.code == 2
    stderr = capsys.readouterr().err
    assert "uv tool install --editable <path to your charlie-work checkout>" in stderr
    assert _FAKE_TOOL_PREFIX not in stderr, (
        "the tool environment's directory name leaked into the recovery command"
    )


def test_verified_root_failure_message_names_the_checkout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: pathlib.Path
) -> None:
    """End to end: a verified root is embedded as the recovery target."""
    monkeypatch.setattr(_ENTRY, "_project_root", lambda: tmp_path)
    _install_metadata(monkeypatch, _STALE)
    monkeypatch.setattr(sys, "prefix", _FAKE_TOOL_PREFIX)

    with pytest.raises(SystemExit) as stop:
        _ENTRY.main()

    assert stop.value.code == 2
    stderr = capsys.readouterr().err
    assert f"uv tool install --editable {tmp_path}" in stderr
    assert "--refresh-package ci-fleet" in stderr


# --- the wiring: every entry surface routes through _entry (review round 1) ---


def test_console_scripts_declare_the_entry_module() -> None:
    """Both [project.scripts] mappings name charlie_work._entry:main.

    Read straight from pyproject.toml with tomllib, the same way
    tests/test_ci_fleet_floor_guard.py reads the dependency floor -- the def
    the entry check story rests on is this mapping, not the module.
    """
    scripts = tomllib.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]["scripts"]

    assert scripts["charlie"] == "charlie_work._entry:main"
    assert scripts["charlie-work"] == "charlie_work._entry:main"


def test_module_entrypoint_is_the_entry_module() -> None:
    """`python -m charlie_work` dispatches the same guarded main, by identity."""
    assert _MODULE_MAIN.main is _ENTRY.main


def test_python_dash_m_package_help_exits_zero() -> None:
    """Subprocess proof: `python -m charlie_work --help` runs the guarded entry.

    The wiring tests above read names; this one RUNS the package form end to
    end (an optional wiring check the review named) through a fresh
    interpreter, PYTHONPATH pinned to this checkout's src the way
    test_cli_module_entrypoint does.
    """
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(_SRC) + (os.pathsep + existing if existing else "")
    result = subprocess.run(
        [sys.executable, "-m", "charlie_work", "--help"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=str(_SRC.parent),
    )

    assert result.returncode == 0, f"stderr={result.stderr[:400]}"
    assert result.stdout.strip(), "no usage text: the package form produced nothing"
    assert "usage" in result.stdout.lower()
