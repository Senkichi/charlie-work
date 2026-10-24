"""Check charlie-work's declared requirements before the CLI imports anything.

A ``uv tool install --editable`` tool environment tracks the checkout's source
but never re-resolves its dependencies. When ``pyproject.toml`` raises a
dependency floor, the tool environment keeps the old version, and the next
``charlie`` invocation dies at import time with a bare ``ModuleNotFoundError``
-- before argparse, so even ``charlie --help`` and ``charlie doctor`` are
unusable, and the traceback names a missing submodule instead of a stale
environment (issue #2808).

This module is the console entry point. It compares the installed
distributions against the requirements charlie-work declares and only on
success imports ``charlie_work.cli`` lazily. The check stays cheap and
hermetic: no network, no subprocess, and no import of ``charlie_work.cli`` or
any ``ci_fleet`` module prior to it passing -- importing those is exactly what
crashes on a stale environment.

Reinstalling the tool environment is deliberately NOT automated: it is a
supply-chain-sensitive action that must stay operator-initiated, so the
failure message hands the operator the recovery command instead.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import sys
import tomllib
from pathlib import Path

_DIST_NAME = "charlie-work"
_EXIT_UNMET_REQUIREMENTS = 2

# `packaging` is the high-fidelity requirement parser when the environment
# happens to have it (dev and CI environments do). A uv tool environment is
# not guaranteed to -- the live one on the operator host does not -- so every
# parse below also has a stdlib fallback, and every fallback fails OPEN: a
# requirement it cannot evaluate is skipped, never reported as unmet. A wrongly
# reported unmet requirement would block every `charlie` start; a skipped one
# only reverts to the pre-#2808 behavior for that requirement.
try:
    from packaging.requirements import Requirement as _PackagingRequirement
except ImportError:  # pragma: no cover - depends on the environment's packaging
    _PackagingRequirement = None


def _stdlib_version_key(version: str) -> tuple[int, ...] | None:
    """Numeric release segments of a fully parseable version string.

    The version is fully parsed only when every ``[._-]``-separated chunk of
    its release part (the text before any ``+local`` segment) is pure digits:
    anything else -- ``0.7.*``, ``1.0.0rc1``, ``1.0.post1``, an empty string --
    is NOT knowably comparable under PEP 440, so ``None`` is returned and
    callers fail open. Guessing an ordering to complete a comparison is
    exactly the false verdict this module refuses to produce.
    """
    source = version.split("+", 1)[0].strip().lower()
    if not source:
        return None
    segments: list[int] = []
    for chunk in re.split(r"[._-]", source):
        if not re.fullmatch(r"\d+", chunk):
            return None
        segments.append(int(chunk))
    return tuple(segments)


def _stdlib_spec_satisfied(installed: str, op: str, spec_version: str) -> bool | None:
    """One fallback specifier comparison; ``None`` when it cannot judge.

    Pure stdlib, no ``packaging``. Comparisons zero-pad the numeric release
    segments so ``0.7`` equals ``0.7.0`` the way PEP 440 says. ``~=`` is
    evaluated as its PEP 440 expansion (``>= X.Y.Z, == X.Y.*``) and needs at
    least two release segments; a one-segment ``~=X`` is unjudgeable. Extreme
    precision (``===``), wildcard specifiers (``0.7.*``), and any version
    ``_stdlib_version_key`` cannot fully parse are equally unjudgeable:
    ``None``. Any unjudgeable comparison fails open (callers treat ``None``
    as satisfied) -- a fallback guess must never report a satisfying version
    as unmet, and must never dress a truncated parse up as a verdict.
    """
    if op == "===":
        return None  # arbitrary-precision equality is not judgeable without packaging
    if "*" in installed or "*" in spec_version:
        return None
    left_raw = _stdlib_version_key(installed)
    right_raw = _stdlib_version_key(spec_version)
    if left_raw is None or right_raw is None:
        return None
    width = max(len(left_raw), len(right_raw))
    left = left_raw + (0,) * (width - len(left_raw))
    right = right_raw + (0,) * (width - len(right_raw))
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == "<":
        return left < right
    if op == "~=":
        if len(right_raw) < 2:
            return None
        fixed = right_raw[:-1]
        return left >= right and left[: len(fixed)] == fixed
    return None


def _stdlib_marker_variable(name: str) -> str | None:
    """Value of one marker variable in this environment, for the fallback.

    ``extra`` is always the empty string: the entry checks the base
    environment, no extras. A variable with no stdlib source here is ``None``,
    which the marker evaluator reports as unjudgeable rather than guessing.
    """
    if name == "extra":
        return ""
    if name == "sys_platform":
        return sys.platform
    if name == "platform_system":
        return platform.system()
    if name == "os_name":
        return os.name
    if name == "platform_machine":
        return platform.machine()
    if name == "platform_python_implementation":
        return platform.python_implementation()
    if name == "implementation_name":
        return sys.implementation.name
    if name == "python_version":
        return f"{sys.version_info[0]}.{sys.version_info[1]}"
    if name == "python_full_version":
        return platform.python_version()
    return None


_ORDERING_OPERATORS = (">=", "<=", ">", "<")


def _stdlib_marker_applies(marker: str) -> bool | None:
    """Fallback marker evaluation; ``None`` when it cannot judge.

    Covers the common shape: single-variable conditions compared against a
    quoted value, joined by ``and``/``or`` (``and`` binds tighter, PEP 508).
    Parenthesized groups, ``in``/``not in``, unknown variables, and ordering
    comparisons over non-numeric strings all read as ``None`` -- the caller
    then skips the requirement instead of blocking startup on the fallback's
    own expressive limits.
    """
    text = marker.strip()
    if not text:
        return True
    if "(" in text or ")" in text or re.search(r"\bin\b", text):
        return None

    def condition_truth(condition: str) -> bool | None:
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(.+?)\s*$", condition)
        if match is None:
            return None
        variable, remainder = match.groups()
        value = _stdlib_marker_variable(variable)
        if value is None:
            return None
        operator = None
        for token in ("==", "!=", ">=", "<=", ">", "<", "~="):
            if remainder.startswith(token):
                operator = token
                break
        if operator is None:
            return None
        operand = remainder[len(operator) :].strip()
        if len(operand) >= 2 and operand[0] == operand[-1] and operand[0] in "'\"":
            operand = operand[1:-1]
        if operator in ("==", "!="):
            return (value == operand) if operator == "==" else (value != operand)
        if operator == "~=":
            return _stdlib_spec_satisfied(value, operator, operand)
        if operator in _ORDERING_OPERATORS and (
            not re.match(r"\d", value) or not re.match(r"\d", operand)
        ):
            return None
        return _stdlib_spec_satisfied(value, operator, operand)

    applies: bool | None = False
    for or_clause in re.split(r"\s+or\s+", text, flags=re.IGNORECASE):
        clause_truth: bool | None = True
        for condition in re.split(r"\s+and\s+", or_clause, flags=re.IGNORECASE):
            truth = condition_truth(condition)
            if truth is None:
                return None
            clause_truth = bool(clause_truth) and truth
        applies = bool(applies) or bool(clause_truth)
    return applies


def _stdlib_parse_requirement(
    text: str,
) -> tuple[str, list[tuple[str, str]], str | None] | None:
    """Split ``name[extras] specifiers ; marker`` without ``packaging``.

    Returns ``(name, [(operator, version), ...], marker)`` or ``None`` when the
    text needs machinery the fallback does not have (a URL requirement, a
    malformed specifier). ``None`` makes the caller skip the requirement,
    never report it.
    """
    body, _, marker = text.partition(";")
    match = re.match(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)", body)
    if match is None:
        return None
    name = match.group(1)
    rest = body[match.end() :].strip()
    if rest.startswith("["):  # extras: this check runs with no extras applied
        close = rest.find("]")
        if close < 0:
            return None
        rest = rest[close + 1 :].strip()
    specs: list[tuple[str, str]] = []
    if rest:
        if rest.startswith("(") and rest.endswith(")"):
            rest = rest[1:-1].strip()
        for part in rest.split(","):
            spec = re.match(r"^(>=|<=|!=|~=|==|>|<)\s*(.+?)\s*$", part.strip())
            if spec is None:
                return None
            specs.append((spec.group(1), spec.group(2).strip("'\"")))
    return name, specs, marker.strip() or None


def _installed_version(name: str) -> str | None:
    """The installed distribution's version, tolerating ``-``/``_`` drift.

    uv writes dist-info directory names with underscores
    (``ci_fleet-0.7.7.dist-info``) while requirements spell the distribution
    with a hyphen (verified live against the operator host's tool environment
    on 2026-10-23), so a direct lookup miss retries the other spelling.
    ``None`` means no installed distribution was recognized.
    """
    for candidate in (name, name.replace("-", "_"), name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _requirement_satisfied(text: str) -> tuple[bool, str, str | None]:
    """One declared requirement against the installed environment.

    Returns ``(satisfied, name, installed version)``. Unjudgeable shapes fail
    open: a requirement the parsers cannot evaluate is reported as satisfied
    with the raw text as its name, leaving the import to surface anything
    real. A requirement whose marker does not match this environment
    (including every ``extra == 'dev'`` entry) is ignored, not checked.
    """
    if _PackagingRequirement is not None:
        requirement = _PackagingRequirement(text)
        name = requirement.name
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            return True, name, None
        installed = _installed_version(name)
        if installed is None:
            return False, name, None
        return (
            bool(requirement.specifier.contains(installed, prereleases=True)),
            name,
            installed,
        )

    parsed = _stdlib_parse_requirement(text)
    if parsed is None:
        return True, text, None
    name, specs, marker = parsed
    if marker:
        applies = _stdlib_marker_applies(marker)
        if applies is None or not applies:
            return True, name, None
    installed = _installed_version(name)
    if installed is None:
        return False, name, None
    for operator, spec_version in specs:
        satisfied = _stdlib_spec_satisfied(installed, operator, spec_version)
        if satisfied is None:
            return True, name, installed
        if not satisfied:
            return False, name, installed
    return True, name, installed


def _declared_requirements() -> list[str]:
    """The requirement strings charlie-work declares, from installed metadata.

    ``requires()`` returns ``None`` for a distribution without requirements;
    that reads as no requirements, not a failure. A distribution that is not
    installed at all raises ``PackageNotFoundError`` -- fail open to the
    import (the resulting import error names a far more familiar problem).
    """
    requires = importlib.metadata.requires(_DIST_NAME)
    return list(requires or [])


def _unmet_requirements() -> list[tuple[str, str, str | None]]:
    """``(name, declared text, installed version)`` for every unmet requirement."""
    unmet: list[tuple[str, str, str | None]] = []
    for text in _declared_requirements():
        if not text.strip():
            continue
        satisfied, name, installed = _requirement_satisfied(text)
        if not satisfied:
            unmet.append((name, text, installed))
    return unmet


def _is_uv_tool_environment(prefix: str) -> bool:
    """Is ``prefix`` a uv tool environment (``<uv dir>/tools/<tool name>``)?

    ``uv tool dir`` is ``%APPDATA%/uv/tools`` on Windows and
    ``~/.local/share/uv/tools`` on POSIX (verified live on the operator host:
    ``C:/Users/senki/AppData/Roaming/uv/tools``), so a tool environment's
    ``sys.prefix`` always ends in the three segments ``uv``, ``tools``,
    ``<tool name>`` -- a plain ``uv sync`` project venv never does.
    """
    parts = Path(prefix).parts
    return len(parts) >= 3 and parts[-2] == "tools" and parts[-3] == "uv"


def _project_root() -> Path | None:
    """The checkout an editable install points at, when discoverable.

    An editable install resolves this module to the real source tree, so
    walking up from the package directory to the first parent whose
    ``pyproject.toml`` declares ``[project].name == 'charlie-work'`` finds the
    project root ``uv tool install --editable`` needs. Verifying the name is
    what stops a stray ``pyproject.toml`` above the checkout (some host tool's
    config file, an unrelated project) from being handed to the recovery
    command: installing *that* project would not repair charlie-work. In a
    non-editable install the package is copied into site-packages and no
    parent qualifies; callers fall back.
    """
    return _project_root_from(Path(__file__).resolve().parent)


def _project_root_from(start: Path) -> Path | None:
    """``_project_root`` over an explicit starting directory (test seam)."""
    for directory in (start, *start.parents):
        pyproject = directory / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError):
            continue
        if declared.get("project", {}).get("name") == _DIST_NAME:
            return directory
    return None


# The recovery command must not guess its target: with no discoverable project
# root, naming the tool environment directory (``uv tool install <dirname>``)
# would resolve through the package index to the *published* distribution, not
# the checkout an editable install tracks -- wrong project, wrong supply
# chain. The placeholder makes the operator fill in the one thing uv needs.
_RECOVERY_CHECKOUT_PLACEHOLDER = "<path to your charlie-work checkout>"


def _recovery_command(unmet: list[tuple[str, str, str | None]]) -> str:
    """The exact ``uv tool install`` command that repairs a stale tool env.

    ``--force`` reinstalls the tool itself; ``--refresh-package`` (one flag
    per unmet requirement -- the observed shape of the 2026-10-23 recovery:
    ``uv tool install --editable <checkout> --force --refresh-package
    ci-fleet``) is what makes uv re-resolve a dependency it would otherwise
    leave as-is. The install is always the editable one against the real
    checkout: when no project root is discoverable the checkout path is
    printed as an explicit placeholder instead of guessed from the
    environment's own directory name.
    """
    refresh = " ".join(f"--refresh-package {name}" for name, _, _ in unmet)
    root = _project_root()
    path = root if root is not None else _RECOVERY_CHECKOUT_PLACEHOLDER
    return f"uv tool install --editable {path} --force {refresh}".rstrip()


def _failure_message(unmet: list[tuple[str, str, str | None]]) -> str:
    """One actionable stderr message: what is missing, and how to recover.

    In a uv tool environment the recovery is reinstalling that tool
    environment; anywhere else the matching recovery is ``uv sync`` against
    the checkout that declares the versions listed above it.
    """
    lines = ["charlie cannot start: required dependencies are missing or too old.", ""]
    for name, text, installed in unmet:
        installed_text = installed or "<not installed>"
        lines.append(f"  {name}: declared {text}, installed {installed_text}")
    lines.append("")
    if _is_uv_tool_environment(sys.prefix):
        lines.extend(
            [
                "A uv tool environment tracks the checkout's source but never",
                "re-resolves its dependencies. Reinstall it to pick up what the",
                "checkout's pyproject.toml declares:",
                "",
                f"  {_recovery_command(unmet)}",
                "",
                "charlie does not run this for you: reinstalling a tool environment",
                "is a supply-chain-sensitive action that stays operator-initiated.",
            ]
        )
    else:
        lines.extend(
            [
                f"This is not a uv tool environment (prefix: {sys.prefix}).",
                "Re-sync the environment from the checkout that declares",
                "the versions listed above:",
                "",
                "  uv sync",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    """Console entry point: check declared requirements, then import the CLI.

    Unmet declared requirements print one actionable message to stderr and
    exit 2 (before `charlie_work.cli` -- which would die with a bare
    `ModuleNotFoundError` on the same environment -- is ever imported). A
    check that cannot run at all (no `charlie-work` metadata in this
    environment, unreadable requirement strings) fails OPEN to the import:
    this guard must never block a healthy start, it only arms a broken one.
    On success the return value is the CLI's own exit code, preserving every
    wire contract read off `main()`'s return value (issues #959 and #862).
    """
    try:
        unmet = _unmet_requirements()
    except Exception:  # noqa: BLE001 - a check failure above must fail open
        unmet = []
    if unmet:
        print(_failure_message(unmet), file=sys.stderr)
        sys.exit(_EXIT_UNMET_REQUIREMENTS)
    from charlie_work.cli import main as cli_main  # noqa: PLC0415 - lazy by design

    return cli_main()
