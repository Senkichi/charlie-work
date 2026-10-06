"""The test command a worker prompt tells the worker to run, derived from the consumer.

The worker/rework templates used to hardcode ``uv run --extra dev pytest`` for every
consumer. ``--extra dev`` only resolves ``[project.optional-dependencies]``, so a
consumer whose dev tools live in a PEP 735 ``[dependency-groups]`` group, or whose
``pyproject.toml`` is not at the repo root, hit a uv error before pytest even started, and
the prompt gave the worker no way to know the repository's own commands should win.

The prompt values are resolved here, once, with a fixed precedence:

1. ``dispatch.test_command`` -- the operator's per-repo runner prefix. It always wins:
   this is how a repo whose ``pyproject.toml`` is not at the root (say
   ``server/pyproject.toml``) says where its tests run.
2. What the consumer's own ``pyproject.toml`` declares: the extra or dependency group
   that actually lists ``pytest``. Nothing is guessed; a repo that declares no pytest
   anywhere yields no runner.
3. Neither: no command is invented. The prompt tells the worker to use the command the
   repository documents in ``CLAUDE.md`` / ``CONTRIBUTING.md``.

Only the *runner* is derived. The impacted test files are named by the worker (they can
live in nested packages such as ``tests/bundle/``), and ``-q --tb=short`` is appended.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .selection_wrapper import SelectionTarget, worker_command

PYTEST_FLAGS = "-q --tb=short"

# What the worker substitutes for the placeholder; the worker, not the orchestrator,
# knows which test files its diff touches.
IMPACTED_TESTS_PLACEHOLDER = "<impacted test files>"

# Rendered when no runner could be resolved. The targeted form sits inside a ```bash
# fence, so it is a comment line; the full-suite form sits inside a parenthetical.
UNRESOLVED_TARGETED = (
    "# No test command could be derived for this repository: run the one documented "
    "in CLAUDE.md / CONTRIBUTING.md, scoped to the impacted test files"
)
UNRESOLVED_FULL_SUITE = "the full-suite command documented in CLAUDE.md / CONTRIBUTING.md"

# uv installs the ``dev`` group by default; ``tool.uv.default-groups`` overrides that.
_UV_DEFAULT_GROUPS = ("dev",)
_PREFERRED_NAME = "dev"


@dataclass(frozen=True)
class WorkerTestCommands:
    """The two command strings the prompt templates splice in.

    ``targeted`` is the body of a fenced ``bash`` block; ``full_suite`` is dropped into a
    parenthetical (``... before pushing ($full_suite_command)``).
    """

    targeted: str
    full_suite: str
    # ``targeted`` runs ``ci-fleet test``, which selects the tests itself.
    selected: bool = False


# The step sentence above the ```bash block, and the execution contract, for each
# form of ``targeted``. The contract's markers are pinned by ``prompts.py``'s
# ``EXECUTION_CONTRACT_MARKERS`` / ``SELECTION_CONTRACT_MARKERS``.
TARGETED_STEP = (
    "Run the tests impacted by your change from the worktree root: the test file(s) you "
    "added or modified, plus `grep tests/` for every module/function/symbol your "
    "production diff touched and run every matching test file — not just the tests you "
    "wrote:"
)
SELECTED_STEP = (
    "Run exactly this command from the worktree root; it selects every test your diff "
    "can affect. To add test files of your own, put `--also <path> ...` before the `--`:"
)
_CI_SENTENCE = "CI runs the same selection or the full suite, and the nightly runs everything."


def _targeted_contract(full_suite: str) -> str:
    return (
        "**Execution contract (self-detect from your diff):** the default is the targeted "
        "command above (changed test files + the named deciders only). Only if the diff "
        "changes any public function signature/return shape, exception type/message "
        "consumed elsewhere, DB schema, or module re-export, run the **FULL suite** "
        f"locally at the final head before pushing ({full_suite}). The blast radius of a "
        "contract change is by definition outside the changed files. For all other diffs, "
        f"do NOT run the full suite locally — {_CI_SENTENCE}"
    )


def _selected_contract(full_suite: str) -> str:
    return (
        "**Execution contract (selection decides scope):** the command above is your whole "
        "local test run. `ci-fleet test` already widens for contract changes: a changed "
        "function signature, return shape or exception selects every test that ran the "
        "changed file, and a module-level or re-export change selects every test that "
        "imports the module. It runs the **FULL suite** itself whenever the change needs "
        "it (it prints `full` and the reason), so do not run a separate full suite "
        f"({full_suite}) unless that command could not start. {_CI_SENTENCE}"
    )


def _dist_name(requirement: str) -> str:
    """The PEP 503-normalized distribution name of a PEP 508 requirement string."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return re.sub(r"[-_.]+", "-", match.group(1)).lower() if match else ""


def _declares_pytest(requirements: object) -> bool:
    """True when ``requirements`` lists ``pytest`` itself (``pytest-cov`` does not count)."""
    if not isinstance(requirements, list):
        return False
    return any(isinstance(item, str) and _dist_name(item) == "pytest" for item in requirements)


def _group_declares_pytest(
    groups: Mapping[str, Any], name: str, seen: frozenset[str] = frozenset()
) -> bool:
    """True when dependency group ``name`` lists pytest, directly or via ``include-group``."""
    entries = groups.get(name)
    if name in seen or not isinstance(entries, list):
        return False
    if _declares_pytest(entries):
        return True
    return any(
        isinstance(entry, dict)
        and isinstance(entry.get("include-group"), str)
        and _group_declares_pytest(groups, entry["include-group"], seen | {name})
        for entry in entries
    )


def _prefer_dev(names: list[str]) -> str:
    """``dev`` when it is a candidate, else the first name. ``names`` is pre-sorted."""
    return _PREFERRED_NAME if _PREFERRED_NAME in names else names[0]


def _default_groups(uv_table: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Groups a bare ``uv run`` installs; ``None`` means every group (``"all"``)."""
    raw = uv_table.get("default-groups", _UV_DEFAULT_GROUPS)
    if raw == "all":
        return None
    if isinstance(raw, list):
        return tuple(item for item in raw if isinstance(item, str))
    return _UV_DEFAULT_GROUPS


def _load_pyproject(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        return None


def _table(value: object) -> dict[str, Any]:
    """``value`` when it is a TOML table, else an empty one.

    Every table read in this module goes through here, so a malformed key (a scalar or
    array where a table belongs) reads as absent instead of raising: this runs at the
    dispatch boundary, where an exception would fail the dispatch, not just the hint.
    """
    return value if isinstance(value, dict) else {}


def derive_pytest_runner(repo_root: Path) -> str | None:
    """The ``uv run`` prefix that puts pytest on the path, per ``repo_root/pyproject.toml``.

    Precedence: pytest as a core dependency, then an extra that lists it
    (``--extra``), then a dependency group that lists it (``--group``, omitted when
    a bare ``uv run`` already installs that group). ``None`` when there is no
    parseable ``pyproject.toml`` or none of those declare pytest -- the caller must
    not invent a command in that case.
    """
    data = _load_pyproject(repo_root / "pyproject.toml")
    if data is None:
        return None
    project = _table(data.get("project"))
    uv_table = _table(_table(data.get("tool")).get("uv"))
    default_groups = _default_groups(uv_table)

    if _declares_pytest(project.get("dependencies")):
        return "uv run pytest"

    extras = _table(project.get("optional-dependencies"))
    extra_names = sorted(name for name, reqs in extras.items() if _declares_pytest(reqs))
    if extra_names:
        return f"uv run --extra {_prefer_dev(extra_names)} pytest"

    # uv's legacy ``[tool.uv] dev-dependencies`` is the ``dev`` group by another name.
    groups = dict(_table(data.get("dependency-groups")))
    legacy = uv_table.get("dev-dependencies")
    if _declares_pytest(legacy):
        groups.setdefault(_PREFERRED_NAME, legacy)
    group_names = sorted(name for name in groups if _group_declares_pytest(groups, name))
    if not group_names:
        return None
    if default_groups is None or any(name in default_groups for name in group_names):
        return "uv run pytest"
    return f"uv run --group {_prefer_dev(group_names)} pytest"


def resolve_test_commands(
    configured_runner: str,
    repo_root: Path | None,
    *,
    selection: SelectionTarget | None = None,
) -> WorkerTestCommands:
    """Resolve the prompt's test commands: config override, then derivation, then neither.

    With a ``selection`` target the targeted command is ``ci-fleet test`` around the
    runner; without one (or with no runner to wrap) it is today's hand-picked form.
    """
    runner = configured_runner.strip()
    if not runner and repo_root is not None:
        runner = derive_pytest_runner(repo_root) or ""
    if not runner:
        return WorkerTestCommands(targeted=UNRESOLVED_TARGETED, full_suite=UNRESOLVED_FULL_SUITE)
    full_suite = f"`{runner} {PYTEST_FLAGS}`"
    if selection is not None:
        return WorkerTestCommands(
            targeted=worker_command(selection, runner, PYTEST_FLAGS),
            full_suite=full_suite,
            selected=True,
        )
    return WorkerTestCommands(
        targeted=f"{runner} {IMPACTED_TESTS_PLACEHOLDER} {PYTEST_FLAGS}",
        full_suite=full_suite,
    )


def prompt_test_command_values(
    configured_runner: str,
    repo_root: Path | None,
    *,
    selection: SelectionTarget | None = None,
) -> Mapping[str, str]:
    """The ``render_prompt`` values every worker/rework writer supplies for the test command."""
    commands = resolve_test_commands(configured_runner, repo_root, selection=selection)
    if commands.selected:
        step, contract = SELECTED_STEP, _selected_contract(commands.full_suite)
    else:
        step, contract = TARGETED_STEP, _targeted_contract(commands.full_suite)
    return {
        "targeted_test_command": commands.targeted,
        "full_suite_command": commands.full_suite,
        "test_step_instruction": step,
        "test_execution_contract": contract,
    }
