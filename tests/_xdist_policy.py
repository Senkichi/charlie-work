"""Small-run serial policy for the ``-n auto`` addopts default (HS-CW-1).

``pyproject.toml`` puts ``-n auto --dist=load`` in ``addopts`` so a bare
``pytest`` distributes. Booting xdist workers (each re-imports and re-collects)
costs 2-4 s, more than a run of one to three files saves, so
``tests/conftest.py`` answers xdist's ``pytest_xdist_auto_num_workers`` hook
from here: ``0`` (serial, no xdist session) for a small explicit run, ``None``
(defer to xdist's own default, which reads ``PYTEST_XDIST_AUTO_NUM_WORKERS``)
for everything else. xdist calls the hook only for ``-n auto``/``-n logical``,
and an explicit ``-n``/``--numprocesses`` on the command line is honoured too,
so a run that names its worker count is never second-guessed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

SMALL_RUN_MAX_ARGS = 3


def _explicit_numprocesses(invocation_args: Sequence[str]) -> bool:
    for arg in invocation_args:
        if arg == "--numprocesses" or arg.startswith("--numprocesses="):
            return True
        if arg.startswith("-n") and not arg.startswith("--"):
            return True
    return False


def _is_file_target(arg: str, invocation_dir: Path) -> bool:
    path = Path(arg.split("::", 1)[0])
    if not path.is_absolute():
        path = invocation_dir / path
    return path.is_file()


def small_run_workers(
    args: Sequence[str],
    invocation_dir: Path,
    invocation_args: Sequence[str],
    *,
    from_command_line: bool,
) -> int | None:
    """``0`` (run serially) for 1-3 explicit file/nodeid targets, else ``None``.

    ``args`` are pytest's resolved positional arguments; ``from_command_line``
    is False when they came from ``testpaths`` (a full run).
    """
    if not from_command_line or _explicit_numprocesses(invocation_args):
        return None
    if not 1 <= len(args) <= SMALL_RUN_MAX_ARGS:
        return None
    if all(_is_file_target(arg, invocation_dir) for arg in args):
        return 0
    return None
