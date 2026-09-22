"""CLI command layer for the private-slug ratchet gate (issue #1502).

This module is the command wrapper extracted from the ``cli.py`` monolith so
that new code does not land in an over-cap file (the file-size high-water-mark
ratchet, issue #1442, forbids growing ``cli.py`` past its recorded mark).  The
pure scanning logic lives in :mod:`charlie_work.private_slug_gate`; this module
owns the argparse subparser registration, the baseline-directory I/O, the
``git diff``/``git ls-files`` subprocess calls, and the
exit-code decision -- the same split ``cli.py``'s ``run_mojibake_check_command``
uses with :mod:`charlie_work.mojibake_gate`.

The baseline is the ``.private-slug-baseline/`` directory (issue #1802):

* ``slugs/<slug>`` -- one empty marker file per gated slug (the config).
* ``files/<repo-relative-path>.count`` -- recorded mention count per file;
  first line is a non-negative integer, remaining lines are ``#`` comments.

The check-mode baseline *increase* is derived from the diff itself
(:func:`charlie_work.ratchet_baseline.count_delta_in_diff`), so no
``git show <base>:...`` round-trip and no stored ``total`` that two
concurrent PRs would conflict on.

``cli`` is imported lazily *inside* the functions that need it
(``cli.bootstrap_command``, ``cli.run_captured``), never at module top level.
This is load-bearing for two reasons:

1. It breaks what would otherwise be a circular import (``cli`` imports this
   module to wire the subparser and dispatch; this module imports ``cli`` for
   the shared bootstrap and subprocess runner).  A top-level ``from . import
   cli`` would re-enter ``cli`` while it is still initialising.
2. ``python -m charlie_work.cli`` runs ``cli.py`` as ``__main__``, a *separate*
   module object from ``charlie_work.cli``.  A top-level ``from . import cli``
   here would trigger a fresh import of ``charlie_work.cli`` (not yet in
   ``sys.modules`` under that name), which would circularly try to import
   ``register_private_slug_check_subparser`` from this still-partially-loaded
   module and raise ``ImportError`` -- observed in
   ``test_cli_module_entrypoint::test_module_form_actually_executes``.  Deferring
   the import to call time means the module imports cleanly even under ``-m``;
   ``register_private_slug_check_subparser`` (the only thing ``--help`` calls)
   does not touch ``cli`` at all.

Deferring also keeps the test surface intact: tests monkeypatch
``cli.run_captured``, ``cli.find_repo_root`` and ``cli.load_layered_config`` on
the ``cli`` module object, and this module reads them off that same object at
call time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import ConfigError
from .private_slug_gate import count_slug_mentions_in_text, find_slug_mentions_in_diff
from .ratchet_baseline import (
    BaselineFormatError,
    count_delta_in_diff,
    load_count_baseline,
    load_set_baseline,
    write_count_baseline,
    write_set_baseline,
)
from .workflow import CommandResult

PRIVATE_SLUG_BASELINE_DIRNAME = ".private-slug-baseline"
_SLUGS_DIRNAME = "slugs"
_FILES_DIRNAME = "files"


def register_private_slug_check_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``private-slug-check`` subcommand on *subparsers*.

    Kept here rather than inline in ``cli.build_parser`` so the monolith does
    not absorb the subparser definition (file-size ratchet, issue #1442).
    """
    private_slug_check = subparsers.add_parser(
        "private-slug-check",
        help=(
            "CI gate (issue #1502): fail if the diff adds net-new mentions "
            "of configured private sibling-repo slugs in tracked files. "
            "Ratchet-style against .private-slug-baseline/: a PR that "
            "adds net-new mentions must raise the matching files/*.count "
            "entries by at least the net-new count (tamper-evident in diff "
            "review). Moves (remove + add) produce zero net-new and do not "
            "trigger the gate."
        ),
    )
    private_slug_check.add_argument(
        "--base",
        default="origin/main",
        help="Git ref to diff against (default: origin/main). Uses the "
        "two-dot diff (base..HEAD), same as mojibake-check, for the same "
        "shallow-clone reason.",
    )
    private_slug_check.add_argument(
        "--regenerate",
        action="store_true",
        help="Instead of checking, scan the working tree and rewrite "
        ".private-slug-baseline/files/ with current per-file mention counts. "
        "The slug list is read from the existing baseline directory's slugs/ "
        "entries (or from --slugs when the directory does not exist yet). Run "
        "this after removing mentions to tighten the ratchet, or after "
        "intentionally adding mentions to update the baseline.",
    )
    private_slug_check.add_argument(
        "--slugs",
        default=None,
        help="Comma-separated slug list for --regenerate when the baseline "
        "directory does not exist yet (e.g. --slugs slug-one,slug-two). Ignored "
        "in check mode (slugs come from the baseline directory).",
    )


def _load_slugs(baseline_dir: Path) -> list[str]:
    """Load the configured slug list from ``<baseline_dir>/slugs/``.

    Raises ``ConfigError`` (caught by the CLI's top-level handler) when the
    baseline directory or the slugs subdirectory is missing -- fail closed,
    matching the heartbeat-suppressions.yaml philosophy: a bad config can
    only ever add friction, never silently pass.
    """
    slugs_dir = baseline_dir / _SLUGS_DIRNAME
    try:
        return sorted(load_set_baseline(slugs_dir))
    except BaselineFormatError as exc:
        raise ConfigError(f"private-slug-check: {exc}") from exc


def _load_head_counts(baseline_dir: Path) -> dict[str, int]:
    """Load per-file mention counts from ``<baseline_dir>/files/``.

    A missing ``files/`` directory is a legitimate zero-mention baseline
    (the ratchet's ideal end state); a present-but-malformed one fails
    closed.
    """
    files_dir = baseline_dir / _FILES_DIRNAME
    if not files_dir.is_dir():
        return {}
    try:
        return load_count_baseline(files_dir)
    except BaselineFormatError as exc:
        raise ConfigError(f"private-slug-check: {exc}") from exc


def _regenerate_private_slug_baseline(
    repo_root: Path, slugs: list[str], baseline_dir: Path
) -> CommandResult:
    """Scan tracked files and rewrite the baseline directory's entries."""
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    ls_result = cli.run_captured(
        ["git", "ls-files"],
        cwd=repo_root,
        timeout_seconds=30,
    )
    if not ls_result.ok:
        return CommandResult(
            False,
            f"private-slug-check: could not list tracked files: "
            f"{ls_result.error or ls_result.stderr or 'git ls-files failed'}",
            {"slugs": slugs},
        )

    baseline_prefix = PRIVATE_SLUG_BASELINE_DIRNAME + "/"
    tracked_files = [
        f
        for f in ls_result.stdout.splitlines()
        if f and not f.startswith(baseline_prefix) and f != ".private-slug-baseline.json"
    ]

    files: dict[str, int] = {}
    total = 0
    for rel_path in tracked_files:
        abs_path = repo_root / rel_path
        try:
            text = abs_path.read_text(encoding="utf-8", errors="surrogateescape")
        except (OSError, UnicodeDecodeError):
            continue
        count = count_slug_mentions_in_text(text, slugs)
        if count > 0:
            files[rel_path] = count
            total += count

    # Per-entry writes (temp-file + replace inside write_count_baseline),
    # per CLAUDE.md's atomic state-write invariant. Stale entries are
    # removed so the directory tracks the live tree exactly.
    write_count_baseline(baseline_dir / _FILES_DIRNAME, files)
    write_set_baseline(baseline_dir / _SLUGS_DIRNAME, slugs)

    return CommandResult(
        True,
        f"private-slug-check: regenerated baseline ({total} mentions across "
        f"{len(files)} file(s), slugs={slugs})",
        {"slugs": slugs, "total": total, "files": files},
    )


def run_private_slug_check_command(args: argparse.Namespace) -> CommandResult:
    """CI gate (issue #1502): fail if the diff adds net-new private-slug mentions.

    Loads the slug list from ``.private-slug-baseline/slugs/`` and the
    per-file mention counts from ``.private-slug-baseline/files/`` at the
    repo root, runs ``git diff <base>..HEAD``, and scans every added
    and removed line for mentions of the configured slugs.  The gate fails
    when ``net_new = added - removed > 0`` AND the baseline entries' net
    increase in the same diff is smaller than ``net_new`` -- i.e. a PR
    that adds new mentions must also bump the baseline, and that bump is
    tamper-evident in the diff.  The baseline increase is derived from the
    diff's own ``.count`` entry changes
    (:func:`charlie_work.ratchet_baseline.count_delta_in_diff`), so there is
    no stored ``total`` for two PRs to conflict over and no second git
    call.

    Moves (remove from one location, add to another) produce zero net-new
    and pass regardless of the baseline, which is why the gate counts
    net-new rather than scanning added lines alone.  The baseline directory
    itself is excluded from the scan because it *lists* slugs as config.

    Uses a two-dot diff (``base..HEAD``) for the same shallow-clone reason
    as the mojibake gate.  If the baseline directory does not exist at the
    base ref (first PR introducing the gate), its entries show up as new
    files in the diff, so the whole directory counts as the increase --
    the same ``base_total = 0`` semantics the JSON baseline had.

    With ``--regenerate``, the command instead scans the working tree and
    rewrites the baseline directory with current per-file mention counts.
    The slug list is read from the existing ``slugs/`` entries (or from
    ``--slugs`` when the directory does not exist yet).

    Errors as values (per CLAUDE.md): git failures come back as
    ``CommandResult(ok=False)`` -- never raised -- so the CI step exits
    non-zero without a Python traceback.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    # Same opt-out as ast-equivalence-check (issue #1600): this command only
    # ever inspects or regenerates against the checkout it was invoked from
    # -- it mutates no orchestrator state.  bootstrap_command defaults to
    # redirecting a linked-worktree cwd to the shared main worktree root
    # (issue #648 state-safety), but that redirect makes check mode scan the
    # main worktree's ``base..HEAD`` diff instead of the invoking
    # worktree's -- in a linked worktree that diff is always empty, so the
    # gate reports a false clean -- and makes --regenerate scan the wrong
    # tree's files.
    ctx = cli.bootstrap_command(args, redirect_to_main_worktree=False)
    baseline_dir = ctx.repo_root / PRIVATE_SLUG_BASELINE_DIRNAME

    # --- --regenerate mode: scan tree, rewrite baseline entries ---
    if getattr(args, "regenerate", False):
        if baseline_dir.is_dir():
            slugs = _load_slugs(baseline_dir)
        else:
            raw = getattr(args, "slugs", None)
            if not raw:
                return CommandResult(
                    False,
                    "private-slug-check: --regenerate requires --slugs when "
                    "the baseline directory does not exist yet.",
                    {},
                )
            slugs = [s.strip() for s in raw.split(",") if s.strip()]
        return _regenerate_private_slug_baseline(ctx.repo_root, slugs, baseline_dir)

    # --- check mode: diff-based ratchet gate ---
    if not baseline_dir.is_dir():
        raise ConfigError(
            f"private-slug-check: baseline directory not found: {baseline_dir}. "
            f"Create it with 'charlie private-slug-check --regenerate --slugs <list>'."
        )
    slugs = _load_slugs(baseline_dir)
    if not slugs:
        return CommandResult(
            False,
            "private-slug-check: baseline directory has an empty slugs/ set -- "
            "nothing to check. Populate it or remove the gate.",
            {"slugs": slugs},
        )

    head_counts = _load_head_counts(baseline_dir)
    head_total = sum(head_counts.values())
    base = getattr(args, "base", "origin/main")

    diff_result = cli.run_captured(
        ["git", "diff", f"{base}..HEAD"],
        cwd=ctx.repo_root,
        timeout_seconds=60,
    )
    if not diff_result.ok:
        return CommandResult(
            False,
            f"private-slug-check: could not run git diff against {base}: "
            f"{diff_result.error or diff_result.stderr or 'git diff failed'}",
            {"base": base},
        )

    delta = find_slug_mentions_in_diff(
        diff_result.stdout,
        slugs,
        exclude_paths=frozenset(
            {".private-slug-baseline.json", PRIVATE_SLUG_BASELINE_DIRNAME + "/"}
        ),
    )

    baseline_increase = count_delta_in_diff(
        diff_result.stdout,
        f"{PRIVATE_SLUG_BASELINE_DIRNAME}/{_FILES_DIRNAME}",
    )

    data: dict[str, Any] = {
        "base": base,
        "slugs": slugs,
        "added_count": len(delta.added),
        "removed_count": len(delta.removed),
        "net_new": delta.net_new,
        "baseline_total": head_total,
        "baseline_increase": baseline_increase,
        "added_findings": [
            {"path": f.path, "line": f.line_number, "slug": f.slug, "content": f.content}
            for f in delta.added
        ],
    }

    if delta.net_new > 0 and baseline_increase < delta.net_new:
        lines = [f"  {f.path}:{f.line_number}: mentions '{f.slug}'" for f in delta.added]
        message = (
            f"private-slug-check: {delta.net_new} net-new private-slug mention(s) "
            f"in diff against {base}\n"
            + "\n".join(lines)
            + f"\nBaseline entries increased by {baseline_increase} but "
            f"{delta.net_new} net-new mention(s) were added. Raise the matching "
            f"{PRIVATE_SLUG_BASELINE_DIRNAME}/{_FILES_DIRNAME}/<path>.count "
            f"entries (or add new ones) to acknowledge the new mentions, "
            f"or remove the mentions. The baseline bump is tamper-evident in "
            f"diff review."
        )
        return CommandResult(False, message, data)

    return CommandResult(
        True,
        f"private-slug-check: clean ({delta.net_new} net-new, "
        f"{len(delta.added)} added, {len(delta.removed)} removed; "
        f"baseline {head_total})",
        data,
    )
