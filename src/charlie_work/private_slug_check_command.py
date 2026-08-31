"""CLI command layer for the private-slug ratchet gate (issue #1502).

This module is the command wrapper extracted from the ``cli.py`` monolith so
that new code does not land in an over-cap file (the file-size high-water-mark
ratchet, issue #1442, forbids growing ``cli.py`` past its recorded mark).  The
pure scanning logic lives in :mod:`charlie_work.private_slug_gate`; this module
owns the argparse subparser registration, the baseline-file I/O, the
``git diff``/``git show``/``git ls-files`` subprocess calls, and the
exit-code decision -- the same split ``cli.py``'s ``run_mojibake_check_command``
uses with :mod:`charlie_work.mojibake_gate`.

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
import json
from pathlib import Path
from typing import Any

from .config import ConfigError
from .private_slug_gate import count_slug_mentions_in_text, find_slug_mentions_in_diff
from .workflow import CommandResult

PRIVATE_SLUG_BASELINE_FILENAME = ".private-slug-baseline.json"


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
            "Ratchet-style against .private-slug-baseline.json: a PR that "
            "adds net-new mentions must bump the baseline total by at least "
            "the net-new count (tamper-evident in diff review). Moves "
            "(remove + add) produce zero net-new and do not trigger the gate."
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
        ".private-slug-baseline.json with current per-file mention counts. "
        "The slug list is read from the existing baseline file (or from "
        "--slugs when the file does not exist yet). Run this after removing "
        "mentions to tighten the ratchet, or after intentionally adding "
        "mentions to update the baseline.",
    )
    private_slug_check.add_argument(
        "--slugs",
        default=None,
        help="Comma-separated slug list for --regenerate when the baseline "
        "file does not exist yet (e.g. --slugs slug-one,slug-two). Ignored "
        "in check mode (slugs come from the baseline file).",
    )


def _load_baseline_file(path: Path) -> dict[str, Any]:
    """Load and parse the private-slug baseline JSON file.

    Returns the parsed dict.  Raises ``ConfigError`` (caught by the CLI's
    top-level handler) if the file is missing or malformed -- fail closed,
    matching the heartbeat-suppressions.yaml philosophy: a bad config can
    only ever add friction, never silently pass.
    """
    if not path.exists():
        raise ConfigError(
            f"private-slug-check: baseline file not found: {path}. "
            f"Create it with 'charlie private-slug-check --regenerate --slugs <list>'."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"private-slug-check: malformed baseline file {path}: {exc}") from exc


def _regenerate_private_slug_baseline(
    repo_root: Path, slugs: list[str], baseline_path: Path
) -> CommandResult:
    """Scan tracked files and rewrite the baseline file with current counts."""
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

    tracked_files = [
        f for f in ls_result.stdout.splitlines() if f and f != PRIVATE_SLUG_BASELINE_FILENAME
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

    baseline = {
        "version": 1,
        "slugs": slugs,
        "files": dict(sorted(files.items())),
        "total": total,
    }

    # Atomic write (temp-file + replace), per CLAUDE.md invariant.
    tmp = baseline_path.with_suffix(baseline_path.suffix + ".tmp")
    tmp.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(baseline_path)

    return CommandResult(
        True,
        f"private-slug-check: regenerated baseline ({total} mentions across "
        f"{len(files)} file(s), slugs={slugs})",
        {"slugs": slugs, "total": total, "files": files},
    )


def run_private_slug_check_command(args: argparse.Namespace) -> CommandResult:
    """CI gate (issue #1502): fail if the diff adds net-new private-slug mentions.

    Loads the slug list and baseline total from ``.private-slug-baseline.json``
    at the repo root, runs ``git diff <base>..HEAD``, and scans every added
    and removed line for mentions of the configured slugs.  The gate fails
    when ``net_new = added - removed > 0`` AND the baseline ``total`` did not
    increase by at least ``net_new`` between *base* and HEAD -- i.e. a PR
    that adds new mentions must also bump the baseline, and that bump is
    tamper-evident in the diff.

    Moves (remove from one location, add to another) produce zero net-new
    and pass regardless of the baseline, which is why the gate counts
    net-new rather than scanning added lines alone.  The baseline file
    itself is excluded from the scan because it *lists* slugs as config.

    Uses a two-dot diff (``base..HEAD``) for the same shallow-clone reason
    as the mojibake gate.  The baseline total at *base* is read via
    ``git show <base>:.private-slug-baseline.json``; if the file does not
    exist at *base* (first PR introducing the gate), the base total is 0
    and any net-new mentions require a matching baseline bump.

    With ``--regenerate``, the command instead scans the working tree and
    rewrites the baseline file with current per-file mention counts.  The
    slug list is read from the existing baseline (or from ``--slugs`` when
    the file does not exist yet).

    Errors as values (per CLAUDE.md): git failures come back as
    ``CommandResult(ok=False)`` -- never raised -- so the CI step exits
    non-zero without a Python traceback.
    """
    from . import cli  # deferred: see module docstring (circular-import / -m guard)

    ctx = cli.bootstrap_command(args)
    baseline_path = ctx.repo_root / PRIVATE_SLUG_BASELINE_FILENAME

    # --- --regenerate mode: scan tree, rewrite baseline ---
    if getattr(args, "regenerate", False):
        if baseline_path.exists():
            existing = _load_baseline_file(baseline_path)
            slugs = existing.get("slugs", [])
        else:
            raw = getattr(args, "slugs", None)
            if not raw:
                return CommandResult(
                    False,
                    "private-slug-check: --regenerate requires --slugs when "
                    "the baseline file does not exist yet.",
                    {},
                )
            slugs = [s.strip() for s in raw.split(",") if s.strip()]
        return _regenerate_private_slug_baseline(ctx.repo_root, slugs, baseline_path)

    # --- check mode: diff-based ratchet gate ---
    baseline = _load_baseline_file(baseline_path)
    slugs = baseline.get("slugs", [])
    if not slugs:
        return CommandResult(
            False,
            "private-slug-check: baseline file has empty 'slugs' list -- "
            "nothing to check. Populate it or remove the gate.",
            {"slugs": slugs},
        )

    head_total = baseline.get("total", 0)
    base = getattr(args, "base", "origin/main")

    # Read the baseline total at the base ref to compute the bump.
    base_show = cli.run_captured(
        ["git", "show", f"{base}:{PRIVATE_SLUG_BASELINE_FILENAME}"],
        cwd=ctx.repo_root,
        timeout_seconds=30,
    )
    if base_show.ok:
        try:
            base_baseline = json.loads(base_show.stdout)
            base_total = base_baseline.get("total", 0)
        except json.JSONDecodeError:
            base_total = 0
    else:
        # File did not exist at base (first PR introducing the gate).
        base_total = 0

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
        exclude_paths=frozenset({PRIVATE_SLUG_BASELINE_FILENAME}),
    )

    baseline_increase = head_total - base_total

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
            + f"\nBaseline total increased by {baseline_increase} but {delta.net_new} "
            f"net-new mention(s) were added. Bump the 'total' (and per-file counts) "
            f"in {PRIVATE_SLUG_BASELINE_FILENAME} to acknowledge the new mentions, "
            f"or remove the mentions. The baseline bump is tamper-evident in diff review."
        )
        return CommandResult(False, message, data)

    return CommandResult(
        True,
        f"private-slug-check: clean ({delta.net_new} net-new, "
        f"{len(delta.added)} added, {len(delta.removed)} removed; "
        f"baseline {head_total})",
        data,
    )
