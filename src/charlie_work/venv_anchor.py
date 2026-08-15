"""Interpreter-anchored editable-``.pth`` guard for long-lived daemons (issue #974).

``ci_fleet_anchor`` and ``worktree.verify_shared_venv`` both derive their
expectation from *importable code* -- ``__file__`` walks or ``repo_root``
parameters that were themselves computed from ``__file__``. That expectation
travels through the same ``.pth`` files it is supposed to check: in the
2026-08-05 incident a scratch clone carried its own ``pyproject.toml`` and its
own ``../ci_runners`` sibling, so declared-vs-loaded agreed while the
supervisor ran unreviewed code. Containment cannot validate its own anchor.

The one identity a scratch clone cannot carry is the interpreter the scheduled
task launches: ``...\\charlie-work\\.venv\\Scripts\\python.exe``. ``sys.prefix``
of the running process therefore names the venv -- and the checkout above it --
independently of what any ``.pth`` file says. This module anchors there.

What is checked, and why the ``.pth`` files rather than import locations
------------------------------------------------------------------------

Worker test runs deliberately shadow the venv's editable install with
``PYTHONPATH`` pointing at a worktree's ``src``, so ``charlie_work.__file__``
legitimately resolves outside the main checkout in those processes. An
import-location comparison would report that as poisoning and block the fleet's
own test lane. The mutable artifact the incident actually corrupted is the
venv's editable ``.pth`` content, which is shared state independent of any one
process's ``sys.path`` -- so that is what gets read.

The allowed-target set is derived, never enumerated: any target inside the
interpreter's own checkout (which contains the venv and ``src``), or inside a
root declared by a ``[tool.uv.sources]`` ``path`` entry, resolved against the
interpreter-derived root. Any ``.pth`` path line resolving outside those roots
is a violation -- deriving what is covered fails closed, where matching
known-bad names would fail open on the next tool that writes a
differently-named file.

Honesty note: this guard is itself code reached through the tree it verifies.
A poisoned install of a revision that lacks the guard would not run it. The
realistic poison shape, though, is a near-HEAD scratch copy of this same repo,
which carries the guard and still fails it: the scratch ``.pth`` target
disagrees with the real interpreter's ``sys.prefix`` parent.

Abstention mirrors ``ci_fleet_anchor``: contexts where the anchor genuinely
answers nothing (not a venv, no ``pyproject.toml`` above the venv) proceed with
a stated reason rather than failing, because refusing there would convert a
check into an outage without evidence of anything being wrong.
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .worktree import _resolve_pth_line, _site_packages_dir

__all__ = ["VenvAnchorResult", "verify_interpreter_anchored_editables"]


@dataclass(frozen=True)
class VenvAnchorResult:
    """Outcome of the interpreter-anchored ``.pth`` check.

    ``ok=True, abstained=True`` means "could not anchor, proceeding" -- the
    caller may log ``detail`` but must not treat it as a verified-clean venv.
    ``ok=False`` is a positive violation and callers that gate daemon startup
    on this must refuse to start.
    """

    ok: bool
    detail: str
    abstained: bool = False


def _declared_path_source_roots(repo_root: Path) -> tuple[Path, ...]:
    """Every ``[tool.uv.sources]`` ``path`` entry, resolved against ``repo_root``.

    Structural failures (unreadable/renamed ``pyproject.toml``) propagate to
    the caller, which reports the exception as the abstention detail --
    same policy as ``ci_fleet_anchor``: breakage must not masquerade as a
    real answer.
    """
    with (repo_root / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    roots: list[Path] = []
    for entry in sources.values():
        declared = entry.get("path") if isinstance(entry, dict) else None
        if declared is not None:
            roots.append((repo_root / declared).resolve())
    return tuple(roots)


def verify_interpreter_anchored_editables(
    prefix: Path | None = None,
) -> VenvAnchorResult:
    """Check every editable ``.pth`` target against the interpreter's own venv.

    ``prefix`` is injectable for tests; production callers pass nothing and get
    ``sys.prefix``. Errors come back as values, never raised.
    """
    try:
        if prefix is None:
            if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
                return VenvAnchorResult(
                    ok=True,
                    detail="not running inside a virtualenv; nothing to anchor against",
                    abstained=True,
                )
            prefix = Path(sys.prefix)
        # resolve() first: worktree ``.venv`` entries on this box are Windows
        # junctions into the shared venv, and the checkout that owns the venv
        # is the junction's *target* parent, not the junction's.
        venv = prefix.resolve()
        repo_root = venv.parent
        if not (repo_root / "pyproject.toml").is_file():
            return VenvAnchorResult(
                ok=True,
                detail=f"no pyproject.toml above venv {venv}; cannot anchor",
                abstained=True,
            )

        site_packages = _site_packages_dir(venv)
        if site_packages is None:
            return VenvAnchorResult(
                ok=True,
                detail=f"could not locate site-packages under {venv}; cannot anchor",
                abstained=True,
            )

        allowed_roots = (repo_root, *_declared_path_source_roots(repo_root))

        violations: list[str] = []
        for pth in sorted(site_packages.glob("*.pth")):
            for raw_line in pth.read_text(encoding="utf-8", errors="replace").splitlines():
                target = _resolve_pth_line(site_packages, raw_line)
                if target == Path():
                    continue
                if any(target == root or target.is_relative_to(root) for root in allowed_roots):
                    continue
                violations.append(f"{pth.name} -> {target}")
        if violations:
            expected = ", ".join(str(root) for root in allowed_roots)
            return VenvAnchorResult(
                ok=False,
                detail=(
                    "editable .pth targets escape the interpreter's checkout: "
                    + "; ".join(violations)
                    + f" (allowed: {expected}). The venv at {venv} has been repointed "
                    "by an install run outside this checkout -- re-run "
                    "`uv sync --all-extras` from the checkout root before starting."
                ),
            )
        return VenvAnchorResult(
            ok=True,
            detail=f"all editable .pth targets resolve inside {repo_root} and its declared sources",
        )
    except Exception as exc:  # noqa: BLE001 -- guard must report, never crash startup
        return VenvAnchorResult(
            ok=True,
            detail=f"venv anchor check raised {type(exc).__name__}: {exc}",
            abstained=True,
        )
