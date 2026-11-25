"""``instructions.md`` rendering with ``_shared`` partials.

A foreman's ``instructions.md`` may include any file under
``foremen/_shared/*.md`` by stem, spelled ``$section_<stem>`` — the same
include mechanism ``prompt_sections`` gives worker sections, adapted here
because partial stems may contain hyphens (``issue-style``), which
``string.Template``'s default identifier pattern rejects. A hyphen-tolerant
``idpattern`` makes ``$section_issue-style`` parse as one placeholder.

A missing instructions file reads as empty; a referenced-but-missing partial
is an error value naming it — the model call is never sent a briefing with a
silently dropped section.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from ..safe_path import contains
from . import Result
from .paths import instructions_path, shared_dir


class _SharedTemplate(Template):
    """``$placeholder`` syntax extended to hyphenated partial stems."""

    idpattern = r"(?a:[_a-z][-._a-z0-9]*)"


def _shared_partials(fleet_dir: Path) -> dict[str, str]:
    """Discover ``foremen/_shared/*.md`` as ``section_<stem>`` -> body.

    Files whose resolved path escapes ``_shared/`` (a symlink planted there)
    are skipped — referencing them then errors as missing, naming the stem.
    """
    root = shared_dir(fleet_dir)
    if not root.is_dir():
        return {}
    partials: dict[str, str] = {}
    for path in sorted(root.glob("*.md")):
        if not path.is_file() or not contains(root, path):
            continue
        try:
            partials[f"section_{path.stem}"] = path.read_text(encoding="utf-8")
        except OSError:
            continue
    return partials


def load_instructions(fleet_dir: Path, foreman_root: Path) -> Result[str]:
    """Render ``instructions.md`` with shared partials substituted.

    ``fleet_dir`` locates ``foremen/_shared``; ``foreman_root`` locates the
    instructions file. A missing file renders as an empty string. An
    unresolvable ``$placeholder`` — a partial that does not exist — is an
    error value naming it.
    """
    try:
        path = instructions_path(foreman_root)
    except ValueError as exc:
        return Result(error=str(exc))
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Result(value="")
    except OSError as exc:
        return Result(error=f"instructions.md: unreadable: {exc}")
    try:
        rendered = _SharedTemplate(text).substitute(_shared_partials(fleet_dir))
    except KeyError as exc:
        missing = exc.args[0]
        stem = missing[len("section_") :] if str(missing).startswith("section_") else missing
        return Result(error=f"instructions.md: shared partial {stem!r} not found")
    except ValueError as exc:
        return Result(error=f"instructions.md: {exc}")
    return Result(value=rendered)
