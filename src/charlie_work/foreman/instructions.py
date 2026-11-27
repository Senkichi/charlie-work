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
    """``$placeholder`` syntax extended to hyphenated partial stems.

    An identifier must start with a letter/underscore, may contain hyphens,
    and must end with a letter, digit, or underscore — so a sentence-final
    ``$section_issue-style.`` keeps its period outside the placeholder and a
    trailing ``-`` is never swallowed into the stem.
    """

    idpattern = r"(?a:[_a-z](?:[-_a-z0-9]*[_a-z0-9])?)"


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
        except (OSError, UnicodeDecodeError):
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
    except (OSError, UnicodeDecodeError) as exc:
        return Result(error=f"instructions.md: unreadable: {exc}")
    template = _SharedTemplate(text)
    partials = _shared_partials(fleet_dir)
    # Only ``$section_<stem>`` identifiers are placeholders this renderer
    # resolves (the ``prompts.py`` contract): any other ``$``-text — ``$HOME``,
    # ``$5`` — is literal and passes through ``safe_substitute`` untouched.
    missing = sorted(
        key
        for key in template.get_identifiers()
        if key.startswith("section_") and key not in partials
    )
    if missing:
        stem = missing[0][len("section_") :]
        return Result(error=f"instructions.md: shared partial {stem!r} not found")
    return Result(value=template.safe_substitute(partials))
