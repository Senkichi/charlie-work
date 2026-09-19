"""Decide whether a worker prompt may tell the worker to use slash-command skills.

``worker.md`` used to state unconditionally that ``/create-branch``, ``/commit``,
``/preflight``, ``/push``, ``/create-pr`` and ``/complete`` are available, and its
implementation loop said to use them. They are not a property of the harness: they
exist only where the consumer ships ``<dir>/<name>/SKILL.md`` in a directory the
harness loads skills from (``HarnessCapabilities.skill_dirs``). A devin-shell worker
in a repo that ships none gets ``Skill "create-branch" not found`` back, having been
told the skill exists.

The prompt now carries the skills loop only when that is true. The sections that
differ live in ``worker_sections/skills/`` (an overlay, see ``prompt_sections``); this
module picks the overlay. Which skill names the loop needs is *derived* from the
overlay's own ``Available skills`` bullet list, so the declaration and the loop text
cannot drift apart: add a bullet and the prompt requires that skill too.

The check is conservative on purpose. Every declared skill must be present; a repo
shipping only some of them gets the plain git/gh loop. A false negative costs the worker
a shortcut it did not need; a false positive is the bug this module exists to prevent.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from .harnesses import HARNESS_REGISTRY
from .prompt_sections import section_variables

# The section-variant directory holding the skills-based loop text.
SKILLS_VARIANT = "skills"

# The partial whose bullet list declares which skills the loop uses.
DECLARATION_SECTION = "section_available_skills"

_SKILL_BULLET = re.compile(r"^- `/([a-z0-9][a-z0-9-]*)`", re.MULTILINE)


def declared_skills(search_dirs: Sequence[Path] = ()) -> tuple[str, ...]:
    """Skill names the skills-variant loop tells the worker to use, in declared order."""
    text = section_variables(tuple(search_dirs), variants=(SKILLS_VARIANT,)).get(
        DECLARATION_SECTION, ""
    )
    return tuple(dict.fromkeys(_SKILL_BULLET.findall(text)))


def provisioned_skills(repo_root: Path, skill_dirs: Iterable[str]) -> frozenset[str]:
    """Skill names ``repo_root`` ships under any of ``skill_dirs``."""
    names: set[str] = set()
    for skill_dir in skill_dirs:
        for manifest in (repo_root / skill_dir).glob("*/SKILL.md"):
            if manifest.is_file():
                names.add(manifest.parent.name)
    return frozenset(names)


def active_prompt_variants(
    worker_harness: str, repo_root: Path | None, search_dirs: Sequence[Path] = ()
) -> tuple[str, ...]:
    """Section variants to render a worker/rework prompt under.

    ``(SKILLS_VARIANT,)`` when ``worker_harness`` loads project skills and
    ``repo_root`` ships every skill the loop declares; otherwise ``()``, the plain
    git/gh loop that needs nothing beyond the shell.
    """
    capabilities = HARNESS_REGISTRY.get(worker_harness)
    if capabilities is None or not capabilities.skill_dirs or repo_root is None:
        return ()
    declared = declared_skills(search_dirs)
    if declared and set(declared) <= provisioned_skills(repo_root, capabilities.skill_dirs):
        return (SKILLS_VARIANT,)
    return ()
