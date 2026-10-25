"""PR-relative interdiff built from archived per-round review diffs (issue #2837).

When a reviewed PR's head moves with a rework push, the next review packet
carried GitHub's three-dot ``gh.compare_diff(prior_head, new_head)`` interdiff.
That comparison includes every commit the rework merged from the base branch,
so after a base merge the interdiff was noisier than the PR itself and
reviewers rightly ignored it -- and every re-review re-read the full diff from
scratch.

This module builds the interdiff from the *differences between two GitHub PR
diffs* instead. Both inputs are PR-relative to begin with (each is relative to
its own merge-base), so base-merge noise can never appear:

* ``prior_diff`` is the last round's ``diff.patch``, archived into
  ``prs/pr-N/rounds/round-K/diff.patch`` by ``record_review``.
* ``new_diff`` is the current packet's ``prs/pr-N/diff.patch``.

A file whose per-file patch is unchanged once ``@@`` hunk-header line numbers
are normalised is listed as unchanged since the prior round; a file whose patch
is new or changed gets its full per-file patch in the interdiff; a file dropped
from the PR is listed by name only.

``classify_review_round_kind`` resolves the ``review_round_kind`` measurement
value (``first`` / ``incremental`` / ``fallback``) for one packet build, and
``count_interdiff`` reads the interdiff artifact's file/line counts back for
the dispatch-side ``review_round_kind`` event payload.

Failure posture mirrors the existing interdiff generation: every read here
returns errors as values (``None`` / empty results), never raises, so a missing
archive or unreadable diff degrades the caller to the legacy fallback instead
of blocking packet generation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .review_decision import _round_history_entries
from .unified_diff import iter_diff_files

# Normalises an ``@@ -a,b +c,d @@`` hunk header to ``@@ @@<context>`` so that a
# patch whose only difference is shifted line numbers (a pure base merge above
# the touched region) compares equal to its previous round's patch.
_HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@(.*)$")


def _normalize_hunk_line(line: str) -> str:
    match = _HUNK_HEADER_RE.match(line)
    if match is None:
        return line
    return f"@@ @@{match.group(1)}"


@dataclass(frozen=True)
class PRInterdiff:
    """The PR-relative interdiff between two PR diffs.

    ``changed`` (in the new diff's file order) and ``unchanged`` (sorted) are
    new-diff paths; ``removed`` (sorted) are paths present in the prior diff
    and absent from the new one. ``text`` is the concatenation of the changed
    files' full per-file patches -- a valid diff by itself -- or ``""`` when
    every file is unchanged.
    """

    changed: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    text: str = ""

    @property
    def file_count(self) -> int:
        """How many files carry a patch in ``text``."""
        return len(self.changed)

    @property
    def line_count(self) -> int:
        """How many lines ``text`` holds."""
        if not self.text:
            return 0
        return len(self.text.splitlines())


def split_diff_sections(diff: str) -> dict[str, str]:
    """Split a unified diff into per-file sections keyed by the ``+++ b/`` path.

    Each value is the full section, from its leading ``diff --git`` line
    through its last hunk line (metadata lines like ``index``/``---`` remain --
    unlike ``iter_diff_files``, which is used for *comparison* bodies only).
    The first section for a path wins (a rename appears keyed by its ``b/``
    path once, like a real diff does).
    """
    sections: dict[str, str] = {}
    for part in diff.split("\ndiff --git"):
        if not part.strip():
            continue
        if not part.startswith("diff --git"):
            part = "diff --git" + part
        new_path: str | None = None
        for line in part.splitlines():
            if line.startswith("+++ b/"):
                new_path = line[6:]
                break
        if new_path is None:
            continue
        sections.setdefault(new_path, part)
    return sections


def _normalized_bodies(diff: str) -> dict[str, str]:
    bodies: dict[str, str] = {}
    for filename, _is_new_file, hunk_lines in iter_diff_files(diff):
        bodies.setdefault(filename, "\n".join(_normalize_hunk_line(line) for line in hunk_lines))
    return bodies


def build_pr_interdiff(prior_diff: str, new_diff: str) -> PRInterdiff:
    """Build the per-file PR-relative interdiff between two full PR diffs."""
    prior_sections = split_diff_sections(prior_diff)
    new_sections = split_diff_sections(new_diff)
    prior_normalized = _normalized_bodies(prior_diff)
    new_normalized = _normalized_bodies(new_diff)

    changed: list[str] = []
    unchanged: list[str] = []
    chunks: list[str] = []
    for path, section in new_sections.items():
        if new_normalized.get(path) != prior_normalized.get(path):
            changed.append(path)
            chunks.append(section)
        else:
            unchanged.append(path)
    removed = sorted(path for path in prior_sections if path not in new_sections)
    text = "\n".join(chunks)
    if text and not text.endswith("\n"):
        text += "\n"
    return PRInterdiff(
        changed=tuple(changed),
        unchanged=tuple(sorted(unchanged)),
        removed=tuple(removed),
        text=text,
    )


def read_optional(path: Path) -> str | None:
    """Read ``path`` as text, or ``None`` when it is missing/unreadable."""
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def archive_round_diff(pr_dir: Path, round_dir: Path) -> Path | None:
    """Copy the packet's live ``diff.patch`` into a round's archive dir.

    Returns the archived path, or ``None`` when the packet has no diff of its
    own (legacy round). Atomic via rework_prompts' helper -- the archive dir
    is polled by readers (import inside the function to avoid the workflow-
    package cycle).
    """
    live_diff = pr_dir / "diff.patch"
    if not live_diff.exists():
        return None
    from .rework_prompts import _write_text_atomic

    _write_text_atomic(round_dir / "diff.patch", live_diff.read_text(encoding="utf-8"))
    return round_dir / "diff.patch"


def prior_round_diff_path(pr_dir: Path, prior_decision: dict[str, Any]) -> Path | None:
    """Return the latest archived round's ``diff.patch``, or ``None``.

    Reads only the ``rounds/`` archive (the same source
    ``_round_history_entries`` renders the prior-review section from), never
    ``state.json``. Legacy rounds (archived before issue #2837) and the
    synthetic-fallback round have no ``diff.patch``; ``None`` is the caller's
    signal to fall back to ``gh.compare_diff``.
    """
    entries = _round_history_entries(pr_dir / "rounds", fallback_decision=prior_decision)
    if not entries:
        return None
    candidate = pr_dir / "rounds" / f"round-{entries[-1][0]}" / "diff.patch"
    return candidate if candidate.exists() else None


def classify_review_round_kind(
    pr_dir: Path,
    prior_decision: dict[str, Any] | None,
    new_head_sha: str | None,
    *,
    incremental_enabled: bool,
    is_round2_review: bool,
) -> str | None:
    """Resolve the ``review_round_kind`` measurement value for one packet build.

    ``None`` when the feature's kill switch is off (today's behaviour; the
    caller stamps nothing). ``"first"`` when no prior terminal review exists.
    ``"incremental"`` when the moved-head packet can build the PR-relative
    interdiff from the latest archived round's diff. ``"fallback"`` when a
    prior review exists but the packet cannot operate incrementally: no
    archived diff (legacy round), an unreadable archive, or a same-head
    re-review (no interdiff exists at all).
    """
    if not incremental_enabled:
        return None
    if not is_round2_review:
        return "first"
    prior_head_sha = (
        str(prior_decision.get("reviewed_head_sha")) if isinstance(prior_decision, dict) else ""
    )
    if prior_head_sha and new_head_sha and prior_head_sha == new_head_sha:
        return "fallback"
    if prior_round_diff_path(pr_dir, prior_decision or {}) is not None:
        return "incremental"
    return "fallback"


def count_interdiff(pr_dir: Path) -> tuple[int, int]:
    """Return ``(file_count, line_count)`` of the current packet's interdiff.

    Reads ``prs/pr-N/interdiff.patch`` -- the artifact ``_build_prior_review_
    section`` wrote moments ago, whether it came from the archive-based
    builder or the ``gh.compare_diff`` fallback. ``(0, 0)`` when absent.
    """
    text = read_optional(pr_dir / "interdiff.patch")
    if not text or not text.strip():
        return (0, 0)
    return (len(split_diff_sections(text)), len(text.splitlines()))


def render_incremental_review_procedure(diff_path: str) -> str:
    """Render the ``$incremental_review_procedure`` slot for a template.

    Empty string on a non-incremental round (the template renders the slot
    away). The text is static across rounds and lanes; the round-relative
    specifics (the interdiff path, the unchanged/removed file lists) are
    rendered into ``$prior_review_section`` by ``_build_prior_review_section``.
    """
    return f"""
## Incremental review procedure

This is an incremental re-review round. The interdiff referenced in the prior
review section above is PR-relative: it is built from the previous round's
archived diff (itself a full PR diff relative to its own merge-base), so it
contains no base-merge noise. Proceed as follows:

1. Re-check every finding in the prior review section above and give each one
   an explicit status in your summary: resolved, still present, or regressed.
   Every prior finding gets an explicit status -- do not silently drop any.
2. Review each file in `interdiff.patch` (the files changed since the last
   round) in full.
3. Open the full `{diff_path}` only to understand context the changed files
   depend on. Do not re-review the unchanged files.
4. Unchanged files keep the earlier round's assessment unless a changed file
   affects them; when one does, say so in your summary.
"""


# ---------------------------------------------------------------------------
# Shared packet-facts / turn-budget plumbing (both review lanes and both
# dispatch claim sites consume these -- issue #2837).
# ---------------------------------------------------------------------------


def round_kind_from_record(pr_state: Mapping[str, Any]) -> str | None:
    """The round kind stamped onto a PR record by the packet builder.

    ``None`` when the stamp is absent -- the ``incremental_review`` kill
    switch is off, or the record predates the stamp.
    """
    value = pr_state.get("review_round_kind")
    return value if isinstance(value, str) else None


def round_info_from_record(pr_state: Mapping[str, Any]) -> dict[str, Any]:
    """The round-kind + interdiff-size fields on a PR record, for dispatch
    events. ``review_round_kind`` is ``None`` when unstamped."""
    return {
        "review_round_kind": round_kind_from_record(pr_state),
        "review_interdiff_files": int(pr_state.get("review_interdiff_files", 0) or 0),
        "review_interdiff_lines": int(pr_state.get("review_interdiff_lines", 0) or 0),
    }


def round_kind_stamp(
    round_kind: str | None, interdiff_files: int, interdiff_lines: int
) -> dict[str, Any]:
    """The PR-record stamp fields; empty when the kind is ``None``."""
    if round_kind is None:
        return {}
    return {
        "review_round_kind": round_kind,
        "review_interdiff_files": interdiff_files,
        "review_interdiff_lines": interdiff_lines,
    }


def base_turn_cap(round_kind: str | None, cfg: Any) -> int:
    """Base reviewer turn budget for one round kind (issue #2837).

    An incremental round runs ``cfg.incremental_review_max_turns``; ``0`` in
    that knob, a ``fallback``/``first``/unstamped kind, or the kill switch,
    gives today's flat ``cfg.review_max_turns``. The #1439 structure/miss
    escalation multiplies on top at the claim site.
    """
    if (
        round_kind == "incremental"
        and bool(cfg.incremental_review)
        and int(cfg.incremental_review_max_turns) > 0
    ):
        return int(cfg.incremental_review_max_turns)
    return int(cfg.review_max_turns)


def packet_round_facts(
    pr_dir: Path,
    prior_decision: dict[str, Any] | None,
    new_head_sha: str | None,
    *,
    cfg: Any,
) -> tuple[str | None, int, int, str]:
    """Resolve one packet build's round fact tuple and
    ``(kind, interdiff_files, interdiff_lines, procedure_text)``.

    ``procedure_text`` is the ``$incremental_review_procedure`` render (its
    ``diff_path`` argument resolved by the caller's template convention) --
    non-empty only for an ``incremental`` kind.
    """
    round_kind = classify_review_round_kind(
        pr_dir,
        prior_decision,
        new_head_sha,
        incremental_enabled=bool(cfg.incremental_review),
        is_round2_review=isinstance(prior_decision, dict)
        and prior_decision.get("decision") not in ("pending", None, "missing")
        and bool(prior_decision.get("reviewed_head_sha")),
    )
    if round_kind is None:
        return (None, 0, 0, "")
    interdiff_files, interdiff_lines = count_interdiff(pr_dir)
    procedure = (
        render_incremental_review_procedure(str(pr_dir / "diff.patch"))
        if round_kind == "incremental"
        else ""
    )
    return (round_kind, interdiff_files, interdiff_lines, procedure)
