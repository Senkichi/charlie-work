"""Shared-prefix shorthand citation resolution (issue #1761).

Extracted from ``cross_repo_gate.py`` so the gate module stays under its
file-size high-water mark (issue #1442) — this module holds the shorthand
machinery; the gate's classification pipeline owns the verdicts.

The shorthand convention: a comma- or newline-separated run of backtick
spans that spells a directory out once and then repeats only the remainder,
prefixed with ``/`` or ``...``::

    `logs/nightly_monitor/.../.credentials.json`, `/.claude.json`, `/sessions/30392.json`
    `logs/nightly_monitor/.../checkpoint_A.json:1`, `.../checkpoint_B.json:1`

No file is literally named ``/b.json`` or ``.../c.json``, so without
resolution the shorthand items extract as bogus missing candidates that
survive every neutralization arm of the gate and escalate otherwise
well-scoped single-repo issues as ``cross_repo_target``.

Two neutralization shapes live here:

- **All-dots first segment of three or more dots** (``...``/longer):
  neutral by construction — never a real path segment — whether or not
  list context is available (:func:`_is_dotdot_shorthand`). ``..`` is
  deliberately excluded: it is the real parent-directory segment, so
  ``../sibling-repo/x.py`` is a genuine cross-repo citation, not
  shorthand, and still escalates.
- **Leading-separator run continuation** (``/x`` or ``/x/y`` following a
  comma/whitespace-separated backtick-span run): resolved against the
  nearest preceding non-shorthand span's directory prefix
  (:func:`_resolve_list_shorthand`), so the item behaves exactly as if the
  full path had been spelled out. A standalone leading-``/`` candidate
  returns ``None`` — it is a genuine absolute path cited on its own terms.
"""

from __future__ import annotations

import re

# A candidate whose first non-separator segment is three or more dots
# (``...``, longer) is a shorthand continuation marker, never a real path
# segment — no file or directory is literally named ``...``. Neutral by
# construction, whether or not list context is available. ``..`` is
# deliberately NOT shorthand: it is the real parent-directory segment, so
# ``../sibling-repo/x.py`` citations still reach the pass/escalate
# decision exactly as they did before this module existed.
_DOTDOT_FIRST_SEGMENT_RE = re.compile(r"[\\/]*\.{3,}(?:[\\/]|$)")

# Characters permitted between backtick spans in a shared-prefix citation
# run: commas and whitespace, per the convention
# `` `dir/sub/a.json`,\n`/b.json` `` (issue #1761). Anything else — a word,
# a colon, a bullet marker — ends the run: a candidate separated from the
# previous span by prose is cited on its own terms, not as a continuation.
_RUN_SEPARATOR_CHARS = " \t\r\n,"


def _is_dotdot_shorthand(candidate: str) -> bool:
    """Return ``True`` when *candidate*'s first non-separator segment is
    three or more dots (``...``, longer) — the shared-prefix shorthand
    marker (issue #1761).

    An all-dots leading segment of three or more dots can never name a
    real file, so the candidate is neutral by construction, whether or not
    it sits in a backtick-span run. ``..`` is deliberately excluded: it is
    the real parent-directory segment — ``../sibling-repo/x.py`` is a
    genuine cross-repo citation that must still reach the pass/escalate
    decision, not a shorthand marker.
    """
    return bool(_DOTDOT_FIRST_SEGMENT_RE.match(candidate))


def _backtick_run_anchor_dir(stripped: str, start: int) -> str | None:
    """Return the directory prefix a shared-prefix shorthand item resolves
    against, or ``None`` when the candidate at ``start`` does not continue a
    backtick-span run (issue #1761).

    A "run" is a sequence of backtick-quoted spans separated only by commas
    and whitespace — the shared-prefix citation shape
    (`` `dir/sub/a.json`, `/b.json`, `/c.json` ``). The anchor is the
    directory part of the nearest preceding span in the run whose content
    is a non-shorthand path: shorthand spans (leading ``/`` or an all-dots
    first segment) and spans with no separator are skipped, so in
    `` `dir/a.json`, `/.claude.json`, `/sessions/30392.json` `` the last
    item resolves against ``dir``'s prefix rather than ``/.claude.json``'s
    empty one.
    """
    # Locate the opening backtick of the candidate's own span. ``_TICK_PATH``
    # matches include both ticks, so ``stripped[start]`` is the opener; a
    # ``_REL_PATH`` match that began just inside the span starts one
    # character after it.
    if stripped[start : start + 1] == "`":
        tick = start
    elif stripped[start - 1 : start] == "`":
        tick = start - 1
    else:
        return None
    pos = tick
    while True:
        i = pos - 1
        while i >= 0 and stripped[i] in _RUN_SEPARATOR_CHARS:
            i -= 1
        if i < 0 or stripped[i] != "`":
            return None
        opener = stripped.rfind("`", 0, i)
        if opener < 0:
            return None
        content = re.sub(r":\d+$", "", stripped[opener + 1 : i])
        if not content.startswith(("/", "\\")) and not _is_dotdot_shorthand(content):
            sep = max(content.rfind("/"), content.rfind("\\"))
            if sep > 0:
                return content[:sep]
        pos = opener


def _resolve_list_shorthand(candidate: str, stripped: str, start: int) -> str | None:
    """Resolve a leading-separator shorthand continuation to the full path
    it denotes (issue #1761), or return ``None`` when the candidate is not
    shorthand.

    The resolved path is the directory prefix of the nearest preceding
    non-shorthand span in the candidate's comma/whitespace-separated
    backtick-span run, plus the candidate's own segments:
    `` `logs/run/a.json`, `/sessions/30392.json` `` resolves to
    ``logs/run/sessions/30392.json``. The shorthand item then behaves
    exactly as if the full path had been spelled out — a resolved path
    that is gitignored or exists is neutral/pass evidence, a resolved path
    that is missing still escalates.

    A leading-separator candidate with no preceding backtick span returns
    ``None``: cited standalone it is a genuine absolute path (``/home/
    user/repo/x.py``), not shorthand, and keeps its existing verdict.
    """
    if not candidate.startswith(("/", "\\")):
        return None
    anchor = _backtick_run_anchor_dir(stripped, start)
    if anchor is None:
        return None
    return anchor.rstrip("/\\") + "/" + candidate.lstrip("/\\").replace("\\", "/")
