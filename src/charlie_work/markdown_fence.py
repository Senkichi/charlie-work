"""CommonMark-safe fencing for text embedded in a rendered prompt (issue #883).

A prompt template that writes its fence literally --

    ```md
    $issue_body
    ```

-- is only correct while the substituted value contains no fence of its own. On
a developer issue tracker that assumption fails most of the time: measured
against this repo, **61 of 100 open issue bodies contain a code fence**. When
one does, the body's own ``` closes the block early and everything after it
stops being quoted material. A heading in the issue body then becomes
indistinguishable from a heading the orchestrator wrote, and in the worker
templates ``$section_scope_contract`` follows immediately after the block, so
body text can merge visually with the scope contract.

That is a correctness bug first: the framing is lost for the *normal* case,
with no adversary involved. It is a prompt-injection vector second, and only
latently -- this repo is private, so issue authors are already trusted
collaborators. Worth noting that ``prompts.render_prompt`` deliberately
substitutes exactly once so that a supplied value is never re-scanned as a
template; a fence that the content can close reintroduces the same hazard one
layer down, at the formatting layer rather than the templating layer.

The fix is the CommonMark rule: a fenced block closes on the first line whose
fence is *at least as long* as the opener, so an opener longer than any
backtick run in the content cannot be terminated from inside it. Computing the
width means the fence has to move out of the template and into the substituted
value, which is why callers supply a pre-fenced ``*_block`` rather than a bare
value.

This lives in its own module rather than in ``issue_comments`` (where the rule
was first implemented, for #872) because it is a CommonMark concern with
several unrelated consumers, and each one would otherwise be tempted to
re-derive it. There are three already:

* ``$issue_body_block`` in the worker templates (this issue);
* the per-comment block in ``issue_comments`` (#872);
* ``$dispatch_note_block`` in ``rework.md`` -- reviewer prose quoting pytest
  output and shell commands, found by sweeping for the same defect class
  rather than assumed absent. 16 of 289 review summaries on disk carry a
  fence, and ``prs/pr-182/rework-prompt.md`` is a rendered instance of the
  break: the reviewer's own fence closed the wrapper early, so the template's
  intended *closing* fence opened a block that swallowed the brief's
  "Required behavior" and push-verification sections.
"""

from __future__ import annotations

import re

__all__ = ["MIN_FENCE_LENGTH", "fence_for", "fenced_block"]

# CommonMark's minimum fence. A shorter run is inline code, not a block.
MIN_FENCE_LENGTH = 3

_BACKTICK_RUN_RE = re.compile(r"`+")


def fence_for(text: str) -> str:
    """Return a backtick fence that ``text`` cannot terminate from inside.

    One longer than the longest backtick run present, never shorter than
    ``MIN_FENCE_LENGTH``.
    """
    longest = max((len(run) for run in _BACKTICK_RUN_RE.findall(text)), default=0)
    return "`" * max(MIN_FENCE_LENGTH, longest + 1)


def fenced_block(text: str, info: str = "") -> str:
    """Wrap ``text`` in a fence it cannot escape, tagged with ``info``.

    ``text`` is embedded verbatim -- no stripping, no normalisation. That is
    deliberate: it keeps the output byte-identical to a literal three-backtick
    fence for every value that contains no fence of its own, which is what makes
    this change auditable. Any prompt diff is then attributable to a body that
    genuinely needed a wider fence.
    """
    fence = fence_for(text)
    return f"{fence}{info}\n{text}\n{fence}"
