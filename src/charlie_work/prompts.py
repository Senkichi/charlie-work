from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from string import Template

TEMPLATE_DIR = Path(__file__).with_name("prompts")

# Markers that must appear in every rendered worker/rework prompt.  The
# no-merge contract (issue #502, #714) is a safety-critical prompt section:
# workers must never merge, close, or relabel their own PRs.  A repo-local
# flat whole-file override can silently drop the ``$section_no_merge_contract``
# reference, so these markers are checked against the *rendered output* — not
# the template source — at the dispatch boundary.
NO_MERGE_CONTRACT_MARKERS: tuple[str, ...] = (
    "## No-merge contract",
    "Your deliverable ENDS at pushing the branch and opening the PR",
)


class MissingNoMergeContractError(RuntimeError):
    """A rendered worker/rework prompt is missing the no-merge contract.

    Issue #714: a repo-local flat whole-file override of ``worker.md`` or
    ``rework.md`` can silently drop the ``$section_no_merge_contract``
    reference, dispatching workers with no instruction against merging,
    closing, or relabeling their own PRs.  This post-render guard catches
    that drift at the dispatch boundary — the single point of enforcement
    — rather than relying on every consumer repo's override to remember
    the section.
    """

    def __init__(self, context: str, missing: tuple[str, ...]) -> None:
        self.context = context
        self.missing = missing
        super().__init__(
            f"{context} is missing the no-merge contract (issue #714): "
            f"required marker(s) not found: {', '.join(missing)}. "
            f"A repo-local flat override may have dropped the "
            f"$section_no_merge_contract reference."
        )


def assert_no_merge_contract(prompt: str, *, context: str = "worker prompt") -> None:
    """Verify a rendered worker/rework prompt carries the no-merge contract.

    Checks the *rendered output* (not the template source) so that a
    repo-local flat override that drops the ``$section_no_merge_contract``
    reference is caught regardless of how the override was structured.
    """

    missing = tuple(m for m in NO_MERGE_CONTRACT_MARKERS if m not in prompt)
    if missing:
        raise MissingNoMergeContractError(context, missing)


# Markers that must appear in every rendered *worker* (not rework) prompt's
# PR-requirements section.  Issue #715: a repo-local flat whole-file override
# of ``worker.md`` can mandate a stale non-conventional-commit title format
# (e.g. ``Fix #$issue_number: <short title>``), which trips the janitor's
# ``_check_title_conventional`` warning on every PR the worker opens.  The
# package templates already use Conventional Commits; these markers are
# checked against the *rendered output* — not the template source — at the
# dispatch boundary so a stale override is caught before it ships.
CONVENTIONAL_TITLE_MARKERS: tuple[str, ...] = ("Conventional-Commits format",)


class MissingConventionalTitleError(RuntimeError):
    """A rendered worker prompt is missing the conventional-commit title instruction.

    Issue #715: a repo-local flat whole-file override of ``worker.md`` can
    mandate a stale non-conventional-commit PR title format (e.g.
    ``Fix #$issue_number: <short title>``), so every PR the worker opens
    trips the janitor's ``_check_title_conventional`` warning.  This
    post-render guard catches that drift at the dispatch boundary — the
    single point of enforcement — rather than relying on every consumer
    repo's override to carry the correct title instruction.
    """

    def __init__(self, context: str, missing: tuple[str, ...]) -> None:
        self.context = context
        self.missing = missing
        super().__init__(
            f"{context} is missing the conventional-commit title instruction "
            f"(issue #715): required marker(s) not found: {', '.join(missing)}. "
            f"A repo-local flat override may mandate a stale "
            f"non-conventional-commit title format (e.g. 'Fix #N: ...')."
        )


def assert_conventional_commit_title(prompt: str, *, context: str = "worker prompt") -> None:
    """Verify a rendered worker prompt instructs conventional-commit PR titles.

    Checks the *rendered output* (not the template source) so that a
    repo-local flat override that drops the conventional-commit title
    instruction is caught regardless of how the override was structured.
    """

    missing = tuple(m for m in CONVENTIONAL_TITLE_MARKERS if m not in prompt)
    if missing:
        raise MissingConventionalTitleError(context, missing)


# Markers that must appear in every rendered worker/rework prompt's execution
# contract.  Issue #717: a repo-local flat whole-file override of ``worker.md``
# or ``rework.md`` can silently drop the ``$section_execution_contract``
# reference, dispatching workers with a blanket "never run the full local suite"
# prohibition and no carve-out for contract-changing diffs (public function
# signature/return shape, exception type/message consumed elsewhere, DB schema,
# or module re-export).  Those are exactly the changes whose blast radius
# extends outside the files they touched, which is why the package's execution
# contract singles them out for a mandatory full-suite run before push rather
# than deferring entirely to CI.  These markers are checked against the
# *rendered output* — not the template source — at the dispatch boundary.
EXECUTION_CONTRACT_MARKERS: tuple[str, ...] = (
    "Execution contract (self-detect from your diff)",
    "run the **FULL suite** locally at the final head before pushing",
)


class MissingExecutionContractError(RuntimeError):
    """A rendered worker/rework prompt is missing the execution contract.

    Issue #717: a repo-local flat whole-file override of ``worker.md`` or
    ``rework.md`` can silently drop the ``$section_execution_contract``
    reference, dispatching workers with no escalation trigger for
    contract-changing diffs.  A worker changing a public function's signature
    or return shape, an exception type a caller depends on, a DB schema, or a
    module re-export can then ship with only the targeted/changed test files
    run locally — exactly the class of change whose blast radius extends
    outside the files it touched.  This post-render guard catches that drift
    at the dispatch boundary — the single point of enforcement — rather than
    relying on every consumer repo's override to remember the section.
    """

    def __init__(self, context: str, missing: tuple[str, ...]) -> None:
        self.context = context
        self.missing = missing
        super().__init__(
            f"{context} is missing the execution contract (issue #717): "
            f"required marker(s) not found: {', '.join(missing)}. "
            f"A repo-local flat override may have dropped the "
            f"$section_execution_contract reference, leaving a blanket "
            f"'never run the full local suite' prohibition with no "
            f"carve-out for contract-changing diffs."
        )


def assert_execution_contract(prompt: str, *, context: str = "worker prompt") -> None:
    """Verify a rendered worker/rework prompt carries the execution contract.

    Checks the *rendered output* (not the template source) so that a
    repo-local flat override that drops the ``$section_execution_contract``
    reference is caught regardless of how the override was structured.
    """

    missing = tuple(m for m in EXECUTION_CONTRACT_MARKERS if m not in prompt)
    if missing:
        raise MissingExecutionContractError(context, missing)


# Markers that must appear in every rendered worker/rework prompt's containment
# clause.  Issue #1010: the containment clause was scoped to "any other checkout
# of the repo" — which does not cover a different repo at all.  A dispatched
# worker edited a sibling repo's shared main checkout, contaminating another
# agent's PR.  The clause is widened to forbid any path outside the assigned
# worktree root, and these markers are checked against the *rendered output* —
# not the template source — at the dispatch boundary so a repo-local flat
# override that drops ``$section_scope_contract`` or reverts to the old
# repo-scoped wording is caught before it ships.
CONTAINMENT_MARKERS: tuple[str, ...] = (
    "**Containment:**",
    "any path outside the assigned worktree root",
)


class MissingContainmentError(RuntimeError):
    """A rendered worker/rework prompt is missing the widened containment clause.

    Issue #1010: the containment clause was scoped to "any other checkout of
    the repo", which does not cover a different repo.  A repo-local flat
    whole-file override of ``worker.md`` or ``rework.md`` can silently drop
    ``$section_scope_contract`` or revert to the old narrow wording,
    dispatching workers with no effective prohibition against editing a
    sibling repo's checkout.  This post-render guard catches that drift at
    the dispatch boundary — the single point of enforcement — rather than
    relying on every consumer repo's override to carry the widened clause.
    """

    def __init__(self, context: str, missing: tuple[str, ...]) -> None:
        self.context = context
        self.missing = missing
        super().__init__(
            f"{context} is missing the widened containment clause (issue #1010): "
            f"required marker(s) not found: {', '.join(missing)}. "
            f"A repo-local flat override may have dropped the "
            f"$section_scope_contract reference or reverted to the old "
            f"repo-scoped wording."
        )


def assert_containment(prompt: str, *, context: str = "worker prompt") -> None:
    """Verify a rendered worker/rework prompt carries the widened containment clause.

    Checks the *rendered output* (not the template source) so that a
    repo-local flat override that drops ``$section_scope_contract`` or
    reverts to the old repo-scoped wording is caught regardless of how the
    override was structured.
    """

    missing = tuple(m for m in CONTAINMENT_MARKERS if m not in prompt)
    if missing:
        raise MissingContainmentError(context, missing)


class PromptTemplateError(RuntimeError):
    """A prompt template references placeholders that nothing supplies.

    ``string.Template.safe_substitute`` leaves an unknown ``$placeholder`` in
    its output as literal text, so a template that drifts out of sync with the
    orchestrator renders a subtly broken prompt rather than failing.

    Issue #589: a repo-local ``review.md`` override kept referencing
    ``$decision_command`` and ``$checks_json_path`` after the orchestrator
    stopped supplying either. Every rendered packet shipped the literal strings
    to reviewers running under ``--permission-mode plan``, which cannot execute
    commands at all, and the prompt no longer carried the fenced-JSON block
    that is such a reviewer's only way to report a verdict. Twenty-one PRs ran
    full multi-turn reviews, reached real conclusions, and had every one of
    them discarded -- burning three paid sessions each before escalating.

    Refusing to render is strictly better than handing a worker or reviewer a
    prompt with literal ``$placeholder`` text in it.
    """

    def __init__(self, template_path: Path, missing: Iterable[str]) -> None:
        self.template_path = template_path
        self.missing = tuple(sorted(missing))
        super().__init__(
            f"prompt template {template_path} references placeholder(s) that "
            f"nothing supplies: {', '.join(self.missing)}. The template is out "
            f"of sync with the orchestrator; refusing to render a prompt "
            f"containing literal $placeholder text."
        )


def resolve_template(template_name: str, search_dirs: Sequence[Path] = ()) -> Path:
    """Repo-local template dirs win over the package defaults, per filename."""
    for directory in search_dirs:
        candidate = Path(directory) / template_name
        if candidate.is_file():
            return candidate
    return TEMPLATE_DIR / template_name


def _unresolved(template_text: str, available: set[str]) -> set[str]:
    """Placeholders ``template_text`` references that ``available`` does not cover.

    Always computed against *template source*, never against rendered output. A
    ``$word`` occurring inside an attacker-controlled value (an issue body, a PR
    title) is a leaf replacement that neither substitution pass re-scans, so it
    is not an unresolved placeholder and must never be reported as one --
    otherwise any issue could deny service to its own dispatch by putting a
    dollar sign in its title.
    """
    return set(Template(template_text).get_identifiers()) - available


def _missing_placeholders(
    template_text: str, sections: Mapping[str, str], available: set[str]
) -> set[str]:
    """Placeholders ``template_text`` (plus referenced section partials) needs
    that ``available`` does not cover.

    Shared by ``render_prompt``'s strict mode and the standalone
    :func:`unsupplied_placeholders` startup/CI check so the two cannot drift
    apart on what counts as an unresolved placeholder. Only section partials
    the template actually references can ship a broken placeholder -- unused
    partials are resolved at render time but never reach the output, so a
    stale one must not block an unrelated render.
    """
    template = Template(template_text)
    missing = _unresolved(template_text, available)
    for key in set(template.get_identifiers()) & set(sections):
        missing |= _unresolved(sections[key], available)
    return missing


def unsupplied_placeholders(
    template_name: str,
    supplied_keys: Iterable[str],
    *,
    search_dirs: Sequence[Path] = (),
) -> set[str]:
    """Placeholders the resolved template references that nothing supplies.

    Pure static check (issue #713): resolve ``template_name`` the way dispatch
    would -- a repo-local override in ``search_dirs`` wins over the package
    default, per filename -- expand every ``$section_*`` partial it references,
    and return the set of ``$placeholder``s the result needs that are neither in
    ``supplied_keys`` nor discovered as a section variable on disk. An empty
    return means the template is safe to render.

    This is the *subset* direction the issue specifies: an override legitimately
    uses fewer placeholders than the writer supplies (a sibling repo's ``worker.md``
    uses 6 of the writer's 8 keys), so the check fails only when the template
    reaches for a placeholder the writer never provides -- the exact shape of
    the #713 crash, where a flat ``rework.md`` override kept referencing
    ``$review_summary`` after the writer renamed its slot to ``$dispatch_note`` /
    ``$required_changes_section``. The reverse direction (every supplied key
    used) is not an error; at most a lint.

    No dispatch, no worker, no network: this reads template and section files
    off disk only, so it can run at supervisor startup and in CI to catch a
    drifting override *before* it crashes a live dispatch with an uncaught
    :class:`PromptTemplateError`.
    """
    from .prompt_sections import section_variables

    template_path = resolve_template(template_name, search_dirs)
    template_text = template_path.read_text(encoding="utf-8")
    sections = section_variables(search_dirs=tuple(search_dirs))
    available = set(supplied_keys) | set(sections)
    return _missing_placeholders(template_text, sections, available)


def render_prompt(
    template_name: str,
    values: Mapping[str, object],
    *,
    search_dirs: Sequence[Path] = (),
    strict: bool = True,
) -> str:
    """Render ``template_name`` against ``values``.

    With ``strict`` (the default), a template referencing a placeholder that
    nothing supplies raises :class:`PromptTemplateError` instead of emitting the
    literal ``$placeholder``. Callers that genuinely want partial rendering must
    opt out explicitly.
    """
    from .prompt_sections import section_variables

    template_path = resolve_template(template_name, search_dirs)
    template_text = template_path.read_text(encoding="utf-8")
    template = Template(template_text)
    sections = section_variables(search_dirs=tuple(search_dirs))
    # Explicit values win over section text on any future key collision.
    merged = {**sections, **values}
    safe_values = {key: str(value) for key, value in merged.items()}
    if strict:
        missing = _missing_placeholders(template_text, sections, set(merged))
        if missing:
            raise PromptTemplateError(template_path, missing)
    # Render section partials first: each partial's internal $placeholders are
    # resolved against safe_values, producing fully-resolved section strings.
    # This prevents attacker-controlled values (issue_body, etc.) from being
    # re-scanned in a second pass, which would allow prompt injection.
    resolved_sections = {
        key: Template(section_text).safe_substitute(safe_values)
        for key, section_text in sections.items()
    }
    # Merge resolved sections with explicit values (explicit values still win).
    final_values = {**resolved_sections, **values}
    # Convert all values to strings for the final substitution
    final_safe_values = {key: str(value) for key, value in final_values.items()}
    # Single substitution over the template: attacker-supplied values are leaf
    # replacements that are never re-scanned.
    return template.safe_substitute(final_safe_values)


def prompt_template_digest(template_name: str, search_dirs: Sequence[Path] = ()) -> str:
    """Return a stable SHA-256 digest of the template text plus every section
    partial it actually references.

    The digest is computed from *resolved source*, mirroring ``render_prompt``:
    the template file ``resolve_template`` picks (repo-local override or
    package default) and the section partials ``section_variables`` supplies
    that the template references via ``$section_<stem>``. Unused partials never
    reach the rendered output, so a stale unused partial must not invalidate
    packets -- only referenced ones are hashed.

    This lets a caller treat a template change as a packet-staleness trigger
    alongside a head-SHA change (issue #592): the packet's freshness becomes a
    function of all of its inputs, not just one, with no version constant to
    bump and no list of template names to maintain. The digest derives from
    the files on disk, so repo-local overrides, package templates, and section
    partials are all covered uniformly.
    """
    from .prompt_sections import section_variables

    template_path = resolve_template(template_name, search_dirs)
    template_text = template_path.read_text(encoding="utf-8")
    template = Template(template_text)
    sections = section_variables(search_dirs=tuple(search_dirs))
    referenced = set(template.get_identifiers()) & set(sections)

    hasher = hashlib.sha256()
    hasher.update(template_name.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(template_text.encode("utf-8"))
    # Deterministic order: section keys are sorted so the digest is stable
    # across processes and platforms regardless of dict iteration order.
    for key in sorted(referenced):
        hasher.update(b"\x00")
        hasher.update(key.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(sections[key].encode("utf-8"))
    return hasher.hexdigest()
