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
        available = set(merged)
        missing = _unresolved(template_text, available)
        # Only section partials the template actually references can ship a
        # broken placeholder -- unused partials are resolved below but never
        # reach the output, so a stale one must not block an unrelated render.
        for key in set(template.get_identifiers()) & set(sections):
            missing |= _unresolved(sections[key], available)
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
