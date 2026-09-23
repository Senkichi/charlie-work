"""Outbound PR/issue body-write secret guard (issue #1505).

The #694 incident: an OAuth token captured in command output was briefly
written into a PR body. Deleting it afterwards did not help -- GitHub keeps
the prior revision in the PR's visible edit history and secret-scanning
surface. The only safe boundary is *before* the body leaves the process.

This module is the single enforcement point both GitHub backends share:

- ``scan_outbound_text`` matches text against a vendored, pinned subset of
  gitleaks' built-in rules (``_vendor/gitleaks/secrets.toml`` -- see
  ``UPSTREAM.md`` there for the pin, the selection rule, and the documented
  deviations from upstream ``gitleaks detect``).
- ``check_outbound_write`` is the chokepoint the write surfaces call with
  the parts about to be submitted (``("title", ...)``, ``("body", ...)``).
  On any match it emits an ``outbound_body_secret_refused`` event to
  ``events.db`` (warning level, surfaced to operators by
  ``OrchestratorApp._maybe_report_outbound_secret_refusals`` and by
  ``scripts/heartbeat_check.py``'s warning-events listing) and returns the
  matches so the caller can fail closed: never redact, never rewrite, refuse.

Fail-closed also covers the *mechanics*: a scanner that cannot load its
ruleset, or a body file that cannot be read, must not silently pass text
through. The loader therefore raises at module use (not import) if the
vendored file is absent or unparseable, and ``check_outbound_write``
propagates rule-loading failure instead of returning "clean".

Nothing secret-shaped is ever logged or persisted: match records carry only
the rule id, the part, the line number, and a SHA-256 of the matched text
(so a post-mortem can tell "the same credential refused twice" from "two
different credentials" without storing either).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import tomllib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import layout

logger = logging.getLogger(__name__)

# Registered in instrumentation._LEVEL_BY_KIND at "warning". Deliberately NOT
# in event_kinds.EXPECTED_OPERATIONAL_KINDS: that bucket exists for kinds that
# routinely dominate warning volume; a refusal is the opposite -- rare, and
# exactly the kind an operator must see in full (see event_kinds.py's own
# comments on the preflight tripwires for the same reasoning).
REFUSAL_EVENT_KIND = "outbound_body_secret_refused"

# Markdown info string whose fenced block contents are masked before
# scanning. Deliberately-labelled secret *examples* (e.g. this issue's own
# discussion of the gho_ incident) cannot otherwise be written to a PR body
# or comment -- the guard would refuse the very prose that documents it.
_EXAMPLE_FENCE_INFO = "example-secret"

_RULES_PATH = Path(__file__).parent / "_vendor" / "gitleaks" / "secrets.toml"

# CommonMark fenced-code-block opening: up to 3 spaces of indent, then at
# least three backticks or tildes, then an optional info string.
_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*(.*)$")


class OutboundBodyGuardError(RuntimeError):
    """The guard itself failed (ruleset unloadable, state path unresolvable).

    Distinct from a credential match: callers translate this into their own
    error vocabulary (``None`` for ``pr_create``, ``GitHubError`` for the
    comment surfaces) while staying fail-closed -- the write is refused either
    way.
    """


@dataclass(frozen=True)
class SecretMatch:
    """One credential-pattern hit in outbound text. Carries no secret material."""

    rule_id: str
    part: str  # the submission part that matched: "title" | "body"
    line: int  # 1-based line number within that part's text
    match_sha256: str  # sha256 of the matched text, for same-credential correlation


@dataclass(frozen=True)
class _Rule:
    """One vendored gitleaks rule, compiled."""

    rule_id: str
    pattern: re.Pattern[str]
    keywords: frozenset[str]  # lowercased
    entropy: float | None
    secret_group: int
    allowlist_regexes: tuple[re.Pattern[str], ...]


def _shannon_entropy(text: str) -> float:
    """Shannon entropy of ``text`` in bits/char, matching gitleaks' gate."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _mask_example_secret_fences(text: str) -> str:
    """Blank out every line of each ``example-secret`` fenced block.

    Returns a string with the same line count as ``text`` so reported line
    numbers still refer to the submitted document. Unclosed blocks mask to
    end-of-file (CommonMark semantics).

    ``in_fence`` tracks *any* open fence, not just exempt ones: inside an
    ordinary block an `````example-secret`` line is literal content (a closing
    fence cannot carry an info string), so it must not start masking -- that
    would let a real credential hide inside a nested-looking fence.
    """
    lines = text.split("\n")
    in_fence = False
    exempt = False
    close_re: re.Pattern[str] | None = None
    for i, line in enumerate(lines):
        stripped = line.rstrip("\r")
        if not in_fence:
            m = _FENCE_OPEN_RE.match(stripped)
            if m is None:
                continue
            fence, info = m.group(1), m.group(2).strip()
            # CommonMark: a backtick fence's info string may not contain `.
            if fence[0] == "`" and "`" in info:
                continue
            in_fence = True
            exempt = info == _EXAMPLE_FENCE_INFO
            close_re = re.compile(rf"^[ \t]{{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
            if exempt:
                lines[i] = ""
        else:
            if exempt:
                lines[i] = ""
            if close_re is not None and close_re.match(stripped):
                in_fence = False
                exempt = False
                close_re = None
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _rules() -> tuple[_Rule, ...]:
    """Load and compile the vendored ruleset. Cached for the process.

    Raises (rather than passing all text) if the vendored file is missing or
    malformed -- a scanner with no rules is a guard that silently admits
    everything, which is the failure mode this module exists to prevent.
    """
    with _RULES_PATH.open("rb") as handle:
        doc = tomllib.load(handle)
    compiled: list[_Rule] = []
    for raw in doc.get("rules", []):
        allowlist = tuple(
            re.compile(pattern)
            for entry in raw.get("allowlists", [])
            for pattern in entry.get("regexes", [])
        )
        compiled.append(
            _Rule(
                rule_id=raw["id"],
                pattern=re.compile(raw["regex"]),
                keywords=frozenset(k.lower() for k in raw.get("keywords", [])),
                entropy=raw.get("entropy"),
                secret_group=int(raw.get("secretGroup") or 0),
                allowlist_regexes=allowlist,
            )
        )
    return tuple(compiled)


def scan_outbound_text(text: str, *, part: str) -> tuple[SecretMatch, ...]:
    """Return every credential-pattern match in ``text`` (empty = clean)."""
    if not text:
        return ()
    masked = _mask_example_secret_fences(text)
    lowered = masked.lower()
    matches: list[SecretMatch] = []
    for rule in _rules():
        # Upstream's keyword prefilter: skip rules whose keywords are all
        # absent (case-insensitive on both sides).
        if rule.keywords and not any(kw in lowered for kw in rule.keywords):
            continue
        for m in rule.pattern.finditer(masked):
            secret = m.group(rule.secret_group) if rule.secret_group else m.group(0)
            if not secret:
                continue
            if rule.entropy is not None and _shannon_entropy(secret) < rule.entropy:
                continue
            if any(allow.search(m.group(0)) for allow in rule.allowlist_regexes):
                continue
            line = masked.count("\n", 0, m.start()) + 1
            matches.append(
                SecretMatch(
                    rule_id=rule.rule_id,
                    part=part,
                    line=line,
                    match_sha256=hashlib.sha256(secret.encode("utf-8")).hexdigest(),
                )
            )
    return tuple(matches)


def refusal_summary(surface: str, matches: tuple[SecretMatch, ...]) -> str:
    """One-line human-readable refusal reason. Carries no secret material."""
    rule_ids = ",".join(sorted({m.rule_id for m in matches}))
    parts = ",".join(sorted({m.part for m in matches}))
    return (
        f"outbound {surface} refused: {len(matches)} credential-pattern "
        f"match(es) [{rule_ids}] in {parts} -- see "
        f"{REFUSAL_EVENT_KIND} event in events.db"
    )


def _state_path_for(repo_root: Path | None, state_dir: str | None) -> Path | None:
    """Resolve ``state.json`` for the repo whose client is writing.

    Deferred import: ``paths`` pulls in ``config``, which imports
    ``charlie_work.github`` -- an import cycle at module load for the
    ``github_capabilities`` callers of this module (same shape as
    ``reconcile.py``'s deferred ``runtime_paths`` import).
    """
    if repo_root is None:
        return None
    from .paths import runtime_paths

    return runtime_paths(
        repo_root, state_dir or layout.DEFAULT_STATE_DIR, check_phantom=False
    ).state_file


def check_outbound_write(
    *,
    surface: str,
    parts: Iterable[tuple[str, str | None]],
    repo_root: Path | None,
    state_dir: str | None = None,
    issue_number: int | None = None,
    pr_number: int | None = None,
) -> tuple[SecretMatch, ...]:
    """Scan the parts of an outbound PR/issue write; emit the refusal event.

    The shared chokepoint for ``pr_create``, ``issue_comment``, and
    ``pr_comment`` on both the ``gh``-backed client and the local-file
    backend. Returns the matches (empty tuple = clean, proceed); callers are
    responsible for refusing the write when it is non-empty.

    The event payload carries only metadata -- rule ids, parts, line
    numbers, match hashes, and the issue/PR target -- never the matched text
    or the body itself.

    Raises ``OutboundBodyGuardError`` if the guard machinery itself fails
    (ruleset unloadable, state path unresolvable) -- callers translate it
    into their own error vocabulary and the write stays refused, so the
    guard fails closed in every direction.
    """
    try:
        matches: list[SecretMatch] = []
        for part, text in parts:
            if text:
                matches.extend(scan_outbound_text(text, part=part))
        if not matches:
            return ()

        logger.warning(
            "outbound body write refused (%s): %d credential-pattern match(es) "
            "across %s; refusing before any API call",
            surface,
            len(matches),
            sorted({m.part for m in matches}),
        )
        state_path = _state_path_for(repo_root, state_dir)
        if state_path is not None:
            from .instrumentation import log_event

            payload: dict[str, object] = {
                "surface": surface,
                "rule_ids": sorted({m.rule_id for m in matches}),
                "parts": sorted({m.part for m in matches}),
                "match_count": len(matches),
                "matches": [
                    {
                        "rule_id": m.rule_id,
                        "part": m.part,
                        "line": m.line,
                        "match_sha256": m.match_sha256,
                    }
                    for m in matches[:20]
                ],
            }
            if issue_number is not None:
                payload["issue_number"] = issue_number
            if pr_number is not None:
                payload["pr_number"] = pr_number
            # Best-effort by contract: log_event never raises, so a broken
            # events.db cannot weaken the refusal itself.
            # `repo_root` is non-None here -- state_path only resolves when
            # it is set.
            log_event(state_path, REFUSAL_EVENT_KIND, payload, repo=repo_root.name)
        return tuple(matches)
    except OutboundBodyGuardError:
        raise
    except Exception as exc:
        raise OutboundBodyGuardError(
            f"outbound body guard failed for {surface}: {type(exc).__name__}: {exc}"
        ) from exc
