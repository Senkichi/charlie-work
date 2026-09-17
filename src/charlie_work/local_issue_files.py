"""Markdown-with-frontmatter issue files: parse, scan, and rewrite.

The pure file layer under ``local_issues.LocalFileGitHub``. An issue is one
file, ``<issues_dir>/NNN_YYYY-MM-DD_slug.md``, whose YAML frontmatter carries
``title`` / ``state`` / ``labels`` and whose remainder is the issue body. The
issue *number* comes from the filename prefix, never from the frontmatter, so a
file cannot claim a number its name does not carry.

Two properties every function here preserves:

- **Mutations are confined to the frontmatter block.** ``rewrite_frontmatter_key``
  splits the file at the closing fence and only ever edits the first half. A
  body line that happens to read ``labels: foo`` or ``state: closed`` is prose,
  not metadata, and must survive a label transition byte-for-byte. (A
  whole-file ``re.sub`` on ``labels:.*`` is the obvious implementation and it
  corrupts exactly those bodies.)
- **Bytes outside the edited key survive.** Files are read and written as bytes
  and the file's own newline convention is reused, so a label flip on a CRLF
  file does not turn into a whole-file diff in the consumer repo's ``git status``.

Writes are atomic (temp file + ``replace``) for the same reason the JSON state
writes are: a worker, an editor, or the consumer repo's own issue CLI may read
the file mid-write.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# ``NNN_anything.md``. Deliberately the *only* selector for "is this an issue":
# issue directories also hold ``README.md`` / ``_template.md``, and a
# ``*.md.tmp`` staging file from an in-flight atomic write must not be read as
# an issue either.
ISSUE_FILENAME_RE = re.compile(r"^(\d+)_.+\.md$")

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*(?P<nl>\r?\n)(?P<fm>.*?)(?:\r?\n)---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)

# Orchestrator comments are appended below this marker so they never become
# part of the issue *body* that is rendered into a worker prompt.
COMMENTS_MARKER = "<!-- charlie-work:comments -->"
_COMMENT_SEPARATOR = "<!-- charlie-work:comment -->"

STATE_OPEN = "OPEN"
STATE_CLOSED = "CLOSED"


@dataclass(frozen=True)
class LocalIssue:
    """One parsed issue file, in orchestrator vocabulary."""

    number: int
    path: Path
    title: str
    state: str  # STATE_OPEN | STATE_CLOSED -- gh's casing, which consumers compare against
    labels: tuple[str, ...]
    body: str
    comments: tuple[str, ...]
    created_at: str
    updated_at: str
    author: str
    resolved: str  # local convenience; stamped on close, never read for workflow state

    @property
    def is_open(self) -> bool:
        return self.state == STATE_OPEN

    def to_github_dict(self, *, url: str) -> dict[str, Any]:
        """The ``gh issue list/view --json`` shape every consumer was written against."""
        return {
            "number": self.number,
            "title": self.title,
            "url": url,
            "body": self.body,
            "labels": [{"name": name} for name in self.labels],
            "assignees": [],
            "author": {"login": self.author},
            "comments": [{"body": text} for text in self.comments],
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "state": self.state,
        }


@dataclass(frozen=True)
class IssueFileProblem:
    """An issue-shaped file that could not be used, and why."""

    path: Path
    reason: str


@dataclass(frozen=True)
class IssueScan:
    """Result of reading an issues directory: what loaded, and what did not."""

    issues: tuple[LocalIssue, ...]
    problems: tuple[IssueFileProblem, ...]

    def by_number(self) -> dict[int, LocalIssue]:
        return {issue.number: issue for issue in self.issues}


class IssueFileError(ValueError):
    """An issue file's frontmatter is missing or unusable."""


def issue_number_from_name(name: str) -> int | None:
    match = ISSUE_FILENAME_RE.match(name)
    return int(match.group(1)) if match else None


def _iso_timestamp(value: Any) -> str:
    """Normalize a frontmatter date (``"2026-09-17"``, a YAML date, or a full
    timestamp) to the ``...T00:00:00Z`` form ``createdAt`` consumers parse."""
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = str(value)  # datetime.date -> "YYYY-MM-DD"
    return text if "T" in text else f"{text}T00:00:00Z"


def _normalize_labels(value: Any, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise IssueFileError(f"{path.name}: 'labels' must be a list of strings")
    # Order-preserving dedupe: a hand-edited duplicate must not make
    # remove-one-label leave a second copy behind.
    return tuple(dict.fromkeys(item.strip() for item in value if item.strip()))


def parse_issue_text(text: str, path: Path, *, updated_at: str = "") -> LocalIssue:
    """Parse one issue file's decoded text. Raises ``IssueFileError``."""
    number = issue_number_from_name(path.name)
    if number is None:
        raise IssueFileError(f"{path.name}: filename does not start with an issue number")
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise IssueFileError(f"{path.name}: no '---' fenced frontmatter block")
    try:
        meta = yaml.safe_load(match.group("fm")) or {}
    except yaml.YAMLError as exc:
        raise IssueFileError(f"{path.name}: frontmatter is not valid YAML ({exc})") from exc
    if not isinstance(meta, dict):
        raise IssueFileError(f"{path.name}: frontmatter must be a YAML mapping")

    raw_state = str(meta.get("state") or "open").strip().lower()
    if raw_state not in ("open", "closed"):
        raise IssueFileError(f"{path.name}: 'state' must be open or closed, got {raw_state!r}")

    remainder = text[match.end() :]
    body, _, comment_block = remainder.partition(COMMENTS_MARKER)
    comments = tuple(
        chunk.strip() for chunk in comment_block.split(_COMMENT_SEPARATOR) if chunk.strip()
    )
    return LocalIssue(
        number=number,
        path=path,
        title=str(meta.get("title") or path.stem),
        state=STATE_OPEN if raw_state == "open" else STATE_CLOSED,
        labels=_normalize_labels(meta.get("labels"), path),
        body=body.strip(),
        comments=comments,
        created_at=_iso_timestamp(meta.get("created")),
        updated_at=updated_at,
        author=str(meta.get("author") or ""),
        resolved=str(meta.get("resolved") or ""),
    )


def read_issue(path: Path) -> LocalIssue:
    """Read and parse one issue file. Raises ``IssueFileError`` / ``OSError``."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IssueFileError(f"{path.name}: not valid UTF-8 ({exc})") from exc
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    return parse_issue_text(text, path, updated_at=mtime.strftime("%Y-%m-%dT%H:%M:%SZ"))


def scan_issues(issues_dir: Path) -> IssueScan:
    """Load every issue file directly inside ``issues_dir`` (non-recursive).

    Never raises for a bad *file*: each unusable file becomes an
    ``IssueFileProblem`` so the caller decides the policy in one place. Two
    files claiming the same number are BOTH reported and BOTH excluded -- a
    label mutation addressed by number has no defensible way to pick one, and
    guessing would write workflow state onto the wrong issue.
    """
    if not issues_dir.is_dir():
        return IssueScan((), (IssueFileProblem(issues_dir, "issues directory does not exist"),))

    loaded: list[LocalIssue] = []
    problems: list[IssueFileProblem] = []
    for path in sorted(issues_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file() or issue_number_from_name(path.name) is None:
            continue
        try:
            loaded.append(read_issue(path))
        except (IssueFileError, OSError) as exc:
            problems.append(IssueFileProblem(path, str(exc)))

    paths_by_number: dict[int, list[Path]] = {}
    for issue in loaded:
        paths_by_number.setdefault(issue.number, []).append(issue.path)
    duplicated = {number for number, paths in paths_by_number.items() if len(paths) > 1}
    for number in sorted(duplicated):
        names = ", ".join(p.name for p in paths_by_number[number])
        problems.extend(
            IssueFileProblem(p, f"issue number {number} is claimed by multiple files: {names}")
            for p in paths_by_number[number]
        )
    issues = tuple(issue for issue in loaded if issue.number not in duplicated)
    return IssueScan(issues, tuple(problems))


def render_flow_list(values: tuple[str, ...]) -> str:
    """Render labels as a single-line YAML flow sequence.

    Emitted by PyYAML rather than ``", ".join`` so a label needing quotes
    (``agent:queued`` is fine bare; ``needs: triage`` is not) round-trips
    through ``parse_issue_text`` by construction. ``width`` is pinned because
    the default (80) wraps a long label list across lines, and everything
    downstream of this treats ``labels:`` as one line.
    """
    return yaml.safe_dump(list(values), default_flow_style=True, width=2**31).strip()


_KEY_LINE_RE_TEMPLATE = r"^{key}[ \t]*:"
_CONTINUATION_RE = re.compile(r"^(?:[ \t]+\S|-[ \t]|-$|[ \t]*$)")


def rewrite_frontmatter_key(text: str, key: str, rendered_value: str) -> str:
    """Return ``text`` with frontmatter ``key`` set to ``rendered_value``.

    Pure. Only the frontmatter block is touched. A block-style value
    (``labels:`` followed by ``- item`` lines) is replaced whole, so a
    hand-written block list collapses to the single-line form instead of
    leaving orphaned ``- item`` lines behind. A missing key is appended to the
    end of the block. Raises ``IssueFileError`` when there is no frontmatter.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise IssueFileError("no '---' fenced frontmatter block")
    newline = match.group("nl")
    lines = match.group("fm").split(newline)
    key_re = re.compile(_KEY_LINE_RE_TEMPLATE.format(key=re.escape(key)))
    new_line = f"{key}: {rendered_value}"

    start = next((i for i, line in enumerate(lines) if key_re.match(line)), None)
    if start is None:
        rewritten = [*lines, new_line]
    else:
        end = start + 1
        while end < len(lines) and _CONTINUATION_RE.match(lines[end]):
            end += 1
        # Trailing blank lines belong to the block's layout, not to the value.
        while end > start + 1 and lines[end - 1].strip() == "":
            end -= 1
        rewritten = [*lines[:start], new_line, *lines[end:]]

    fm_start, fm_end = match.span("fm")
    return text[:fm_start] + newline.join(rewritten) + text[fm_end:]


def append_comment(text: str, comment: str, *, timestamp: str) -> str:
    """Return ``text`` with ``comment`` appended under the comments marker."""
    match = _FRONTMATTER_RE.match(text)
    newline = match.group("nl") if match else "\n"
    entry = newline.join((f"**charlie-work** ({timestamp})", "", *comment.strip().splitlines()))
    base = text.rstrip("\r\n")
    if COMMENTS_MARKER not in text:
        base = newline.join((base, "", COMMENTS_MARKER))
    return newline.join((base, "", _COMMENT_SEPARATOR, entry, ""))


def write_text_atomic(path: Path, text: str) -> None:
    """Temp file + ``replace``; bytes, so no newline translation on Windows."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(text.encode("utf-8"))
    tmp.replace(path)
