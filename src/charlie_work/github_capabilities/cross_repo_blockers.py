"""Cross-repo native issue blockers (issue #2005).

GitHub's native ``blocked_by`` relationship can point at an issue in a
*different* repository. The dependency surface used to reduce every blocker to
a bare issue number, so a foreign blocker ``owner/other#60`` was resolved as
``#60`` in the current repo -- typically a closed, unrelated issue -- and the
blocked issue looked unblocked.

``CrossRepoBlocker`` is an ``int`` subclass so every existing consumer that
sorts, dedupes, and passes blocker lists to ``are_issues_open`` keeps working
unchanged, while its equality/hash include the repo so a foreign ``#60`` never
collides with the local ``#60``. Same-repo blockers stay plain ``int``.
"""

from __future__ import annotations

from typing import Any


class CrossRepoBlocker(int):
    """A blocker issue number qualified by the ``owner/name`` repo it lives in."""

    repo: str

    def __new__(cls, number: int, repo: str) -> CrossRepoBlocker:
        obj = super().__new__(cls, number)
        obj.repo = repo.lower()
        return obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CrossRepoBlocker):
            return int(self) == int(other) and self.repo == other.repo
        return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.repo, int(self)))

    def __repr__(self) -> str:
        return f"CrossRepoBlocker({self.repo}#{int(self)})"


def repo_from_repository_url(url: Any) -> str | None:
    """``https://api.github.com/repos/owner/name`` -> ``owner/name`` (else None)."""
    if not isinstance(url, str) or "/repos/" not in url:
        return None
    slug = url.rsplit("/repos/", 1)[1].strip("/")
    return slug if slug.count("/") == 1 else None


def make_blocker(number: int, repo: str | None, current_repo: str | None) -> int:
    """Return ``number`` unchanged for a same-repo blocker, else a ``CrossRepoBlocker``.

    ``repo`` is ``None`` when the payload carried no repository (legacy shapes /
    test doubles): treated as same-repo. When ``repo`` is present but the
    current repo is unknown, the blocker is treated as foreign -- resolving it
    by its own repo is correct either way, whereas assuming same-repo is the
    original bug.
    """
    if not repo:
        return number
    if current_repo is not None and repo.lower() == current_repo.lower():
        return number
    return CrossRepoBlocker(number, repo)
