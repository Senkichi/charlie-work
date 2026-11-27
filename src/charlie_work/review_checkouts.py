"""``pr-<N>`` review checkouts: create, tear down, and forward their roots.

A reviewer session checks out the PR's head SHA in an isolated, detached-HEAD
git worktree keyed ``pr-<N>`` under the review-checkouts root -- never the
worker's branch-slug worktree, never a real branch. Issue #3224 split that
root in two: the checkout half of ``reviews_dir`` (the ``pr-<N>`` dirs, the
only part that churns a full file tree per dispatch) follows the host I/O
volume (``<worker_root>/<repo name>/reviews``) while the bookkeeping half
(sidecars, logs, session records) stays in ``reviews_dir``.
``create_review_checkout`` and ``remove_review_checkout`` therefore take the
resolved pair -- ``review_checkouts`` (the active root; ``reviews_dir``
unchanged when the volume is not relocating) and ``legacy_review_checkouts``
(the pre-switch root, swept while it still has ``pr-*`` dirs) -- and
``review_checkout_kwargs`` is the single place callers fold the pair into
keyword arguments.

Extracted from ``worktree.py`` under the file-size ratchet (issue #1442): new
code must not land in the over-cap monolith, so the family lives here and
``worktree.py`` re-exports it through a facade import block. The worktree
internals the family runs on (``remove_worktree``, ``list_worktrees``, the
fetch helpers, ``WorktreeInfo``) stay in ``worktree.py`` and are imported
lazily inside each function -- a module-level ``from .worktree import`` here
would close the import cycle, the same reason ``worktree_stranded_tip``
injects its runner. The call-time lookup also keeps the historical
``charlie_work.worktree.<helper>`` monkeypatch seam resolving exactly as it
did when the functions lived there.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .safe_ref import require_valid_sha
from .subprocess_runner import run_captured

if TYPE_CHECKING:
    from .worktree import WorktreeInfo


def _review_checkout_path(checkouts_root: Path, pr_number: int) -> Path:
    """The ``pr-<N>`` checkout path under a review-checkouts root.

    The single place the keyed path is built -- every caller (create, remove,
    the stalled-review orphan scan) resolves a root and lets this name the
    checkout inside it.
    """
    return checkouts_root / f"pr-{pr_number}"


def review_checkout_kwargs(
    reviews_dir: Path,
    review_checkouts: Path | None = None,
    legacy_review_checkouts: Path | None = None,
) -> dict[str, Path]:
    """Kwargs forwarding the resolved checkout roots to the checkout helpers.

    Includes a root only when it changes the outcome: ``review_checkouts``
    only when it differs from ``reviews_dir`` (the value the parameter falls
    back to), ``legacy_review_checkouts`` only when set and not the same root
    as the active one. A caller holding :class:`ResolvedLayout` roots can pass
    them unconditionally through this helper -- the call keeps its historical
    ``reviews_dir=``-only shape whenever the roots coincide, which is also
    what test doubles patched on the checkout-helper seams accept.
    """
    kwargs: dict[str, Path] = {}
    if review_checkouts is not None and review_checkouts != reviews_dir:
        kwargs["review_checkouts"] = review_checkouts
    if legacy_review_checkouts is not None and legacy_review_checkouts != (
        review_checkouts or reviews_dir
    ):
        kwargs["legacy_review_checkouts"] = legacy_review_checkouts
    return kwargs


def create_review_checkout(
    repo_root: Path,
    pr_number: int,
    head_sha: str,
    *,
    reviews_dir: Path,
    review_checkouts: Path | None = None,
    legacy_review_checkouts: Path | None = None,
) -> WorktreeInfo:
    """Create an isolated, detached-HEAD checkout of a PR's head SHA for a
    reviewer session.

    This is the reviewer-side sibling of ``create_worktree``, but deliberately
    NOT built on top of it: a reviewer must never share a worker's branch or
    worktree path. Where ``create_worktree`` keys the checkout path by
    branch-slug under ``worktrees_dir`` and (in ``rework``/``review`` mode)
    attaches to the branch's live tip, this function keys the path by PR
    number under the review-checkouts root and always checks out in detached
    HEAD state at the exact ``head_sha``. That guarantees:

    - The reviewer's checkout can never alias a worker's worktree directory,
      even for the same branch (different parent dir, different key scheme).
    - The reviewer never has a branch checked out, so nothing it does (or a
      bug in a permissive command template) can land a commit on the PR's
      real branch — the checkout is read-only in spirit even before the
      caller's ``--permission-mode`` is applied.

    The checkout goes under ``review_checkouts`` when given (the resolved
    ``ResolvedLayout.review_checkouts`` — the host I/O volume's
    ``<worker_root>/<repo>/reviews`` when the manifest relocates checkouts),
    else ``reviews_dir``. Only the ``pr-<N>`` checkout directory moves with
    that root; sidecars, logs and every other session file stay in
    ``reviews_dir``.

    Idempotent: if a checkout already exists at this PR's keyed path (e.g.
    from a previous review round at a different head_sha), it is torn down
    and recreated fresh rather than reused or fast-forwarded in place. The
    teardown covers the legacy root (``legacy_review_checkouts``) too, so a
    pre-switch checkout left under ``reviews_dir`` cannot linger once the
    checkout root moved onto the volume.

    Raises ``ValueError`` if ``head_sha`` is empty (caller error — never a
    transient condition). Raises ``RuntimeError`` if the fetch or worktree-add
    fails.
    """
    from .worktree import (
        WorktreeInfo,
        _commit_exists_locally,
        _DEFAULT_TIMEOUT_SECONDS,
        _has_origin_remote,
        _run_remote_captured,
    )

    if not head_sha:
        raise ValueError(
            f"create_review_checkout requires a non-empty head_sha for PR #{pr_number}"
        )
    # Validate head_sha format before it reaches git argv (issue #659).
    # head_sha is passed as a plain positional to ``git fetch origin`` and
    # ``git worktree add --detach``, so a flag-like value would be parsed as
    # an option without this guard.
    head_sha = require_valid_sha(head_sha, context="create_review_checkout head_sha")

    target_dir = review_checkouts if review_checkouts is not None else reviews_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    checkout_path = _review_checkout_path(target_dir, pr_number)

    # Tear down any stale checkout at this path first (idempotent replace) —
    # never reuse/fast-forward a review checkout in place. The legacy root is
    # swept too so a pre-switch checkout cannot outlive the move.
    remove_review_checkout(
        repo_root,
        pr_number,
        reviews_dir=target_dir,
        **review_checkout_kwargs(target_dir, legacy_review_checkouts=legacy_review_checkouts),
    )

    # Only fetch if an origin remote exists (mirrors create_worktree's own
    # guard). Pure-local repos (test fixtures) have no origin to fetch from —
    # the caller-supplied head_sha must already be reachable locally there.
    # Skip the network round-trip entirely when the commit is already local.
    if _has_origin_remote(repo_root) and not _commit_exists_locally(repo_root, head_sha):
        fetch_result = _run_remote_captured(
            ["git", "fetch", "origin", head_sha],
            cwd=repo_root,
        )
        if not fetch_result.ok:
            # Raw-SHA fetches are refused by the server when the SHA is not
            # currently advertised (e.g. read just before a push updated the
            # branch). The PR head ref is always advertised — fall back to it
            # so a transient race doesn't block the review checkout; the
            # worktree add below still pins the exact caller-supplied head_sha.
            _run_remote_captured(
                ["git", "fetch", "origin", f"refs/pull/{pr_number}/head"],
                cwd=repo_root,
            )
            if not _commit_exists_locally(repo_root, head_sha):
                raise RuntimeError(
                    f"[git fetch origin {head_sha}] Failed to fetch head_sha {head_sha!r} "
                    f"for PR #{pr_number} review checkout: "
                    f"{fetch_result.error or fetch_result.stderr}"
                )

    result = run_captured(
        ["git", "worktree", "add", "--detach", str(checkout_path), head_sha],
        cwd=repo_root,
        timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
    )
    if not result.ok:
        raise RuntimeError(
            f"git worktree add --detach failed for PR #{pr_number} review checkout at "
            f"{head_sha!r}: {result.error or result.stderr}"
        )

    return WorktreeInfo(path=checkout_path, branch=head_sha, venv_junction=None)


def remove_review_checkout(
    repo_root: Path,
    pr_number: int,
    *,
    reviews_dir: Path,
    review_checkouts: Path | None = None,
    legacy_review_checkouts: Path | None = None,
) -> bool:
    """Idempotent teardown of a PR's isolated review checkout.

    Tears the ``pr-<N>`` checkout down under the active checkouts root
    (``review_checkouts`` when given, else ``reviews_dir``) AND under
    ``legacy_review_checkouts`` when that root is also set -- the second half
    is the switchover guarantee: a checkout created before the root moved to
    the host I/O volume is still found and removed.

    Returns True if the checkout was removed or was already absent; never
    raises. Safe to call speculatively (e.g. before creating a fresh checkout,
    or during a stale-claim/completed-verdict sweep that isn't sure whether a
    checkout exists for a given PR).
    """
    from .worktree import list_worktrees, remove_worktree

    active_root = review_checkouts if review_checkouts is not None else reviews_dir
    candidates = [_review_checkout_path(active_root, pr_number)]
    if legacy_review_checkouts is not None:
        legacy_path = _review_checkout_path(legacy_review_checkouts, pr_number)
        if legacy_path != candidates[0]:
            candidates.append(legacy_path)

    existing_worktrees = list_worktrees(repo_root)
    registered_paths = {Path(wt["worktree"]) for wt in existing_worktrees}
    ok = True
    for checkout_path in candidates:
        if checkout_path not in registered_paths and not checkout_path.exists():
            continue
        ok = remove_worktree(repo_root, checkout_path, force=True, branch=None) and ok
    return ok
