"""Print the base SHA the collect-only gate must collect at (issue #2123).

On a ``pull_request`` run, ``actions/checkout`` checks out ``refs/pull/N/merge``:
GitHub's merge of the PR head onto the *current* base tip. The head-side
collection therefore already contains every PR merged since this one branched.
``github.event.pull_request.base.sha`` is the base tip when the PR was last
synchronized and can be older, so collecting the base side there attributes
another PR's removals to this one. The merge commit's first parent is the exact
base the head was merged onto, so that is the only consistent base-side ref.

Fails loudly (exit 1, nothing on stdout) unless ``HEAD`` is a two-parent merge
commit: a non-merge checkout (e.g. the head SHA itself) would silently compare
the wrong snapshots.
"""

from __future__ import annotations

import subprocess
import sys


class NotAMergeCommitError(RuntimeError):
    """HEAD is not a two-parent merge commit."""


def merge_base_parent(repo: str = ".", rev: str = "HEAD") -> str:
    """Return the first-parent SHA of *rev*, which must be a two-parent merge."""
    proc = subprocess.run(
        ["git", "-C", repo, "rev-list", "--parents", "-n", "1", rev],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise NotAMergeCommitError(f"git rev-list failed for {rev}: {proc.stderr.strip()}")
    fields = proc.stdout.split()
    parents = fields[1:]
    if len(parents) != 2:
        raise NotAMergeCommitError(
            f"{rev} has {len(parents)} parent(s), expected exactly 2: it is not the "
            "pull_request merge ref (refs/pull/N/merge). Refusing to compare "
            "inconsistent snapshots. Is the checkout shallow (needs fetch-depth: 2)?"
        )
    return parents[0]


def main(argv: list[str]) -> int:
    try:
        print(merge_base_parent(argv[1] if len(argv) > 1 else "."))
    except NotAMergeCommitError as exc:
        print(f"collect_gate_base_sha: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
