"""Local-file issue backend: a ``GitHubLike`` for repos with no GitHub remote.

``OrchestratorApp`` never constructs its GitHub client -- it is handed one
(``gh: GitHubLike``). That injection point is the whole integration: the
dispatch path reaches ~10 distinct ``gh`` members (``issue_list``, ``pr_list``,
``merged_pr_list``, ``are_issues_open``, ``labels.transition(gh, ...)``, ...),
so a ``if config.local_issues.enabled`` branch at the ``issue_list`` call site
would need a sibling branch at every one of the others. Substituting the
injected object needs none, and ``labels.transition`` -- the single owner of
every label edge -- works unmodified because it only ever calls
``add_issue_label`` / ``remove_issue_label``.

The surface splits in two:

- **Issues and labels are real**, backed by ``local_issue_files``.
- **Everything PR/CI/merge-shaped is a null object.** A repo with no remote has
  no pull requests, so *reads* answer in each member's existing "nothing
  there" vocabulary (``[]`` / ``{}`` / ``None`` -- the same values the real
  client returns for an empty repo) and the PR-driven lanes (review dispatch,
  merge train, finalization) find nothing to do. *Writes* to that surface fail
  as values or raise ``GitHubError``, exactly as the real client does when
  ``gh`` fails: nothing should reach them when ``pr_list()`` is empty, so
  reaching one is a bug worth hearing about, not a no-op worth hiding.

``github_client_for`` is the only place that chooses between the two backends.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from .config import OrchestratorConfig
from .github import GitHubError, GitHubLike, GitHubRunResult
from .github_capabilities.pull_requests import MergedPRSearchResult
from .outbound_body_guard import (
    OutboundBodyGuardError,
    check_outbound_write,
    refusal_summary,
)
from .local_issue_files import (
    IssueFileError,
    IssueFileProblem,
    IssueScan,
    LocalIssue,
    append_comment,
    render_flow_list,
    rewrite_frontmatter_key,
    scan_issues,
    write_text_atomic,
)
from .safe_path import require_contained

logger = logging.getLogger(__name__)

_NO_REMOTE = "local-file issue source: this repo has no GitHub remote"

# ``check_graphql_rate_limit`` reports (sufficient, remaining, reset_at). There
# is no quota to exhaust; any positive ``remaining`` reads as "plenty".
_UNLIMITED_REMAINING = 5000


@dataclass(frozen=True)
class LocalFileGitHub:
    """``GitHubLike`` over a directory of markdown issue files."""

    # Capability flag, probed with ``getattr(gh, "publishes_pull_requests",
    # True)`` so the real client and every existing test double keep the
    # default. ``dead_worker_reap._attempt_salvage`` reads it to decide what
    # "publish the worker's commits" means: push + PR, or -- here -- leave the
    # branch where it is and hand the issue to a human as review-ready.
    publishes_pull_requests: ClassVar[bool] = False

    repo_root: Path
    issues_dir: Path
    dry_run: bool = False
    # ``runtime.state_dir`` carried through ``github_client_for`` so the
    # outbound-body secret guard (issue #1505) can locate ``events.db`` for
    # the refusal event -- the same resolution the real client's
    # ``RuntimeConfig`` provides. ``None`` falls back to the default state
    # dir; only test/direct constructions omit it.
    state_dir: str | None = None
    # Holds exactly one kind of entry: ``("issue_dependencies", n) -> []``,
    # the warm-cache contract ``Issues.issue_dependencies`` documents. The
    # per-issue ``get_github_issue_dependencies`` reads this key before it
    # falls back to ``gh api``; without the entry it would call ``run()``,
    # fail, and log a "dependencies API failed" warning per issue per pass
    # for a condition that is not a failure. The entry can never go stale --
    # a file-backed issue has no GitHub-native dependency edges, ever.
    #
    # Issue *content* is deliberately never cached: re-reading a few dozen
    # small files is cheaper than being wrong about a label this same process
    # changed a moment ago.
    _list_cache: dict[Any, Any] = field(default_factory=dict, compare=False, repr=False)
    # Problems already warned about by THIS instance. ``_scan`` runs many
    # times per pass (every ``issue_list`` / ``issue_view`` / ``_mutate``), and
    # the fleet loop rebuilds the app -- and so this client -- every pass, so
    # the set's lifetime is "one pass": one warning per broken file per pass,
    # not one per read.
    _reported: set[tuple[Path, str]] = field(default_factory=set, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Boundary check, once: ``issues_dir`` is config-derived, and every
        # write below targets a path beneath it.
        require_contained(self.repo_root, self.issues_dir, context="local_issues.issues_dir")

    # -- file access -------------------------------------------------------

    def _scan(self) -> IssueScan:
        scan = scan_issues(self.issues_dir)
        self._handle_scan_problems(scan.problems)
        for issue in scan.issues:
            self._list_cache.setdefault(("issue_dependencies", issue.number), [])
        return scan

    def _handle_scan_problems(self, problems: tuple[IssueFileProblem, ...]) -> None:
        """Decide what an unusable issue file costs the pass.

        Called on every scan with the files that were issue-shaped by name but
        could not be loaded (bad YAML, bad ``state``, duplicate number, missing
        directory). Those files are already excluded from the returned issues.

        Policy: a **missing directory** raises ``GitHubError`` -- there is no
        issue set to act on, the config points at nothing, and a pass that
        proceeds would conclude "zero issues" with a clean exit; deferring the
        pass is the same loud outcome a ``gh`` outage gets. Every **per-file**
        problem is a warning and the pass goes on: the broken file is already
        excluded, so one typo must not hold the other issues hostage, and a
        duplicate number is safe to skip precisely because the scan drops
        *both* claimants rather than guessing. Warnings are deduplicated per
        instance (see ``_reported``): the operator reads the log, and the same
        line forty times a pass is how a log stops being read. No event is
        emitted -- nothing would consume it, and the label state is untouched.
        """
        for problem in problems:
            if problem.path == self.issues_dir:
                raise GitHubError(f"local_issues.issues_dir: {problem.path}: {problem.reason}")
            key = (problem.path, problem.reason)
            if key in self._reported:
                continue
            self._reported.add(key)
            logger.warning("local issue file skipped: %s: %s", problem.path.name, problem.reason)

    def _issue_url(self, issue: LocalIssue) -> str:
        return f"local://{issue.path.relative_to(self.repo_root).as_posix()}"

    def _to_dict(self, issue: LocalIssue) -> dict[str, Any]:
        return issue.to_github_dict(url=self._issue_url(issue))

    def _mutate(self, number: int, key: str, update: Any) -> bool:
        """Rewrite one frontmatter key of issue ``number``. Never raises.

        ``update`` maps the freshly-read ``LocalIssue`` to the rendered value,
        or ``None`` for "already as desired" (idempotent success, no write --
        so an unchanged label set does not touch the file's mtime or dirty the
        consumer repo's working tree).
        """
        issue = self._scan().by_number().get(number)
        if issue is None:
            logger.warning("local issue #%d not found under %s", number, self.issues_dir)
            return False
        rendered = update(issue)
        if rendered is None or self.dry_run:
            return True
        try:
            text = issue.path.read_bytes().decode("utf-8")
            write_text_atomic(issue.path, rewrite_frontmatter_key(text, key, rendered))
        except (OSError, UnicodeDecodeError, IssueFileError) as exc:
            logger.warning("local issue #%d: could not update '%s': %s", number, key, exc)
            return False
        return True

    # -- IssuesLike --------------------------------------------------------

    def issue_list(self, labels: Any = None, state: Any = None) -> list[dict[str, Any]]:
        wanted = {labels} if isinstance(labels, str) else set(labels or ())
        effective_state = str(state or "open").lower()
        selected = [
            issue
            for issue in self._scan().issues
            if wanted <= set(issue.labels)  # gh's repeated --label is AND
            and (effective_state == "all" or issue.state.lower() == effective_state)
        ]
        # gh lists newest first.
        return [self._to_dict(i) for i in sorted(selected, key=lambda i: i.number, reverse=True)]

    def issue_view(self, number: int) -> dict[str, Any]:
        issue = self._scan().by_number().get(number)
        return self._to_dict(issue) if issue is not None else {}

    def close_issue(self, number: int) -> bool:
        """Set ``state: closed`` (and stamp ``resolved`` if unset). Idempotent."""
        closed = self._mutate(number, "state", lambda i: "closed" if i.is_open else None)
        if closed and not self.dry_run:
            today = datetime.now(UTC).date().isoformat()
            self._mutate(number, "resolved", lambda i: None if i.resolved else f'"{today}"')
        return closed

    def are_issues_open(self, issue_numbers: list[int]) -> set[int]:
        by_number = self._scan().by_number()
        return {n for n in issue_numbers if n in by_number and by_number[n].is_open}

    def issue_dependencies(self, issue_numbers: list[int]) -> dict[int, list[int]]:
        # GitHub-native "blocked by" edges have no file equivalent. Blockers
        # declared in the issue *body* are still honoured: ``parse_blockers``
        # reads them and resolves state through ``are_issues_open`` above.
        for number in issue_numbers:
            self._list_cache.setdefault(("issue_dependencies", number), [])
        return {number: [] for number in issue_numbers}

    # -- LabelsLike --------------------------------------------------------

    def add_issue_label(self, number: int, label: str) -> bool:
        return self._mutate(
            number,
            "labels",
            lambda i: None if label in i.labels else render_flow_list((*i.labels, label)),
        )

    def remove_issue_label(self, number: int, label: str) -> bool:
        return self._mutate(
            number,
            "labels",
            lambda i: (
                render_flow_list(tuple(name for name in i.labels if name != label))
                if label in i.labels
                else None
            ),
        )

    def label_list(self) -> list[dict[str, Any]]:
        names = sorted({name for issue in self._scan().issues for name in issue.labels})
        return [{"name": name} for name in names]

    def label_create(self, label: str, color: str, description: str) -> None:
        # Labels are free-form strings in frontmatter; there is no registry to
        # create one in. Success by definition.
        return None

    def add_pr_label(self, number: int, label: str) -> bool:
        return False

    def remove_pr_label(self, number: int, label: str) -> bool:
        return False

    # -- CommentsLike ------------------------------------------------------

    def issue_comment(self, number: int, body_file: Path) -> None:
        """Append the comment to the issue file, below the comments marker.

        With no remote there is no other operator-facing channel attached to
        the issue itself: the review-ready hand-off (which names the branch to
        review) and the citation-check notice both arrive this way.
        """
        issue = self._scan().by_number().get(number)
        if issue is None:
            raise GitHubError(f"local issue #{number} not found under {self.issues_dir}")
        if self.dry_run:
            return
        stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            comment = body_file.read_text(encoding="utf-8")
            # Issue #1505: the appended comment persists inside a git-visible
            # issue file in the consumer repo, so the same credential guard
            # that fronts the gh API boundary fronts this write too.
            matches = check_outbound_write(
                surface="issue_comment",
                parts=(("body", comment),),
                repo_root=self.repo_root,
                state_dir=self.state_dir,
                issue_number=number,
            )
            if matches:
                raise GitHubError(refusal_summary("issue_comment", matches))
            text = issue.path.read_bytes().decode("utf-8")
            write_text_atomic(issue.path, append_comment(text, comment, timestamp=stamp))
        except OutboundBodyGuardError as exc:
            raise GitHubError(f"issue_comment #{number}: {exc}") from exc
        except (OSError, UnicodeDecodeError) as exc:
            raise GitHubError(f"local issue #{number}: could not append comment: {exc}") from exc

    def pr_comment(self, number: int, body_file: Path) -> None:
        raise GitHubError(_NO_REMOTE)

    # -- owner members -----------------------------------------------------

    def run(
        self,
        args: list[str],
        *,
        json_output: bool = False,
        allow_failure: bool = False,
        long_call: bool = False,
    ) -> Any:
        """There is no ``gh`` to run. Same failure contract as the real client."""
        if allow_failure:
            return GitHubRunResult(ok=False, returncode=1, stdout="", stderr="", error=_NO_REMOTE)
        raise GitHubError(f"{_NO_REMOTE} (gh {' '.join(args[:2])})")

    def _fail(self) -> GitHubRunResult:
        return self.run([], allow_failure=True)

    # -- RepoMetaLike ------------------------------------------------------

    def name_with_owner(self) -> str:
        return f"local/{self.repo_root.name}"

    def _repo_owner_name(self) -> tuple[str, str]:
        # Not a ``GitHubLike`` member, but ``cli._assert_not_sibling_clone``
        # calls it on whatever client bootstrap produced and reads
        # ``GitHubError`` as "not a GitHub-fleet repo, guard does not apply" --
        # which is the literal truth here.
        raise GitHubError(_NO_REMOTE)

    def compare(self, base: str, head: str) -> dict[str, Any] | None:
        return None

    def compare_diff(self, base: str, head: str) -> str | None:
        return None

    def commit(self, sha: str) -> GitHubRunResult:
        return self._fail()

    def invalidate_list_cache(self) -> None:
        return None

    # -- PullRequestsLike / MergeBranchLike / ChecksLike: null object ------

    def pr_create(self, head: str, base: str, title: str, body: str) -> int | None:
        return None

    def pr_view(self, number: int, *, fields: str = "") -> dict[str, Any]:
        return {}

    def pr_list(self) -> list[dict[str, Any]]:
        return []

    def pr_diff(self, number: int) -> str:
        return ""

    def pr_commits(self, number: int) -> list[dict[str, Any]] | None:
        return None

    def merged_pr_list(self) -> list[dict[str, Any]]:
        return []

    def merged_prs_for_issue(self, issue_number: int, branch_prefix: str) -> MergedPRSearchResult:
        return MergedPRSearchResult([], ok=True)

    def pr_ready(self, number: int) -> GitHubRunResult:
        return self._fail()

    def pr_close(self, number: int) -> GitHubRunResult:
        return self._fail()

    def pr_reopen(self, number: int) -> GitHubRunResult:
        return self._fail()

    def push_empty_commit(self, branch: str) -> GitHubRunResult:
        return self._fail()

    def merge_pr(
        self,
        number: int,
        strategy: str,
        admin: bool = False,
        merge_flags: tuple[str, ...] = (),
    ) -> str:
        raise GitHubError(_NO_REMOTE)

    def delete_branch(self, branch: str) -> bool:
        return False

    def pr_update_branch(self, pr_number: int) -> bool:
        return False

    def branch_protection(self, base: str) -> dict[str, Any] | None:
        return None

    def pr_checks(self, number: int) -> list[dict[str, Any]] | None:
        return None

    def check_run_annotations(self, check_run_id: int) -> list[dict[str, Any]]:
        return []

    def commit_check_runs(self, sha: str) -> list[dict[str, Any]] | None:
        return None

    def actions_job(self, job_id: int) -> dict[str, Any] | None:
        return None

    def workflow_runs_for_head(self, head_sha: str) -> list[dict[str, Any]] | None:
        return None

    def check_graphql_rate_limit(self, threshold: int = 0) -> tuple[bool, int, int | None]:
        return True, _UNLIMITED_REMAINING, None


def github_client_for(
    repo_root: Path,
    config: OrchestratorConfig,
    *,
    github: Callable[..., GitHubLike],
    dry_run: bool = False,
) -> GitHubLike:
    """The issue/PR backend this repo's config selects.

    Single point of choice between the real ``gh``-backed client and the
    local-file backend; every ``OrchestratorApp`` construction path goes
    through here so the two cannot drift per call site.

    ``github`` is the real client's constructor, passed by the caller as its
    own module-level ``GitHub`` name -- required, with no default, on purpose.
    ``cli.GitHub`` / ``fleet_dispatch.GitHub`` are the injection seam ~60 tests
    patch (``monkeypatch.setattr(cli, "GitHub", Fake)``); a default bound in
    *this* module's namespace would silently bypass every one of them and run
    real ``gh`` from a test. Resolving the name at the call site keeps the
    seam where it has always been.
    """
    local = config.local_issues
    if not local.enabled:
        return github(repo_root=repo_root, runtime=config.runtime, dry_run=dry_run)
    return LocalFileGitHub(
        repo_root=repo_root,
        issues_dir=repo_root / local.issues_dir,
        dry_run=dry_run,
        state_dir=config.runtime.state_dir,
    )
