"""PreToolUse hook: block raw merges that fail ``charlie merge-check`` (#894).

Merge authorization is enforced on the orchestrator's own lanes (``merge_ready``,
the Aviator re-queue) but a raw ``gh pr merge`` typed in an agent session bypasses
both, and the #502 tripwire only reports the bypass after the merge is
irreversible (the PR #759 incident). ``merge_check`` is the pure preflight built
for exactly this interception point; this module is the caller that wires it in.

Registered in the repo's checked-in ``.claude/settings.json`` for two tools:

- ``Bash``: commands containing a ``gh pr merge`` invocation.
- ``mcp__github__merge_pull_request``: the GitHub MCP merge tool.

Scope is the fleet, not the world: the target repo is resolved to a local state
root via the fleet registry (``fleet.json``), the same source the orchestrator
uses. A merge into a repo the fleet does not manage is out of scope and passes
through undecided (the normal permission flow still applies). Within the fleet
the hook fails closed: an unparseable PR number, an unreadable registry, or a
failing ``merge_check`` all deny. Operators with a legitimate exception record
it first via ``charlie merge-authorize`` — ``merge_check`` honors the override,
so the hook does too, and the authorization is on the audit record instead of
in a chat justification.

Protocol: reads the PreToolUse JSON on stdin; emits a ``permissionDecision``
of ``deny`` with a reason, or nothing (undecided) for out-of-scope calls.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

# ``gh ... pr ... merge`` anywhere in a shell command, including after
# ``&&``/``;``/``|``. gh (cobra) allows global flags interspersed between the
# words (``gh -R o/r pr merge``, ``gh pr -R o/r merge``), so the intervening
# tokens are matched too — but never across a shell separator, so a mention
# followed by an unrelated ``merge`` in a later command does not trip it.
_GH_PR_MERGE = re.compile(r"\bgh(?:\s+[^\s;|&]+)*?\s+pr(?:\s+[^\s;|&]+)*?\s+merge\b")


def _load_fleet_roots() -> dict[str, Path] | None:
    """Map ``owner/name`` -> local repo root from the fleet registry.

    The two failure shapes are deliberately distinct, because they demand
    opposite answers from an authorization gate:

    - ``None``: the registry file exists but cannot be read or parsed. We
      cannot confirm *any* repo is outside the fleet, so every merge-shaped
      call must deny (fail closed) — on both the Bash and MCP paths.
    - ``{}``: no registry file, or a well-formed registry with no repos.
      Nothing is fleet-managed, so there is nothing for this hook to guard
      and merges pass through to the normal permission flow.

    ``fleet_registry._load_registry`` swallows read errors into an empty
    registry, which would collapse the two shapes; read the file directly.
    """
    from charlie_work import layout

    path = layout.fleet_registry_path()
    try:
        if not path.exists():
            return {}
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(registry, dict):
        return None
    roots: dict[str, Path] = {}
    repos = registry.get("repos")
    for name, entry in repos.items() if isinstance(repos, dict) else ():
        root = entry.get("repo_root") if isinstance(entry, dict) else None
        if isinstance(root, str) and root:
            roots[name.lower()] = Path(root)
    return roots


def _repo_for_cwd(roots: dict[str, Path], cwd: Path) -> str | None:
    """Which fleet repo contains ``cwd``, by exact-prefix containment."""
    resolved = cwd.resolve()
    for name, root in roots.items():
        try:
            root_resolved = root.resolve()
        except OSError:
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return name
    return None


_SHELL_SEPARATORS = frozenset({";", "|", "&", "&&", "||", "\n"})

# ``gh pr merge``'s own flag surface, used only to decide whether the token
# after a flag is the flag's value or a positional argument. This is a UX
# optimization, not a safety boundary: a flag NOT listed here (new gh release,
# typo) makes the following positional ambiguous and the invocation fails
# CLOSED (pr=None -> deny), never open. Staleness degrades to an inconvenient
# deny that tells the operator to put the PR number first.
_MERGE_VALUE_FLAGS = frozenset(
    {"t", "subject", "b", "body", "F", "body-file", "A", "author-email", "match-head-commit"}
)
_MERGE_BOOLEAN_FLAGS = frozenset(
    {
        "admin",
        "auto",
        "disable-auto",
        "delete-branch",
        "d",
        "merge",
        "m",
        "squash",
        "s",
        "rebase",
        "r",
        "help",
    }
)


def _parse_gh_merge_targets(command: str) -> list[dict[str, Any]]:
    """Extract every ``gh pr merge`` invocation's PR number and ``--repo``.

    Detection is token-based, not substring-based: the command is shlex-split
    and an invocation is the words ``gh``, ``pr``, ``merge`` in order with only
    flag tokens between them (gh's cobra CLI accepts global flags interspersed
    anywhere — see ``_match_gh_merge``). A mention of the command inside a
    quoted string (a commit message, an echo) is a single token after splitting
    and therefore does not match — the first version of this hook denied its
    own feature commit because the message *described* the command it guards.

    Splitting uses ``punctuation_chars`` so shell separators (``;``, ``&&``,
    ``|``) become their own tokens instead of gluing onto neighbors (``5;``),
    which would otherwise let a first invocation's argument scan swallow a
    second invocation entirely.

    A ``GH_REPO=owner/repo`` assignment token anywhere before an invocation is
    tracked and applied to invocations that lack an explicit ``-R``/``--repo``,
    mirroring gh's own precedence (flag > GH_REPO > cwd).

    If the command cannot be tokenized (unbalanced quotes), fall back to the
    raw-text regex: a match there yields ``pr=None``, which the caller must
    treat as fail-closed.

    A ``pr`` of ``None`` means the invocation merges "the current branch's PR"
    or the number could not be parsed — the caller must fail closed on that.
    """
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:
        return [{"pr": None, "repo": None}] if _GH_PR_MERGE.search(command) else []

    targets: list[dict[str, Any]] = []
    gh_repo_env: str | None = None
    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        # gh's documented GH_REPO override applies to any later gh call in the
        # same shell command (prefix assignment, ``env``, or ``export``).
        # Tracking every assignment seen so far and applying the latest is
        # conservative in the deny direction: it may attribute a repo to an
        # invocation the shell would not have given it to, never the reverse.
        env_match = re.fullmatch(r"GH_REPO=(\S+)", tok)
        if env_match:
            gh_repo_env = env_match.group(1)
            idx += 1
            continue
        matched_end, flag_repo = _match_gh_pr_merge(tokens, idx)
        if matched_end is None:
            idx += 1
            continue
        pr: int | None = None
        pr_ambiguous = False
        repo: str | None = flag_repo
        i = matched_end
        while i < len(tokens):
            tok = tokens[i]
            if tok in _SHELL_SEPARATORS:
                break
            if tok in ("-R", "--repo") and i + 1 < len(tokens):
                repo = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--repo="):
                repo = tok.split("=", 1)[1]
            elif tok.startswith("-"):
                if "=" in tok or tok.lstrip("-") in _MERGE_BOOLEAN_FLAGS:
                    pass  # self-contained; the next token is a real argument
                elif tok.lstrip("-") in _MERGE_VALUE_FLAGS and i + 1 < len(tokens):
                    i += 2  # consume the flag's value so it can't be read as a PR
                    continue
                elif pr is None:
                    # Unknown flag before the PR number: if it takes a value,
                    # the next token is that value, and reading it as the PR
                    # would make merge-check validate the WRONG pull request
                    # (round-3 review finding). Unknowable here -> ambiguous,
                    # which the caller treats as fail-closed.
                    pr_ambiguous = True
            elif pr is None and not pr_ambiguous and re.fullmatch(r"\d+", tok):
                pr = int(tok)
            elif pr is None and not pr_ambiguous and tok.startswith("https://github.com/"):
                url_match = re.search(r"/pull/(\d+)", tok)
                if url_match:
                    pr = int(url_match.group(1))
                    repo_match = re.search(r"github\.com/([^/]+/[^/]+)/pull/", tok)
                    if repo_match:
                        repo = repo_match.group(1)
            i += 1
        # Same precedence as gh itself: explicit flag > GH_REPO > cwd (None).
        if repo is None:
            repo = gh_repo_env
        # ``gh`` accepts OWNER/REPO or a full URL for --repo; normalize URLs.
        if repo and repo.startswith("https://github.com/"):
            repo = "/".join(repo.rstrip("/").split("/")[-2:])
        targets.append({"pr": pr, "repo": repo})
        idx = i
    return targets


def _match_gh_pr_merge(tokens: list[str], start: int) -> tuple[int | None, str | None]:
    """Match a ``gh ... pr ... merge`` invocation starting at ``tokens[start]``.

    gh (cobra) accepts flags interspersed anywhere, so ``gh pr -R o/r merge 5``
    and ``gh -R o/r pr merge 5`` are both valid invocations — strict 3-token
    adjacency would silently pass them (round-2 review finding on PR #1195).
    This skips flag tokens between the words, consuming a value token for the
    repo flags it understands and capturing it.

    Returns ``(index_after_merge, repo_from_flags)`` on a match, else
    ``(None, None)``.
    """
    if tokens[start] != "gh":
        return None, None
    repo: str | None = None
    i = start + 1
    for expected_word in ("pr", "merge"):
        while i < len(tokens):
            tok = tokens[i]
            if tok in _SHELL_SEPARATORS:
                return None, None
            if tok in ("-R", "--repo") and i + 1 < len(tokens):
                repo = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--repo="):
                repo = tok.split("=", 1)[1]
                i += 1
                continue
            if tok.startswith("-"):
                # Unknown flag: assume boolean and keep scanning. If it
                # actually consumed the next word, the worst case is a
                # spurious match on a merge-shaped command, which errs
                # toward deny — never toward bypass.
                i += 1
                continue
            break
        if i >= len(tokens) or tokens[i] != expected_word:
            return None, None
        i += 1
    return i, repo


def _run_merge_check(repo_root: Path, pr_number: int) -> tuple[bool, str]:
    """Run ``charlie --repo <root> merge-check <pr>`` in-process."""
    import contextlib
    import io

    from charlie_work.cli import main as charlie_main

    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            rc = charlie_main(["--repo", str(repo_root), "merge-check", str(pr_number)])
    except SystemExit as exc:  # argparse exits are still an answer
        rc = int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001 - a broken preflight must deny, not crash the hook
        return False, f"merge-check raised {type(exc).__name__}: {exc}"
    return rc == 0, out.getvalue().strip()


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def _decide(tool_name: str, tool_input: dict[str, Any], cwd: Path) -> str | None:
    """Return a deny reason, or ``None`` to leave the call undecided."""
    if tool_name == "mcp__github__merge_pull_request":
        owner = tool_input.get("owner")
        repo = tool_input.get("repo")
        pr = tool_input.get("pullNumber")
        name = f"{owner}/{repo}".lower() if owner and repo else None
        roots = _load_fleet_roots()
        if roots is None:
            # Registry unreadable: cannot confirm this repo is outside the
            # fleet, so the gate must not answer yes (same contract as the
            # Bash path).
            return (
                "merge_pull_request intercepted but the fleet registry is "
                "unreadable; cannot verify merge authorization (#894). Fix "
                "fleet.json or merge via charlie ship-it."
            )
        if name is None or name not in roots:
            return None  # not a fleet repo — out of scope
        if not isinstance(pr, int) or isinstance(pr, bool):
            return f"cannot determine PR number from merge_pull_request input {pr!r}"
        ok, detail = _run_merge_check(roots[name], pr)
        if ok:
            return None
        return (
            f"merge-check denied PR #{pr} in {name}: {detail or 'not authorized'}. "
            f"Record a verdict (charlie verdict) or an operator override "
            f"(charlie merge-authorize) first."
        )

    if tool_name != "Bash":
        return None
    command = tool_input.get("command") or ""
    targets = _parse_gh_merge_targets(command)
    if not targets:
        return None

    roots = _load_fleet_roots()
    if roots is None:
        # Registry unreadable: cannot confirm any target is outside the
        # fleet — even an explicit --repo. The gate must not answer yes.
        return (
            "gh pr merge intercepted but the fleet registry is unreadable; "
            "cannot verify merge authorization (#894). Fix fleet.json or "
            "merge via charlie ship-it."
        )
    cwd_repo = _repo_for_cwd(roots, cwd)
    for target in targets:
        name = (target["repo"] or cwd_repo or "").lower() or None
        if name is None or name not in roots:
            continue  # merging outside the fleet — out of scope
        pr = target["pr"]
        if pr is None:
            return (
                f"gh pr merge without an explicit PR number in fleet repo {name}; "
                f"cannot preflight authorization (#894). Re-run as "
                f"'gh pr merge <number> ...' so merge-check can verify it."
            )
        ok, detail = _run_merge_check(roots[name], pr)
        if not ok:
            return (
                f"merge-check denied PR #{pr} in {name}: {detail or 'not authorized'}. "
                f"Record a verdict (charlie verdict) or an operator override "
                f"(charlie merge-authorize) first."
            )
    return None


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return 0  # malformed hook input — leave undecided rather than break the tool
    tool_name = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return 0
    try:
        reason = _decide(tool_name, tool_input, Path.cwd())
    except Exception as exc:  # noqa: BLE001 - hook crash must not deny unrelated tools
        # Only merge-shaped calls reach the risky code; deny those on error.
        command = tool_input.get("command") or ""
        is_merge_shaped = tool_name == "mcp__github__merge_pull_request" or (
            tool_name == "Bash"
            and bool(_GH_PR_MERGE.search(command) or _parse_gh_merge_targets(command))
        )
        if is_merge_shaped:
            _deny(f"merge preflight hook errored ({type(exc).__name__}: {exc}); failing closed")
            return 0
        return 0
    if reason is not None:
        _deny(reason)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the hook subprocess
    sys.exit(main())
