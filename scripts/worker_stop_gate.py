#!/usr/bin/env python3
"""Fail-closed Stop-hook gate for worker sessions (rework-RCA W2/#1259, W4/#1262).

Moves two prose worker-brief instructions ("lint before you stop", "run the
tests you touched", "register new event kinds") into a mechanism that fails
closed instead of a prompt a model can skim past (see
``docs/plans/rework-rca-mitigations-plan-2026-08-15.md`` section W2/W4).

Registered as a ``Stop`` hook in the repo's tracked ``.claude/settings.json``.
Claude Code invokes this script as a fresh subprocess every time a worker
session is about to stop, feeding it a JSON payload on stdin. It:

1. Fast-paths to an immediate allow (exit 0) when the session has no diff at
   all -- nothing to lint or test.
2. Derives its changed-file set from the UNION of two sources (see
   ``_all_changed_files``): the working tree (``git status --porcelain``)
   and everything committed on this branch since it diverged from
   ``origin/main`` (``git merge-base HEAD origin/main`` .. ``HEAD``). Both
   are needed -- a compliant worker that commits and pushes *before*
   stopping has nothing left in the working tree, so working-tree-only
   scope would silently never check its actual diff.
3. Runs ``ruff check`` and ``ruff format --check`` scoped to exactly the
   changed ``*.py`` files (not the whole tree) -- an untouched, pre-existing
   lint/format issue elsewhere in the repo must never spuriously block a
   session that didn't touch it. Skips ruff entirely when the changed set
   contains no ``.py`` files.
4. Derives a targeted test set from the same changed-file union: any changed
   ``tests/*.py`` file, plus (W4/#1262) ``tests/test_instrumentation.py``
   whenever a changed ``src/*.py`` file's *content* matches an event-emit
   call (``log_event(``, ``append_event(``, ``_record_event(``). W4's issue
   is explicit that the event-kind registry deriver already exists
   (``tests/test_instrumentation.py::test_event_kind_registry_exhaustive``,
   AST-scanning ``src/``) -- this script does not reimplement it, it only
   pulls the existing test earlier (inside the session, before push) instead
   of leaving the omission for CI to catch a round late. No hardcoded
   source-to-test mapping beyond this one explicit rule: the targeted set is
   otherwise derived entirely from the session's own changed files.
5. Blocks the stop on any failure, bounded by a consecutive-block cap
   (``MAX_BLOCKS_PER_SESSION``) so a session that cannot converge still ends
   instead of looping forever; on cap exhaustion it allows the stop while
   printing the ``WORKER_STOP_GATE_EXHAUSTED`` marker. A pass -- or hitting
   the cap and giving up -- resets the streak, so one early stumble does not
   silently disarm the gate for the rest of the session; the counter tracks
   *consecutive* blocks, not a lifetime session budget.

Enforcement surface, disclosed honestly (review round finding, #1259): this
gate sees uncommitted working-tree state plus everything committed on this
branch past its ``origin/main`` merge-base. It cannot see, and does not try
to see, anything already merged into ``origin/main`` itself, and branch-base
derivation is deliberately fail-OPEN for *that one surface* when ambiguous
(detached HEAD, no ``origin/main`` ref, ``HEAD`` already equal to
``origin/main``, or any ``merge-base``/``diff`` failure) -- it silently
narrows to working-tree-only scope rather than blocking on an inability to
determine the branch base. This is the one deliberate exception to this
module's fail-closed default; every other error path in this file blocks.
Two further disclosed limitations, kept as-is per operator decision: (a)
``stop_hook_active`` is recorded to the invocation log only and does not
independently gate anything -- the persisted per-session block counter is
the robust bound against a retry loop, not this stdin flag; (b) the
``.claude/settings.json`` command line has no fallback if the worker's
``.venv`` is missing -- a missing interpreter is treated as fail-OPEN (the
Stop hook simply cannot launch), which is the intentionally safe direction
for an environment-setup failure that is not this gate's job to diagnose.
That command line anchors both the interpreter and this script to
``$CLAUDE_PROJECT_DIR`` (the harness-exported project root) rather than
the hook process's cwd: Claude Code runs hooks from the session's
*current* directory, which drifts whenever a compound command ``cd``s
into a worktree, and a cwd-relative ``.venv/...`` then resolves against a
tree that has no venv and dies with bash's "No such file or directory" on
every subsequent turn (observed 2026-09-03 in a downstream repo that ports
this gate). ``tests/test_worker_stop_gate.py`` pins that invariant: every
path token in the command must be absolute once ``$CLAUDE_PROJECT_DIR``
is expanded.
A third, empirically-confirmed limitation (review round, #1259; resolved as
#1306): scoping ruff to "changed" files did not distinguish a file this
session just wrote from pre-existing untracked debris sitting in a
long-lived interactive checkout -- ``git status`` reports both identically,
so a stray untracked ``.py`` file with real lint/format issues left over
from an unrelated earlier session could still trigger a block in an
interactive (non-worker) session that never touched it. The fix chosen in
#1306 narrows ruff's scope to tracked-modified + committed-since-base only,
excluding untracked (``??``) files from the *ruff* surface specifically.
Untracked files are still included in the W4 emit-site rule and test
targeting, where a brand-new file legitimately needs coverage -- only the
ruff lint/format check is narrowed. The residual cost (a worker's genuinely
new untracked ``.py`` file is not lint/formatted by the gate until it is
staged or committed) is acceptable: the worker workflow ends with a
commit+push, at which point the committed-diff surface covers it, and CI
runs ruff on the whole tree on every push regardless.

Hook contract (verified against the current Claude Code hooks docs and
working reference implementations at implementation time -- the repo has no
prior Stop-hook precedent to anchor to, see issue #1259's binding recon
comment): a Stop hook blocks ONLY via exit code 2 (reason read from stderr)
or stdout JSON ``{"decision": "block", "reason": "..."}``. A plain nonzero
exit that is not 2, with no JSON, is treated as a non-blocking hook error and
the stop proceeds -- i.e. an uncaught exception that produces a bare
traceback and exit 1 is a fail-OPEN trap. This module's ``main()`` therefore
wraps every enforcement decision in a catch-all and always exits via
``_decide_and_report``, which is the single choke point that emits exit code
2 for every block path, including its own internal-error path.

Stdlib-only (worker sessions are not guaranteed to have the ``dev`` extra
installed -- see issue #1259's binding recon comment on
``scripts/merge_autonomy_ratio.py`` being the correct "stdlib-only scripts"
citation, not ``scripts/heartbeat_check.py`` which imports ``psutil``/
``yaml``). It shells out to ``ruff``/``pytest`` via ``uv run --no-sync``
rather than importing them, so the script itself never needs third-party
packages to run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The only reliable way to block a Claude Code Stop hook (verified 2026-08).
BLOCK_EXIT_CODE = 2

#: Consecutive blocks allowed before the gate gives up on the current streak
#: and lets the stop through. A pass, or hitting this cap, resets the
#: counter -- it is not a lifetime session budget. A self-imposed bound,
#: independent of any platform-level retry limit (undocumented as of this
#: verification -- do not assume one).
MAX_BLOCKS_PER_SESSION = 3

#: Printed when the cap above is hit, so an operator grepping worker output
#: can distinguish "gate gave up" from "gate never ran" (silence is not
#: success -- see the module docstring and the invocation log below).
EXHAUSTED_MARKER = "WORKER_STOP_GATE_EXHAUSTED"

GIT_TIMEOUT_SECONDS = 30
RUFF_TIMEOUT_SECONDS = 120
PYTEST_TIMEOUT_SECONDS = 300

#: Age after which a per-session ``.count``/``.log`` pair is opportunistically
#: pruned at startup -- otherwise the state dir accumulates one pair per
#: worker session forever. 14 days comfortably outlives any single session.
STATE_FILE_MAX_AGE_SECONDS = 14 * 24 * 60 * 60

#: W4/#1262 target: the existing exhaustive event-kind registry test.
INSTRUMENTATION_TEST_PATH = "tests/test_instrumentation.py"

#: W4/#1262 rule: a changed src file whose *content* calls one of these three
#: emit functions must pull INSTRUMENTATION_TEST_PATH into this session's
#: targeted-test set. Deliberately a cheap text grep, not an AST call-site
#: analysis -- the plan calls for "a cheap grep of changed files"; the
#: registry-exhaustiveness itself is already enforced by the existing test,
#: this only decides whether to run it early.
_EMIT_SITE_RE = re.compile(r"\b(?:log_event|append_event|_record_event)\(")

_SESSION_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")
_FALLBACK_SESSION_ID = "unknown-session"


class GateError(Exception):
    """Raised when the gate cannot complete a check -- always fails closed."""


@dataclass(frozen=True)
class ChangedFile:
    """One line of ``git status --porcelain`` output, parsed."""

    path: str
    deleted: bool
    #: True only for ``??`` entries -- a file git has never tracked or staged.
    #: Used to narrow ruff's scope to tracked-modified + committed-since-base
    #: only (#1306): an untracked file may be session-written debris that
    #: predates this session, and ``git status`` cannot tell the two apart, so
    #: it is excluded from the *ruff* surface while still appearing in the
    #: W4 emit-site rule and test targeting (a brand-new file legitimately
    #: needs coverage there). Committed-diff entries are never untracked.
    untracked: bool = False


@dataclass(frozen=True)
class GateResult:
    """The gate's enforcement decision for this invocation."""

    block: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Session-keyed state (survives across the fresh-subprocess-per-invocation
# boundary -- env vars and in-memory state do not).
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    path = Path(tempfile.gettempdir()) / "worker_stop_gate"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_session_id(raw: Any) -> str:
    """Sanitize an untrusted stdin field into a safe filename component."""
    if not isinstance(raw, str) or not raw:
        return _FALLBACK_SESSION_ID
    cleaned = _SESSION_ID_SAFE_RE.sub("_", raw)[:128]
    return cleaned or _FALLBACK_SESSION_ID


def _counter_path(session_id: str) -> Path:
    return _state_dir() / f"{session_id}.count"


def _log_path(session_id: str) -> Path:
    return _state_dir() / f"{session_id}.log"


def _read_block_count(path: Path) -> int:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    count = raw.get("count") if isinstance(raw, dict) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return 0
    return count


def _write_block_count(path: Path, count: int) -> None:
    """Atomic temp-file + replace, per the repo's JSON-write invariant."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"count": count}), encoding="utf-8")
    tmp.replace(path)


def _append_invocation_log(session_id: str, message: str) -> None:
    """Best-effort diagnostic trail: absence of evidence != never ran."""
    try:
        with _log_path(session_id).open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip("\n") + "\n")
    except OSError:
        pass  # logging must never be why the gate itself fails


def _prune_stale_state_files(state_dir: Path) -> None:
    """Opportunistically delete ``.count``/``.log`` files older than
    ``STATE_FILE_MAX_AGE_SECONDS``.

    Best-effort only, by design: a failure here (permission error, a file
    disappearing mid-iteration, a full disk) must never affect the gate's
    enforcement decision, so every ``OSError`` is swallowed at both the
    directory-listing level and the per-file level. Bounded to a single,
    non-recursive pass over whatever is in the state dir right now -- never
    a recursive walk, never retried.
    """
    try:
        cutoff = time.time() - STATE_FILE_MAX_AGE_SECONDS
        for entry in state_dir.iterdir():
            if entry.suffix not in (".count", ".log"):
                continue
            try:
                if entry.stat().st_mtime < cutoff:
                    entry.unlink()
            except OSError:
                continue  # one stubborn file must not stop the rest
    except OSError:
        pass  # listing the dir itself failed -- nothing to prune, move on


# ---------------------------------------------------------------------------
# Subprocess plumbing.
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"{' '.join(cmd)!r} failed to run: {exc}") from exc


def _repo_root(cwd: Path) -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, timeout=GIT_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        raise GateError(f"git rev-parse --show-toplevel failed: {proc.stderr.strip()}")
    return Path(proc.stdout.strip())


def _changed_files(repo_root: Path) -> tuple[ChangedFile, ...]:
    """Parse ``git status --porcelain`` (staged + unstaged + untracked).

    ``--untracked-files=all`` is required, not cosmetic: the default
    ``normal`` mode collapses a brand-new untracked directory into a single
    ``?? src/`` line instead of listing the files inside it, which would
    make a worker's newly added source or test file invisible to both the
    W4 emit-site check and the changed-test-file rule below.

    ``-c core.quotePath=false`` is required, not cosmetic, too: git's
    default quoting C-escapes any path containing a non-ASCII byte into a
    quoted octal form (e.g. ``"caf\\303\\251.py"``), which does not
    ``.endswith(".py")`` as a plain string and would silently drop that
    file from every rule below.
    """
    proc = _run(
        ["git", "-c", "core.quotePath=false", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo_root,
        timeout=GIT_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        raise GateError(f"git status --porcelain failed: {proc.stderr.strip()}")
    files: list[ChangedFile] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        status, rest = line[:2], line[3:]
        if " -> " in rest:  # rename: "old -> new"; the new path is the live one
            rest = rest.split(" -> ", 1)[1]
        files.append(ChangedFile(path=rest, deleted="D" in status, untracked=status == "??"))
    return tuple(files)


def _parse_diff_name_status(output: str) -> list[ChangedFile]:
    """Parse ``git diff --name-status`` output (used for the committed-diff
    surface below). Status codes: ``A``/``M``/``D`` are ``STATUS<TAB>path``;
    ``R``/``C`` (rename/copy, with a trailing similarity score digit) are
    ``STATUS<TAB>old<TAB>new`` -- the new path is the live one.
    """
    files: list[ChangedFile] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        path = parts[-1] if status[:1] in ("R", "C") else (parts[1] if len(parts) > 1 else "")
        if not path:
            continue
        files.append(ChangedFile(path=path, deleted=status.startswith("D")))
    return files


def _committed_diff_files(repo_root: Path) -> tuple[ChangedFile, ...]:
    """Files changed in commits on this branch since it diverged from
    ``origin/main``.

    Catches a compliant worker that already committed (and possibly pushed)
    before Stop: its working tree is clean, so ``_changed_files`` alone
    would see nothing and this gate would silently never check its actual
    diff. See the module docstring's "Enforcement surface" section.

    Best-effort and deliberately fail-OPEN for *this surface only*: branch
    base derivation can be ambiguous -- detached HEAD (no branch to define
    "diverged from"), no ``origin/main`` ref, ``HEAD`` already equal to
    ``origin/main`` (nothing committed beyond it), or any ``merge-base``/
    ``diff`` command failure. None of those mean "the committed scope is
    dirty" -- they mean "cannot determine the committed scope" -- so every
    one of them returns an empty tuple and the gate silently narrows to
    working-tree-only scope instead of blocking on the ambiguity. This is
    the ONE deliberate exception to this module's fail-closed default;
    every other error path in this file blocks (see ``GateError``).
    """
    try:
        head_ref = _run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo_root, timeout=GIT_TIMEOUT_SECONDS
        )
        if head_ref.returncode != 0:
            # Detached HEAD: no branch, no well-defined "diverged from" base.
            # merge-base/diff below would likely still resolve fine even on
            # a detached HEAD -- this check is here anyway, as its own named
            # fallback trigger (review round, #1259), rather than folding
            # detached-HEAD into "some later git command happened to fail".
            return ()

        origin_main = _run(
            ["git", "rev-parse", "--verify", "origin/main"],
            cwd=repo_root,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        if origin_main.returncode != 0:
            return ()  # no origin/main ref to diverge from

        base_proc = _run(
            ["git", "merge-base", "HEAD", "origin/main"],
            cwd=repo_root,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        if base_proc.returncode != 0:
            return ()
        base_sha = base_proc.stdout.strip()
        if not base_sha:
            return ()

        diff_proc = _run(
            ["git", "-c", "core.quotePath=false", "diff", "--name-status", base_sha, "HEAD"],
            cwd=repo_root,
            timeout=GIT_TIMEOUT_SECONDS,
        )
        if diff_proc.returncode != 0:
            return ()
        # HEAD == origin/main falls out here for free: merge-base(HEAD,
        # origin/main) == HEAD in that case too, so base_sha == HEAD, the
        # diff is empty, and _parse_diff_name_status returns [] -- no
        # separate "git rev-parse HEAD" round-trip needed to detect it.
        return tuple(_parse_diff_name_status(diff_proc.stdout))
    except GateError:
        return ()


def _all_changed_files(repo_root: Path) -> tuple[ChangedFile, ...]:
    """Union of working-tree state and the committed-since-base diff.

    Used for the W4 emit-site rule and test targeting (review-round
    decision, #1259) -- a file only needs to appear in one of the two
    sources to count as "changed this session". Working-tree state wins on
    a path collision (freshest -- e.g. a file committed earlier in the
    session and then edited again since).

    Ruff scope is a *narrowing* of this union, not the union itself: the
    caller (``_evaluate``) excludes untracked (``??``) files from the ruff
    surface (#1306), because ``git status`` cannot distinguish a file this
    session just wrote from pre-existing untracked debris. Untracked files
    remain in this union so the W4 and test-targeting rules still see them
    -- a brand-new file legitimately needs test coverage even before it is
    staged.
    """
    committed = _committed_diff_files(repo_root)
    working_tree = _changed_files(repo_root)
    by_path: dict[str, ChangedFile] = {cf.path: cf for cf in committed}
    by_path.update((cf.path, cf) for cf in working_tree)
    return tuple(by_path[path] for path in sorted(by_path))


# ---------------------------------------------------------------------------
# W4/#1262 targeting rule.
# ---------------------------------------------------------------------------


def _touches_emit_site(repo_root: Path, changed: tuple[ChangedFile, ...]) -> bool:
    for cf in changed:
        if cf.deleted or not cf.path.startswith("src/") or not cf.path.endswith(".py"):
            continue
        try:
            text = (repo_root / cf.path).read_text(encoding="utf-8")
        except OSError:
            # Can't prove this file is emit-site-free -- fail closed toward
            # running the registry test rather than silently skipping it.
            return True
        if _EMIT_SITE_RE.search(text):
            return True
    return False


def _targeted_tests(repo_root: Path, changed: tuple[ChangedFile, ...]) -> tuple[str, ...]:
    targets = {
        cf.path
        for cf in changed
        if not cf.deleted and cf.path.startswith("tests/") and cf.path.endswith(".py")
    }
    if _touches_emit_site(repo_root, changed):
        targets.add(INSTRUMENTATION_TEST_PATH)
    return tuple(sorted(targets))


# ---------------------------------------------------------------------------
# Lint / test enforcement.
# ---------------------------------------------------------------------------


def _run_ruff(repo_root: Path, py_files: tuple[str, ...]) -> GateResult:
    """Run ``ruff check``/``ruff format --check`` scoped to exactly
    ``py_files`` -- never the whole tree (review-round fix, #1259: a
    pre-existing lint/format issue in a file this session never touched
    must not spuriously block the stop). The caller (``_evaluate``)
    further narrows this to tracked-modified + committed-since-base only,
    excluding untracked ``??`` files (#1306: ``git status`` cannot
    distinguish session-written debris from pre-existing debris). A caller
    with no ``.py`` files in its changed set should not call this at all;
    as a second line of defense it is also a no-op here.
    """
    if not py_files:
        return GateResult(block=False)
    check = _run(
        ["uv", "run", "--no-sync", "ruff", "check", "--force-exclude", *py_files],
        cwd=repo_root,
        timeout=RUFF_TIMEOUT_SECONDS,
    )
    if check.returncode != 0:
        return GateResult(
            block=True, reason="ruff check failed:\n" + (check.stdout + check.stderr).strip()
        )
    fmt = _run(
        ["uv", "run", "--no-sync", "ruff", "format", "--check", "--force-exclude", *py_files],
        cwd=repo_root,
        timeout=RUFF_TIMEOUT_SECONDS,
    )
    if fmt.returncode != 0:
        return GateResult(
            block=True,
            reason="ruff format --check failed:\n" + (fmt.stdout + fmt.stderr).strip(),
        )
    return GateResult(block=False)


def _run_targeted_tests(repo_root: Path, targets: tuple[str, ...]) -> GateResult:
    if not targets:
        return GateResult(block=False)
    proc = _run(
        ["uv", "run", "--no-sync", "pytest", "-q", "--tb=short", *targets],
        cwd=repo_root,
        timeout=PYTEST_TIMEOUT_SECONDS,
    )
    if proc.returncode != 0:
        return GateResult(
            block=True,
            reason=(
                f"targeted tests failed ({', '.join(targets)}):\n"
                + (proc.stdout + proc.stderr).strip()
            ),
        )
    return GateResult(block=False)


def _evaluate(repo_root: Path) -> GateResult:
    changed = _all_changed_files(repo_root)
    if not changed:
        return GateResult(block=False)  # fast path: nothing changed, nothing to check
    # Ruff scope is narrowed to tracked-modified + committed-since-base only
    # (#1306): untracked (``??``) files are excluded because ``git status``
    # cannot distinguish a file this session just wrote from pre-existing
    # untracked debris that predates the session -- and the latter must never
    # spuriously block a session that never touched it. Untracked files are
    # still in ``changed`` for the W4 emit-site rule and test targeting below,
    # where a brand-new file legitimately needs coverage.
    py_files = tuple(
        sorted(
            cf.path
            for cf in changed
            if not cf.deleted and cf.path.endswith(".py") and not cf.untracked
        )
    )
    ruff_result = _run_ruff(repo_root, py_files)
    if ruff_result.block:
        return ruff_result
    targets = _targeted_tests(repo_root, changed)
    return _run_targeted_tests(repo_root, targets)


# ---------------------------------------------------------------------------
# Decision -> exit code, with the bounded-retry / exhaustion contract.
# ---------------------------------------------------------------------------


def _decide_and_report(session_id: str, result: GateResult) -> int:
    counter_path = _counter_path(session_id)
    if not result.block:
        # A pass breaks the consecutive-block streak: reset so a later,
        # unrelated failure gets its own fresh budget instead of inheriting
        # an exhausted counter from an earlier stumble in the same session.
        if _read_block_count(counter_path) != 0:
            _write_block_count(counter_path, 0)
        return 0
    count = _read_block_count(counter_path) + 1
    if count > MAX_BLOCKS_PER_SESSION:
        # Give up on *this* streak and reset it, mirroring the platform's own
        # "force-allow after N consecutive blocks" semantics: exhaustion ends
        # the streak, it does not spend a lifetime session budget.
        _write_block_count(counter_path, 0)
        message = (
            f"{EXHAUSTED_MARKER}: gate blocked {MAX_BLOCKS_PER_SESSION} consecutive "
            f"times; allowing stop. Last reason:\n{result.reason}"
        )
        sys.stderr.write(message + "\n")
        _append_invocation_log(session_id, f"exhausted count={count} reason={result.reason!r}")
        return 0
    _write_block_count(counter_path, count)
    _append_invocation_log(session_id, f"block count={count} reason={result.reason!r}")
    sys.stderr.write(result.reason.rstrip("\n") + "\n")
    return BLOCK_EXIT_CODE


def main() -> int:
    try:
        raw_stdin = sys.stdin.read()
    except OSError:
        raw_stdin = ""
    try:
        payload = json.loads(raw_stdin) if raw_stdin.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except json.JSONDecodeError:
        payload = {}

    session_id = _safe_session_id(payload.get("session_id"))
    stop_hook_active = bool(payload.get("stop_hook_active", False))

    try:
        _prune_stale_state_files(_state_dir())
        counter_path = _counter_path(session_id)
        already = _read_block_count(counter_path)
        if already >= MAX_BLOCKS_PER_SESSION:
            # Already exhausted in a prior invocation this session: skip
            # re-running (possibly slow) checks and let the stop through.
            # Reset the streak too -- exhaustion ends it, it must not
            # silently disarm the rest of the session (see
            # _decide_and_report's matching reset on the normal path).
            _write_block_count(counter_path, 0)
            message = (
                f"{EXHAUSTED_MARKER}: prior consecutive blocks={already}; "
                "allowing stop without re-check."
            )
            sys.stderr.write(message + "\n")
            _append_invocation_log(session_id, message)
            return 0

        try:
            cwd_value = payload.get("cwd")
            cwd = Path(cwd_value) if isinstance(cwd_value, str) and cwd_value else Path.cwd()
            repo_root = _repo_root(cwd)
            result = _evaluate(repo_root)
        except GateError as exc:
            result = GateResult(block=True, reason=f"internal gate error: {exc}")
        except Exception as exc:  # noqa: BLE001 - any unexpected failure must fail CLOSED
            result = GateResult(
                block=True, reason=f"internal gate error: {type(exc).__name__}: {exc}"
            )

        _append_invocation_log(
            session_id, f"invocation stop_hook_active={stop_hook_active} block={result.block}"
        )
        return _decide_and_report(session_id, result)
    except Exception as exc:  # noqa: BLE001 - ultimate fail-closed backstop
        # Anything unexpected here -- including a failure while recording the
        # decision itself -- must still block. A bare nonzero/non-2 exit is
        # what the hook contract treats as fail-OPEN (see module docstring).
        sys.stderr.write(f"worker_stop_gate: unhandled error, failing closed: {exc}\n")
        return BLOCK_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
