from __future__ import annotations

import ast
import io
import threading
import tokenize
from pathlib import Path

import pytest

from charlie_work import state as state_module
from charlie_work.config import OrchestratorConfig
from charlie_work.paths import runtime_paths
from charlie_work.state import (
    append_event,
    load_state,
    load_state_locked,
    save_state,
    state_lock,
)
from charlie_work.workflow import OrchestratorApp
from _src_ast import parsed, source_files, source_text

# The single sanctioned lock-free state reader (issue #3141):
# ``state.read_state_snapshot`` never quarantines, renames, or writes, and
# reports read/parse failure to its caller instead of returning empty state.
# Every lock-free read goes through it; ``load_state`` -- which quarantines
# on failure -- is never permitted unlocked, in ANY spelling.
_LOCK_FREE_STATE_READER = "read_state_snapshot"

# Unlocked ``load_state`` call sites exempted from the lint below, as
# ``(path relative to src/charlie_work, enclosing function)`` -- the same
# module-qualified convention ``_SANCTIONED_MODULE_FUNCTIONS`` uses in
# ``test_state_decision_cache_enforcement``. It is deliberately EMPTY: after
# the issue #3141 rework every lock-free read goes through
# ``read_state_snapshot``, so no unlocked ``load_state`` may exist at all.
# Adding an entry here requires the same justification comment the sibling
# lint demands -- an unlocked ``load_state`` quarantines a live state.json
# on read failure (issue #310), which is why the reader primitive exists.
_SANCTIONED_UNLOCKED_LOAD_STATE: frozenset[tuple[str, str]] = frozenset()

_LOCK_CONTEXT_NAMES = frozenset({"state_lock", "advisory_file_lock"})

# Insignificant tokens that may sit between a callee and its ``(`` (a
# ``load_state`` line-continued inside enclosing brackets is still a call).
_TOKEN_GATE_SKIP = frozenset({tokenize.NL, tokenize.COMMENT, tokenize.NEWLINE})


def _fstring_hides_load_state(literal: str) -> bool:
    """True when a STRING token is an f-string whose text names ``load_state``.

    Pre-PEP-701 (Python <3.12) an f-string tokenizes as ONE STRING token,
    so ``f"{load_state(p)}"`` emits neither the NAME nor the ``(`` the pair
    check below needs -- yet ``ast`` still parses the interpolation as a
    real ``Call`` the visitor would flag. Any f-prefixed literal containing
    the name therefore keeps the file: doubt lands on the AST stage, which
    parses interpolations and decides (a text-only ``f"load_state failed"``
    mention is simply an extra ``parsed()`` call, never a bypass). On 3.12+
    f-strings split into FSTRING_* tokens and the ordinary pair check fires
    inside them, so an f-prefixed STRING token can only appear on <3.12.
    """
    prefix_end = 0
    while prefix_end < len(literal) and literal[prefix_end].isalpha():
        prefix_end += 1
    return "f" in literal[:prefix_end].lower() and "load_state" in literal


def _has_load_state_call_tokens(text: str) -> bool:
    """True when ``text`` tokenizes a ``load_state`` NAME that is called.

    Both flaggable call spellings -- ``load_state(...)`` and
    ``x.load_state(...)`` -- emit the token pair ``NAME 'load_state'``
    followed by ``OP '('``, so a file without the pair cannot hold a call
    the visitor below would flag and never needs ``parsed()``. A paren-
    wrapped callee (``(load_state)(args)``) hides the pair behind ``)``s,
    so a ``)``-run after the NAME still counts when a ``(`` follows.
    ``load_state`` inside strings/comments or inside a longer identifier
    (``load_state_locked``, ``reload_state``) produces no matching NAME
    token and is correctly skipped -- except inside an f-string on
    Python <3.12, where the whole literal is one STRING token that can
    hide an interpolated call; ``_fstring_hides_load_state`` keeps those
    files so the pair check cannot wave an f-string-interpolated call
    past the lint. A tokenize failure returns True so the file falls
    through to the authoritative AST stage -- including an ERRORTOKEN,
    the anomaly a <3.12 tokenizer emits (instead of raising TokenError)
    for an unterminated string.
    """
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        saw_name = False
        for token in tokens:
            if token.type in _TOKEN_GATE_SKIP:
                continue
            if token.type == tokenize.ERRORTOKEN:
                return True
            if saw_name and token.type == tokenize.OP and token.string in "()":
                if token.string == "(":
                    return True
                # A ``)``-run: ``(load_state)(args)`` -- keep watching.
                continue
            if token.type == tokenize.STRING and _fstring_hides_load_state(token.string):
                return True
            saw_name = token.type == tokenize.NAME and token.string == "load_state"
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return True
    return False


class _LoadStateLockVisitor(ast.NodeVisitor):
    """Flag ``load_state`` calls not inside a lock context.

    Both spellings are matched, deliberately: the pre-#3141 lint only caught
    the bare ``load_state(...)`` name form, which is exactly how an unlocked
    read slipped in via ``_state.load_state(...)`` -- the attribute spelling
    this visitor also flags. Lock contexts are recognized in both spellings
    too (``with state_lock(...)`` and ``with _wf.state_lock(...)`` /
    ``with ports.state_lock(...)``), plus ``advisory_file_lock``, the generic
    primitive ``state_lock`` wraps.
    """

    def __init__(self) -> None:
        self._lock_depth = 0
        self._func_stack: list[str] = []
        self.errors: list[str] = []
        # (lineno, enclosing function) of every flagged call, for the
        # sanction/ staleness checks in the test below.
        self.flagged: list[tuple[int, str]] = []

    def _enter_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._func_stack.append(node.name)
        self.generic_visit(node)
        self._func_stack.pop()

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    def _current_function(self) -> str:
        return self._func_stack[-1] if self._func_stack else "<module>"

    @staticmethod
    def _is_lock_context(expr: ast.expr) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func
        if isinstance(func, ast.Name):
            return func.id in _LOCK_CONTEXT_NAMES
        return isinstance(func, ast.Attribute) and func.attr in _LOCK_CONTEXT_NAMES

    @staticmethod
    def _is_load_state_call(func: ast.expr) -> bool:
        if isinstance(func, ast.Name):
            return func.id == "load_state"
        return isinstance(func, ast.Attribute) and func.attr == "load_state"

    def visit_With(self, node: ast.With) -> None:
        entered = sum(1 for item in node.items if self._is_lock_context(item.context_expr))
        self._lock_depth += entered
        self.generic_visit(node)
        self._lock_depth -= entered

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_load_state_call(node.func) and self._lock_depth <= 0:
            self.errors.append(f"line {node.lineno}: {ast.unparse(node)}")
            self.flagged.append((node.lineno, self._current_function()))
        self.generic_visit(node)


class FakeGitHub:
    """Minimal stub sufficient for OrchestratorApp.status()."""

    def __init__(self) -> None:
        self.issues = [
            {
                "number": 123,
                "title": "Fix search",
                "url": "https://example.test/issues/123",
                "body": "Search is broken",
                "labels": [{"name": "automated-ready"}],
                "state": "OPEN",
            }
        ]
        self.prs = [
            {
                "number": 456,
                "title": "Fix #123: search",
                "url": "https://example.test/pull/456",
                "headRefName": "agent/issue-123-fix-search",
                "headRefOid": "sha-abc123",
                "mergeStateStatus": "CLEAN",
                "body": "Closes #123\n\nTests: regression coverage added.",
                "labels": [],
                "isCrossRepository": False,
                "state": "OPEN",
            }
        ]

    def issue_list(self, labels=None, state=None):
        if isinstance(labels, str):
            return [
                issue
                for issue in self.issues
                if labels in {label["name"] for label in issue.get("labels", [])}
            ]
        elif labels:
            return [
                issue
                for issue in self.issues
                if any(
                    label in {label_obj["name"] for label_obj in issue.get("labels", [])}
                    for label in labels
                )
            ]
        return self.issues

    def pr_list(self):
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "OPEN"]

    def merged_pr_list(self):
        # Issue #1337: status() now calls merged_pr_list() to compute the
        # merged-PR coverage exclusion set for the reachability classifier.
        return [pr for pr in self.prs if pr.get("state", "OPEN").upper() == "MERGED"]

    def check_graphql_rate_limit(self, threshold: int) -> tuple[bool, int, int | None]:
        return (True, 10000, 0)

    def run(self, args, *, json_output=False, allow_failure=False):
        return [] if json_output else ""


def test_load_state_locked_uses_state_lock(tmp_path: Path, monkeypatch) -> None:
    """load_state_locked must acquire the advisory lock before reading."""
    state_path = tmp_path / "state.json"
    lock_calls: list[Path] = []
    orig_state_lock = state_module.state_lock

    def tracking_state_lock(path: Path):
        lock_calls.append(path)
        return orig_state_lock(path)

    monkeypatch.setattr(state_module, "state_lock", tracking_state_lock)

    result = load_state_locked(state_path)

    assert result["version"] == 1
    assert lock_calls == [state_path]


def test_load_state_locked_matches_load_state(tmp_path: Path) -> None:
    """A locked read returns the same state an explicit lock+load would."""
    state_path = tmp_path / "state.json"
    save_state(state_path, {"issues": {"1": {"number": 1}}})

    with state_lock(state_path):
        expected = load_state(state_path)

    actual = load_state_locked(state_path)

    assert actual == expected


def test_status_concurrent_writer_no_quarantine(tmp_path: Path) -> None:
    """Concurrent status() calls and a state writer must never quarantine a healthy state file.

    Regression for issue #310: status() read state.json without state_lock, so a
    writer's tmp+replace could race a read and (after retry exhaustion) cause the
    reader to quarantine a perfectly healthy state.json.
    """
    config = OrchestratorConfig()
    paths = runtime_paths(tmp_path, config.runtime.state_dir)
    app = OrchestratorApp(tmp_path, paths, config, FakeGitHub())
    state_file = paths.state_file

    iterations = 200
    errors: list[Exception] = []

    def writer() -> None:
        for i in range(iterations):
            try:
                with state_lock(state_file):
                    state = load_state(state_file)
                    state = append_event(state, "stress", {"i": i})
                    save_state(state_file, state)
            except Exception as exc:  # pragma: no cover - failures fail the test
                errors.append(exc)
                break

    def reader() -> None:
        for _ in range(iterations):
            try:
                app.status()
            except Exception as exc:  # pragma: no cover - failures fail the test
                errors.append(exc)
                break

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    writer_thread.join(timeout=30)
    reader_thread.join(timeout=30)

    assert not errors, f"exceptions during concurrent stress: {errors}"

    quarantine_files = list(state_file.parent.glob(f"{state_file.name}.corrupt-*"))
    assert not quarantine_files, f"healthy state file was quarantined: {quarantine_files}"

    final_state = load_state(state_file)
    assert final_state["version"] == 1
    assert len(final_state.get("events", [])) == iterations


@pytest.fixture(scope="module")
def unlocked_load_state_calls() -> list[str]:
    """Every ``load_state`` call outside a lock context under ``src/charlie_work``.

    Scanning the package is this guard's dominant cost, so it is charged to
    a module fixture -- the test's ``setup`` -- instead of the ``call``
    phase the ledger times (the accepted #2662/#2867/#2875/#2882 remedy
    family, applied to this guard for #3196; ``test_env_sanitize_call_sites``
    scans the same corpus the same way). If setup ever flags, ``ci-fleet
    ledger report --section slow-fixtures`` is its owner.

    Files reach ``parsed()`` through two gates: the ``"load_state"``
    substring prefilter (a violation needs the literal characters), then
    ``_has_load_state_call_tokens`` -- a file without a called
    ``load_state`` NAME token can only *mention* it (imports,
    ``load_state_locked`` sites, comments) and cannot hold a flaggable
    call. Measured at this tree: 541 scanned files shrink to 63 parse
    candidates.
    """
    src_root = Path(__file__).resolve().parents[1] / "src" / "charlie_work"

    errors: list[str] = []
    scanned = 0
    for source_file in source_files(src_root):
        if source_file.name == "test_load_state_locked.py":
            continue
        rel = source_file.relative_to(src_root).as_posix()
        sanctioned = {func for (path, func) in _SANCTIONED_UNLOCKED_LOAD_STATE if path == rel}
        text = source_text(source_file)
        flagged_funcs: set[str] = set()
        if "load_state" in text and _has_load_state_call_tokens(text):
            scanned += 1
            tree = parsed(source_file)
            visitor = _LoadStateLockVisitor()
            visitor.visit(tree)
            flagged_funcs = {func for _lineno, func in visitor.flagged}
            for error, (lineno, func) in zip(visitor.errors, visitor.flagged):
                if func in sanctioned:
                    continue
                errors.append(f"{rel}:{error} in {func}()")
        # A sanction whose flagged call vanished must not sit here silently
        # allow-listing nothing (same staleness rule as the decision-cache
        # lint in test_state_decision_cache_enforcement.py). The check runs
        # for every file, gated or not -- a gated-out file has no calls at
        # all, so any sanction naming it is stale by definition.
        stale = sanctioned - flagged_funcs
        assert not stale, (
            f"_SANCTIONED_UNLOCKED_LOAD_STATE lists {sorted(stale)} for {rel} "
            "but no unlocked load_state call was found there anymore -- "
            "remove the stale entry."
        )

    # Positive control: a scan that matched nothing reports "zero offenders"
    # the same as a clean tree, so the guard must prove it parsed real
    # call sites before its empty error list means anything.
    assert scanned, (
        "the scan found no load_state call sites under src/charlie_work -- "
        "the guard is blind, not green"
    )
    return errors


def test_no_unlocked_load_state_in_production_code(
    unlocked_load_state_calls: list[str],
) -> None:
    """Every load_state call in src/charlie_work must be inside a lock context.

    This is the lint-style enforcement from issue #310: the locked read helper
    (load_state_locked) should be used for read-only callers, and all raw
    load_state calls must occur inside a state_lock block.

    Issue #3141: the scan now flags the attribute spelling (``_state.load_state``,
    ``_wf.load_state``, ``ports.load_state``) too -- the original Name-only
    matcher is how the quarantining reader slipped into the fleet census --
    and recognizes attribute-form lock contexts (``with _wf.state_lock(...)``,
    ``with ports.state_lock(...)``). Lock-free reads that need no lock must
    use ``state.read_state_snapshot`` -- the single sanctioned unlocked
    reader, which never quarantines -- or be listed in
    ``_SANCTIONED_UNLOCKED_LOAD_STATE`` above with justification.
    """
    assert not unlocked_load_state_calls, (
        f"load_state calls outside a state lock: {unlocked_load_state_calls}. "
        f"Unlocked reads go through {_LOCK_FREE_STATE_READER}() -- the "
        "sanctioned non-mutating reader -- or need an explicit entry in "
        "_SANCTIONED_UNLOCKED_LOAD_STATE."
    )


def test_load_state_token_gate_fails_closed() -> None:
    """The token gate must keep every file a flaggable call could hide in.

    The corpus scan skips files ``_has_load_state_call_tokens`` clears, so a
    blind spot in the gate is a lint bypass the AST stage never sees. Pin the
    flagged call spellings (bare, attribute, paren-wrapped, line-continued,
    f-string-interpolated) as kept, and non-call mentions as cleared. A
    tokenize failure must keep the file -- gate doubt always lands on the
    AST side.
    """
    assert _has_load_state_call_tokens("load_state(path)\n")
    assert _has_load_state_call_tokens("_state.load_state(path)\n")
    assert _has_load_state_call_tokens("(load_state)(path)\n")
    assert _has_load_state_call_tokens("x = f(load_state\n(path))\n")
    assert _has_load_state_call_tokens("await load_state(path)\n")
    # TokenError on 3.12+, ERRORTOKEN on <3.12 -- either way, keep.
    assert _has_load_state_call_tokens("x = 'unclosed\n")

    # An interpolated call must keep on EVERY supported interpreter: on
    # <3.12 the f-string is one STRING token kept by
    # ``_fstring_hides_load_state``; on 3.12+ PEP 701 splits it into
    # FSTRING_* tokens and the NAME+``(`` pair fires. Both spellings,
    # both prefix cases.
    assert _has_load_state_call_tokens('x = f"{load_state(p)}"\n')
    assert _has_load_state_call_tokens('x = f"{_state.load_state(p)}"\n')
    assert _has_load_state_call_tokens("x = F'{load_state(p)}'\n")

    assert not _has_load_state_call_tokens("from .state import load_state\n")
    assert not _has_load_state_call_tokens("load_state_locked(path)\n")
    assert not _has_load_state_call_tokens("reload_state(path)\n")
    assert not _has_load_state_call_tokens("load_state\n")
    assert not _has_load_state_call_tokens("# load_state(path)\n")
    assert not _has_load_state_call_tokens('"""load_state(path)"""\n')
    # A plain string carrying the same text is still cleared -- the
    # f-string keep is prefix-scoped, not a blanket STRING keep.
    assert not _has_load_state_call_tokens('x = "load_state(p)"\n')
    assert not _has_load_state_call_tokens("x = b'load_state(p)'\n")


def test_load_state_lint_fails_closed_on_attribute_spelling() -> None:
    """The scanner must flag the attribute spelling it was extended to catch.

    A lint that only matches ``load_state(...)`` bare-Name calls fails open
    the moment an unlocked read is written ``_state.load_state(...)`` --
    the exact bypass the issue #3141 review caught. This pins both call
    spellings AND both lock-context spellings against a synthetic module.
    """
    source = """
import something as _state


def unlocked_bare(path):
    return load_state(path)


def unlocked_attr(path):
    return _state.load_state(path)


def locked_bare(path):
    with state_lock(path):
        return load_state(path)


def locked_attr(path):
    with _wf.state_lock(path):
        return _wf.load_state(path)


def locked_generic(path):
    with ports.advisory_file_lock(path):
        return ports.load_state(path)


def sanctioned_reader(path):
    return _state.read_state_snapshot(path)
"""
    visitor = _LoadStateLockVisitor()
    visitor.visit(ast.parse(source))

    flagged = {func for _lineno, func in visitor.flagged}
    assert flagged == {"unlocked_bare", "unlocked_attr"}, (
        f"expected exactly the two unlocked spellings flagged, got: {visitor.errors}"
    )
