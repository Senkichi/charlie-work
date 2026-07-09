"""TOOL-V2-01 / ADR 006C structural guard — no byte reads of Claude Code credentials.

This module enforces ADR 006C Option B at the source level. It walks every
``*.py`` file under ``src/freecode/`` and raises if any code path:

1. Reads *content bytes* from paths associated with the Claude Code Linux/Windows
   credential cache (``~/.claude/.credentials.json``); or
2. Calls ``keyring.get_password("Claude Code", ...)`` outside the allowlisted
   tool module — the structural deny rule for the macOS Keychain surface
   (ADR 006C OQ4).

Permitted (stat-only, per ADR 006C Locked behaviors mirror of ADR 006A):
  - ``Path.exists()``
  - ``Path.stat()`` / ``os.stat()``
  - ``os.path.getmtime()`` / ``os.path.getsize()``

Forbidden (reads content — could exfiltrate token/cookie bytes):
  - ``open(...)`` with a target string co-locating ``.claude`` and ``.credentials.json``
  - ``Path.read_text(...)`` / ``Path.read_bytes(...)`` on such paths
  - ``keyring.get_password("Claude Code", ...)`` outside ``src/freecode/tooling/claude_code.py``

The composite path rule reflects ADR 006C §Structural-guard fragment constants:
``CLAUDE_DIR_FRAGMENT`` co-located with ``CLAUDE_CREDENTIALS_FILE_FRAGMENT`` in
the same path literal. Bare ``.credentials.json`` (without ``.claude``) does not
match — this prevents incidental false positives against any other tool that
happens to use a similar filename.

This file mirrors the structure of ``test_gemini_oauth_credential_isolation.py``,
``test_gemini_cli_credential_path_guard.py``, and ``test_codex_credential_isolation.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "freecode"


# Composite rule: a path literal matches when BOTH fragments appear in the same
# string (ADR 006C §Structural-guard fragment constants). The composite check is
# what distinguishes Claude Code credential paths from incidental uses of
# ``.credentials.json`` in unrelated code.
_CLAUDE_DIR_FRAGMENT = ".claude"
_CLAUDE_CREDENTIALS_FILE_FRAGMENT = ".credentials.json"


def _literal_is_claude_credential_path(s: str) -> bool:
    """ADR 006C composite rule: ``.claude`` AND ``.credentials.json`` co-located."""

    return _CLAUDE_DIR_FRAGMENT in s and _CLAUDE_CREDENTIALS_FILE_FRAGMENT in s


def _node_targets_claude_credential_path(node: ast.AST) -> bool:
    """Walk a Call argument expression looking for a composite-matching literal."""

    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _literal_is_claude_credential_path(sub.value):
                return True
        if isinstance(sub, ast.JoinedStr):
            joined = "".join(
                v.value
                for v in sub.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            if _literal_is_claude_credential_path(joined):
                return True
    return False


def _collect_python_files() -> list[Path]:
    return sorted(_REPO_SRC.rglob("*.py"))


_FORBIDDEN_CALLS = frozenset({"open", "read_text", "read_bytes"})


def _ast_call_names(path: Path, target_names: frozenset[str]) -> list[tuple[int, str]]:
    """Return (lineno, qualified_func_name) for every Call node whose function
    name (Name.id or Attribute.attr) appears in ``target_names``.

    Docstrings and comments are ignored — only real Call expressions count
    (mirrors the opencode test pattern).
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname: str | None = None
        if isinstance(node.func, ast.Name):
            fname = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fname = node.func.attr
        if fname in target_names:
            hits.append((node.lineno, fname))
    return hits


def test_claude_code_tool_module_has_no_byte_reads() -> None:
    """ADR 006C — ``tooling/claude_code.py`` must not open or read credential files.

    AST-only scan: docstrings and comments that *describe* the prohibition are
    permitted (and expected); only real Call expressions are flagged.
    """

    path = _REPO_SRC / "tooling" / "claude_code.py"
    assert path.exists(), f"claude_code.py not found at {path}"
    hits = _ast_call_names(path, _FORBIDDEN_CALLS)
    assert not hits, (
        "tooling/claude_code.py must remain stat-only on credential paths "
        f"(ADR 006C); found forbidden calls: {hits!r}"
    )


def test_claude_code_env_module_has_no_byte_reads() -> None:
    """ADR 006C — ``cli_lane/claude_code_env.py`` must remain content-read-free.

    AST scan: the env builder is keyring → in-memory → subprocess; it has no
    legitimate reason to call open / read_text / read_bytes anywhere.
    """

    path = _REPO_SRC / "cli_lane" / "claude_code_env.py"
    assert path.exists(), f"claude_code_env.py not found at {path}"
    hits = _ast_call_names(path, _FORBIDDEN_CALLS)
    assert not hits, (
        "cli_lane/claude_code_env.py must not open files (ADR 006C Option B); "
        f"found forbidden calls: {hits!r}"
    )


def test_claude_code_credential_paths_ast_guard() -> None:
    """AST scan — no open/read_text/read_bytes on a string literal that
    co-locates ``.claude`` and ``.credentials.json`` under src/freecode/.

    Mirrors GEM-10e (ADR 006A) for Claude Code. Allowlist is intentionally
    empty: ADR 006C Locked behaviors explicitly prohibit byte reads with no
    exception.
    """

    offenders: list[str] = []
    for py in _collect_python_files():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name: str | None = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name not in {"open", "read_bytes", "read_text"}:
                continue
            # Receiver expression (for ``obj.read_text()``) may itself encode
            # the path (e.g. ``Path("~/.claude/.credentials.json").read_text()``).
            receivers: list[ast.AST] = []
            if isinstance(node.func, ast.Attribute):
                receivers.append(node.func.value)
            for arg in node.args:
                receivers.append(arg)
            for expr in receivers:
                if _node_targets_claude_credential_path(expr):
                    rel = py.relative_to(_REPO_SRC.parent.parent)
                    offenders.append(
                        f"{rel}:{node.lineno}: forbidden {func_name}() near "
                        f"Claude Code credential path literal"
                    )

    assert not offenders, (
        "ADR 006C structural guard found forbidden Claude Code credential byte reads:\n"
        + "\n".join(f"  {v}" for v in offenders)
        + "\n\nOnly stat/exists calls are permitted on Claude Code credential paths "
        "(ADR 006C Locked behaviors)."
    )


# ---------------------------------------------------------------------------
# ADR 006C OQ4 — macOS Keychain deny rule
# ---------------------------------------------------------------------------

# Only this module may legitimately reference the literal service name
# "Claude Code" (and even there it must not be passed to keyring.get_password).
# Allowlisted module: the structural guard test itself defines the rule's
# allowlist; the tool module is the policy holder. In practice, no source file
# should call ``keyring.get_password("Claude Code", ...)`` at all under
# ADR 006C Option B — freecode's keyring usage is scoped to its own
# ``ANTHROPIC_API_KEY`` slot keyed by ``provider_id="claude_code"``, never to
# Anthropic's own ``"Claude Code"`` service name.
_CLAUDE_CODE_TOOL_MODULE = _REPO_SRC / "tooling" / "claude_code.py"


def _call_targets_claude_code_service(node: ast.Call) -> bool:
    """True if this Call passes the literal "Claude Code" as the first positional arg.

    Matches ``keyring.get_password("Claude Code", ...)``-style calls regardless of
    whether the call target is bound as ``keyring.get_password`` or via an alias.
    """

    if not node.args:
        return False
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value == "Claude Code"
    return False


def test_macos_keychain_deny_rule_no_get_password_for_claude_code() -> None:
    """ADR 006C OQ4 — no source file may call ``keyring.get_password("Claude Code", ...)``.

    Allowlist is documentation-only: ``tooling/claude_code.py`` is the module
    that owns Claude Code credential policy, but even it must not call into
    Anthropic's own Keychain service slot. The keyring usage in the tool is
    scoped to freecode's own ``provider_id="claude_code"`` slot (via
    ``freecode.security.keys.get_api_key``), never to the literal
    ``"Claude Code"`` service.
    """

    offenders: list[str] = []
    for py in _collect_python_files():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func_name: str | None = None
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            if func_name != "get_password":
                continue
            if _call_targets_claude_code_service(node):
                rel = py.relative_to(_REPO_SRC.parent.parent)
                offenders.append(
                    f"{rel}:{node.lineno}: forbidden keyring.get_password("
                    f"\"Claude Code\", ...) — ADR 006C OQ4 deny rule"
                )

    assert not offenders, (
        "ADR 006C OQ4 structural deny rule violated:\n"
        + "\n".join(f"  {v}" for v in offenders)
        + "\n\nfreecode's keyring usage for Claude Code must scope to the "
        "freecode-managed ANTHROPIC_API_KEY slot keyed by provider_id='claude_code', "
        "never to Anthropic's own 'Claude Code' service name."
    )


def test_claude_code_constants_in_known_credential_paths() -> None:
    """ADR 006C §Structural-guard fragment constants — extend the aggregate set."""

    from freecode.tooling import _known_credential_paths as kcp

    assert kcp.CLAUDE_DIR_FRAGMENT == ".claude"
    assert kcp.CLAUDE_CREDENTIALS_FILE_FRAGMENT == ".credentials.json"
    assert kcp.CLAUDE_DIR_FRAGMENT in kcp.CLI_CREDENTIAL_PATH_FRAGMENTS
    assert kcp.CLAUDE_CREDENTIALS_FILE_FRAGMENT in kcp.CLI_CREDENTIAL_PATH_FRAGMENTS


def test_claude_code_helper_paths_do_not_read_filesystem() -> None:
    """``claude_config_dir`` and ``claude_credentials_path`` are pure ``Path`` joins."""

    from pathlib import Path as _P

    from freecode.tooling._known_credential_paths import (
        claude_config_dir,
        claude_credentials_path,
    )

    home = _P("/tmp/never-touched")
    cfg = claude_config_dir(home)
    creds = claude_credentials_path(home)
    assert cfg == home / ".claude"
    assert creds == home / ".claude" / ".credentials.json"
    # Pure ``Path`` operations should not touch the filesystem.
    assert not cfg.exists()


def test_guard_failure_template_references_006c() -> None:
    """Failure diagnostics must cite ADR 006C (grep-verifiable)."""

    text = Path(__file__).read_text(encoding="utf-8")
    assert "006C" in text


# ---------------------------------------------------------------------------
# Negative case — the composite rule rejects a fabricated offender
# ---------------------------------------------------------------------------


def test_composite_rule_matches_fabricated_offender() -> None:
    """Sanity check: a synthetic literal containing both fragments matches.

    Mirrors the spirit of the codex ``_literal_contains_codex_fragment`` self-test:
    if the matcher silently ignored the composite rule, every guard above would
    pass vacuously.
    """

    assert _literal_is_claude_credential_path("/home/me/.claude/.credentials.json")
    # Negative: bare ``.credentials.json`` without ``.claude`` does NOT match
    # (this is the documented false-positive guard from ADR 006C
    # §Structural-guard fragment constants).
    assert not _literal_is_claude_credential_path("/etc/some/.credentials.json")
    # Negative: ``.claude`` without the credentials filename does NOT match.
    assert not _literal_is_claude_credential_path("/home/me/.claude/config.yaml")
