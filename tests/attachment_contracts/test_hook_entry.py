"""Tests for hook_entry.py: PreToolUse stdin protocol.

Drives `hook_entry.main()` directly with stdin/stdout/stderr StringIO and an
explicit env mapping (never touches real sys.stdin/os.environ), against
tmp_path repos.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from charlie_work.attachment_contracts.baseline import BASELINE_FILENAME, dump, generate
from charlie_work.attachment_contracts.excludes import load_excludes
from charlie_work.attachment_contracts.archetypes import scan_tree
from charlie_work.attachment_contracts.hook_entry import main
from charlie_work.attachment_contracts.outliers import saturate_all

_SMALL_CLASS = """
class {name}:
    def alpha(self): pass
    def beta(self): pass
"""

_BIG_CLASS_TEMPLATE = """
class Big:
{methods}
"""


def _big_class_source(count: int) -> str:
    # Non-digit-suffixed method names -- see test_check.py for why a bare
    # `<prefix><int>` sequence would misclassify as a ledger.
    methods = "\n".join(f"    def m{i}x(self): pass" for i in range(count))
    return _BIG_CLASS_TEMPLATE.format(methods=methods)


def _build_repo(root: Path, big_member_count: int = 25) -> None:
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "a.py").write_text(_SMALL_CLASS.format(name="A"), encoding="utf-8")
    (root / "src" / "pkg" / "b.py").write_text(_SMALL_CLASS.format(name="B"), encoding="utf-8")
    (root / "src" / "pkg" / "c.py").write_text(_SMALL_CLASS.format(name="C"), encoding="utf-8")
    (root / "src" / "pkg" / "big.py").write_text(
        _big_class_source(big_member_count), encoding="utf-8"
    )


def _freeze_baseline_at(root: Path, big_member_count: int) -> None:
    """Freeze a baseline where Big is saturated at `big_member_count`, then
    grow Big past that in the actual file (the on-disk repo already has the
    grown version from `_build_repo`; this writes a shrunk snapshot first,
    freezes it, then restores growth) -- simpler: freeze straight from a
    smaller on-disk state, then grow.
    """
    excludes = load_excludes(root)
    scan = scan_tree(root, excludes)
    kinds = sorted({p.kind for p in scan.points})
    verdicts = saturate_all(scan.points, kinds)
    document = generate(verdicts, generated_by="test", generated_at="t", floor=4)
    dump(document, root / BASELINE_FILENAME)


def _over_budget_repo(root: Path) -> Path:
    """Repo with a committed baseline where `big.py`'s `Big` class is already
    over its frozen ceiling -- the standard fixture for advisory/enforce tests.
    """
    _build_repo(root, big_member_count=20)
    _freeze_baseline_at(root, big_member_count=20)
    (root / "src" / "pkg" / "big.py").write_text(_big_class_source(25), encoding="utf-8")
    return root / "src" / "pkg" / "big.py"


def _run(
    file_path: Path, root_env: dict[str, str] | None = None, tool_name: str = "Edit"
) -> tuple[int, str, str]:
    stdin = io.StringIO(
        json.dumps({"tool_name": tool_name, "tool_input": {"file_path": str(file_path)}})
    )
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(stdin=stdin, stdout=stdout, stderr=stderr, env=root_env or {})
    return code, stdout.getvalue(), stderr.getvalue()


# ---------------------------------------------------------------------------
# fast no-op: no baseline found anywhere upward
# ---------------------------------------------------------------------------


def test_no_baseline_upward_is_fast_noop(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    target = tmp_path / "src" / "pkg" / "target.py"
    target.write_text("def f(): pass\n", encoding="utf-8")

    code, out, err = _run(target)

    assert code == 0
    assert out == ""
    assert err == ""


# ---------------------------------------------------------------------------
# advisory JSON output (default mode = advise)
# ---------------------------------------------------------------------------


def test_advisory_json_output_default_mode(tmp_path: Path) -> None:
    target = _over_budget_repo(tmp_path)

    code, out, err = _run(target)

    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert "additionalContext" in payload["hookSpecificOutput"]
    assert "Big" in payload["hookSpecificOutput"]["additionalContext"]

    log_path = tmp_path / ".var" / "attachment-contracts" / "advisories.jsonl"
    assert log_path.is_file()
    logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(entry["identity"] == "Big" for entry in logged)


# ---------------------------------------------------------------------------
# enforce mode: interactive -> exit 2
# ---------------------------------------------------------------------------


def test_enforce_mode_interactive_exits_2(tmp_path: Path) -> None:
    target = _over_budget_repo(tmp_path)

    code, out, err = _run(target, root_env={"ATTACHMENT_CONTRACTS_MODE": "enforce"})

    assert code == 2
    assert out == ""
    assert "Big" in err


# ---------------------------------------------------------------------------
# unattended: never blocks, even in enforce mode
# ---------------------------------------------------------------------------


def test_unattended_never_blocks_even_in_enforce_mode(tmp_path: Path) -> None:
    target = _over_budget_repo(tmp_path)

    code, out, err = _run(
        target,
        root_env={"ATTACHMENT_CONTRACTS_MODE": "enforce", "CHARLIE_FLEET_WORKER": "1"},
    )

    assert code == 0
    assert err == ""
    payload = json.loads(out)
    assert "additionalContext" in payload["hookSpecificOutput"]


def test_unattended_via_claude_code_env_var(tmp_path: Path) -> None:
    target = _over_budget_repo(tmp_path)

    code, out, err = _run(
        target,
        root_env={"ATTACHMENT_CONTRACTS_MODE": "enforce", "CLAUDE_CODE_UNATTENDED": "1"},
    )

    assert code == 0
    assert err == ""


# ---------------------------------------------------------------------------
# G6 parse failure surfaces through the hook too
# ---------------------------------------------------------------------------


def test_g6_parse_failure_surfaces_as_advisory_finding(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=2)
    _freeze_baseline_at(tmp_path, big_member_count=2)
    broken = tmp_path / "src" / "pkg" / "broken.py"
    broken.write_text("def broken(:\n    pass\n", encoding="utf-8")

    code, out, err = _run(broken)

    assert code == 0
    payload = json.loads(out)
    assert "G6" in payload["hookSpecificOutput"]["additionalContext"]


# ---------------------------------------------------------------------------
# clean file under a piloted repo: no findings -> no output at all
# ---------------------------------------------------------------------------


def test_clean_file_produces_no_output(tmp_path: Path) -> None:
    _build_repo(tmp_path, big_member_count=2)
    _freeze_baseline_at(tmp_path, big_member_count=2)

    code, out, err = _run(tmp_path / "src" / "pkg" / "a.py")

    assert code == 0
    assert out == ""
    assert err == ""
