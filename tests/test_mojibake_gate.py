"""Tests for the mojibake CI gate (issue #1057).

Covers the pure scanning functions (``is_mojibake``, ``recover_mojibake``,
``find_mojibake_in_diff``) and the CLI command that wires them together for
``charlie mojibake-check --base <ref>``.

The centerpiece is ``test_em_dash_mojibake_from_issue_1057_is_detected``: a
regression fixture using the exact byte sequence documented in the issue
(``\\xc3\\xa2\\xe2\\x82\\xac\\xe2\\x80\\x9d`` -- the UTF-8 encoding of the
cp1252 misdecoding of an em-dash).  The detection is derived from the
encoding process (reverse the corruption and check whether the result
differs), not a hardcoded list of bad byte sequences.
"""

from __future__ import annotations

import argparse

import pytest

from charlie_work import cli as cli_module
from charlie_work.config import OrchestratorConfig
from charlie_work.mojibake_gate import (
    MojibakeFinding,
    find_mojibake_in_diff,
    is_mojibake,
    recover_mojibake,
)
from charlie_work.subprocess_runner import RunResult

# --- The exact mojibake from issue #1057 ---
#
# An em-dash (U+2014) is UTF-8 \\xe2\\x80\\x94.  When those three bytes are
# decoded as cp1252, they become: \\xe2 -> \\u00e2 (a-circumflex), \\x80 ->
# \\u20ac (euro sign), \\x94 -> \\u201d (right double quotation mark).  That
# three-character string re-encoded as UTF-8 is the
# \\xc3\\xa2\\xe2\\x82\\xac\\xe2\\x80\\x9d byte sequence the issue documents.
#
# Constructed programmatically from the encoding process itself, so the
# fixture cannot rot if the test file's own encoding is ever mis-handled.
_EM_DASH = "\u2014"
_MOJIBAKE_EM_DASH = _EM_DASH.encode("utf-8").decode("cp1252")  # \u00e2\u20ac\u201d


# --- is_mojibake / recover_mojibake: core detection behavior ---


def test_em_dash_mojibake_from_issue_1057_is_detected() -> None:
    # The exact corruption: em-dash -> a-circumflex/euro/right-double-quote.
    assert is_mojibake(_MOJIBAKE_EM_DASH) is True
    recovered = recover_mojibake(_MOJIBAKE_EM_DASH)
    assert recovered == _EM_DASH


def test_em_dash_mojibake_byte_sequence_matches_issue() -> None:
    # The byte sequence the issue documents: \\xc3\\xa2\\xe2\\x82\\xac\\xe2\\x80\\x9d
    assert _MOJIBAKE_EM_DASH.encode("utf-8") == b"\xc3\xa2\xe2\x82\xac\xe2\x80\x9d"


def test_pure_ascii_is_not_mojibake() -> None:
    assert is_mojibake("def foo(): return 42") is False
    assert is_mojibake("") is False
    assert recover_mojibake("def foo(): return 42") is None


def test_legitimate_latin1_accented_character_is_not_mojibake() -> None:
    # "cafe" with e-acute (U+00E9) -- a legitimate non-ASCII character that
    # is NOT the result of a UTF-8/cp1252 round trip.  The round-trip fails
    # because \\xe9 is not a valid UTF-8 lead byte.
    assert is_mojibake("caf\u00e9") is False
    assert recover_mojibake("caf\u00e9") is None


def test_legitimate_em_dash_is_not_mojibake() -> None:
    # A correctly-encoded em-dash is not mojibake -- it is the *correct* text.
    # The round-trip: "\\u2014".encode("cp1252") succeeds (em-dash is in
    # cp1252 at 0x97) -> \\x97, which is not valid UTF-8 -> fails.  Not flagged.
    assert is_mojibake(_EM_DASH) is False


def test_mojibake_inside_a_comment_line_is_detected() -> None:
    # The real-world shape: a comment line with mojibake punctuation.
    line = "# some text " + _MOJIBAKE_EM_DASH + " more text"
    assert is_mojibake(line) is True
    recovered = recover_mojibake(line)
    assert recovered == "# some text " + _EM_DASH + " more text"


def test_mojibake_mixed_with_emoji_is_detected() -> None:
    # A line that contains both mojibake and a character outside cp1252
    # (emoji, U+1F600).  The whole-line round-trip fails (emoji can't encode
    # as cp1252), but the chunk-based fallback catches the mojibake substring.
    line = "# comment " + _MOJIBAKE_EM_DASH + " with emoji \U0001f600"
    assert is_mojibake(line) is True


def test_multiple_mojibake_sequences_in_one_line() -> None:
    # Two em-dashes corrupted in the same line.
    line = "# a " + _MOJIBAKE_EM_DASH + " b " + _MOJIBAKE_EM_DASH + " c"
    assert is_mojibake(line) is True
    recovered = recover_mojibake(line)
    assert recovered == "# a " + _EM_DASH + " b " + _EM_DASH + " c"


def test_other_common_mojibake_sequences_are_detected() -> None:
    # The issue mentions the detection should not be limited to em-dashes.
    # Verify a few other common UTF-8/cp1252 mojibake sequences are caught.
    # en-dash (U+2013, UTF-8 \\xe2\\x80\\x93):
    en_dash = "\u2013"
    mojibake_en_dash = en_dash.encode("utf-8").decode("cp1252")
    assert is_mojibake(mojibake_en_dash) is True
    assert recover_mojibake(mojibake_en_dash) == en_dash

    # left double quotation mark (U+201C, UTF-8 \\xe2\\x80\\x9c):
    ldquo = "\u201c"
    mojibake_ldquo = ldquo.encode("utf-8").decode("cp1252")
    assert is_mojibake(mojibake_ldquo) is True
    assert recover_mojibake(mojibake_ldquo) == ldquo

    # e-acute (U+00E9, UTF-8 \\xc3\\xa9) -> cp1252: \\u00c3 (A-tilde) + \\u00a9 (copyright):
    eacute = "\u00e9"
    mojibake_eacute = eacute.encode("utf-8").decode("cp1252")
    assert is_mojibake(mojibake_eacute) is True
    assert recover_mojibake(mojibake_eacute) == eacute


def test_mojibake_finding_is_a_frozen_dataclass() -> None:
    # CLAUDE.md invariant: config/value objects are frozen dataclasses.
    finding = MojibakeFinding(path="src/file.py", line_number=42, content="bad", recovered="good")
    with pytest.raises(AttributeError):
        finding.path = "other"  # type: ignore[misc]


# --- find_mojibake_in_diff: unified diff parsing ---


def _make_diff(path: str, added_lines: list[str], context: str = "context line") -> str:
    """Build a minimal unified diff with the given added lines."""
    lines = [
        f"diff --git a/{path} b/{path}",
        f"--- a/{path}",
        f"+++ b/{path}",
        "@@ -1,3 +1,{} @@".format(2 + len(added_lines)),
        f" {context}",
    ]
    for added in added_lines:
        lines.append(f"+{added}")
    lines.append(f" {context}")
    return "\n".join(lines) + "\n"


def test_diff_with_mojibake_in_added_lines_is_flagged() -> None:
    diff = _make_diff("src/example.py", ["# comment " + _MOJIBAKE_EM_DASH + " here"])
    findings = find_mojibake_in_diff(diff)
    assert len(findings) == 1
    f = findings[0]
    assert f.path == "src/example.py"
    assert f.line_number == 2  # first added line in the new file
    assert _EM_DASH in f.recovered


def test_diff_with_multiple_mojibake_lines() -> None:
    diff = _make_diff(
        "src/example.py",
        [
            "# line1 " + _MOJIBAKE_EM_DASH + " text",
            "# line2 " + _MOJIBAKE_EM_DASH + " text",
        ],
    )
    findings = find_mojibake_in_diff(diff)
    assert len(findings) == 2
    assert {f.line_number for f in findings} == {2, 3}


def test_clean_diff_has_no_findings() -> None:
    diff = _make_diff("src/example.py", ["# clean comment with em-dash " + _EM_DASH])
    findings = find_mojibake_in_diff(diff)
    assert findings == []


def test_context_and_removed_lines_are_not_scanned() -> None:
    # Mojibake in a context or removed line should NOT be flagged -- only
    # added lines are scanned, because the gate checks what the PR introduces.
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n"
        " # context " + _MOJIBAKE_EM_DASH + " here\n"
        "-" + "# removed " + _MOJIBAKE_EM_DASH + " here\n"
        "+# clean added line\n"
    )
    findings = find_mojibake_in_diff(diff)
    assert findings == []


def test_multiple_files_in_diff() -> None:
    diff = (
        "diff --git a/file1.py b/file1.py\n"
        "--- a/file1.py\n"
        "+++ b/file1.py\n"
        "@@ -1,1 +1,2 @@\n"
        " # ctx\n"
        "+# file1 " + _MOJIBAKE_EM_DASH + " bad\n"
        "diff --git a/file2.py b/file2.py\n"
        "--- a/file2.py\n"
        "+++ b/file2.py\n"
        "@@ -1,1 +1,2 @@\n"
        " # ctx\n"
        "+# file2 " + _MOJIBAKE_EM_DASH + " bad\n"
    )
    findings = find_mojibake_in_diff(diff)
    assert len(findings) == 2
    assert {f.path for f in findings} == {"file1.py", "file2.py"}


def test_empty_diff_has_no_findings() -> None:
    assert find_mojibake_in_diff("") == []


def test_deleted_file_is_not_scanned() -> None:
    # /dev/null as the new file means the file was deleted -- no added lines.
    diff = (
        "diff --git a/gone.py b/gone.py\n"
        "--- a/gone.py\n"
        "+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n"
        "-" + "# removed " + _MOJIBAKE_EM_DASH + " here\n"
    )
    findings = find_mojibake_in_diff(diff)
    assert findings == []


def test_added_line_starting_with_plus_plus_is_scanned() -> None:
    # Regression for the "+++" collision bug: an added line whose content
    # starts with "++" (so the raw diff line is "+++...") must NOT be mistaken
    # for the "+++ b/path" file header.  The true header requires a trailing
    # space and is consumed earlier in the loop; this added line has no space
    # after "+++" (the 4th char is "#") and must be scanned as a normal added
    # line.
    #
    # The added line content is "++#bad " + mojibake em-dash + " here", so the
    # raw diff line is "+++#bad <mojibake> here".  Before the fix, the
    # `line.startswith("+++")` guard in the added/context/removed
    # classification skipped this line entirely (and did not advance the line
    # counter).
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,3 @@\n"
        " # ctx\n"
        "+++#bad " + _MOJIBAKE_EM_DASH + " here\n"
        " # ctx\n"
    )
    findings = find_mojibake_in_diff(diff)
    assert len(findings) == 1
    f = findings[0]
    assert f.path == "f.py"
    assert f.line_number == 2  # second line in the new file
    assert f.content == "++#bad " + _MOJIBAKE_EM_DASH + " here"
    assert f.recovered == "++#bad " + _EM_DASH + " here"


def test_added_line_starting_with_plus_plus_keeps_line_counter_in_sync() -> None:
    # The "+++" collision bug also desynced the line counter: the skipped
    # added line did not advance new_line_number, so subsequent findings in
    # the same hunk reported wrong line numbers.  This test places a
    # "++"-prefixed added line (clean, no mojibake) BEFORE a mojibake added
    # line and asserts the mojibake finding's line number accounts for the
    # earlier added line.
    diff = (
        "diff --git a/f.py b/f.py\n"
        "--- a/f.py\n"
        "+++ b/f.py\n"
        "@@ -1,2 +1,4 @@\n"
        " # ctx\n"
        "+++#clean line starting with ++\n"
        "+#bad " + _MOJIBAKE_EM_DASH + " here\n"
        " # ctx\n"
    )
    findings = find_mojibake_in_diff(diff)
    assert len(findings) == 1
    f = findings[0]
    assert f.path == "f.py"
    # Line 2 = "+++#clean...", line 3 = "+#bad <mojibake>".
    # If the counter desynced (the "++" line skipped without advancing),
    # this would report line 2 instead of 3.
    assert f.line_number == 3


# --- CLI wiring: charlie mojibake-check --base <ref> ---


def _cli_args(base: str = "origin/main") -> argparse.Namespace:
    return argparse.Namespace(repo=None, config=None, fleet_dir=None, dry_run=False, base=base)


def _mock_git_diff(stdout: str, ok: bool = True) -> RunResult:
    if ok:
        return RunResult(returncode=0, stdout=stdout, stderr="", error=None)
    return RunResult(returncode=1, stdout="", stderr="boom", error="git diff failed")


def test_cli_mojibake_check_fails_on_corrupted_diff(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit, **kw: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    diff_output = _make_diff("src/example.py", ["# bad " + _MOJIBAKE_EM_DASH + " line"])
    monkeypatch.setattr(
        cli_module,
        "run_captured",
        lambda *a, **k: _mock_git_diff(diff_output),
    )

    result = cli_module.run_mojibake_check_command(_cli_args())

    assert result.ok is False
    assert len(result.data["findings"]) == 1
    assert result.data["findings"][0]["path"] == "src/example.py"
    assert "mojibake-check" in result.message


def test_cli_mojibake_check_passes_on_clean_diff(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit, **kw: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    diff_output = _make_diff("src/example.py", ["# clean line with " + _EM_DASH + " dash"])
    monkeypatch.setattr(
        cli_module,
        "run_captured",
        lambda *a, **k: _mock_git_diff(diff_output),
    )

    result = cli_module.run_mojibake_check_command(_cli_args())

    assert result.ok is True
    assert result.data["findings"] == []
    assert "clean" in result.message


def test_cli_mojibake_check_reports_git_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit, **kw: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(
        cli_module,
        "run_captured",
        lambda *a, **k: _mock_git_diff("", ok=False),
    )

    result = cli_module.run_mojibake_check_command(_cli_args("bad-ref"))

    assert result.ok is False
    assert "could not run git diff" in result.message
    assert result.data["base"] == "bad-ref"


def test_cli_mojibake_check_passes_base_to_git_diff(monkeypatch, tmp_path) -> None:
    # Verify the --base argument is forwarded to `git diff base..HEAD`.
    # Two-dot (not three-dot) is used because CI runs against a shallow
    # clone (fetch-depth: 1) where three-dot cannot resolve the merge-base.
    captured_args: list[list[str]] = []

    def fake_run_captured(command, **kwargs):
        captured_args.append(command)
        return _mock_git_diff("")

    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit, **kw: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(cli_module, "run_captured", fake_run_captured)

    cli_module.run_mojibake_check_command(_cli_args("abc123"))

    assert len(captured_args) == 1
    assert captured_args[0] == ["git", "diff", "abc123..HEAD"]


def test_cli_mojibake_check_uses_two_dot_not_three_dot(monkeypatch, tmp_path) -> None:
    # Regression test for the CI failure on PR #1206: the three-dot diff
    # (base...HEAD) exits 128 in a shallow clone because the merge-base
    # cannot be resolved without the ancestry chain.  The two-dot diff
    # (base..HEAD) compares trees directly and works once both commits are
    # present.  This test asserts the command string contains ".." not
    # "..." so a regression to three-dot is caught immediately.
    captured_args: list[list[str]] = []

    def fake_run_captured(command, **kwargs):
        captured_args.append(command)
        return _mock_git_diff("")

    monkeypatch.setattr(cli_module, "find_repo_root", lambda repo, explicit, **kw: tmp_path)
    monkeypatch.setattr(cli_module, "load_layered_config", lambda *a, **k: OrchestratorConfig())
    monkeypatch.setattr(cli_module, "run_captured", fake_run_captured)

    cli_module.run_mojibake_check_command(_cli_args("b4d1bf3c37424ff943f9032020157067e1d28f69"))

    cmd = captured_args[0]
    assert cmd == [
        "git",
        "diff",
        "b4d1bf3c37424ff943f9032020157067e1d28f69..HEAD",
    ]
    # Explicitly assert no three-dot: "..." must not appear in the ref spec.
    ref_spec = cmd[2]
    assert "..." not in ref_spec, (
        f"three-dot diff would fail in shallow clones (CI exit 128); got {ref_spec!r}"
    )
