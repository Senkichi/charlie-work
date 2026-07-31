"""Tests for safe_ref's SHA / ref-name format validators (issue #659).

Covers the defense-in-depth boundary that prevents attacker-influenced or
hand-edited persisted values from being parsed as git flags when they reach
a ``subprocess`` argv list. The validators are format-check only — git's own
ref/SHA naming rules already reject flag-like strings today; these guards
keep that true after future refactors.
"""

from __future__ import annotations

import pytest

from charlie_work.safe_ref import (
    require_valid_ref_name,
    require_valid_rev,
    require_valid_sha,
)


# --- require_valid_sha --------------------------------------------------------


class TestRequireValidSha:
    def test_accepts_full_sha1(self) -> None:
        sha = "a" * 40
        assert require_valid_sha(sha, context="test") == sha

    def test_accepts_full_sha256(self) -> None:
        sha = "a" * 64
        assert require_valid_sha(sha, context="test") == sha

    def test_accepts_abbreviated_sha(self) -> None:
        assert require_valid_sha("abc123", context="test") == "abc123"

    def test_accepts_min_length_4(self) -> None:
        assert require_valid_sha("abcd", context="test") == "abcd"

    def test_accepts_uppercase_hex(self) -> None:
        assert require_valid_sha("ABCD1234", context="test") == "ABCD1234"

    def test_rejects_too_short(self) -> None:
        with pytest.raises(ValueError, match="not a valid git object SHA"):
            require_valid_sha("abc", context="test")

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ValueError, match="not a valid git object SHA"):
            require_valid_sha("a" * 65, context="test")

    def test_rejects_non_hex(self) -> None:
        with pytest.raises(ValueError, match="not a valid git object SHA"):
            require_valid_sha("xyz123", context="test")

    def test_rejects_flag_like(self) -> None:
        """A leading ``-`` must be rejected — this is the primary defense."""
        with pytest.raises(ValueError, match="not a valid git object SHA"):
            require_valid_sha("--exec=foo", context="test")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="not a valid git object SHA"):
            require_valid_sha("", context="test")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match="not a valid git object SHA"):
            require_valid_sha(123456, context="test")  # type: ignore[arg-type]

    def test_error_includes_context(self) -> None:
        with pytest.raises(ValueError, match="my-context"):
            require_valid_sha("bad", context="my-context")


# --- require_valid_ref_name ---------------------------------------------------


class TestRequireValidRefName:
    def test_accepts_simple_branch(self) -> None:
        assert require_valid_ref_name("main", context="test") == "main"

    def test_accepts_hierarchical_branch(self) -> None:
        assert (
            require_valid_ref_name("agent/issue-659-fix", context="test") == "agent/issue-659-fix"
        )

    def test_accepts_remote_tracking_ref(self) -> None:
        assert require_valid_ref_name("origin/main", context="test") == "origin/main"

    def test_accepts_plus_in_branch(self) -> None:
        """Git-legal ref names containing '+' must not be rejected."""
        assert require_valid_ref_name("feature/+hotfix", context="test") == "feature/+hotfix"

    def test_accepts_head(self) -> None:
        assert require_valid_ref_name("HEAD", context="test") == "HEAD"

    def test_rejects_leading_dash(self) -> None:
        """Leading ``-`` is flag injection — the primary defense."""
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name("--exec=foo", context="test")

    def test_rejects_leading_dot(self) -> None:
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name(".hidden", context="test")

    def test_rejects_leading_slash(self) -> None:
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name("/branch", context="test")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            require_valid_ref_name("", context="test")

    def test_rejects_double_dot(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            require_valid_ref_name("foo..bar", context="test")

    def test_rejects_at_brace(self) -> None:
        with pytest.raises(ValueError, match="@"):
            require_valid_ref_name("foo@{bar", context="test")

    def test_rejects_trailing_slash(self) -> None:
        with pytest.raises(ValueError, match="ends with"):
            require_valid_ref_name("foo/", context="test")

    def test_rejects_trailing_dot(self) -> None:
        with pytest.raises(ValueError, match="ends with"):
            require_valid_ref_name("foo.", context="test")

    def test_rejects_rev_syntax_caret(self) -> None:
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name("foo^bar", context="test")

    def test_rejects_rev_syntax_tilde(self) -> None:
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name("foo~1", context="test")

    def test_rejects_rev_syntax_colon(self) -> None:
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name("foo:bar", context="test")

    def test_rejects_whitespace(self) -> None:
        with pytest.raises(ValueError, match="not a valid git ref name"):
            require_valid_ref_name("foo bar", context="test")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            require_valid_ref_name(123, context="test")  # type: ignore[arg-type]

    def test_error_includes_context(self) -> None:
        with pytest.raises(ValueError, match="my-ctx"):
            require_valid_ref_name("foo^", context="my-ctx")


# --- require_valid_rev --------------------------------------------------------


class TestRequireValidRev:
    def test_accepts_sha(self) -> None:
        sha = "abcd1234" * 5  # 40 chars
        assert require_valid_rev(sha, context="test") == sha

    def test_accepts_ref_name(self) -> None:
        assert require_valid_rev("origin/main", context="test") == "origin/main"

    def test_accepts_head(self) -> None:
        assert require_valid_rev("HEAD", context="test") == "HEAD"

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            require_valid_rev("", context="test")

    def test_rejects_flag_like(self) -> None:
        with pytest.raises(ValueError):
            require_valid_rev("--exec=foo", context="test")

    def test_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            require_valid_rev(None, context="test")  # type: ignore[arg-type]
