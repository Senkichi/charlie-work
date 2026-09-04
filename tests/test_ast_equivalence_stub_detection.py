"""Tests for stub-body detection in the AST-equivalence gate (issue #1607).

Extracted from :mod:`tests.test_ast_equivalence_gate` to keep that file under
the file-size ratchet cap (issue #1442).  Covers:

* :func:`charlie_work.ast_equivalence_stub_detection._is_trivial_stub_dump` --
  unit tests for the stub classifier (each stub shape, string-literal masking).
* :func:`charlie_work.ast_equivalence_gate.derive_moved_symbols` -- end-to-end
  tests for the non-stub preference when a ``Protocol`` stub and concrete
  implementation share the same bare method name (the false negative #1607
  reports).

Mutation controls: each regression test names the exact edit it reverts and
verifies the test fails against the unfixed code.
"""

from __future__ import annotations

import textwrap

from charlie_work.ast_equivalence_gate import (
    derive_moved_symbols,
    extract_symbols,
)
from charlie_work.ast_equivalence_stub_detection import _is_trivial_stub_dump


# ---------------------------------------------------------------------------
# derive_moved_symbols -- non-stub preference (issue #1607)
# ---------------------------------------------------------------------------


def test_derive_moved_prefers_concrete_impl_over_protocol_stub() -> None:
    """A Protocol stub and concrete impl added together: prefer the impl (issue #1607).

    When a destination file adds both a ``Protocol`` stub (body ``...``) and a
    concrete class with the real moved method body, both sharing the same bare
    method name, ``derive_moved_symbols`` must pick the concrete implementation
    as the move target -- not the Protocol stub, whose ``...`` body never
    equals a real method body's dump (a false negative on a byte-identical
    move).

    Positive control from issue #1607: a synthetic diff whose head adds
    ``class FooLike(Protocol)`` with a stub ``foo`` AND ``class Foo`` with the
    moved ``foo`` body, from a base that has neither (the L01 facade-leaf
    shape: Protocol and implementation introduced together).
    """
    base_source = textwrap.dedent("""
        class GitHub:
            def foo(self):
                return 42
    """)
    head_old_source = textwrap.dedent("""
        class GitHub:
            pass
    """)
    head_new_source = textwrap.dedent("""
        from typing import Protocol

        class FooLike(Protocol):
            def foo(self) -> int: ...

        class Foo:
            def foo(self):
                return 42
    """)
    base = {"src/github.py": extract_symbols(base_source, "github.py")}
    head = {
        "src/github.py": extract_symbols(head_old_source, "github.py"),
        "src/foo.py": extract_symbols(head_new_source, "foo.py"),
    }
    moved = derive_moved_symbols(base, head)
    foo_moves = [m for m in moved if m.name == "foo"]
    assert len(foo_moves) == 1
    # The concrete implementation (Foo) is the move target, not the Protocol
    # stub (FooLike) which is first in file order.
    assert foo_moves[0].dest_class == "Foo"
    assert foo_moves[0].dest_file == "src/foo.py"
    # The move is byte-identical (same body) -> equivalent, not a false negative.
    assert foo_moves[0].equivalent is True


def test_derive_moved_falls_back_to_stub_when_only_candidate() -> None:
    """When the only candidate is a stub, it is still used (issue #1607).

    The non-stub preference never drops a move: if the sole added candidate is
    a Protocol stub, it is reported as the destination (with whatever
    equivalence the dump comparison yields) rather than silently recording no
    move. This preserves the one-removal-one-move invariant.
    """
    base_source = textwrap.dedent("""
        class GitHub:
            def foo(self):
                return 42
    """)
    head_old_source = textwrap.dedent("""
        class GitHub:
            pass
    """)
    head_new_source = textwrap.dedent("""
        from typing import Protocol

        class FooLike(Protocol):
            def foo(self) -> int: ...
    """)
    base = {"src/github.py": extract_symbols(base_source, "github.py")}
    head = {
        "src/github.py": extract_symbols(head_old_source, "github.py"),
        "src/foo.py": extract_symbols(head_new_source, "foo.py"),
    }
    moved = derive_moved_symbols(base, head)
    foo_moves = [m for m in moved if m.name == "foo"]
    assert len(foo_moves) == 1
    assert foo_moves[0].dest_class == "FooLike"
    # Stub body (...) != real body -> non-equivalent (correctly flagged).
    assert foo_moves[0].equivalent is False


# ---------------------------------------------------------------------------
# _is_trivial_stub_dump -- unit tests for the stub classifier
# ---------------------------------------------------------------------------


def test_is_trivial_stub_dump_detects_each_stub_shape() -> None:
    """``_is_trivial_stub_dump`` recognizes ``...``, ``pass``, and docstring stubs.

    And it does NOT classify a real method (including one with a nested
    trivial-body function) as a stub -- the nested ``body=[Pass()]`` is not
    the first ``body=[`` in the dump, and the outer body has a top-level comma.
    """
    assert _is_trivial_stub_dump(
        extract_symbols("class C:\n    def m(self): ...\n", "t.py")["C.m"]
    )
    assert _is_trivial_stub_dump(
        extract_symbols("class C:\n    def m(self):\n        pass\n", "t.py")["C.m"]
    )
    assert _is_trivial_stub_dump(
        extract_symbols('class C:\n    def m(self):\n        """stub"""\n', "t.py")["C.m"]
    )
    # A real method is not a stub.
    assert not _is_trivial_stub_dump(
        extract_symbols("class C:\n    def m(self):\n        return 42\n", "t.py")["C.m"]
    )
    # A real method with a nested trivial-body function is not a stub: the
    # nested body=[Pass()] is not the first body=[, and the outer body has a
    # top-level comma separating the nested def from the return.
    nested = (
        "class C:\n    def m(self):\n        def inner():\n            pass\n        return 42\n"
    )
    assert not _is_trivial_stub_dump(extract_symbols(nested, "t.py")["C.m"])
    # A docstring followed by a real statement is not a stub (multi-statement).
    multi = 'class C:\n    def m(self):\n        """doc"""\n        return 42\n'
    assert not _is_trivial_stub_dump(extract_symbols(multi, "t.py")["C.m"])


def test_is_trivial_stub_dump_string_literal_brackets_do_not_desync() -> None:
    """Brackets inside string literals must not desynchronize the depth scan.

    Regression for the #1607 rework: ``_is_trivial_stub_dump``'s ``body=[``
    marker search and closing-bracket depth scan (and ``_has_top_level_comma``)
    operate on the raw dump string, so a bracket character inside a quoted
    ``Constant(value='...')`` span -- a default-arg value, docstring, or
    annotation -- could be mistaken for structural brackets. The confirmed
    false negative: a ``pass``-only stub with a default arg containing the
    literal text ``body=[`` was classified as non-stub because ``find("body=[")``
    matched the marker *inside* the string literal instead of the real body
    list, silently reintroducing the exact Protocol/impl false negative this
    gate exists to catch.

    Both directions are covered: a genuine stub still detected correctly, and a
    genuine multi-statement implementation not misclassified as a stub.
    """
    # Genuine stub: pass-only body, but a default arg whose repr contains the
    # literal marker text "body=[". Without string-literal masking, find()
    # matches the marker inside the string and the stub is misclassified.
    stub_marker = 'class C:\n    def m(self, x="body=["):\n        pass\n'
    assert _is_trivial_stub_dump(extract_symbols(stub_marker, "t.py")["C.m"]), (
        "pass-only stub with default arg 'body=[' must still be detected as a stub"
    )

    # Genuine stub: pass-only body, default arg whose repr contains an
    # unmatched closing bracket that would prematurely end the body-list scan.
    stub_unmatched = 'class C:\n    def m(self, x="has ] inside"):\n        pass\n'
    assert _is_trivial_stub_dump(extract_symbols(stub_unmatched, "t.py")["C.m"]), (
        "pass-only stub with default arg containing ']' must still be detected"
    )

    # Genuine stub: ... body, docstring-shaped default with an unmatched '['.
    stub_ellipsis = 'class C:\n    def m(self, x="open [ bracket"):\n        ...\n'
    assert _is_trivial_stub_dump(extract_symbols(stub_ellipsis, "t.py")["C.m"]), (
        "... stub with default arg containing '[' must still be detected"
    )

    # Genuine multi-statement implementation: a docstring whose text contains
    # an unmatched ']' followed by a real statement. Must NOT be classified as
    # a stub -- the body has two statements separated by a top-level comma.
    impl = 'class C:\n    def m(self):\n        """doc with ] bracket"""\n        return 42\n'
    assert not _is_trivial_stub_dump(extract_symbols(impl, "t.py")["C.m"]), (
        "docstring-with-bracket + return must not be misclassified as a stub"
    )

    # Genuine multi-statement implementation: a docstring containing the
    # literal "body=[" text followed by a real statement. The marker inside
    # the docstring must not be mistaken for the real body list.
    impl_marker = 'class C:\n    def m(self):\n        """see body=[ ..."""\n        return 42\n'
    assert not _is_trivial_stub_dump(extract_symbols(impl_marker, "t.py")["C.m"]), (
        "docstring containing 'body=[' + return must not be misclassified as a stub"
    )


def test_derive_moved_prefers_non_stub_with_bracket_in_default_arg() -> None:
    """End-to-end: the non-stub preference survives a bracket in a default arg.

    The #1607 rework's false negative is not just a unit-test artifact: when a
    destination adds a Protocol stub whose method has a default arg containing
    a bracket character, the stub-detection helper must still recognize it as a
    stub so the concrete implementation is preferred. Without string-literal
    masking the stub is misclassified as non-stub and chosen first, reproducing
    the byte-identical-move false negative in :func:`derive_moved_symbols`.
    """
    base_source = textwrap.dedent("""
        class GitHub:
            def foo(self):
                return 42
    """)
    head_old_source = textwrap.dedent("""
        class GitHub:
            pass
    """)
    # The Protocol stub's method has a default arg whose repr contains "body=[",
    # the exact literal that desynchronized the unmasked marker search.
    head_new_source = textwrap.dedent("""
        from typing import Protocol

        class FooLike(Protocol):
            def foo(self, x="body=[") -> int: ...

        class Foo:
            def foo(self):
                return 42
    """)
    base = {"src/github.py": extract_symbols(base_source, "github.py")}
    head = {
        "src/github.py": extract_symbols(head_old_source, "github.py"),
        "src/foo.py": extract_symbols(head_new_source, "foo.py"),
    }
    moved = derive_moved_symbols(base, head)
    foo_moves = [m for m in moved if m.name == "foo"]
    assert len(foo_moves) == 1
    # The concrete implementation (Foo) is the move target, not the Protocol
    # stub (FooLike) -- even though the stub's default arg contains "body=[".
    assert foo_moves[0].dest_class == "Foo"
    assert foo_moves[0].equivalent is True
