"""Stub-body detection for the AST-equivalence gate (issue #1607 rework).

Extracted from :mod:`charlie_work.ast_equivalence_gate` to keep that module
under the file-size ratchet cap (issue #1442).  These helpers classify an
``ast.dump`` string as a trivial stub body (``...``, ``pass``, or a bare
docstring) so :func:`charlie_work.ast_equivalence_gate.derive_moved_symbols`
can prefer a concrete implementation over a ``Protocol`` stub when both share
the same bare method name -- the false negative issue #1607 reports.

All three functions are pure string-processing utilities with no dependency on
the rest of the gate module, so the extraction is byte-identical (no logic
change, just a module split).
"""

from __future__ import annotations


def _mask_string_literals(s: str) -> str:
    """Return *s* with string-literal contents replaced by spaces.

    ``ast.dump`` renders every string value with ``repr()``, so a ``'`` or ``"``
    in the dump always opens a string-literal span (the repr of some
    ``Constant`` value, a node ``name``, an ``arg`` name, etc.). Brackets and
    commas inside such a span are part of the literal's *text*, not dump
    structure, so structural scans -- the ``body=[`` marker search, the
    closing-bracket depth scan, and the top-level comma scan -- must ignore
    them. Without this, a default-arg value or docstring containing the literal
    text ``body=[`` or an unmatched ``]`` desynchronizes the depth counter and
    silently reintroduces the false negative this gate exists to catch (issue
    #1607 rework).

    The returned string has the same length as *s*; only characters *inside*
    string spans are replaced with spaces (the opening/closing quote chars are
    preserved -- they are not brackets or commas, so they are inert for the
    structural scans). Backslash escapes inside the span are honored so an
    escaped quote (``\\'`` / ``\\"``) does not prematurely close the span, and
    the escape character itself is blanked.
    """
    out = list(s)
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "'" or ch == '"':
            quote = ch
            i += 1
            while i < n:
                c = s[i]
                if c == "\\":
                    # Blank the backslash and its escaped char (if any).
                    out[i] = " "
                    if i + 1 < n:
                        out[i + 1] = " "
                        i += 2
                    else:
                        i += 1
                    continue
                if c == quote:
                    i += 1  # close the span; leave the closing quote unmasked
                    break
                out[i] = " "
                i += 1
            continue
        i += 1
    return "".join(out)


def _has_top_level_comma(s: str) -> bool:
    """Return True if *s* has a comma at bracket depth 0 (a list-element separator).

    Tracks both ``()`` and ``[]`` nesting so a comma inside a call's argument
    list (e.g. ``Call(func=..., args=[...])``) is not mistaken for a
    top-level list separator. The scan runs over :func:`_mask_string_literals`
    output so a comma inside a string constant (a docstring, default-arg value,
    or annotation) cannot reach depth 0 -- without masking, a comma in a
    ``Constant(value='a,b')`` span at depth 0 of the *body* substring would be
    mistaken for a multi-statement separator.
    """
    masked = _mask_string_literals(s)
    depth = 0
    for ch in masked:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            return True
    return False


def _is_trivial_stub_dump(dump: str) -> bool:
    """Return True if an ``ast.dump`` string is a function/class with a stub body.

    A trivial stub body is exactly one statement that is ``...`` (Ellipsis),
    ``pass``, or a bare docstring -- the body of a ``Protocol`` method stub or
    abstract placeholder, never the body of a real moved method. When
    :func:`charlie_work.ast_equivalence_gate.derive_moved_symbols` has multiple
    same-bare-name candidates in one destination file (a ``Protocol`` stub and
    a concrete implementation added together, issue #1607), preferring the
    non-stub candidate avoids a false negative: the stub's dump never equals a
    real method body's dump, so first-in-file-order can report
    ``equivalent=False`` for a byte-identical move.

    Detection operates on the dump string: the first ``body=[`` in a
    ``FunctionDef``/``ClassDef`` dump is the node's own body list
    (``ast.arguments`` has no ``body=`` field), so we extract that list's
    balanced-bracket content and check it is a single trivial element. A
    nested function whose own body is trivial does not cause a false positive:
    it is not the first ``body=[``, and at the outer level it is one of
    several statements separated by a top-level comma (caught by
    :func:`_has_top_level_comma`).

    Both the ``body=[`` marker search and the closing-bracket depth scan run
    over :func:`_mask_string_literals` output, so a bracket character inside a
    string literal (a default-arg value or docstring containing the literal
    text ``body=[`` or an unmatched ``]``) cannot desynchronize the depth
    counter. The extracted *body* substring is taken from the original
    (unmasked) dump so the docstring-shape check still sees the real quote
    characters.
    """
    masked = _mask_string_literals(dump)
    marker = "body=["
    idx = masked.find(marker)
    if idx < 0:
        return False
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(masked) and depth > 0:
        ch = masked[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        i += 1
    if depth != 0:
        return False  # malformed / unbalanced brackets
    body = dump[start : i - 1]
    # A stub body has exactly one statement; a top-level comma separates
    # multiple statements.
    if _has_top_level_comma(body):
        return False
    if body == "Pass()":
        return True
    if body == "Expr(value=Constant(value=Ellipsis))":
        return True
    # A bare docstring: Expr(value=Constant(value='...')) or double-quoted.
    return (body.startswith("Expr(value=Constant(value='") and body.endswith("'))")) or (
        body.startswith('Expr(value=Constant(value="') and body.endswith('"))')
    )
