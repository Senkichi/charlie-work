"""Shared helpers for ``test_githublike_protocol*.py`` (issue #1284 sanctions
bare-name ``tests/_*.py`` modules for cross-test-file helper sharing, as
distinct from banned ``test_*.py`` -> ``test_*.py`` imports).

Hoisted out of ``test_githublike_protocol.py`` in Track 2 L06 (issue #1590)
so the new ``test_githublike_protocol_l06.py`` can reuse them without
importing from another ``test_*.py`` module. Bodies are unchanged from their
original location -- only this module boundary is new.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import charlie_work.github as _github_module


def _compatible_signature(proto_sig: inspect.Signature, concrete_sig: inspect.Signature) -> None:
    proto_params = [p for n, p in proto_sig.parameters.items() if n != "self"]
    concrete_params = [p for n, p in concrete_sig.parameters.items() if n != "self"]
    assert len(proto_params) == len(concrete_params), (
        f"parameter count differs: {proto_sig} vs {concrete_sig}"
    )
    for proto_param, concrete_param in zip(proto_params, concrete_params):
        assert proto_param.name == concrete_param.name, (
            f"parameter name differs for {proto_sig}: "
            f"{proto_param.name!r} vs {concrete_param.name!r}"
        )
        assert proto_param.kind == concrete_param.kind, (
            f"parameter kind differs for {proto_param.name}: "
            f"{proto_param.kind} vs {concrete_param.kind}"
        )
    if proto_sig.return_annotation is not inspect.Signature.empty:
        assert concrete_sig.return_annotation is not inspect.Signature.empty, (
            f"concrete missing return annotation for {proto_sig}"
        )
        assert proto_sig.return_annotation == concrete_sig.return_annotation, (
            f"return annotation differs: {proto_sig.return_annotation} "
            f"vs {concrete_sig.return_annotation}"
        )


def _lexical_github_defs() -> set[str]:
    """Names GitHub defines directly via ``def`` in source (AST-derived).

    Mirrors the member_count ratchet's own counting rule (only direct
    ``FunctionDef``/``AsyncFunctionDef`` children of the ``ClassDef`` body),
    so this reflects "lexically defined" independent of whatever
    ``_install_delegates()`` has since added via ``setattr`` -- a class-level
    assignment is not an AST ``FunctionDef`` node and would not show up here.
    """
    source = Path(_github_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GitHub":
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                # Dunder-shaped names (including hand-written ones like
                # __post_init__) are excluded from this comparison the same
                # way they are excluded from `actual` below -- both sides
                # must apply the identical filter or a real dataclass hook
                # like __post_init__ reads as spurious "drift".
                and not (child.name.startswith("__") and child.name.endswith("__"))
            }
    raise AssertionError("GitHub class definition not found in charlie_work/github.py source")
