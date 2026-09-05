"""Direct unit tests for ``CapabilityCollaborator.__getattr__`` and the
``github_delegation._build_routes`` collision path (issue #1585, PR #1596
round-1 review).

Kept in its own module rather than folded into ``test_githublike_protocol.py``
so that file's ``test_module`` attachment point stays within its baselined
member-count ceiling (``.attachment-budgets.json``).
"""

from __future__ import annotations

from typing import Any

import pytest

import charlie_work.github_delegation as _github_delegation
from charlie_work.github_capabilities import Comments
from charlie_work.github_capabilities._base import CapabilityCollaborator


class _RecordingOwner:
    """A plain owner stub whose every attribute access is logged.

    Used to prove a code path never touches the owner at all (rather than
    merely asserting the outcome), by asserting the log stays empty.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "accessed", [])

    def __getattribute__(self, name: str) -> Any:
        if name != "accessed":
            object.__getattribute__(self, "accessed").append(name)
        return object.__getattribute__(self, name)


class _StubOwner:
    """A minimal owner with one method and one plain data attribute."""

    def __init__(self) -> None:
        self.some_attr = "owner-value"

    def run(self) -> str:
        return "ran"


def test_dunder_lookup_raises_without_forwarding_to_owner() -> None:
    """A dunder-shaped name must raise AttributeError without ever touching
    the owner (design doc Section 3.3: dunder probes are guarded before the
    owner lookup runs at all, not merely before returning something useful).
    """
    owner = _RecordingOwner()
    collab = CapabilityCollaborator(owner)

    with pytest.raises(AttributeError):
        getattr(collab, "__missing_dunder__")

    assert owner.accessed == [], (
        f"dunder lookup must never reach the owner; recorded accesses: {owner.accessed}"
    )


def test_missing_owner_raises_clean_attribute_error_for_arbitrary_name() -> None:
    """An instance constructed without ``__init__`` (no ``_owner`` in
    ``__dict__``) must raise a clean AttributeError naming the requested
    attribute -- not RecursionError from re-entering ``__getattr__`` while
    trying to resolve ``_owner`` itself.
    """
    collab = object.__new__(Comments)

    with pytest.raises(AttributeError) as exc_info:
        collab.some_missing_attr

    assert "some_missing_attr" in str(exc_info.value)


def test_missing_owner_raises_clean_attribute_error_for_owner_itself() -> None:
    """Accessing ``_owner`` directly on an instance where it was never set
    must also raise a clean, named AttributeError (not recurse).
    """
    collab = object.__new__(Comments)

    with pytest.raises(AttributeError) as exc_info:
        collab._owner

    assert "_owner" in str(exc_info.value)


def test_forwards_method_and_data_attribute_to_owner() -> None:
    """A name absent on the collaborator resolves through to the owner, for
    both a callable and a plain data attribute.
    """
    owner = _StubOwner()
    collab = CapabilityCollaborator(owner)

    assert collab.run() == "ran"
    assert collab.some_attr == "owner-value"


def test_missing_on_both_forwards_and_raises_owners_attribute_error() -> None:
    """A name absent on both the collaborator and the owner still resolves
    via a plain ``getattr(owner, name)`` forward, so the AttributeError that
    surfaces is the owner's own -- unchanged, naming the owner's type.

    (This is a distinct path from the no-owner-at-all case above, where
    ``CapabilityCollaborator`` raises its own AttributeError naming *its own*
    type because there is no owner to forward to. Here an owner exists; the
    forward simply fails on the owner's side, and nothing wraps or rewrites
    that failure.)
    """
    owner = _StubOwner()
    collab = CapabilityCollaborator(owner)

    with pytest.raises(AttributeError) as exc_info:
        collab.totally_missing_name

    message = str(exc_info.value)
    assert "totally_missing_name" in message
    assert type(owner).__name__ in message


def test_build_routes_raises_valueerror_on_name_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two collaborator classes declaring the same method name must fail
    ``_build_routes`` with a ``ValueError`` naming the colliding member --
    a silent last-writer-wins would otherwise drop one route outright.

    Uses throwaway subclasses (not the real ``Comments``/``Labels``/...
    classes, which are still empty in L01) monkeypatched onto the module's
    ``_COLLABORATORS`` registry, since ``_build_routes`` takes no parameters
    of its own.
    """

    class _CollabOne(CapabilityCollaborator):
        def dup(self) -> None: ...

    class _CollabTwo(CapabilityCollaborator):
        def dup(self) -> None: ...

    monkeypatch.setattr(
        _github_delegation,
        "_COLLABORATORS",
        (("_one", _CollabOne), ("_two", _CollabTwo)),
    )

    with pytest.raises(ValueError) as exc_info:
        _github_delegation._build_routes()

    assert "dup" in str(exc_info.value)


def test_build_routes_positive_control_disjoint_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two collaborator classes with disjoint method names must produce the
    expected routing map instead of raising -- the positive control proving
    the collision test above isn't merely detecting a broken registry.
    """

    class _CollabFoo(CapabilityCollaborator):
        def foo(self) -> None: ...

    class _CollabBar(CapabilityCollaborator):
        def bar(self) -> None: ...

    monkeypatch.setattr(
        _github_delegation,
        "_COLLABORATORS",
        (("_foo", _CollabFoo), ("_bar", _CollabBar)),
    )

    routes, signature_source = _github_delegation._build_routes()

    assert routes == {"foo": "_foo", "bar": "_bar"}
    assert set(signature_source) == {"foo", "bar"}
