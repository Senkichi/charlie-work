"""Turn output schema and action types (design §8.3, issue #3119).

Decide's only output is a JSON document — a ``turn_summary`` string and an
``actions`` list — and ``TURN_OUTPUT_SCHEMA`` is the contract it is prompted
against. ``parse_turn_output`` validates a decoded document and returns it as
frozen dataclasses. Anything that fails the schema fails the *whole* Turn
output as an error value: a Turn carrying one malformed action applies
nothing, because a resume reads the same document back.

The package carries no ``jsonschema`` dependency; ``validate_turn_output``
covers the subset of JSON Schema the document uses (``type``, ``const``,
``enum``, ``required``, ``properties``, ``additionalProperties``, ``items``,
``pattern``/``minLength``/``minimum``, ``minItems``/``maxItems``, ``oneOf``,
``allOf``, and local ``#/$defs/`` refs). Unknown keywords are ignored, so the
document can grow richer over time without the gate loosening.

``blocked_by`` is data, never prose: each entry is either ``key:<key>``
referencing another ``file_issue`` action in the same Turn, or ``#<number>``
naming an existing issue. Apply resolves keys to real numbers and renders the
``Blocked by`` section itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Union

from . import Result

#: ``key`` on a keyed action: the model-chosen stable identifier used by
#: ``blocked_by`` refs, issue markers and item ids.
KEY_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_-]*"

#: One ``blocked_by`` entry: ``key:<key>`` for a same-Turn ``file_issue``
#: dependency, or ``#<number>`` for an existing issue.
BLOCKED_BY_PATTERN = rf"^(?:key:{KEY_PATTERN}|#[0-9]+)$"

_KIND_PATTERN = r"decision|permission|report"

TURN_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Foreman turn output",
    "type": "object",
    "additionalProperties": False,
    "required": ["turn_summary", "actions"],
    "properties": {
        "turn_summary": {"type": "string", "minLength": 1},
        "actions": {"type": "array", "items": {"$ref": "#/$defs/action"}},
    },
    "$defs": {
        "action": {
            "type": "object",
            "required": ["type"],
            "oneOf": [
                {"$ref": "#/$defs/file_issue"},
                {"$ref": "#/$defs/amend_issue"},
                {"$ref": "#/$defs/withdraw_issue"},
                {"$ref": "#/$defs/comment_issue"},
                {"$ref": "#/$defs/raise_item"},
                {"$ref": "#/$defs/write_memory"},
                {"$ref": "#/$defs/delete_memory"},
                {"$ref": "#/$defs/propose_done"},
            ],
        },
        "file_issue": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "key", "repo", "title", "body"],
            "properties": {
                "type": {"const": "file_issue"},
                "key": {"type": "string", "pattern": f"^{KEY_PATTERN}$"},
                "repo": {"type": "string", "minLength": 1},
                "title": {"type": "string", "minLength": 1},
                "body": {"type": "string"},
                "blocked_by": {
                    "type": "array",
                    "items": {"type": "string", "pattern": BLOCKED_BY_PATTERN},
                },
            },
        },
        "amend_issue": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "repo", "number", "body"],
            "properties": {
                "type": {"const": "amend_issue"},
                "repo": {"type": "string", "minLength": 1},
                "number": {"type": "integer", "minimum": 1},
                "body": {"type": "string"},
            },
        },
        "withdraw_issue": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "repo", "number", "reason"],
            "properties": {
                "type": {"const": "withdraw_issue"},
                "repo": {"type": "string", "minLength": 1},
                "number": {"type": "integer", "minimum": 1},
                "reason": {"type": "string", "minLength": 1},
            },
        },
        "comment_issue": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "repo", "number", "body"],
            "properties": {
                "type": {"const": "comment_issue"},
                "repo": {"type": "string", "minLength": 1},
                "number": {"type": "integer", "minimum": 1},
                "body": {"type": "string", "minLength": 1},
            },
        },
        "raise_item": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "key", "kind", "question"],
            "properties": {
                "type": {"const": "raise_item"},
                "key": {"type": "string", "pattern": f"^{KEY_PATTERN}$"},
                "kind": {"type": "string", "enum": ["decision", "permission", "report"]},
                # For a ``report`` the question field carries the report body;
                # ``options``/``recommended`` apply to decision and permission
                # kinds and are validated by the item store in Apply.
                "question": {"type": "string", "minLength": 1},
                "options": {"type": "array", "items": {"type": "string", "minLength": 1}},
                "recommended": {"type": "integer", "minimum": 0},
                "command": {"type": "string", "minLength": 1},
            },
        },
        "write_memory": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "path", "content"],
            "properties": {
                "type": {"const": "write_memory"},
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
        },
        "delete_memory": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "path"],
            "properties": {
                "type": {"const": "delete_memory"},
                "path": {"type": "string", "minLength": 1},
            },
        },
        "propose_done": {
            "type": "object",
            "additionalProperties": False,
            "required": ["type", "evidence"],
            "properties": {
                "type": {"const": "propose_done"},
                "evidence": {"type": "string", "minLength": 1},
            },
        },
    },
}


@dataclass(frozen=True)
class FileIssueAction:
    """File a new issue; ``blocked_by`` entries are ``key:<key>`` or ``#N`` refs."""

    type: ClassVar[str] = "file_issue"
    key: str
    repo: str
    title: str
    body: str
    blocked_by: tuple[str, ...] = ()


@dataclass(frozen=True)
class AmendIssueAction:
    """Rewrite the body of an authored issue still matching its recorded hash."""

    type: ClassVar[str] = "amend_issue"
    repo: str
    number: int
    body: str


@dataclass(frozen=True)
class WithdrawIssueAction:
    """Unarm and close an authored issue, leaving a comment with the reason."""

    type: ClassVar[str] = "withdraw_issue"
    repo: str
    number: int
    reason: str


@dataclass(frozen=True)
class CommentIssueAction:
    """Comment on any issue carrying this foreman's ownership label."""

    type: ClassVar[str] = "comment_issue"
    repo: str
    number: int
    body: str


@dataclass(frozen=True)
class RaiseItemAction:
    """Raise a decision, permission or report item for the operator."""

    type: ClassVar[str] = "raise_item"
    key: str
    kind: str
    question: str
    options: tuple[str, ...] = ()
    recommended: int | None = None
    command: str | None = None


@dataclass(frozen=True)
class WriteMemoryAction:
    """Write one file under ``memory/`` (bounds enforced by the memory store)."""

    type: ClassVar[str] = "write_memory"
    path: str
    content: str


@dataclass(frozen=True)
class DeleteMemoryAction:
    """Delete one file under ``memory/`` (an absent file is a clean no-op)."""

    type: ClassVar[str] = "delete_memory"
    path: str


@dataclass(frozen=True)
class ProposeDoneAction:
    """Ask the operator to confirm the charter outcome is met."""

    type: ClassVar[str] = "propose_done"
    evidence: str


Action = Union[
    FileIssueAction,
    AmendIssueAction,
    WithdrawIssueAction,
    CommentIssueAction,
    RaiseItemAction,
    WriteMemoryAction,
    DeleteMemoryAction,
    ProposeDoneAction,
]

ACTION_TYPES: frozenset[str] = frozenset(
    {
        FileIssueAction.type,
        AmendIssueAction.type,
        WithdrawIssueAction.type,
        CommentIssueAction.type,
        RaiseItemAction.type,
        WriteMemoryAction.type,
        DeleteMemoryAction.type,
        ProposeDoneAction.type,
    }
)


@dataclass(frozen=True)
class TurnOutput:
    """One Turn's decided output: the summary plus the action list."""

    turn_summary: str
    actions: tuple[Action, ...]


# ---------------------------------------------------------------------------
# Minimal JSON Schema validator
# ---------------------------------------------------------------------------

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    # ``bool`` is an ``int`` subclass; JSON's integer/number exclude it.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a local ``#/$defs/<name>`` ref; anything else is unsupported."""
    match = re.fullmatch(r"#/\$defs/([A-Za-z0-9_-]+)", ref)
    if not match:
        return None
    target = root.get("$defs", {}).get(match.group(1))
    return target if isinstance(target, dict) else None


def _validate(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> list[str]:
    """One human-readable problem per violation; empty means valid."""
    ref = schema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_ref(ref, root)
        if target is None:
            return [f"{path}: unsupported $ref {ref!r} in the turn output schema"]
        return _validate(instance, target, root, path)

    problems: list[str] = []

    expected_type = schema.get("type")
    if expected_type is not None:
        names = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        if not any(_TYPE_CHECKS[name](instance) for name in names):
            return [f"{path}: expected type {expected_type!r}"]

    if "const" in schema and instance != schema["const"]:
        problems.append(f"{path}: must be {schema['const']!r}")
        return problems

    if "enum" in schema and instance not in schema["enum"]:
        problems.append(f"{path}: must be one of {schema['enum']!r}")
        return problems

    if isinstance(instance, dict):
        for name in schema.get("required", []):
            if name not in instance:
                problems.append(f"{path}.{name}: required property missing")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                problems.extend(_validate(value, properties[key], root, f"{path}.{key}"))
            elif additional is False:
                problems.append(f"{path}.{key}: additional property not allowed")
            elif isinstance(additional, dict):
                problems.extend(_validate(value, additional, root, f"{path}.{key}"))

    if isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, value in enumerate(instance):
                problems.extend(_validate(value, items, root, f"{path}[{i}]"))
        if "minItems" in schema and len(instance) < schema["minItems"]:
            problems.append(f"{path}: at least {schema['minItems']} item(s) required")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            problems.append(f"{path}: at most {schema['maxItems']} item(s) allowed")

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            problems.append(f"{path}: at least {schema['minLength']} character(s) required")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            problems.append(f"{path}: at most {schema['maxLength']} character(s) allowed")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            problems.append(f"{path}: must match pattern {schema['pattern']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            problems.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            problems.append(f"{path}: must be <= {schema['maximum']}")

    for i, sub in enumerate(schema.get("allOf", [])):
        problems.extend(_validate(instance, sub, root, path))

    any_of = schema.get("anyOf")
    if any_of is not None:
        if not any(not _validate(instance, sub, root, path) for sub in any_of):
            problems.append(f"{path}: must match at least one allowed shape")

    one_of = schema.get("oneOf")
    if one_of is not None:
        branch_problems = [_validate(instance, sub, root, path) for sub in one_of]
        clean = [i for i, ps in enumerate(branch_problems) if not ps]
        if len(clean) > 1:
            problems.append(f"{path}: matches more than one allowed action shape")
        elif not clean:
            # Discriminator heuristic: when exactly one branch declares a
            # ``type`` const equal to the instance's, report that branch's
            # problems — the model wrote a known action type with a bad body,
            # which is a far more useful error than "matches none of 8".
            chosen: list[str] | None = None
            discriminator = instance.get("type") if isinstance(instance, dict) else None
            matching = [
                i
                for i, sub in enumerate(one_of)
                if isinstance(sub, dict)
                and _branch_type_const(sub, root) == discriminator
                and discriminator is not None
            ]
            if len(matching) == 1:
                chosen = branch_problems[matching[0]]
            problems.extend(chosen or [f"{path}: does not match any allowed action shape"])

    return problems


def _branch_type_const(subschema: dict[str, Any], root: dict[str, Any]) -> Any:
    """The ``properties.type.const`` a (possibly ``$ref``) branch declares."""
    target = subschema
    ref = subschema.get("$ref")
    if isinstance(ref, str):
        target = _resolve_ref(ref, root) or {}
    properties = target.get("properties", {})
    type_schema = properties.get("type", {})
    return type_schema.get("const")


def validate_turn_output(document: Any) -> tuple[str, ...]:
    """Validate a decoded Turn output document; returns the problems."""
    return tuple(_validate(document, TURN_OUTPUT_SCHEMA, TURN_OUTPUT_SCHEMA, "$"))


_BUILDERS = {
    "file_issue": lambda a: FileIssueAction(
        key=a["key"],
        repo=a["repo"],
        title=a["title"],
        body=a["body"],
        blocked_by=tuple(a.get("blocked_by", ())),
    ),
    "amend_issue": lambda a: AmendIssueAction(repo=a["repo"], number=a["number"], body=a["body"]),
    "withdraw_issue": lambda a: WithdrawIssueAction(
        repo=a["repo"], number=a["number"], reason=a["reason"]
    ),
    "comment_issue": lambda a: CommentIssueAction(
        repo=a["repo"], number=a["number"], body=a["body"]
    ),
    "raise_item": lambda a: RaiseItemAction(
        key=a["key"],
        kind=a["kind"],
        question=a["question"],
        options=tuple(a.get("options", ())),
        recommended=a.get("recommended"),
        command=a.get("command"),
    ),
    "write_memory": lambda a: WriteMemoryAction(path=a["path"], content=a["content"]),
    "delete_memory": lambda a: DeleteMemoryAction(path=a["path"]),
    "propose_done": lambda a: ProposeDoneAction(evidence=a["evidence"]),
}


def parse_turn_output(document: Any) -> Result[TurnOutput]:
    """Validate *document* against the schema and return it as dataclasses.

    A schema failure is one error value naming every offending path — the
    whole Turn output fails, never a single action.
    """
    problems = validate_turn_output(document)
    if problems:
        return Result(error="turn output failed schema: " + "; ".join(problems))
    actions = tuple(_BUILDERS[a["type"]](a) for a in document["actions"])
    return Result(value=TurnOutput(turn_summary=document["turn_summary"], actions=actions))


def load_turn_output(path: Path) -> Result[TurnOutput]:
    """Read ``path`` (a caller-written ``actions.json``) and parse it."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Result(error=f"{path.name}: not found at {path}")
    except OSError as exc:
        return Result(error=f"{path.name}: unreadable: {exc}")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        return Result(error=f"{path.name}: invalid JSON: {exc}")
    return parse_turn_output(document)
