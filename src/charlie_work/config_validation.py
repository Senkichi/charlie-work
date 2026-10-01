"""Config sections validate themselves from field metadata (ADR-0007).

A section declares its type and range rules as ``typing.Annotated`` metadata on
the fields of its existing frozen dataclass (``Annotated[int, NonNeg]``).
:func:`validate_section` is the single generic validator that reads that
metadata, coerces structurally, and constructs the section. Rules that span
fields or sections live in a per-section ``validate(self, config)`` hook that
:func:`run_section_hooks` calls once the whole ``OrchestratorConfig`` exists.

Every failure reads ``<dotted.key.path>: expected <rule>, got <value>``.

Design notes that are easy to get wrong:

* A field with NO rule-bearing marker is not checked at all (its current
  laxness is preserved); any rule-bearing marker turns on the base-type check.
* ``key: null`` is preserved as ``None`` (design P2): no marker runs on it and the
  default does NOT apply, because a falsy ``None`` differs from a default ``True``.
  ``NullIsDefault`` opts a key into the default instead (where main normalized it),
  ``NotNull`` rejects it, and a required key still reports ``missing``.
* ``bool`` is an ``int`` in Python; ``int``/``number`` fields reject it unless
  the field carries ``BoolTolerant`` (a legacy tolerance, preserved per key).
* ``ConfigError`` lives here so section modules can raise it without the
  ``config.py`` circular-import workaround; ``config`` re-exports it.
* No state writes or event emission happen here (write-gate ratchet n/a).
"""

from __future__ import annotations

import functools
import math
import re
import types
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeVar

GotKind = Literal["value", "type", "raw"]
T = TypeVar("T")
_MAX_REPR = 80


class ConfigError(ValueError):
    """A config file was structurally invalid (unknown keys, wrong shapes)."""


class ConstructionError(ConfigError):
    """A section dataclass's own constructor raised a plain ``ValueError``/``TypeError``.

    Callers that catch ``ConfigError`` handle it (design F1), but the #665 layered-load
    rescue must NOT: such errors come from host-wide sections (``ci_fleet``'s
    ``__post_init__``) and ``rescue.worker``, which live in the very global layer that
    rescue discards. Before the wrap they escaped the rescue as raw ``ValueError``."""


def _trunc(text: str) -> str:
    return text if len(text) <= _MAX_REPR else text[: _MAX_REPR - 3] + "..."


def render_got(value: object, kind: GotKind = "value") -> str:
    """Render the ``got`` clause; ``repr`` output is capped at 80 characters."""
    if kind == "raw":
        return str(value)
    if kind == "type":
        return f"{_trunc(repr(value))} ({type(value).__name__})"
    return _trunc(repr(value))


class FieldError(ConfigError):
    """One field violated a rule: ``<key>: expected <expected>, got <got>``."""

    def __init__(self, key: str, expected: str, got: object, got_kind: GotKind = "value") -> None:
        self.key, self.expected, self.got, self.got_kind = key, expected, got, got_kind
        super().__init__(f"{key}: expected {expected}, got {render_got(got, got_kind)}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (self.key, self.expected, self.got, self.got_kind))

    def with_prefix(self, prefix: str) -> FieldError:
        """Copy with ``prefix`` prepended to the key path (``[i]`` joins without a dot)."""
        if not prefix:
            return self
        sep = "" if self.key.startswith("[") or not self.key else "."
        return FieldError(f"{prefix}{sep}{self.key}", self.expected, self.got, self.got_kind)


# --------------------------------------------------------------------- markers
class Marker:
    """Base of every constraint marker. ``kinds`` = base kinds it may annotate."""

    kinds: ClassVar[frozenset[str] | None] = None  # None = any supported kind
    checks_type: ClassVar[bool] = True  # does its presence turn on the base-type check?

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:  # noqa: ARG002
        """``(expected, got, got_kind)`` if ``value`` violates the rule, else None."""
        return None


_NUM = frozenset({"int", "number"})
_STR = frozenset({"str"})


@dataclass(frozen=True)
class _Typed(Marker):
    pass


@dataclass(frozen=True)
class Ge(Marker):
    bound: float
    kinds: ClassVar[frozenset[str] | None] = _NUM

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        # Violation is ``value < bound`` (not ``not value >= bound``): NaN passes, as it
        # always has; pair with ``Finite`` where NaN must be rejected.
        return (f">= {self.bound}", value, "value") if value < self.bound else None


@dataclass(frozen=True)
class Gt(Marker):
    bound: float
    kinds: ClassVar[frozenset[str] | None] = _NUM

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        return (f"> {self.bound}", value, "value") if value <= self.bound else None


@dataclass(frozen=True)
class InRange(Marker):
    lo: float
    hi: float
    kinds: ClassVar[frozenset[str] | None] = _NUM

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        ok = self.lo <= value <= self.hi
        return None if ok else (f"in [{self.lo}, {self.hi}]", value, "value")


@dataclass(frozen=True)
class _Finite(Marker):
    kinds: ClassVar[frozenset[str] | None] = _NUM

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        return None if math.isfinite(value) else ("finite number", value, "value")


@dataclass(frozen=True)
class _NonEmpty(Marker):
    kinds: ClassVar[frozenset[str] | None] = _STR

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        return None if value.strip() else ("non-empty string", value, "value")


@dataclass(frozen=True, init=False)
class OneOf(Marker):
    choices: tuple[Any, ...]
    kinds: ClassVar[frozenset[str] | None] = frozenset({"int", "number", "str"})

    def __init__(self, *choices: Any) -> None:
        object.__setattr__(self, "choices", tuple(choices))

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        if value in self.choices:
            return None
        return ("one of " + ", ".join(repr(c) for c in self.choices), value, "value")


@dataclass(frozen=True, init=False)
class Placeholders(Marker):
    """``{name}`` placeholders in a command string must come from ``allowed``."""

    allowed: frozenset[str]
    kinds: ClassVar[frozenset[str] | None] = _STR
    _PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"\{([^{}]*)\}")

    def __init__(self, allowed: Iterable[str]) -> None:
        object.__setattr__(self, "allowed", frozenset(allowed))

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        names = sorted(self.allowed)
        expected = "placeholders from " + ", ".join(f"{{{n}}}" for n in names)
        if not names:
            expected = "no placeholders"
        for name in self._PATTERN.findall(value):
            if name == "":
                return (expected, "empty placeholder {}", "raw")
            if name not in self.allowed:
                return (expected, f"unknown placeholder {{{name}}}", "raw")
        try:  # the regex misses bare '{', unclosed '{x', stray '}', positional '{0}'
            value.format(**{n: "" for n in self.allowed})
        except (ValueError, KeyError, IndexError) as exc:
            return (expected, f"malformed placeholder in {value!r} ({exc})", "raw")
        return None


@dataclass(frozen=True)
class _NonEmptyRaw(Marker):
    """Legacy: rejects only ``""``; a whitespace-only string is accepted (preserved per key)."""

    kinds: ClassVar[frozenset[str] | None] = _STR

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        return None if value != "" else ("non-empty string", value, "value")


@dataclass(frozen=True)
class _Regex(Marker):
    kinds: ClassVar[frozenset[str] | None] = _STR

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        try:
            re.compile(value)
        except re.error as exc:
            return ("valid regex", f"{_trunc(repr(value))} ({exc})", "raw")
        return None


@dataclass(frozen=True)
class _RelativePath(Marker):
    kinds: ClassVar[frozenset[str] | None] = _STR

    def problem(self, value: Any) -> tuple[str, object, GotKind] | None:
        # Legacy semantics kept exactly: a drive-relative ``C:foo`` is NOT absolute.
        posix = PurePosixPath(value.replace("\\", "/"))
        if posix.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in posix.parts:
            return ("relative path without '..'", value, "value")
        return None


@dataclass(frozen=True)
class _BoolTolerant(Marker):
    """Legacy: an int/number field also accepts True/False (preserved per key)."""

    kinds: ClassVar[frozenset[str] | None] = _NUM


@dataclass(frozen=True)
class _LenientMapping(Marker):
    """Legacy: a non-mapping value for a nested-section field is treated as ``{}``."""

    kinds: ClassVar[frozenset[str] | None] = frozenset({"nested"})
    checks_type: ClassVar[bool] = False


@dataclass(frozen=True)
class _NotNull(Marker):
    """Legacy: ``key: null`` is rejected (the default is NOT substituted) for this field."""

    checks_type: ClassVar[bool] = False


@dataclass(frozen=True)
class _NullIsDefault(Marker):
    """Legacy: ``key: null`` builds the field's default (base normalized it, e.g. ``or ""``).

    Without this marker an explicit null is preserved as ``None`` (design P2)."""

    checks_type: ClassVar[bool] = False


@dataclass(frozen=True)
class _Coerced(Marker):
    """Legacy: a list has every element ``str()``-coerced with no element check.

    ``strict`` additionally requires a list; lenient lets any other value through
    unchanged (``notify.shell_command``)."""

    strict: bool = True
    kinds: ClassVar[frozenset[str] | None] = frozenset({"str"})  # element kind of seq_str


@dataclass(frozen=True, init=False)
class CommandTemplate(Marker):
    """Legacy command-template field: a list is ``str()``-coerced to a tuple; a truthy value
    must then be a string or tuple whose parts only use ``allowed`` placeholders.  A falsy
    value (``""``, ``0``, ``{}``) passes through untouched (preserved per key)."""

    allowed: frozenset[str]
    checks_type: ClassVar[bool] = False

    def __init__(self, allowed: Iterable[str]) -> None:
        object.__setattr__(self, "allowed", frozenset(allowed))

    def apply(self, value: Any, path: str) -> Any:
        if isinstance(value, list):
            value = tuple(str(elem) for elem in value)
        if not value:
            return value
        if not isinstance(value, (str, tuple)):
            raise FieldError(path, "string or list of strings", value, "type")
        rule = Placeholders(self.allowed)
        parts = (value,) if isinstance(value, str) else value
        for i, part in enumerate(parts):
            found = rule.problem(part)
            if found is not None:
                raise FieldError(path if isinstance(value, str) else f"{path}[{i}]", *found)
        return value


@dataclass(frozen=True)
class Note(Marker):
    """Parenthetical appended to the ``expected`` clause of any failure on the field."""

    text: str
    checks_type: ClassVar[bool] = False


@dataclass(frozen=True, init=False)
class FieldRules(Marker):
    """Use-site markers for fields of a nested/section class (incl. classes we cannot edit)."""

    items: tuple[tuple[str, tuple[Marker, ...]], ...]
    kinds: ClassVar[frozenset[str] | None] = frozenset({"nested"})
    checks_type: ClassVar[bool] = False

    def __init__(self, **rules: Marker | tuple[Marker, ...]) -> None:
        pairs = ((k, v if isinstance(v, tuple) else (v,)) for k, v in rules.items())
        object.__setattr__(self, "items", tuple(pairs))

    def as_mapping(self) -> dict[str, tuple[Marker, ...]]:
        return dict(self.items)


@dataclass(frozen=True, init=False)
class Entries(Marker):
    """Use-site rules for a ``tuple[SomeDataclass, ...]`` field: a length cap, per-element
    ``FieldRules``-style markers, and element keys the class declares but this site forbids
    (reported exactly like an unknown key)."""

    max_len: int | None
    forbid: tuple[str, ...]
    items: tuple[tuple[str, tuple[Marker, ...]], ...]
    kinds: ClassVar[frozenset[str] | None] = frozenset({"seq_dc"})
    checks_type: ClassVar[bool] = False

    def __init__(
        self,
        *,
        max_len: int | None = None,
        forbid: Iterable[str] = (),
        **rules: Marker | tuple[Marker, ...],
    ) -> None:
        pairs = tuple((k, v if isinstance(v, tuple) else (v,)) for k, v in rules.items())
        object.__setattr__(self, "max_len", max_len)
        object.__setattr__(self, "forbid", tuple(forbid))
        object.__setattr__(self, "items", pairs)


@dataclass(frozen=True)
class _Verbatim(Marker):
    """Legacy: a sequence-of-sections field keeps the value exactly as written, unparsed."""

    kinds: ClassVar[frozenset[str] | None] = frozenset({"seq_dc"})
    checks_type: ClassVar[bool] = False


@dataclass(frozen=True)
class _HostWideOnly(Marker):
    """The section may not appear in a per-repo layer (declared on the root field)."""

    checks_type: ClassVar[bool] = False


@dataclass(frozen=True)
class Check(Marker):
    """Section hook ``fn(section_value, config)`` for a class we cannot edit (ci_fleet)."""

    fn: Callable[[Any, Any], None]
    checks_type: ClassVar[bool] = False


Typed = _Typed()
NonNeg = Ge(0)
AtLeastOne = Ge(1)
Positive = Gt(0)
Finite = _Finite()
NonEmpty = _NonEmpty()
NonEmptyRaw = _NonEmptyRaw()
Regex = _Regex()
RelativePath = _RelativePath()
BoolTolerant = _BoolTolerant()
LenientMapping = _LenientMapping()
NotNull = _NotNull()
NullIsDefault = _NullIsDefault()
Coerced = _Coerced()
CoercedLenient = _Coerced(strict=False)
HostWideOnly = _HostWideOnly()
Verbatim = _Verbatim()


# ----------------------------------------------------------- annotation parsing
@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # int number bool str union seq_str seq_dc map_str map_dc nested opaque
    item: type | None  # nested / element dataclass
    scalars: tuple[type, ...]  # union members
    markers: tuple[Marker, ...]
    required: bool
    mapping_factory: str  # "dict" or "proxy" for map kinds


_SCALARS: dict[Any, str] = {int: "int", float: "number", bool: "bool", str: "str"}
_SCALAR_NAMES = {str: "string", bool: "bool", int: "int", float: "number"}


def _markers_of(hint: Any) -> tuple[Any, tuple[Marker, ...]]:
    if typing.get_origin(hint) is typing.Annotated:
        base, *meta = typing.get_args(hint)
        return base, tuple(m for m in meta if isinstance(m, Marker))
    return hint, ()


def _is_dc(tp: Any) -> bool:
    return isinstance(tp, type) and is_dataclass(tp)


def _classify(tp: Any) -> tuple[str, type | None, tuple[type, ...], str]:
    tp, _ = _markers_of(tp)
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        members = tuple(a for a in args if a is not type(None))
        if len(members) == 1:
            return _classify(members[0])
        if all(m in _SCALARS for m in members):
            return "union", None, members, "dict"
        return "opaque", None, (), "dict"
    if tp in _SCALARS:
        return _SCALARS[tp], None, (), "dict"
    if _is_dc(tp):
        return "nested", tp, (), "dict"
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        if args[0] is str:
            return "seq_str", None, (), "dict"
        if _is_dc(args[0]):
            return "seq_dc", args[0], (), "dict"
    if origin in (dict, Mapping) and len(args) == 2 and args[0] is str:
        factory = "dict" if origin is dict else "proxy"
        if args[1] is str:
            return "map_str", None, (), factory
        if _is_dc(args[1]):
            return "map_dc", args[1], (), factory
    return "opaque", None, (), "dict"


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    """Resolved annotations incl. ``Annotated`` extras (resolution is slow; classes are static)."""
    return typing.get_type_hints(cls, include_extras=True)


@functools.cache
def field_specs(cls: type) -> tuple[FieldSpec, ...]:
    """Per-class field specs, in dataclass order (``init=False`` fields excluded)."""
    hints = _hints(cls)
    specs = []
    for f in fields(cls):
        if not f.init:
            continue
        kind, item, scalars, factory = _classify(hints[f.name])
        _, markers = _markers_of(hints[f.name])
        required = f.default is MISSING and f.default_factory is MISSING
        specs.append(FieldSpec(f.name, kind, item, scalars, markers, required, factory))
    return tuple(specs)


def field_rules(cls: type, name: str) -> dict[str, tuple[Marker, ...]]:
    """Use-site ``FieldRules`` declared on ``cls.name`` (for the nested section's fields)."""
    return _rules_of(field_markers(cls, name))


def field_markers(cls: type, name: str) -> tuple[Marker, ...]:
    """Annotation markers on one field (e.g. ``HostWideOnly`` on an OrchestratorConfig field)."""
    hint = _hints(cls)[name]
    return _markers_of(hint)[1]


def host_wide_sections(root: type | None = None) -> frozenset[str]:
    """Names of root-config fields marked ``HostWideOnly`` (derived, never hand-listed)."""
    if root is None:
        from .config import OrchestratorConfig as root  # noqa: N813  (lazy: avoids a cycle)
    return frozenset(
        f.name
        for f in fields(root)
        if not f.metadata.get("provenance") and HostWideOnly in field_markers(root, f.name)
    )


def unknown_sections_error(unknown: Iterable[str], valid: Iterable[str]) -> FieldError:
    return FieldError(
        "config",
        f"known sections (valid: {', '.join(sorted(valid))})",
        f"unknown config section(s) {', '.join(sorted(unknown))}",
        "raw",
    )


def host_wide_error(section: str, *, fleet_dir: object, repo_path: object) -> FieldError:
    return FieldError(
        section,
        f"host-wide only (declare it in {fleet_dir}/config.yaml)",
        f"per-repo config {repo_path}",
        "raw",
    )


# ------------------------------------------------------------------ validation
_NO_RULES: Mapping[str, tuple[Marker, ...]] = MappingProxyType({})


def _element_kind(kind: str) -> str:
    return "str" if kind in ("seq_str", "map_str") else kind


def _check_compat(spec: FieldSpec, markers: tuple[Marker, ...], cls: type) -> None:
    """Programming errors (a marker on an unsupported annotation) raise TypeError."""
    where = f"{cls.__name__}.{spec.name}"
    if spec.kind == "opaque" and any(m.checks_type for m in markers):
        raise TypeError(f"{where}: markers on an unsupported annotation")
    for m in markers:
        allowed = m.kinds
        if allowed is not None and _element_kind(spec.kind) not in allowed:
            raise TypeError(f"{where}: {type(m).__name__} cannot annotate a {spec.kind} field")


def _scalar_ok(kind: str, value: Any, tolerant: bool, spec: FieldSpec) -> bool:
    is_bool = isinstance(value, bool)
    if kind == "int":
        return isinstance(value, int) and (tolerant or not is_bool)
    if kind == "number":
        return isinstance(value, (int, float)) and (tolerant or not is_bool)
    if kind == "bool":
        return is_bool
    if kind == "str":
        return isinstance(value, str)
    return isinstance(value, spec.scalars)  # union


def _scalar_expected(spec: FieldSpec, kind: str) -> str:
    if kind == "union":
        return " or ".join(_SCALAR_NAMES[s] for s in spec.scalars)
    return {"int": "int", "number": "number", "bool": "bool", "str": "string"}[kind]


def _value_markers(path: str, value: Any, markers: tuple[Marker, ...]) -> None:
    for m in markers:
        found = m.problem(value)
        if found is not None:
            raise FieldError(path, *found)


def _is_seq(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _rules_of(markers: tuple[Marker, ...]) -> dict[str, tuple[Marker, ...]]:
    merged: dict[str, tuple[Marker, ...]] = {}
    for m in markers:
        if isinstance(m, FieldRules):
            for key, ms in m.items:
                merged[key] = merged.get(key, ()) + ms
    return merged


def _entry(spec: FieldSpec, entries: Entries | None, elem: Mapping[str, Any], path: str) -> Any:
    rules: Mapping[str, tuple[Marker, ...]] = _NO_RULES
    if entries is not None:
        if entries.forbid:
            valid = sorted(s.name for s in field_specs(spec.item) if s.name not in entries.forbid)  # type: ignore[arg-type]
            barred = sorted(str(k) for k in elem if k not in valid)
        else:
            valid, barred = [], []
        if barred:
            raise FieldError(
                path,
                f"known keys (valid: {', '.join(valid)})",
                f"unknown key(s) {', '.join(barred)}",
                "raw",
            )
        rules = dict(entries.items)
    return validate_section(spec.item, elem, path=path, rules=rules)  # type: ignore[arg-type]


def _coerce(spec: FieldSpec, markers: tuple[Marker, ...], value: Any, path: str) -> Any:
    kind = spec.kind
    if kind == "nested":
        if not isinstance(value, Mapping):
            if LenientMapping not in markers:
                raise FieldError(path, "mapping", value, "type")
            value = {}
        return validate_section(spec.item, value, path=path, rules=_rules_of(markers))  # type: ignore[arg-type]
    if kind == "seq_dc":
        entries = next((m for m in markers if isinstance(m, Entries)), None)
        if Verbatim in markers and entries is None:  # a use-site ``Entries`` overrides it
            return value
        if not _is_seq(value):
            raise FieldError(path, "list of mappings", value, "type")
        if entries is not None and entries.max_len is not None and len(value) > entries.max_len:
            raise FieldError(
                path, f"at most {entries.max_len} entries", f"{len(value)} entries", "raw"
            )
        out = []
        for i, elem in enumerate(value):
            if not isinstance(elem, Mapping):
                raise FieldError(f"{path}[{i}]", "mapping", elem, "type")
            out.append(_entry(spec, entries, elem, f"{path}[{i}]"))
        return tuple(out)
    if kind == "map_dc":
        if not isinstance(value, Mapping):
            raise FieldError(path, "mapping", value, "type")
        built = {}
        for name, sub in value.items():
            sub_path = f"{path}.{name}"
            if not isinstance(sub, Mapping):
                raise FieldError(sub_path, "mapping", sub, "type")
            built[str(name)] = validate_section(spec.item, sub, path=sub_path)  # type: ignore[arg-type]
        return MappingProxyType(built) if spec.mapping_factory == "proxy" else built
    command = next((m for m in markers if isinstance(m, CommandTemplate)), None)
    if command is not None:
        return command.apply(value, path)
    coerced = next((m for m in markers if isinstance(m, _Coerced)), None)
    if coerced is not None and kind == "map_str":
        if isinstance(value, Mapping):
            return {str(k): str(v) for k, v in value.items()}
        raise FieldError(path, "mapping of env-var names to values", value, "type")
    if coerced is not None and kind == "seq_str":
        if _is_seq(value):
            return tuple(str(elem) for elem in value)
        if coerced.strict:
            raise FieldError(path, "list", value, "type")
        return value
    if kind == "opaque" or not any(m.checks_type for m in markers):
        # Unchecked field: only the structural list -> tuple coercion applies.
        return tuple(value) if kind == "seq_str" and _is_seq(value) else value
    tolerant = BoolTolerant in markers
    if kind == "seq_str":
        if not _is_seq(value):
            raise FieldError(path, "list of strings", value, "type")
        for i, elem in enumerate(value):
            if not isinstance(elem, str):
                raise FieldError(f"{path}[{i}]", "string", elem, "type")
            _value_markers(f"{path}[{i}]", elem, markers)
        return tuple(value)
    if kind == "map_str":
        if not isinstance(value, Mapping):
            raise FieldError(path, "mapping of string to string", value, "type")
        for k, v in value.items():
            if not isinstance(k, str):
                raise FieldError(path, "mapping of string to string", k, "type")
            if not isinstance(v, str):
                raise FieldError(f"{path}.{k}", "string", v, "type")
            _value_markers(f"{path}.{k}", v, markers)
        return dict(value)
    if not _scalar_ok(kind, value, tolerant, spec):
        raise FieldError(path, _scalar_expected(spec, kind), value, "type")
    _value_markers(path, value, markers)
    return value


def validate_section(
    cls: type[T],
    raw: Mapping[str, Any],
    *,
    path: str,
    rules: Mapping[str, tuple[Marker, ...]] = _NO_RULES,
) -> T:
    """Check ``raw`` against ``cls``'s field metadata (plus use-site ``rules``) and build it.

    Never mutates ``raw``. Raises :class:`ConfigError` (a :class:`FieldError` for
    rule failures) whose message starts with the dotted key path. Order is fixed so
    the first error is deterministic: unknown keys, missing required keys, then
    each present field in dataclass order.
    """
    if not isinstance(raw, Mapping):
        raise FieldError(path, "mapping", raw, "type")
    specs = field_specs(cls)
    by_name = {s.name: s for s in specs}
    bad_rules = sorted(set(rules) - set(by_name))
    if bad_rules:
        raise TypeError(f"{cls.__name__}: rules for unknown field(s) {', '.join(bad_rules)}")
    unknown = sorted(str(k) for k in raw if k not in by_name)
    if unknown:
        raise FieldError(
            path,
            f"known keys (valid: {', '.join(sorted(by_name))})",
            f"unknown key(s) {', '.join(unknown)}",
            "raw",
        )
    for spec in specs:
        if spec.required and raw.get(spec.name) is None:
            raise FieldError(f"{path}.{spec.name}", "required key", "missing", "raw")
    built: dict[str, Any] = {}
    for spec in specs:
        value = raw.get(spec.name)
        markers = spec.markers + tuple(rules.get(spec.name, ()))
        if value is None and LenientMapping in markers and spec.name in raw:
            value = {}  # legacy: ``key: null`` on a lenient nested section builds a bare one
        if value is None:
            if spec.name in raw:
                if NotNull in markers:
                    raise FieldError(f"{path}.{spec.name}", "a value", "null", "raw")
                # ``key: null`` is preserved, never checked and never replaced by the
                # default (design P2): a falsy None differs from a default True.
                if NullIsDefault not in markers:
                    built[spec.name] = None
            continue
        _check_compat(spec, markers, cls)
        key_path = f"{path}.{spec.name}"
        notes = [m.text for m in markers if isinstance(m, Note)]
        try:
            built[spec.name] = _coerce(spec, markers, value, key_path)
        except FieldError as err:
            if not notes:
                raise
            note = f" ({'; '.join(notes)})"
            raise FieldError(err.key, err.expected + note, err.got, err.got_kind) from None
    try:
        return cls(**built)
    except FieldError as err:
        raise err.with_prefix(path) from None
    except ConfigError:
        raise
    except (ValueError, TypeError) as exc:
        text = str(exc)
        if text.startswith(f"{path}."):  # ci_fleet text already names the key path
            raise ConstructionError(text) from exc
        nulls = sorted(k for k, v in built.items() if v is None)
        # A ``key: null`` the constructor cannot compare surfaces as an opaque TypeError
        # ("'<' not supported ... 'NoneType'"); name the candidate keys so it stays actionable.
        hint = f" (null keys: {', '.join(f'{path}.{k}' for k in nulls)})" if nulls else ""
        raise ConstructionError(f"{path}: expected valid construction, got {text}{hint}") from exc


# ----------------------------------------------------------------------- hooks
def _call_hook(path: str, fn: Callable[..., None], *args: Any) -> None:
    try:
        fn(*args)
    except FieldError as err:
        raise err.with_prefix(path) from None


def _hook_value(value: Any, config: Any, path: str) -> None:
    hook = getattr(value, "validate", None)
    if callable(hook):
        _call_hook(path, hook, config)
    for f in fields(value):
        child = getattr(value, f.name)
        child_path = f"{path}.{f.name}"
        if is_dataclass(child) and not isinstance(child, type):
            _hook_value(child, config, child_path)
        elif isinstance(child, Mapping):
            for name, sub in child.items():
                if is_dataclass(sub) and not isinstance(sub, type):
                    _hook_value(sub, config, f"{child_path}.{name}")
        elif isinstance(child, tuple):
            for i, sub in enumerate(child):
                if is_dataclass(sub) and not isinstance(sub, type):
                    _hook_value(sub, config, f"{child_path}[{i}]")


def run_section_hooks(config: Any) -> None:
    """Phase 2: run cross-field/cross-section rules over a fully built root config.

    For each section field (declaration order): the value's ``validate(config)``
    method, then any ``Check`` markers on the field, then the same for nested
    dataclass values. A ``FieldError`` leaves prefixed with the section path.
    """
    hints = _hints(type(config))
    for f in fields(config):
        if f.metadata.get("provenance"):
            continue
        value = getattr(config, f.name)
        if value is None or not is_dataclass(value) or isinstance(value, type):
            continue
        _hook_value(value, config, f.name)
        for m in _markers_of(hints[f.name])[1]:
            if isinstance(m, Check):
                _call_hook(f.name, m.fn, value, config)


__all__ = [
    "AtLeastOne", "BoolTolerant", "Check", "ConfigError", "FieldError", "FieldRules",
    "Coerced", "CoercedLenient", "CommandTemplate", "ConstructionError", "Entries", "Finite", "Ge", "Gt", "HostWideOnly", "InRange", "LenientMapping", "Marker",
    "NonEmpty", "NonEmptyRaw", "NonNeg", "NotNull", "Note", "NullIsDefault", "OneOf", "Placeholders", "Positive", "Regex", "RelativePath", "Typed", "Verbatim",
    "field_markers", "field_rules", "field_specs", "host_wide_error", "host_wide_sections", "render_got",
    "run_section_hooks", "unknown_sections_error", "validate_section",
]  # fmt: skip
