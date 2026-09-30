"""Unit tests for the generic config validator (ADR-0007): markers, messages, hooks.

Decision tables: each row is (field value -> accepted) or (field value -> exact
message). No real config section is exercised here; the fixtures below are the
only classes carrying markers.
"""

from __future__ import annotations

import ast
import inspect
import pickle
import textwrap
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Annotated

import pytest

from charlie_work import config as config_mod
from charlie_work import config_validation as cv
from charlie_work.config_validation import (
    AtLeastOne,
    BoolTolerant,
    Check,
    ConfigError,
    FieldError,
    FieldRules,
    Finite,
    Ge,
    HostWideOnly,
    InRange,
    LenientMapping,
    NonEmpty,
    NonNeg,
    Note,
    OneOf,
    Placeholders,
    Positive,
    Regex,
    RelativePath,
    Typed,
    render_got,
    run_section_hooks,
    validate_section,
)


@dataclass(frozen=True)
class Inner:
    depth: Annotated[int, NonNeg] = 1


@dataclass(frozen=True)
class Entry:
    name: Annotated[str, NonEmpty]
    weight: Annotated[int, AtLeastOne]


@dataclass(frozen=True)
class Sec:
    count: Annotated[int, NonNeg] = 0
    tol: Annotated[int, Typed, BoolTolerant] = 0
    ratio: Annotated[float, InRange(0.0, 1.0)] = 0.5
    wait: Annotated[float, Finite] = 1.0
    limit: Annotated[int, Positive] = 1
    flag: Annotated[bool, Typed] = False
    label: Annotated[str, NonEmpty] = "x"
    order: Annotated[str, OneOf("fifo", "lifo", "rand")] = "fifo"
    cmd: Annotated[tuple[str, ...], Placeholders({"prompt", "path"})] = ()
    pattern: Annotated[str, Regex] = "a"
    rel: Annotated[str, RelativePath] = "a/b"
    env: Annotated[dict[str, str], Typed] = field(default_factory=dict)
    depth: Annotated[int, NonNeg, Note("blocked-ready-issue count; see #1768")] = 0
    free: int = 0  # unannotated: never checked
    inner: Inner = field(default_factory=Inner)
    lenient: Annotated[Inner, LenientMapping] = field(default_factory=Inner)
    entries: tuple[Entry, ...] = ()
    named: Mapping[str, Entry] = field(default_factory=lambda: MappingProxyType({}))
    maybe: Annotated[str | bool | None, Typed] = None
    names: Annotated[tuple[str, ...], NonEmpty] = ()


def build(**raw: object) -> Sec:
    return validate_section(Sec, raw, path="sec")


def err(**raw: object) -> str:
    with pytest.raises(ConfigError) as info:
        build(**raw)
    return str(info.value)


# --------------------------------------------------------------- render_got
@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (5, "value", "5"),
        ("5", "value", "'5'"),
        ("5", "type", "'5' (str)"),
        (True, "type", "True (bool)"),
        ([1], "type", "[1] (list)"),
        ("missing", "raw", "missing"),
        ("x" * 200, "value", repr("x" * 200)[:77] + "..."),
    ],
)
def test_render_got_table(value, kind, expected):
    assert render_got(value, kind) == expected
    assert len(render_got(value, kind)) <= 89  # 80 cap plus " (type)" suffix at most


# ------------------------------------------------ scalar decision table
ACCEPT = [
    {"count": 0},
    {"count": 7},
    {"tol": True},
    {"tol": 3},
    {"ratio": 0},
    {"ratio": 1},
    {"ratio": 0.25},
    {"wait": 3},
    {"wait": 0.0},
    {"limit": 1},
    {"flag": True},
    {"label": " y "},
    {"order": "lifo"},
    {"cmd": ["{prompt}", "--x", "{path}"]},
    {"cmd": ("plain",)},
    {"pattern": r"^a.*\d+$"},
    {"rel": "a/b/c"},
    {"rel": "a..b"},
    {"env": {"A": "1"}},
    {"free": "anything"},
    {"free": [1, 2]},
    {"maybe": "s"},
    {"maybe": False},
    {"names": ["a", "b"]},
    {"count": None},  # key: null behaves like an absent key
    {"limit": None},
]


@pytest.mark.parametrize("raw", ACCEPT, ids=[str(r) for r in ACCEPT])
def test_accepts(raw):
    build(**raw)


REJECT = [
    ({"count": -1}, "sec.count: expected >= 0, got -1"),
    ({"count": "5"}, "sec.count: expected int, got '5' (str)"),
    ({"count": 1.5}, "sec.count: expected int, got 1.5 (float)"),
    ({"count": True}, "sec.count: expected int, got True (bool)"),
    ({"tol": "1"}, "sec.tol: expected int, got '1' (str)"),
    ({"ratio": -0.1}, "sec.ratio: expected in [0.0, 1.0], got -0.1"),
    ({"ratio": 1.1}, "sec.ratio: expected in [0.0, 1.0], got 1.1"),
    ({"ratio": "half"}, "sec.ratio: expected number, got 'half' (str)"),
    ({"ratio": True}, "sec.ratio: expected number, got True (bool)"),
    ({"ratio": float("nan")}, "sec.ratio: expected in [0.0, 1.0], got nan"),
    ({"wait": float("inf")}, "sec.wait: expected finite number, got inf"),
    ({"wait": float("nan")}, "sec.wait: expected finite number, got nan"),
    ({"limit": 0}, "sec.limit: expected > 0, got 0"),
    ({"limit": -3}, "sec.limit: expected > 0, got -3"),
    ({"flag": 1}, "sec.flag: expected bool, got 1 (int)"),
    ({"flag": "true"}, "sec.flag: expected bool, got 'true' (str)"),
    ({"label": "  "}, "sec.label: expected non-empty string, got '  '"),
    ({"label": ""}, "sec.label: expected non-empty string, got ''"),
    ({"label": 3}, "sec.label: expected string, got 3 (int)"),
    ({"order": "zig"}, "sec.order: expected one of 'fifo', 'lifo', 'rand', got 'zig'"),
    ({"order": 7}, "sec.order: expected string, got 7 (int)"),
    ({"cmd": "{prompt}"}, "sec.cmd: expected list of strings, got '{prompt}' (str)"),
    ({"cmd": ["ok", 3]}, "sec.cmd[1]: expected string, got 3 (int)"),
    (
        {"cmd": ["{prompt}", "{nope}"]},
        "sec.cmd[1]: expected placeholders from {path}, {prompt}, got unknown placeholder {nope}",
    ),
    (
        {"cmd": ["{}"]},
        "sec.cmd[0]: expected placeholders from {path}, {prompt}, got empty placeholder {}",
    ),
    (
        {"cmd": ["{prompt"]},
        "sec.cmd[0]: expected placeholders from {path}, {prompt}, "
        "got malformed placeholder in '{prompt' (expected '}' before end of string)",
    ),
    ({"pattern": "("}, None),  # message text comes from re.error; checked separately
    ({"rel": "/abs"}, "sec.rel: expected relative path without '..', got '/abs'"),
    ({"rel": "a/../b"}, "sec.rel: expected relative path without '..', got 'a/../b'"),
    ({"rel": "a\\..\\b"}, "sec.rel: expected relative path without '..', got 'a\\\\..\\\\b'"),
    ({"rel": "C:\\x"}, "sec.rel: expected relative path without '..', got 'C:\\\\x'"),
    ({"env": ["A"]}, "sec.env: expected mapping of string to string, got ['A'] (list)"),
    ({"env": {"A": 1}}, "sec.env.A: expected string, got 1 (int)"),
    (
        {"depth": -1},
        "sec.depth: expected >= 0 (blocked-ready-issue count; see #1768), got -1",
    ),
    (
        {"depth": "a"},
        "sec.depth: expected int (blocked-ready-issue count; see #1768), got 'a' (str)",
    ),
    ({"maybe": 3}, "sec.maybe: expected string or bool, got 3 (int)"),
    ({"names": ["a", " "]}, "sec.names[1]: expected non-empty string, got ' '"),
]


@pytest.mark.parametrize(("raw", "message"), REJECT, ids=[str(r) for r, _ in REJECT])
def test_rejects_with_exact_message(raw, message):
    text = err(**raw)
    if message is None:
        assert text.startswith("sec.pattern: expected valid regex, got '('")
    else:
        assert text == message


def test_validation_error_carries_structured_fields():
    with pytest.raises(FieldError) as info:
        build(count=-2)
    e = info.value
    assert (e.key, e.expected, e.got, e.got_kind) == ("sec.count", ">= 0", -2, "value")
    assert isinstance(e, ConfigError) and isinstance(e, ValueError)


def test_field_error_pickles():
    e = pickle.loads(pickle.dumps(FieldError("a.b", ">= 0", -1)))  # noqa: S301
    assert str(e) == "a.b: expected >= 0, got -1"


def test_none_is_omitted_so_default_applies():
    assert build(count=None).count == 0 and build(limit=None).limit == 1


def test_raw_is_not_mutated_and_lists_become_tuples():
    raw = {"cmd": ["{prompt}"], "names": ["a"], "entries": [{"name": "n", "weight": 2}]}
    snapshot = {k: list(v) for k, v in raw.items()}
    sec = build(**raw)
    assert {k: list(v) for k, v in raw.items()} == snapshot
    assert sec.cmd == ("{prompt}",) and sec.names == ("a",)
    assert sec.entries == (Entry("n", 2),)


# ------------------------------------------------------------ keys and nesting
def test_unknown_keys_message_lists_sorted_valid_keys():
    text = err(zeta=1, alpha=2)
    assert text == (
        "sec: expected known keys (valid: "
        + ", ".join(sorted(f.name for f in fields(Sec)))
        + "), got unknown key(s) alpha, zeta"
    )


def test_unknown_keys_are_reported_before_type_errors():
    assert err(count="x", bogus=1).startswith("sec: expected known keys")


def test_first_error_follows_dataclass_field_order():
    assert err(limit=0, count=-1).startswith("sec.count:")


def test_non_mapping_section_is_rejected():
    with pytest.raises(FieldError) as info:
        validate_section(Sec, [1], path="sec")
    assert str(info.value) == "sec: expected mapping, got [1] (list)"


def test_nested_dataclass_paths_and_defaults():
    assert build().inner == Inner()
    assert err(inner={"depth": -1}) == "sec.inner.depth: expected >= 0, got -1"
    assert err(inner={"nope": 1}).startswith("sec.inner: expected known keys (valid: depth)")
    assert err(inner=[]) == "sec.inner: expected mapping, got [] (list)"
    assert build(inner={"depth": 4}).inner == Inner(4)


def test_lenient_mapping_treats_non_mapping_as_empty():
    assert build(lenient=5).lenient == Inner()
    assert build(lenient="x").lenient == Inner()
    assert err(lenient={"depth": -1}) == "sec.lenient.depth: expected >= 0, got -1"


def test_sequence_of_sections_paths_and_required_keys():
    assert err(entries=[{"name": "n", "weight": 0}]) == (
        "sec.entries[0].weight: expected >= 1, got 0"
    )
    assert (
        err(entries=[{"name": "n"}]) == "sec.entries[0].weight: expected required key, got missing"
    )
    assert err(entries=[{"name": "n", "weight": None}]) == (
        "sec.entries[0].weight: expected required key, got missing"
    )
    assert err(entries=[3]) == "sec.entries[0]: expected mapping, got 3 (int)"
    assert err(entries="x") == "sec.entries: expected list of mappings, got 'x' (str)"


def test_mapping_of_sections_uses_dot_paths_and_freezes_proxies():
    sec = build(named={"kimi-k3": {"name": "n", "weight": 1}})
    assert sec.named["kimi-k3"] == Entry("n", 1)
    assert err(named={"kimi-k3": {"name": "", "weight": 1}}) == (
        "sec.named.kimi-k3.name: expected non-empty string, got ''"
    )
    assert err(named={"p": 3}) == "sec.named.p: expected mapping, got 3 (int)"
    assert err(named=[]) == "sec.named: expected mapping, got [] (list)"


def test_unannotated_field_is_never_checked_but_lists_stay_lists():
    assert build(free={"a": 1}).free == {"a": 1}


# ------------------------------------------------------------- FieldRules
@dataclass(frozen=True)
class Plain:
    n: int = 0
    s: str = ""
    flag: bool = False


@dataclass(frozen=True)
class Holder:
    plain: Annotated[Plain, FieldRules(n=(Typed, BoolTolerant), s=NonEmpty)] = field(
        default_factory=Plain
    )


def test_field_rules_refine_fields_of_an_unannotated_class():
    assert validate_section(Holder, {"plain": {"n": True, "s": "x"}}, path="h").plain.n is True
    with pytest.raises(FieldError, match=r"^h\.plain\.s: expected non-empty string, got ''$"):
        validate_section(Holder, {"plain": {"s": ""}}, path="h")
    with pytest.raises(FieldError, match=r"^h\.plain\.n: expected int, got 'x' \(str\)$"):
        validate_section(Holder, {"plain": {"n": "x"}}, path="h")
    # fields without a rule stay unchecked
    assert (
        validate_section(Holder, {"plain": {"flag": "whatever"}}, path="h").plain.flag
        == "whatever"
    )


def test_use_site_rules_argument_applies_to_top_level_fields():
    rules = {"n": (NonNeg,)}
    with pytest.raises(FieldError, match=r"^p\.n: expected >= 0, got -1$"):
        validate_section(Plain, {"n": -1}, path="p", rules=rules)
    assert validate_section(Plain, {"n": -1}, path="p") == Plain(n=-1)


def test_rules_naming_a_missing_field_are_a_programming_error():
    with pytest.raises(TypeError, match="unknown field"):
        validate_section(Plain, {}, path="p", rules={"nope": (NonNeg,)})


def test_marker_on_incompatible_annotation_is_a_programming_error():
    with pytest.raises(TypeError, match="cannot annotate"):
        validate_section(Plain, {"s": "x"}, path="p", rules={"s": (NonNeg,)})
    with pytest.raises(TypeError, match="unsupported annotation"):
        validate_section(Opaque, {"x": 1}, path="o")


@dataclass(frozen=True)
class Opaque:
    x: Annotated[frozenset[int], Typed] = frozenset()


# --------------------------------------------------------- construction errors
@dataclass(frozen=True)
class Strict:
    a: Annotated[int, NonNeg] = 0
    b: int = 0

    def __post_init__(self) -> None:
        if self.b > 10:
            raise FieldError("b", "<= 10", self.b)
        if self.b == 7:
            raise ValueError("strict.b must be != 7 -- a ci_fleet style message")
        if self.b == 8:
            raise ValueError("some bare failure")
        if self.b == 9:
            raise ConfigError("already a ConfigError")


@pytest.mark.parametrize(
    ("path", "b", "message"),
    [
        ("s", 11, "s.b: expected <= 10, got 11"),
        ("strict", 7, "strict.b must be != 7 -- a ci_fleet style message"),
        ("s", 8, "s: expected valid construction, got some bare failure"),
        ("s", 9, "already a ConfigError"),
    ],
)
def test_construction_errors_become_config_errors(path, b, message):
    with pytest.raises(ConfigError) as info:
        validate_section(Strict, {"b": b}, path=path)
    assert str(info.value) == message


def test_post_init_field_error_gets_the_section_prefix_and_direct_use_does_not():
    with pytest.raises(FieldError) as info:
        validate_section(Strict, {"b": 11}, path="outer.s")
    assert str(info.value) == "outer.s.b: expected <= 10, got 11"
    with pytest.raises(FieldError, match=r"^b: expected <= 10, got 11$"):
        Strict(b=11)


# ------------------------------------------------------------------ hooks
@dataclass(frozen=True)
class Cross:
    floor: int = 0
    ceiling: int = 5

    def validate(self, config: Root) -> None:
        if self.floor > self.ceiling:
            raise FieldError("floor", "<= ceiling", f"{self.floor} > {self.ceiling}", "raw")
        if config.other.value < 0:
            raise FieldError("floor", "a non-negative other.value", config.other.value)


@dataclass(frozen=True)
class Other:
    value: int = 0


@dataclass(frozen=True)
class Ext:  # stands in for a class we cannot edit
    lo: int = 0


def _ext_check(ext: Ext, config: Root) -> None:
    if ext.lo > config.cross.ceiling:
        raise FieldError("lo", "<= cross.ceiling", ext.lo)


@dataclass(frozen=True)
class Root:
    cross: Cross = field(default_factory=Cross)
    other: Other = field(default_factory=Other)
    ext: Annotated[Ext, HostWideOnly, Check(_ext_check)] = field(default_factory=Ext)
    plain: Annotated[Other, FieldRules(value=NonNeg)] = field(default_factory=Other)
    tagged: dict = field(default_factory=dict, metadata={"provenance": True})


def test_hooks_pass_on_valid_config():
    run_section_hooks(Root())


def test_method_hook_error_is_prefixed_with_the_section_name():
    with pytest.raises(FieldError) as info:
        run_section_hooks(Root(cross=Cross(floor=9, ceiling=3)))
    assert str(info.value) == "cross.floor: expected <= ceiling, got 9 > 3"


def test_method_hook_can_read_other_sections():
    with pytest.raises(FieldError) as info:
        run_section_hooks(Root(other=Other(-4)))
    assert str(info.value) == "cross.floor: expected a non-negative other.value, got -4"


def test_check_marker_hook_runs_for_classes_we_do_not_own():
    with pytest.raises(FieldError) as info:
        run_section_hooks(Root(ext=Ext(lo=6)))
    assert str(info.value) == "ext.lo: expected <= cross.ceiling, got 6"


@dataclass(frozen=True)
class Leaf:
    v: int = 0

    def validate(self, config: object) -> None:
        if self.v:
            raise FieldError("v", "zero", self.v)


@dataclass(frozen=True)
class Branch:
    leaf: Leaf = field(default_factory=Leaf)
    leaves: Mapping[str, Leaf] = field(default_factory=lambda: MappingProxyType({}))
    many: tuple[Leaf, ...] = ()


@dataclass(frozen=True)
class NestedRoot:
    branch: Branch = field(default_factory=Branch)


@pytest.mark.parametrize(
    ("branch", "message"),
    [
        (Branch(leaf=Leaf(1)), "branch.leaf.v: expected zero, got 1"),
        (
            Branch(leaves=MappingProxyType({"kimi": Leaf(2)})),
            "branch.leaves.kimi.v: expected zero, got 2",
        ),
        (Branch(many=(Leaf(), Leaf(3))), "branch.many[1].v: expected zero, got 3"),
    ],
)
def test_hooks_recurse_into_nested_sections(branch, message):
    with pytest.raises(FieldError) as info:
        run_section_hooks(NestedRoot(branch=branch))
    assert str(info.value) == message


# ------------------------------------------------ host-wide + grammar helpers
def test_host_wide_sections_derived_from_root_field_metadata():
    assert cv.host_wide_sections(Root) == frozenset({"ext"})
    assert cv.host_wide_sections(NestedRoot) == frozenset()


def test_host_wide_error_and_unknown_sections_messages():
    assert str(cv.host_wide_error("ext", fleet_dir="F", repo_path="R")) == (
        "ext: expected host-wide only (declare it in F/config.yaml), got per-repo config R"
    )
    assert str(cv.unknown_sections_error(["z", "a"], ["m", "b"])) == (
        "config: expected known sections (valid: b, m), got unknown config section(s) a, z"
    )


def test_host_wide_sections_are_derived_from_root_field_markers():
    assert cv.host_wide_sections() == frozenset(
        {"runner_allocation", "runner_capacity_escalation", "fleet_supervisor"}
    )


# ------------------------------------------------------------- marker tables
@pytest.mark.parametrize(
    ("marker", "value", "expected"),
    [
        (Ge(2), 2, None),
        (Ge(2), 1, (">= 2", 1, "value")),
        (cv.Gt(2), 2, ("> 2", 2, "value")),
        (cv.Gt(2), 3, None),
        (InRange(1, 3), 3, None),
        (InRange(1, 3), 4, ("in [1, 3]", 4, "value")),
        (NonEmpty, "a", None),
        (NonEmpty, "", ("non-empty string", "", "value")),
        (OneOf(1, 2), 2, None),
        (OneOf(1, 2), 3, ("one of 1, 2", 3, "value")),
        (Placeholders(()), "{x}", ("no placeholders", "unknown placeholder {x}", "raw")),
        (Placeholders(("x",)), "{x}{x}", None),
        (RelativePath, "a", None),
        (Regex, "[a-z]+", None),
    ],
)
def test_marker_problem_table(marker, value, expected):
    assert marker.problem(value) == expected


def test_constants_are_the_documented_bounds():
    assert NonNeg == Ge(0) and AtLeastOne == Ge(1) and Positive == cv.Gt(0)


def test_config_error_is_shared_with_config_module():
    assert config_mod.ConfigError is ConfigError


def test_every_field_rules_key_names_a_real_field():
    """Catches a renamed ci_fleet / section field at CI time, not at a live load."""
    hints_root = config_mod.OrchestratorConfig
    for f in fields(hints_root):
        if f.metadata.get("provenance"):
            continue
        for marker in cv.field_markers(hints_root, f.name):
            if isinstance(marker, FieldRules):
                target = next(s.item for s in cv.field_specs(hints_root) if s.name == f.name)
                assert set(marker.as_mapping()) <= {x.name for x in fields(target)}


def _section_classes(root: type) -> list[type]:
    seen: list[type] = []
    stack = [s.item for s in cv.field_specs(root) if s.item is not None]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        stack.extend(s.item for s in cv.field_specs(cls) if s.item is not None)
    return seen


def test_every_section_class_annotation_resolves_with_supported_markers():
    """Fails in CI, not at a live load, if a marked annotation is unsupported."""
    classes = _section_classes(config_mod.OrchestratorConfig)
    assert len(classes) > 10  # positive control: the walk actually found sections
    for cls in classes:
        for spec in cv.field_specs(cls):
            cv._check_compat(spec, spec.markers, cls)


# ------------------------------------------------------ single point of enforcement
def _hand_written_checks(fn) -> list[str]:
    """``isinstance`` calls and ``raise ConfigError(...)`` inside ``fn`` (what ADR-0007 bans)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "isinstance"
        ):
            found.append(f"isinstance@{node.lineno}")
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            callee = node.exc.func
            if isinstance(callee, ast.Name) and callee.id in {"ConfigError", "FieldError"}:
                found.append(f"raise {callee.id}@{node.lineno}")
    return found


def test_guard_detects_hand_written_checks():
    """Positive control: the AST guard below would fail on the pre-migration shape."""

    def legacy(data):
        if not isinstance(data, dict):
            raise ConfigError("config section 'x' key 'y' must be an int")

    assert len(_hand_written_checks(legacy)) == 2


def test_build_config_from_data_has_no_hand_written_validation():
    """Section rules live in field metadata; the router only routes (ADR-0007)."""
    assert _hand_written_checks(config_mod.build_config_from_data) == []
