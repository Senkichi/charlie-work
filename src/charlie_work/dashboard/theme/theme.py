"""Generate dashboard CSS custom properties from the vendored Living Journal tokens.

``tokens.json`` is a verbatim copy of swole's ``docs/design/living-journal/tokens.json``
including its ``status`` group (swole PR #467), plus a charlie-work-only ``dashboardLocal``
group (spacing/type scales) that is never compared against swole.
The CSS is pure and deterministic: same tokens in, same bytes out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

STATUS_GROUP = "status"
LOCAL_GROUP = "dashboardLocal"
VENDORED_GROUPS_EXCLUDED = (STATUS_GROUP, LOCAL_GROUP)
SWOLE_TOKENS_RELPATH = Path("docs/design/living-journal/tokens.json")

# swole token name -> CSS custom property. Dark-only tokens (raised, equipment,
# dialTrack) have fallbacks in _theme_vars so every property exists in both themes.
_COLOR_VARS: tuple[tuple[str, str], ...] = (
    ("page", "--lj-page"),
    ("card", "--lj-card"),
    ("ink", "--lj-ink"),
    ("inkSecondary", "--lj-gray"),
    ("inkSecondaryText", "--lj-gray-text"),
    ("hairline", "--lj-hair"),
    ("semantic", "--lj-green"),
    ("semanticText", "--lj-green-text"),
)
_STATUS_VARS: tuple[tuple[str, str], ...] = (
    ("danger", "--lj-danger"),
    ("warn", "--lj-warn"),
)
_CHART_VARS: tuple[tuple[str, str], ...] = (
    ("series1", "--lj-series-1"),
    ("series2", "--lj-series-2"),
    ("series3", "--lj-series-3"),
)


def load_tokens() -> dict[str, Any]:
    """Read the vendored tokens.json via importlib.resources (ships in the wheel)."""
    text = resources.files(__package__).joinpath("tokens.json").read_text(encoding="utf-8")
    return json.loads(text)


def base_groups(tokens: Mapping[str, Any]) -> dict[str, Any]:
    """Groups swole's ``color``-parity base: everything but ``status`` and ``dashboardLocal``."""
    return {k: v for k, v in tokens.items() if k not in VENDORED_GROUPS_EXCLUDED}


def _value(group: Mapping[str, Any], name: str) -> str:
    return str(group[name]["$value"])


def _theme_vars(tokens: Mapping[str, Any], mode: str) -> list[tuple[str, str]]:
    color = tokens["color"][mode]
    status = tokens[STATUS_GROUP][mode]
    chart = tokens[STATUS_GROUP]["chart"][mode]
    out = [(var, _value(color, name)) for name, var in _COLOR_VARS]
    # Dark-only tokens fall back so a property never goes missing across a theme switch:
    # raised -> card, equipment -> gray, rule -> dialTrack (dark) / rule (light).
    out.append(("--lj-raised", _value(color, "raised" if "raised" in color else "card")))
    out.append(
        ("--lj-equipment", _value(color, "equipment" if "equipment" in color else "inkSecondary"))
    )
    out.append(("--lj-rule", _value(color, "rule" if "rule" in color else "dialTrack")))
    out.extend((var, _value(status, name)) for name, var in _STATUS_VARS)
    out.extend((var, _value(chart, name)) for name, var in _CHART_VARS)
    return out


def _font_stack(fonts: Mapping[str, Any]) -> str:
    families = fonts["fontFamily"]["$value"]
    generic = {"serif", "sans-serif", "system-ui", "monospace"}
    return ", ".join(f if f in generic else f"'{f}'" for f in families)


def _static_vars(tokens: Mapping[str, Any]) -> list[tuple[str, str]]:
    typo, local = tokens["typography"], tokens[LOCAL_GROUP]
    out = [
        ("--lj-serif", _font_stack(typo["display"])),
        ("--lj-sans", _font_stack(typo["body"])),
        ("--lj-label-spacing", _value(typo["label"], "letterSpacing")),
    ]
    out += [
        (f"--lj-stroke-{k}", _value(tokens["stroke"], k)) for k in sorted_keys(tokens["stroke"])
    ]
    out += [
        (f"--lj-radius-{k}", _value(tokens["radius"], k)) for k in sorted_keys(tokens["radius"])
    ]
    out += [
        (f"--lj-space-{i}", f"{v}px") for i, v in enumerate(local["spacing"]["$value"], start=1)
    ]
    out += [(f"--lj-text-{i}", f"{v}px") for i, v in enumerate(local["typeSize"]["$value"], 1)]
    return out


def sorted_keys(group: Mapping[str, Any]) -> list[str]:
    """Token names in a group, skipping ``$``-prefixed metadata, in file order."""
    return [k for k in group if not k.startswith("$")]


def _block(selector: str, decls: list[tuple[str, str]], indent: str = "") -> str:
    body = "".join(f"{indent}  {k}: {v};\n" for k, v in decls)
    return f"{indent}{selector} {{\n{body}{indent}}}\n"


def generate_css(tokens: Mapping[str, Any] | None = None) -> str:
    """CSS custom properties: light on :root, dark via media query and explicit attribute."""
    tokens = load_tokens() if tokens is None else tokens
    light = _static_vars(tokens) + _theme_vars(tokens, "light")
    dark = _theme_vars(tokens, "dark")
    parts = [
        "/* generated from tokens.json; do not edit */\n",
        _block(":root", light),
        "@media (prefers-color-scheme: dark) {\n"
        + _block(':root:not([data-theme="light"])', dark, "  ")
        + "}\n",
        _block(':root[data-theme="dark"]', dark),
        _block(
            "body",
            [
                ("background", "var(--lj-page)"),
                ("color", "var(--lj-ink)"),
                ("font-family", "var(--lj-sans)"),
            ],
        ),
    ]
    return "\n".join(parts)


def _channel(c: int) -> float:
    s = c / 255
    return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.1 contrast ratio of two opaque ``#RRGGBB`` colours."""

    def lum(h: str) -> float:
        r, g, b = (int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
        return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)

    hi, lo = sorted((lum(fg), lum(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@dataclass(frozen=True)
class DriftResult:
    """Outcome of comparing the vendored base and status groups to a swole checkout.

    ``status`` is ``in_sync``, ``drifted`` (``differing`` names the base groups), or
    ``unavailable`` (no checkout, or its tokens.json is unreadable; ``detail`` says why).
    """

    status: str
    differing: tuple[str, ...] = ()
    detail: str = ""


def swole_drift(swole_root: Path | str | None) -> DriftResult:
    """Compare base and status groups with ``<swole_root>/docs/design/living-journal/tokens.json``.

    Never raises: a missing/unreadable/malformed checkout yields ``unavailable``.
    """
    if swole_root is None:
        return DriftResult("unavailable", detail="no swole checkout configured")
    path = Path(swole_root) / SWOLE_TOKENS_RELPATH
    try:
        upstream = json.loads(path.read_text(encoding="utf-8"))
        vendored = load_tokens()
        if not isinstance(upstream, dict):
            raise ValueError("tokens.json is not an object")
    except (OSError, ValueError) as exc:
        return DriftResult("unavailable", detail=f"{path}: {exc}")
    v, u = base_groups(vendored), base_groups(upstream)
    differing = {g for g in v.keys() | u.keys() if v.get(g) != u.get(g)}
    # The status group is vendored verbatim from swole PR #467; compare it too. Only an
    # upstream that has it can be compared (a pre-#467 checkout simply lacks the group).
    if STATUS_GROUP in upstream and vendored.get(STATUS_GROUP) != upstream[STATUS_GROUP]:
        differing.add(STATUS_GROUP)
    differing = tuple(sorted(differing))
    return DriftResult("drifted" if differing else "in_sync", differing)


def find_swole_root(registry: Mapping[str, Any]) -> Path | None:
    """The swole checkout registered in the fleet registry, or None.

    ``registry`` is the loaded ``fleet.json`` dict (``{"repos": {nameWithOwner: entry}}``).
    Picks the repo whose name (the part after ``owner/``) ends with ``swole`` and whose
    ``repo_root`` is an existing directory. Entries are examined in sorted key order so the
    choice is deterministic. Never raises on a malformed registry.
    """
    repos = registry.get("repos") if isinstance(registry, Mapping) else None
    if not isinstance(repos, Mapping):
        return None
    for key in sorted(k for k in repos if isinstance(k, str)):
        entry = repos[key]
        if not key.rsplit("/", 1)[-1].lower().endswith("swole") or not isinstance(entry, Mapping):
            continue
        root = entry.get("repo_root")
        if isinstance(root, str) and root and Path(root).is_dir():
            return Path(root)
    return None
