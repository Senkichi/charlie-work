"""G2: sibling-destination naming + pre-wired scaffold.

`suggest` finds the nearest non-saturated same-kind sibling attachment point
(fewest members, same package directory preferred) as the destination for new
growth; when no such sibling exists it proposes a new module name derived
structurally from the attachment point's own shape (its most recently defined
member, its kind, its file's directory) — never a hand-maintained list of
destination names.

`scaffold` renders the sibling file's content (relevant imports + the
registration wiring for the point's kind) so the caller only has to write the
member body. It returns a `ScaffoldPlan` and writes nothing itself; callers
(the hook, the CLI) decide whether and how to present or persist it.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from charlie_work.attachment_contracts.model import (
    AttachmentPoint,
    Kind,
    Redirect,
    ScaffoldPlan,
    ScanResult,
)
from charlie_work.attachment_contracts.outliers import saturate


def _package_dir(file: str) -> str:
    parent = PurePosixPath(file).parent
    return "" if str(parent) == "." else str(parent)


def _derive_verb(member_name: str) -> str:
    """First underscore-separated token of a member name, or a safe fallback."""
    stripped = member_name.lstrip("_")
    token = stripped.split("_")[0] if stripped else ""
    return token or "extracted"


def _last_member(point: AttachmentPoint) -> str:
    return point.members[-1] if point.members else ""


def _join(directory: str, filename: str) -> str:
    return f"{directory}/{filename}" if directory else filename


def _propose_new_module(point: AttachmentPoint) -> str:
    """Derive a new sibling module path from the point's own shape.

    class -> `<verb>_ops.py` beside the source file.
    test_module -> `tests/<topic>/test_<topic>.py`.
    typer_app / click_group / blueprint / migration_runner -> `<verb>_commands.py`
    beside the source file (same shape as the class case: a verb pulled from
    the point's most recently defined member).
    """
    verb = _derive_verb(_last_member(point))
    directory = _package_dir(point.file)
    if point.kind == "test_module":
        stem = PurePosixPath(point.file).stem
        topic = stem[len("test_") :] if stem.startswith("test_") else stem
        topic = topic[: -len("_test")] if topic.endswith("_test") else topic
        return f"tests/{topic}/test_{topic}.py"
    if point.kind == "class":
        return _join(directory, f"{verb}_ops.py")
    return _join(directory, f"{verb}_commands.py")


def suggest(point: AttachmentPoint, scan: ScanResult) -> Redirect:
    """Nearest non-saturated same-kind sibling, else a proposed new module."""
    verdicts = saturate(scan.points, point.kind)
    saturated_keys = {(v.point.file, v.point.identity) for v in verdicts if v.saturated}
    point_dir = _package_dir(point.file)

    siblings = [
        p
        for p in scan.points
        if p.kind == point.kind
        and not p.is_linear_ledger
        and not p.is_structurally_trivial
        and (p.file, p.identity) != (point.file, point.identity)
        and (p.file, p.identity) not in saturated_keys
    ]

    if siblings:

        def sort_key(p: AttachmentPoint) -> tuple[int, int, str, str]:
            same_dir = 0 if _package_dir(p.file) == point_dir else 1
            return (same_dir, p.member_count, p.file, p.identity)

        best = min(siblings, key=sort_key)
        return Redirect(
            destination=best.file,
            rationale=(
                f"{best.identity} ({best.kind}) has {best.member_count} members and is "
                "not saturated; nearest non-saturated sibling."
            ),
            is_new_module=False,
        )

    destination = _propose_new_module(point)
    return Redirect(
        destination=destination,
        rationale=(
            f"no non-saturated sibling {point.kind} attachment point found in the repo; "
            "proposing a new module."
        ),
        is_new_module=True,
    )


def _class_name_for(verb: str) -> str:
    return "".join(part.capitalize() for part in verb.split("_") if part) + "Ops"


def scaffold(point: AttachmentPoint, redirect: Redirect, member_name: str) -> ScaffoldPlan:
    """Render the redirect destination's content + registration wiring note."""
    kind: Kind = point.kind
    if kind in ("typer_app", "click_group"):
        content = (
            "from __future__ import annotations\n\n"
            "import typer\n\n"
            "app = typer.Typer()\n\n\n"
            f"@app.command()\n"
            f"def {member_name}() -> None:\n"
            f'    """TODO: implement {member_name}."""\n'
            "    raise NotImplementedError\n"
        )
        registration_note = (
            f"Wire into the source router: `{point.identity.split(':')[-1]}.add_typer"
            f'(app, name="{member_name}")` (or the click-group equivalent) in {point.file}.'
        )
    elif kind == "blueprint":
        content = (
            "from __future__ import annotations\n\n"
            "from flask import Blueprint\n\n"
            f'bp = Blueprint("{member_name}", __name__)\n\n\n'
            f'@bp.route("/{member_name}")\n'
            f"def {member_name}() -> str:\n"
            f'    """TODO: implement {member_name}."""\n'
            "    raise NotImplementedError\n"
        )
        registration_note = "Register in the app factory: `app.register_blueprint(bp)`."
    elif kind == "test_module":
        content = (
            "from __future__ import annotations\n\n\n"
            f"def {member_name}() -> None:\n"
            f'    """TODO: implement {member_name}."""\n'
            "    raise NotImplementedError\n"
        )
        registration_note = (
            f"Import shared fixtures from the original test module ({point.file}) or its "
            "conftest as needed."
        )
    else:  # class, migration_runner fallback
        verb = _derive_verb(member_name)
        class_name = _class_name_for(verb)
        content = (
            "from __future__ import annotations\n\n\n"
            f"class {class_name}:\n"
            f'    """Extracted ops for {point.identity}."""\n\n'
            f"    def {member_name}(self) -> None:\n"
            f'        """TODO: implement {member_name}."""\n'
            "        raise NotImplementedError\n"
        )
        registration_note = (
            f"Compose {class_name} into {point.identity} (delegation or mixin) rather than "
            f"adding {member_name} to it directly."
        )
    return ScaffoldPlan(
        path=redirect.destination, content=content, registration_note=registration_note
    )
