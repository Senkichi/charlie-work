"""``memory/*.md``: bounded notes a foreman keeps for itself.

The foreman's own scratchpad — markdown files it can list, read, write, and
delete. The bounds are deliberately small (32 files, 8 KB each): memory is
for distilled knowledge that earns its keep across Turns, not an archive.
Every bound violation or path escape is a refusal value, not an exception —
the model call gets told "too many memory files" and can act on it.
"""

from __future__ import annotations

from pathlib import Path

from ..atomic_write import write_text_atomic
from . import Result
from .paths import memory_dir, memory_file

MAX_MEMORY_FILES = 32
MAX_MEMORY_BYTES = 8 * 1024


def list_memory(foreman_root: Path) -> tuple[str, ...]:
    """Return the stems of every ``memory/*.md``, sorted."""
    try:
        root = memory_dir(foreman_root)
    except ValueError:
        return ()
    if not root.is_dir():
        return ()
    return tuple(sorted(path.stem for path in root.glob("*.md") if path.is_file()))


def read_memory(foreman_root: Path, name: str) -> Result[str]:
    """Read one memory file by stem or ``*.md`` name."""
    try:
        path = memory_file(foreman_root, name)
    except ValueError as exc:
        return Result(error=str(exc))
    try:
        return Result(value=path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Result(error=f"memory file {name!r}: not found")
    except OSError as exc:
        return Result(error=f"memory file {name!r}: unreadable: {exc}")


def write_memory(foreman_root: Path, name: str, text: str) -> Result[None]:
    """Write one memory file, enforcing the file-count and size bounds.

    A new file is refused once 32 exist (rewriting an existing file always
    fits); the body is bounded at 8 KB encoded.
    """
    try:
        path = memory_file(foreman_root, name)
    except ValueError as exc:
        return Result(error=str(exc))
    try:
        memory_dir(foreman_root).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Result(error=f"memory dir: cannot create: {exc}")
    if len(text.encode("utf-8")) > MAX_MEMORY_BYTES:
        return Result(
            error=f"memory file {name!r}: at most {MAX_MEMORY_BYTES} bytes, "
            f"got {len(text.encode('utf-8'))}"
        )
    existing = {f"{stem}.md" for stem in list_memory(foreman_root)}
    if path.name not in existing and len(existing) >= MAX_MEMORY_FILES:
        return Result(
            error=f"memory: at most {MAX_MEMORY_FILES} files, already have {len(existing)}"
        )
    try:
        write_text_atomic(path, text)
    except OSError as exc:
        return Result(error=f"memory file {name!r}: unwritable: {exc}")
    return Result()


def delete_memory(foreman_root: Path, name: str) -> Result[None]:
    """Delete one memory file; deleting an absent file is a clean no-op."""
    try:
        path = memory_file(foreman_root, name)
    except ValueError as exc:
        return Result(error=str(exc))
    try:
        path.unlink()
    except FileNotFoundError:
        return Result()
    except OSError as exc:
        return Result(error=f"memory file {name!r}: not deletable: {exc}")
    return Result()
