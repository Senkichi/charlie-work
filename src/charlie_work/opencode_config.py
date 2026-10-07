"""``opencode:`` config section -- settings for the opencode worker harness.

Lives outside ``config.py`` (which sits at its file-size ratchet mark) and is
re-exported there as the type of ``OrchestratorConfig.opencode``. The model a
launch runs is ``worker.model`` / the selected ``worker.fallbacks`` entry's
model, never a field here -- the same single-source-of-truth rule
``ClaudeCodeConfig`` documents. ``opencode_worker.launch_opencode_worker``
pins it as ``--model <provider>/<model>`` on every launch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from .config_validation import CommandTemplate, Coerced, Typed


@dataclass(frozen=True)
class OpenCodeConfig:
    """Settings for ``worker.harness: opencode`` (or an ``opencode`` fallback entry).

    ``command``: empty means ``opencode_worker.DEFAULT_COMMAND_TEMPLATE``. The
    rendered worker prompt is fed via stdin unless the template names
    ``{prompt_path}``; the launcher appends ``--model``/``--variant`` pins.

    ``provider``: prefixed onto a bare ``worker.model`` (``glm-5.3-flash`` ->
    ``opencode-go/glm-5.3-flash``). A model that already names a provider
    (contains ``/``) is passed through unchanged.

    ``variant``: opencode's provider-specific reasoning-effort variant
    (``--variant``); empty means the model default.

    ``venv_source`` / ``worker_env``: same semantics as ``ClaudeCodeConfig``'s
    fields of the same name (``worker_env`` is merged after env sanitisation,
    so operator values win).
    """

    command: Annotated[
        tuple[str, ...], CommandTemplate({"prompt_path", "issue_number", "branch"})
    ] = ()
    provider: Annotated[str, Typed] = "opencode-go"
    variant: Annotated[str, Typed] = ""
    venv_source: str | None = None
    worker_env: Annotated[dict[str, str], Coerced] = field(default_factory=dict)
