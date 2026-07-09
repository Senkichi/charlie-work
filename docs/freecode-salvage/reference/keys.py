"""Provider API key storage — OS keyring only.

**v1.0 lockdown (R-D, principal decision Q-4).** The plaintext fallback path
that existed in v0.x (write the key to a JSON file under the app data dir)
is removed entirely. Two reasons:

1. The fallback wrote with default umask, surfaced no audit metadata, and was
   silently exercised on any keyring failure. That is a security primitive
   we cannot defend.
2. Every supported platform ships a keyring backend. If the user's
   environment lacks one (typically minimal headless Linux), the correct
   answer is a clear, actionable error — not "freecode silently weakened
   your security posture".

Public surface:

- `store_api_key(provider_id, api_key)` — writes to OS keyring or raises
  `KeyringUnavailable` with platform-specific remediation guidance.
- `get_api_key(provider_id)` — returns the stored key, or `None` if absent;
  emits a `security.key.accessed` decision event on success (provider id only,
  never the key bytes).
- `delete_api_key(provider_id)` — best-effort delete from OS keyring; never
  raises on absence.

There is no `--allow-plaintext-keys` flag, no `plaintext_keys.json` file, and
no parameter on this surface that re-enables the legacy path. Tests assert
this explicitly.
"""

from __future__ import annotations

import sys
from typing import cast

import keyring
import keyring.errors

from freecode.observability.events import DecisionEvent, emit_decision

SERVICE_NAME = "freecode"


class KeyringUnavailable(RuntimeError):
    """Raised when no usable OS keyring backend is available.

    The string carries platform-specific remediation guidance so the user
    sees a clear next step, not a stack trace. Caller (CLI) maps this to
    exit code `EXIT_KEYRING_UNAVAILABLE`.
    """


_PLATFORM_REMEDIATION: dict[str, str] = {
    "win32": (
        "Windows Credential Manager unavailable. Verify the 'Credential Manager' "
        "service is running. If you are on Windows Server Core, install a keyring "
        "backend (`pip install keyrings.alt` for an encrypted file backend) and "
        "configure it before retrying."
    ),
    "darwin": (
        "macOS Keychain unavailable. Run `security list-keychains` to confirm a "
        "keychain is present and unlocked. If running over SSH, log in to a GUI "
        "session first to unlock the login keychain."
    ),
    "linux": (
        "No Secret Service backend available. Install `gnome-keyring` (or any "
        "other Secret-Service-compatible daemon), or `pip install keyrings.alt` "
        "for an encrypted file backend. Headless servers can use `keyring-pass`."
    ),
}


def _platform_remediation() -> str:
    if sys.platform.startswith("linux"):
        return _PLATFORM_REMEDIATION["linux"]
    return _PLATFORM_REMEDIATION.get(sys.platform, _PLATFORM_REMEDIATION["linux"])


def _wrap_keyring_failure(operation: str, exc: BaseException) -> KeyringUnavailable:
    return KeyringUnavailable(
        f"freecode {operation} failed: {exc.__class__.__name__}: {exc}\n"
        f"Remediation: {_platform_remediation()}"
    )


def _win_vault_get_password(provider_id: str) -> str | None:
    """Windows-only: read credential with stable lookup order.

    :class:`keyring.backends.Windows.WinVaultKeyring` checks the bare
    ``TargetName=service`` first. When that row exists with the same
    ``UserName`` as requested, it returns immediately — even if a compound
    row ``{username}@{service}`` holds the real UTF-16 secret (common after
    Credential Manager edits). The bare row can then contain a stale one-byte
    blob while ``groq@freecode`` is correct. Prefer compound first, then bare,
    matching how credentials are displayed in Credential Manager.
    """
    try:
        from keyring.backends.Windows import DecodingCredential, WinVaultKeyring
    except ImportError:
        return None
    try:
        vault = WinVaultKeyring()  # type: ignore[no-untyped-call]
    except RuntimeError:
        return None
    compound = vault._compound_name(provider_id, SERVICE_NAME)  # type: ignore[no-untyped-call]
    for target in (compound, SERVICE_NAME):
        try:
            res = vault._read_credential(target)  # type: ignore[no-untyped-call]
        except Exception:
            continue
        if res is None or res.get("UserName") != provider_id:
            continue
        return cast(str | None, DecodingCredential(res).value)
    return None


# Tests monkeypatch this symbol so ``keyring.get_password`` mocks stay effective
# without disabling :func:`_win_vault_get_password` itself (see ``tests/conftest.py``).
_win_vault_reader = _win_vault_get_password


def store_api_key(provider_id: str, api_key: str) -> None:
    """Write `api_key` to the OS keyring for `provider_id`.

    Raises:
        KeyringUnavailable: if no usable backend is configured.
    """
    if not provider_id:
        raise ValueError("provider_id must be non-empty")
    if not api_key:
        raise ValueError("api_key must be non-empty")
    try:
        keyring.set_password(SERVICE_NAME, provider_id, api_key)
    except keyring.errors.KeyringError as exc:
        raise _wrap_keyring_failure("set_password", exc) from exc
    except Exception as exc:
        raise _wrap_keyring_failure("set_password", exc) from exc


def get_api_key(provider_id: str) -> str | None:
    """Return the stored API key for `provider_id`, or None if absent.

    Emits exactly one `security.key.accessed` decision event when a key is
    returned (with `provider_id` and `present=true`); emits nothing when
    the key is absent so we don't generate noise on routine misses.
    """
    if not provider_id:
        raise ValueError("provider_id must be non-empty")
    try:
        value: str | None = None
        if sys.platform == "win32":
            value = _win_vault_reader(provider_id)
        if value is None:
            value = keyring.get_password(SERVICE_NAME, provider_id)
    except keyring.errors.KeyringError as exc:
        raise _wrap_keyring_failure("get_password", exc) from exc
    except Exception as exc:
        raise _wrap_keyring_failure("get_password", exc) from exc
    if value is None:
        return None
    # CRLF/whitespace from paste → invalid Authorization header (some APIs return
    # HTTP 400 with empty body instead of 401).
    value = value.strip()
    if not value:
        return None
    emit_decision(
        DecisionEvent(
            "security.key.accessed",
            {"provider_id": provider_id, "present": True},
        )
    )
    return value


def delete_api_key(provider_id: str) -> None:
    """Best-effort delete from the OS keyring; silent on absence."""
    if not provider_id:
        raise ValueError("provider_id must be non-empty")
    try:
        keyring.delete_password(SERVICE_NAME, provider_id)
    except keyring.errors.PasswordDeleteError:
        return
    except keyring.errors.KeyringError as exc:
        raise _wrap_keyring_failure("delete_password", exc) from exc
    except Exception as exc:
        raise _wrap_keyring_failure("delete_password", exc) from exc
