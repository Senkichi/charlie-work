"""Reviewer env-pin and bash-permission policy for opencode_worker.

Split out of test_opencode_worker.py (file-size ratchet): the
worker_env_for/review-pin tests, the _oc_match Wildcard harness, and
the bash deny-matrix live here.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from charlie_work.env_sanitize import build_worker_env
from charlie_work.opencode_worker import _REVIEWER_CONFIG, worker_env_for

SECRET = "sk-oc-secret-key-98765"
OTHER_SECRET = "oauth-refresh-OTHER-PROVIDER-4321"
GO_ENTRY = {"type": "api", "key": SECRET}
AUTH_JSON = json.dumps({"opencode-go": GO_ENTRY, "anthropic": {"refresh": OTHER_SECRET}})


@pytest.fixture(autouse=True)
def _isolate_host_auth(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the host-auth lookup at an empty dir unless a test seeds it."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "host-data"))
    monkeypatch.delenv("OPENCODE_AUTH_CONTENT", raising=False)


def _seed_host_auth(tmp_path: Path, text: str = AUTH_JSON) -> None:
    auth = tmp_path / "host-data" / "opencode" / "auth.json"
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text(text, encoding="utf-8")


# --- worker_env_for ---------------------------------------------------------


def test_worker_env_for_posture_and_per_issue_data_dir(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    empty = {"XDG_DATA_HOME": str(tmp_path / "empty-host")}
    env = worker_env_for(7, sessions, environ=empty)
    assert env["XDG_DATA_HOME"] == str(sessions / "opencode-data" / "issue-7")
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] == "1"
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["permission"] == {"*": "allow"}
    assert config["autoupdate"] is False
    assert worker_env_for(8, sessions, environ=empty)["XDG_DATA_HOME"] != env["XDG_DATA_HOME"]
    assert "OPENCODE_AUTH_CONTENT" not in env  # no host login present


def test_worker_env_for_forwards_host_auth_json(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    env = worker_env_for(1, tmp_path / "s", environ={"XDG_DATA_HOME": str(tmp_path / "host-data")})
    # Only the selected provider's entry is forwarded, never the whole host file.
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in env["OPENCODE_AUTH_CONTENT"]


def test_worker_env_for_scopes_auth_to_requested_provider(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    environ = {"XDG_DATA_HOME": str(tmp_path / "host-data")}
    env = worker_env_for(1, tmp_path / "s", environ=environ, provider="anthropic")
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"anthropic": {"refresh": OTHER_SECRET}}
    assert SECRET not in env["OPENCODE_AUTH_CONTENT"]


def test_worker_env_for_forwards_nothing_for_absent_provider(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    environ = {"XDG_DATA_HOME": str(tmp_path / "host-data")}
    env = worker_env_for(1, tmp_path / "s", environ=environ, provider="not-logged-in")
    assert "OPENCODE_AUTH_CONTENT" not in env


@pytest.mark.parametrize("text", ["not json", "[1, 2]", ""])
def test_worker_env_for_forwards_nothing_for_invalid_auth_json(tmp_path: Path, text: str) -> None:
    _seed_host_auth(tmp_path, text)
    environ = {"XDG_DATA_HOME": str(tmp_path / "host-data")}
    assert "OPENCODE_AUTH_CONTENT" not in worker_env_for(1, tmp_path / "s", environ=environ)


def test_worker_env_for_skips_disk_read_when_auth_already_in_environ(tmp_path: Path) -> None:
    _seed_host_auth(tmp_path)
    env = worker_env_for(
        1,
        tmp_path / "s",
        environ={
            "XDG_DATA_HOME": str(tmp_path / "host-data"),
            "OPENCODE_AUTH_CONTENT": json.dumps(
                {"opencode-go": GO_ENTRY, "anthropic": {"refresh": OTHER_SECRET}}
            ),
        },
    )
    # An ambient OPENCODE_AUTH_CONTENT is the documented operator injection
    # point: it is parsed and SCOPED to the selected provider (the other
    # providers' entries cannot ride into the child env) instead of being
    # dropped -- an operator who authenticates through it stays authenticated.
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"opencode-go": GO_ENTRY}
    assert OTHER_SECRET not in env["OPENCODE_AUTH_CONTENT"]


def test_worker_env_for_operator_env_overrides_posture(tmp_path: Path) -> None:
    env = worker_env_for(
        1,
        tmp_path / "s",
        {"OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "0", "XDG_DATA_HOME": "/custom"},
        environ={"XDG_DATA_HOME": str(tmp_path / "empty-host")},
    )
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_PROMPT"] == "0"
    assert env["XDG_DATA_HOME"] == "/custom"


# --- review posture ---------------------------------------------------------


def test_worker_env_for_review_denies_writes_and_allowlists_read_bash(
    tmp_path: Path,
) -> None:
    env = worker_env_for(
        3, tmp_path / "s", environ={"XDG_DATA_HOME": str(tmp_path / "h")}, review=True
    )
    permission = json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]
    assert permission["*"] == "deny"
    assert permission["edit"] == "deny"
    assert permission["read"] == "allow"
    # Egress is closed: the claude-code reviewer has no web tools, the packet
    # already carries the diff, and read/external_directory stay allow (the
    # per-PR packet path) -- without this deny a prompt injection could read
    # host files and POST them out (verified live against a local listener).
    assert "webfetch" not in permission
    assert "websearch" not in permission
    bash = permission["bash"]
    assert bash["*"] == "deny"
    assert bash["git diff *"] == "allow"
    assert bash["gh pr view *"] == "allow"
    # The paired write verbs stay denied: no git push/commit or gh pr merge/edit.
    allowed = [pattern for pattern, effect in bash.items() if effect == "allow"]
    assert not any("push" in p or "commit" in p or "merge" in p or "edit" in p for p in allowed)


def test_worker_env_for_review_deny_policy_cannot_be_overridden(tmp_path: Path) -> None:
    """The reviewer's read-only invariant wins over operator worker_env --
    the OPENCODE_CONFIG_CONTENT counterpart of the --permission-mode plan pin."""
    hostile = json.dumps({"permission": {"*": "allow"}})
    env = worker_env_for(
        3,
        tmp_path / "s",
        {"OPENCODE_CONFIG_CONTENT": hostile},
        environ={"XDG_DATA_HOME": str(tmp_path / "h")},
        review=True,
    )
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["*"] == "deny"


def test_worker_env_for_review_pins_every_opencode_control_env(tmp_path: Path) -> None:
    """Every OPENCODE_* control surface is pinned after the worker_env merge:
    opencode merges OPENCODE_PERMISSION after OPENCODE_CONFIG_CONTENT, and the
    review checkout's .opencode/ dir, plugin list, or an ambient/extra config
    file could otherwise override the deny policy or run code at startup."""
    hostile_permission = json.dumps({"*": "allow"})
    sessions_dir = tmp_path / "s"
    env = worker_env_for(
        3,
        sessions_dir,
        {
            "OPENCODE_PERMISSION": hostile_permission,
            "OPENCODE_CONFIG": "/evil/opencode.json",
            "OPENCODE_CONFIG_DIR": "/evil/.opencode",
            "OPENCODE_DISABLE_PROJECT_CONFIG": "0",
            # The 2.x alias takes precedence over the legacy name (verified
            # in the 2.3.0 bundle: CONFIG_PROJECT_DISABLE ?? DISABLE_PROJECT_CONFIG)
            # and re-enabled the PR's project config live when left unpinned.
            "OPENCODE_CONFIG_PROJECT_DISABLE": "0",
            "OPENCODE_CLI_CONFIG_CONTENT": json.dumps({"permission": {"*": "allow"}}),
            "OPENCODE_MODELS_URL": "https://evil.example/models",
            "OPENCODE_MODELS_PATH": "/evil/models.json",
            "OPENCODE_LOG_LEVEL": "debug",
            "OPENCODE_PRINT_LOGS": "0",
            "OPENCODE_SIMULATE": "1",
            "OPENCODE_API_KEY": "EVIL-KEY",
            "OPENCODE_PURE": "0",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS": "0",
            "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "0",
            "OPENCODE_DB": "/evil/opencode.db",
            "OPENCODE_TEST_MANAGED_CONFIG_DIR": "/evil/managed",
            "OPENCODE_TEST_HOME": "/evil/home",
            "XDG_DATA_HOME": "/evil/xdg-data",
            "XDG_CONFIG_HOME": "/evil/xdg-config",
            "XDG_STATE_HOME": "/evil/xdg-state",
            "XDG_CACHE_HOME": "/evil/xdg-cache",
        },
        environ={
            "XDG_DATA_HOME": str(tmp_path / "h"),
            # Ambient os.environ values flow through the sanitized base env --
            # they must lose to the pins too, not just hostile worker_env.
            "OPENCODE_PERMISSION": hostile_permission,
        },
        review=True,
    )
    # Pinned to the same deny policy the config content carries (pin, not
    # strip: deleting it would leave whatever ambient value survives).
    assert json.loads(env["OPENCODE_PERMISSION"])["*"] == "deny"
    assert json.loads(env["OPENCODE_PERMISSION"])["edit"] == "deny"
    # The untrusted review checkout's project config, agents, plugins and
    # .claude/.agents skills never load; external plugins are off entirely.
    # BOTH project-config spellings are pinned: the 2.x alias wins when set.
    assert env["OPENCODE_CONFIG_PROJECT_DISABLE"] == "1"
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_PURE"] == "1"
    assert env["OPENCODE_DISABLE_EXTERNAL_SKILLS"] == "1"
    # The 2.x claude-code import gate: base pins carry OPENCODE_DISABLE_
    # CLAUDE_CODE, the reviewer additionally pins the skills switch.
    assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] == "1"
    # Extra config source neutralized (empty == falsy in opencode's loader).
    assert env["OPENCODE_CONFIG"] == ""
    # The inline CLI-config channel is pinned to the empty document.
    assert env["OPENCODE_CLI_CONFIG_CONTENT"] == "{}"
    # Path-shaped surfaces must be REAL empty dirs, never "": the binary's
    # ?? / || fallbacks turn "" into a cwd-relative or real-host-dir path
    # (verified against the 1.18.35 binary; apply_review_env_pins now also
    # creates the pinned directories so the "real dir" claim holds on disk,
    # see the dedicated pin test).
    data_dir = sessions_dir / "opencode-data" / "issue-3"
    assert env["OPENCODE_CONFIG_DIR"] == str(data_dir / "xdg" / "config" / "opencode")
    assert env["OPENCODE_TEST_MANAGED_CONFIG_DIR"] == str(data_dir / "empty-managed-config")
    assert env["OPENCODE_TEST_HOME"] == str(data_dir / "empty-home")
    # DB + XDG roots re-pinned after the worker_env merge: XDG_CONFIG_HOME is
    # the global-config dir the agent.*.permission override vector reads, and
    # a redirected OPENCODE_DB / XDG_DATA_HOME would relocate the reviewer's
    # DB off the per-issue dir.
    assert env["XDG_DATA_HOME"] == str(data_dir)
    assert env["XDG_CONFIG_HOME"] == str(data_dir / "xdg" / "config")
    assert env["XDG_STATE_HOME"] == str(data_dir / "xdg" / "state")
    assert env["XDG_CACHE_HOME"] == str(data_dir / "xdg" / "cache")
    # Pinned to the CONCRETE per-issue path, not "": on 2.3.0 an empty
    # OPENCODE_DB reaches bun:sqlite as the literal path and kills the
    # standalone server before readiness (runtime-verified; "" was
    # truthy-gated back to the default on 1.18.35).
    assert env["OPENCODE_DB"] == str(data_dir / "opencode.db")
    # 2.x control surfaces that are stripped outright (no safe neutral value):
    # model-metadata redirection, logging re-widening, simulation, and the
    # host credential channels (the reviewer authenticates through the
    # seeded per-issue auth.json only).
    for stripped in (
        "OPENCODE_MODELS_URL",
        "OPENCODE_MODELS_PATH",
        "OPENCODE_LOG_LEVEL",
        "OPENCODE_PRINT_LOGS",
        "OPENCODE_SIMULATE",
        "OPENCODE_API_KEY",
    ):
        assert stripped not in env, stripped


def test_worker_env_for_review_strips_case_variant_pin_twins(tmp_path: Path) -> None:
    """Windows env lookup is case-insensitive and CreateProcessW resolves
    duplicate names last-wins: a lowercase/mixed-case worker_env key appended
    after an uppercase pin slot would silently hand the hostile value to the
    spawned child. Every case-variant of a pinned name must be REMOVED from
    the dict, not merely out-valued."""
    hostile_permission = json.dumps({"*": "allow"})
    env = worker_env_for(
        3,
        tmp_path / "s",
        {
            "opencode_config_content": json.dumps({"permission": {"*": "allow"}}),
            "Opencode_Permission": hostile_permission,
            "xdg_data_home": "C:/evil/xdg-data",
            "xdg_config_home": "C:/evil/xdg-config",
            "opencode_test_home": "C:/evil/home",
            "opencode_db": "C:/evil/opencode.db",
        },
        environ={"XDG_DATA_HOME": str(tmp_path / "h")},
        review=True,
    )
    # The lowercase twins are gone entirely -- the pinned uppercase keys are
    # the only spellings present, so nothing sits later in the env block.
    for pinned in (
        "OPENCODE_CONFIG_CONTENT",
        "OPENCODE_PERMISSION",
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "OPENCODE_TEST_HOME",
        "OPENCODE_DB",
    ):
        assert [key for key in env if key.upper() == pinned] == [pinned]
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"])["permission"]["*"] == "deny"
    assert json.loads(env["OPENCODE_PERMISSION"])["*"] == "deny"
    data_dir = tmp_path / "s" / "opencode-data" / "issue-3"
    assert env["XDG_DATA_HOME"] == str(data_dir)
    assert env["XDG_CONFIG_HOME"] == str(data_dir / "xdg" / "config")


def test_worker_env_for_review_creates_the_pinned_config_dir(tmp_path: Path) -> None:
    """The pinned OPENCODE_CONFIG_DIR must be a real, empty directory.

    Nothing creates it implicitly at launch: opencode only loads a config
    DIRECTORY that exists, and the empty-string trap documented on the pin
    (Instruction resolves AGENTS.md against the checkout) is the reason it is
    a path rather than "". A regression that leaves the path dangling makes
    the doctor's provider-visibility message true for the wrong reason and
    hides an operator's custom providers silently -- so assert the directory
    exists and is empty, not just that the env names it.
    """
    env = worker_env_for(3, tmp_path / "s", None, environ={}, review=True)
    config_dir = Path(env["OPENCODE_CONFIG_DIR"])
    assert config_dir.is_dir()
    assert list(config_dir.iterdir()) == []
    # It lives under the per-issue data dir, so the reap removes it with the
    # session (never a shared host path).
    assert config_dir.is_relative_to(Path(env["XDG_DATA_HOME"]))


def test_worker_env_for_review_repins_auth_content(tmp_path: Path) -> None:
    """OPENCODE_AUTH_CONTENT is re-pinned on review=True: a 1.x binary's
    Auth.all() reads it directly, a "wellknown" entry makes the child fetch
    remote config whose agent.*.permission merges after OPENCODE_PERMISSION,
    and an ambient value leaks every host credential into the child env.
    Operator-supplied values are parsed and SCOPED to the selected provider,
    not dropped."""
    _seed_host_auth(tmp_path)
    # 1. Hostile worker_env value for the selected provider is scoped to
    #    exactly the operator's declared entry (an operator value is
    #    authoritative -- here it deliberately replaces the host file).
    env = worker_env_for(
        3,
        tmp_path / "s",
        {"OPENCODE_AUTH_CONTENT": json.dumps({"opencode-go": {"api": "EVIL"}})},
        environ={"XDG_DATA_HOME": str(tmp_path / "host-data")},
        review=True,
    )
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"opencode-go": {"api": "EVIL"}}
    # 2. A case-variant twin is stripped before pinning.
    env = worker_env_for(
        3,
        tmp_path / "s",
        {"opencode_auth_content": json.dumps({"opencode-go": {"api": "EVIL"}})},
        environ={"XDG_DATA_HOME": str(tmp_path / "host-data")},
        review=True,
    )
    assert [k for k in env if k.upper() == "OPENCODE_AUTH_CONTENT"] == ["OPENCODE_AUTH_CONTENT"]
    # 3. An ambient value naming a DIFFERENT provider scopes to the empty
    #    map: the wellknown remote-config vector is contained, and the
    #    review pin overrides the hostile ambient value downstream in
    #    build_worker_env's merge.
    env = worker_env_for(
        3,
        tmp_path / "s",
        None,
        environ={
            "XDG_DATA_HOME": str(tmp_path / "host-data"),
            "OPENCODE_AUTH_CONTENT": json.dumps(
                {"evil": {"type": "wellknown", "url": "https://evil.example"}}
            ),
        },
        review=True,
    )
    assert env["OPENCODE_AUTH_CONTENT"] == "{}"
    # 4. A well-scoped AMBIENT value survives the review pin as the scoped
    #    credential (the silent blanking an earlier review flagged: an
    #    operator authenticating through the documented env channel gets an
    #    authenticated reviewer).
    env = worker_env_for(
        3,
        tmp_path / "s",
        None,
        environ={
            "XDG_DATA_HOME": str(tmp_path / "no-auth-here"),
            "OPENCODE_AUTH_CONTENT": json.dumps({"opencode-go": GO_ENTRY}),
        },
        review=True,
    )
    assert json.loads(env["OPENCODE_AUTH_CONTENT"]) == {"opencode-go": GO_ENTRY}
    # 5. No credential anywhere (worker_env blank, no ambient value, empty
    #    host dir) -> pinned to an empty map, never left absent for a later
    #    layer to smuggle in.
    env = worker_env_for(
        3,
        tmp_path / "s",
        None,
        environ={"XDG_DATA_HOME": str(tmp_path / "no-auth-here")},
        review=True,
    )
    assert env["OPENCODE_AUTH_CONTENT"] == "{}"


def test_worker_env_for_review_strips_exec_env_vectors(tmp_path: Path) -> None:
    """worker_env exec vectors that ride ALLOWED verbs are removed outright:
    pager/editor hooks, repo-discovery redirection, inline git config pairs,
    git trace writes, python/uv hooks, and addopts/plugin env injection."""
    env = worker_env_for(
        3,
        tmp_path / "s",
        {
            "GIT_EDITOR": "sh -c 'touch pwned'",
            "GIT_PAGER": "sh",
            "GIT_EXTERNAL_DIFF": "evil",
            "GIT_DIR": "/evil/repo",
            "GIT_WORK_TREE": "/evil/tree",
            "GIT_INDEX_FILE": "/evil/index",
            "GIT_CONFIG_KEY_0": "alias.st=!sh -c id",
            "GIT_CONFIG_VALUE_0": "x",
            "GIT_CONFIG_COUNT": "1",
            "GIT_TRACE": "/evil/trace.log",
            "GIT_TRACE2_EVENT": "/evil/trace.json",
            "SHELL": "/bin/evil",
            "BASH_ENV": "/evil/env.sh",
            "ENV": "/evil/env.sh",
            "PYTHONPATH": "/evil",
            "PYTHONSTARTUP": "/evil/s.py",
            "NODE_OPTIONS": "--require /evil",
            "NODE_PATH": "/evil",
            "LD_PRELOAD": "/evil.so",
            "UV_INDEX_URL": "https://evil.example/simple",
            "UV_INDEX": "https://evil.example/simple",
            "UV_PYTHON": "/evil/python",
            "UV_PROJECT": "/evil/project",
            "UV_CONFIG_FILE": "/evil/uv.toml",
            "PYTHONUSERBASE": "/evil/userbase",
            "GH_CONFIG_DIR": "/real/gh",
            "GH_HOST": "evil.example",
            "GH_HOSTS_FILE": "/evil/hosts.yml",
            "GH_PAGER": "evil-pager",
            "GH_FORCE_TTY": "1",
            "GH_BROWSER": "evil-browser",
            "BROWSER": "evil-browser",
            "GH_EDITOR": "evil-editor",
            "LESSOPEN": "|evil",
            "LESSCLOSE": "evil %s",
            "LESSSECURE": "0",
            "LESSEDIT": "evil %s",
            "OPENCODE_GIT_BASH_PATH": "C:/evil/bash.exe",
            "COMSPEC": "C:/evil/cmd.exe",
            "PYTEST_ADDOPTS": "-p evilplugin",
            "PYTEST_PLUGINS": "evil.module",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "0",
            "INNOCENT_VAR": "kept",
        },
        environ={"XDG_DATA_HOME": str(tmp_path / "h")},
        review=True,
    )
    for key in env:
        assert "evil" not in str(env[key]).lower(), key
    for stripped in (
        "GIT_EDITOR",
        "GIT_PAGER",
        "GIT_EXTERNAL_DIFF",
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GIT_TRACE",
        "GIT_TRACE2_EVENT",
        "SHELL",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "NODE_OPTIONS",
        "NODE_PATH",
        "LD_PRELOAD",
        "UV_INDEX_URL",
        "UV_INDEX",
        "UV_PYTHON",
        "UV_PROJECT",
        "UV_CONFIG_FILE",
        "GH_CONFIG_DIR",
        "GH_HOST",
        "GH_HOSTS_FILE",
        "GH_PAGER",
        "GH_FORCE_TTY",
        "GH_BROWSER",
        "BROWSER",
        "GH_EDITOR",
        "LESSOPEN",
        "LESSCLOSE",
        "LESSSECURE",
        "LESSEDIT",
        "OPENCODE_GIT_BASH_PATH",
        "COMSPEC",
    ):
        assert stripped not in env, stripped
    # These are pinned to empty values (off), not stripped.
    assert env["GIT_CONFIG_COUNT"] == "0"
    assert env["GIT_CONFIG_PARAMETERS"] == ""
    assert env["PYTEST_ADDOPTS"] == ""
    assert env["PYTEST_PLUGINS"] == ""
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["INNOCENT_VAR"] == "kept"
    # BASH_ENV/ENV are stripped from the caller value and RE-PINNED to the
    # orchestrator-written GLOBIGNORE startup file (see the dedicated test).
    assert env["BASH_ENV"] == str(tmp_path / "s" / "opencode-data" / "issue-3" / "bash-startup.sh")
    assert env["ENV"] == env["BASH_ENV"]


def test_worker_env_for_review_pins_bash_startup_globignore(tmp_path: Path) -> None:
    """The GLOBIGNORE closure is delivered by a pinned BASH_ENV/ENV startup
    file, not by an env var: Git Bash 5.2 ignores an env-inherited
    GLOBIGNORE (verified -- the variable is present and exported yet
    pathname expansion still matches dash-leading names) and only honors it
    when it is assigned inside bash. The pinned file drops every
    dash-leading name from glob expansion (the only tokens a shell can turn
    into a flag) and restores the default dotfile behavior that a non-null
    GLOBIGNORE would otherwise disable."""
    sessions = tmp_path / "s"
    data_dir = sessions / "opencode-data" / "issue-3"
    env = worker_env_for(
        3,
        sessions,
        {"BASH_ENV": "/evil/env.sh", "env": "/evil/env2.sh"},
        environ={"XDG_DATA_HOME": str(tmp_path / "h")},
        review=True,
    )
    startup = data_dir / "bash-startup.sh"
    assert env["BASH_ENV"] == str(startup)
    assert env["ENV"] == str(startup)
    # Case twins of the pinned names are gone: the pinned spellings are the
    # only ones present, so nothing sits later in the child's env block.
    assert [k for k in env if k.upper() == "BASH_ENV"] == ["BASH_ENV"]
    assert [k for k in env if k.upper() == "ENV"] == ["ENV"]
    # The GLOBIGNORE assignment lives in the sourced file, not the env.
    assert [k for k in env if k.upper() == "GLOBIGNORE"] == []
    assert startup.read_text(encoding="ascii") == "GLOBIGNORE='-*:.*'\n"
    # The other path-shaped pins are real directories now, not just strings.
    assert (data_dir / "empty-home").is_dir()
    assert (data_dir / "empty-managed-config").is_dir()
    assert (data_dir / "xdg" / "config").is_dir()
    # A non-review launch does not carry the pins.
    worker_env = worker_env_for(4, sessions, environ={"XDG_DATA_HOME": str(tmp_path / "h")})
    assert "BASH_ENV" not in worker_env
    assert "ENV" not in worker_env
    assert not (sessions / "opencode-data" / "issue-4" / "bash-startup.sh").exists()


def test_build_worker_env_dedupes_case_variants_on_windows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Windows build_worker_env leaves exactly one spelling per env name: a
    lowercase twin appended after an uppercase pin would otherwise win the
    spawned child's env-block lookup. The last layer's value and casing win.

    On POSIX the collapse must NOT run: there a case variant is a different
    variable name, so deleting the canonical spelling would silently remove
    PATH/PYTHONPATH/... from the child while keeping a name nothing reads.
    """
    base = {"OPENCODE_PERMISSION": "ambient", "PATH": "C:/bin"}
    layer = {"opencode_permission": '{"*":"allow"}', "PATH": "C:/evil"}

    monkeypatch.setattr("charlie_work.env_sanitize._collapses_env_case", lambda: True)
    merged = build_worker_env(base, layer)
    assert merged["opencode_permission"] == '{"*":"allow"}'
    assert "OPENCODE_PERMISSION" not in merged
    assert [k for k in merged if k.upper() == "PATH"] == ["PATH"]
    assert merged["PATH"] == "C:/evil"

    # POSIX: both spellings survive, and the canonical one is untouched.
    monkeypatch.setattr("charlie_work.env_sanitize._collapses_env_case", lambda: False)
    posix = build_worker_env({"PYTHONPATH": "/safe", "PATH": "/usr/bin"}, {"pythonpath": "/evil"})
    assert posix["PYTHONPATH"] == "/safe"
    assert posix["pythonpath"] == "/evil"
    assert posix["PATH"] == "/usr/bin"


def _oc_match(command: str, pattern: str) -> bool:
    """opencode's Wildcard.match: ^pattern$ anchored, * -> .*, ? -> ., case-insensitive.

    Mirrors the binary: both sides normalize backslashes to '/', regex
    metachars are escaped, and a pattern ending in " *" is rewritten to
    "( .*)?" so bare verbs like "pytest" also match the "pytest *" wildcard.

    Verification status per binary version: the escape set
    (``[.+^${}()|[\\]\\\\]``), the ``*``->``.*`` / ``?``->``.`` rewrites and the
    ``( .*)?`` tail rewrite were re-confirmed against the installed 2.3.0
    binary's embedded bundle (byte offset 128141428); the ``\\Z``-vs-``$``
    anchor and case-insensitivity were verified against 1.18.35 and could
    not be re-extracted from 2.3.0's bytecode. Worst case on drift is
    over-deny (a command the harness predicts "allow" that the binary
    denies), never an uncaught write.
    """
    actual = command.replace("\\", "/")
    # JS escapes only [.+^${}()|[\]\\] -- NOT space or * -- so re.escape is
    # wrong here (it would escape the space and break the " .*" rewrite).
    regex = re.sub(r"([.+^${}()|[\]\\])", r"\\\1", pattern.replace("\\", "/"))
    regex = regex.replace("*", ".*").replace("?", ".")
    if regex.endswith(" .*"):
        regex = regex[:-3] + "( .*)?"
    # \Z, not $: Python's $ also matches before a trailing \n; JS RegExp "$"
    # does not. Exact parity with Wildcard.match matters for adversarial
    # inputs like "git status\n".
    return re.match(f"^{regex}\\Z", actual, re.IGNORECASE | re.DOTALL) is not None


def _oc_bash_effect(bash: dict[str, str], command: str) -> str:
    """findLast semantics: the last matching rule decides."""
    effect = "ask"
    for pattern, action in bash.items():
        if _oc_match(command, pattern):
            effect = action
    return effect


def test_review_bash_trailing_denies_close_wildcard_write_escapes() -> None:
    """The "verb *" allows are prefix-anchored; a command like "git diff > f"
    still matches "git diff *" (opencode checks the whole redirected_statement
    text), so the trailing deny patterns must win under findLast evaluation.

    Wildcard.match sees the UNEXPANDED command text, so the denies also have
    to cover every spelling an attacker can produce that the shell will later
    expand into the denied flag: quote/backtick/backslash splitting inside a
    flag word, unicode-whitespace separators, and git's long-option prefix
    abbreviations. The metacharacter deny layer is what makes the flag-name
    patterns trustworthy at all.

    No test runner is allowed (the code-execution posture: the checkout is the
    PR's code, so a test run imports the PR's conftest.py). Every pytest
    spelling below -- including the ones this policy used to re-allow -- must
    therefore deny, which also keeps prompt-injected "run the tests" text
    inert.
    """
    bash = _REVIEWER_CONFIG["permission"]["bash"]
    # The structural invariant the layer design relies on: after the LAST
    # "allow" every entry is "deny". An "allow" placed inside or after the
    # deny layers beats every earlier deny for any command its wildcard
    # happens to match -- the shape that let "*--log-failed*" resurrect
    # rm/git-push/gh-merge and the old pytest re-allows resurrect
    # redirection/command-substitution/quoted flags. Legitimate lookalikes are
    # handled by NARROWING the deny to the verb that owns the dangerous flag
    # ("*git *--ex*"), never by re-allowing after it.
    last_allow = max(i for i, v in enumerate(bash.values()) if v == "allow")
    assert all(v == "deny" for v in list(bash.values())[last_allow + 1 :])
    # Nothing on the allowlist may execute PR code: no test runner, no
    # interpreter, no package manager.
    assert not any(
        _oc_bash_effect(bash, command) == "allow"
        for command in (
            "pytest",
            "pytest -q --tb=short",
            "python -m pytest",
            "uv run pytest",
            "uv run --extra dev pytest tests/x.py",
            "python -c print(1)",
            "uv run --version",
            "npm test",
            "sh -c id",
        )
    )

    for attack in (
        # Write/exec escapes on allowed verbs (whole-statement prefix match).
        "git diff > hacked.txt",
        "git diff --output=hacked.txt",
        "git diff --output hacked.txt",
        "git diff --out hacked.txt",  # unambiguous prefix abbrev of --output
        "git diff --output-indicator=x",
        "git log --output=../evil",
        "git show HEAD --output=/tmp/x",
        "git diff --ext-diff",
        "git diff --ext",
        "git log --exec=x",
        "git log --ex=x",  # prefix abbrev of --exec
        "gh pr view 1 > notes.md",
        "gh pr diff 1 > /tmp/pr.diff",
        "pytest tests --junitxml=C:/evil/x.xml",
        "pytest --junitxml x.xml",
        "pytest --junit-xml=C:/evil/x.xml",  # hyphenated alias
        "pytest --junit-xml x.xml",
        "pytest --junitx x.xml",  # prefix abbrev
        "pytest --junit x.xml",
        "pytest --log-file=C:/evil/x.log",
        "pytest --log-file x.log",
        "pytest --cache-clear",
        "pytest --cache-show",
        "pytest --basetemp=C:/evil",
        "pytest --ba=C:/evil",  # prefix abbrev of --basetemp
        "pytest --debug",
        "pytest --debug=/tmp/pytestdebug.log",
        "pytest --deb",  # prefix abbrev of --debug
        "pytest --pyargs=evil.module",  # imports arbitrary installed module
        "pytest --pyargs evil.module",
        # -o / --override-ini ini injection, every spelling.
        "pytest -o addopts=--junitxml=/tmp/x",
        "pytest -oaddopts=--cache-clear",
        "pytest -o=addopts=x",
        "pytest --override-ini addopts=--junitxml=/tmp/x",
        "pytest --override-ini=addopts=x",
        "pytest -o x=y",  # any ini override, not just addopts
        # pytest bundled short flags: argparse packs no-arg shorts before a
        # value flag at unbounded depth, so every pytest " -" is denied --
        # including deep and repeat bundles a "?"-depth pattern could miss.
        "pytest -qo addopts=foo",
        "pytest -xo addopts=foo",
        "pytest -vo addopts=foo",
        "pytest -so addopts=foo",
        "pytest -qxvo addopts=foo",
        "pytest -qxlNvo addopts=foo",
        "pytest -xqvslNo addopts=foo",
        "pytest -xxxxxxo addopts=foo",  # repeat chars, beyond any fixed depth
        "pytest -p evilplugin",  # imports a python module at startup
        "pytest -pevilplugin",
        "pytest -qp evilplugin",
        "pytest -xvp evilplugin",
        "pytest -c evil.ini",  # loads an arbitrary ini file
        "pytest -cevil.ini",
        "pytest -qc evil.ini",
        "pytest --confcutdir=/evil",
        # Every other pytest flag form dies too: the pinned -q --tb=short
        # invocation is the only flagged pytest allowed, and even its tail
        # is positional-only.
        "pytest -q",
        "pytest -x tests/test_x.py",
        "pytest -k filter_expr",
        "pytest -m slow",
        "pytest -n auto",
        "pytest -sv",
        "pytest -rx",
        "pytest --tb=line",
        "pytest --tb=long",
        "pytest --no-header",
        "pytest --deselect tests/x",
        "pytest --disable-warnings",
        "pytest --import-mode=importlib",
        "pytest --rootdir=.",
        "pytest tests/test_x.py -x",
        "pytest -q -k filter_expr",
        "python -m pytest -q",
        "uv run pytest -q",
        "uv run --extra dev pytest -q",
        # The pinned invocation accepts positionals only -- no further flags.
        "pytest -q --tb=short -x",
        "pytest -q --tb=short --co",
        "pytest -q --tb=short -m foo",
        "pytest -q --tb=short tests/ -x",
        "uv run --extra dev pytest -q --tb=short -k foo",
        # git branch --list takes pattern args only -- a trailing -d/-D is a
        # branch delete, not a list filter.
        "git branch --list -D feature-x",
        "git branch --list -d feature-x",
        "git branch --list main -D feature-x",
        "git ls-files -o",  # belt: --others is the allowed spelling
        # --textconv runs the repo-config diff.<driver>.textconv command.
        "git diff --textconv",
        "git log --textconv HEAD",
        "git show --textconv",
        # Documented collateral of the ordered-letter denies -- read-
        # only flags that share the denied letter surface. The reviewer has an
        # equivalent for each (--branches via --all, --exclude-standard via
        # the grep tool).
        "git log --exclude=x --all",
        "git ls-files --exclude-standard",
        # Whitespace other than ASCII space: pwsh tokenizes on unicode ws.
        "pytest -q\t-o addopts=x",
        "pytest -q\xa0-o addopts=x",
        "pytest -q -o addopts=x",
        "pytest -q　-o addopts=x",
        "git status\ngit push",
        # Shell quote/backtick/backslash splitting reconstructs the flag word
        # after matching, so the flag-name substring is never present.
        'pytest "-o" addopts=x',
        "pytest '-o' addopts=x",
        'git grep "-Osh -c id" pat',
        'git diff --out""put=hacked.txt',
        "git diff --out''put=hacked.txt",
        'git log --ex""ec=sh',
        "git diff --out`put=hacked.txt",  # pwsh backtick escape
        "git diff `--output=hacked.txt",
        # bash backslash escapes -- Wildcard.match normalizes \ -> / before
        # matching, so these land on the ordered-letter deny patterns.
        "git diff --out\\put=hacked.txt",
        "git diff --\\output=hacked.txt",
        "git diff --o\\utp\\ut=hacked.txt",
        "git diff --e\\xt-d\\iff",
        "git diff --ex\\ec=sh",
        "pytest --junit\\xml=x.xml",
        "pytest --log-f\\ile=x.log",
        "pytest --cache-\\clear",
        "pytest --b\\asetemp=/x",
        "pytest --d\\ebug",
        "pytest --py\\args=mod",
        "pytest --override\\-ini=x",
        "pytest -\\o addopts=x",
        "pytest -\\p mod",
        "pytest -\\c f.ini",
        "pytest -q\\o addopts=x",
        "git diff --o{u,t}put=x",  # brace expansion
        "git diff --$fput=x",  # variable expansion
        "git diff --${f}put=x",
        "git diff --$(printf out)put=x",  # command substitution
        # git grep flag surface: -O/--open* spawn a pager via sh -c; bundling
        # and prefix abbrev make them unmatchable by flag-name patterns, so
        # every " -" after git grep is denied.
        "git grep --open-files-in-pager='sh -c x' pat",
        "git grep --open-files-in-pager=evil pat",
        "git grep --open-files=evil pat",
        'git grep --open="sh -c id" pat',  # prefix abbrev of --open-files-*
        "git grep --open=echo pat",
        "git grep -O'sh -c \"touch pwned\"' pattern",
        "git grep -Oecho pat",
        "git grep -nOecho pat",  # bundled short flags
        "git grep -iOsh pat",
        "git grep -cOsh pat",
        "git grep -wOsh pat",
        "git grep -vOsh pat",
        "git grep -FOsh pat",
        "git grep -EOsh pat",
        "git grep -POsh pat",
        "git grep -HOsh pat",
        "git grep -hOsh pat",
        "git grep -lOsh pat",
        "git grep -n pat",  # harmless flag, denied with the exec surface
        "git grep pat -n",  # flag-after-pattern form
        "git grep pat -Osh",
        "git grep pat -- README",  # even the -- path separator goes
        "uv run --extra dev pytest -o addopts=x",
        "uv run pytest -qp evilplugin",
        "python -m pytest -qo addopts=x",
        # Statements outside the allowlist entirely.
        "git push origin HEAD",
        "git -c core.editor=sh rebase -i",
        "sh -c 'rm -rf .'",
        "rm -rf .",
        "git commit -m x",
        "gh pr merge 1",
        "git status && git push",
        "git status; rm x",
        "git diff | sh",
        "git diff & evil",
        "git status !x",
        "git diff <(evil)",
        "git diff >(evil)",
        "git show HEAD~1!",
        # gh *view --web spawns GH_BROWSER/BROWSER -- process exec off a
        # read-only verb.
        "gh pr view --web",
        "gh repo view --web",
        "gh run view 123 --web",
        "gh issue view 9 --web",
        # A trailing --log-failed must never act as an allow-token: an
        # earlier unanchored re-allow beat every deny above it under
        # findLast. `gh run view --log-failed` stays allowed (it is a
        # read-only verb on an allowlisted allow), and every non-allowlisted
        # verb still dies at the catch-all.
        "rm -rf . --log-failed",
        "git push origin HEAD --log-failed",
        "git commit -m pwned --log-failed",
        "gh pr merge 1 --log-failed",
        'sh -c "touch pwned" --log-failed',
        "git diff --output=hacked.txt --log-failed",
        "git log --exec=sh --log-failed",
        "git grep -nOsh pat --log-failed",
        "git grep --open=sh pat --log-failed",
        "git branch --list -D main --log-failed",
        "gh run view 123 --log-failed > hacked.txt",
        # A pinned-invocation tail must never re-allow the Layer-1 vectors:
        # redirection, pipes, substitution, quoted flags, backtick escapes,
        # and tab-separated flags all stay denied under findLast.
        "pytest -q --tb=short > hacked.txt",
        "pytest -q --tb=short | sh",
        "pytest -q --tb=short $(touch pwned)",
        'pytest -q --tb=short "-o" addopts=x',
        'pytest -q --tb=short "-p" evilplugin',
        'pytest -q --tb=short "-c" evil.ini',
        'pytest -q --tb=short "--junitxml=C:/evil/x.xml"',
        'pytest -q --tb=short "--log-file=C:/evil/x.log"',
        "pytest -q --tb=short \t-o\taddopts=x",
        "pytest -q --tb=short `-o addopts=x",
        "uv run --extra dev pytest > hacked.txt",
        'uv run --extra dev pytest "-o" addopts=x',
        'uv run --extra dev pytest "-p" evilplugin',
        "uv run --extra dev pytest -q --tb=short > hacked.txt",
        "python -m pytest -q --tb=short | sh",
        # bash escapes the dash itself: \ -> / under Wildcard.match, but the
        # shell strips it to a real "-". ' /-' (escaped dash at a word
        # boundary) and '-/-' (a long flag's dashes split) are denied.
        "git diff -\\-output=hacked.txt",
        "git log -\\-output=hacked.txt",
        "git show HEAD -\\-output=hacked.txt",
        "git diff -\\-ext-diff",
        "git diff -\\-textconv",
        "git log -\\-exec=sh",
        "git grep \\-Osh pat",
        "git grep \\-\\-open-files-in-pager=sh pat",
        "git branch --list \\-D feature-x",
        "pytest \\-p evilplugin",
        "pytest \\-c evil.ini",
        "pytest \\-o console_output_style=classic",
        "pytest \\-\\-junitxml=C:/evil/x.xml",
        "pytest \\-\\-log-file=C:/evil/x.log",
        "pytest \\-\\-cache-clear",
        "pytest \\-\\-basetemp=C:/evil",
        "pytest \\-\\-debug",
        "pytest \\-\\-pyargs=evil.module",
        "pytest \\-\\-override-ini=x=y",
        "pytest -q --tb=short \\-o addopts=x",
        "pytest -q --tb=short \\-\\-junitxml=x.xml",
    ):
        assert _oc_bash_effect(bash, attack) == "deny", attack
    for allowed in (
        "git status",
        "git status --short",
        "git status -sb",  # bundled read-only shorts: git status is not flagged
        "git diff HEAD~1",
        "git diff --stat",
        "git diff HEAD~1..HEAD",
        "git diff -- file.py",
        "git log --oneline -20",
        "git log --pretty=format:%H",
        "git log --graph --all --decorate",
        "git log -p src/x.py",  # -p is git's patch flag; no pytest tail rule exists
        "git log -G foo",
        "git log -S foo",
        "git log --decorate --all",
        "git log --topo-order --name-only",  # regression: --?o*p*e*n* would have denied this
        "git show HEAD:file.py",
        "git blame file.py",
        "git blame -- file.py",
        "git grep foo",
        "git grep foo.py",  # flagless search still works
        "git ls-files -d",
        "git ls-files --deleted",
        "git ls-files --modified",
        "git ls-files --others",  # long form; the bare -o short is denied
        "git ls-files --cached",  # --cache-* deny must not catch --cached
        "git ls-files --stage",
        "git rev-parse HEAD",
        "git branch --show-current",
        "git branch --list feature/x",
        "git remote -v",
        "git remote show origin",
        "git describe --always",
        "gh pr view",
        "gh pr view 123 --json files",
        "gh pr view 123 --jq .body",  # unquoted jq path is plain words
        "gh pr diff 123",
        "gh pr checks 123",
        "gh issue view 9",
        "gh repo view owner/repo",
        "gh run list",
        "gh run view 123 --log",
        "gh run view 123 --log-failed",  # --log-file deny must not catch this
        "gh search issues flaky",
        # Read-only flags that the removed pytest-scoped flag denies used to
        # catch collaterally (--b*a, --junit*, --cache-*, --override-ini*).
        "git log --branches",
        "gh pr view --branch feature-x",
    ):
        assert _oc_bash_effect(bash, allowed) == "allow", allowed

    # Present in the DENY corpus above, restated here as the positive control
    # for the posture change: every form the policy formerly re-allowed (and
    # the flagless forms) is now denied, so a reviewer cannot execute PR code
    # even if a prompt injection asks it to "run the tests".
    for pytest_command in (
        "pytest",
        "pytest tests/test_x.py",
        "pytest tests/test_x.py::TestY::test_z",
        "pytest -q --tb=short",
        "pytest -q --tb=short tests/",
        "python -m pytest -q --tb=short",
        "uv run pytest -q --tb=short",
        "uv run --extra dev pytest -q --tb=short tests/test_x.py",
    ):
        assert _oc_bash_effect(bash, pytest_command) == "deny", pytest_command


# --- V2 permissions ruleset (opencode 2.x) -----------------------------------


def test_review_config_carries_v2_permissions_ruleset() -> None:
    """opencode 2.x's config schema has no top-level ``permission`` map
    field -- only ``permissions``, an ordered ``{action, resource, effect}``
    ruleset evaluated last-match (runtime-verified against 2.3.0 with
    ``opencode run --standalone``). The reviewer config therefore ships
    BOTH shapes: the V1 object (1.x binaries) and the flattened V2 list.

    Shape contract pinned here:
    - the V1 map's insertion order is preserved, with the nested ``bash``
      command-pattern map flattened under the 2.x action name ``shell``;
    - the V1 ``edit`` deny gains a sibling ``write`` deny (2.x split the
      tool; runtime-verified that the legacy ``edit`` deny does NOT cover
      2.x's write tool);
    - ``{"*","*","deny"}`` appears exactly once, LEADING the list: the same
      deny trailing it would withhold the shell tool entirely
      (runtime-verified), while leading it leaves the tool surfaced and
      denies unmatched commands per-command ("Permission denied: shell").
    """
    import json

    from charlie_work.opencode_review_policy import _v2_permission_rules

    rules = _v2_permission_rules(_REVIEWER_CONFIG["permission"])
    assert rules[0] == {"action": "*", "resource": "*", "effect": "deny"}
    # Exactly one resource-"*" deny, leading: a deeper one is the shape that
    # runtime-verified WITHHOLDS the shell tool on 2.3.0.
    star_denies = [r for r in rules if r["resource"] == "*" and r["effect"] == "deny"]
    assert star_denies == [
        {"action": "*", "resource": "*", "effect": "deny"},
        {"action": "edit", "resource": "*", "effect": "deny"},
        {"action": "write", "resource": "*", "effect": "deny"},
    ]
    # The per-tool denies (edit/write) keep their own resource-"*" rules --
    # withholding THOSE tools is the point -- but no later rule may be a
    # resource-"*" deny on the global or shell action: that is the shape
    # that runtime-verified WITHHOLDS the shell tool on 2.3.0.
    assert not any(
        rule["action"] in ("*", "shell") and rule["resource"] == "*" and rule["effect"] == "deny"
        for rule in rules[1:]
    )
    # bash map flattened under "shell", in map order, with the same effects.
    effects = [(rule["action"], rule["resource"], rule["effect"]) for rule in rules]
    assert effects.count(("shell", "git status", "allow")) == 1
    assert effects.count(("shell", "git status *", "allow")) == 1
    # 2.x write tool is denied by its own rule, not the legacy edit key.
    assert ("edit", "*", "deny") in effects
    assert ("write", "*", "deny") in effects
    # Read-side tools stay allows; the list carries no effect but the V1
    # object's values.
    assert {rule["effect"] for rule in rules} <= {"allow", "deny"}
    # The shipped config carries both shapes and serializes compactly.
    assert _REVIEWER_CONFIG["permissions"] == rules
    assert json.loads(json.dumps(_REVIEWER_CONFIG, separators=(",", ":")))["permissions"] == rules


def _git_bash() -> str | None:
    """Git Bash for the glob-closure integration test, else None (skip)."""
    import shutil

    candidate = Path(r"C:\Program Files\Git\bin\bash.exe")
    if os.name == "nt" and candidate.exists():
        return str(candidate)
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        return found
    return None


def test_pinned_bash_startup_file_kills_glob_flag_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end closure of the glob bypass (adversarial review round 2):
    an attacker-controlled checkout containing dash-leading files must not
    be able to turn the permitted ``pytest -q --tb=short *`` invocation into
    ``pytest -c <attacker.ini>`` (which imports an attacker module via
    addopts). With the pinned BASH_ENV startup file sourced, bash's
    pathname expansion must drop every dash-leading name, so the permission
    text and the executed argv agree that no flag is present."""
    bash = _git_bash()
    if bash is None:
        pytest.skip("Git Bash not available")
    import subprocess

    from charlie_work.opencode_review_policy import apply_review_env_pins
    from charlie_work.opencode_worker import _WORKER_ENV

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "-c").write_text("", encoding="ascii")
    (checkout / "0.ini").write_text("[pytest]\naddopts = -p0evil\n", encoding="ascii")
    pkg = checkout / "0evil"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("open('PWNED.txt', 'w').write('pwned')\n", encoding="ascii")
    (checkout / "conftest.py").write_text("", encoding="ascii")

    env: dict[str, str] = {
        **os.environ,
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
    }
    apply_review_env_pins(env, base_pins=_WORKER_ENV, data_dir=tmp_path / "data", scoped_auth="{}")
    script = (
        f'cd "{checkout.as_posix()}" && '
        'python -c "import sys; print(sys.argv[1:])" * ?p && '
        "python -m pytest -q --tb=short * ; "
        "test -f PWNED.txt && echo PWNED-EXECUTED || echo not-executed"
    )
    proc = subprocess.run(
        [bash, "-c", script], env=env, capture_output=True, text=True, timeout=60
    )
    argv_line = proc.stdout.strip().splitlines()
    # No dash-leading token may appear in the expanded argv, and the ?-glob
    # must pass through unexpanded (no match) rather than resolving to "-p".
    for line in argv_line:
        assert not line.lstrip("['").startswith("-"), proc.stdout
    assert "PWNED-EXECUTED" not in proc.stdout
    # Control: without the startup pin the expansion produces the flag and
    # the pytest plugin import runs -- this is the bug the pin closes.
    control = dict(env)
    control.pop("BASH_ENV", None)
    control.pop("ENV", None)
    proc2 = subprocess.run(
        [bash, "-c", script], env=control, capture_output=True, text=True, timeout=60
    )
    assert "PWNED-EXECUTED" in proc2.stdout, proc2.stdout
