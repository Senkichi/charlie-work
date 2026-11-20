"""Reviewer-only launch posture for opencode (`opencode_worker.worker_env_for`).

Everything that turns a *worker* env/config into a *reviewer* one lives here:
the read-only permission policy pinned via ``OPENCODE_CONFIG_CONTENT`` /
``OPENCODE_PERMISSION`` -- shipped in BOTH config shapes (the 1.x
``permission`` object and the 2.x ``permissions`` ruleset; see the version
notes above ``_REVIEWER_PERMISSION``) -- plus the exec/redirection env scrub
applied to the caller-supplied layer, and :func:`apply_review_env_pins`,
which re-pins every control surface opencode reads AFTER the operator
``worker_env`` merge (the findLast/env-collision reasons are documented
inline). ``opencode_worker`` re-imports ``_REVIEWER_CONFIG``/
``_AUTH_CONTENT_ENV`` so existing ``charlie_work.opencode_worker.<name>``
references keep resolving.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .atomic_write import write_text_atomic

log = logging.getLogger(__name__)

# Deny-character sets for the reviewer bash policy below: opencode's
# Permission.evaluate uses findLast (the LAST matching rule wins), so a
# wildcarded read verb ("git diff *") would otherwise admit quoting, escaping,
# expansion, chaining and redirection into a command the shell re-parses.
# Denying the metacharacters themselves closes that whole class: every command
# a legitimate reviewer runs is plain unquoted words.
_REVIEW_BASH_METACHARS = (
    "'",
    '"',
    "`",
    "$",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    "<",
    ">",
    "|",
    "&",
    ";",
    "!",
)

# Non-ASCII-space separators. pwsh/powershell tokenizes on unicode
# whitespace, so a \t- or NBSP-separated "-o" never matches " -o*".
_REVIEW_BASH_NON_ASCII_WS = (
    "\t",
    "\n",
    "\r",
    "\v",
    "\f",
    "\x85",
    "\xa0",
    "\u1680",
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
    "\ufeff",
    *[chr(0x2000 + i) for i in range(11)],
)

# Reviewer posture, shipped in BOTH config shapes opencode has used.
#
# Version history of the enforcement surface (all runtime-verified against a
# live ``opencode run --standalone --auto`` session on the installed binary,
# 2026-11-17):
#
# - opencode 1.18.35: top-level ``permission`` object (tool -> action, with a
#   nested ``bash`` command-pattern map), evaluated findLast (last matching
#   rule wins). ``bash`` was the tool name and ``edit`` covered edit/write.
# - opencode 2.3.0: the top-level schema field is ``permissions`` -- an
#   ORDERED list of ``{action, resource, effect}`` rules, also evaluated
#   last-match (runtime-verified: ``[shell "echo*" deny, shell "*" allow]``
#   ran ``echo``). The tool was renamed ``bash`` -> ``shell`` and gained a
#   separate ``write`` action; ``list``/``todowrite`` no longer exist. Two
#   runtime-verified traps shape the translation below:
#   * a trailing ``{"action":"*","resource":"*","effect":"deny"}`` rule
#     WITHHOLDS the shell tool entirely (last-match at tool registration),
#     while the same deny LEADING the list leaves the tool present and denies
#     unmatched commands per-command ("Permission denied: shell"). The V2
#     list therefore carries the catch-all deny FIRST and never repeats it
#     at the end.
#   * the legacy top-level ``edit`` deny does NOT cover 2.x's separate
#     ``write`` tool (runtime-verified: a file was created with
#     ``permission: {"edit": "deny"}``), so ``write`` is denied explicitly.
#   * specific command-pattern allows do not surface the shell tool when a
#     later ``*`` deny exists; only the leading-catch-all shape does.
#
# The V1 object stays for 1.x binaries; the V2 list stays for 2.x binaries.
# Both describe the same policy: ``edit``/``write`` deny, an exact-prefix
# allowlist of the read-only git/gh verbs the review packet names, and every
# other action defaults to deny. There is deliberately NO test-runner entry
# (see the code-execution note below). ``external_directory`` stays allow --
# the packet artifacts ($diff_path, $pr_json_path) live under the
# orchestrator's state ``prs/`` dir outside the review checkout. A ``deny``
# surfaces to the model as a per-command tool error ("Permission denied:
# shell"), so straying outside the allowlist costs the reviewer a retry, not
# the review.
#
# Code-execution posture (decided 2026-11-19, superseding the pytest allowance
# this policy shipped with): the reviewer's context embeds attacker-controlled
# PR body/diff/commit text, so any allowed surface that EXECUTES attacker-
# chosen code turns a prompt injection into arbitrary code execution in the
# orchestrator's own process tree. A test run is exactly that surface: the
# checkout is the PR's code, so ``pytest`` imports the PR's ``conftest.py``
# and tests. It is therefore not allowed here -- parity with the devin-shell
# reviewer, whose review prompt explicitly forbids tests/linters/interpreters
# and package managers. Reading the diff, the surrounding code and the CI
# verdict is the review loop; executing the PR is the worker's and CI's job.
# The shared review prompt's "run a single targeted test if a specific doubt
# remains" therefore costs this harness one denied-tool retry in the rare case
# it is attempted; it can never execute PR code.
#
# Egress posture (deliberate difference from the claude-code reviewer,
# decided 2026-11-18): ``webfetch`` and ``websearch`` are DENIED -- the
# claude-code reviewer's tool scope (Bash,Read,Grep,Glob,Write) carries no
# web tools, and the review packet already contains the diff and PR metadata,
# so there is nothing to fetch. This closes the exfil channel that host reads
# would otherwise open: without the deny a prompt injection could read host
# files (read/external_directory are allow) and POST them to an attacker
# server through the webfetch tool (verified live against a local listener).
# With both denied and no code-executing verb allowed, the remaining surface
# is read-only git/gh verbs (no ``gh api``), read/grep/glob/list, and
# edit/write are denied. What that does NOT claim: host reads stay allow
# (the packet path is per-PR dynamic -- the launcher does not know the state
# ``prs/`` dir when the policy constant is built), so a reviewer can still
# READ a host secret; the boundary is that no allowed tool can carry it
# off-host or write it, and the reviewer's own posted verdict text is
# constrained to the verdict schema, not free-form exfiltration.
_AUTH_CONTENT_ENV = "OPENCODE_AUTH_CONTENT"

_REVIEWER_PERMISSION: dict[str, Any] = {
    "*": "deny",
    "read": "allow",
    "glob": "allow",
    "grep": "allow",
    "list": "allow",
    "todowrite": "allow",
    "external_directory": "allow",
    # webfetch/websearch DENIED (absent => the leading catch-all denies
    # them): the claude-code reviewer has no web tools and the packet
    # already carries the diff -- see the egress-posture note above the
    # policy. On 1.x these keys would have allowed the tools; on 2.x the
    # V2 ruleset carries no allow for them either.
    # Explicitly denied tools. On 1.x ``edit`` covers edit/write; on 2.x
    # ``write`` is a separate action the legacy ``edit`` key does NOT
    # cover (runtime-verified 2.3.0: a file was created under
    # ``permission: {"edit": "deny"}``), so ``write`` is denied by its
    # own key. On 1.x a "write" rule names a tool that does not exist --
    # a dead entry, harmless.
    "edit": "deny",
    "write": "deny",
    "bash": {
        "*": "deny",
        "git status": "allow",
        "git status *": "allow",
        "git diff": "allow",
        "git diff *": "allow",
        "git log": "allow",
        "git log *": "allow",
        "git show": "allow",
        "git show *": "allow",
        "git blame *": "allow",
        "git grep *": "allow",
        "git ls-files *": "allow",
        "git rev-parse *": "allow",
        "git branch --show-current": "allow",
        "git branch --list": "allow",
        "git branch --list *": "allow",
        "git remote -v": "allow",
        "git remote show *": "allow",
        "git describe *": "allow",
        "gh pr view": "allow",
        "gh pr view *": "allow",
        "gh pr diff *": "allow",
        "gh pr checks": "allow",
        "gh pr checks *": "allow",
        "gh issue view": "allow",
        "gh issue view *": "allow",
        "gh repo view *": "allow",
        "gh run list": "allow",
        "gh run list *": "allow",
        "gh run view *": "allow",
        "gh search *": "allow",
        # findLast ordering: opencode's Permission.evaluate returns the
        # LAST matching rule, so these denies sit after the allowlist and
        # win over every "verb *" wildcard above.
        #
        # Layer 1 -- shell metacharacters. Wildcard.match sees the
        # UNEXPANDED AST text, so quoting, escaping or expansion can
        # reconstruct a denied flag without the substring pattern ever
        # seeing it: ""/'' in-word splitting ("--out""put" -> --output),
        # quoted flags ("-o"/'-O'), pwsh backtick escapes (`-o), $IFS/
        # ${var}/$( ) expansion, and brace expansion ("--o{u,t}put").
        # Chaining (;&|) and redirection (<>) can smuggle a second
        # statement past a whole-statement prefix match. Denying the
        # metacharacters themselves closes the entire class: every
        # command a legitimate reviewer runs is plain unquoted words.
        # (Literal * and ? can't be expressed as deny patterns -- both
        # are wildcard syntax -- and a backslash can't be denied either
        # -- Wildcard.match normalizes \ -> / on BOTH sides first, so
        # "*\*" would become "*/*" and deny every slashed path; bash's
        # \-escape splits are closed by the ordered-letter patterns in
        # layer 3.)
        #
        # GLOB CHARS ARE A REAL BYPASS of everything above, closed by the
        # pinned GLOBIGNORE startup file (apply_review_env_pins), NOT
        # here: a reviewer command whose tail contains a literal * or ?
        # matches an allow before opencode ever sees the shell's
        # expansion, and bash expands it against the review checkout --
        # where the UNTRUSTED PR controls the filenames. A committed file
        # named "-Osh" (legal on NTFS, sorts first because '-' < digits)
        # turns `git grep *` into git grep -Osh ... -> matched lines fed to
        # `sh`; reproduced live (2026-11-17). (The `pytest -c 0.ini` variant
        # of the same attack is moot now that no test runner is allowed.)
        # Bash resolves GLOBIGNORE at expansion time and excludes matching
        # names from every pathname expansion, so `-*:.*` removes exactly the
        # dash-leading tokens a flag can be built from while restoring
        # the default dotfile behavior; subdirectory glob results carry
        # their directory prefix ("tests/-p"), which no option parser
        # treats as a flag. The env var ALONE is not enough -- Git Bash
        # 5.2 ignores an env-inherited GLOBIGNORE (verified: present as
        # an exported shell variable yet honored only when assigned
        # inside bash) -- hence the BASH_ENV/ENV startup-file pin.
        **{f"*{ch}*": "deny" for ch in _REVIEW_BASH_METACHARS},
        **{f"*{ch}*": "deny" for ch in _REVIEW_BASH_NON_ASCII_WS},
        # Escaped-dash shapes. Wildcard.match normalizes \ -> / in the
        # command text but bash strips an unquoted \ before -, so every
        # " -" / "--" deny above can be dodged by escaping the dash
        # itself: \-o -> /-o (a short flag) and -\-o -> -/-o (a split
        # "--"). '/' can't be denied (slashed paths are legitimate), so
        # the two shapes an escaped dash produces are denied instead --
        # ' /-' (escaped dash at a word boundary) and '-/-' (a long
        # flag's dashes split by an escape). Double-backslash inputs
        # leave a literal \ in the arg and are harmless.
        "* */-*": "deny",
        "*-/-*": "deny",
        # Layer 2 -- git grep is flag-free. -O<cmd>/
        # --open-files-in-pager spawn a pager via sh -c, and git's
        # parse-options accepts bundled short flags (-nOcmd) plus
        # unambiguous long-option prefixes (--open=cmd). Two deny forms
        # are needed: "git grep -*" covers flag-first
        # ("git grep -nOcmd") and "git grep * -*" covers
        # flag-after-pattern ("git grep pat -Ocmd"); neither alone
        # covers the other. Flagless `git grep <pattern>` still works
        # and the grep tool covers the rest.
        "git grep -*": "deny",
        "git grep * -*": "deny",
        # `git branch --list` takes pattern arguments only -- a
        # "-d"/"-D" anywhere in the tail would delete branches. Two
        # forms again: flag directly after "--list " and flag later.
        "git branch --list -*": "deny",
        "git branch --list * -*": "deny",
        # Layer 3 -- dangerous flag names on the verbs that remain allowed,
        # each widened to its shortest covering substring because git accepts
        # unambiguous long-option prefixes: --open= for
        # --open-files-in-pager, --ex/--ext for --exec/--ext-diff. --output
        # writes a file from a read verb, --textconv runs the repo-config
        # diff.<driver>.textconv command, and --web makes gh spawn
        # GH_BROWSER/BROWSER.
        #
        # Command-scoped where an allowed verb legitimately carries the same
        # letters. findLast cannot express "deny X except on verb Y" -- a late
        # re-allow would have to sit after every deny and would re-expose ALL
        # of them (a `*--log-failed*` allow resurrected `rm -rf x --log-failed`),
        # so "--ex*" is narrowed to git: its --exclude is harmless, its
        # --exec/--ext-diff are not.
        "*--out*": "deny",  # git diff/log/show --output[=f]
        "*git *--ex*": "deny",  # git --exec/--ext-diff (+ --exclude, harmless)
        "*--open*": "deny",  # --open-files-in-pager prefix abbrevs
        "*--textconv*": "deny",
        "*--web*": "deny",  # gh *view --web spawns GH_BROWSER/BROWSER
        # Bash backslash escapes reconstruct a flag AFTER
        # Wildcard.match normalizes \ -> / in the command text
        # ("--out\put" -> "--out/put" -> shell strips to "--output");
        # '/' itself can't be denied (slashed paths are legitimate),
        # so the covering pattern is the flag's letters in order --
        # .* between letters absorbs inserted slashes. Each entry has
        # two forms: "--L" (letters from the first flag char) and
        # "--?L" (one char split before the first letter, i.e. --\x
        # -> --/x). --open* letter-splits are omitted:
        # --open-files-in-pager lives only on git grep, which layer 2
        # already renders flag-free.
        "*--o*u*t*p*u*t*": "deny",  # --o/u/t/p/u/t -> --output
        "*--?o*u*t*p*u*t*": "deny",
        "*git *--e*x*": "deny",  # --e/x* -> --ex*/--ext* abbrev + splits
        "*git *--?e*x*": "deny",
        "*--t*e*x*t*c*o*n*v*": "deny",  # --t/e/x/t/c/o/n/v -> textconv
        "*--?t*e*x*t*c*o*n*v*": "deny",
        "*--w*e*b*": "deny",  # --w/e/b* -> --web + splits
        "*--?w*e*b*": "deny",
        # Layer 4 -- catch-all for an attached -o/-O on any verb (" -o"
        # needs a bare single dash before the o, so "--oneline" is
        # unaffected): git grep -O<cmd> (already unreachable above; kept as
        # belt) and git diff -o<file>.
        "* -o*": "deny",
    },
}


def _v2_permission_rules(permission: dict[str, Any]) -> list[dict[str, str]]:
    """Flatten the V1 ``permission`` object into 2.x ``permissions`` rules.

    opencode 2.x's config schema has no top-level ``permission`` map field --
    only ``permissions``, an ORDERED list of ``{action, resource, effect}``
    rules (verified against the 2.3.0 bundle's ``Config.Info`` schema, and
    runtime-verified with ``opencode run --standalone``: the list is
    evaluated last-match, and the action for shell commands is ``shell``,
    not the 1.x ``bash``).

    The V1 map's insertion order is preserved verbatim, with two
    runtime-verified shape corrections:

    - the nested ``bash`` command-pattern map flattens under the action name
      ``shell`` (the 1.x ``bash`` key does not match 2.x's tool name);
    - a trailing ``{"*","*","deny"}`` rule is NOT emitted: runtime-verified
      2.3.0 withholds the shell tool entirely when a ``*`` deny sits last in
      the list, while the same catch-all deny LEADING the list leaves the
      tool present and denies unmatched commands per-command. The V1 map's
      own ``"*": "deny"`` entry becomes the leading rule; the trailing
      deny-layer entries are command-scoped patterns that never match
      ``*`` at tool level, so the tool stays surfaced (runtime-verified
      end-to-end with the shipped policy).

    Under last-match evaluation the flattened list is semantically identical
    to the V1 object. If a future opencode evaluates the list FIRST-match,
    the leading ``{"*","*","deny"}`` makes every tool fail CLOSED -- an
    unrecognized evaluation order can degrade the review, never re-open
    writes.

    Any LATER rule with resource ``*`` and effect ``deny`` is dropped as
    redundant: the leading catch-all already matches every command, so both
    match orders reach the same verdict without it, and a resource-``*``
    deny deeper in the list is exactly the shape that runtime-verified
    WITHHOLDS the shell tool on 2.3.0 (the V1 bash map's own ``"*": "deny"``
    entry is such a rule).
    """
    rules: list[dict[str, str]] = []
    for action, value in permission.items():
        tool = "shell" if action == "bash" else action
        if isinstance(value, str):
            rules.append({"action": tool, "resource": "*", "effect": value})
        else:
            for pattern, effect in value.items():
                if pattern == "*" and effect == "deny":
                    continue  # covered by the leading catch-all (see docstring)
                rules.append({"action": tool, "resource": pattern, "effect": effect})
    return rules


_REVIEWER_CONFIG: dict[str, Any] = {
    "$schema": "https://opencode.ai/config.json",
    "permission": _REVIEWER_PERMISSION,
    # 2.x-native shape (see _v2_permission_rules). The V1 object above stays
    # for 1.x binaries; on 2.x both keys load (2.3.0 accepts unknown legacy
    # keys without rejecting the document -- runtime-verified) and the V2
    # list is the one the tool evaluation consults. On a 1.x binary the
    # ``permissions`` list is expected to be an ignored unknown key the same
    # way (not verified on 1.18.35 -- the fleet binary claims to be that
    # version); if a future 1.x rejects unknown keys outright, the failure
    # mode is a refused launch, which fails loud rather than fail-open.
    "permissions": _v2_permission_rules(_REVIEWER_PERMISSION),
    "autoupdate": False,
    "share": "disabled",
    "snapshot": False,
}


# Reviewer-only env scrub: names a hostile worker_env must never inject into a
# review launch because an ALLOWED verb turns them into code execution or a
# redirect of the files the reviewer inspects. Removal (not pinning to "") is
# correct here -- each absent name lets the tool fall back to its safe
# default, whereas "" keeps e.g. a session-tmp path alive. Ambient copies in
# os.environ are the operator's own trust boundary, same as PATH; this strip
# only governs the caller-supplied layer. Match is case-insensitive: the
# spawned Windows process collapses case, so a lowercase twin must go too.
_REVIEW_ENV_STRIP = frozenset(
    {
        # Pager/editor/diff/ssh hooks an allowed `git <verb>` spawns through.
        "GIT_EDITOR",
        "GIT_PAGER",
        "GIT_EXTERNAL_DIFF",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_PROXY_COMMAND",
        "PAGER",
        "EDITOR",
        "VISUAL",
        # gh pager/browser/editor hooks: `gh *view` (allowed) pages through
        # GH_PAGER/PAGER when GH_FORCE_TTY coerces terminal detection, opens
        # GH_BROWSER/BROWSER on `--web`, and falls back to GH_EDITOR.
        "GH_PAGER",
        "GH_FORCE_TTY",
        "GH_BROWSER",
        "BROWSER",
        "GH_EDITOR",
        # gh host/config redirection: GH_HOST repoints every allowed
        # `gh <verb>` at an attacker-chosen server (review-integrity, not
        # exec), and a hosts-file override would bypass the isolated
        # GH_CONFIG_DIR pin's empty-credential posture.
        "GH_HOST",
        "GH_HOSTS_FILE",
        # `less` preprocessor/postprocessor hooks (any pager invocation),
        # plus the in-pager shell/editor escapes: LESSSECURE=0 re-enables
        # less's `!` shell escape and LESSEDIT is its editor spawn. Inert
        # while pager output is captured non-interactively, but they cost
        # nothing to remove.
        "LESSOPEN",
        "LESSCLOSE",
        "LESSSECURE",
        "LESSEDIT",
        # opencode shell selection: OPENCODE_GIT_BASH_PATH is honored before
        # PATH lookup when Git Bash is selected, COMSPEC is the cmd fallback
        # (verified 1.18.35 Shell.gitbash/EU()).
        "OPENCODE_GIT_BASH_PATH",
        "COMSPEC",
        # Git repo-discovery redirection: an allowed read verb inside the
        # review checkout could instead open (and update-ref an `update`
        # subcommand on) an attacker-chosen repository.
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_NAMESPACE",
        "GIT_EXEC_PATH",
        # Shell selection/startup files the reviewer bash tool sources.
        # BASH_ENV/ENV are stripped here and then RE-PINNED in
        # apply_review_env_pins to the orchestrator-written GLOBIGNORE
        # startup file (the glob-char closure -- see the Layer-1 comment):
        # stripping alone is not enough because Git Bash 5.2 ignores an
        # env-inherited GLOBIGNORE, so the assignment must run inside bash.
        "SHELL",
        "BASH_ENV",
        "ENV",
        "ZDOTDIR",
        # Python import-path hooks. No test runner or interpreter is on the
        # reviewer allowlist (see the code-execution posture above), so these
        # are belt rather than load-bearing: they exist so a future allow of
        # any Python-executing verb cannot be turned into an arbitrary import
        # or startup-exec by an ambient/operator value.
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONBREAKPOINT",
        "PYTHONEXECUTABLE",
        # $PYTHONUSERBASE/.../site-packages is added to sys.path (and its
        # .pth files executed) whenever user-site is active.
        "PYTHONUSERBASE",
        # Native loader preloads (inherited into every spawned process).
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        # Node runtime hooks (opencode is a bun-compiled binary; --require is
        # honored for node compat). NODE_PATH is a module-resolution hook --
        # the compiled bundle is self-contained, but removing it costs
        # nothing.
        "NODE_OPTIONS",
        "NODE_PATH",
        # uv resolver hooks: an attacker index or interpreter path is remote
        # package execution. Belt -- no uv verb is allowed today.
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_INDEX",
        "UV_FIND_LINKS",
        "UV_PYTHON",
        "UV_PYTHON_INSTALL_MIRROR",
        "UV_ENV_FILE",
        "UV_PROJECT_ENVIRONMENT",
        # `uv run` project/config redirection: UV_PROJECT repoints the run
        # at a different directory's pyproject.toml/conftest.py, and
        # UV_CONFIG_FILE loads an arbitrary uv.toml.
        "UV_PROJECT",
        "UV_CONFIG_FILE",
        # gh/gh-credential redirection: GH_CONFIG_DIR must stay on
        # sanitize_env's isolated pin or the reviewer can read the
        # orchestrator's gh tokens out of the pointed-to hosts.yml.
        "GH_CONFIG_DIR",
        # Session-tmp and repo-escape pins owned by sanitize_env; removing a
        # caller-supplied value restores the isolated layer underneath.
        "TMP",
        "TEMP",
        "TMPDIR",
        "GIT_CEILING_DIRECTORIES",
        "XDG_RUNTIME_DIR",
        "SSH_AUTH_SOCK",
        # Deliberately NOT stripped, with reasons (probed 2026-11-17):
        # - SYSTEMROOT/WINDIR: Windows loader and SChannel derive system
        #   paths from them; stripping breaks the child outright, and an
        #   attacker value reaches no allowed-verb exec surface that
        #   ambient PATH does not already reach (operator trust boundary).
        # - PATHEXT: only consulted by cmd.exe resolution; the reviewer
        #   shell is Git Bash.
        # - PSModulePath: only powershell/pwsh module loading; the reviewer
        #   shell is Git Bash, and pwsh does not glob for native commands.
        # - BUN_CONFIG_*/BUN_INSTALL: opencode is a self-contained compiled
        #   bun binary; no install or preload path is reachable.
        # - LESSHISTFILE/LESS: pager cosmetics; LESSSECURE/LESSEDIT (the
        #   exec-capable ones) ARE stripped above.
        # - GH_TOKEN/GH_ENTERPRISE_TOKEN: sanitize_env strips the ambient
        #   copies; worker_env is the documented operator injection point
        #   for a scoped token, and the reviewer only runs read verbs.
        # - GH_HOST: see GH_HOSTS_FILE above (it IS stripped).
    }
)

# Case-insensitive name prefixes: git reads GIT_CONFIG_KEY_n/GIT_CONFIG_VALUE_n
# as inline config pairs (alias/gitconfig exec vectors) and GIT_TRACE* writes
# debug output to arbitrary paths.
_REVIEW_ENV_STRIP_PREFIXES = (
    "GIT_CONFIG_KEY_",
    "GIT_CONFIG_VALUE_",
    "GIT_TRACE",
)


# Reviewer-only names beyond _REVIEW_ENV_STRIP: 2.x control surfaces a
# caller-supplied or ambient value must not inject through the review env
# (see apply_review_env_pins' inline comments). Path/value-shaping names are
# pinned to neutral values in review_pins instead of stripped.
_REVIEW_OPENCODE_STRIPS = frozenset(
    {
        "OPENCODE_MODELS_URL",
        "OPENCODE_MODELS_PATH",
        "OPENCODE_LOG_LEVEL",
        "OPENCODE_PRINT_LOGS",
        "OPENCODE_SIMULATE",
        "OPENCODE_API_KEY",
        "OPENCODE_PASSWORD",
        "OPENCODE_SERVER_PASSWORD",
    }
)


def apply_review_env_pins(
    env: dict[str, str],
    *,
    base_pins: Mapping[str, str],
    data_dir: Path,
    scoped_auth: str | None,
) -> None:
    """Re-pin every opencode control surface for a review launch, in place.

    Called by ``worker_env_for`` AFTER the operator ``worker_env`` merge --
    see the inline comments for why pins (not strips) are the enforcement
    point. ``base_pins`` is the caller's worker-posture env block
    (``opencode_worker._WORKER_ENV``): its safety flags were merged BEFORE
    ``worker_env``, so they are re-pinned here too.
    """
    # Read-only invariant, the opencode counterpart of the
    # --permission-mode plan pin _sanitize_review_command_template applies
    # for claude-code reviewers: no operator worker_env -- and no ambient
    # OPENCODE_* the sanitized base env inherited from os.environ -- may
    # grant a reviewer write access. Every control surface opencode reads
    # is therefore pinned AFTER the worker_env merge, not stripped before
    # it (ambient values would sail straight through a strip):
    # - OPENCODE_CONFIG_CONTENT: the deny policy itself. opencode merges
    #   it after global and project config files. On 2.x the V2
    #   ``permissions`` ruleset it carries is the shape the tool
    #   evaluation consults; on 1.x the V1 ``permission`` object is.
    # - OPENCODE_PERMISSION: opencode merges this env AFTER
    #   OPENCODE_CONFIG_CONTENT (verified against the 1.18.35 binary; the
    #   env var name no longer appears anywhere in the 2.3.0 bundle's
    #   string table -- verified 2026-11-18 -- so on 2.x this pin is a
    #   no-op, pinned anyway so an older binary never sees an ambient
    #   value; the 2.x enforcement surface is the ``permissions`` list
    #   inside OPENCODE_CONFIG_CONTENT).
    # - OPENCODE_DISABLE_PROJECT_CONFIG **and its 2.x alias
    #   OPENCODE_CONFIG_PROJECT_DISABLE**: the review checkout holds
    #   UNTRUSTED PR code; without these pins the PR's .opencode/
    #   opencode.json (agent.*.permission, tools) and .opencode/plugin/
    #   load ahead of the deny policy and can override it or run code at
    #   startup. The 2.3.0 bundle reads
    #   ``OPENCODE_CONFIG_PROJECT_DISABLE ?? OPENCODE_DISABLE_PROJECT_CONFIG``
    #   -- the ALIAS takes precedence, so pinning only the legacy name
    #   re-enables the PR's project config (verified live: with the alias
    #   set, a PR-committed .opencode/opencode.json loaded and its shell
    #   ran a write the pinned policy denies). Also gates the project
    #   AGENTS.md/CLAUDE.md walk.
    # - OPENCODE_PURE: skip external plugins entirely -- belt on the
    #   project-config plugin vector, and no plugin install runs in the
    #   review worktree. 1.x-only name (0 occurrences in the 2.3.0
    #   bundle, verified 2026-11-18); on 2.x the same vector is closed by
    #   the project-config disable pins above (the PR's plugin list loads
    #   through project config) and the pinned-empty global config dir
    #   (no global plugins exist to load).
    # - OPENCODE_CONFIG: truthy-gated in the loader, so "" safely skips
    #   an extra config file.
    # - OPENCODE_CONFIG_DIR / OPENCODE_TEST_MANAGED_CONFIG_DIR /
    #   OPENCODE_TEST_HOME: pinned to real EMPTY DIRECTORIES, never "".
    #   All three are read with ?? / || fallbacks in the binary (verified
    #   1.18.35): OPENCODE_CONFIG_DIR="" survives `?? G.config` and leaves
    #   Global.config == "", which makes Instruction resolve "AGENTS.md"
    #   against the review checkout's cwd -- re-enabling untrusted-PR
    #   instruction injection despite OPENCODE_DISABLE_PROJECT_CONFIG.
    #   OPENCODE_TEST_MANAGED_CONFIG_DIR="" falls through `|| g0()` to
    #   the real managed dir (C:\\ProgramData\\opencode on Windows),
    #   whose opencode.json would load as GLOBAL config -- including
    #   agent.<name>.permission, which Agent.state merges AFTER
    #   OPENCODE_PERMISSION and could therefore un-deny the policy.
    #   OPENCODE_TEST_HOME="" has the same ?? trap for the ~/.opencode
    #   and ~/.claude walks. The pins also close the ambient/host files
    #   entirely (~/.config/opencode, ~/.opencode, the managed dir).
    # - XDG_* / OPENCODE_DB: re-pinned AFTER the worker_env merge (not
    #   just defaulted before it): XDG_CONFIG_HOME/opencode is the
    #   global config dir the agent-permission override vector reads,
    #   and an ambient or operator OPENCODE_DB / XDG_DATA_HOME could
    #   redirect the reviewer's DB off the per-issue dir. OPENCODE_DB is
    #   pinned to the CONCRETE per-issue path: on 1.18.35 "" was
    #   truthy-gated back to the default (verified), but on 2.3.0 ""
    #   reaches bun:sqlite as the literal path -- runtime-verified
    #   "unable to open database file", the standalone server dies before
    #   readiness and the launch fails outright. An absolute per-issue
    #   path redirects nothing and works on both.
    # - OPENCODE_DISABLE_EXTERNAL_SKILLS: the review checkout's
    #   .claude/ and .agents/ SKILL.md files (and ~/.claude, ~/.agents)
    #   are scanned unless this is set -- it is NOT covered by
    #   OPENCODE_DISABLE_PROJECT_CONFIG. 1.x-only name (0 occurrences in
    #   the 2.3.0 bundle); on 2.x the same scan is gated by the
    #   ``config.claude`` switch -- OPENCODE_DISABLE_CLAUDE_CODE (pinned
    #   via base_pins) and OPENCODE_DISABLE_CLAUDE_CODE_SKILLS below,
    #   verified against the 2.3.0 env table.
    # - 2.x-only control surfaces, stripped or pinned here because a
    #   caller-supplied/ambient value would otherwise survive the merge
    #   (verified against the 2.3.0 bundle's env table, 2026-11-18):
    #   * OPENCODE_CLI_CONFIG_CONTENT: an inline CLI-config document (the
    #     TUI/cli.json channel) -- pinned "{}" so no operator or ambient
    #     value injects config into the reviewer.
    #   * OPENCODE_MODELS_URL / OPENCODE_MODELS_PATH: repoint the
    #     models.dev metadata source -- a hostile value could serve a
    #     custom provider definition (baseURL under attacker control).
    #     Stripped: removal falls back to the binary's default source.
    #   * OPENCODE_LOG_LEVEL / OPENCODE_PRINT_LOGS: logging is owned by
    #     the argv pins (--log-level warn --print-logs); an ambient env
    #     value must not re-widen or silence it.
    #   * OPENCODE_SIMULATE: a dev-only simulation flag that could fake
    #     provider behaviour; stripped.
    #   * OPENCODE_API_KEY / OPENCODE_PASSWORD / OPENCODE_SERVER_PASSWORD:
    #     credential channels -- the reviewer authenticates through the
    #     seeded per-issue auth.json only; a host-wide key must not ride
    #     into the review child env.
    # - OPENCODE_AUTH_CONTENT: Auth.all() reads it directly even in a
    #   review launch. An ambient or worker_env value bypasses
    #   _scoped_auth_content (leaking every host credential into the
    #   child env), and a "wellknown" entry makes the child fetch remote
    #   config whose agent.*.permission merges AFTER OPENCODE_PERMISSION
    #   -- un-denying the policy. The pin is the scoped credential or {}.
    # - _WORKER_ENV's safety flags (autoupdate/LSP/CLAUDE-prompt
    #   disables, bash timeout): merged BEFORE worker_env above, so they
    #   too are re-pinned here.
    # - Git config surface: `git <verb>` is on the bash allowlist, so
    #   ~/.gitconfig / inline-config env vectors (alias.<verb> = "!cmd",
    #   GIT_CONFIG_KEY_n/VALUE_n, GIT_CONFIG_PARAMETERS) would exec a
    #   pager-level command despite the deny policy. Global and system
    #   config are pinned to a nonexistent file and inline config off.
    # - PYTEST_ADDOPTS / PYTEST_PLUGINS / PYTEST_DISABLE_PLUGIN_AUTOLOAD:
    #   env-level addopts/plugin injection. No test runner is allowed in a
    #   review launch (see the code-execution posture), so these are belt for
    #   a future re-allow: an operator/ambient value must not be able to
    #   inject a plugin into a Python run the policy later permits.
    #
    # Case-insensitive collisions: the spawned child's env map is
    # case-insensitive (CreateProcessW resolves last-entry-wins) but a
    # Python dict happily holds "opencode_permission" and
    # "OPENCODE_PERMISSION" side by side, and worker_env keys land AFTER
    # the pin slots in build_worker_env's merge order. Every key whose
    # .upper() collides with a pin name -- or matches the exec-vector
    # strip list -- is therefore removed before the pins are applied.
    xdg_config = data_dir / "xdg" / "config"
    empty_home = data_dir / "empty-home"
    empty_managed = data_dir / "empty-managed-config"
    # GLOBIGNORE startup file -- the glob-char closure (see the Layer-1
    # comment in _REVIEWER_PERMISSION). Bash only honors GLOBIGNORE when it
    # is ASSIGNED inside the shell (Git Bash 5.2 ignores an env-inherited
    # value -- verified), so the file is sourced via pinned BASH_ENV/ENV:
    # `-*` drops every dash-leading name from pathname expansion (the only
    # tokens an option parser can mistake for a flag) and `.*` restores the
    # default dotfile behavior that setting GLOBIGNORE would otherwise
    # disable (setting it non-null implicitly enables dotfile matching).
    # Note: 2.3.0 falls back to PowerShell when Git Bash is absent, where
    # the startup file is inert (PowerShell does not read BASH_ENV) -- the
    # closure is Git-Bash-only, but the metachar denies above hold in any
    # shell, so no hole opens either way.
    bash_startup = data_dir / "bash-startup.sh"
    startup_text = "GLOBIGNORE='-*:.*'\n"
    # The pinned global config dir: created EMPTY (the pin is only meaningful
    # as a real directory -- "" re-resolves "AGENTS.md" against the checkout
    # per the OPENCODE_CONFIG_DIR note above). Consequence, deliberately
    # accepted and surfaced by `charlie doctor`: a review launch sees only the
    # providers/models built into the binary, never the operator's
    # ~/.config/opencode/opencode.json provider definitions -- that file is
    # the same agent-permission override vector the deny policy closes.
    review_config_dir = xdg_config / "opencode"
    try:
        for directory in (data_dir, xdg_config, review_config_dir, empty_home, empty_managed):
            directory.mkdir(parents=True, exist_ok=True)
        # Atomic: a partially-written startup file sourced by bash would
        # leave the GLOBIGNORE closure half-applied (or unset) while the
        # launch proceeds. A real storage failure is logged and the launch
        # continues -- the metachar denies still hold, so the residual risk
        # is the glob-char layer only.
        write_text_atomic(bash_startup, startup_text, encoding="ascii")
    except OSError as exc:
        log.warning(
            "opencode review env: could not write %s (%s: %s) -- the "
            "GLOBIGNORE glob-char closure is not in place for this review",
            bash_startup,
            type(exc).__name__,
            exc,
        )
    review_pins: dict[str, str] = {
        **base_pins,
        "OPENCODE_CONFIG_CONTENT": json.dumps(_REVIEWER_CONFIG, separators=(",", ":")),
        "OPENCODE_PERMISSION": json.dumps(_REVIEWER_CONFIG["permission"], separators=(",", ":")),
        # BOTH spellings: the 2.3.0 bundle reads
        # ``OPENCODE_CONFIG_PROJECT_DISABLE ?? OPENCODE_DISABLE_PROJECT_CONFIG``
        # -- the alias wins when set, so pinning only the legacy name lets a
        # hostile OPENCODE_CONFIG_PROJECT_DISABLE re-enable the PR's
        # .opencode/ project config (verified live 2026-11-18).
        "OPENCODE_CONFIG_PROJECT_DISABLE": "1",
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_PURE": "1",
        "OPENCODE_DISABLE_EXTERNAL_SKILLS": "1",
        # 2.x names for the claude-code import gate (prompt + skills scan):
        # base_pins already carries OPENCODE_DISABLE_CLAUDE_CODE.
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
        # Inline CLI-config document: pinned to the empty document so no
        # caller/ambient value injects config through the cli.json channel.
        "OPENCODE_CLI_CONFIG_CONTENT": "{}",
        "OPENCODE_CONFIG": "",
        "OPENCODE_CONFIG_DIR": str(review_config_dir),
        "OPENCODE_TEST_MANAGED_CONFIG_DIR": str(empty_managed),
        "OPENCODE_TEST_HOME": str(empty_home),
        "OPENCODE_DB": str(data_dir / "opencode.db"),
        "XDG_DATA_HOME": str(data_dir),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_STATE_HOME": str(data_dir / "xdg" / "state"),
        "XDG_CACHE_HOME": str(data_dir / "xdg" / "cache"),
        _AUTH_CONTENT_ENV: scoped_auth or "{}",
        "GIT_CONFIG_GLOBAL": str(empty_home / "gitconfig"),
        "GIT_CONFIG_SYSTEM": str(empty_home / "gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_PARAMETERS": "",
        "GIT_CONFIG_COUNT": "0",
        "PYTEST_ADDOPTS": "",
        "PYTEST_PLUGINS": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        # The GLOBIGNORE startup file (stripped from caller/env above, then
        # re-pinned here -- same pin-not-strip reasoning as the OPENCODE_*
        # controls). ENV covers a posix-mode sh read; BASH_ENV covers bash.
        "BASH_ENV": str(bash_startup),
        "ENV": str(bash_startup),
    }
    pinned_names = {name.upper() for name in review_pins}
    for key in list(env):
        upper = key.upper()
        if (
            upper in pinned_names
            or upper in _REVIEW_ENV_STRIP
            or upper in _REVIEW_OPENCODE_STRIPS
            or upper.startswith(_REVIEW_ENV_STRIP_PREFIXES)
        ):
            del env[key]
    env.update(review_pins)
