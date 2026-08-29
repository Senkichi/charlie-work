# Security Policy

## Reporting a vulnerability

Use **GitHub private vulnerability reporting** (Security tab → "Report a
vulnerability") on this repository. Please do not open a public issue for
anything you believe is exploitable.

You can expect an acknowledgement within a few days. There is no bug bounty.

## Scope and threat model

charlie-work is a GitHub-issue orchestrator that dispatches autonomous AI
workers on the operator's own machine, with CI on self-hosted runners. The
security-relevant surfaces, in rough priority order:

1. **Untrusted GitHub content reaching a worker or shell.** Issue bodies, PR
   titles/bodies, and comments are attacker-writable. Anything that lets that
   content escape its role as inert text — prompt-injection via template
   re-scanning, closing-keyword hijack of PR→issue linking, unfiltered
   comment authors — is a vulnerability. See the trust-model section in
   `CONTRIBUTING.md` for the existing mitigations and where they live.
2. **Path traversal out of managed roots.** Runner-slot discovery and
   worktree handling enforce resolved-path containment (junction/reparse
   points must not let an entry escape the configured root). Weakening those
   checks is a vulnerability even when no exploit is demonstrated.
3. **Secrets in state or logs.** Worker subprocess environments are
   sanitized (`sanitize_env`); state files and event logs must not capture
   tokens.

Out of scope: denial of service against the operator's own orchestrator by
the operator's own configuration; issues requiring an already-compromised
operator account.

## Supported versions

Only the current `main` branch is supported. There are no maintained release
branches.
