# devin-orchestrator

Deterministic GitHub-issue orchestration for AI worker fleets. One labeled
issue → one worker session → one PR → adversarial review → gated auto-merge.
Devin workers by default; Claude Code workers first-class.

Extracted from the battle-tested orchestrators that ran inside
[job-cannon](../job-cannon) and [empericus](../empericus) — this repo is the
union of both forks plus the fixes each learned separately.

## How it works

```
                    ┌─────────────────────────────────────────────┐
                    │            GitHub labels = state            │
                    └─────────────────────────────────────────────┘
  intake ──► dispatch ──► worker session ──► PR ──► review ──► merge-ready
  (issues        │        (Devin / Claude       (adversarial      │
   labeled       │         Code, one issue       packet, cross-   ├─ merge
   automated-    │         per session)          family pass)     ├─ labels
   ready)        └─ writes worker-prompt.md                       └─ branch
                    + session manifest                               delete
                                                                  (best-effort)
```

- **Hub**: a deterministic Python CLI (`devin-orch`). No chat memory, no
  LLM-driven control flow — state lives in GitHub labels plus a JSON state
  file under `.var/devin-orchestrator/`.
- **Workers**: hermetic one-shot sessions, each bound to exactly one issue,
  driven by a generated `worker-prompt.md`.
- **Review**: the orchestrator generates an adversarial review packet per PR
  (diff, checks, metadata) and optionally runs a **cross-family pass** — a
  non-Claude model (codex via the Devin CLI) attacks the PR; its findings are
  leads, never merge gates.
- **Merge**: gated on required CI checks + a recorded `approved` decision.
  Branch deletion is remote-only and best-effort — it can never abort the
  merge/label sequence.

## Quickstart

```powershell
# in this repo
uv sync

# in your consumer repo's pyproject.toml
# [tool.uv.sources]
# devin-orchestrator = { path = "../devin-orchestrator", editable = true }

# copy an example config to your repo root and adjust
cp ../devin-orchestrator/examples/orchestrator.config.devin.yaml orchestrator.config.yaml

devin-orch doctor              # preflight: env, labels, CI-check names, config
devin-orch bootstrap-labels    # create the agent:* labels once
devin-orch status --json       # what is ready / active / linked
devin-orch dispatch --limit 3  # newest-first wave
devin-orch dispatch --issues 565,570   # dependency-ordered wave
devin-orch review --pr 123     # generate adversarial review packet
devin-orch record-review --pr 123 --decision approved --summary-file review.md
devin-orch merge-ready --pr 123
devin-orch loop --limit 3      # intake + dispatch + review + merge in one pass
devin-orch spec-review --file docs/SPEC.md   # cross-family pass on a design doc
```

## Configuration

Config is discovered at `<repo-root>/orchestrator.config.yaml` (override with
`--config`). Absent file → dataclass defaults. See [examples/](examples/) for
the two shipped profiles:

| Profile | Worker runtime | Notes |
|---|---|---|
| `orchestrator.config.devin.yaml` | Devin sessions | skills-based worker loop, cross-family review on |
| `orchestrator.config.claude-code.yaml` | Claude Code | direct-shell worker loop, Claude-only review |

Key knobs: `labels.*` (state-machine label names), `dispatch.default_limit` /
`branch_prefix` / `worker_template`, `review.max_rework_cycles`,
`auto_merge.required_checks` (verify with `doctor`), `runtime.prompts_dir`
(repo-local template overrides), `devin.adapter` (`manual` | `command`),
`cross_family.*` (non-Claude adversarial pass).

## Prompt templates

Package defaults live in `src/devin_orchestrator/prompts/`. A repo-local
directory (`runtime.prompts_dir`) overrides them **by filename** — drop in
your own `worker.md` carrying your repo's invariants and canonical commands,
and everything else keeps the defaults. `worker.md` targets Devin sessions;
`worker_claude_code.md` targets Claude Code workers
(`dispatch.worker_template` selects).

## State

`.var/devin-orchestrator/` in the consumer repo holds `state.json`
(issues/prs/events, schema v1 — compatible with pre-extraction state), the
per-issue worker prompts, per-PR review packets and decisions, and the
session dispatch manifest/results. All JSON writes are atomic
(temp-file + rename).

## Provenance

Unioned from two production forks (June–July 2026): job-cannon contributed
cross-family review and `spec-review`; empericus contributed `--issues` wave
dispatch, the `gh pr merge` stdout fix, and the worktree/branch-deletion
failure report that drove the decoupled merge sequence.
