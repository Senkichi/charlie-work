# Provenance

Frozen verbatim copies from the archived **freecode** repository
(a private, now-archived sibling repo, archived 2026-07-09), taken at commit
`c1beaef` ("fix(codex): drift cleanup vs locked ADR-006D (D1+D2+D3) (#132)",
2026-05-29 — the final pre-archive code commit).

These files are **reference material, not live code** — they are not imported by
`charlie_work` and are excluded from lint/test scope by location. Port them into
`src/charlie_work/` per the numbered specs in the parent directory when the
corresponding trigger fires.

| File here | Source path in freecode | LOC | Loose imports to replace when porting |
|---|---|---|---|
| `events.py` | `src/freecode/observability/events.py` | 103 | `freecode.config.logs_dir` → charlie-work's `paths.py` state dir; sibling `redaction`/`sinks` imports |
| `sinks.py` | `src/freecode/observability/sinks.py` | 109 | none (stdlib only) |
| `redaction.py` | `src/freecode/observability/redaction.py` | 65 | none (stdlib only) |
| `keys.py` | `src/freecode/security/keys.py` | ~180 | `keyring` (new dep for charlie-work); optional `DecisionEvent` emit → drop or wire to ported events |
| `capacity.py` | `src/freecode/planner/capacity.py` | 155 | `freecode.planner.models.CapacityModel` → `planner_models.py` (copied alongside) |
| `planner_models.py` | `src/freecode/planner/models.py` | 135 | none (stdlib only) |
| `limits.py` | `src/freecode/planner/limits.py` | 125 | yaml seed-file loader for `capacity.py`'s `CapacityModel` rows |
| `example_structural_guard_test.py` | `tests/test_claude_code_credential_isolation.py` | ~300 | pattern exemplar only — rewrite targets/literals for charlie-work invariants |

Verification notes (from the 2026-07-09 evaluation, freecode
`docs/evaluation-2026-07/`): `events.py`/`sinks.py`/`redaction.py` were grep-verified to have
zero `freecode.state`/`freecode.providers` coupling (277 LOC combined); `capacity.py` is pure
(no I/O — was choke-guard-enforced in the source repo); `keys.py:85-113` contains the Windows
Credential Manager compound-vs-bare lookup-order fix worth preserving verbatim.
