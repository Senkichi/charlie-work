# Private-slug baseline (issue #1502, per-entry layout #1802)

This directory is the ratchet baseline for `charlie private-slug-check`
(`src/charlie_work/private_slug_gate.py`). It replaced the single
`.private-slug-baseline.json` document, which was a shared append point:
concurrent PRs conflicted on it on merge, and a CONFLICTING PR gets no
pull_request CI run.

Layout:

- `slugs/<slug>` — one empty marker file per configured private repo slug.
  Presence means the slug is gated.
- `files/<repo-relative-path>.count` — the recorded mention count for that
  repo file. First line is a non-negative integer; any following lines are
  `#` comments. There is no stored total: the check computes it by summing
  the entries, and the baseline *increase* a PR claims is derived from the
  diff itself.

Regenerate after intentionally adding or removing mentions:

    uv run charlie private-slug-check --regenerate

To acknowledge net-new mentions in a PR without regenerating, add or raise
the matching `files/<path>.count` entry — two PRs that touch different
entries merge cleanly.
