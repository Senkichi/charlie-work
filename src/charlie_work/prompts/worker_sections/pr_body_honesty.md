## PR body honesty

Never claim work you did not do. Every sentence in the PR body must be verifiable against the diff:

- Do not say "tests added" for tests you only renamed or moved.
- Do not stub a test to `pass` while claiming its coverage is provided elsewhere, without pasting the proof.
- Do not describe intended or planned behavior as implemented behavior.

If a claim cannot be checked against the diff by someone who has not read your reasoning, rewrite it or remove it.
