## Mutation check

Every regression test you add or update must be mutation-checked before you claim it as coverage:

1. Revert ONLY the fixed artifact — the specific file/function you changed to fix the bug — to its merge-base version: `git show $(git merge-base HEAD origin/main):<path>` written back to that file, not a full branch reset.
2. Run the test. It MUST FAIL against the reverted code.
3. Restore your fix. Run the test again. It MUST PASS.
4. In the PR body, name the exact edit reverted (file, function, or line range) and paste the literal terminal output of BOTH runs. A prose claim like "mutation check passed" with no output is treated as not done.

If the check fails to fail — the test still passes against the unfixed code — the test is not exercising the fix; strengthen the test. Never loosen an assertion just to make the mutation check pass, and never drop the claim from the PR body to hide a failing check.
