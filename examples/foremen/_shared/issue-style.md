# Issue style

Every issue you file must pass the issue-contract lint and give a worker
everything it needs in one body.

## Body contract

- One deliverable per issue. A piece that needs two commits to land safely
  is two issues.
- Acceptance criteria: the checks a reviewer can run to know the piece
  landed.
- An out-of-scope list naming what the issue deliberately does not do.
- No issue references or ordering in prose. Apply renders the `Blocked by
  #N` lines itself from your `blocked_by` keys — never type a `#N`, a
  "depends on" phrase, or a closing keyword in the body.
- No path-and-line citations — line numbers rot. Cite the symbol.

## Planning rules

- Reading is cheap and a wasted worker run is not: take as many tool calls as the
  code needs.
- Before you propose to move or change a symbol, find everything that depends on
  where it lives: every caller, every test that imports or patches it, and every
  test that asserts something about the structure of the file it lives in. Name
  each affected test file in the issue, and state any invariant such a test
  protects.
- A worker never makes a design decision. Where a piece has more than one
  reasonable design, you decide, and the issue states the decision as an
  instruction. Never offer the worker alternatives. If later pieces depend on the
  decision, the piece that settles it is filed first.
- Read the charter's outcome literally and check each issue against its exact
  words.
- Never state a number you did not measure.
