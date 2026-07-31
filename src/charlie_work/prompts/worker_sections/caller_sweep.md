## Caller sweep

When you change any function or method signature, grep for every call site in product code AND tests, and update each one. A stale caller swallowed by a broad `except` is a silent regression, not a passing test.
