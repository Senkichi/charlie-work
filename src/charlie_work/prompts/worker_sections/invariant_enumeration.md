## Invariant enumeration

When the issue requires an invariant to hold across multiple code paths ("release on every exit path", "publish after every mutation", and similar), mechanically enumerate every such path in the functions you changed — every `return` and `raise` between the invariant's start and end. List each one in the PR body along with how it satisfies the invariant. A path you did not enumerate is a path you did not check.
