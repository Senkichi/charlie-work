"""Destination package for ``OrchestratorApp`` method-body delegates (Track 2 Phase B).

Phase B of the ``OrchestratorApp`` god-object decomposition (umbrella #1582,
design ``docs/design/2026-09-04-orchestratorapp-mikado-graph-and-delegation-plan.md``)
moves ``OrchestratorApp`` method *bodies* out of ``charlie_work.workflow``, one
Mikado leaf at a time, into the domain-named submodules of this package. Each
moved body becomes a module-level ``def <name>(self, ...)`` here; the installer
in ``charlie_work.workflow_delegation`` introspects these submodules, derives a
``name -> module`` route table, and re-attaches every function onto
``OrchestratorApp`` as a class attribute (so ``self`` binds through the
descriptor protocol exactly as a lexical method did).

At L00 (issue #1631) this package is **empty on purpose**: it is the destination
skeleton only. It moves zero members, so ``workflow_delegation.discover_delegate_modules``
finds no submodules and the post-class install call in ``workflow.py`` is a
no-op. Later leaves add one submodule (or a small cluster) at a time; the same
installer picks them up with no edit to ``workflow.py``.

Module-namespace rule (design Section 3.1 rule 2, #1627 -- load-bearing):
a submodule of this package that needs a ``charlie_work.workflow`` module-level
free function MUST reference it through the module object
(``import charlie_work.workflow as _wf; _wf.<fn>(...)``), never via a fresh
``from charlie_work.workflow import <fn>``. The module-object form is the only
one that (a) keeps the ~140 Tier D ``patch("charlie_work.workflow.<fn>")`` sites
intercepting the moved body, and (b) is cycle-safe when ``workflow.py`` itself
imports the submodule at import time (it binds the partially-initialized module
object registered in ``sys.modules`` and resolves the attribute at call time).
Correspondingly, this package must never import ``charlie_work.workflow`` at
package-import time -- importing ``charlie_work.orchestration`` must not pull in
``charlie_work.workflow``.
"""

from __future__ import annotations
