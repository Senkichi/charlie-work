# Triage labels

The mattpocock-skills plugin speaks in five canonical triage roles. This file maps them to
this repo's labels. The fleet dispatcher (charlie-work) reads labels only, so this mapping is
what turns a skill's "ready" decision into dispatch.

| Label in mattpocock/skills | Label in our tracker | Meaning |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | Maintainer needs to evaluate this issue |
| `needs-info` | `question` | Waiting on the reporter for more information |
| `ready-for-agent` | `automated-ready` | Fully specified; the fleet dispatches it as soon as the label lands |
| `ready-for-human` | `human-action` | Requires human implementation; the fleet never dispatches it |
| `wontfix` | `wontfix` | Will not be actioned |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the label string
from the middle column.

Applying `automated-ready` starts work immediately. Apply it only to an issue whose blockers
are written as `issue-tracker.md` describes.
