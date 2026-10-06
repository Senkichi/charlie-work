"""Record ``gh --json`` vs GraphQL fixtures for the G4 contract tests.

Reads only. For each case below this runs the real ``gh <resource> view|checks
--json FIELDS`` and the GraphQL document ``gh_json_fields`` builds for the same
object, and writes both to ``tests/fixtures/gh_json/<case>.json``. The contract
test (``tests/test_github_transport_gh_json_fields.py``) then asserts the
normalizer turns the GraphQL data into exactly what gh printed.

Run it from the repo root, authenticated, against this repository::

    uv run --no-sync python scripts/record_gh_json_fixtures.py

Tokens never reach the output (only response bodies are stored); a
``ghp_``/``gho_``/``github_pat_`` substring aborts the write.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from charlie_work.github_capabilities.issues import (  # noqa: E402
    ISSUE_LIST_FIELDS,
    ISSUE_VIEW_FIELDS,
)
from charlie_work.github_capabilities.pull_requests import (  # noqa: E402
    PR_LIST_FIELDS,
    PR_VIEW_FIELDS,
)
from charlie_work.github_transport import gh_json_fields as g  # noqa: E402

OUT = REPO_ROOT / "tests" / "fixtures" / "gh_json"
CLOSING = "closingIssuesReferences,statusCheckRollup,labels,author,state,mergedAt,closedAt"
CHECKS = "name,state,bucket,link,workflow,event,startedAt,completedAt,description"

# (case name, resource, number, fields)
CASES = [
    ("issue_view_closed", "issue", 2090, ISSUE_VIEW_FIELDS),
    ("issue_list_row_closed", "issue", 2090, ISSUE_LIST_FIELDS),
    ("pr_view_merged", "pr", 2121, PR_VIEW_FIELDS + ",closingIssuesReferences,closedAt,mergedAt"),
    ("pr_view_open", "pr", 2130, PR_VIEW_FIELDS),
    ("pr_list_row_merged", "pr", 2121, PR_LIST_FIELDS),
    ("pr_closing_refs", "pr", 2121, CLOSING),
]
CHECK_CASES = [("pr_checks_merged", 2121), ("pr_checks_open", 2130)]
_SECRET = re.compile(r"(ghp_|gho_|ghs_|github_pat_)[A-Za-z0-9_]{10,}")


def _run(*argv: str) -> str:
    done = subprocess.run(
        list(argv), capture_output=True, text=True, encoding="utf-8", check=False
    )
    if done.returncode != 0:
        raise SystemExit(f"{' '.join(argv[:4])} failed: {done.stderr.strip()}")
    return done.stdout


def _repo() -> tuple[str, str]:
    data = json.loads(_run("gh", "repo", "view", "--json", "owner,name"))
    return data["owner"]["login"], data["name"]


def _graphql(document: str, **variables: object) -> dict:
    argv = ["gh", "api", "graphql", "-f", f"query={document}"]
    for key, value in variables.items():
        flag = "-F" if isinstance(value, int) else "-f"
        argv += [flag, f"{key}={value}"]
    return json.loads(_run(*argv))["data"]


def _write(name: str, payload: dict) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if _SECRET.search(text):
        raise SystemExit(f"{name}: token-shaped string in fixture; refusing to write")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {name}")


def main() -> None:
    owner, repo = _repo()
    for name, resource, number, fields in CASES:
        gh_json = json.loads(_run("gh", resource, "view", str(number), "--json", fields))
        document = g.document_for(resource, fields, "view")
        data = _graphql(document, owner=owner, name=repo, number=number)
        _write(
            name,
            {
                "resource": resource,
                "fields": fields,
                "number": number,
                "gh_json": gh_json,
                "graphql_data": data,
            },
        )
    for name, number in CHECK_CASES:
        gh_json = json.loads(_run("gh", "pr", "checks", str(number), "--json", CHECKS))
        data = _graphql(g.checks_document(), owner=owner, name=repo, number=number)
        _write(
            name,
            {
                "resource": "checks",
                "fields": CHECKS,
                "number": number,
                "gh_json": gh_json,
                "graphql_data": data,
            },
        )


if __name__ == "__main__":
    main()
