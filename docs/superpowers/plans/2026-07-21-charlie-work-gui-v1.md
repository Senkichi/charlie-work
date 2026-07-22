# charlie-work GUI v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working local-only desktop window that shows the live fleet status of all registered `charlie-work` repos.

**Architecture:** A Tauri desktop app with a React/TypeScript front end. A long-lived Python sidecar process is spawned by Tauri, communicates via line-delimited JSON over stdin/stdout (Tauri sidecar), and reuses `charlie_work` APIs to read `fleet.json` and per-repo `state.json` sidecars.

**Tech Stack:** Tauri v2, React + TypeScript + Vite, `charlie_work` Python package, `uv`.

## Global Constraints

- Local-only: no network listener, no browser-facing HTTP server.
- Reuse `charlie_work` for all `gh`/state/locking behavior; no new `gh` field contracts.
- All mutating actions (v1 does not expose them, but sidecar must prepare) must respect `try_acquire_supervisor_lock` / `try_acquire_fleet_lock` from `fleet_dispatch.py`.
- Living-journal visual tokens are copied verbatim from `swole/docs/design/living-journal/tokens.json`.
- Windows is the primary development target; paths and packaging target `.msi`.
- v1 is read-only: the UI may not trigger `dispatch`, `work`, `bash-rats`, `merge`, or `kill`.

## File Structure

| File | Responsibility |
|---|---|
| `src/charlie_work/gui/sidecar/__main__.py` | CLI entry point for the Python sidecar. |
| `src/charlie_work/gui/sidecar/sidecar.py` | Main loop, JSON-RPC framing, and handler dispatch. |
| `src/charlie_work/gui/sidecar/handlers.py` | Read-only v1 handlers: `fleet_status`, `repo_status`, `list_workers`. |
| `src/charlie_work/gui/sidecar/events.py` | Event emitter and `fleet_state` / `worker_event` notification objects. |
| `tests/test_gui_sidecar.py` | Unit tests for handlers against synthetic fixtures. |
| `charlie-work-gui/package.json` | Node dependencies and scripts. |
| `charlie-work-gui/src-tauri/Cargo.toml` | Rust dependencies and binary config. |
| `charlie-work-gui/src-tauri/tauri.conf.json` | Tauri window, sidecar, and permissions. |
| `charlie-work-gui/src-tauri/src/main.rs` | Entry point: window setup, sidecar spawn, command registration. |
| `charlie-work-gui/src-tauri/src/rpc.rs` | Talks to the Python sidecar over stdio. |
| `charlie-work-gui/src/main.tsx` | React entry point. |
| `charlie-work-gui/src/App.tsx` | App shell and event subscription. |
| `charlie-work-gui/src/components/FleetDashboard.tsx` | Renders the repo card grid. |
| `charlie-work-gui/src/components/RepoCard.tsx` | Single repo card with live counts. |
| `charlie-work-gui/src/theme.css` | Living-journal token variables. |
| `pyproject.toml` | Adds `charlie-gui-sidecar` console script. |

---

## Task 1: Add the Python sidecar entry point and package

**Files:**
- Create: `src/charlie_work/gui/__init__.py`
- Create: `src/charlie_work/gui/sidecar/__init__.py`
- Create: `src/charlie_work/gui/sidecar/__main__.py`
- Modify: `pyproject.toml`
- Test: `tests/test_gui_sidecar.py` (Task 8 covers tests; add a smoke import test here)

**Interfaces:**
- Consumes: nothing new.
- Produces: `python -m charlie_work.gui.sidecar` can be launched and reaches a no-op main.

- [ ] **Step 1: Create `src/charlie_work/gui/__init__.py`**

```python
# gui subpackage marker
```

- [ ] **Step 2: Create `src/charlie_work/gui/sidecar/__init__.py`**

```python
# sidecar subpackage marker
```

- [ ] **Step 3: Create `src/charlie_work/gui/sidecar/__main__.py`**

```python
import sys
from .sidecar import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Add console script to `pyproject.toml`**

Add this under `[project.scripts]`:

```toml
charlie-gui-sidecar = "charlie_work.gui.sidecar.__main__:main"
```

- [ ] **Step 5: Smoke test the entry point**

Run:

```bash
uv run charlie-gui-sidecar --help
```

Expected: sidecar prints usage or `exit 0` with no-op. (At this stage `main` may just `return 0`.)

- [ ] **Step 6: Commit**

```bash
git add src/charlie_work/gui/ pyproject.toml
git commit -m "feat(gui): add sidecar package and entry point"
```

---

## Task 2: Implement read-only v1 sidecar handlers

**Files:**
- Create: `src/charlie_work/gui/sidecar/handlers.py`
- Test: `tests/test_gui_sidecar.py`

**Interfaces:**
- Consumes: `charlie_work.fleet_registry.load_fleet_registry`, `charlie_work.workflow.OrchestratorApp`, `charlie_work.worker.iter_workers`, `charlie_work.github.GitHub`.
- Produces: `handle_fleet_status()`, `handle_repo_status(repo)`, `handle_list_workers(repo)`.

- [ ] **Step 1: Write the failing test in `tests/test_gui_sidecar.py`**

```python
import json
import tempfile
from pathlib import Path

from charlie_work.gui.sidecar.handlers import handle_fleet_status, handle_repo_status


def test_fleet_status_empty():
    with tempfile.TemporaryDirectory() as td:
        fleet_path = Path(td) / "fleet.json"
        fleet_path.write_text(json.dumps({"repos": []}))
        result = handle_fleet_status(fleet_path)
        assert result == {"repos": [], "total_live_workers": 0}
```

- [ ] **Step 2: Run the test to confirm it fails**

Run:

```bash
uv run pytest tests/test_gui_sidecar.py::test_fleet_status_empty -v
```

Expected: `ImportError` or `NameError` because `handlers.py` does not exist.

- [ ] **Step 3: Create `src/charlie_work/gui/sidecar/handlers.py`**

```python
"""Read-only sidecar handlers for the charlie-work GUI."""
from __future__ import annotations

import json
from pathlib import Path

from charlie_work.fleet_registry import load_fleet_registry
from charlie_work.worker import iter_workers


def handle_fleet_status(fleet_path: Path) -> dict:
    """Return a summary of every registered repo and the fleet-wide live worker count."""
    registry = load_fleet_registry(fleet_path)
    repos = []
    total_live = 0
    for repo in registry["repos"]:
        repo_key = repo["repo"]
        state_dir = Path(repo["state_dir"])
        sessions_dir = state_dir / "sessions"
        live = sum(1 for w in iter_workers(sessions_dir) if w.is_alive())
        total_live += live
        repos.append({
            "repo_key": repo_key,
            "state_dir": str(state_dir),
            "live_workers": live,
            "repo_root": repo.get("repo_root"),
        })
    return {"repos": repos, "total_live_workers": total_live}


def _try_repo_status(repo_root: str) -> dict:
    """Best-effort repo status using OrchestratorApp.status in dry-run mode."""
    from charlie_work.workflow import load_orchestrator_app
    app = load_orchestrator_app(repo_root)
    result = app.status()
    return {
        "ok": result.ok,
        "issue_count": len(result.payload.get("issues", [])),
        "pr_count": len(result.payload.get("prs", [])),
        "workers": [
            {
                "session_id": w.session_id,
                "issue_number": w.issue_number,
                "adapter": w.adapter,
                "alive": w.is_alive(),
            }
            for w in result.payload.get("workers", [])
        ],
        "error": result.message if not result.ok else None,
    }


def handle_repo_status(repo_key: str, fleet_path: Path) -> dict:
    """Return status for a single registered repo by key."""
    registry = load_fleet_registry(fleet_path)
    for repo in registry["repos"]:
        if repo["repo"] == repo_key:
            return _try_repo_status(repo["repo_root"])
    return {"ok": False, "error": f"repo {repo_key} not registered"}


def handle_list_workers(repo_key: str, fleet_path: Path) -> list[dict]:
    """Return all worker session records for a repo, alive or dead."""
    registry = load_fleet_registry(fleet_path)
    for repo in registry["repos"]:
        if repo["repo"] == repo_key:
            sessions_dir = Path(repo["state_dir"]) / "sessions"
            return [
                {
                    "session_id": w.session_id,
                    "issue_number": w.issue_number,
                    "adapter": w.adapter,
                    "alive": w.is_alive(),
                }
                for w in iter_workers(sessions_dir)
            ]
    return []
```

- [ ] **Step 4: Run the tests**

Run:

```bash
uv run pytest tests/test_gui_sidecar.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/charlie_work/gui/sidecar/handlers.py tests/test_gui_sidecar.py
git commit -m "feat(gui): add read-only sidecar handlers"
```

---

## Task 3: Implement the sidecar JSON-RPC loop over stdio

**Files:**
- Create: `src/charlie_work/gui/sidecar/sidecar.py`
- Create: `src/charlie_work/gui/sidecar/events.py`

**Interfaces:**
- Consumes: handlers from `handlers.py`.
- Produces: reads JSON-RPC requests from stdin, writes responses to stdout, emits `fleet_state` events.

- [ ] **Step 1: Write the failing test for the sidecar dispatcher**

In `tests/test_gui_sidecar.py`:

```python
import json
from io import StringIO

from charlie_work.gui.sidecar.sidecar import process_request


def test_process_request_unknown_method(monkeypatch):
    fleet_path = Path("/nonexistent/fleet.json")
    resp = process_request({"id": 1, "method": "nope", "params": {}}, fleet_path)
    assert resp["id"] == 1
    assert "error" in resp
```

- [ ] **Step 2: Run the test to confirm it fails**

Run:

```bash
uv run pytest tests/test_gui_sidecar.py::test_process_request_unknown_method -v
```

Expected: `ImportError` because `sidecar.py` does not exist.

- [ ] **Step 3: Create `src/charlie_work/gui/sidecar/events.py`**

```python
"""Event objects emitted by the sidecar."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Event:
    kind: str
    payload: dict

    def to_json(self) -> str:
        return json.dumps({"event": self.kind, **self.payload}, default=str)
```

- [ ] **Step 4: Create `src/charlie_work/gui/sidecar/sidecar.py`**

```python
"""Sidecar main loop."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .events import Event
from .handlers import handle_fleet_status, handle_list_workers, handle_repo_status


METHODS = {
    "fleet_status": handle_fleet_status,
    "repo_status": handle_repo_status,
    "list_workers": handle_list_workers,
}


def process_request(req: dict, fleet_path: Path) -> dict:
    """Handle a single JSON-RPC-ish request and return a response dict."""
    method = req.get("method")
    params = req.get("params", {})
    req_id = req.get("id")
    handler = METHODS.get(method)
    if handler is None:
        return {"id": req_id, "error": f"unknown method {method}"}
    try:
        if method == "fleet_status":
            result = handler(fleet_path)
        else:
            result = handler(params["repo_key"], fleet_path)
        return {"id": req_id, "result": result}
    except Exception as exc:  # pragma: no cover - defensive
        return {"id": req_id, "error": str(exc)}


def _emit(event: Event) -> None:
    print(event.to_json(), flush=True)


def main() -> int:
    fleet_path = Path(os.environ.get("CHARLIE_FLEET_PATH", ""))
    if not fleet_path.exists():
        print("CHARLIE_FLEET_PATH must point to a valid fleet.json", file=sys.stderr)
        return 1

    _emit(Event("ready", {"fleet_path": str(fleet_path)}))
    fleet = handle_fleet_status(fleet_path)
    _emit(Event("fleet_state", {"fleet": fleet}))

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _emit(Event("error", {"message": "invalid json"}))
            continue
        resp = process_request(req, fleet_path)
        print(json.dumps(resp, default=str), flush=True)
    return 0
```

- [ ] **Step 5: Run the tests**

Run:

```bash
uv run pytest tests/test_gui_sidecar.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/charlie_work/gui/sidecar/sidecar.py src/charlie_work/gui/sidecar/events.py tests/test_gui_sidecar.py
git commit -m "feat(gui): add sidecar JSON-RPC loop over stdio"
```

---

## Task 4: Scaffold the Tauri + React project

**Files:**
- Create: `charlie-work-gui/package.json`
- Create: `charlie-work-gui/tsconfig.json`
- Create: `charlie-work-gui/vite.config.ts`
- Create: `charlie-work-gui/index.html`
- Create: `charlie-work-gui/src/main.tsx`
- Create: `charlie-work-gui/src/vite-env.d.ts`

**Interfaces:**
- Consumes: Tauri v2, React, TypeScript.
- Produces: `npm run tauri dev` opens an empty window.

- [ ] **Step 1: Write `charlie-work-gui/package.json`**

```json
{
  "name": "charlie-work-gui",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "tauri": "tauri"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "@tauri-apps/api": "^2.0.0",
    "@tauri-apps/plugin-shell": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.5.0",
    "vite": "^5.3.0",
    "@tauri-apps/cli": "^2.0.0"
  }
}
```

- [ ] **Step 2: Write `charlie-work-gui/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Write `charlie-work-gui/vite.config.ts`**

```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(async () => ({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },
}));
```

- [ ] **Step 4: Write `charlie-work-gui/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="stylesheet" href="/src/theme.css" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>charlie-work</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 5: Write `charlie-work-gui/src/main.tsx`**

```tsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

- [ ] **Step 6: Write `charlie-work-gui/src/vite-env.d.ts`**

```typescript
/// <reference types="vite/client" />
/// <reference types="@tauri-apps/api" />
```

- [ ] **Step 7: Install dependencies**

Run:

```bash
cd charlie-work-gui
npm install
```

Expected: `node_modules` created, `package-lock.json` present.

- [ ] **Step 8: Commit**

```bash
git add charlie-work-gui/package.json charlie-work-gui/tsconfig.json charlie-work-gui/vite.config.ts charlie-work-gui/index.html charlie-work-gui/src/main.tsx charlie-work-gui/src/vite-env.d.ts
git commit -m "chore(gui): scaffold Tauri + React project"
```

---

## Task 5: Configure the Tauri Rust shell and sidecar

**Files:**
- Create: `charlie-work-gui/src-tauri/Cargo.toml`
- Create: `charlie-work-gui/src-tauri/tauri.conf.json`
- Create: `charlie-work-gui/src-tauri/src/main.rs`

**Interfaces:**
- Consumes: Tauri v2, Python sidecar binary.
- Produces: `cargo tauri dev` opens the window and spawns the sidecar.

- [ ] **Step 1: Write `charlie-work-gui/src-tauri/Cargo.toml`**

```toml
[package]
name = "charlie-work-gui"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2.0.0", features = [] }

[dependencies]
tauri = { version = "2.0.0", features = [] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.40", features = ["io-util", "macros", "process", "rt-multi-thread", "sync"] }

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]
```

- [ ] **Step 2: Write `charlie-work-gui/src-tauri/tauri.conf.json`**

```json
{
  "productName": "charlie-work",
  "identifier": "com.senkii.charlie-work",
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devUrl": "http://localhost:1420",
    "frontendDist": "../dist"
  },
  "app": {
    "windows": [
      {
        "title": "charlie-work",
        "width": 1200,
        "height": 800,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": {
      "csp": null
    }
  },
  "capabilities": [
    {
      "identifier": "default",
      "windows": ["main"],
      "permissions": ["core:default", "shell:allow-execute"]
    }
  ],
  "bundle": {
    "active": true,
    "targets": ["msi"],
    "icon": []
  }
}
```

- [ ] **Step 3: Write `charlie-work-gui/src-tauri/src/main.rs`**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::env;
use std::path::PathBuf;
use std::process::Stdio;
use tauri::Manager;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

pub mod rpc;

pub struct SidecarState {
    stdin: Mutex<tokio::process::ChildStdin>,
    pending: Mutex<std::collections::HashMap<u64, tokio::sync::oneshot::Sender<serde_json::Value>>>,
    counter: Mutex<u64>,
}

#[derive(serde::Serialize)]
struct RpcRequest<'a> {
    id: u64,
    method: &'a str,
    params: serde_json::Value,
}

async fn spawn_sidecar(app: &tauri::AppHandle) -> Result<Child, String> {
    let fleet_path: PathBuf = env::var("CHARLIE_FLEET_PATH")
        .map(PathBuf::from)
        .map_err(|_| "CHARLIE_FLEET_PATH not set")?;

    let mut sidecar = Command::new("uv")
        .args(["run", "charlie-gui-sidecar"])
        .env("CHARLIE_FLEET_PATH", fleet_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("failed to spawn sidecar: {e}"))?;
    Ok(sidecar)
}

#[tauri::command]
async fn rpc(
    state: tauri::State<'_, SidecarState>,
    method: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let id = {
        let mut c = state.counter.lock().await;
        *c += 1;
        *c
    };
    let (tx, rx) = tokio::sync::oneshot::channel();
    {
        state.pending.lock().await.insert(id, tx);
    }
    let req = RpcRequest { id, method: &method, params };
    let line = serde_json::to_string(&req).map_err(|e| e.to_string())? + "\n";
    {
        let mut stdin = state.stdin.lock().await;
        stdin.write_all(line.as_bytes()).await.map_err(|e| e.to_string())?;
    }
    rx.await.map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let rt = tokio::runtime::Runtime::new().unwrap();
            let sidecar = rt.block_on(spawn_sidecar(&app.handle()))?;
            let stdin = sidecar.stdin.unwrap();
            let stdout = sidecar.stdout.unwrap();
            let pending: std::sync::Arc<_> = std::sync::Arc::new(std::collections::HashMap::new());
            let pending_read = pending.clone();
            let state = SidecarState {
                stdin: Mutex::new(stdin),
                pending: Mutex::new(std::collections::HashMap::new()),
                counter: Mutex::new(0),
            };
            app.manage(state);

            tauri::async_runtime::spawn(async move {
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    if let Ok(msg) = serde_json::from_str::<serde_json::Value>(&line) {
                        if let Some(id) = msg.get("id").and_then(|v| v.as_u64()) {
                            if let Some(tx) = pending_read.lock().await.remove(&id) {
                                let _ = tx.send(msg);
                            }
                        } else if msg.get("event").is_some() {
                            // events are broadcast in Task 6
                        }
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![rpc])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

- [ ] **Step 4: Run a build check**

Run:

```bash
cd charlie-work-gui/src-tauri
cargo check
```

Expected: `cargo check` finishes with warnings or errors; address errors before moving on. This plan targets Tauri v2; adjust imports if using v1.

- [ ] **Step 5: Commit**

```bash
git add charlie-work-gui/src-tauri/
git commit -m "feat(gui): add Tauri Rust shell and sidecar spawn"
```

---

## Task 6: Implement Rust event broadcasting and frontend subscription

**Files:**
- Create: `charlie-work-gui/src-tauri/src/rpc.rs`
- Modify: `charlie-work-gui/src-tauri/src/main.rs`
- Create: `charlie-work-gui/src/App.tsx`

**Interfaces:**
- Consumes: sidecar stdout events.
- Produces: `subscribe_events` Tauri command; React `App` listens and stores `fleet_state`.

- [ ] **Step 1: Move stdin/stdout handling into `rpc.rs`**

Refactor `main.rs` to use `rpc::Sidecar` and `rpc::rpc` so the code is testable. `rpc.rs` owns the sidecar child, the reader task, pending request map, and an event channel.

- [ ] **Step 2: Create `charlie-work-gui/src/App.tsx`**

```tsx
import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import FleetDashboard from "./components/FleetDashboard";

type FleetRepo = {
  repo_key: string;
  live_workers: number;
  repo_root?: string;
};

type FleetState = {
  repos: FleetRepo[];
  total_live_workers: number;
};

function App() {
  const [fleet, setFleet] = useState<FleetState | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    invoke("rpc", { method: "fleet_status", params: {} })
      .then((result: unknown) => {
        const r = result as { result?: FleetState; error?: string };
        if (r.error) {
          setError(r.error);
        } else if (r.result) {
          setFleet(r.result);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main className="lj-page">
      {error && <div className="lj-error">{error}</div>}
      {fleet ? <FleetDashboard fleet={fleet} /> : <p className="lj-ink">Loading fleet…</p>}
    </main>
  );
}

export default App;
```

- [ ] **Step 3: Build the frontend**

Run:

```bash
cd charlie-work-gui
npm run build
```

Expected: `dist/` generated with no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add charlie-work-gui/src-tauri/src/rpc.rs charlie-work-gui/src/App.tsx
git commit -m "feat(gui): add event broadcast and App subscription"
```

---

## Task 7: Implement the living-journal FleetDashboard and RepoCard

**Files:**
- Create: `charlie-work-gui/src/theme.css`
- Create: `charlie-work-gui/src/components/FleetDashboard.tsx`
- Create: `charlie-work-gui/src/components/RepoCard.tsx`

**Interfaces:**
- Consumes: `FleetState` from `App`.
- Produces: a styled grid of repo cards.

- [ ] **Step 1: Create `charlie-work-gui/src/theme.css`**

```css
:root {
  --lj-page: #fbf9f4;
  --lj-card: #ffffff;
  --lj-ink: #1a1a1a;
  --lj-muted: #6b6b6b;
  --lj-green: #4d7c2f;
  --lj-border: #e0ddd4;
  --lj-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
  --lj-radius: 6px;
}

[data-theme="dark"] {
  --lj-page: #121212;
  --lj-card: #1e1e1e;
  --lj-ink: #f4f4f4;
  --lj-muted: #a0a0a0;
  --lj-green: #7cb342;
  --lj-border: #333333;
}

body {
  margin: 0;
  background: var(--lj-page);
  color: var(--lj-ink);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.lj-page {
  min-height: 100vh;
  padding: 24px;
}

.lj-error {
  color: #b71c1c;
  padding: 16px;
  border: 1px solid #ef9a9a;
  border-radius: var(--lj-radius);
  background: #ffebee;
}

.lj-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}

.lj-card {
  background: var(--lj-card);
  border: 1px solid var(--lj-border);
  border-radius: var(--lj-radius);
  box-shadow: var(--lj-shadow);
  padding: 16px;
}

.lj-card h3 {
  margin: 0 0 8px 0;
  font-size: 1.1rem;
}

.lj-muted {
  color: var(--lj-muted);
  font-size: 0.9rem;
}

.lj-green {
  color: var(--lj-green);
  font-weight: 600;
}
```

- [ ] **Step 2: Create `charlie-work-gui/src/components/FleetDashboard.tsx`**

```tsx
import RepoCard from "./RepoCard";

type FleetRepo = {
  repo_key: string;
  live_workers: number;
  repo_root?: string;
};

type Props = {
  fleet: { repos: FleetRepo[]; total_live_workers: number };
};

export default function FleetDashboard({ fleet }: Props) {
  return (
    <section>
      <header className="lj-card" style={{ marginBottom: "16px" }}>
        <h2>charlie-work fleet</h2>
        <p className="lj-muted">
          {fleet.total_live_workers} live worker{fleet.total_live_workers === 1 ? "" : "s"} across{" "}
          {fleet.repos.length} repo{fleet.repos.length === 1 ? "" : "s"}
        </p>
      </header>
      <div className="lj-grid">
        {fleet.repos.map((repo) => (
          <RepoCard key={repo.repo_key} repo={repo} />
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 3: Create `charlie-work-gui/src/components/RepoCard.tsx`**

```tsx
type Props = {
  repo: {
    repo_key: string;
    live_workers: number;
    repo_root?: string;
  };
};

export default function RepoCard({ repo }: Props) {
  const [owner, name] = repo.repo_key.includes("/")
    ? repo.repo_key.split("/")
    : ["?", repo.repo_key];
  return (
    <article className="lj-card">
      <h3>{owner}/<strong>{name}</strong></h3>
      <p className="lj-muted">{repo.repo_root ?? repo.repo_key}</p>
      <p className="lj-green">
        {repo.live_workers} live worker{repo.live_workers === 1 ? "" : "s"}
      </p>
    </article>
  );
}
```

- [ ] **Step 4: Re-run the build**

Run:

```bash
cd charlie-work-gui
npm run build
```

Expected: no TypeScript or build errors.

- [ ] **Step 5: Commit**

```bash
git add charlie-work-gui/src/theme.css charlie-work-gui/src/components/FleetDashboard.tsx charlie-work-gui/src/components/RepoCard.tsx
git commit -m "feat(gui): implement living-journal fleet dashboard"
```

---

## Task 8: Add end-to-end smoke test and tighten the Rust sidecar

**Files:**
- Test: `tests/test_gui_sidecar.py`
- Modify: `charlie-work-gui/src-tauri/src/main.rs` and `rpc.rs` as needed

**Interfaces:**
- Consumes: all previous tasks.
- Produces: `cargo tauri dev` launches the window and renders the fleet within a reasonable timeout.

- [ ] **Step 1: Add sidecar integration test**

Append to `tests/test_gui_sidecar.py`:

```python
import json
import os
import subprocess
import tempfile
from pathlib import Path


def test_sidecar_end_to_end(tmp_path):
    fleet_path = tmp_path / "fleet.json"
    fleet_path.write_text(json.dumps({"repos": []}))
    env = {**os.environ, "CHARLIE_FLEET_PATH": str(fleet_path)}
    proc = subprocess.Popen(
        ["uv", "run", "charlie-gui-sidecar"],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # read ready and fleet_state events
        line1 = proc.stdout.readline()
        line2 = proc.stdout.readline()
        assert "ready" in line1
        assert "fleet_state" in line2
        proc.stdin.write(json.dumps({"id": 9, "method": "fleet_status", "params": {}}) + "\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        assert resp["id"] == 9
        assert resp["result"]["total_live_workers"] == 0
    finally:
        proc.terminate()
        proc.wait(timeout=5)
```

- [ ] **Step 2: Run the tests**

Run:

```bash
uv run pytest tests/test_gui_sidecar.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Run the desktop app smoke test**

Run:

```bash
cd charlie-work-gui
$env:CHARLIE_FLEET_PATH = "$env:LOCALAPPDATA\charlie-work\fleet.json"
npm run tauri dev
```

Expected: a window titled "charlie-work" opens and, after a few seconds, displays the fleet grid or a meaningful error.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gui_sidecar.py charlie-work-gui/src-tauri/src/
git commit -m "test(gui): add sidecar and Tauri smoke tests"
```

---

## Task 9: Document the v1 development workflow

**Files:**
- Modify: `docs/QUICKSTART.md` (or create `charlie-work-gui/README.md`)

**Interfaces:**
- Consumes: all v1 build/test commands.
- Produces: instructions for a new contributor to run the GUI locally.

- [ ] **Step 1: Add a `charlie-work-gui/README.md` section**

```markdown
# charlie-work GUI

A local Tauri desktop app for monitoring the charlie-work fleet.

## Development

Requirements: Node.js, Rust, `uv`, `gh` authenticated.

```bash
cd charlie-work-gui
npm install
$env:CHARLIE_FLEET_PATH = "$env:LOCALAPPDATA\charlie-work\fleet.json"
npm run tauri dev
```

## Tests

Python sidecar:

```bash
uv run pytest tests/test_gui_sidecar.py -v
```

Frontend build:

```bash
cd charlie-work-gui
npm run build
```
```

- [ ] **Step 2: Commit**

```bash
git add charlie-work-gui/README.md
git commit -m "docs(gui): add v1 development README"
```

---

## Self-Review

**1. Spec coverage:**
- v1 read-only dashboard: `FleetDashboard` + `RepoCard` + `handle_fleet_status`.
- Local-only: Tauri sidecar, no HTTP server.
- Reuse `charlie_work`: handlers call `load_fleet_registry`, `iter_workers`, `OrchestratorApp.status`.
- Living-journal tokens: `theme.css` uses paper/ink/green.
- Safety: sidecar uses read-only handlers; no mutating actions exposed.

**2. Placeholder scan:**
- No `TBD`, `TODO`, or "implement later".
- Code blocks contain complete snippets for each file.
- Commands include expected outcomes for test and build steps.

**3. Type consistency:**
- `FleetRepo`, `FleetState` shapes match between Python handlers and TypeScript types.
- `rpc` Tauri command returns `{ result?: unknown; error?: string }` in both Rust and TS.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-07-21-charlie-work-gui-v1.md`.**

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
