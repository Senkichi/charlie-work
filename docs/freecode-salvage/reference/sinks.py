from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RotatingPolicy:
    max_bytes: int = 16 * 1024 * 1024
    max_rotations: int = 5
    retention_days: int = 30


class JsonlFileSink:
    """Daily JSONL decision-log sink with bounded size rotation."""

    def __init__(self, log_dir: Path, *, policy: RotatingPolicy | None = None) -> None:
        self.log_dir = log_dir
        self.policy = policy or RotatingPolicy()
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write_record(self, record: dict[str, Any]) -> None:
        self._prune_old_logs()
        path = self.current_path()
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        self._rotate_if_needed(path, len(line.encode("utf-8")))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def write(self, text: str) -> None:
        for line in text.splitlines():
            if not line.strip():
                continue
            self.write_record(json.loads(line))

    def current_path(self) -> Path:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        return self.log_dir / f"decisions-{today}.jsonl"

    def _rotate_if_needed(self, path: Path, incoming_bytes: int) -> None:
        if not path.exists() or path.stat().st_size + incoming_bytes <= self.policy.max_bytes:
            return
        oldest = path.with_name(f"{path.name}.{self.policy.max_rotations}")
        if oldest.exists():
            oldest.unlink()
        for idx in range(self.policy.max_rotations - 1, 0, -1):
            src = path.with_name(f"{path.name}.{idx}")
            if src.exists():
                src.rename(path.with_name(f"{path.name}.{idx + 1}"))
        path.rename(path.with_name(f"{path.name}.1"))

    def _prune_old_logs(self) -> None:
        if self.policy.retention_days <= 0:
            return
        cutoff = datetime.now(UTC).timestamp() - (self.policy.retention_days * 24 * 60 * 60)
        for path in self.log_dir.glob("decisions-*.jsonl*"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue


class HmacChainSink:
    """Adds per-line HMAC chain fields before delegating to another sink."""

    def __init__(self, inner: JsonlFileSink, *, seed: bytes) -> None:
        if not seed:
            raise ValueError("HMAC seed must be non-empty")
        self.inner = inner
        self.seed = seed
        self._prev_hash = self._load_last_hash()

    def write_record(self, record: dict[str, Any]) -> None:
        chained = dict(record)
        chained["prev_hash"] = self._prev_hash
        payload = json.dumps(chained, sort_keys=True, separators=(",", ":"), default=str)
        digest = hmac.new(self.seed, payload.encode("utf-8"), hashlib.sha256).hexdigest()
        chained["hash"] = digest
        self.inner.write_record(chained)
        self._prev_hash = digest

    def write(self, text: str) -> None:
        for line in text.splitlines():
            if not line.strip():
                continue
            self.write_record(json.loads(line))

    def _load_last_hash(self) -> str | None:
        paths = sorted(self.inner.log_dir.glob("decisions-*.jsonl*"))
        for path in reversed(paths):
            try:
                lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
            except OSError:
                continue
            for line in reversed(lines):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                digest = obj.get("hash")
                if isinstance(digest, str) and digest:
                    return digest
        return None
