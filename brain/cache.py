"""Tiny SQLite key/value cache with TTL. Used for HTTP fetches, API responses and LLM calls,
so re-running a campaign costs almost nothing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from typing import Any, Optional


class Cache:
    def __init__(self, path: str = ":memory:", ignore_ttl: bool = False):
        self.ignore_ttl = ignore_ttl  # replay mode: a recorded answer never expires
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (ns TEXT, k TEXT, v TEXT, expires REAL, PRIMARY KEY (ns, k))"
        )
        self._db.commit()

    @staticmethod
    def key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, ns: str, key: str) -> Optional[Any]:
        with self._lock:
            row = self._db.execute("SELECT v, expires FROM cache WHERE ns=? AND k=?", (ns, key)).fetchone()
        if not row:
            return None
        value, expires = row
        if expires and expires < time.time() and not self.ignore_ttl:
            return None
        return json.loads(value)

    def set(self, ns: str, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires = time.time() + ttl if ttl else 0
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache (ns, k, v, expires) VALUES (?,?,?,?)",
                (ns, key, json.dumps(value), expires),
            )
            self._db.commit()
