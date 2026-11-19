"""Persistent SQLite request caching with Time-To-Live (TTL) expiration."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union


@dataclass
class CachedResponse:
    """Cached HTTP response representation."""

    status_code: int
    text: str
    created_at: str
    expires_at: str
    cache_hit: bool = True

    def json(self) -> Any:
        """Parse cached JSON body."""
        return json.loads(self.text)


class RequestCache:
    """SQLite-backed persistent cache for API responses to protect quota."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @staticmethod
    def compute_cache_key(method: str, url: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Generate deterministic cache key using SHA-256."""
        norm_method = method.upper().strip()
        norm_url = url.strip().lower()
        norm_params = json.dumps(params or {}, sort_keys=True)
        raw = f"{norm_method}:{norm_url}:{norm_params}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, method: str, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[CachedResponse]:
        """Retrieve valid (unexpired) cached response if available."""
        cache_key = self.compute_cache_key(method, url, params)
        now_iso = datetime.now(timezone.utc).isoformat()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT status_code, response_body, created_at, expires_at
            FROM request_cache
            WHERE cache_key = ? AND expires_at > ?
            LIMIT 1;
            """,
            (cache_key, now_iso),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return CachedResponse(
            status_code=row["status_code"],
            text=row["response_body"],
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            cache_hit=True,
        )

    def set(
        self,
        method: str,
        url: str,
        status_code: int,
        response_body: str,
        ttl_minutes: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store API response in cache with specified TTL in minutes."""
        cache_key = self.compute_cache_key(method, url, params)
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=ttl_minutes)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO request_cache (
                cache_key, method, url, params_json, status_code, response_body, created_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                status_code = excluded.status_code,
                response_body = excluded.response_body,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at;
            """,
            (
                cache_key,
                method.upper().strip(),
                url.strip(),
                json.dumps(params or {}, sort_keys=True),
                status_code,
                response_body,
                now.isoformat(),
                expires_at.isoformat(),
            ),
        )
        self.conn.commit()

    def clear_expired(self) -> int:
        """Remove expired entries from the cache."""
        now_iso = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM request_cache WHERE expires_at <= ?;", (now_iso,))
        deleted_count = cursor.rowcount
        self.conn.commit()
        return deleted_count

    def clear_all(self) -> int:
        """Purge entire request cache."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM request_cache;")
        deleted_count = cursor.rowcount
        self.conn.commit()
        return deleted_count
