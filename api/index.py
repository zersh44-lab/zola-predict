"""Vercel Serverless Function entry point for Zola Predict."""

from __future__ import annotations

import os
import sys
import urllib.parse
from pathlib import Path

# Ensure root workspace directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Pre-seed serverless SQLite database in /tmp if running on Vercel
from zola.database.connection import resolve_db_path
resolve_db_path()

from zola.dashboard.server import DashboardRequestHandler


class handler(DashboardRequestHandler):
    """Vercel serverless request handler inheriting from DashboardRequestHandler."""

    def _normalize_vercel_path(self) -> None:
        """Restore true client URL path if rewritten by Vercel Edge routing."""
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

        if "__route__" in qs:
            actual_path = qs.pop("__route__")[0]
            if not actual_path.startswith("/"):
                actual_path = "/" + actual_path
            pairs = []
            for k, vals in qs.items():
                for v in vals:
                    pairs.append((k, v))
            remaining = urllib.parse.urlencode(pairs)
            self.path = actual_path + ("?" + remaining if remaining else "")
            return

        # Fallback to headers if present and valid
        for hdr in ("x-now-route-matches", "x-matched-path", "x-forwarded-uri"):
            val = self.headers.get(hdr)
            if val and not val.startswith("/api/index.py") and val != "/api":
                self.path = val
                return

    def do_GET(self) -> None:
        self._normalize_vercel_path()
        super().do_GET()

    def do_POST(self) -> None:
        self._normalize_vercel_path()
        super().do_POST()

    def do_OPTIONS(self) -> None:
        self._normalize_vercel_path()
        super().do_OPTIONS()
