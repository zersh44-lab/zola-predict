"""Vercel Serverless Function entry point for Zola Predict."""

from __future__ import annotations

import os
import sys
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
        matched = (
            self.headers.get("x-matched-path")
            or self.headers.get("x-forwarded-uri")
        )
        if matched and (self.path.startswith("/api/index.py") or self.path == "/api"):
            if "?" in self.path and "?" not in matched:
                matched += "?" + self.path.split("?", 1)[1]
            self.path = matched

    def do_GET(self) -> None:
        self._normalize_vercel_path()
        super().do_GET()

    def do_POST(self) -> None:
        self._normalize_vercel_path()
        super().do_POST()

    def do_OPTIONS(self) -> None:
        self._normalize_vercel_path()
        super().do_OPTIONS()
