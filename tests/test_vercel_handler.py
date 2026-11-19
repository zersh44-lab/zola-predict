"""Unit tests for the Vercel serverless request handler."""

import io
import json
import urllib.parse
from http.server import HTTPServer
from unittest.mock import MagicMock

import pytest
from api.index import handler
from zola.config import Settings


def test_vercel_normalize_path_route_param():
    """Verify that _normalize_vercel_path extracts __route__ and restores path and query."""
    mock_request = MagicMock()
    mock_server = MagicMock()
    mock_server.settings = Settings()

    # Create handler instance without calling standard BaseHTTPRequestHandler socket init
    h = handler.__new__(handler)
    h.headers = {}
    h.path = "/api/index.py?__route__=/api/fixtures&status=SCHEDULED&limit=25"

    h._normalize_vercel_path()

    assert h.path.startswith("/api/fixtures?")
    parsed = urllib.parse.urlparse(h.path)
    assert parsed.path == "/api/fixtures"
    qs = urllib.parse.parse_qs(parsed.query)
    assert qs["status"] == ["SCHEDULED"]
    assert qs["limit"] == ["25"]
    assert "__route__" not in qs


def test_vercel_normalize_path_status_route():
    """Verify that _normalize_vercel_path handles plain route."""
    h = handler.__new__(handler)
    h.headers = {}
    h.path = "/api/index.py?__route__=/api/status"

    h._normalize_vercel_path()

    assert h.path == "/api/status"
