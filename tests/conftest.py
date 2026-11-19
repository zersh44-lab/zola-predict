"""Pytest fixtures and mock API payloads for Zola Predict."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Callable, Dict, Generator, List, Optional
import pytest

from zola.database.connection import SCHEMA_SQL
from zola.database.repository import FootballRepository

# Mock raw response for GET /v4/competitions
MOCK_COMPETITIONS_PAYLOAD = {
    "count": 2,
    "competitions": [
        {
            "id": 2021,
            "name": "Premier League",
            "code": "PL",
            "type": "LEAGUE",
            "emblem": "https://crests.football-data.org/PL.png",
            "area": {"name": "England", "code": "ENG"},
            "currentSeason": {"id": 1564, "startDate": "2024-08-16", "endDate": "2025-05-25", "currentMatchday": 12},
        },
        {
            "id": 2014,
            "name": "Primera Division",
            "code": "PD",
            "type": "LEAGUE",
            "emblem": "https://crests.football-data.org/laliga.png",
            "area": {"name": "Spain", "code": "ESP"},
            "currentSeason": {"id": 1565, "startDate": "2024-08-15", "endDate": "2025-05-24", "currentMatchday": 13},
        },
    ],
}

# Mock raw response for GET /v4/competitions/PL/matches?status=SCHEDULED
MOCK_SCHEDULED_FIXTURES_PAYLOAD = {
    "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
    "matches": [
        {
            "id": 5001,
            "utcDate": "2024-11-23T12:30:00Z",
            "status": "TIMED",
            "matchday": 12,
            "homeTeam": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS", "crest": "https://crests/arsenal.png"},
            "awayTeam": {"id": 65, "name": "Manchester City FC", "shortName": "Man City", "tla": "MCI", "crest": "https://crests/city.png"},
            "score": {
                "winner": None,
                "duration": "REGULAR",
                "fullTime": {"home": None, "away": None},
                "halfTime": {"home": None, "away": None},
            },
            "venue": "Emirates Stadium",
            "referees": [{"name": "Michael Oliver"}],
            "season": {"startDate": "2024-08-16"},
        },
        {
            "id": 5002,
            "utcDate": "2024-11-23T15:00:00Z",
            "status": "TIMED",
            "matchday": 12,
            "homeTeam": {"id": 64, "name": "Liverpool FC", "shortName": "Liverpool", "tla": "LIV", "crest": "https://crests/liverpool.png"},
            "awayTeam": {"id": 61, "name": "Chelsea FC", "shortName": "Chelsea", "tla": "CHE", "crest": "https://crests/chelsea.png"},
            "score": {
                "winner": None,
                "duration": "REGULAR",
                "fullTime": {"home": None, "away": None},
                "halfTime": {"home": None, "away": None},
            },
            "venue": "Anfield",
            "referees": [{"name": "Anthony Taylor"}],
            "season": {"startDate": "2024-08-16"},
        },
    ],
}

# Mock raw response for GET /v4/competitions/PL/matches?status=FINISHED
MOCK_HISTORICAL_RESULTS_PAYLOAD = {
    "competition": {"id": 2021, "name": "Premier League", "code": "PL"},
    "matches": [
        {
            "id": 5001,
            "utcDate": "2024-11-23T12:30:00Z",
            "status": "FINISHED",
            "matchday": 12,
            "homeTeam": {"id": 57, "name": "Arsenal FC", "shortName": "Arsenal", "tla": "ARS", "crest": "https://crests/arsenal.png"},
            "awayTeam": {"id": 65, "name": "Manchester City FC", "shortName": "Man City", "tla": "MCI", "crest": "https://crests/city.png"},
            "score": {
                "winner": "HOME_TEAM",
                "duration": "REGULAR",
                "fullTime": {"home": 2, "away": 1},
                "halfTime": {"home": 1, "away": 0},
                "extraTime": {"home": None, "away": None},
                "penalties": {"home": None, "away": None},
            },
            "venue": "Emirates Stadium",
            "referees": [{"name": "Michael Oliver"}],
            "season": {"startDate": "2024-08-16"},
        }
    ],
}


class MockHttpResponse:
    """Pure-Python mock HTTP response."""

    def __init__(
        self,
        status_code: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        raw_text: Optional[str] = None,
    ):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._json_data = json_data or {}
        self.text = raw_text if raw_text is not None else json.dumps(self._json_data)
        self.headers = headers or {}

    def json(self) -> Dict[str, Any]:
        return self._json_data


class MockHttpSession:
    """Pure-Python mock requests.Session to avoid external asyncio/C-extension dependencies."""

    def __init__(self, handler: Optional[Callable[..., MockHttpResponse]] = None):
        self.handler = handler
        self.calls: List[Dict[str, Any]] = []

    def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> MockHttpResponse:
        call_info = {
            "method": method,
            "url": url,
            "headers": headers or {},
            "params": params or {},
            "timeout": timeout,
        }
        self.calls.append(call_info)
        if self.handler:
            return self.handler(method, url, headers=headers, params=params, timeout=timeout)
        return MockHttpResponse(200, {})


@pytest.fixture
def in_memory_db() -> Generator[sqlite3.Connection, None, None]:
    """Provide an in-memory SQLite database initialized with schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    yield conn
    conn.close()


@pytest.fixture
def repo(in_memory_db: sqlite3.Connection) -> FootballRepository:
    """Provide repository backed by in-memory database."""
    return FootballRepository(in_memory_db)
