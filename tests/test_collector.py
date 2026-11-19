"""Tests for data collection service and end-to-end local persistence."""

from __future__ import annotations

import pytest

from zola.client.football_data import FootballDataOrgClient
from zola.database.repository import FootballRepository
from zola.services.collector import FootballCollectorService
from tests.conftest import (
    MOCK_COMPETITIONS_PAYLOAD,
    MOCK_HISTORICAL_RESULTS_PAYLOAD,
    MOCK_SCHEDULED_FIXTURES_PAYLOAD,
    MockHttpResponse,
    MockHttpSession,
)


@pytest.fixture
def mock_client() -> FootballDataOrgClient:
    def handler(method: str, url: str, **kwargs) -> MockHttpResponse:
        if "competitions/PL/matches" in url:
            params = kwargs.get("params") or {}
            if params.get("status") == "FINISHED":
                payload = MOCK_HISTORICAL_RESULTS_PAYLOAD
            else:
                payload = MOCK_SCHEDULED_FIXTURES_PAYLOAD
        elif "competitions" in url:
            payload = MOCK_COMPETITIONS_PAYLOAD
        else:
            payload = {}
        return MockHttpResponse(200, payload)

    session = MockHttpSession(handler=handler)
    return FootballDataOrgClient(api_key="mock_key", session=session)


def test_sync_competitions_pipeline(repo: FootballRepository, mock_client: FootballDataOrgClient):
    """Test fetching and saving competitions into local database."""
    service = FootballCollectorService(mock_client, repo)
    comps = service.sync_competitions()

    assert len(comps) == 2
    stored = repo.get_all_competitions()
    assert len(stored) == 2
    assert any(c.code == "PL" for c in stored)

    # Verify API log entry
    logs = repo.get_recent_api_logs()
    assert len(logs) == 1
    assert logs[0].endpoint == "competitions"
    assert logs[0].status_code == 200


def test_sync_fixtures_and_deduplication(repo: FootballRepository, mock_client: FootballDataOrgClient):
    """Test syncing fixtures twice ensures zero duplicate entries in database."""
    service = FootballCollectorService(mock_client, repo)

    # First sync
    res1 = service.sync_fixtures("PL", status="SCHEDULED")
    assert res1["fixtures_synced"] == 2

    fixtures_round1 = repo.get_fixtures()
    assert len(fixtures_round1) == 2

    # Second sync (same fixtures from API)
    res2 = service.sync_fixtures("PL", status="SCHEDULED")
    assert res2["fixtures_synced"] == 2

    # Verification: total fixture count MUST still be 2, NOT 4!
    fixtures_round2 = repo.get_fixtures()
    assert len(fixtures_round2) == 2

    # Verify teams were also deduplicated
    stats = repo.get_database_stats()
    # Arsenal, Man City, Liverpool, Chelsea = 4 distinct teams
    assert stats["teams"] == 4
    assert stats["fixtures"] == 2


def test_sync_historical_results_pipeline(repo: FootballRepository, mock_client: FootballDataOrgClient):
    """Test syncing historical matches populates match_results table."""
    service = FootballCollectorService(mock_client, repo)
    res = service.sync_historical_results("PL", limit=10)

    assert res["historical_results_synced"] == 1

    stats = repo.get_database_stats()
    assert stats["match_results"] == 1

    fixtures = repo.get_fixtures(status="FINISHED")
    assert len(fixtures) == 1
    assert fixtures[0]["full_time_home_score"] == 2
    assert fixtures[0]["full_time_away_score"] == 1
    assert fixtures[0]["winner"] == "HOME_TEAM"
