"""Tests for football API clients using mocked responses (no live API calls)."""

from __future__ import annotations

import json
import pytest

from zola.client.base import (
    AuthenticationError,
    FootballApiError,
    NotFoundError,
    RateLimitError,
)
from zola.client.cache import RequestCache
from zola.client.football_data import FootballDataOrgClient
from tests.conftest import (
    MOCK_COMPETITIONS_PAYLOAD,
    MOCK_HISTORICAL_RESULTS_PAYLOAD,
    MOCK_SCHEDULED_FIXTURES_PAYLOAD,
    MockHttpResponse,
    MockHttpSession,
)


def test_client_is_configured():
    """Verify is_configured returns True only when non-empty key is provided."""
    client_empty = FootballDataOrgClient(api_key=None)
    assert not client_empty.is_configured()

    client_blank = FootballDataOrgClient(api_key="   ")
    assert not client_blank.is_configured()

    client_valid = FootballDataOrgClient(api_key="mock_key_abc")
    assert client_valid.is_configured()


def test_missing_api_key_raises_auth_error():
    """Verify that attempting to fetch data without an API key raises AuthenticationError."""
    client = FootballDataOrgClient(api_key=None)
    with pytest.raises(AuthenticationError) as exc:
        client.get_competitions()
    assert "FOOTBALL_API_KEY is missing or empty" in str(exc.value)


def test_get_competitions_mocked():
    """Test parsing competitions from mocked response."""
    session = MockHttpSession(handler=lambda m, u, **kw: MockHttpResponse(200, MOCK_COMPETITIONS_PAYLOAD))
    client = FootballDataOrgClient(api_key="dummy_mock_key", session=session)

    comps, from_cache, elapsed_ms = client.get_competitions()

    assert not from_cache
    assert len(comps) == 2
    assert comps[0].external_id == "2021"
    assert comps[0].name == "Premier League"
    assert comps[0].code == "PL"
    assert comps[0].country == "England"
    assert comps[1].name == "Primera Division"
    assert len(session.calls) == 1


def test_get_fixtures_mocked():
    """Test parsing upcoming fixtures from mocked response."""
    session = MockHttpSession(handler=lambda m, u, **kw: MockHttpResponse(200, MOCK_SCHEDULED_FIXTURES_PAYLOAD))
    client = FootballDataOrgClient(api_key="dummy_mock_key", session=session)

    fixtures, from_cache, _ = client.get_fixtures("PL", status="SCHEDULED")

    assert len(fixtures) == 2
    assert fixtures[0].external_id == "5001"
    assert fixtures[0].home_team.name == "Arsenal FC"
    assert fixtures[0].away_team.name == "Manchester City FC"
    assert fixtures[0].status == "TIMED"
    assert fixtures[0].result is None  # Not played yet


def test_get_historical_results_mocked():
    """Test parsing finished match results with scores."""
    session = MockHttpSession(handler=lambda m, u, **kw: MockHttpResponse(200, MOCK_HISTORICAL_RESULTS_PAYLOAD))
    client = FootballDataOrgClient(api_key="dummy_mock_key", session=session)

    results, from_cache, _ = client.get_historical_results("PL", limit=10)

    assert len(results) == 1
    assert results[0].status == "FINISHED"
    assert results[0].result is not None
    assert results[0].result.winner == "HOME_TEAM"
    assert results[0].result.full_time_home_score == 2
    assert results[0].result.full_time_away_score == 1


def test_client_handles_401_without_leaking_key():
    """Verify HTTP 401 raises AuthenticationError and never leaks the key."""
    fake_key = "secret_auth_token_999"
    session = MockHttpSession(handler=lambda m, u, **kw: MockHttpResponse(401, {"message": "Unauthorized"}))
    client = FootballDataOrgClient(api_key=fake_key, session=session)

    with pytest.raises(AuthenticationError) as exc:
        client.get_competitions()

    assert fake_key not in str(exc.value)
    assert "Authentication failed" in str(exc.value)


def test_client_handles_429_rate_limit():
    """Verify HTTP 429 raises RateLimitError with retry_after header."""
    session = MockHttpSession(
        handler=lambda m, u, **kw: MockHttpResponse(429, {"message": "Too Many Requests"}, headers={"Retry-After": "60"})
    )
    client = FootballDataOrgClient(api_key="mock_key", session=session)

    with pytest.raises(RateLimitError) as exc:
        client.get_competitions()

    assert exc.value.retry_after == 60


def test_client_caching_skips_network_call(in_memory_db):
    """Verify that cached data returns immediately with zero HTTP requests made."""
    cache = RequestCache(in_memory_db)
    session = MockHttpSession()

    # Pre-populate cache
    url = "https://api.football-data.org/v4/competitions"
    cache.set("GET", url, status_code=200, response_body=json.dumps(MOCK_COMPETITIONS_PAYLOAD), ttl_minutes=60)

    client = FootballDataOrgClient(api_key="mock_key", cache=cache, session=session)
    comps, from_cache, elapsed_ms = client.get_competitions()

    assert from_cache is True
    assert len(comps) == 2
    # Ensure network request was NOT called
    assert len(session.calls) == 0


def test_api_sports_status_translation():
    """Verify that generic SCHEDULED status is translated to API-Football NS-TBD."""
    from zola.client.api_sports import ApiSportsClient

    calls = []

    def mock_handler(method, url, headers=None, params=None, timeout=None):
        calls.append(params)
        return MockHttpResponse(200, {"response": [], "errors": []})

    session = MockHttpSession(handler=mock_handler)
    client = ApiSportsClient(api_key="test_key", session=session)
    client.get_fixtures(competition_code_or_id="39", status="SCHEDULED", season="2026")

    assert len(calls) == 1
    assert calls[0]["status"] == "NS-TBD"
    assert calls[0]["league"] == "39"


def test_api_sports_handles_plan_error_without_caching(in_memory_db):
    """Verify that API-Football plan error raises FootballApiError and is not cached."""
    from zola.client.api_sports import ApiSportsClient

    cache = RequestCache(in_memory_db)
    session = MockHttpSession(
        handler=lambda m, u, **kw: MockHttpResponse(
            200,
            {"errors": {"plan": "Free plans do not have access to this season, try from 2022 to 2024."}, "response": []},
        )
    )
    client = ApiSportsClient(api_key="test_key", cache=cache, session=session)

    # Calling with a season that triggers plan error without fallback
    with pytest.raises(FootballApiError) as exc:
        client._execute_http("GET", "fixtures", params={"league": "39", "season": "2026"})

    assert "plan" in str(exc.value).lower()
    # Ensure it was not cached
    cached = cache.get("GET", "https://v3.football.api-sports.io/fixtures", {"league": "39", "season": "2026"})
    assert cached is None
