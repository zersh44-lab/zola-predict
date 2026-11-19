"""Tests for persistent request caching and API rate limiting / quota protection."""

import sqlite3
import time
from datetime import datetime, timedelta, timezone
import pytest

from zola.client.cache import RequestCache
from zola.client.rate_limiter import QuotaExceededError, RateLimiter


def test_cache_set_and_get(in_memory_db: sqlite3.Connection):
    """Test caching response and successful cache retrieval."""
    cache = RequestCache(in_memory_db)
    url = "https://api.football-data.org/v4/competitions"
    params = {"status": "SCHEDULED"}

    # Initially cache is empty
    assert cache.get("GET", url, params) is None

    # Save to cache with 10 min TTL
    test_json = '{"count": 1, "competitions": [{"id": 2021}]}'
    cache.set("GET", url, status_code=200, response_body=test_json, ttl_minutes=10, params=params)

    cached = cache.get("GET", url, params)
    assert cached is not None
    assert cached.status_code == 200
    assert cached.cache_hit is True
    assert cached.json() == {"count": 1, "competitions": [{"id": 2021}]}


def test_cache_expiration(in_memory_db: sqlite3.Connection):
    """Test that expired cache entries are not returned."""
    cache = RequestCache(in_memory_db)
    url = "https://api.football-data.org/v4/matches"

    # Set with negative TTL (already expired)
    cache.set("GET", url, status_code=200, response_body='{"test": 1}', ttl_minutes=-5)

    assert cache.get("GET", url) is None

    # Test clear_expired cleans the record
    cleared = cache.clear_expired()
    assert cleared == 1


def test_rate_limiter_rpm_enforcement():
    """Test that rate limiter detects and enforces maximum requests per minute."""
    limiter = RateLimiter(max_rpm=3, enable_sleep=False)

    # 3 calls should succeed immediately
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    # 4th call within the same minute should raise QuotaExceededError
    with pytest.raises(QuotaExceededError) as exc_info:
        limiter.acquire()

    assert "Rate limit exceeded" in str(exc_info.value)


def test_rate_limiter_daily_quota():
    """Test daily quota guardian protection."""
    limiter = RateLimiter(max_rpm=100, daily_limit=5, enable_sleep=False)

    for _ in range(5):
        limiter.acquire()

    # 6th call should be blocked by daily quota
    with pytest.raises(QuotaExceededError) as exc_info:
        limiter.acquire()

    assert "Daily API request quota reached" in str(exc_info.value)
    status = limiter.get_status()
    assert status["daily_requests_used"] == 5
    assert status["daily_limit"] == 5
