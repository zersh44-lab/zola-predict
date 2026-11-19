"""Football API client module."""

from __future__ import annotations

import sqlite3
from typing import Optional

from zola.client.api_sports import ApiSportsClient
from zola.client.base import (
    AuthenticationError,
    BaseFootballClient,
    FootballApiError,
    NetworkError,
    NotFoundError,
    RateLimitError,
    StandardCompetition,
    StandardFixture,
    StandardMatchResult,
    StandardTeam,
)
from zola.client.cache import RequestCache
from zola.client.football_data import FootballDataOrgClient
from zola.client.rate_limiter import QuotaExceededError, RateLimiter
from zola.config import Settings


def create_football_client(
    settings: Settings,
    conn: Optional[sqlite3.Connection] = None,
) -> BaseFootballClient:
    """Factory creating the appropriate football API client based on configuration."""
    cache = RequestCache(conn) if conn is not None else None
    rate_limiter = RateLimiter(max_rpm=settings.rate_limit_rpm)

    provider = (settings.football_api_provider or "api-football").lower().strip()
    if provider in ("api-football", "api-sports"):
        return ApiSportsClient(
            api_key=settings.football_api_key,
            base_url=settings.football_api_base_url,
            cache=cache,
            rate_limiter=rate_limiter,
        )
    elif provider in ("football-data", "football-data.org"):
        return FootballDataOrgClient(
            api_key=settings.football_api_key,
            base_url=settings.football_api_base_url,
            cache=cache,
            rate_limiter=rate_limiter,
        )
    else:
        raise ValueError(
            f"Unsupported FOOTBALL_API_PROVIDER: {provider!r}. "
            "Choose 'api-football' or 'football-data'."
        )


__all__ = [
    "create_football_client",
    "BaseFootballClient",
    "FootballDataOrgClient",
    "ApiSportsClient",
    "RequestCache",
    "RateLimiter",
    "FootballApiError",
    "AuthenticationError",
    "RateLimitError",
    "NotFoundError",
    "NetworkError",
    "QuotaExceededError",
    "StandardCompetition",
    "StandardTeam",
    "StandardFixture",
    "StandardMatchResult",
]
