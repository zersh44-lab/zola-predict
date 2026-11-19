"""Abstract base client and standardized data structures for football APIs."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from zola.client.cache import RequestCache
from zola.client.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (Strictly credential-safe: never output secret values)
# ---------------------------------------------------------------------------
class FootballApiError(Exception):
    """Base exception for all football API client operations."""
    pass


class AuthenticationError(FootballApiError):
    """Raised when API key is missing, unauthorized, or invalid."""
    pass


class RateLimitError(FootballApiError):
    """Raised when the remote API returns HTTP 429 Too Many Requests."""

    def __init__(self, message: str = "Remote API rate limit reached (HTTP 429)", retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class NotFoundError(FootballApiError):
    """Raised when requested competition, match, or resource is not found (HTTP 404)."""
    pass


class ApiServerError(FootballApiError):
    """Raised when the remote API returns a 5xx server error."""
    pass


class NetworkError(FootballApiError):
    """Raised for network connection or socket timeouts."""
    pass


# ---------------------------------------------------------------------------
# Standardized Data Transfer Classes
# ---------------------------------------------------------------------------
@dataclass
class StandardCompetition:
    external_id: str
    name: str
    code: Optional[str] = None
    country: Optional[str] = None
    emblem_url: Optional[str] = None
    current_season: Optional[str] = None


@dataclass
class StandardTeam:
    external_id: str
    name: str
    short_name: Optional[str] = None
    tla: Optional[str] = None
    crest_url: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class StandardMatchResult:
    winner: Optional[str] = None
    duration: Optional[str] = "REGULAR"
    full_time_home_score: Optional[int] = None
    full_time_away_score: Optional[int] = None
    half_time_home_score: Optional[int] = None
    half_time_away_score: Optional[int] = None
    extra_time_home_score: Optional[int] = None
    extra_time_away_score: Optional[int] = None
    penalty_home_score: Optional[int] = None
    penalty_away_score: Optional[int] = None


@dataclass
class StandardFixture:
    external_id: str
    competition_external_id: str
    utc_date: str
    status: str
    home_team: StandardTeam
    away_team: StandardTeam
    season: Optional[str] = None
    matchday: Optional[int] = None
    venue: Optional[str] = None
    referee: Optional[str] = None
    result: Optional[StandardMatchResult] = None


# ---------------------------------------------------------------------------
# Base Client
# ---------------------------------------------------------------------------
class BaseFootballClient(ABC):
    """Abstract base football API client with caching, rate limiting, and error handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        cache: Optional[RequestCache] = None,
        rate_limiter: Optional[RateLimiter] = None,
        timeout_seconds: int = 15,
        session: Optional[requests.Session] = None,
    ):
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider (e.g. 'football-data.org')."""
        pass

    @property
    @abstractmethod
    def default_base_url(self) -> str:
        """Default base URL for the API."""
        pass

    @abstractmethod
    def get_auth_headers(self) -> Dict[str, str]:
        """Headers required for authentication."""
        pass

    def is_configured(self) -> bool:
        """Check if API key is present without disclosing it."""
        return bool(self._api_key and len(self._api_key) > 0)

    def _execute_http(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        ttl_minutes: int = 60,
    ) -> Tuple[Dict[str, Any], bool, int]:
        """Execute request with caching, rate limiting, and safe error handling.

        Returns (response_data, cache_hit, elapsed_ms).
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        # 1. Check persistent cache
        if self.cache is not None:
            cached = self.cache.get(method, url, params)
            if cached is not None:
                try:
                    return cached.json(), True, 0
                except Exception as e:
                    logger.warning("Failed to decode cached response, refetching: %s", e)

        # 2. Check credentials
        if not self.is_configured():
            raise AuthenticationError(
                f"Cannot make API call to {self.provider_name}: FOOTBALL_API_KEY is missing or empty. "
                "Please configure it in your local .env file."
            )

        # 3. Apply rate limiter
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        # 4. Prepare request
        headers = self.get_auth_headers()
        start_time = time.perf_counter()

        try:
            resp = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=self.timeout_seconds,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise NetworkError(f"Network error communicating with {self.provider_name}: {e}") from e
        except requests.RequestException as e:
            raise FootballApiError(f"HTTP request to {self.provider_name} failed: {e}") from e

        # 5. Handle HTTP status codes
        if resp.status_code in (401, 403):
            raise AuthenticationError(
                f"Authentication failed for {self.provider_name} (HTTP {resp.status_code}). "
                "Verify your FOOTBALL_API_KEY in .env."
            )
        if resp.status_code == 429:
            retry_after_hdr = resp.headers.get("Retry-After")
            retry_after = int(retry_after_hdr) if retry_after_hdr and retry_after_hdr.isdigit() else None
            raise RateLimitError(
                f"Rate limit exceeded on {self.provider_name} (HTTP 429).",
                retry_after=retry_after,
            )
        if resp.status_code == 404:
            raise NotFoundError(f"Resource not found at {endpoint} on {self.provider_name} (HTTP 404)")
        if resp.status_code >= 500:
            raise ApiServerError(f"Server error from {self.provider_name} (HTTP {resp.status_code})")
        if not resp.ok:
            raise FootballApiError(f"API request failed with HTTP {resp.status_code}: {resp.text[:200]}")

        # 6. Parse valid JSON and verify provider-level errors
        try:
            data = resp.json()
        except ValueError as e:
            raise FootballApiError(f"Failed to parse JSON response from {self.provider_name}: {e}") from e

        # Check for provider-level API errors returned with HTTP 200
        errors = data.get("errors")
        if errors:
            if isinstance(errors, dict) and errors:
                if "token" in errors or "apiKey" in errors:
                    raise AuthenticationError(
                        f"Authentication failed on {self.provider_name}: {errors}. "
                        "Verify your FOOTBALL_API_KEY in .env."
                    )
                if "requests" in errors:
                    raise RateLimitError(f"Daily quota or rate limit exceeded on {self.provider_name}: {errors}")
                raise FootballApiError(f"{self.provider_name} API error: {errors}")
            elif isinstance(errors, list) and errors:
                raise FootballApiError(f"{self.provider_name} API error: {', '.join(str(e) for e in errors)}")

        if self.cache is not None:
            self.cache.set(
                method=method,
                url=url,
                params=params,
                status_code=resp.status_code,
                response_body=resp.text,
                ttl_minutes=ttl_minutes,
            )

        return data, False, elapsed_ms

    @abstractmethod
    def get_competitions(self) -> Tuple[List[StandardCompetition], bool, int]:
        """Fetch list of supported competitions."""
        pass

    @abstractmethod
    def get_fixtures(
        self,
        competition_code_or_id: Optional[str] = None,
        status: Optional[str] = None,
        season: Optional[str] = None,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch fixtures for a competition or date."""
        pass

    @abstractmethod
    def get_historical_results(
        self,
        competition_code_or_id: str,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch historical completed match results for a competition."""
        pass
