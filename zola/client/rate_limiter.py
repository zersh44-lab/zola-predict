"""Rate limiting and quota protection for free-tier football APIs."""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Optional


class QuotaExceededError(Exception):
    """Raised when the configured API quota has been exhausted."""
    pass


class RateLimiter:
    """Sliding-window rate limiter and daily quota guardian."""

    def __init__(
        self,
        max_rpm: int = 10,
        daily_limit: Optional[int] = 100,
        enable_sleep: bool = True,
    ):
        self.max_rpm = max(1, max_rpm)
        self.daily_limit = daily_limit
        self.enable_sleep = enable_sleep
        self.timestamps: Deque[float] = deque()
        self.daily_count = 0
        self.current_day: Optional[str] = None
        self._lock = threading.Lock()

    def _check_and_reset_day(self) -> None:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.current_day != today_str:
            self.current_day = today_str
            self.daily_count = 0

    def acquire(self) -> float:
        """Acquire permission to make an API call.

        Waits if rate limits are exceeded (when enable_sleep=True),
        or raises QuotaExceededError if daily limits are reached.
        Returns the duration waited in seconds.
        """
        with self._lock:
            self._check_and_reset_day()

            # Check daily quota
            if self.daily_limit is not None and self.daily_count >= self.daily_limit:
                raise QuotaExceededError(
                    f"Daily API request quota reached ({self.daily_count}/{self.daily_limit}). "
                    "Halting requests to protect free tier quota."
                )

            now = time.time()
            # Clean timestamps older than 60 seconds
            while self.timestamps and (now - self.timestamps[0]) >= 60.0:
                self.timestamps.popleft()

            waited = 0.0
            if len(self.timestamps) >= self.max_rpm:
                # Rate limit exceeded in the current minute window
                oldest = self.timestamps[0]
                wait_needed = max(0.01, 60.0 - (now - oldest) + 0.1)
                if not self.enable_sleep:
                    raise QuotaExceededError(
                        f"Rate limit exceeded: {len(self.timestamps)}/{self.max_rpm} requests in last 60s. "
                        f"Wait {wait_needed:.1f}s."
                    )
                time.sleep(wait_needed)
                waited = wait_needed
                now = time.time()
                while self.timestamps and (now - self.timestamps[0]) >= 60.0:
                    self.timestamps.popleft()

            self.timestamps.append(now)
            self.daily_count += 1
            return waited

    def get_status(self) -> dict:
        """Return current rate limiter metrics."""
        with self._lock:
            self._check_and_reset_day()
            now = time.time()
            while self.timestamps and (now - self.timestamps[0]) >= 60.0:
                self.timestamps.popleft()
            return {
                "current_minute_requests": len(self.timestamps),
                "max_rpm": self.max_rpm,
                "daily_requests_used": self.daily_count,
                "daily_limit": self.daily_limit,
            }
