"""Strict Date and Time Validation for Zola Predict.

Ensures all match timestamps are strictly normalized to timezone-aware UTC,
correctly handles Ethiopia time (UTC+3 / EAT), detects midnight boundary shifts,
and prevents comparing raw un-normalized date strings.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional, Union

# Fixed UTC+3 timezone for Ethiopia / East Africa Time (EAT)
ETHIOPIA_TZ = timezone(timedelta(hours=3), name="EAT")


def parse_to_utc(val: Union[str, datetime, date, None]) -> datetime:
    """Parse any datetime input into a timezone-aware UTC datetime.

    Raises:
        ValueError: If value is None, empty, or cannot be parsed into a valid timestamp.
    """
    if val is None:
        raise ValueError("Cannot parse None to UTC datetime.")

    if isinstance(val, datetime):
        if val.tzinfo is None:
            # Naive datetime: assume UTC
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)

    if isinstance(val, date) and not isinstance(val, datetime):
        return datetime.combine(val, time.min, tzinfo=timezone.utc)

    val_str = str(val).strip()
    if not val_str:
        raise ValueError("Cannot parse empty string to UTC datetime.")

    # Replace 'Z' with +00:00 for ISO standard parsing
    cleaned = val_str.replace("Z", "+00:00")

    # Attempt ISO format parsing
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    # Common football API fallback formats
    formats = [
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"Unable to parse timestamp '{val}' into UTC datetime.")


def format_kickoff_utc(val: Union[str, datetime, date]) -> str:
    """Format timestamp as standard ISO-8601 UTC string (e.g. 2026-11-21T15:00:00+00:00)."""
    dt_utc = parse_to_utc(val)
    return dt_utc.isoformat()


def format_kickoff_ethiopia(val: Union[str, datetime, date]) -> str:
    """Format timestamp in Ethiopia Time (UTC+3 / EAT) (e.g. 2026-11-21 18:00 EAT)."""
    dt_utc = parse_to_utc(val)
    dt_eat = dt_utc.astimezone(ETHIOPIA_TZ)
    return dt_eat.strftime("%Y-%m-%d %H:%M EAT")


def is_kickoff_in_future(
    val: Union[str, datetime, date],
    now_utc: Optional[datetime] = None,
    grace_minutes: int = 0,
) -> bool:
    """Check if kickoff timestamp is strictly in the future relative to UTC now.

    Args:
        val: Kickoff timestamp.
        now_utc: Reference UTC datetime (defaults to datetime.now(timezone.utc)).
        grace_minutes: Tolerated pre-match margin in minutes (e.g. 0 for strict pre-match).
    """
    kickoff_utc = parse_to_utc(val)
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    cutoff = now_utc - timedelta(minutes=grace_minutes)
    return kickoff_utc >= cutoff


def validate_match_date_for_target(
    kickoff_val: Union[str, datetime, date],
    target_date_str: str,
    tz: timezone = timezone.utc,
) -> bool:
    """Validate whether a kickoff occurs on target_date_str in specified timezone.

    Prevents timezone boundary errors where e.g. a match scheduled for 2026-11-22 00:30 UTC
    is misclassified as 2026-11-21 due to local offset.
    """
    kickoff_utc = parse_to_utc(kickoff_val)
    target_clean = str(target_date_str).strip()
    try:
        target_d = datetime.strptime(target_clean, "%Y-%m-%d").date()
    except ValueError as e:
        raise ValueError(f"Invalid target date format: '{target_date_str}'. Expected YYYY-MM-DD.") from e

    localized_dt = kickoff_utc.astimezone(tz)
    return localized_dt.date() == target_d
