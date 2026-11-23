"""Strict Fixture Validation for Zola Predict.

Guarantees every match entering the prediction engine has verified provider credentials,
valid teams, real competitions, future kickoff timestamps, and valid pre-match status.
Rejects synthetic, calendar-seeded, postponed, completed, or cancelled fixtures.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

from zola.validation.datetime import is_kickoff_in_future, parse_to_utc

# Valid pre-match statuses allowed into prediction pipeline
VALID_PRE_MATCH_STATUSES: Set[str] = {
    "SCHEDULED",
    "TIMED",
    "NOT_STARTED",
    "NS",
    "TBD",
}

# Explicitly ineligible statuses that must be rejected immediately
INELIGIBLE_MATCH_STATUSES: Set[str] = {
    "FINISHED",
    "FT",
    "AET",
    "PEN",
    "COMPLETED",
    "IN_PLAY",
    "LIVE",
    "1H",
    "HT",
    "2H",
    "ET",
    "P",
    "PAUSED",
    "BT",
    "INT",
    "POSTPONED",
    "PST",
    "CANCELLED",
    "CANC",
    "SUSPENDED",
    "SUSP",
    "ABANDONED",
    "ABD",
    "AWARDED",
    "AWD",
    "INTERRUPTED",
    "WO",
    "INVALID_SYNTHETIC",
    "VOID",
}

# Disallowed synthetic ID regex patterns
SYNTHETIC_ID_PATTERNS = [
    re.compile(r"^(pl|pd|sa|bl1|cl|fl1)_\d{8}_\d+$", re.IGNORECASE),
    re.compile(r"^(synthetic|fake)[\-_]?", re.IGNORECASE),
    re.compile(r".*calendar.*", re.IGNORECASE),
]

DISALLOWED_TEAM_NAMES: Set[str] = {
    "unknown",
    "tbd",
    "dummy",
    "test",
    "team a",
    "team b",
    "placeholder",
    "n/a",
    "none",
}


def is_synthetic_external_id(external_id: Any) -> bool:
    """Check if external_id matches known synthetic or calendar seeding patterns."""
    if not external_id:
        return True
    ext_str = str(external_id).strip()
    for pat in SYNTHETIC_ID_PATTERNS:
        if pat.match(ext_str):
            return True
    return False


def validate_fixture_for_prediction(
    fixture: Dict[str, Any],
    now_utc: Optional[datetime] = None,
    enforce_future_kickoff: bool = True,
) -> Tuple[bool, Optional[str]]:
    """Validate a fixture record against strict production data integrity rules.

    Args:
        fixture: Dictionary representation of fixture.
        now_utc: Reference UTC datetime for kickoff freshness.
        enforce_future_kickoff: If True, rejects fixtures whose kickoff is in the past.

    Returns:
        (is_valid, rejection_reason)
    """
    if not fixture or not isinstance(fixture, dict):
        return False, "Fixture data is missing, null, or not a valid dictionary."

    # 1. Provider and Match ID Validation
    fid = fixture.get("id")
    ext_id = fixture.get("external_id")
    provider = fixture.get("provider")

    if not ext_id and not fid:
        return False, "Fixture rejected: Missing both internal ID and provider external ID."

    if ext_id and is_synthetic_external_id(ext_id):
        return False, f"Fixture rejected: Detected synthetic or calendar-seeded external ID '{ext_id}'."

    if provider:
        p_clean = str(provider).strip().lower()
        if p_clean in ("synthetic", "calendar", "fake"):
            return False, f"Fixture rejected: Invalid or synthetic provider '{provider}'."

    # 2. Team Validation
    home_name = str(fixture.get("home_team_name") or fixture.get("home") or "").strip()
    away_name = str(fixture.get("away_team_name") or fixture.get("away") or "").strip()
    home_id = fixture.get("home_team_id")
    away_id = fixture.get("away_team_id")

    if not home_name and not home_id:
        return False, "Fixture rejected: Home team information is completely missing."
    if not away_name and not away_id:
        return False, "Fixture rejected: Away team information is completely missing."

    if home_name.lower() in DISALLOWED_TEAM_NAMES:
        return False, f"Fixture rejected: Unverified or placeholder home team '{home_name}'."
    if away_name.lower() in DISALLOWED_TEAM_NAMES:
        return False, f"Fixture rejected: Unverified or placeholder away team '{away_name}'."

    if home_name and away_name and home_name.lower() == away_name.lower():
        return False, f"Fixture rejected: Identical home and away team '{home_name}'."
    if home_id and away_id and home_id == away_id:
        return False, f"Fixture rejected: Identical home and away team IDs ({home_id})."

    # Check for placeholder team ID patterns (e.g. 'team_1', 'dummy_team')
    if isinstance(home_id, str) and (home_id.startswith("team_") or "dummy" in home_id):
        return False, f"Fixture rejected: Unverified placeholder home team ID '{home_id}'."
    if isinstance(away_id, str) and (away_id.startswith("team_") or "dummy" in away_id):
        return False, f"Fixture rejected: Unverified placeholder away team ID '{away_id}'."

    # 3. Competition Validation
    comp_name = str(fixture.get("competition_name") or fixture.get("comp_name") or "").strip()
    comp_code = str(fixture.get("competition_code") or fixture.get("comp_code") or "").strip()
    comp_id = fixture.get("competition_id")

    if not comp_name and not comp_code and not comp_id:
        return False, "Fixture rejected: Competition information is completely missing."
    if comp_name.lower() in ("unknown", "dummy", "test", "fake"):
        return False, f"Fixture rejected: Invalid competition '{comp_name}'."

    # 4. Status Validation
    status = str(fixture.get("status") or "").strip().upper()
    if not status:
        return False, "Fixture rejected: Match status is missing."

    if status in INELIGIBLE_MATCH_STATUSES:
        return False, f"Fixture rejected: Ineligible match status '{status}' (must be scheduled pre-match)."

    if status not in VALID_PRE_MATCH_STATUSES:
        return False, f"Fixture rejected: Unrecognized or non-pre-match status '{status}'."

    # 5. Kickoff Timestamp Validation
    raw_date = fixture.get("utc_date") or fixture.get("date")
    if not raw_date:
        return False, "Fixture rejected: Missing kickoff timestamp."

    try:
        kickoff_utc = parse_to_utc(raw_date)
    except ValueError as e:
        return False, f"Fixture rejected: Invalid kickoff timestamp '{raw_date}': {e}"

    if enforce_future_kickoff:
        if not is_kickoff_in_future(kickoff_utc, now_utc=now_utc, grace_minutes=0):
            return (
                False,
                f"Fixture rejected: Kickoff timestamp ({kickoff_utc.isoformat()}) is in the past or already underway.",
            )

    return True, None
