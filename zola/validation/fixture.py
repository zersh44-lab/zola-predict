"""Strict Fixture Validation for Zola Predict.

Guarantees every match entering the prediction engine or database has verified provider credentials,
valid teams, real competitions, consistent kickoff timestamps, and valid pre-match status.
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
    "INVALID_FIXTURE_IDENTITY",
    "INVALID",
    "VOID",
}

# Disallowed synthetic ID regex patterns
SYNTHETIC_ID_PATTERNS = [
    re.compile(r"^(pl|pd|sa|bl1|cl|fl1|e0|sp1|d1|i1|f1)[\-_]\d{8}[\-_]?\d*$", re.IGNORECASE),
    re.compile(r"^(synthetic|fake|calendar|mock|test)[\-_]?.*$", re.IGNORECASE),
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


def validate_fixture_identity(
    fixture: Dict[str, Any],
    repository: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate core fixture identity independent of kickoff freshness.

    Ensures the fixture possesses a valid external provider ID, authentic team pairings,
    valid competition and season associations, and consistent kickoff dates.
    """
    if not fixture or not isinstance(fixture, dict):
        return False, "Fixture data is missing, null, or not a valid dictionary."

    # 1. Status quarantine check
    status = str(fixture.get("status") or "").strip().upper()
    if status.startswith("INVALID") or status == "VOID":
        return False, f"Status indicates invalid fixture identity: '{status}'."

    # 2. Provider and Match ID Validation
    fid = fixture.get("id")
    ext_id = fixture.get("external_id")
    provider = fixture.get("provider")

    if not ext_id:
        return False, "Fixture rejected: Missing provider external ID."

    if is_synthetic_external_id(ext_id):
        return False, f"Fixture rejected: Detected synthetic or calendar-seeded external ID '{ext_id}'."

    if not provider:
        return False, "Fixture rejected: Missing data provider."

    p_clean = str(provider).strip().lower()
    if p_clean in ("synthetic", "calendar", "fake"):
        return False, f"Fixture rejected: Invalid or synthetic provider '{provider}'."

    # 3. Team Validation
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

    # Check for placeholder team ID patterns
    if isinstance(home_id, str) and (home_id.startswith("team_") or "dummy" in home_id):
        return False, f"Fixture rejected: Unverified placeholder home team ID '{home_id}'."
    if isinstance(away_id, str) and (away_id.startswith("team_") or "dummy" in away_id):
        return False, f"Fixture rejected: Unverified placeholder away team ID '{away_id}'."

    # 4. Competition & Season Validation
    comp_name = str(fixture.get("competition_name") or fixture.get("comp_name") or "").strip()
    comp_code = str(fixture.get("competition_code") or fixture.get("comp_code") or "").strip()
    comp_id = fixture.get("competition_id")

    if not comp_name and not comp_code and not comp_id:
        return False, "Fixture rejected: Competition information is completely missing."
    if comp_name.lower() in ("unknown", "dummy", "test", "fake"):
        return False, f"Fixture rejected: Invalid competition '{comp_name}'."

    season = str(fixture.get("season") or "").strip()
    raw_date = fixture.get("utc_date") or fixture.get("date")
    if not season and raw_date:
        try:
            season = str(parse_to_utc(raw_date).year)
        except Exception:
            pass

    if not season or season.lower() in ("unknown", "test", "dummy"):
        return False, f"Fixture rejected: Invalid or missing season '{season}'."

    # 5. Kickoff Timestamp & Season Consistency Validation
    raw_date = fixture.get("utc_date") or fixture.get("date")
    if not raw_date:
        return False, "Fixture rejected: Missing kickoff timestamp."

    try:
        kickoff_utc = parse_to_utc(raw_date)
    except ValueError as e:
        return False, f"Fixture rejected: Invalid kickoff timestamp '{raw_date}': {e}"

    # Year consistency check with season
    season_digits = re.findall(r"\b(20\d\d)\b", season)
    if season_digits:
        season_year = int(season_digits[0])
        if abs(kickoff_utc.year - season_year) > 1:
            return (
                False,
                f"Fixture rejected: Inconsistent kickoff year ({kickoff_utc.year}) for season '{season}'.",
            )

    # 6. Database Identity Verification (if repository provided)
    if repository and home_id and away_id:
        try:
            cursor = repository.conn.cursor()
            cursor.execute(
                "SELECT id, name FROM teams WHERE id IN (?, ?);",
                (home_id, away_id),
            )
            t_rows = {r["id"]: r["name"] for r in cursor.fetchall()}
            db_h_name = t_rows.get(home_id)
            db_a_name = t_rows.get(away_id)

            if db_h_name and home_name and db_h_name.lower() != home_name.lower():
                # Check for reversed mapping
                if db_a_name and home_name.lower() == db_a_name.lower():
                    return False, "Fixture rejected: Reversed home and away team mapping."
                # Check if team name substantially mismatches stored team
                if db_h_name.lower() not in home_name.lower() and home_name.lower() not in db_h_name.lower():
                    return False, f"Fixture rejected: Home team ID {home_id} ({db_h_name}) does not match '{home_name}'."

            if db_a_name and away_name and db_a_name.lower() != away_name.lower():
                # Check for reversed mapping
                if db_h_name and away_name.lower() == db_h_name.lower():
                    return False, "Fixture rejected: Reversed home and away team mapping."
                if db_a_name.lower() not in away_name.lower() and away_name.lower() not in db_a_name.lower():
                    return False, f"Fixture rejected: Away team ID {away_id} ({db_a_name}) does not match '{away_name}'."
        except Exception:
            pass

    return True, None


def validate_fixture_for_prediction(
    fixture: Dict[str, Any],
    now_utc: Optional[datetime] = None,
    enforce_future_kickoff: bool = True,
    repository: Optional[Any] = None,
) -> Tuple[bool, Optional[str]]:
    """Validate a fixture record against strict production data integrity rules.

    Args:
        fixture: Dictionary representation of fixture.
        now_utc: Reference UTC datetime for kickoff freshness.
        enforce_future_kickoff: If True, rejects fixtures whose kickoff is in the past.
        repository: Optional repository instance for database identity verification.

    Returns:
        (is_valid, rejection_reason)
    """
    # 1. Core Identity Validation
    is_identity_valid, id_reason = validate_fixture_identity(fixture, repository=repository)
    if not is_identity_valid:
        return False, f"INVALID FIXTURE — PREDICTION BLOCKED: {id_reason}"

    # 2. Status Validation for Predictions
    status = str(fixture.get("status") or "").strip().upper()
    if not status:
        return False, "INVALID FIXTURE — PREDICTION BLOCKED: Match status is missing."

    if status in INELIGIBLE_MATCH_STATUSES:
        return (
            False,
            f"INVALID FIXTURE — PREDICTION BLOCKED: Ineligible match status '{status}' (must be scheduled pre-match).",
        )

    if status not in VALID_PRE_MATCH_STATUSES:
        return (
            False,
            f"INVALID FIXTURE — PREDICTION BLOCKED: Unrecognized or non-pre-match status '{status}'.",
        )

    # 3. Kickoff Timestamp Freshness
    raw_date = fixture.get("utc_date") or fixture.get("date")
    try:
        kickoff_utc = parse_to_utc(raw_date)
    except ValueError as e:
        return False, f"INVALID FIXTURE — PREDICTION BLOCKED: Invalid kickoff timestamp '{raw_date}': {e}"

    if enforce_future_kickoff:
        if not is_kickoff_in_future(kickoff_utc, now_utc=now_utc, grace_minutes=0):
            return (
                False,
                f"INVALID FIXTURE — PREDICTION BLOCKED: Kickoff timestamp ({kickoff_utc.isoformat()}) is in the past or already underway.",
            )

    return True, None
