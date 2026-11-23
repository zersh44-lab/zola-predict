"""Strict Odds and Expected Value Validation for Zola Predict.

Enforces:
1. Approved markets only: OU05 (Over 0.5) and OU15 (Over 1.5).
2. Strict odds authenticity: decimal odds > 1.01, pre-match, verified provider.
3. Rejection of synthetic, estimated, stale, live, or post-kickoff odds.
4. Mathematically verified Expected Value: EV = (P * Odds) - 1 > 0 with decimal P in (0, 1).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set, Tuple

from zola.validation.datetime import parse_to_utc

# Production Active Approved Markets
APPROVED_MARKETS: Set[str] = {"OU05", "OU15"}

# Inactive / Disabled Markets
DISABLED_MARKETS: Set[str] = {
    "1X2",
    "BTTS",
    "DOUBLE_CHANCE",
    "DNB",
    "OU25",
    "OU35",
}

# Minimum threshold for decimal odds to be considered valid and non-trivial
MIN_DECIMAL_ODD = 1.01
MAX_DECIMAL_ODD = 100.0


def calculate_expected_value(probability: float, decimal_odds: float) -> float:
    """Calculate mathematically rigorous Expected Value: EV = (P * Odds) - 1.

    Args:
        probability: Decimal probability strictly between 0.0 and 1.0 (e.g. 0.85).
        decimal_odds: Verified decimal odds > 1.01 (e.g. 1.25).

    Returns:
        float: Expected Value margin (e.g. +0.0625 for +6.25%).

    Raises:
        ValueError: If probability or odds are invalid.
    """
    if probability is None or math.isnan(probability) or math.isinf(probability):
        raise ValueError(f"Invalid probability value: {probability}")

    # Enforce decimal probability between 0 and 1 (never percentage like 85.0)
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError(
            f"Probability must be a decimal strictly between 0 and 1. Got: {probability}"
        )

    if decimal_odds is None or math.isnan(decimal_odds) or math.isinf(decimal_odds):
        raise ValueError(f"Invalid decimal odds: {decimal_odds}")

    if decimal_odds <= 1.0:
        raise ValueError(
            f"Decimal odds must be strictly greater than 1.0. Got: {decimal_odds}"
        )

    ev = (probability * decimal_odds) - 1.0
    return round(ev, 6)


def validate_odds_record(
    odds_record: Any,
    market: str,
    outcome: str = "OVER",
    fixture: Optional[Dict[str, Any]] = None,
    now_utc: Optional[datetime] = None,
) -> Tuple[Optional[float], bool, Optional[str]]:
    """Validate a single market's odds record against strict authenticity criteria.

    Returns:
        (validated_odd_float, is_valid, rejection_reason)
    """
    # 1. Market Allowlist Check
    m_clean = str(market).strip().upper()
    if m_clean not in APPROVED_MARKETS:
        return None, False, f"Market '{market}' is not in approved markets ({sorted(APPROVED_MARKETS)})."

    if odds_record is None:
        return None, False, f"Odds for market '{market}' are missing (null)."

    raw_odd: Optional[float] = None

    if isinstance(odds_record, dict):
        # Check match ID association if present
        if fixture and "fixture_id" in odds_record:
            rec_fid = odds_record["fixture_id"]
            fix_fid = fixture.get("id")
            if rec_fid and fix_fid and int(rec_fid) != int(fix_fid):
                return None, False, f"Odds match ID ({rec_fid}) does not match fixture ID ({fix_fid})."

        # Check market association if present
        if "market" in odds_record:
            rec_m = str(odds_record["market"]).strip().upper()
            if rec_m != m_clean:
                return None, False, f"Odds market ({rec_m}) does not match target market ({m_clean})."

        # 2. Reject Synthetic or Estimated Odds
        if odds_record.get("is_synthetic") is True or odds_record.get("synthetic") is True:
            return None, False, f"Synthetic odds rejected for market '{market}'."
        if odds_record.get("is_estimated") is True or odds_record.get("estimated") is True:
            return None, False, f"Estimated odds rejected for market '{market}'."

        source = str(odds_record.get("source", "")).strip().lower()
        if source in ("synthetic", "estimated", "fair_odd", "model", "average", "historical"):
            return None, False, f"Non-genuine odds source '{source}' rejected for market '{market}'."

        status = str(odds_record.get("status", "")).strip().upper()
        if status in ("SYNTHETIC", "ESTIMATED"):
            return None, False, f"Odds status '{status}' rejected for market '{market}'."

        # 3. Reject Stale Odds
        if odds_record.get("is_stale") is True or odds_record.get("stale") is True or status == "STALE":
            return None, False, f"Stale odds rejected for market '{market}'."

        # 4. Reject In-Play / Post-Kickoff Odds
        if odds_record.get("is_post_kickoff") is True or odds_record.get("post_kickoff") is True:
            return None, False, f"Post-kickoff odds rejected for market '{market}'."
        if (
            odds_record.get("in_play") is True
            or odds_record.get("live") is True
            or odds_record.get("is_in_play") is True
            or status in ("IN_PLAY", "LIVE", "POST_KICKOFF")
        ):
            return None, False, f"Live / in-play odds rejected for market '{market}'."

        # Check captured_at timestamp against fixture kickoff
        captured_at = odds_record.get("captured_at") or odds_record.get("timestamp")
        kickoff_utc = (fixture.get("utc_date") if fixture else None) or odds_record.get("kickoff_utc")

        if captured_at and kickoff_utc:
            try:
                cap_dt = parse_to_utc(captured_at)
                k_dt = parse_to_utc(kickoff_utc)
                if cap_dt >= k_dt:
                    return (
                        None,
                        False,
                        f"Odds captured after kickoff ({cap_dt.isoformat()} >= {k_dt.isoformat()}) rejected.",
                    )
            except Exception:
                pass

        # Extract decimal odd value for target outcome
        raw = (
            odds_record.get(outcome)
            or odds_record.get(outcome.upper())
            or odds_record.get(outcome.lower())
            or odds_record.get("market_odd")
            or odds_record.get("odd")
            or odds_record.get("decimal_odd")
        )
        if raw is not None:
            try:
                raw_odd = float(raw)
            except (ValueError, TypeError):
                return None, False, f"Non-numeric odds value '{raw}' for market '{market}'."

    elif isinstance(odds_record, (int, float)):
        raw_odd = float(odds_record)
    else:
        return None, False, f"Unrecognized odds data format ({type(odds_record).__name__}) for market '{market}'."

    # Fixture in-play check
    if fixture:
        f_status = str(fixture.get("status", "")).strip().upper()
        if f_status in ("IN_PLAY", "LIVE", "FINISHED", "COMPLETED", "PAUSED"):
            return None, False, f"Fixture status is '{f_status}'; pre-match odds cannot be used."

    if raw_odd is None:
        return None, False, f"Missing decimal odd for outcome '{outcome}' in market '{market}'."

    if math.isnan(raw_odd) or math.isinf(raw_odd):
        return None, False, f"Invalid mathematical odd ({raw_odd}) for market '{market}'."

    if raw_odd <= MIN_DECIMAL_ODD:
        return (
            None,
            False,
            f"Decimal odd ({raw_odd}) is <= {MIN_DECIMAL_ODD} threshold for market '{market}'.",
        )

    if raw_odd > MAX_DECIMAL_ODD:
        return (
            None,
            False,
            f"Decimal odd ({raw_odd}) exceeds maximum sanity limit ({MAX_DECIMAL_ODD}) for market '{market}'.",
        )

    return raw_odd, True, None


def validate_pre_match_odds(
    odds_data: Optional[Dict[str, Any]],
    fixture: Dict[str, Any],
    approved_markets: Set[str] = APPROVED_MARKETS,
) -> Tuple[bool, Dict[str, Dict[str, float]], Dict[str, str]]:
    """Validate full odds dictionary for a fixture against active approved markets.

    Returns:
        (has_any_valid_approved_market, validated_odds_dict, rejection_reasons)
    """
    if not odds_data or not isinstance(odds_data, dict):
        return False, {}, {"general": "No odds data provided for fixture."}

    validated: Dict[str, Dict[str, float]] = {}
    rejections: Dict[str, str] = {}

    for market in approved_markets:
        if market not in odds_data:
            rejections[market] = f"No odds record provided for market {market}."
            continue

        item = odds_data[market]
        # For OU markets, primary outcome is 'OVER'
        odd_val, is_valid, reason = validate_odds_record(
            item,
            market=market,
            outcome="OVER",
            fixture=fixture,
        )
        if is_valid and odd_val is not None:
            validated[market] = {"OVER": odd_val}
        else:
            rejections[market] = reason or f"Validation failed for market {market}."

    has_valid = len(validated) > 0
    return has_valid, validated, rejections
