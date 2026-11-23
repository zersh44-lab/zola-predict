"""Automated Regression Test Suite for Zola Predict Production Data Integrity.

Covers all 20 required regression test cases:
1. Valid fixture with valid odds
2. Nonexistent fixture
3. Incorrect team mapping
4. Wrong match ID
5. Wrong competition
6. Incorrect kickoff date
7. Timezone boundary error
8. Completed match
9. Postponed match
10. Missing odds
11. Stale odds
12. Odds received after kickoff
13. Invalid probability
14. Negative EV
15. Unapproved market
16. Insufficient consensus
17. API timeout
18. Invalid API response
19. Duplicate fixture
20. Telegram report containing mismatched data
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict
import pytest

from zola.client.base import FootballApiError
from zola.database.models import Competition, Fixture, Team
from zola.database.repository import FootballRepository
from zola.prediction.calibration import compute_expected_value
from zola.prediction.engine import PredictionEngine
from zola.prediction.high_confidence import (
    APPROVED_HIGH_CONFIDENCE_MARKETS,
    HighConfidenceSelector,
)
from zola.telegram.reporter import TelegramReportSender
from zola.validation.datetime import (
    ETHIOPIA_TZ,
    format_kickoff_ethiopia,
    format_kickoff_utc,
    is_kickoff_in_future,
    parse_to_utc,
    validate_match_date_for_target,
)
from zola.validation.fixture import (
    INELIGIBLE_MATCH_STATUSES,
    VALID_PRE_MATCH_STATUSES,
    is_synthetic_external_id,
    validate_fixture_for_prediction,
)
from zola.validation.odds import (
    APPROVED_MARKETS,
    DISABLED_MARKETS,
    calculate_expected_value,
    validate_odds_record,
    validate_pre_match_odds,
)


@pytest.fixture
def memory_repo(in_memory_db):
    """Provides a fresh in-memory repository initialized with schema."""
    return FootballRepository(in_memory_db)


@pytest.fixture
def future_kickoff():
    """Returns an ISO timestamp in UTC 24 hours into the future."""
    dt = datetime.now(timezone.utc) + timedelta(days=1)
    return dt.isoformat()


@pytest.fixture
def valid_fixture_dict(future_kickoff):
    """Returns a fully valid dictionary representation of a pre-match fixture."""
    return {
        "id": 101,
        "external_id": "auth_match_101",
        "provider": "api-football",
        "competition_id": 1,
        "competition_name": "Premier League",
        "competition_code": "PL",
        "home_team_id": 10,
        "home_team_name": "Arsenal",
        "away_team_id": 20,
        "away_team_name": "Chelsea",
        "utc_date": future_kickoff,
        "status": "SCHEDULED",
        "season": "2026",
    }


# ==============================================================================
# TEST 1: Valid fixture with valid odds
# ==============================================================================
def test_01_valid_fixture_with_valid_odds(valid_fixture_dict, future_kickoff):
    """Case 1: Fully verified fixture with authentic pre-match odds."""
    # 1. Validate fixture
    is_valid, reason = validate_fixture_for_prediction(valid_fixture_dict)
    assert is_valid is True, f"Valid fixture failed validation: {reason}"

    # 2. Validate odds
    odds_data = {
        "OU15": {
            "fixture_id": 101,
            "market": "OU15",
            "OVER": 1.35,
            "source": "bet365",
            "captured_at": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            "kickoff_utc": future_kickoff,
        }
    }
    has_valid, validated_odds, rejections = validate_pre_match_odds(odds_data, valid_fixture_dict)
    assert has_valid is True
    assert "OU15" in validated_odds
    assert validated_odds["OU15"]["OVER"] == 1.35


# ==============================================================================
# TEST 2: Nonexistent fixture
# ==============================================================================
def test_02_nonexistent_fixture(memory_repo):
    """Case 2: Attempting to predict a nonexistent fixture ID raises ValueError."""
    engine = PredictionEngine(memory_repo)
    with pytest.raises(ValueError, match="not found in database"):
        engine.predict_fixture(fixture_id=999999)


# ==============================================================================
# TEST 3: Incorrect team mapping
# ==============================================================================
def test_03_incorrect_team_mapping(valid_fixture_dict):
    """Case 3: Fixture with identical teams or placeholder/dummy team names is rejected."""
    # Identical teams
    same_team = dict(valid_fixture_dict, home_team_name="Arsenal", away_team_name="Arsenal")
    is_valid, reason = validate_fixture_for_prediction(same_team)
    assert is_valid is False
    assert "Identical home and away" in reason

    # Placeholder / unknown team
    dummy_team = dict(valid_fixture_dict, home_team_name="Unknown", away_team_name="Chelsea")
    is_valid, reason = validate_fixture_for_prediction(dummy_team)
    assert is_valid is False
    assert "Unverified or placeholder" in reason

    # Synthetic placeholder team ID
    placeholder_id = dict(valid_fixture_dict, home_team_id="team_dummy_1")
    is_valid, reason = validate_fixture_for_prediction(placeholder_id)
    assert is_valid is False
    assert "placeholder" in reason.lower()


# ==============================================================================
# TEST 4: Wrong match ID
# ==============================================================================
def test_04_wrong_match_id(valid_fixture_dict):
    """Case 4: Odds containing a mismatched fixture_id are rejected."""
    odds_record = {
        "fixture_id": 999,  # Mismatched ID (fixture is 101)
        "market": "OU15",
        "OVER": 1.30,
        "source": "bet365",
    }
    odd_val, is_valid, reason = validate_odds_record(
        odds_record, market="OU15", outcome="OVER", fixture=valid_fixture_dict
    )
    assert is_valid is False
    assert "does not match fixture ID" in reason


# ==============================================================================
# TEST 5: Wrong competition
# ==============================================================================
def test_05_wrong_competition(valid_fixture_dict):
    """Case 5: Fixture with missing or dummy competition is rejected."""
    no_comp = dict(valid_fixture_dict, competition_name=None, competition_code=None, competition_id=None)
    is_valid, reason = validate_fixture_for_prediction(no_comp)
    assert is_valid is False
    assert "Competition information is completely missing" in reason

    fake_comp = dict(valid_fixture_dict, competition_name="Dummy")
    is_valid, reason = validate_fixture_for_prediction(fake_comp)
    assert is_valid is False
    assert "Invalid competition" in reason


# ==============================================================================
# TEST 6: Incorrect kickoff date
# ==============================================================================
def test_06_incorrect_kickoff_date(valid_fixture_dict):
    """Case 6: Fixture with an unparseable or invalid date timestamp is rejected."""
    bad_date = dict(valid_fixture_dict, utc_date="not-a-valid-date-timestamp")
    is_valid, reason = validate_fixture_for_prediction(bad_date)
    assert is_valid is False
    assert "Invalid kickoff timestamp" in reason


# ==============================================================================
# TEST 7: Timezone boundary error
# ==============================================================================
def test_07_timezone_boundary_error():
    """Case 7: Kickoff at 2026-11-22 00:30 UTC is NOT misclassified as 2026-11-21 in UTC."""
    kickoff = "2026-11-22T00:30:00+00:00"

    # In UTC, it belongs strictly to 2026-11-22
    assert validate_match_date_for_target(kickoff, "2026-11-22", tz=timezone.utc) is True
    assert validate_match_date_for_target(kickoff, "2026-11-21", tz=timezone.utc) is False

    # In Ethiopia (UTC+3), it is 03:30 AM on Nov 22
    eat_str = format_kickoff_ethiopia(kickoff)
    assert "2026-11-22" in eat_str
    assert "03:30 EAT" in eat_str


# ==============================================================================
# TEST 8: Completed match
# ==============================================================================
def test_08_completed_match(valid_fixture_dict):
    """Case 8: Match with status FINISHED or COMPLETED is rejected from pre-match pipeline."""
    finished = dict(valid_fixture_dict, status="FINISHED")
    is_valid, reason = validate_fixture_for_prediction(finished)
    assert is_valid is False
    assert "Ineligible match status 'FINISHED'" in reason

    completed = dict(valid_fixture_dict, status="COMPLETED")
    is_valid, reason = validate_fixture_for_prediction(completed)
    assert is_valid is False
    assert "Ineligible match status 'COMPLETED'" in reason


# ==============================================================================
# TEST 9: Postponed match
# ==============================================================================
def test_09_postponed_match(valid_fixture_dict):
    """Case 9: Match with status POSTPONED, CANCELLED, or ABANDONED is rejected."""
    for st in ("POSTPONED", "CANCELLED", "ABANDONED", "SUSPENDED"):
        fixture = dict(valid_fixture_dict, status=st)
        is_valid, reason = validate_fixture_for_prediction(fixture)
        assert is_valid is False
        assert st in reason


# ==============================================================================
# TEST 10: Missing odds
# ==============================================================================
def test_10_missing_odds(valid_fixture_dict):
    """Case 10: Missing or null odds records are rejected."""
    odd_val, is_valid, reason = validate_odds_record(
        None, market="OU15", outcome="OVER", fixture=valid_fixture_dict
    )
    assert is_valid is False
    assert "missing" in reason.lower()

    # Empty dictionary passed to pre-match odds validator
    has_valid, validated, rejections = validate_pre_match_odds({}, valid_fixture_dict)
    assert has_valid is False


# ==============================================================================
# TEST 11: Stale odds
# ==============================================================================
def test_11_stale_odds(valid_fixture_dict):
    """Case 11: Odds flagged as stale are rejected."""
    stale_record = {
        "market": "OU15",
        "OVER": 1.25,
        "is_stale": True,
        "status": "STALE",
    }
    odd_val, is_valid, reason = validate_odds_record(
        stale_record, market="OU15", outcome="OVER", fixture=valid_fixture_dict
    )
    assert is_valid is False
    assert "stale" in reason.lower()


# ==============================================================================
# TEST 12: Odds received after kickoff
# ==============================================================================
def test_12_odds_received_after_kickoff(valid_fixture_dict):
    """Case 12: Odds captured after kickoff timestamp are rejected."""
    kickoff = "2026-11-21T15:00:00+00:00"
    fixture = dict(valid_fixture_dict, utc_date=kickoff)

    post_kickoff_odds = {
        "market": "OU15",
        "OVER": 1.25,
        "captured_at": "2026-11-21T15:30:00+00:00",  # 30 min after kickoff
        "kickoff_utc": kickoff,
    }
    odd_val, is_valid, reason = validate_odds_record(
        post_kickoff_odds, market="OU15", outcome="OVER", fixture=fixture
    )
    assert is_valid is False
    assert "after kickoff" in reason.lower()

    # Also test is_post_kickoff flag
    flagged_odds = {
        "market": "OU15",
        "OVER": 1.25,
        "is_post_kickoff": True,
    }
    odd_val, is_valid, reason = validate_odds_record(
        flagged_odds, market="OU15", outcome="OVER", fixture=fixture
    )
    assert is_valid is False
    assert "post-kickoff" in reason.lower()


# ==============================================================================
# TEST 13: Invalid probability
# ==============================================================================
def test_13_invalid_probability():
    """Case 13: Probabilities outside (0, 1), percentages (e.g. 75), and NaN are rejected."""
    # Percentage instead of decimal (e.g. 75.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calculate_expected_value(75.0, 1.25)

    # Negative probability
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calculate_expected_value(-0.1, 1.25)

    # Probability >= 1.0
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        calculate_expected_value(1.0, 1.25)

    # NaN probability
    with pytest.raises(ValueError, match="Invalid probability"):
        calculate_expected_value(float("nan"), 1.25)


# ==============================================================================
# TEST 14: Negative EV
# ==============================================================================
def test_14_negative_ev():
    """Case 14: Selections with negative expected value are calculated correctly and rejected."""
    # Probability 0.70 at odds 1.20: EV = (0.70 * 1.20) - 1 = -0.16 (-16%)
    ev = calculate_expected_value(0.70, 1.20)
    assert ev == -0.16
    assert ev < 0.0

    # compute_expected_value from calibration also yields negative
    cal_ev = compute_expected_value(0.70, 1.20)
    assert cal_ev < 0.0


# ==============================================================================
# TEST 15: Unapproved market
# ==============================================================================
def test_15_unapproved_market(valid_fixture_dict):
    """Case 15: Blocked markets (1X2, BTTS, DOUBLE_CHANCE, etc.) are rejected."""
    for dis in DISABLED_MARKETS:
        odd_val, is_valid, reason = validate_odds_record(
            {"market": dis, "HOME": 2.10}, market=dis, fixture=valid_fixture_dict
        )
        assert is_valid is False
        assert "not in approved markets" in reason.lower()


# ==============================================================================
# TEST 16: Insufficient consensus
# ==============================================================================
def test_16_insufficient_consensus(valid_fixture_dict):
    """Case 16: Single-model or insufficient consensus outputs do not produce unverified picks."""
    # When no candidate models provide high-confidence consensus, HighConfidenceSelector issues skip
    report = HighConfidenceSelector.evaluate_fixture(
        fixture=valid_fixture_dict,
        ensemble_predictions={},  # Empty predictions
        require_valid_odds=True,
    )
    assert report.is_skipped is True
    assert "NO HIGH-CONFIDENCE PREDICTION" in report.skip_reason


# ==============================================================================
# TEST 17: API timeout
# ==============================================================================
def test_17_api_timeout():
    """Case 17: API timeouts raise NetworkError and do not fabricate fake fixtures."""
    from zola.client.api_sports import ApiSportsClient
    from zola.client.base import NetworkError
    import requests

    client = ApiSportsClient(api_key="mock_key")

    def mock_req(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timed out to API-Football")

    client.session.request = mock_req
    with pytest.raises(NetworkError, match="timed out"):
        client.get_fixtures(date="2026-11-21")


# ==============================================================================
# TEST 18: Invalid API response
# ==============================================================================
def test_18_invalid_api_response():
    """Case 18: API plan limitations or error responses raise FootballApiError instead of returning empty."""
    from zola.client.api_sports import ApiSportsClient
    from unittest.mock import MagicMock

    client = ApiSportsClient(api_key="mock_key")

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.json.return_value = {
        "message": "Free plans do not have access to this date, try from 2026-11-18 to 2026-11-20."
    }
    mock_resp.text = str(mock_resp.json.return_value)

    client.session.request = MagicMock(return_value=mock_resp)

    with pytest.raises(FootballApiError, match="HTTP 500"):
        client.get_fixtures(date="2026-11-21")


# ==============================================================================
# TEST 19: Duplicate fixture
# ==============================================================================
def test_19_duplicate_fixture(memory_repo, future_kickoff):
    """Case 19: Inserting a fixture with identical external_id is idempotent and prevents duplicate rows."""
    comp_id = memory_repo.upsert_competition(
        Competition(external_id="pl_comp_1", provider="api-football", name="Premier League", code="PL")
    )
    h_id = memory_repo.upsert_team(
        Team(external_id="arsenal_1", provider="api-football", name="Arsenal")
    )
    a_id = memory_repo.upsert_team(
        Team(external_id="chelsea_1", provider="api-football", name="Chelsea")
    )

    f1 = Fixture(
        external_id="match_uniq_12345",
        provider="api-football",
        competition_id=comp_id,
        home_team_id=h_id,
        away_team_id=a_id,
        utc_date=future_kickoff,
        status="SCHEDULED",
    )
    fid1 = memory_repo.upsert_fixture(f1)
    fid2 = memory_repo.upsert_fixture(f1)

    assert fid1 == fid2, "Duplicate external_id must return the same local fixture ID"

    cursor = memory_repo.conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fixtures WHERE external_id = 'match_uniq_12345'")
    assert cursor.fetchone()[0] == 1, "There must be exactly one row for match_uniq_12345"


# ==============================================================================
# TEST 20: Telegram report containing mismatched data
# ==============================================================================
def test_20_telegram_report_containing_mismatched_data(valid_fixture_dict):
    """Case 20: Telegram publication is blocked when report contains invalid/unverified fields."""
    sender = TelegramReportSender(token="mock_token_123")

    # 1. Report with unapproved market (e.g. 1X2)
    bad_hc_report = {
        "is_skipped": False,
        "picks": [
            {
                "market": "1X2",
                "predicted_outcome": "HOME_TEAM",
                "probability": 0.75,
                "market_odd": 1.45,
                "expected_value": 0.0875,
            }
        ],
    }
    is_valid, reason = sender.validate_report_payload(valid_fixture_dict, bad_hc_report)
    assert is_valid is False
    assert "Unapproved market '1X2'" in reason

    # 2. Report with negative EV
    neg_ev_report = {
        "is_skipped": False,
        "picks": [
            {
                "market": "OU15",
                "predicted_outcome": "OVER_15",
                "probability": 0.75,
                "market_odd": 1.20,
                "expected_value": -0.10,
            }
        ],
    }
    is_valid, reason = sender.validate_report_payload(valid_fixture_dict, neg_ev_report)
    assert is_valid is False
    assert "Non-positive expected value" in reason

    # 3. Report with invalid fixture (e.g. synthetic ID)
    synthetic_fixture = dict(valid_fixture_dict, external_id="pl_20261121_1")
    is_valid, reason = sender.validate_report_payload(synthetic_fixture, bad_hc_report)
    assert is_valid is False
    assert "synthetic" in reason.lower()
