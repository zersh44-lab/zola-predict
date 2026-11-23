"""Comprehensive Regression Test Suite for Fixture Identity & Match Mapping.

Covers all 20 critical Phase 7 data integrity and mapping requirements:
1. Correct Liverpool vs Manchester United fixture on November 22 accepted.
2. Incorrect Manchester United vs Ipswich fixture on November 22 rejected.
3. Tottenham vs Ipswich on November 21 accepted with matching provider identity.
4. Correct home and away team mapping accepted.
5. Reversed home and away team mapping rejected.
6. Duplicate provider fixture ID handled idempotently.
7. Same local ID assigned to different matches detected as conflict.
8. Missing provider fixture ID rejected.
9. Wrong competition rejected.
10. Wrong season rejected.
11. Incorrect kickoff date rejected.
12. UTC and Ethiopia UTC+3 timezone boundary validation.
13. Stale cached fixture rejected for upcoming prediction.
14. Conflicting provider response is quarantined as INVALID_FIXTURE_IDENTITY.
15. Invalid fixture cannot generate a prediction (returns 'INVALID FIXTURE — PREDICTION BLOCKED').
16. Invalid fixture cannot enter prediction history.
17. Invalid fixture cannot be settled.
18. Invalid fixture cannot be sent to Telegram.
19. Invalid fixture excluded from accuracy and ROI statistics.
20. Frontend API (/api/discovery, /api/fixtures) cannot return quarantined fixtures.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict

import pytest

from zola.database.connection import get_connection, init_db
from zola.database.models import Fixture, MatchResult, Prediction
from zola.database.repository import FootballRepository
from zola.prediction.engine import PredictionEngine
from zola.telegram.reporter import TelegramReportSender
from zola.validation.datetime import format_kickoff_ethiopia, format_kickoff_utc, parse_to_utc
from zola.validation.fixture import (
    is_synthetic_external_id,
    validate_fixture_for_prediction,
    validate_fixture_identity,
)


@pytest.fixture
def test_repo(in_memory_db: sqlite3.Connection) -> FootballRepository:
    """Create in-memory SQLite database populated with standard test teams and competitions."""
    c = in_memory_db.cursor()

    # Competitions
    c.execute("""
        INSERT INTO competitions (id, external_id, provider, name, code, country)
        VALUES (8, '39', 'api-football', 'Premier League', 'PL', 'England');
    """)

    # Teams
    c.execute("""
        INSERT INTO teams (id, external_id, provider, name, short_name)
        VALUES 
            (51, '40', 'api-football', 'Liverpool', 'LIV'),
            (45, '33', 'api-football', 'Manchester United', 'MUN'),
            (47, '47', 'api-football', 'Tottenham', 'TOT'),
            (39, '57', 'api-football', 'Ipswich', 'IPS');
    """)
    in_memory_db.commit()
    return FootballRepository(in_memory_db)


# 1. Correct Liverpool vs Manchester United fixture on November 22
def test_01_correct_liverpool_vs_manchester_united_accepted(test_repo):
    fix = {
        "id": 101,
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": 8,
        "competition_name": "Premier League",
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix, repository=test_repo)
    assert is_valid is True
    assert reason is None


# 2. Incorrect Manchester United vs Ipswich fixture on November 22 rejected
def test_02_incorrect_manchester_united_vs_ipswich_rejected(test_repo):
    # Case A: Synthetic external ID
    fix_syn = {
        "id": 102,
        "external_id": "pl_20261122_2",
        "provider": "api-football",
        "home_team_id": 45,
        "away_team_id": 39,
        "home_team_name": "Manchester United",
        "away_team_name": "Ipswich",
        "competition_id": 8,
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix_syn, repository=test_repo)
    assert is_valid is False
    assert "synthetic" in reason.lower()

    # Case B: Status quarantined
    fix_quar = dict(fix_syn, external_id="999999", status="INVALID_FIXTURE_IDENTITY")
    is_valid, reason = validate_fixture_identity(fix_quar, repository=test_repo)
    assert is_valid is False
    assert "invalid fixture identity" in reason.lower()


# 3. Tottenham vs Ipswich on November 21 accepted with matching provider identity
def test_03_tottenham_vs_ipswich_nov_21_accepted(test_repo):
    fix = {
        "id": 103,
        "external_id": "1557476",
        "provider": "api-football",
        "home_team_id": 47,
        "away_team_id": 39,
        "home_team_name": "Tottenham",
        "away_team_name": "Ipswich",
        "competition_id": 8,
        "competition_name": "Premier League",
        "season": "2026",
        "utc_date": "2026-11-21T15:00:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix, repository=test_repo)
    assert is_valid is True
    assert reason is None

    # Rejected if provider is synthetic calendar
    fix_cal = dict(fix, provider="calendar")
    is_cal_valid, cal_reason = validate_fixture_identity(fix_cal, repository=test_repo)
    assert is_cal_valid is False
    assert "provider" in cal_reason.lower()


# 4. Correct home and away team mapping accepted
def test_04_correct_home_away_mapping_accepted(test_repo):
    fix = {
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": 8,
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix, repository=test_repo)
    assert is_valid is True


# 5. Reversed home and away team mapping rejected
def test_05_reversed_home_away_mapping_rejected(test_repo):
    fix = {
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 45,  # 45 is Manchester United in DB
        "away_team_id": 51,  # 51 is Liverpool in DB
        "home_team_name": "Liverpool",  # Mismatch: claiming Liverpool is team 45
        "away_team_name": "Manchester United",  # Mismatch: claiming Man Utd is team 51
        "competition_id": 8,
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix, repository=test_repo)
    assert is_valid is False
    assert "reversed" in reason.lower() or "match" in reason.lower()


# 6. Duplicate provider fixture ID handled idempotently
def test_06_duplicate_provider_fixture_id(test_repo):
    f1 = Fixture(
        external_id="1557473",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="SCHEDULED",
        home_team_id=51,
        away_team_id=45,
    )
    id1 = test_repo.upsert_fixture(f1)
    id2 = test_repo.upsert_fixture(f1)
    assert id1 == id2

    cursor = test_repo.conn.cursor()
    cursor.execute("SELECT count(*) FROM fixtures WHERE provider = 'api-football' AND external_id = '1557473';")
    assert cursor.fetchone()[0] == 1


# 7. Same local ID assigned to different matches rejected / conflict quarantined
def test_07_conflicting_team_ids_quarantined(test_repo):
    f1 = Fixture(
        external_id="1557473",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="SCHEDULED",
        home_team_id=51,
        away_team_id=45,
    )
    fid = test_repo.upsert_fixture(f1)

    # Ingestion of conflicting match under same provider fixture ID
    f_conflict = Fixture(
        external_id="1557473",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="SCHEDULED",
        home_team_id=47,  # Conflicting home team
        away_team_id=39,  # Conflicting away team
    )
    ret_id = test_repo.upsert_fixture(f_conflict)
    assert ret_id == fid

    # Record must be quarantined with INVALID_FIXTURE_IDENTITY
    db_fix = test_repo.get_fixture_by_id(fid)
    assert db_fix["status"] == "INVALID_FIXTURE_IDENTITY"


# 8. Missing provider fixture ID rejected
def test_08_missing_provider_fixture_id_rejected():
    fix = {
        "external_id": None,
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": 8,
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix)
    assert is_valid is False
    assert "missing" in reason.lower() and "external" in reason.lower()


# 9. Wrong competition rejected
def test_09_wrong_competition_rejected():
    fix = {
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": None,
        "competition_name": "unknown",
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix)
    assert is_valid is False
    assert "competition" in reason.lower()


# 10. Wrong season rejected
def test_10_wrong_season_rejected():
    # Season in 2021 with match in 2026
    fix = {
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": 8,
        "season": "2021",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix)
    assert is_valid is False
    assert "season" in reason.lower() or "inconsistent" in reason.lower()


# 11. Incorrect kickoff date rejected
def test_11_incorrect_kickoff_date_rejected():
    fix = {
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": 8,
        "season": "2026",
        "utc_date": "invalid-datetime-string",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_identity(fix)
    assert is_valid is False
    assert "kickoff" in reason.lower()


# 12. UTC and Ethiopia UTC+3 timezone boundary validation
def test_12_utc_and_ethiopia_timezone_boundary():
    # Kickoff at 2026-11-22 22:30 UTC -> 2026-11-23 01:30 EAT
    raw_utc = "2026-11-22T22:30:00+00:00"
    parsed_utc = parse_to_utc(raw_utc)
    assert parsed_utc.day == 22
    assert parsed_utc.hour == 22

    eat_str = format_kickoff_ethiopia(raw_utc)
    # Ethiopia is UTC+3, so 22:30 + 3h = 01:30 on Nov 23
    assert "01:30" in eat_str
    assert "2026-11-23" in eat_str

    # UTC display remains on Nov 22
    utc_str = format_kickoff_utc(raw_utc)
    assert "2026-11-22" in utc_str


# 13. Stale cached fixture rejected for upcoming predictions
def test_13_stale_cached_fixture_rejected():
    # Kickoff is in the past relative to reference time
    ref_now = datetime(2026, 11, 23, 12, 0, 0, tzinfo=timezone.utc)
    stale_fix = {
        "external_id": "1557473",
        "provider": "api-football",
        "home_team_id": 51,
        "away_team_id": 45,
        "home_team_name": "Liverpool",
        "away_team_name": "Manchester United",
        "competition_id": 8,
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "SCHEDULED",
    }
    is_valid, reason = validate_fixture_for_prediction(stale_fix, now_utc=ref_now, enforce_future_kickoff=True)
    assert is_valid is False
    assert "INVALID FIXTURE — PREDICTION BLOCKED" in reason
    assert "past" in reason.lower()


# 14. Conflicting provider response is quarantined
def test_14_conflicting_provider_response_quarantined(test_repo):
    f1 = Fixture(
        external_id="conflict_fixture_1",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="SCHEDULED",
        home_team_id=51,
        away_team_id=45,
    )
    fid = test_repo.upsert_fixture(f1)
    assert test_repo.get_fixture_by_id(fid)["status"] == "SCHEDULED"

    # Conflicting provider response arrives
    f2 = Fixture(
        external_id="conflict_fixture_1",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="SCHEDULED",
        home_team_id=47,
        away_team_id=39,
    )
    test_repo.upsert_fixture(f2)
    assert test_repo.get_fixture_by_id(fid)["status"] == "INVALID_FIXTURE_IDENTITY"


# 15. Invalid fixture cannot generate a prediction
def test_15_invalid_fixture_cannot_generate_prediction(test_repo):
    f_inv = Fixture(
        external_id="pl_20261122_inv",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="INVALID_FIXTURE_IDENTITY",
        home_team_id=51,
        away_team_id=45,
    )
    fid = test_repo.upsert_fixture(f_inv)

    engine = PredictionEngine(test_repo)
    with pytest.raises(ValueError) as excinfo:
        engine.predict_fixture(fid)
    assert "INVALID FIXTURE — PREDICTION BLOCKED" in str(excinfo.value)


# 16. Invalid fixture cannot enter history
def test_16_invalid_fixture_cannot_enter_history(test_repo):
    # Insert invalid fixture
    f_inv = Fixture(
        external_id="inv_hist_fixture",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="INVALID_FIXTURE_IDENTITY",
        home_team_id=51,
        away_team_id=45,
    )
    fid = test_repo.upsert_fixture(f_inv)
    test_repo.upsert_match_result(
        MatchResult(fixture_id=fid, full_time_home_score=2, full_time_away_score=0, winner="HOME_TEAM")
    )
    test_repo.upsert_prediction(
        Prediction(
            fixture_id=fid,
            model_name="ensemble",
            market="OU05",
            predicted_outcome="OVER_05",
            probability=0.90,
            probabilities_json="{}",
            confidence="HIGH",
            status="INVALID",
            settlement_status="VOID",
        )
    )

    history = test_repo.get_prediction_history(date="2026-11-22")
    fixture_ids = [m["fixture_id"] for m in history.get("matches", [])]
    assert fid not in fixture_ids


# 17. Invalid fixture cannot be settled
def test_17_invalid_fixture_cannot_be_settled(test_repo):
    f_inv = Fixture(
        external_id="settle_test_fixture",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="INVALID_FIXTURE_IDENTITY",
        home_team_id=51,
        away_team_id=45,
    )
    fid = test_repo.upsert_fixture(f_inv)

    res = test_repo.settle_fixture(fid, 2, 0)
    assert res.get("status") == "BLOCKED"
    assert "INVALID FIXTURE — CANNOT BE SETTLED" in res.get("error", "")

    eval_res = test_repo.evaluate_predictions(fixture_id=fid)
    assert eval_res.get("status") == "BLOCKED"
    assert "INVALID FIXTURE — CANNOT BE SETTLED" in eval_res.get("error", "")


# 18. Invalid fixture cannot be sent to Telegram
def test_18_invalid_fixture_cannot_be_sent_to_telegram():
    sender = TelegramReportSender(token="dummy_token")
    inv_fix = {
        "external_id": "pl_20261122_2",
        "provider": "api-football",
        "home_team_id": 45,
        "away_team_id": 39,
        "home_team_name": "Manchester United",
        "away_team_name": "Ipswich",
        "competition_id": 8,
        "season": "2026",
        "utc_date": "2026-11-22T16:30:00+00:00",
        "status": "INVALID_FIXTURE_IDENTITY",
    }
    report = {"is_skipped": False, "picks": [{"market": "OU05", "probability": 0.90, "market_odd": 1.15, "expected_value": 0.035}]}
    is_valid, rej = sender.validate_report_payload(inv_fix, report)
    assert is_valid is False
    assert "INVALID FIXTURE — PREDICTION BLOCKED" in rej


# 19. Invalid fixture excluded from accuracy statistics
def test_19_invalid_fixture_excluded_from_accuracy_statistics(test_repo):
    # 1. Add valid completed fixture and settled prediction
    f_val = Fixture(
        external_id="val_acc_fix",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-20T16:30:00+00:00",
        status="FINISHED",
        home_team_id=51,
        away_team_id=45,
    )
    fid_val = test_repo.upsert_fixture(f_val)
    test_repo.upsert_match_result(
        MatchResult(fixture_id=fid_val, full_time_home_score=1, full_time_away_score=0, winner="HOME_TEAM")
    )
    test_repo.upsert_prediction(
        Prediction(
            fixture_id=fid_val,
            model_name="ensemble",
            market="OU05",
            predicted_outcome="OVER_05",
            probability=0.90,
            probabilities_json="{}",
            confidence="HIGH",
            status="WIN",
            settlement_status="WIN",
            is_correct=1,
        )
    )

    # 2. Add quarantined fixture with invalid predictions
    f_inv = Fixture(
        external_id="inv_acc_fix",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-20T16:30:00+00:00",
        status="INVALID_FIXTURE_IDENTITY",
        home_team_id=47,
        away_team_id=39,
    )
    fid_inv = test_repo.upsert_fixture(f_inv)
    test_repo.upsert_prediction(
        Prediction(
            fixture_id=fid_inv,
            model_name="ensemble",
            market="OU05",
            predicted_outcome="OVER_05",
            probability=0.85,
            probabilities_json="{}",
            confidence="HIGH",
            status="INVALID",
            settlement_status="VOID",
            is_correct=0,
        )
    )

    metrics = test_repo.get_accuracy_metrics(model_name="ensemble", market="OU05", timeframe="30d")
    assert len(metrics) == 1
    # Only 1 prediction evaluated, total must be 1, not 2
    assert metrics[0].total_predictions == 1
    assert metrics[0].correct_predictions == 1
    assert metrics[0].accuracy_rate == 1.0


# 20. Frontend API cannot return quarantined fixtures as valid
def test_20_frontend_api_cannot_return_quarantined_fixtures(test_repo):
    # Insert genuine Liverpool vs Manchester United
    f_real = Fixture(
        external_id="1557473",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="FINISHED",
        home_team_id=51,
        away_team_id=45,
    )
    test_repo.upsert_fixture(f_real)

    # Insert quarantined synthetic Manchester United vs Ipswich
    f_fake = Fixture(
        external_id="pl_20261122_2",
        provider="api-football",
        competition_id=8,
        season="2026",
        matchday=12,
        utc_date="2026-11-22T16:30:00+00:00",
        status="INVALID_FIXTURE_IDENTITY",
        home_team_id=45,
        away_team_id=39,
    )
    test_repo.upsert_fixture(f_fake)

    # Check discovery API query
    disc = test_repo.get_discovery_fixtures(date="2026-11-22", category="all")
    fixtures = disc.get("all_fixtures", [])
    assert len(fixtures) == 1
    assert fixtures[0]["external_id"] == "1557473"
    assert fixtures[0]["home_team_name"] == "Liverpool"
    assert fixtures[0]["away_team_name"] == "Manchester United"

    # Check standard fixtures query
    standard_fixtures = test_repo.get_fixtures(date="2026-11-22")
    assert len(standard_fixtures) == 1
    assert standard_fixtures[0]["external_id"] == "1557473"
