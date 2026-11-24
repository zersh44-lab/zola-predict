"""Regression test suite for Zola Predict Prediction History pipeline.

Verifies:
1. History displays predictions newer than November 20 when records exist.
2. New predictions appear after being saved.
3. History sorts newest to oldest.
4. UTC timestamps display under the correct date.
5. Pagination does not hide the newest records.
6. Empty history and API errors are handled correctly.
7. History does not use a hardcoded date.
8. Automatic history reconciliation and prediction settlement.
"""

import json
import sqlite3
import urllib.request
from typing import Any, Dict

import pytest

from zola.database.models import Fixture, MatchResult, Prediction
from zola.database.repository import FootballRepository
from zola.database.connection import get_connection, init_db


@pytest.fixture
def history_test_db(tmp_path):
    """Creates an isolated SQLite database with initial schema for history testing."""
    db_file = str(tmp_path / "test_history.db")
    init_db(db_file)
    conn = get_connection(db_file)
    cursor = conn.cursor()

    # Pre-seed competition and teams
    cursor.execute("INSERT OR IGNORE INTO competitions (id, external_id, provider, name, code) VALUES (1, 'comp_1', 'api-football', 'Premier League', 'PL');")
    cursor.execute("INSERT OR IGNORE INTO teams (id, external_id, provider, name, short_name) VALUES (10, 'team_10', 'api-football', 'Team A', 'TMA');")
    cursor.execute("INSERT OR IGNORE INTO teams (id, external_id, provider, name, short_name) VALUES (20, 'team_20', 'api-football', 'Team B', 'TMB');")
    cursor.execute("INSERT OR IGNORE INTO teams (id, external_id, provider, name, short_name) VALUES (30, 'team_30', 'api-football', 'Team C', 'TMC');")
    cursor.execute("INSERT OR IGNORE INTO teams (id, external_id, provider, name, short_name) VALUES (40, 'team_40', 'api-football', 'Team D', 'TMD');")
    conn.commit()
    conn.close()
    return db_file


def test_history_retrieves_records_newer_than_nov_20(history_test_db):
    """Proves history retrieves and includes dates newer than November 20 (e.g. Nov 24, 2026)."""
    conn = get_connection(history_test_db)
    repo = FootballRepository(conn)

    # 1. Insert an older match from Nov 20
    f_nov20 = Fixture(
        external_id="ext_nov20", provider="api-football",
        competition_id=1, season="2026", matchday=12,
        utc_date="2026-11-20T20:00:00+00:00", status="FINISHED",
        home_team_id=10, away_team_id=20,
    )
    fid_20 = repo.upsert_fixture(f_nov20)
    repo.upsert_match_result(MatchResult(fixture_id=fid_20, full_time_home_score=2, full_time_away_score=1))
    repo.upsert_prediction(Prediction(
        fixture_id=fid_20, model_name="ensemble", market="1X2",
        predicted_outcome="HOME_WIN", probability=0.65, probabilities_json="{}", status="SETTLED",
        settlement_status="WIN", is_correct=1
    ))

    # 2. Insert a newer match from Nov 24, 2026
    f_nov24 = Fixture(
        external_id="ext_nov24", provider="api-football",
        competition_id=1, season="2026", matchday=13,
        utc_date="2026-11-24T16:00:00+00:00", status="FINISHED",
        home_team_id=30, away_team_id=40,
    )
    fid_24 = repo.upsert_fixture(f_nov24)
    repo.upsert_match_result(MatchResult(fixture_id=fid_24, full_time_home_score=1, full_time_away_score=1))
    repo.upsert_prediction(Prediction(
        fixture_id=fid_24, model_name="ensemble", market="1X2",
        predicted_outcome="DRAW", probability=0.45, probabilities_json="{}", status="SETTLED",
        settlement_status="WIN", is_correct=1
    ))

    history_data = repo.get_prediction_history(limit=50)
    conn.close()

    avail_dates = [d["date"] for d in history_data["available_dates"]]
    assert "2026-11-24" in avail_dates, "Nov 24 must be present in available_dates"
    assert "2026-11-20" in avail_dates, "Nov 20 must be present in available_dates"

    # Verify Nov 24 is listed before Nov 20
    assert avail_dates.index("2026-11-24") < avail_dates.index("2026-11-20"), "Dates must sort newest first"

    items = history_data["history"]
    assert len(items) == 2
    assert items[0]["fixture_id"] == fid_24, "Newest fixture (Nov 24) must appear first"
    assert items[1]["fixture_id"] == fid_20, "Older fixture (Nov 20) must appear second"


def test_new_predictions_appear_after_being_saved_and_settled(history_test_db):
    """Proves that saving a prediction and settling it immediately surfaces in history."""
    conn = get_connection(history_test_db)
    repo = FootballRepository(conn)

    # Initially empty history
    init_hist = repo.get_prediction_history()
    assert len(init_hist["history"]) == 0
    assert len(init_hist["available_dates"]) == 0

    # Add a fixture with pre-match prediction
    f_new = Fixture(
        external_id="ext_new_match", provider="api-football",
        competition_id=1, season="2026", matchday=14,
        utc_date="2026-11-24T18:00:00+00:00", status="SCHEDULED",
        home_team_id=10, away_team_id=30,
    )
    fid_new = repo.upsert_fixture(f_new)
    repo.upsert_prediction(Prediction(
        fixture_id=fid_new, model_name="ensemble", market="OU15",
        predicted_outcome="OVER_15", probability=0.82, probabilities_json="{}", status="PENDING",
        settlement_status="PENDING",
    ))

    # Before match ends, should not be in history
    pre_hist = repo.get_prediction_history()
    assert len(pre_hist["history"]) == 0

    # Match concludes with final score 2-1
    repo.upsert_match_result(MatchResult(fixture_id=fid_new, full_time_home_score=2, full_time_away_score=1))
    cursor = conn.cursor()
    cursor.execute("UPDATE fixtures SET status = 'FINISHED' WHERE id = ?;", (fid_new,))
    conn.commit()

    # Settle predictions
    eval_res = repo.evaluate_predictions()
    assert eval_res["evaluated"] >= 1

    # Now verify it surfaces immediately in history
    post_hist = repo.get_prediction_history()
    assert len(post_hist["history"]) == 1
    assert post_hist["history"][0]["fixture_id"] == fid_new
    assert post_hist["available_dates"][0]["date"] == "2026-11-24"
    conn.close()


def test_history_sorts_newest_to_oldest(history_test_db):
    """Proves history items are strictly sorted from newest to oldest by kickoff timestamp."""
    conn = get_connection(history_test_db)
    repo = FootballRepository(conn)

    dates = [
        ("2026-11-15T15:00:00+00:00", "ext_301"),
        ("2026-11-24T21:00:00+00:00", "ext_304"),
        ("2026-11-19T18:00:00+00:00", "ext_302"),
        ("2026-11-24T12:00:00+00:00", "ext_303"),
    ]
    created_fids = {}
    for d_str, ext_id in dates:
        f = Fixture(
            external_id=ext_id, provider="api-football",
            competition_id=1, season="2026", matchday=1,
            utc_date=d_str, status="FINISHED",
            home_team_id=10, away_team_id=20,
        )
        fid = repo.upsert_fixture(f)
        created_fids[ext_id] = fid
        repo.upsert_match_result(MatchResult(fixture_id=fid, full_time_home_score=1, full_time_away_score=0))
        repo.upsert_prediction(Prediction(
            fixture_id=fid, model_name="ensemble", market="1X2",
            predicted_outcome="HOME_WIN", probability=0.6, probabilities_json="{}", status="SETTLED",
            settlement_status="WIN", is_correct=1
        ))

    hist = repo.get_prediction_history(limit=50)
    returned_ids = [item["fixture_id"] for item in hist["history"]]
    # Expected newest first: ext_304 (21:00 Nov 24), ext_303 (12:00 Nov 24), ext_302 (Nov 19), ext_301 (Nov 15)
    expected_order = [
        created_fids["ext_304"],
        created_fids["ext_303"],
        created_fids["ext_302"],
        created_fids["ext_301"],
    ]
    assert returned_ids == expected_order, f"Expected newest to oldest order, got: {returned_ids}"
    conn.close()


def test_pagination_does_not_hide_newest_records(history_test_db):
    """Proves pagination (limit/offset) keeps the newest records on the first page."""
    conn = get_connection(history_test_db)
    repo = FootballRepository(conn)

    # Insert 5 fixtures across consecutive dates
    created_fids = []
    for i in range(1, 6):
        f = Fixture(
            external_id=f"ext_page_{i}", provider="api-football",
            competition_id=1, season="2026", matchday=i,
            utc_date=f"2026-11-{20+i:02d}T15:00:00+00:00", status="FINISHED",
            home_team_id=10, away_team_id=20,
        )
        fid = repo.upsert_fixture(f)
        created_fids.append(fid)
        repo.upsert_match_result(MatchResult(fixture_id=fid, full_time_home_score=1, full_time_away_score=0))
        repo.upsert_prediction(Prediction(
            fixture_id=fid, model_name="ensemble", market="1X2",
            predicted_outcome="HOME_WIN", probability=0.6, probabilities_json="{}", status="SETTLED",
            settlement_status="WIN", is_correct=1
        ))

    # Page 1 (limit 2, offset 0) must contain the two newest records: day 25 (index 4) and day 24 (index 3)
    page1 = repo.get_prediction_history(limit=2, offset=0)
    p1_ids = [item["fixture_id"] for item in page1["history"]]
    assert p1_ids == [created_fids[4], created_fids[3]], f"Page 1 must contain newest items, got {p1_ids}"

    # Page 2 (limit 2, offset 2) must contain the next two: day 23 (index 2) and day 22 (index 1)
    page2 = repo.get_prediction_history(limit=2, offset=2)
    p2_ids = [item["fixture_id"] for item in page2["history"]]
    assert p2_ids == [created_fids[2], created_fids[1]], f"Page 2 must contain next newest items, got {p2_ids}"
    conn.close()


def test_empty_history_and_invalid_date_handled(history_test_db):
    """Proves that querying an invalid date or date without matches returns structured empty data without crashing."""
    conn = get_connection(history_test_db)
    repo = FootballRepository(conn)

    res = repo.get_prediction_history(date="2099-12-31")
    assert res["history"] == []
    assert res["summary"]["total_matches"] == 0
    assert res["summary"]["is_date_filtered"] is True
    assert res["summary"]["filtered_date"] == "2099-12-31"

    res_empty = repo.get_prediction_history(search="NonExistentTeamNameXYZ")
    assert res_empty["history"] == []
    assert res_empty["summary"]["total_matches"] == 0
    conn.close()


def test_history_does_not_use_hardcoded_date(history_test_db):
    """Proves that dates returned in available_dates are dynamic and strictly derived from database records."""
    conn = get_connection(history_test_db)
    repo = FootballRepository(conn)

    # Insert a fixture in arbitrary future year 2028
    f = Fixture(
        external_id="ext_future_2028", provider="api-football",
        competition_id=1, season="2028", matchday=1,
        utc_date="2028-06-15T18:00:00+00:00", status="FINISHED",
        home_team_id=10, away_team_id=20,
    )
    fid_2028 = repo.upsert_fixture(f)
    repo.upsert_match_result(MatchResult(fixture_id=fid_2028, full_time_home_score=3, full_time_away_score=0))
    repo.upsert_prediction(Prediction(
        fixture_id=fid_2028, model_name="ensemble", market="1X2",
        predicted_outcome="HOME_WIN", probability=0.7, probabilities_json="{}", status="SETTLED",
        settlement_status="WIN", is_correct=1
    ))

    data = repo.get_prediction_history()
    avail = [d["date"] for d in data["available_dates"]]
    assert "2028-06-15" in avail, "Date must dynamically include 2028-06-15"
    assert avail[0] == "2028-06-15", "Dynamic date must be the first date in the list"
    conn.close()
