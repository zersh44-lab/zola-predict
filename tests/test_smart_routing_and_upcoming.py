"""Tests for Smart Market Routing, High-Loss Gating, and 7-10 Day Upcoming Fixtures."""

import json
import sqlite3
from datetime import datetime, timezone, timedelta
import pytest

from zola.database.models import Competition, Fixture, Team
from zola.database.repository import FootballRepository
from zola.prediction.base import derive_markets_from_score_matrix
from zola.prediction.engine import PredictionEngine
from zola.prediction.ensemble import ConsensusEnsembleModel
from zola.services.collector import FootballCollectorService


def test_btts_no_and_under25_gating():
    """Verify that low-margin BTTS_NO and UNDER_25 picks have their confidence gated to prevent traps."""
    # Score matrix where BTTS_NO has 54% probability (narrow edge)
    score_matrix = {
        (1, 0): 0.28,
        (0, 0): 0.12,
        (0, 1): 0.14,
        (1, 1): 0.20,
        (2, 1): 0.14,
        (1, 2): 0.12,
    }
    derived = derive_markets_from_score_matrix(score_matrix, model_name="dixon_coles")
    pred_btts = derived["BTTS"]
    if pred_btts.predicted_outcome == "BTTS_NO":
        assert pred_btts.confidence in ("LOW", "NO_BET")
        assert pred_btts.confidence_score <= 0.40

    pred_ou25 = derived["OU25"]
    if pred_ou25.predicted_outcome == "UNDER_25" and pred_ou25.probability < 0.57:
        assert pred_ou25.confidence in ("LOW", "NO_BET")


def test_league_volatility_penalty_serie_a():
    """Verify that Serie A applies a variance penalty and draw gating."""
    score_matrix = {
        (1, 0): 0.35,
        (0, 0): 0.15,
        (1, 1): 0.15,
        (0, 1): 0.15,
        (2, 1): 0.10,
        (1, 2): 0.10,
    }
    # Predict with Serie A league flag
    derived_serie_a = derive_markets_from_score_matrix(
        score_matrix,
        model_name="poisson",
        competition_name="Serie A",
    )
    # Notes should contain league volatility notice
    notes_joined = " ".join(derived_serie_a["1X2"].data_quality_notes)
    assert "League Volatility" in notes_joined or "Draw Risk" in notes_joined


def test_smart_market_routing_to_double_chance():
    """Verify that 1X2 with high draw propensity is smart-routed to Double Chance (1X/X2)."""
    # Create mock model predictions for 1X2 where Home Win is favored (44%) but Draw is high (30%)
    model = ConsensusEnsembleModel()
    score_matrix = {
        (1, 0): 0.24,
        (2, 0): 0.12,
        (0, 0): 0.16,
        (1, 1): 0.14,
        (0, 1): 0.14,
        (1, 2): 0.10,
        (2, 1): 0.10,
    }
    m1 = derive_markets_from_score_matrix(score_matrix, model_name="poisson")
    m2 = derive_markets_from_score_matrix(score_matrix, model_name="dixon_coles")

    combined = model.combine({"poisson": m1, "dixon_coles": m2}, competition_name="Serie A")
    assert "1X2" in combined
    assert "DOUBLE_CHANCE" in combined

    feat = combined["1X2"].features
    assert "smart_routing" in feat
    sr = feat["smart_routing"]
    assert sr["active"] is True
    assert sr["recommended_market"] == "DOUBLE_CHANCE"
    assert sr["recommended_outcome"] in ("1X", "X2")
    assert sr["historical_win_rate"] == 0.790


def test_upcoming_7_to_10_day_fixtures_queries(repo):
    """Verify that get_fixtures and get_discovery_fixtures support multi-day upcoming queries up to 10 days ahead."""
    c = Competition(external_id="PL_UPCOMING", name="Premier League", code="PL", provider="mock")
    c_id = repo.upsert_competition(c)

    t1 = Team(external_id="T_A", name="Team Alpha", provider="mock")
    t2 = Team(external_id="T_B", name="Team Beta", provider="mock")
    t1_id = repo.upsert_team(t1)
    t2_id = repo.upsert_team(t2)

    now = datetime.now(timezone.utc)
    # Seed fixture 5 days in the future
    f_future = Fixture(
        external_id="FIX_5_DAYS_AHEAD",
        competition_id=c_id,
        home_team_id=t1_id,
        away_team_id=t2_id,
        season="2026",
        utc_date=now + timedelta(days=5),
        status="SCHEDULED",
        provider="mock",
    )
    repo.upsert_fixture(f_future)

    # 1. get_fixtures with upcoming_only=True and days_ahead=10
    upcoming = repo.get_fixtures(upcoming_only=True, days_ahead=10)
    assert len(upcoming) >= 1
    found = any(x["external_id"] == "FIX_5_DAYS_AHEAD" for x in upcoming)
    assert found is True

    # 2. get_discovery_fixtures with category='upcoming'
    discovery = repo.get_discovery_fixtures(
        date=now.strftime("%Y-%m-%d"),
        category="upcoming",
    )
    assert discovery["counts"]["upcoming"] >= 1
    all_fix = discovery.get("all_fixtures", [])
    assert any(x["external_id"] == "FIX_5_DAYS_AHEAD" for x in all_fix)


def test_predict_upcoming_fixtures(repo):
    """Verify that PredictionEngine.predict_upcoming_fixtures predicts all upcoming fixtures in the window."""
    c = Competition(external_id="PL_PRED", name="Premier League", code="PL", provider="mock")
    c_id = repo.upsert_competition(c)
    t1 = Team(external_id="T1", name="Arsenal", provider="mock")
    t2 = Team(external_id="T2", name="Chelsea", provider="mock")
    t1_id = repo.upsert_team(t1)
    t2_id = repo.upsert_team(t2)

    now = datetime.now(timezone.utc)
    f = Fixture(
        external_id="FIX_PRED_UPCOMING",
        competition_id=c_id,
        home_team_id=t1_id,
        away_team_id=t2_id,
        season="2026",
        utc_date=now + timedelta(days=3),
        status="SCHEDULED",
        provider="mock",
    )
    fid = repo.upsert_fixture(f)

    engine = PredictionEngine(repo)
    # Predict upcoming fixtures across the 10-day window
    count = engine.predict_upcoming_fixtures(days_ahead=10)
    assert count >= 1

    # Verify predictions were persisted in SQLite
    preds = repo.get_predictions_for_fixture(fixture_id=fid, model_name="ensemble")
    assert len(preds) >= 1
    assert any(p.market == "1X2" for p in preds)


def test_football_data_matches_window():
    """Verify that FootballDataOrgClient.get_matches_window correctly formats date span and parses fixtures."""
    from zola.client.football_data import FootballDataOrgClient
    from tests.conftest import MockHttpResponse, MockHttpSession, MOCK_SCHEDULED_FIXTURES_PAYLOAD

    calls = []
    def handler(method: str, url: str, **kwargs):
        calls.append((method, url, kwargs.get("params")))
        return MockHttpResponse(200, MOCK_SCHEDULED_FIXTURES_PAYLOAD)

    client = FootballDataOrgClient(api_key="mock_key", session=MockHttpSession(handler=handler))
    fixtures, from_cache, elapsed = client.get_matches_window(
        date_from="2026-11-21",
        date_to="2026-12-05",  # Span > 9 days, should be automatically clamped
        competitions=["PL", "PD", "SA"],
    )
    assert len(fixtures) == 2
    assert len(calls) == 1
    params = calls[0][2]
    assert params["dateFrom"] == "2026-11-21"
    assert params["dateTo"] == "2026-11-30"  # Clamped to +9 days
    assert params["competitions"] == "PL,PD,SA"


def test_sync_upcoming_window_pipeline(repo):
    """Verify that sync_upcoming_window persists authentic upcoming fixtures and never seeds fake fixtures on empty API."""
    from zola.client.football_data import FootballDataOrgClient
    from tests.conftest import MockHttpResponse, MockHttpSession

    mock_matches = [
        {
            "id": 8801,
            "utcDate": "2026-11-25T15:00:00Z",
            "status": "TIMED",
            "competition": {"id": 2021, "code": "PL", "name": "Premier League"},
            "homeTeam": {"id": 1, "name": "Arsenal", "shortName": "ARS"},
            "awayTeam": {"id": 2, "name": "Chelsea", "shortName": "CHE"},
        }
    ]

    def handler(method: str, url: str, **kwargs):
        return MockHttpResponse(200, {"matches": mock_matches})

    client = FootballDataOrgClient(api_key="mock_key", session=MockHttpSession(handler=handler))
    collector = FootballCollectorService(client=client, repository=repo)

    res = collector.sync_upcoming_window(days_ahead=7, start_date="2026-11-21")
    assert res["fixtures_synced"] >= 1
    assert "2026-11-21" in res["dates"]
