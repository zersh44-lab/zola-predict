"""Unit tests for the High-Confidence Prediction Selection System."""

import pytest
from zola.prediction.base import ModelPrediction
from zola.prediction.high_confidence import HighConfidenceSelector, MatchHighConfidenceReport


@pytest.fixture
def sample_fixture():
    return {
        "id": 1001,
        "home_team_name": "Arsenal",
        "away_team_name": "Crystal Palace",
        "competition_name": "Premier League",
        "utc_date": "2026-11-21T15:00:00+00:00",
    }


def make_pred(market: str, outcome: str, prob: float, conf: str = "HIGH", conf_score: float = 0.85):
    dist = {outcome: prob}
    if market == "1X2":
        rem = round(1.0 - prob, 4)
        dist["DRAW"] = round(rem * 0.6, 4)
        dist["AWAY_WIN"] = round(rem * 0.4, 4)
    else:
        dist["OTHER"] = round(1.0 - prob, 4)

    return ModelPrediction(
        model_name="ensemble",
        market=market,
        predicted_outcome=outcome,
        probability=prob,
        probabilities=dist,
        confidence=conf,
        confidence_score=conf_score,
    )


def test_high_confidence_selector_returns_max_3_picks(sample_fixture):
    """Test that selector never returns more than 3 picks even if all 8 markets qualify."""
    preds = {
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.82),
        "OU15": make_pred("OU15", "OVER_15", 0.78),
        "OU05": make_pred("OU05", "OVER_05", 0.94),
        "DNB": make_pred("DNB", "1_DNB", 0.75),
        "1X2": make_pred("1X2", "HOME_WIN", 0.66),
        "BTTS": make_pred("BTTS", "BTTS_YES", 0.66),
        "OU25": make_pred("OU25", "OVER_25", 0.67),
    }
    stats = {"home_xg": 2.1, "away_xg": 0.8, "home_goals_scored": 2.2, "away_goals_scored": 1.2}

    report = HighConfidenceSelector.evaluate_fixture(
        sample_fixture,
        preds,
        comparison_stats=stats,
        home_form=["W", "W", "D", "W", "W"],
        away_form=["L", "D", "L", "W", "L"],
    )

    assert isinstance(report, MatchHighConfidenceReport)
    assert not report.is_skipped
    assert 1 <= report.total_picks <= 3
    assert len(report.picks) <= 3
    assert report.match_name == "Arsenal vs Crystal Palace"
    assert report.competition_name == "Premier League"

    # Verify each pick has required fields
    for pick in report.picks:
        assert pick.market in ("DOUBLE_CHANCE", "OU15", "OU05", "DNB", "1X2", "BTTS", "OU25")
        assert pick.probability >= 0.60
        assert pick.confidence in ("HIGH", "MEDIUM")
        assert len(pick.rationale) > 20
        assert len(pick.risks) > 10
        assert pick.estimated_odd >= 1.0


def test_high_confidence_selector_skips_low_confidence_match(sample_fixture):
    """Test that when markets fail the confidence threshold, a SKIP report is returned."""
    # Narrow toss-up match with low conviction
    preds = {
        "1X2": make_pred("1X2", "HOME_WIN", 0.42, conf="LOW", conf_score=0.35),
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.62, conf="LOW", conf_score=0.40),
        "OU15": make_pred("OU15", "OVER_15", 0.58, conf="LOW", conf_score=0.38),
        "OU25": make_pred("OU25", "UNDER_25", 0.52, conf="NO_BET", conf_score=0.30),
        "BTTS": make_pred("BTTS", "BTTS_NO", 0.51, conf="LOW", conf_score=0.32),
    }
    stats = {"home_xg": 1.1, "away_xg": 1.0}

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds, comparison_stats=stats)

    assert report.is_skipped is True
    assert report.skip_reason == "NO HIGH-CONFIDENCE PREDICTION — SKIP THIS MATCH."
    assert "SKIP THIS MATCH" in report.skip_rationale
    assert report.total_picks == 0
    assert len(report.picks) == 0


def test_high_confidence_selector_allows_1_or_2_picks_without_forcing_3(sample_fixture):
    """Test that system does not force 3 picks if only 1 is reliable."""
    # Only Double Chance qualifies; goals markets are weak or coin-flips
    preds = {
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.81, conf="HIGH", conf_score=0.88),
        "OU15": make_pred("OU15", "OVER_15", 0.55, conf="LOW", conf_score=0.35),
        "OU25": make_pred("OU25", "UNDER_25", 0.52, conf="LOW", conf_score=0.30),
        "1X2": make_pred("1X2", "HOME_WIN", 0.51, conf="LOW", conf_score=0.38),
    }

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds)

    assert not report.is_skipped
    assert report.total_picks == 1
    assert len(report.picks) == 1
    assert report.picks[0].market == "DOUBLE_CHANCE"
    assert report.picks[0].predicted_outcome == "1X"


def test_high_confidence_selector_deduplication(sample_fixture):
    """Test that redundant picks (Over 1.5 + Over 0.5, or 1X + 1_DNB) are deduplicated."""
    preds = {
        "DOUBLE_CHANCE": make_pred("DOUBLE_CHANCE", "1X", 0.82),
        "DNB": make_pred("DNB", "1_DNB", 0.78),
        "OU15": make_pred("OU15", "OVER_15", 0.79),
        "OU05": make_pred("OU05", "OVER_05", 0.95),
    }

    report = HighConfidenceSelector.evaluate_fixture(sample_fixture, preds)

    assert not report.is_skipped
    markets = [p.market for p in report.picks]
    # Shouldn't include both DOUBLE_CHANCE and DNB
    assert not ("DOUBLE_CHANCE" in markets and "DNB" in markets)
    # Shouldn't include both OU15 and OU05
    assert not ("OU15" in markets and "OU05" in markets)


def test_high_confidence_engine_persistence(repo):
    """Test that PredictionEngine executes HighConfidenceSelector and persists metadata to SQLite."""
    import json
    from zola.database.models import Competition, Fixture, Team
    from zola.prediction.engine import PredictionEngine

    cid = repo.upsert_competition(
        Competition(external_id="PL", provider="api-football", name="Premier League", code="PL")
    )
    t1_id = repo.upsert_team(Team(external_id="1", provider="api-football", name="Arsenal", short_name="Arsenal"))
    t2_id = repo.upsert_team(Team(external_id="2", provider="api-football", name="Chelsea", short_name="Chelsea"))

    fid = repo.upsert_fixture(
        Fixture(
            external_id="upcoming_derby_hc_101",
            provider="api-football",
            competition_id=cid,
            utc_date="2026-11-25 20:00:00",
            status="SCHEDULED",
            home_team_id=t1_id,
            away_team_id=t2_id,
        )
    )

    engine = PredictionEngine(repo)
    results = engine.predict_fixture(fid, persist=True)

    assert "ensemble" in results
    ens = results["ensemble"]
    # Check that high_confidence_report is attached
    sample_market = next(iter(ens.values()))
    assert "high_confidence_report" in sample_market.features
    report = sample_market.features["high_confidence_report"]
    assert "is_skipped" in report
    assert "total_picks" in report

    # Verify persisted prediction in database
    db_preds = repo.get_predictions_for_fixture(fid)
    assert len(db_preds) > 0
    ensemble_preds = [p for p in db_preds if p.model_name == "ensemble"]
    assert len(ensemble_preds) == 8
    for p in ensemble_preds:
        feat = json.loads(p.features_json) if p.features_json else {}
        assert "high_confidence_report" in feat

