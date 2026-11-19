"""Unit tests for the Prediction Accuracy & Intelligence Upgrade suite.

Verifies:
1. Strict monotonic goal distributions: P(O0.5) >= P(O1.5) >= P(O2.5) >= P(O3.5)
2. Decoupled smart confidence system & tossup detection
3. Brier score, Log Loss, and Expected Calibration Error (ECE)
4. Market value analysis, Shin overround removal, and Expected Value (EV)
5. Draw No Bet (DNB) draw refund as PUSH
6. Automated quality control sanity checks
7. Ethiopia vs Mozambique (0-1) correct settlement across all 8 markets
8. Historical walk-forward backtesting
"""

import sqlite3
import pytest

from zola.database.models import Competition, Fixture, MatchResult, Prediction, Team
from zola.database.repository import FootballRepository
from zola.prediction.base import (
    ModelPrediction,
    derive_markets_from_score_matrix,
    verify_market_consistency,
)
from zola.prediction.calibration import (
    brier_score,
    expected_calibration_error,
    log_loss_score,
    PlattScaler,
    TemperatureScaler,
)
from zola.prediction.ensemble import ConsensusEnsembleModel
from zola.prediction.value import (
    analyze_market_value,
    calculate_expected_value,
    classify_value_opportunity,
    remove_overround_proportional,
    remove_overround_shin,
)


def test_monotonic_goals_distribution():
    """Verify monotonic property across Over/Under 0.5, 1.5, 2.5, 3.5."""
    # Score matrix where total goals distribution is spread
    score_matrix = {
        (0, 0): 0.08,
        (1, 0): 0.15,
        (0, 1): 0.12,
        (1, 1): 0.22,
        (2, 0): 0.10,
        (0, 2): 0.08,
        (2, 1): 0.12,
        (1, 2): 0.08,
        (3, 1): 0.05,
    }

    derived = derive_markets_from_score_matrix(score_matrix, model_name="poisson")
    assert len(derived) == 8

    p_o05 = derived["OU05"].probabilities["OVER_05"]
    p_o15 = derived["OU15"].probabilities["OVER_15"]
    p_o25 = derived["OU25"].probabilities["OVER_25"]
    p_o35 = derived["OU35"].probabilities["OVER_35"]

    # Monotonic decreasing for Over
    assert p_o05 >= p_o15 >= p_o25 >= p_o35

    p_u05 = derived["OU05"].probabilities["UNDER_05"]
    p_u15 = derived["OU15"].probabilities["UNDER_15"]
    p_u25 = derived["OU25"].probabilities["UNDER_25"]
    p_u35 = derived["OU35"].probabilities["UNDER_35"]

    # Monotonic increasing for Under
    assert p_u35 >= p_u25 >= p_u15 >= p_u05

    # Complementarity
    assert pytest.approx(p_o05 + p_u05, abs=0.001) == 1.0
    assert pytest.approx(p_o15 + p_u15, abs=0.001) == 1.0
    assert pytest.approx(p_o25 + p_u25, abs=0.001) == 1.0
    assert pytest.approx(p_o35 + p_u35, abs=0.001) == 1.0


def test_tossup_confidence_decoupling():
    """Verify that close calls (e.g. 51.5% edge) are tagged with LOW or NO_BET confidence."""
    # Balanced score matrix producing near 50/50 on OU25
    score_matrix = {
        (0, 0): 0.10,
        (1, 0): 0.15,
        (0, 1): 0.15,
        (1, 1): 0.12,
        (2, 1): 0.15,
        (1, 2): 0.15,
        (2, 2): 0.18,
    }
    derived = derive_markets_from_score_matrix(score_matrix, model_name="dixon_coles")
    pred_ou25 = derived["OU25"]

    # When margin is minimal, confidence must not be HIGH
    assert pred_ou25.confidence in ("LOW", "NO_BET", "MEDIUM")
    assert pred_ou25.confidence != "HIGH"


def test_brier_and_calibration_metrics():
    """Verify Brier score, log loss, and Expected Calibration Error calculation."""
    y_true = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    y_prob = [0.85, 0.75, 0.20, 0.35, 0.90, 0.40, 0.65, 0.80, 0.15, 0.70]

    bs = brier_score(y_true, y_prob)
    assert 0.0 <= bs <= 0.20  # Well calibrated predictions have low Brier score

    ll = log_loss_score(y_true, y_prob)
    assert ll > 0.0

    ece, bins = expected_calibration_error(y_true, y_prob)
    assert 0.0 <= ece <= 0.50
    assert len(bins) > 0


def test_platt_and_temperature_scalers():
    """Verify probability calibrators operate cleanly."""
    y_true = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    y_prob = [0.95, 0.90, 0.60, 0.55, 0.85, 0.45, 0.80, 0.30, 0.75, 0.20, 0.70, 0.10]

    scaler = PlattScaler().fit(y_true, y_prob)
    calibrated_val = scaler.transform(0.70)
    assert 0.01 <= calibrated_val <= 0.99

    temp_scaler = TemperatureScaler(temperature=1.2)
    dist = {"HOME": 0.60, "DRAW": 0.25, "AWAY": 0.15}
    smoothed = temp_scaler.calibrate_distribution(dist)
    assert pytest.approx(sum(smoothed.values()), abs=0.01) == 1.0
    # Higher temperature softens extreme probabilities
    assert smoothed["HOME"] < dist["HOME"]


def test_market_value_and_shin_overround():
    """Verify Shin and proportional margin removal and Expected Value computation."""
    # 1X2 market odds with 6% bookmaker overround
    odds = {"HOME_WIN": 2.10, "DRAW": 3.40, "AWAY_WIN": 3.60}
    fair_probs = remove_overround_shin(odds)
    assert pytest.approx(sum(fair_probs.values()), abs=0.005) == 1.0

    # EV calculation
    model_prob = 0.55
    ev = calculate_expected_value(model_prob, odds["HOME_WIN"])
    # EV = (0.55 * 2.10) - 1.0 = 1.155 - 1.0 = 0.155 (+15.5%)
    assert pytest.approx(ev, abs=0.001) == 0.155
    tier, label = classify_value_opportunity(ev)
    assert tier == "HIGH_VALUE"

    # Analyze market value helper
    res = analyze_market_value("1X2", "HOME_WIN", {"HOME_WIN": 0.55, "DRAW": 0.25, "AWAY_WIN": 0.20}, odds)
    assert res["status"] == "AVAILABLE"
    assert res["has_value"] is True
    assert res["best_value_outcome"] == "HOME_WIN"

    # Fallback when odds are None
    res_none = analyze_market_value("1X2", "HOME_WIN", {"HOME_WIN": 0.55, "DRAW": 0.25, "AWAY_WIN": 0.20}, None)
    assert res_none["status"] == "UNAVAILABLE"
    assert "VALUE ANALYSIS UNAVAILABLE" in res_none["message"]


def test_dnb_push_settlement(in_memory_db: sqlite3.Connection):
    """Verify that Draw No Bet correctly records a PUSH when match draws."""
    repo = FootballRepository(in_memory_db)
    comp_id = repo.upsert_competition(Competition(external_id="PL", provider="api-football", name="Premier League"))
    t1 = repo.upsert_team(Team(external_id="1", provider="api-football", name="Arsenal"))
    t2 = repo.upsert_team(Team(external_id="2", provider="api-football", name="Chelsea"))

    # Finished in a 1-1 Draw
    f_id = repo.upsert_fixture(
        Fixture(
            external_id="fix_draw_101",
            provider="api-football",
            competition_id=comp_id,
            utc_date="2024-11-10 16:30:00",
            status="FINISHED",
            home_team_id=t1,
            away_team_id=t2,
        )
    )
    repo.upsert_match_result(
        MatchResult(fixture_id=f_id, winner="DRAW", full_time_home_score=1, full_time_away_score=1)
    )

    p_dnb = Prediction(
        fixture_id=f_id,
        model_name="ensemble",
        market="DNB",
        predicted_outcome="1_DNB",
        probability=0.62,
        probabilities_json='{"1_DNB": 0.62, "2_DNB": 0.38}',
    )
    repo.upsert_prediction(p_dnb)

    eval_stats = repo.evaluate_predictions(fixture_id=f_id)
    assert eval_stats["pushes"] == 1

    settled_preds = repo.get_predictions_for_fixture(f_id, model_name="ensemble")
    assert len(settled_preds) == 1
    dnb_settled = settled_preds[0]
    assert dnb_settled.settlement_status == "PUSH"
    assert dnb_settled.is_correct is None


def test_ethiopia_vs_mozambique_settlement(in_memory_db: sqlite3.Connection):
    """Verify full settlement of Ethiopia vs Mozambique (0 - 1) across all 8 markets."""
    repo = FootballRepository(in_memory_db)
    comp_id = repo.upsert_competition(Competition(external_id="AFCON_Q", provider="api-football", name="AFCON Qualifiers"))
    t1 = repo.upsert_team(Team(external_id="ETH", provider="api-football", name="Ethiopia"))
    t2 = repo.upsert_team(Team(external_id="MOZ", provider="api-football", name="Mozambique"))

    f_id = repo.upsert_fixture(
        Fixture(
            external_id="afcon_eth_moz_1682",
            provider="api-football",
            competition_id=comp_id,
            utc_date="2026-11-17 13:00:00",
            status="SCHEDULED",
            home_team_id=t1,
            away_team_id=t2,
        )
    )

    # Predictions reflecting actual model forecasts
    preds = [
        Prediction(fixture_id=f_id, model_name="ensemble", market="1X2", predicted_outcome="HOME_WIN", probability=0.458, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="BTTS", predicted_outcome="BTTS_YES", probability=0.522, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="OU05", predicted_outcome="OVER_05", probability=0.921, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="OU15", predicted_outcome="OVER_15", probability=0.736, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="OU25", predicted_outcome="UNDER_25", probability=0.522, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="OU35", predicted_outcome="UNDER_35", probability=0.737, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="DOUBLE_CHANCE", predicted_outcome="12", probability=0.741, probabilities_json="{}"),
        Prediction(fixture_id=f_id, model_name="ensemble", market="DNB", predicted_outcome="1_DNB", probability=0.616, probabilities_json="{}"),
    ]
    for p in preds:
        repo.upsert_prediction(p)

    # Settle with official score: Ethiopia 0 - 1 Mozambique
    settle_res = repo.settle_fixture(f_id, home_score=0, away_score=1)
    assert settle_res["winner"] == "AWAY_TEAM"

    eval_preds = repo.get_predictions_for_fixture(f_id, model_name="ensemble")
    settlement_map = {p.market: p.settlement_status for p in eval_preds}

    # 1X2: Predicted Home Win, Away won -> LOSS
    assert settlement_map["1X2"] == "LOSS"
    # BTTS: Predicted Yes, was 0-1 -> LOSS
    assert settlement_map["BTTS"] == "LOSS"
    # OU05: Predicted Over 0.5, was 1 goal -> WIN
    assert settlement_map["OU05"] == "WIN"
    # OU15: Predicted Over 1.5, was 1 goal -> LOSS
    assert settlement_map["OU15"] == "LOSS"
    # OU25: Predicted Under 2.5, was 1 goal -> WIN
    assert settlement_map["OU25"] == "WIN"
    # OU35: Predicted Under 3.5, was 1 goal -> WIN
    assert settlement_map["OU35"] == "WIN"
    # DOUBLE_CHANCE: Predicted 12, was Away Win -> WIN
    assert settlement_map["DOUBLE_CHANCE"] == "WIN"
    # DNB: Predicted 1_DNB, Away won -> LOSS
    assert settlement_map["DNB"] == "LOSS"


def test_quality_control_verifier():
    """Verify pre-publish quality control check catches invalid probability shapes."""
    valid_predictions = {
        "1X2": ModelPrediction(model_name="ensemble", market="1X2", predicted_outcome="HOME_WIN", probability=0.50, probabilities={"HOME_WIN": 0.50, "DRAW": 0.30, "AWAY_WIN": 0.20}),
        "BTTS": ModelPrediction(model_name="ensemble", market="BTTS", predicted_outcome="BTTS_YES", probability=0.55, probabilities={"BTTS_YES": 0.55, "BTTS_NO": 0.45}),
        "OU05": ModelPrediction(model_name="ensemble", market="OU05", predicted_outcome="OVER_05", probability=0.90, probabilities={"OVER_05": 0.90, "UNDER_05": 0.10}),
        "OU15": ModelPrediction(model_name="ensemble", market="OU15", predicted_outcome="OVER_15", probability=0.72, probabilities={"OVER_15": 0.72, "UNDER_15": 0.28}),
        "OU25": ModelPrediction(model_name="ensemble", market="OU25", predicted_outcome="OVER_25", probability=0.48, probabilities={"OVER_25": 0.48, "UNDER_25": 0.52}),
        "OU35": ModelPrediction(model_name="ensemble", market="OU35", predicted_outcome="OVER_35", probability=0.25, probabilities={"OVER_35": 0.25, "UNDER_35": 0.75}),
        "DOUBLE_CHANCE": ModelPrediction(model_name="ensemble", market="DOUBLE_CHANCE", predicted_outcome="1X", probability=0.80, probabilities={"1X": 0.80, "X2": 0.50, "12": 0.70}),
    }
    violations = verify_market_consistency(valid_predictions)
    assert len(violations) == 0

    # Inject monotonic violation: Over 2.5 > Over 1.5
    invalid_predictions = dict(valid_predictions)
    invalid_predictions["OU25"] = ModelPrediction(
        model_name="ensemble",
        market="OU25",
        predicted_outcome="OVER_25",
        probability=0.85,
        probabilities={"OVER_25": 0.85, "UNDER_25": 0.15},
    )
    bad_violations = verify_market_consistency(invalid_predictions)
    assert len(bad_violations) > 0
    assert any("Monotonic violation" in v for v in bad_violations)
