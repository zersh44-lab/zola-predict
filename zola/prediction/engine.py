"""Orchestrator for the 6 independent statistical models, quality checks, and database persistence."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from zola.database.models import Prediction
from zola.database.repository import FootballRepository
from zola.prediction.base import (
    BasePredictionModel,
    ModelPrediction,
    verify_market_consistency,
)
from zola.prediction.dixon_coles import DixonColesModel
from zola.prediction.elo import EloRatingModel
from zola.prediction.ensemble import ConsensusEnsembleModel
from zola.prediction.form_momentum import FormMomentumModel
from zola.prediction.head_to_head import HeadToHeadModel
from zola.prediction.poisson import PoissonModel
from zola.prediction.value import analyze_market_value

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Core prediction engine coordinating all 6 independent prediction models."""

    def __init__(self, repository: FootballRepository):
        self.repository = repository
        self.base_models: List[BasePredictionModel] = [
            PoissonModel(),
            DixonColesModel(),
            EloRatingModel(),
            FormMomentumModel(),
            HeadToHeadModel(),
        ]
        self.ensemble = ConsensusEnsembleModel()

    def predict_fixture(
        self,
        fixture_id: int,
        persist: bool = True,
        odds_data: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, Dict[str, ModelPrediction]]:
        """Run all 6 models for a fixture and return their market predictions for all 8 markets."""
        fixture = self.repository.get_fixture_by_id(fixture_id)
        if not fixture:
            raise ValueError(f"Fixture with ID {fixture_id} not found in database.")

        home_team_id = fixture["home_team_id"]
        away_team_id = fixture["away_team_id"]
        competition_id = fixture["competition_id"]

        # Fetch historical inputs
        home_matches = self.repository.get_team_matches(home_team_id, limit=30)
        away_matches = self.repository.get_team_matches(away_team_id, limit=30)
        h2h_matches = self.repository.get_head_to_head_matches(home_team_id, away_team_id, limit=15)
        league_matches = self.repository.get_competition_matches(competition_id, limit=200)

        all_results: Dict[str, Dict[str, ModelPrediction]] = {}

        # 1. Run 5 independent statistical models
        for model in self.base_models:
            try:
                preds = model.predict(
                    fixture=fixture,
                    home_matches=home_matches,
                    away_matches=away_matches,
                    h2h_matches=h2h_matches,
                    league_matches=league_matches,
                )
                all_results[model.name] = preds
            except Exception as e:
                logger.error("Error running prediction model %s for fixture %s: %s", model.name, fixture_id, e)

        # 2. Retrieve historical Brier scores to dynamically weight models
        historical_brier: Dict[str, Dict[str, float]] = {}
        try:
            metrics = self.repository.get_accuracy_metrics()
            for m in metrics:
                if m.brier_score is not None:
                    historical_brier.setdefault(m.model_name, {})[m.market] = m.brier_score
        except Exception:
            pass

        # 3. Run Model 6: Decorrelated Consensus Ensemble Meta-Model
        comp_name = fixture.get("competition_name") if fixture else None
        ensemble_preds = self.ensemble.combine(
            all_results,
            historical_brier_scores=historical_brier,
            competition_name=comp_name,
        )
        all_results["ensemble"] = ensemble_preds

        # 4. Automated Quality Control & Pre-Publish Sanity Check
        violations = verify_market_consistency(ensemble_preds)
        if violations:
            logger.warning("Quality Control flagged consistency issues for fixture %s: %s", fixture_id, violations)

        # 5. Market Value Analysis
        for model_name, markets in all_results.items():
            for market, p in markets.items():
                mkt_odds = odds_data.get(market) if odds_data else None
                p.value_analysis = analyze_market_value(
                    market=market,
                    predicted_outcome=p.predicted_outcome,
                    model_probabilities=p.probabilities,
                    bookmaker_odds=mkt_odds,
                )

        # 6. Persist to SQLite if requested
        if persist:
            for model_name, markets in all_results.items():
                for market, p in markets.items():
                    notes_str = "; ".join(p.data_quality_notes) if p.data_quality_notes else None
                    feat_dict = dict(p.features or {})
                    if p.value_analysis:
                        feat_dict["value_analysis"] = p.value_analysis
                    features_str = json.dumps(feat_dict)
                    odds_str = json.dumps(odds_data.get(market)) if odds_data and market in odds_data else None

                    db_pred = Prediction(
                        fixture_id=fixture_id,
                        model_name=model_name,
                        market=market,
                        predicted_outcome=p.predicted_outcome,
                        probability=p.probability,
                        probabilities_json=json.dumps(p.probabilities),
                        confidence=p.confidence,
                        confidence_score=p.confidence_score,
                        data_quality_notes=notes_str,
                        features_json=features_str,
                        status="PENDING",
                        settlement_status="PENDING",
                        model_version="v1.1.0",
                        odds_json=odds_str,
                    )
                    self.repository.upsert_prediction(db_pred)

        return all_results

    def predict_upcoming_fixtures(
        self,
        days_ahead: int = 7,
        competition_id: Optional[int] = None,
        limit: int = 500,
    ) -> int:
        """Find upcoming scheduled fixtures within the next `days_ahead` days and generate predictions."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        now_iso = now.isoformat()
        end_iso = (now + timedelta(days=days_ahead)).isoformat()

        cursor = self.repository.conn.cursor()
        query = """
            SELECT f.id FROM fixtures f
            WHERE f.status IN ('SCHEDULED', 'TIMED', 'TBD', 'NOT_STARTED')
              AND f.utc_date >= ? AND f.utc_date <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM predictions p WHERE p.fixture_id = f.id AND p.model_name = 'ensemble'
              )
        """
        params: List[Any] = [now_iso, end_iso]
        if competition_id:
            query += " AND f.competition_id = ?"
            params.append(competition_id)
        query += " ORDER BY f.utc_date ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        fixture_ids = [r[0] if isinstance(r, (tuple, list)) else r["id"] for r in rows]

        count = 0
        for fid in fixture_ids:
            try:
                self.predict_fixture(int(fid), persist=True)
                count += 1
            except Exception as e:
                logger.error("Failed to predict upcoming fixture %s: %s", fid, e)

        return count
