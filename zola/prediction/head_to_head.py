"""Head-to-Head (H2H) Historical Matchup prediction model."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from zola.prediction.base import (
    BasePredictionModel,
    ModelPrediction,
    clamp,
    derive_markets_from_score_matrix,
    poisson_pmf,
)


class HeadToHeadModel(BasePredictionModel):
    """Analyzes direct historical encounters and matchup dynamics between the two clubs."""

    @property
    def name(self) -> str:
        return "head_to_head"

    @property
    def description(self) -> str:
        return (
            "Focuses on historical head-to-head match dynamics, direct rivalry trends, "
            "and venue-specific results between the two competing clubs."
        )

    def predict(
        self,
        fixture: Dict[str, Any],
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
        h2h_matches: List[Dict[str, Any]],
        league_matches: List[Dict[str, Any]],
    ) -> Dict[str, ModelPrediction]:
        notes: List[str] = []
        n_h2h = len(h2h_matches)
        home_team_id = fixture.get("home_team_id")

        # Baseline goals
        default_h_lambda = 1.50
        default_a_lambda = 1.15

        if n_h2h == 0:
            confidence = "MEDIUM"
            confidence_score = 0.50
            notes.append("No direct meetings on record; comparative strength calibrated to competition baseline.")
            lambda_h = default_h_lambda
            lambda_a = default_a_lambda
            h2h_home_wins = 0
            h2h_draws = 0
            h2h_away_wins = 0
        elif n_h2h < 3:
            confidence = "MEDIUM"
            confidence_score = 0.58
            notes.append(f"Direct matchup history ({n_h2h} meeting{'s' if n_h2h > 1 else ''}).")
        elif n_h2h < 6:
            confidence = "HIGH"
            confidence_score = 0.75
        else:
            confidence = "HIGH"
            confidence_score = 0.90

        if n_h2h > 0:
            h_scored_total = 0.0
            a_scored_total = 0.0
            h2h_home_wins = 0
            h2h_draws = 0
            h2h_away_wins = 0

            # Weight more recent H2H matches higher
            weight_total = 0.0
            for idx, m in enumerate(h2h_matches):
                w = 0.90 ** idx
                weight_total += w
                is_home = m.get("home_team_id") == home_team_id
                h_goals = m.get("full_time_home_score", 0) or 0
                a_goals = m.get("full_time_away_score", 0) or 0

                team_h_goals = h_goals if is_home else a_goals
                team_a_goals = a_goals if is_home else h_goals

                h_scored_total += team_h_goals * w
                a_scored_total += team_a_goals * w

                if team_h_goals > team_a_goals:
                    h2h_home_wins += 1
                elif team_h_goals == team_a_goals:
                    h2h_draws += 1
                else:
                    h2h_away_wins += 1

            avg_h_goals = h_scored_total / weight_total if weight_total > 0 else default_h_lambda
            avg_a_goals = a_scored_total / weight_total if weight_total > 0 else default_a_lambda

            # Bayesian shrinkage toward baseline when sample is small
            shrinkage = min(1.0, n_h2h / 6.0)
            lambda_h = (shrinkage * avg_h_goals) + ((1.0 - shrinkage) * default_h_lambda)
            lambda_a = (shrinkage * avg_a_goals) + ((1.0 - shrinkage) * default_a_lambda)

            lambda_h = max(0.3, min(4.0, round(lambda_h, 2)))
            lambda_a = max(0.3, min(4.0, round(lambda_a, 2)))

        score_matrix: Dict[Tuple[int, int], float] = {}
        for h in range(8):
            for a in range(8):
                score_matrix[(h, a)] = poisson_pmf(h, lambda_h) * poisson_pmf(a, lambda_a)

        features = {
            "h2h_total_matches": n_h2h,
            "h2h_home_wins": h2h_home_wins,
            "h2h_draws": h2h_draws,
            "h2h_away_wins": h2h_away_wins,
            "lambda_home_h2h": lambda_h,
            "lambda_away_h2h": lambda_a,
        }

        return derive_markets_from_score_matrix(
            score_matrix=score_matrix,
            model_name=self.name,
            confidence=confidence,
            confidence_score=confidence_score,
            notes=notes,
            features=features,
        )
