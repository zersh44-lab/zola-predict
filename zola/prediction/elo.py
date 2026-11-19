"""Elo Dynamic Power Rating prediction model for football matches."""

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


class EloRatingModel(BasePredictionModel):
    """Dynamic Elo rating system tracking team strength with home advantage adjustment."""

    def __init__(self, base_rating: float = 1500.0, home_advantage: float = 65.0, k_factor: float = 32.0):
        self.base_rating = base_rating
        self.home_advantage = home_advantage
        self.k_factor = k_factor

    @property
    def name(self) -> str:
        return "elo"

    @property
    def description(self) -> str:
        return (
            "Calculates team strength ratings using the standard Elo rating framework, "
            "incorporating home field advantage and goal-margin sensitivity."
        )

    def _calculate_team_elo(self, matches: List[Dict[str, Any]]) -> float:
        """Estimate team's current Elo rating by replaying chronological match outcomes."""
        rating = self.base_rating
        # Sort chronologically (oldest to newest)
        chrono_matches = sorted(matches, key=lambda m: m.get("utc_date", ""))

        for m in chrono_matches:
            h_score = m.get("full_time_home_score")
            a_score = m.get("full_time_away_score")
            if h_score is None or a_score is None:
                continue

            is_home = bool(m.get("is_home", 1))
            my_score = h_score if is_home else a_score
            opp_score = a_score if is_home else h_score

            if my_score > opp_score:
                actual = 1.0
            elif my_score == opp_score:
                actual = 0.5
            else:
                actual = 0.0

            # Opponent assumed around base_rating or slight differential
            opp_rating = self.base_rating
            diff = (rating + (self.home_advantage if is_home else 0.0)) - opp_rating
            expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            goal_diff = abs(my_score - opp_score)
            multiplier = 1.0 if goal_diff <= 1 else math.sqrt(goal_diff)

            rating += self.k_factor * multiplier * (actual - expected)

        return round(rating, 1)

    def predict(
        self,
        fixture: Dict[str, Any],
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
        h2h_matches: List[Dict[str, Any]],
        league_matches: List[Dict[str, Any]],
    ) -> Dict[str, ModelPrediction]:
        notes: List[str] = []
        n_home = len(home_matches)
        n_away = len(away_matches)

        home_elo = self._calculate_team_elo(home_matches)
        away_elo = self._calculate_team_elo(away_matches)

        # Adjust for home advantage
        effective_home_elo = home_elo + self.home_advantage
        elo_diff = effective_home_elo - away_elo

        sample_depth = min(1.0, (n_home + n_away) / 12.0)
        elo_decisiveness = min(1.0, abs(elo_diff) / 280.0)
        confidence_score = round(max(0.45, min(0.95, 0.40 * sample_depth + 0.60 * elo_decisiveness + 0.15)), 2)

        if confidence_score >= 0.70:
            confidence = "HIGH"
        elif confidence_score >= 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if n_home < 3 or n_away < 3:
            notes.append(f"Elo regularized with baseline strength rating (Home: {n_home}, Away: {n_away}).")

        # Expected score for home team
        e_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # Model draw probability with bell curve around equality calibrated to competition
        comp_str = str(fixture.get("competition_name") or fixture.get("competition_code") or "").lower()
        if any(c in comp_str for c in ("serie a", "la liga", "laliga")):
            base_draw_peak = 0.29
        elif any(c in comp_str for c in ("bundesliga", "eredivisie")):
            base_draw_peak = 0.25
        elif "premier league" in comp_str:
            base_draw_peak = 0.23
        else:
            base_draw_peak = 0.26

        p_draw = base_draw_peak * math.exp(-((elo_diff) ** 2) / (2.0 * (240.0 ** 2)))
        p_draw = clamp(p_draw, 0.12, 0.34)

        p_home = clamp(e_home - (0.5 * p_draw), 0.05, 0.85)
        p_away = clamp((1.0 - e_home) - (0.5 * p_draw), 0.05, 0.85)

        total_1x2 = p_home + p_draw + p_away
        p_home = round(p_home / total_1x2, 4)
        p_draw = round(p_draw / total_1x2, 4)
        p_away = round(1.0 - p_home - p_draw, 4)

        # Estimate goal expectancies corresponding to competition & Elo ratings
        if league_matches and len(league_matches) >= 8:
            tot_goals = sum((m.get("full_time_home_score", 0) or 0) + (m.get("full_time_away_score", 0) or 0) for m in league_matches)
            lambda_total = max(1.8, min(3.8, tot_goals / len(league_matches)))
        else:
            lambda_total = 2.70

        ratio_home = p_home / (p_home + p_away or 1.0)
        lambda_h = max(0.4, min(3.8, round(lambda_total * ratio_home, 2)))
        lambda_a = max(0.4, min(3.8, round(lambda_total * (1.0 - ratio_home), 2)))

        score_matrix: Dict[Tuple[int, int], float] = {}
        for h in range(8):
            for a in range(8):
                score_matrix[(h, a)] = poisson_pmf(h, lambda_h) * poisson_pmf(a, lambda_a)

        features = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "effective_home_elo": effective_home_elo,
            "elo_diff": round(elo_diff, 1),
            "expected_score_home": round(e_home, 3),
            "base_draw_peak": base_draw_peak,
            "lambda_total": round(lambda_total, 2),
        }

        return derive_markets_from_score_matrix(
            score_matrix=score_matrix,
            model_name=self.name,
            confidence=confidence,
            confidence_score=confidence_score,
            notes=notes,
            features=features,
        )
