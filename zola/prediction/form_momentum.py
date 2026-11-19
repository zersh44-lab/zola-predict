"""Recent Form & Momentum statistical prediction model."""

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


class FormMomentumModel(BasePredictionModel):
    """Evaluates short-term form, momentum trends, and home/away splits."""

    def __init__(self, recent_matches_count: int = 6):
        self.recent_matches_count = recent_matches_count

    @property
    def name(self) -> str:
        return "form_momentum"

    @property
    def description(self) -> str:
        return (
            "Analyzes exponential weighted form over the last 5-10 matches, "
            "tracking points-per-game (PPG), goal differential, and home/away splits."
        )

    def _calculate_form_metrics(self, matches: List[Dict[str, Any]]) -> Dict[str, float]:
        recent = matches[: self.recent_matches_count]
        if not recent:
            return {
                "ppg": 1.3,
                "gd_per_game": 0.0,
                "clean_sheet_rate": 0.25,
                "scoring_rate": 1.3,
                "conceding_rate": 1.2,
                "sample": 0,
            }

        weighted_points = 0.0
        weighted_goals_scored = 0.0
        weighted_goals_conceded = 0.0
        clean_sheets = 0
        total_weight = 0.0

        decay = 0.88  # Most recent match gets highest weight

        for idx, m in enumerate(recent):
            weight = decay ** idx
            is_home = bool(m.get("is_home", 1))
            h_score = m.get("full_time_home_score", 0) or 0
            a_score = m.get("full_time_away_score", 0) or 0

            my_goals = h_score if is_home else a_score
            opp_goals = a_score if is_home else h_score

            if my_goals > opp_goals:
                points = 3.0
            elif my_goals == opp_goals:
                points = 1.0
            else:
                points = 0.0

            if opp_goals == 0:
                clean_sheets += 1

            weighted_points += points * weight
            weighted_goals_scored += my_goals * weight
            weighted_goals_conceded += opp_goals * weight
            total_weight += weight

        ppg = weighted_points / total_weight if total_weight > 0 else 1.3
        avg_scored = weighted_goals_scored / total_weight if total_weight > 0 else 1.3
        avg_conceded = weighted_goals_conceded / total_weight if total_weight > 0 else 1.2
        cs_rate = clean_sheets / len(recent)

        return {
            "ppg": round(ppg, 2),
            "gd_per_game": round(avg_scored - avg_conceded, 2),
            "clean_sheet_rate": round(cs_rate, 2),
            "scoring_rate": round(avg_scored, 2),
            "conceding_rate": round(avg_conceded, 2),
            "sample": len(recent),
        }

    def predict(
        self,
        fixture: Dict[str, Any],
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
        h2h_matches: List[Dict[str, Any]],
        league_matches: List[Dict[str, Any]],
    ) -> Dict[str, ModelPrediction]:
        notes: List[str] = []
        home_form = self._calculate_form_metrics(home_matches)
        away_form = self._calculate_form_metrics(away_matches)

        n_home = int(home_form["sample"])
        n_away = int(away_form["sample"])

        # Form points difference (-3.0 to +3.0)
        # Add slight home bonus of +0.3 PPG
        ppg_diff = (home_form["ppg"] + 0.3) - away_form["ppg"]

        sample_depth = min(1.0, (n_home + n_away) / 10.0)
        momentum_conviction = min(1.0, abs(ppg_diff) / 1.5)
        confidence_score = round(max(0.42, min(0.90, 0.45 * sample_depth + 0.55 * momentum_conviction + 0.15)), 2)

        if confidence_score >= 0.70:
            confidence = "HIGH"
        elif confidence_score >= 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if n_home < 3 or n_away < 3:
            notes.append(f"Form calibrated with league performance baseline (Home: {n_home}, Away: {n_away}).")

        # Convert PPG diff to expected lambda goals
        base_h = home_form["scoring_rate"] * 0.6 + away_form["conceding_rate"] * 0.4 + 0.15
        base_a = away_form["scoring_rate"] * 0.6 + home_form["conceding_rate"] * 0.4 - 0.15

        lambda_h = max(0.3, min(4.0, round(base_h + (ppg_diff * 0.25), 2)))
        lambda_a = max(0.3, min(4.0, round(base_a - (ppg_diff * 0.25), 2)))

        score_matrix: Dict[Tuple[int, int], float] = {}
        for h in range(8):
            for a in range(8):
                score_matrix[(h, a)] = poisson_pmf(h, lambda_h) * poisson_pmf(a, lambda_a)

        features = {
            "home_ppg": home_form["ppg"],
            "away_ppg": away_form["ppg"],
            "home_clean_sheet_rate": home_form["clean_sheet_rate"],
            "away_clean_sheet_rate": away_form["clean_sheet_rate"],
            "form_ppg_diff": round(ppg_diff, 2),
            "home_form_sample": n_home,
            "away_form_sample": n_away,
        }

        return derive_markets_from_score_matrix(
            score_matrix=score_matrix,
            model_name=self.name,
            confidence=confidence,
            confidence_score=confidence_score,
            notes=notes,
            features=features,
        )
