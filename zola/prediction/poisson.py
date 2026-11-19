"""Poisson Distribution match prediction model with venue-adjusted attack and defense ratings."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from zola.prediction.base import (
    BasePredictionModel,
    ModelPrediction,
    derive_markets_from_score_matrix,
    poisson_pmf,
)


class PoissonModel(BasePredictionModel):
    """Independent Poisson distribution model based on venue-adjusted attack and defense ratings."""

    @property
    def name(self) -> str:
        return "poisson"

    @property
    def description(self) -> str:
        return (
            "Calculates independent goal expectancies (lambda) for home and away sides "
            "using league averages, team attacking strength, and defensive resilience."
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
        n_home = len(home_matches)
        n_away = len(away_matches)

        # Baseline League Averages (dynamic from league_matches if available)
        default_league_home_avg = 1.55
        default_league_away_avg = 1.25

        if league_matches and len(league_matches) >= 8:
            tot_h_goals = sum(m.get("full_time_home_score", 0) or 0 for m in league_matches)
            tot_a_goals = sum(m.get("full_time_away_score", 0) or 0 for m in league_matches)
            league_home_avg = max(0.6, tot_h_goals / len(league_matches))
            league_away_avg = max(0.6, tot_a_goals / len(league_matches))
        else:
            league_home_avg = default_league_home_avg
            league_away_avg = default_league_away_avg
            notes.append("Using standard league baseline due to limited competition history.")

        # Data Quality & Baseline Assessment
        sample_depth = min(1.0, (n_home + n_away) / 10.0)
        league_depth = min(1.0, len(league_matches) / 20.0) if league_matches else 0.45
        confidence_score = round(max(0.45, min(0.95, 0.65 * sample_depth + 0.35 * league_depth)), 2)

        if confidence_score >= 0.70:
            confidence = "HIGH"
        elif confidence_score >= 0.45:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if n_home < 3 or n_away < 3:
            notes.append(f"Calibrated with competition baseline priors (Home: {n_home} matches, Away: {n_away} matches).")
        elif n_home < 6 or n_away < 6:
            notes.append(f"Moderate sample size: Home ({n_home} matches), Away ({n_away} matches).")

        # Home Team Stats: overall + venue-specific
        home_scored_all = [
            m["full_time_home_score"] if m.get("is_home") else m["full_time_away_score"]
            for m in home_matches
            if m.get("full_time_home_score") is not None
        ]
        home_conceded_all = [
            m["full_time_away_score"] if m.get("is_home") else m["full_time_home_score"]
            for m in home_matches
            if m.get("full_time_home_score") is not None
        ]
        home_scored_at_home = [
            m["full_time_home_score"]
            for m in home_matches
            if m.get("is_home") and m.get("full_time_home_score") is not None
        ]
        home_conceded_at_home = [
            m["full_time_away_score"]
            for m in home_matches
            if m.get("is_home") and m.get("full_time_away_score") is not None
        ]

        # Blend venue-specific and overall performance (60% venue, 40% overall)
        avg_h_scored_all = sum(home_scored_all) / len(home_scored_all) if home_scored_all else league_home_avg
        avg_h_conceded_all = sum(home_conceded_all) / len(home_conceded_all) if home_conceded_all else league_away_avg

        if len(home_scored_at_home) >= 2:
            avg_h_scored_venue = sum(home_scored_at_home) / len(home_scored_at_home)
            avg_home_scored = 0.60 * avg_h_scored_venue + 0.40 * avg_h_scored_all
        else:
            avg_home_scored = avg_h_scored_all

        if len(home_conceded_at_home) >= 2:
            avg_h_conceded_venue = sum(home_conceded_at_home) / len(home_conceded_at_home)
            avg_home_conceded = 0.60 * avg_h_conceded_venue + 0.40 * avg_h_conceded_all
        else:
            avg_home_conceded = avg_h_conceded_all

        # Away Team Stats: overall + venue-specific
        away_scored_all = [
            m["full_time_home_score"] if m.get("is_home") else m["full_time_away_score"]
            for m in away_matches
            if m.get("full_time_home_score") is not None
        ]
        away_conceded_all = [
            m["full_time_away_score"] if m.get("is_home") else m["full_time_home_score"]
            for m in away_matches
            if m.get("full_time_home_score") is not None
        ]
        away_scored_away = [
            m["full_time_away_score"]
            for m in away_matches
            if not m.get("is_home") and m.get("full_time_away_score") is not None
        ]
        away_conceded_away = [
            m["full_time_home_score"]
            for m in away_matches
            if not m.get("is_home") and m.get("full_time_home_score") is not None
        ]

        avg_a_scored_all = sum(away_scored_all) / len(away_scored_all) if away_scored_all else league_away_avg
        avg_a_conceded_all = sum(away_conceded_all) / len(away_conceded_all) if away_conceded_all else league_home_avg

        if len(away_scored_away) >= 2:
            avg_a_scored_venue = sum(away_scored_away) / len(away_scored_away)
            avg_away_scored = 0.60 * avg_a_scored_venue + 0.40 * avg_a_scored_all
        else:
            avg_away_scored = avg_a_scored_all

        if len(away_conceded_away) >= 2:
            avg_a_conceded_venue = sum(away_conceded_away) / len(away_conceded_away)
            avg_away_conceded = 0.60 * avg_a_conceded_venue + 0.40 * avg_a_conceded_all
        else:
            avg_away_conceded = avg_a_conceded_all

        # Relative Attack and Defense Strengths
        home_attack = avg_home_scored / league_home_avg
        away_defense = avg_away_conceded / league_home_avg
        lambda_home = max(0.25, home_attack * away_defense * league_home_avg)

        away_attack = avg_away_scored / league_away_avg
        home_defense = avg_home_conceded / league_away_avg
        lambda_away = max(0.25, away_attack * home_defense * league_away_avg)

        # Cap lambda to reasonable bounds (0.25 to 4.5)
        lambda_home = min(4.5, round(lambda_home, 3))
        lambda_away = min(4.5, round(lambda_away, 3))

        # Generate bivariate score matrix (up to 7x7)
        score_matrix: Dict[Tuple[int, int], float] = {}
        for h in range(8):
            for a in range(8):
                score_matrix[(h, a)] = poisson_pmf(h, lambda_home) * poisson_pmf(a, lambda_away)

        features = {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "home_attack_strength": round(home_attack, 3),
            "home_defense_weakness": round(home_defense, 3),
            "away_attack_strength": round(away_attack, 3),
            "away_defense_weakness": round(away_defense, 3),
            "sample_home_matches": n_home,
            "sample_away_matches": n_away,
            "league_home_avg": round(league_home_avg, 2),
            "league_away_avg": round(league_away_avg, 2),
        }

        comp_name = fixture.get("competition_name") if fixture else None

        return derive_markets_from_score_matrix(
            score_matrix=score_matrix,
            model_name=self.name,
            confidence=confidence,
            confidence_score=confidence_score,
            notes=notes,
            features=features,
            competition_name=comp_name,
        )
