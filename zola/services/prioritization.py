"""Intelligent Match Prioritization Engine inspired by BeSoccer discovery workflows.

Computes a calibrated 0-100 priority score for every football fixture using:
1. Competition Importance (30%)
2. Competition Popularity (20%)
3. Team Popularity (20%)
4. Match Relevance & High-Stakes Context (15%)
5. Kickoff Proximity (10%)
6. Data Completeness (5%)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PrioritizationWeights:
    """Configurable scoring weights summing to 1.0 (or 100%)."""
    competition_importance: float = 0.30
    competition_popularity: float = 0.20
    team_popularity: float = 0.20
    match_relevance: float = 0.15
    kickoff_proximity: float = 0.10
    data_completeness: float = 0.05

    def normalize(self) -> "PrioritizationWeights":
        total = (
            self.competition_importance
            + self.competition_popularity
            + self.team_popularity
            + self.match_relevance
            + self.kickoff_proximity
            + self.data_completeness
        )
        if total <= 0:
            return PrioritizationWeights()
        return PrioritizationWeights(
            competition_importance=round(self.competition_importance / total, 3),
            competition_popularity=round(self.competition_popularity / total, 3),
            team_popularity=round(self.team_popularity / total, 3),
            match_relevance=round(self.match_relevance / total, 3),
            kickoff_proximity=round(self.kickoff_proximity / total, 3),
            data_completeness=round(self.data_completeness / total, 3),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# Tier 1 Competitions (Major International & Top 5 European Leagues)
TIER_1_COMPETITIONS = {
    "uefa nations league",
    "premier league",
    "la liga",
    "laliga",
    "primera division",
    "serie a",
    "bundesliga",
    "ligue 1",
    "uefa champions league",
    "fifa world cup",
    "uefa european championship",
    "euro",
    "copa america",
}

# Tier 2 Competitions (Second Tier Continental & Strong Domestic Leagues/Cups)
TIER_2_COMPETITIONS = {
    "uefa europa league",
    "uefa europa conference league",
    "copa libertadores",
    "championship",
    "eredivisie",
    "primeira liga",
    "liga portugal",
    "fa cup",
    "copa del rey",
    "dfb pokal",
    "coppa italia",
    "coupe de france",
    "major league soccer",
    "mls",
    "saudi pro league",
    "brasileirão",
    "serie a - brazil",
    "liga mx",
    "africa cup of nations",
    "concacaf nations league",
    "asian cup",
}

# Tier 3 Competitions (Competitive European / Global Leagues)
TIER_3_COMPETITIONS = {
    "super lig",
    "scottish premiership",
    "premiership",
    "belgian pro league",
    "pro league",
    "super league",
    "2. bundesliga",
    "segunda division",
    "laliga 2",
    "serie b",
    "ligue 2",
    "caf champions league",
    "afc champions league",
    "copa sudamericana",
    "argentina liga profesional",
}

# High-Profile Global Clubs & National Teams (Top Popularity Tier)
GLOBAL_MARQUEE_TEAMS = {
    # Elite European Clubs
    "real madrid", "barcelona", "manchester city", "liverpool", "arsenal",
    "bayern munich", "bayern münchen", "paris saint germain", "psg",
    "inter", "inter milan", "juventus", "ac milan", "atletico madrid",
    "chelsea", "manchester united", "tottenham", "borussia dortmund",
    "bayer leverkusen", "aston villa", "newcastle", "napoli", "roma",
    "benfica", "sporting cp", "porto", "ajax", "psv", "galatasaray", "fenerbahce",
    # Global Elite National Teams
    "france", "england", "spain", "germany", "portugal", "netherlands",
    "belgium", "italy", "croatia", "brazil", "argentina", "uruguay",
    "colombia", "morocco", "senegal", "japan", "denmark", "switzerland",
    "norway", "sweden", "wales", "poland", "austria", "turkey", "türkiye",
    "scotland", "serbia", "czech republic", "hungary", "ukraine", "georgia",
}

SECOND_TIER_TEAMS = {
    "west ham", "brighton", "crystal palace", "everton", "wolves", "fulham",
    "sevilla", "real betis", "athletic bilbao", "real sociedad", "villarreal",
    "lazio", "fiorentina", "atalanta", "bologna", "torino",
    "stuttgart", "eintracht frankfurt", "rb leipzig", "freiburg",
    "monaco", "marseille", "lyon", "lille", "lens",
    "al hilal", "al nassr", "inter miami", "flamengo", "palmeiras",
    "boca juniors", "river plate", "celtic", "rangers",
    # Notable National Teams
    "usa", "united states", "mexico", "south korea", "australia",
    "egypt", "nigeria", "ghana", "ivory coast", "algeria", "cameroon",
    "romania", "slovakia", "slovenia", "albania", "finland", "greece",
    "republic of ireland", "northern ireland", "bosnia & herzegovina",
    "israel", "kosovo", "bulgaria", "estonia", "montenegro", "cyprus",
}


class MatchPrioritizationEngine:
    """Evaluates and ranks football matches for optimal user discovery."""

    def __init__(self, weights: Optional[PrioritizationWeights] = None):
        self.weights = weights.normalize() if weights else PrioritizationWeights()

    def update_weights(self, new_weights: Dict[str, float]) -> PrioritizationWeights:
        """Update prioritization scoring weights dynamically."""
        self.weights = PrioritizationWeights(
            competition_importance=float(new_weights.get("competition_importance", self.weights.competition_importance)),
            competition_popularity=float(new_weights.get("competition_popularity", self.weights.competition_popularity)),
            team_popularity=float(new_weights.get("team_popularity", self.weights.team_popularity)),
            match_relevance=float(new_weights.get("match_relevance", self.weights.match_relevance)),
            kickoff_proximity=float(new_weights.get("kickoff_proximity", self.weights.kickoff_proximity)),
            data_completeness=float(new_weights.get("data_completeness", self.weights.data_completeness)),
        ).normalize()
        return self.weights

    def get_competition_importance(self, comp_name: str, comp_country: Optional[str] = None) -> float:
        """Score competition importance from 0 to 100 based on sporting tier."""
        name_clean = (comp_name or "").lower().strip()
        country_clean = (comp_country or "").lower().strip()

        # Check Tier 1
        for t1 in TIER_1_COMPETITIONS:
            if t1 in name_clean:
                # Disambiguate lower divisions like "U18 Premier League" or "Premier League 2"
                if any(youth in name_clean for youth in ("u17", "u18", "u19", "u21", "u23", "2 division", "cup")):
                    return 55.0
                return 98.0

        # Check Tier 2
        for t2 in TIER_2_COMPETITIONS:
            if t2 in name_clean:
                if any(youth in name_clean for youth in ("u17", "u18", "u19", "u21", "u23")):
                    return 45.0
                return 80.0

        # Check Tier 3
        for t3 in TIER_3_COMPETITIONS:
            if t3 in name_clean:
                return 65.0

        # International youth competitions
        if "uefa" in name_clean or "fifa" in name_clean:
            return 50.0

        # National top-tier leagues
        if "premier" in name_clean or "first" in name_clean or "liga 1" in name_clean:
            return 45.0

        # Reserve or lower division leagues
        if any(w in name_clean for w in ("reserve", "res.", "regional", "3. liga", "tercera", "division 2", "division 3")):
            return 20.0

        return 30.0

    def get_competition_popularity(self, comp_name: str) -> float:
        """Score global viewership and popularity of the competition."""
        name_clean = (comp_name or "").lower().strip()

        if any(k in name_clean for k in ("champions league", "world cup", "euro")):
            return 100.0
        if "premier league" in name_clean and not any(y in name_clean for y in ("u18", "u21", "2")):
            return 98.0
        if any(k in name_clean for k in ("nations league", "la liga", "serie a", "bundesliga")):
            return 90.0
        if "ligue 1" in name_clean and not any(y in name_clean for y in ("congo", "algeria")):
            return 82.0
        if any(k in name_clean for k in ("europa league", "copa libertadores", "copa america")):
            return 80.0
        if any(k in name_clean for k in ("mls", "saudi", "championship", "fa cup")):
            return 68.0
        if "africa cup of nations" in name_clean:
            return 72.0
        if "qualif" in name_clean:
            return 60.0
        return 35.0

    def get_team_popularity(self, team_name: str) -> float:
        """Score global fanbase, following, and stature of the team."""
        t_clean = (team_name or "").lower().strip()

        for marquee in GLOBAL_MARQUEE_TEAMS:
            if marquee == t_clean or (len(marquee) > 4 and marquee in t_clean):
                return 95.0

        for second in SECOND_TIER_TEAMS:
            if second == t_clean or (len(second) > 4 and second in t_clean):
                return 75.0

        # National teams generally have higher interest than obscure club reserve teams
        if not any(w in t_clean for w in ("res", "u17", "u18", "u19", "u21", "u23", "ii", " b", "w")):
            return 45.0

        return 20.0

    def get_match_relevance(self, fixture: Dict[str, Any]) -> float:
        """Score match stakes, derby factor, and head-to-head relevance."""
        home_team = (fixture.get("home_team_name") or "").lower()
        away_team = (fixture.get("away_team_name") or "").lower()
        comp_name = (fixture.get("competition_name") or "").lower()
        matchday = str(fixture.get("matchday") or "").lower()

        score = 50.0

        # Marquee clash: Both teams are global marquee teams
        h_pop = self.get_team_popularity(home_team)
        a_pop = self.get_team_popularity(away_team)
        if h_pop >= 90 and a_pop >= 90:
            score += 35.0  # Massive derby/clash
        elif h_pop >= 75 and a_pop >= 75:
            score += 20.0

        # Knockout or Final stakes
        if any(k in matchday for k in ("final", "semi-final", "quarter-final", "play-off", "1/8")):
            score += 15.0

        # Group decider / League A clash
        if "league a" in matchday or "group stage - 6" in matchday:
            score += 10.0

        return min(100.0, score)

    def get_kickoff_proximity(self, utc_date_str: str, status: str) -> float:
        """Score proximity to kickoff: live matches and imminent games get highest score."""
        status_norm = (status or "").upper()
        if status_norm in ("IN_PLAY", "LIVE", "PAUSED"):
            return 100.0

        if not utc_date_str:
            return 40.0

        try:
            clean_str = utc_date_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean_str)
            now = datetime.now(timezone.utc)
            delta_minutes = (dt - now).total_seconds() / 60.0

            if delta_minutes < 0:
                # Finished match
                hours_ago = abs(delta_minutes) / 60.0
                if hours_ago <= 3:
                    return 75.0  # Just finished, high post-match review interest
                return max(10.0, 60.0 - (hours_ago * 2.0))
            elif delta_minutes <= 60:
                return 95.0  # Kicking off within 1 hour
            elif delta_minutes <= 180:
                return 85.0  # Kicking off within 3 hours
            elif delta_minutes <= 720:
                return 70.0  # Kicking off today
            elif delta_minutes <= 1440:
                return 60.0  # Tomorrow
            else:
                return max(20.0, 50.0 - (delta_minutes / 1440.0 * 5.0))
        except Exception:
            return 40.0

    def get_data_completeness(self, fixture: Dict[str, Any]) -> float:
        """Score whether logos, venues, referee, and scores are present."""
        points = 0
        if fixture.get("home_team_logo"):
            points += 20
        if fixture.get("away_team_logo"):
            points += 20
        if fixture.get("competition_logo"):
            points += 20
        if fixture.get("venue"):
            points += 15
        if fixture.get("competition_country"):
            points += 15
        if fixture.get("referee") or fixture.get("matchday"):
            points += 10
        return float(points)

    def compute_priority_score(self, fixture: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
        """Calculate the 0-100 composite priority score and return category."""
        comp_name = fixture.get("competition_name") or ""
        comp_country = fixture.get("competition_country") or ""
        home_team = fixture.get("home_team_name") or ""
        away_team = fixture.get("away_team_name") or ""
        utc_date = fixture.get("utc_date") or ""
        status = fixture.get("status") or "SCHEDULED"

        s_comp_imp = self.get_competition_importance(comp_name, comp_country)
        s_comp_pop = self.get_competition_popularity(comp_name)
        s_team_pop = (self.get_team_popularity(home_team) + self.get_team_popularity(away_team)) / 2.0
        s_relevance = self.get_match_relevance(fixture)
        s_proximity = self.get_kickoff_proximity(utc_date, status)
        s_data = self.get_data_completeness(fixture)

        w = self.weights
        total_score = (
            (s_comp_imp * w.competition_importance)
            + (s_comp_pop * w.competition_popularity)
            + (s_team_pop * w.team_popularity)
            + (s_relevance * w.match_relevance)
            + (s_proximity * w.kickoff_proximity)
            + (s_data * w.data_completeness)
        )

        final_score = round(max(0.0, min(100.0, total_score)), 1)

        # Categorization based on prompt specifications
        if final_score >= 80.0:
            category = "FEATURED"
        elif final_score >= 60.0:
            category = "HIGH"
        elif final_score >= 35.0:
            category = "NORMAL"
        else:
            category = "LOW"

        breakdown = {
            "competition_importance": round(s_comp_imp, 1),
            "competition_popularity": round(s_comp_pop, 1),
            "team_popularity": round(s_team_pop, 1),
            "match_relevance": round(s_relevance, 1),
            "kickoff_proximity": round(s_proximity, 1),
            "data_completeness": round(s_data, 1),
            "priority_score": final_score,
            "category": category,
        }

        return final_score, category, breakdown
