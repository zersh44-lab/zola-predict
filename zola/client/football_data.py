"""Implementation of the football-data.org (v4) client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from zola.client.base import (
    BaseFootballClient,
    StandardCompetition,
    StandardFixture,
    StandardMatchResult,
    StandardTeam,
)


class FootballDataOrgClient(BaseFootballClient):
    """Client for https://api.football-data.org/v4."""

    @property
    def provider_name(self) -> str:
        return "football-data.org"

    @property
    def default_base_url(self) -> str:
        return "https://api.football-data.org/v4"

    def get_auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "ZolaPredict/0.1.0",
        }
        if self._api_key:
            headers["X-Auth-Token"] = self._api_key
        return headers

    def get_competitions(self) -> Tuple[List[StandardCompetition], bool, int]:
        """Fetch list of competitions from football-data.org."""
        # Cache competitions for 24 hours (1440 minutes) since leagues rarely change
        data, from_cache, elapsed_ms = self._execute_http("GET", "competitions", ttl_minutes=1440)
        comps: List[StandardCompetition] = []
        for raw in data.get("competitions", []):
            area = raw.get("area") or {}
            curr_season = raw.get("currentSeason") or {}
            season_str = str(curr_season.get("startDate", ""))[:4] if curr_season else None

            comps.append(
                StandardCompetition(
                    external_id=str(raw.get("id")),
                    name=raw.get("name", "Unknown Competition"),
                    code=raw.get("code"),
                    country=area.get("name"),
                    emblem_url=raw.get("emblem"),
                    current_season=season_str,
                )
            )
        return comps, from_cache, elapsed_ms

    def _parse_match(self, raw_match: Dict[str, Any], default_comp_id: str) -> StandardFixture:
        """Parse raw match object into a StandardFixture with optional result."""
        raw_comp = raw_match.get("competition") or {}
        comp_id = str(raw_comp.get("id") or default_comp_id)

        home_raw = raw_match.get("homeTeam") or {}
        away_raw = raw_match.get("awayTeam") or {}

        home_team = StandardTeam(
            external_id=str(home_raw.get("id", "0")),
            name=home_raw.get("name", "Unknown Home Team"),
            short_name=home_raw.get("shortName"),
            tla=home_raw.get("tla"),
            crest_url=home_raw.get("crest"),
        )
        away_team = StandardTeam(
            external_id=str(away_raw.get("id", "0")),
            name=away_raw.get("name", "Unknown Away Team"),
            short_name=away_raw.get("shortName"),
            tla=away_raw.get("tla"),
            crest_url=away_raw.get("crest"),
        )

        # Parse match score if available
        result: Optional[StandardMatchResult] = None
        raw_score = raw_match.get("score") or {}
        full_time = raw_score.get("fullTime") or {}
        half_time = raw_score.get("halfTime") or {}
        extra_time = raw_score.get("extraTime") or {}
        penalties = raw_score.get("penalties") or {}

        has_scores = full_time.get("home") is not None and full_time.get("away") is not None
        status = str(raw_match.get("status", "SCHEDULED")).upper()

        if has_scores or status == "FINISHED":
            result = StandardMatchResult(
                winner=raw_score.get("winner"),
                duration=raw_score.get("duration", "REGULAR"),
                full_time_home_score=full_time.get("home"),
                full_time_away_score=full_time.get("away"),
                half_time_home_score=half_time.get("home"),
                half_time_away_score=half_time.get("away"),
                extra_time_home_score=extra_time.get("home"),
                extra_time_away_score=extra_time.get("away"),
                penalty_home_score=penalties.get("home"),
                penalty_away_score=penalties.get("away"),
            )

        season_raw = raw_match.get("season") or {}
        season_str = str(season_raw.get("startDate", ""))[:4] if season_raw else None

        referees = raw_match.get("referees") or []
        referee_name = referees[0].get("name") if referees else None

        return StandardFixture(
            external_id=str(raw_match.get("id")),
            competition_external_id=comp_id,
            utc_date=raw_match.get("utcDate", ""),
            status=status,
            home_team=home_team,
            away_team=away_team,
            season=season_str,
            matchday=raw_match.get("matchday"),
            venue=raw_match.get("venue"),
            referee=referee_name,
            result=result,
        )

    def get_fixtures(
        self,
        competition_code_or_id: Optional[str] = None,
        status: Optional[str] = "SCHEDULED",
        season: Optional[str] = None,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch scheduled fixtures for a given competition code or ID."""
        params: Dict[str, Any] = {}
        if status:
            params["status"] = status
        if season:
            params["season"] = season
        if date:
            params["date"] = date
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to

        endpoint = f"competitions/{competition_code_or_id}/matches" if competition_code_or_id else "matches"
        data, from_cache, elapsed_ms = self._execute_http("GET", endpoint, params=params, ttl_minutes=30)

        fixtures: List[StandardFixture] = []
        for match in data.get("matches", []):
            fixtures.append(self._parse_match(match, str(competition_code_or_id)))
        return fixtures, from_cache, elapsed_ms

    def get_matches_window(
        self,
        date_from: str,
        date_to: str,
        competitions: Optional[List[str]] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch matches across a rolling date window (max span 10 days, date delta <= 9 days)."""
        from datetime import datetime, timedelta

        try:
            d_from = datetime.strptime(date_from, "%Y-%m-%d")
            d_to = datetime.strptime(date_to, "%Y-%m-%d")
            if (d_to - d_from).days > 9:
                d_to = d_from + timedelta(days=9)
                date_to = d_to.strftime("%Y-%m-%d")
        except ValueError:
            pass

        params: Dict[str, Any] = {
            "dateFrom": date_from,
            "dateTo": date_to,
        }
        if competitions:
            params["competitions"] = ",".join(competitions)

        data, from_cache, elapsed_ms = self._execute_http("GET", "matches", params=params, ttl_minutes=30)
        fixtures: List[StandardFixture] = []
        for match in data.get("matches", []):
            comp_code = (match.get("competition") or {}).get("code", "PL")
            fixtures.append(self._parse_match(match, default_comp_id=str(comp_code)))
        return fixtures, from_cache, elapsed_ms

    def get_historical_results(
        self,
        competition_code_or_id: str,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch completed/finished match results for a competition."""
        params: Dict[str, Any] = {"status": "FINISHED"}
        if season:
            params["season"] = season

        # Finished historical matches cached for 24 hours (1440 minutes) since completed scores do not change
        endpoint = f"competitions/{competition_code_or_id}/matches"
        data, from_cache, elapsed_ms = self._execute_http("GET", endpoint, params=params, ttl_minutes=1440)

        matches = data.get("matches", [])
        if limit and limit > 0:
            matches = matches[-limit:]  # get most recent finished matches

        fixtures: List[StandardFixture] = []
        for match in matches:
            fixtures.append(self._parse_match(match, str(competition_code_or_id)))
        return fixtures, from_cache, elapsed_ms
