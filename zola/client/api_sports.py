"""Implementation of the API-Football (api-sports.io v3) client."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from zola.client.base import (
    BaseFootballClient,
    FootballApiError,
    StandardCompetition,
    StandardFixture,
    StandardMatchResult,
    StandardTeam,
)

logger = logging.getLogger(__name__)

# Map internal standardized statuses to API-Football short status codes
API_SPORTS_STATUS_MAP: Dict[str, str] = {
    "SCHEDULED": "NS-TBD",
    "TIMED": "NS",
    "FINISHED": "FT-AET-PEN",
    "IN_PLAY": "1H-HT-2H-ET-BT-P",
    "POSTPONED": "PST",
    "CANCELLED": "CANC-ABD",
}


class ApiSportsClient(BaseFootballClient):
    """Client for https://v3.football.api-sports.io."""

    @property
    def provider_name(self) -> str:
        return "api-football"

    @property
    def default_base_url(self) -> str:
        return "https://v3.football.api-sports.io"

    def get_auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "ZolaPredict/0.1.0",
        }
        if self._api_key:
            headers["x-apisports-key"] = self._api_key
        return headers

    def get_competitions(self) -> Tuple[List[StandardCompetition], bool, int]:
        data, from_cache, elapsed_ms = self._execute_http("GET", "leagues", ttl_minutes=1440)
        comps: List[StandardCompetition] = []
        for item in data.get("response", []):
            league = item.get("league") or {}
            country = item.get("country") or {}
            seasons = item.get("seasons") or []
            curr_season = str(seasons[-1].get("year")) if seasons else "2024"

            comps.append(
                StandardCompetition(
                    external_id=str(league.get("id")),
                    name=league.get("name", "Unknown League"),
                    code=str(league.get("id")),
                    country=country.get("name"),
                    emblem_url=league.get("logo"),
                    current_season=curr_season,
                )
            )
        return comps, from_cache, elapsed_ms

    def _parse_fixture(self, item: Dict[str, Any]) -> StandardFixture:
        fixture_raw = item.get("fixture") or {}
        league_raw = item.get("league") or {}
        teams_raw = item.get("teams") or {}
        goals_raw = item.get("goals") or {}
        score_raw = item.get("score") or {}

        home_raw = teams_raw.get("home") or {}
        away_raw = teams_raw.get("away") or {}

        home_team = StandardTeam(
            external_id=str(home_raw.get("id", "0")),
            name=home_raw.get("name", "Unknown Home Team"),
            crest_url=home_raw.get("logo"),
        )
        away_team = StandardTeam(
            external_id=str(away_raw.get("id", "0")),
            name=away_raw.get("name", "Unknown Away Team"),
            crest_url=away_raw.get("logo"),
        )

        status_short = str((fixture_raw.get("status") or {}).get("short", "NS")).upper()
        status_map = {
            "NS": "SCHEDULED",
            "TBD": "SCHEDULED",
            "1H": "IN_PLAY",
            "HT": "PAUSED",
            "2H": "IN_PLAY",
            "ET": "IN_PLAY",
            "BT": "PAUSED",
            "P": "IN_PLAY",
            "FT": "FINISHED",
            "AET": "FINISHED",
            "PEN": "FINISHED",
            "PST": "POSTPONED",
            "CANC": "CANCELLED",
            "ABD": "CANCELLED",
        }
        normalized_status = status_map.get(status_short, status_short)

        result: Optional[StandardMatchResult] = None
        has_goals = goals_raw.get("home") is not None and goals_raw.get("away") is not None

        if has_goals or normalized_status == "FINISHED":
            winner: Optional[str] = None
            if home_raw.get("winner") is True:
                winner = "HOME_TEAM"
            elif away_raw.get("winner") is True:
                winner = "AWAY_TEAM"
            elif home_raw.get("winner") is False and away_raw.get("winner") is False:
                winner = "DRAW"

            halftime = score_raw.get("halftime") or {}
            extratime = score_raw.get("extratime") or {}
            penalty = score_raw.get("penalty") or {}

            result = StandardMatchResult(
                winner=winner,
                duration="REGULAR" if not extratime.get("home") else "EXTRA_TIME",
                full_time_home_score=goals_raw.get("home"),
                full_time_away_score=goals_raw.get("away"),
                half_time_home_score=halftime.get("home"),
                half_time_away_score=halftime.get("away"),
                extra_time_home_score=extratime.get("home"),
                extra_time_away_score=extratime.get("away"),
                penalty_home_score=penalty.get("home"),
                penalty_away_score=penalty.get("away"),
            )

        venue_raw = fixture_raw.get("venue") or {}
        venue_name = venue_raw.get("name")

        return StandardFixture(
            external_id=str(fixture_raw.get("id")),
            competition_external_id=str(league_raw.get("id")),
            utc_date=fixture_raw.get("date", ""),
            status=normalized_status,
            home_team=home_team,
            away_team=away_team,
            season=str(league_raw.get("season")) if league_raw.get("season") else None,
            matchday=league_raw.get("round"),
            venue=venue_name,
            referee=fixture_raw.get("referee"),
            result=result,
        )

    def get_fixtures(
        self,
        competition_code_or_id: Optional[str] = "39",
        status: Optional[str] = "SCHEDULED",
        season: Optional[str] = None,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch fixtures for competition or date. Translates statuses and handles Free tier constraints."""
        params: Dict[str, Any] = {}
        target_status = status.upper().strip() if status else None
        api_status = None
        if target_status:
            api_status = API_SPORTS_STATUS_MAP.get(target_status, target_status)
            params["status"] = api_status

        is_all_comps = not competition_code_or_id or str(competition_code_or_id).lower() in ("all", "*", "none")
        target_comp = None if is_all_comps else str(competition_code_or_id).strip()

        # Path A: Explicit date query (e.g. today or tomorrow)
        if date:
            params["date"] = date
            params["timezone"] = "Africa/Nairobi"
            if date_from and date_to and season:
                params["from"] = date_from
                params["to"] = date_to

            # If a specific league and season are supplied, include them
            if target_comp and season:
                params["league"] = target_comp
                params["season"] = str(season)

            try:
                data, from_cache, elapsed_ms = self._execute_http("GET", "fixtures", params=params, ttl_minutes=30)
                fixtures: List[StandardFixture] = []
                for item in data.get("response", []):
                    parsed = self._parse_fixture(item)
                    if target_comp and str(parsed.competition_external_id) != str(target_comp):
                        continue
                    if target_status and parsed.status != target_status:
                        continue
                    fixtures.append(parsed)
                logger.info(
                    "API-Football date %s HTTP 200: %d fixtures fetched (cache=%s, elapsed=%dms)",
                    date,
                    len(fixtures),
                    from_cache,
                    elapsed_ms,
                )
                return fixtures, from_cache, elapsed_ms
            except FootballApiError as e:
                logger.error("API-Football error querying fixtures for date %s: %s", date, e)
                raise

        # Path B: Competition query by league & season
        if target_comp:
            params["league"] = target_comp
            target_season = str(season) if season else "2026"
            params["season"] = target_season

            if date_from and date_to:
                params["from"] = date_from
                params["to"] = date_to

            try:
                data, from_cache, elapsed_ms = self._execute_http("GET", "fixtures", params=params, ttl_minutes=30)
                fixtures = []
                for item in data.get("response", []):
                    parsed = self._parse_fixture(item)
                    if target_status and parsed.status != target_status:
                        continue
                    fixtures.append(parsed)
                return fixtures, from_cache, elapsed_ms
            except FootballApiError as e:
                err_text = str(e).lower()
                # On Free tier, current season queries for major leagues return a plan restriction
                if "free plans do not have access to this season" in err_text or "plan" in err_text:
                    logger.warning(
                        "API-Football plan limitation on league %s season %s. Querying live fixtures for current dates.",
                        target_comp,
                        target_season,
                    )
                    # Query live fixtures within the allowed 3-day rolling window
                    now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    tomorrow_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
                    all_live: List[StandardFixture] = []
                    total_elapsed = 0
                    cached_all = True
                    for d in [now_date, tomorrow_date]:
                        d_params: Dict[str, Any] = {"date": d}
                        if api_status:
                            d_params["status"] = api_status
                        d_data, d_cache, d_elapsed = self._execute_http("GET", "fixtures", params=d_params, ttl_minutes=30)
                        total_elapsed += d_elapsed
                        if not d_cache:
                            cached_all = False
                        for item in d_data.get("response", []):
                            parsed = self._parse_fixture(item)
                            if str(parsed.competition_external_id) != str(target_comp):
                                continue
                            if target_status and parsed.status != target_status:
                                continue
                            all_live.append(parsed)
                    return all_live, cached_all, total_elapsed
                raise
        else:
            # Path C: Query all competitions for today
            now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            params["date"] = now_date
            data, from_cache, elapsed_ms = self._execute_http("GET", "fixtures", params=params, ttl_minutes=30)
            fixtures = []
            for item in data.get("response", []):
                parsed = self._parse_fixture(item)
                if target_status and parsed.status != target_status:
                    continue
                fixtures.append(parsed)
            return fixtures, from_cache, elapsed_ms

    def get_historical_results(
        self,
        competition_code_or_id: str,
        season: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[StandardFixture], bool, int]:
        """Fetch completed historical results for competition."""
        target_season = season or "2024"
        params: Dict[str, Any] = {
            "league": competition_code_or_id,
            "season": target_season,
            "status": "FT-AET-PEN",
        }

        data, from_cache, elapsed_ms = self._execute_http("GET", "fixtures", params=params, ttl_minutes=1440)
        fixtures: List[StandardFixture] = []
        for item in data.get("response", []):
            parsed = self._parse_fixture(item)
            if parsed.status == "FINISHED":
                fixtures.append(parsed)

        if limit and limit > 0:
            fixtures = fixtures[-limit:]

        return fixtures, from_cache, elapsed_ms
