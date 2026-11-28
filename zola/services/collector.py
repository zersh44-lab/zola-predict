"""Data collection service orchestrating API clients, caching, and repository storage."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from zola.client.base import (
    BaseFootballClient,
    FootballApiError,
    StandardCompetition,
    StandardFixture,
    StandardTeam,
)
from zola.database.models import (
    ApiCollectionLog,
    Competition,
    Fixture,
    MatchResult,
    Team,
)
from zola.database.repository import FootballRepository
from zola.validation.fixture import is_synthetic_external_id, validate_fixture_identity

logger = logging.getLogger(__name__)


CALENDAR_FIXTURES: List[Dict[str, Any]] = []
CALENDAR_RESULTS: Dict[str, Dict[str, Any]] = {}


class FootballCollectorService:
    """Service coordinating API data fetching, deduplication, and database persistence."""

    def __init__(self, client: BaseFootballClient, repository: FootballRepository):
        self.client = client
        self.repository = repository

    def sync_competitions(self) -> List[Competition]:
        """Fetch competitions from API and persist them idempotently."""
        endpoint = "competitions"
        try:
            standard_comps, from_cache, elapsed_ms = self.client.get_competitions()
            # Log successful API operation
            self.repository.log_api_call(
                ApiCollectionLog(
                    provider=self.client.provider_name,
                    endpoint=endpoint,
                    status_code=200,
                    response_time_ms=elapsed_ms,
                    cache_hit=from_cache,
                )
            )
        except FootballApiError as e:
            self.repository.log_api_call(
                ApiCollectionLog(
                    provider=self.client.provider_name,
                    endpoint=endpoint,
                    status_code=500,
                    response_time_ms=0,
                    cache_hit=False,
                    error_message=str(e),
                )
            )
            raise

        saved_comps: List[Competition] = []
        for sc in standard_comps:
            comp = Competition(
                external_id=sc.external_id,
                provider=self.client.provider_name,
                name=sc.name,
                code=sc.code,
                country=sc.country,
                emblem_url=sc.emblem_url,
                current_season=sc.current_season,
            )
            comp_id = self.repository.upsert_competition(comp)
            comp.id = comp_id
            saved_comps.append(comp)

        return saved_comps

    def _persist_standard_fixture(
        self,
        sf: StandardFixture,
        local_competition_id: int,
    ) -> int:
        """Helper to upsert teams, fixture, and optional result safely."""
        provider = self.client.provider_name

        # Strict fixture verification
        if is_synthetic_external_id(sf.external_id):
            logger.warning("Rejecting fixture with synthetic external_id: %s", sf.external_id)
            return 0
        if not sf.home_team or not sf.home_team.name or not sf.away_team or not sf.away_team.name:
            logger.warning("Rejecting fixture with missing team info: %s", sf.external_id)
            return 0
        if sf.home_team.name.strip().lower() == sf.away_team.name.strip().lower():
            logger.warning("Rejecting fixture with identical teams: %s", sf.home_team.name)
            return 0

        fix_dict = {
            "external_id": sf.external_id,
            "provider": provider,
            "competition_id": local_competition_id,
            "season": sf.season or "2026",
            "utc_date": sf.utc_date,
            "status": sf.status,
            "home_team_name": sf.home_team.name,
            "away_team_name": sf.away_team.name,
        }
        is_val, val_rej = validate_fixture_identity(fix_dict)
        if not is_val:
            logger.warning("Rejecting fixture failing identity validation: %s (%s)", sf.external_id, val_rej)
            return 0

        # 1. Upsert home team
        home_id = self.repository.upsert_team(
            Team(
                external_id=sf.home_team.external_id,
                provider=provider,
                name=sf.home_team.name,
                short_name=sf.home_team.short_name,
                tla=sf.home_team.tla,
                crest_url=sf.home_team.crest_url,
                venue=sf.home_team.venue,
            )
        )

        # 2. Upsert away team
        away_id = self.repository.upsert_team(
            Team(
                external_id=sf.away_team.external_id,
                provider=provider,
                name=sf.away_team.name,
                short_name=sf.away_team.short_name,
                tla=sf.away_team.tla,
                crest_url=sf.away_team.crest_url,
                venue=sf.away_team.venue,
            )
        )

        # 3. Upsert fixture (idempotent, prevents duplicates)
        fixture_id = self.repository.upsert_fixture(
            Fixture(
                external_id=sf.external_id,
                provider=provider,
                competition_id=local_competition_id,
                utc_date=sf.utc_date,
                status=sf.status,
                home_team_id=home_id,
                away_team_id=away_id,
                season=sf.season,
                matchday=sf.matchday,
                venue=sf.venue,
                referee=sf.referee,
            )
        )

        # 4. Upsert result if match has scores or is finished
        if sf.result is not None:
            self.repository.upsert_match_result(
                MatchResult(
                    fixture_id=fixture_id,
                    winner=sf.result.winner,
                    duration=sf.result.duration or "REGULAR",
                    full_time_home_score=sf.result.full_time_home_score,
                    full_time_away_score=sf.result.full_time_away_score,
                    half_time_home_score=sf.result.half_time_home_score,
                    half_time_away_score=sf.result.half_time_away_score,
                    extra_time_home_score=sf.result.extra_time_home_score,
                    extra_time_away_score=sf.result.extra_time_away_score,
                    penalty_home_score=sf.result.penalty_home_score,
                    penalty_away_score=sf.result.penalty_away_score,
                )
            )

        return fixture_id

    def sync_fixtures(
        self,
        competition_code_or_id: Optional[str] = "39",
        status: Optional[str] = "SCHEDULED",
        season: Optional[str] = None,
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch fixtures for a competition or date and store them locally, handling duplicates safely."""
        provider = self.client.provider_name
        is_all = not competition_code_or_id or str(competition_code_or_id).lower() in ("all", "*", "none")
        endpoint = f"fixtures?date={date}" if date else (f"fixtures?all" if is_all else f"fixtures?league={competition_code_or_id}")

        # Determine target season if querying specific competition
        target_season = season
        comp_id_default = None
        if not is_all and competition_code_or_id:
            comp = self.repository.get_competition_by_code_or_external_id(provider, str(competition_code_or_id))
            if comp and comp.id:
                comp_id_default = comp.id
                if not target_season and comp.current_season:
                    target_season = comp.current_season

        try:
            standard_fixtures, from_cache, elapsed_ms = self.client.get_fixtures(
                competition_code_or_id=None if is_all else str(competition_code_or_id),
                status=status,
                season=target_season,
                date=date,
            )
            self.repository.log_api_call(
                ApiCollectionLog(
                    provider=provider,
                    endpoint=endpoint,
                    status_code=200,
                    response_time_ms=elapsed_ms,
                    cache_hit=from_cache,
                )
            )
        except FootballApiError as e:
            self.repository.log_api_call(
                ApiCollectionLog(
                    provider=provider,
                    endpoint=endpoint,
                    status_code=500,
                    response_time_ms=0,
                    cache_hit=False,
                    error_message=str(e),
                )
            )
            raise

        saved_count = 0
        comp_cache: Dict[str, int] = {}
        if comp_id_default and competition_code_or_id:
            comp_cache[str(competition_code_or_id)] = comp_id_default

        for sf in standard_fixtures:
            ext_cid = sf.competition_external_id or str(competition_code_or_id)
            if ext_cid not in comp_cache:
                c_record = self.repository.get_competition_by_code_or_external_id(provider, ext_cid)
                if c_record and c_record.id:
                    comp_cache[ext_cid] = c_record.id
                else:
                    new_cid = self.repository.upsert_competition(
                        Competition(
                            external_id=str(ext_cid),
                            provider=provider,
                            name=f"Competition {ext_cid}",
                            code=str(ext_cid),
                        )
                    )
                    comp_cache[ext_cid] = new_cid

            target_cid = comp_cache[ext_cid]
            self._persist_standard_fixture(sf, target_cid)
            saved_count += 1

        return {
            "competition": "all" if is_all else str(competition_code_or_id),
            "fixtures_synced": saved_count,
            "cache_hit": from_cache,
            "response_time_ms": elapsed_ms,
        }

    def _ensure_scheduled_calendar_fixtures(self, start_date: str, days_ahead: int = 10) -> int:
        """Disabled for data integrity. Calendar seeding is strictly prohibited in production."""
        logger.info("Calendar fixture seeding is permanently disabled to enforce data integrity.")
        return 0

    def sync_upcoming_window(
        self,
        competition_code_or_id: Optional[str] = "all",
        days_ahead: int = 7,
        football_data_client: Optional[BaseFootballClient] = None,
        secondary_client: Optional[BaseFootballClient] = None,
        start_date: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch and persist fixtures over a rolling multi-day window (today + 7 days).
        
        When status is None, fetches all fixture statuses so rescheduled and postponed
        matches are updated accurately in the database.
        """
        import os
        import time
        from datetime import datetime, timezone, timedelta
        from zola.client.football_data import FootballDataOrgClient

        now = datetime.now(timezone.utc)
        base_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc) if start_date else now
        today_str = base_dt.strftime("%Y-%m-%d")
        total_synced = 0
        dates_synced = []

        # 1. Primary Bulk Sync via football-data.org (GET /v4/matches)
        fb_client = football_data_client
        if not fb_client and isinstance(self.client, FootballDataOrgClient):
            fb_client = self.client

        if fb_client is not None:
            tier_one = ["PL", "PD", "SA", "BL1", "FL1", "DED", "PPL", "ELC", "BSA", "CL", "EC", "WC"]
            end_span = (base_dt + timedelta(days=min(days_ahead, 9))).strftime("%Y-%m-%d")
            try:
                logger.info("Initiating primary bulk sync via football-data.org (%s to %s)", today_str, end_span)
                standard_fixtures, from_cache, elapsed_ms = fb_client.get_matches_window(
                    date_from=today_str,
                    date_to=end_span,
                    competitions=tier_one,
                )
                comp_cache: Dict[str, int] = {}
                for sf in standard_fixtures:
                    ext_cid = sf.competition_external_id or "PL"
                    if ext_cid not in comp_cache:
                        c_rec = self.repository.get_competition_by_code_or_external_id(fb_client.provider_name, ext_cid)
                        if c_rec and c_rec.id:
                            comp_cache[ext_cid] = c_rec.id
                        else:
                            new_cid = self.repository.upsert_competition(
                                Competition(
                                    external_id=ext_cid,
                                    provider=fb_client.provider_name,
                                    name=f"Competition {ext_cid}",
                                    code=ext_cid,
                                )
                            )
                            comp_cache[ext_cid] = new_cid
                    self._persist_standard_fixture(sf, comp_cache[ext_cid])
                    total_synced += 1
                logger.info("football-data.org bulk sync: %d fixtures synced across %s to %s", len(standard_fixtures), today_str, end_span)
            except Exception as e:
                logger.warning("football-data.org bulk sync notice: %s", e)

        # 2. Secondary / Extra Competitions Sync via API-Football (GET /fixtures?date=...&timezone=Africa/Nairobi)
        api_client = secondary_client or (self.client if getattr(self.client, "provider_name", "") == "api-football" else None)
        if api_client is not None:
            for d in range(min(days_ahead + 1, 8)):
                cur_date = (base_dt + timedelta(days=d)).strftime("%Y-%m-%d")
                dates_synced.append(cur_date)
                try:
                    res = self.sync_fixtures(
                        competition_code_or_id=competition_code_or_id,
                        status=status,
                        date=cur_date,
                    )
                    count = res.get("fixtures_synced", 0)
                    total_synced += count
                    cached = res.get("cache_hit", False)
                    logger.info("API-Football date %s: %d fixtures synced (cache=%s)", cur_date, count, cached)
                    # If free-plan date limit reached via API-Sports client, stop probing future dates
                    if getattr(api_client, "_plan_limit_reached", False):
                        logger.info("API-Football Free tier date limit reached at date %s. Stopping rolling window scan.", cur_date)
                        break
                    # Respect API-Football 10 RPM free-tier limit with 6.5s pacing between live calls outside serverless/tests
                    if not cached and d < min(days_ahead, 7) and not os.environ.get("VERCEL") and not os.environ.get("PYTEST_CURRENT_TEST"):
                        time.sleep(6.5)
                except Exception as e:
                    logger.warning("API-Football sync date %s notice: %s", cur_date, e)
                    if "plan" in str(e).lower():
                        break
        else:
            for d in range(days_ahead + 1):
                cur_date = (base_dt + timedelta(days=d)).strftime("%Y-%m-%d")
                dates_synced.append(cur_date)
                try:
                    res = self.sync_fixtures(
                        competition_code_or_id=competition_code_or_id,
                        status=status,
                        date=cur_date,
                    )
                    total_synced += res.get("fixtures_synced", 0)
                except Exception as e:
                    logger.warning("Error syncing date %s: %s", cur_date, e)

        end_date_str = (base_dt + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        return {
            "competition": competition_code_or_id or "all",
            "start_date": today_str,
            "end_date": end_date_str,
            "days_ahead": days_ahead,
            "dates": dates_synced,
            "fixtures_synced": total_synced,
        }

    def sync_historical_results(
        self,
        competition_code_or_id: str,
        season: Optional[str] = None,
        limit: Optional[int] = 50,
    ) -> Dict[str, Any]:
        """Fetch completed historical matches and store scores in match_results."""
        provider = self.client.provider_name
        endpoint = f"competitions/{competition_code_or_id}/matches?status=FINISHED"

        comp = self.repository.get_competition_by_code_or_external_id(provider, competition_code_or_id)
        if not comp:
            self.sync_competitions()
            comp = self.repository.get_competition_by_code_or_external_id(provider, competition_code_or_id)

        if not comp or comp.id is None:
            comp_id = self.repository.upsert_competition(
                Competition(
                    external_id=str(competition_code_or_id),
                    provider=provider,
                    name=f"Competition {competition_code_or_id}",
                    code=str(competition_code_or_id),
                )
            )
        else:
            comp_id = comp.id

        try:
            standard_fixtures, from_cache, elapsed_ms = self.client.get_historical_results(
                competition_code_or_id=competition_code_or_id,
                season=season,
                limit=limit,
            )
            self.repository.log_api_call(
                ApiCollectionLog(
                    provider=provider,
                    endpoint=endpoint,
                    status_code=200,
                    response_time_ms=elapsed_ms,
                    cache_hit=from_cache,
                )
            )
        except FootballApiError as e:
            self.repository.log_api_call(
                ApiCollectionLog(
                    provider=provider,
                    endpoint=endpoint,
                    status_code=500,
                    response_time_ms=0,
                    cache_hit=False,
                    error_message=str(e),
                )
            )
            raise

        saved_count = 0
        for sf in standard_fixtures:
            self._persist_standard_fixture(sf, comp_id)
            saved_count += 1

        return {
            "competition": competition_code_or_id,
            "historical_results_synced": saved_count,
            "cache_hit": from_cache,
            "response_time_ms": elapsed_ms,
        }
