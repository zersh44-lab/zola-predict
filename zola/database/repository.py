"""Database repository layer for safe CRUD and idempotent upsert operations."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from zola.database.models import (
    AccuracyMetric,
    ApiCollectionLog,
    Competition,
    Fixture,
    HistoricalDataImport,
    MatchResult,
    Prediction,
    ResearchSource,
    Team,
)


class FootballRepository:
    """Repository handling all database interactions for football entities."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---------------------------------------------------------
    # Competitions
    # ---------------------------------------------------------
    def upsert_competition(self, competition: Competition) -> int:
        """Insert or update a competition and return its local ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO competitions (external_id, provider, name, code, country, emblem_url, current_season)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                name = excluded.name,
                code = COALESCE(excluded.code, competitions.code),
                country = COALESCE(excluded.country, competitions.country),
                emblem_url = COALESCE(excluded.emblem_url, competitions.emblem_url),
                current_season = COALESCE(excluded.current_season, competitions.current_season),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (
                str(competition.external_id),
                competition.provider,
                competition.name,
                competition.code,
                competition.country,
                competition.emblem_url,
                competition.current_season,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_competition_by_code_or_external_id(self, provider: str, code_or_id: str) -> Optional[Competition]:
        """Find competition by external_id or code."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM competitions
            WHERE (provider = ? OR ? IS NULL) AND (external_id = ? OR code = ?)
            LIMIT 1;
            """,
            (provider, provider, str(code_or_id), str(code_or_id)),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Competition(
            id=row["id"],
            external_id=row["external_id"],
            provider=row["provider"],
            name=row["name"],
            code=row["code"],
            country=row["country"],
            emblem_url=row["emblem_url"],
            current_season=row["current_season"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_all_competitions(self) -> List[Competition]:
        """List all stored competitions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM competitions ORDER BY name ASC;")
        rows = cursor.fetchall()
        return [
            Competition(
                id=r["id"],
                external_id=r["external_id"],
                provider=r["provider"],
                name=r["name"],
                code=r["code"],
                country=r["country"],
                emblem_url=r["emblem_url"],
                current_season=r["current_season"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------
    # Teams
    # ---------------------------------------------------------
    def upsert_team(self, team: Team) -> int:
        """Insert or update a football team and return its local ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO teams (external_id, provider, name, short_name, tla, crest_url, venue)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                name = excluded.name,
                short_name = COALESCE(excluded.short_name, teams.short_name),
                tla = COALESCE(excluded.tla, teams.tla),
                crest_url = COALESCE(excluded.crest_url, teams.crest_url),
                venue = COALESCE(excluded.venue, teams.venue),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (
                str(team.external_id),
                team.provider,
                team.name,
                team.short_name,
                team.tla,
                team.crest_url,
                team.venue,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_team_by_external_id(self, provider: str, external_id: str) -> Optional[Team]:
        """Lookup team by external id and provider."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM teams WHERE provider = ? AND external_id = ? LIMIT 1;",
            (provider, str(external_id)),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Team(
            id=row["id"],
            external_id=row["external_id"],
            provider=row["provider"],
            name=row["name"],
            short_name=row["short_name"],
            tla=row["tla"],
            crest_url=row["crest_url"],
            venue=row["venue"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_team_by_name(self, name: str) -> Optional[Team]:
        """Lookup team by name or short name."""
        cursor = self.conn.cursor()
        clean_name = name.strip()
        cursor.execute(
            "SELECT * FROM teams WHERE LOWER(name) = LOWER(?) OR LOWER(short_name) = LOWER(?) LIMIT 1;",
            (clean_name, clean_name),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Team(
            id=row["id"],
            external_id=row["external_id"],
            provider=row["provider"],
            name=row["name"],
            short_name=row["short_name"],
            tla=row["tla"],
            crest_url=row["crest_url"],
            venue=row["venue"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ---------------------------------------------------------
    # Fixtures & Duplicate Handling
    # ---------------------------------------------------------
    def upsert_fixture(self, fixture: Fixture) -> int:
        """Insert or safely update fixture, preventing duplicates. Returns fixture ID."""
        cursor = self.conn.cursor()

        # Check for existing fixture to detect identity conflicts
        cursor.execute(
            "SELECT id, home_team_id, away_team_id, competition_id, status FROM fixtures WHERE provider = ? AND external_id = ? LIMIT 1;",
            (fixture.provider, str(fixture.external_id)),
        )
        existing = cursor.fetchone()
        if existing:
            # Preserve quarantined records unconditionally
            if existing["status"] == "INVALID_FIXTURE_IDENTITY":
                logger.warning(
                    "Skipping update for quarantined fixture %s:%s (status=INVALID_FIXTURE_IDENTITY).",
                    fixture.provider,
                    fixture.external_id,
                )
                return int(existing["id"])

            # If incoming provider fixture has conflicting team IDs, quarantine existing and fail closed
            if (
                existing["home_team_id"] != fixture.home_team_id
                or existing["away_team_id"] != fixture.away_team_id
            ):
                logger.error(
                    "Conflict detected for provider fixture %s:%s. Existing teams (%s, %s) != Incoming teams (%s, %s). Quarantining record.",
                    fixture.provider,
                    fixture.external_id,
                    existing["home_team_id"],
                    existing["away_team_id"],
                    fixture.home_team_id,
                    fixture.away_team_id,
                )
                cursor.execute(
                    """
                    UPDATE fixtures
                    SET status = 'INVALID_FIXTURE_IDENTITY',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?;
                    """,
                    (existing["id"],),
                )
                self.conn.commit()
                return int(existing["id"])

        cursor.execute(
            """
            INSERT INTO fixtures (
                external_id, provider, competition_id, season, matchday,
                utc_date, status, home_team_id, away_team_id, venue, referee
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider, external_id) DO UPDATE SET
                competition_id = excluded.competition_id,
                season = COALESCE(excluded.season, fixtures.season),
                matchday = COALESCE(excluded.matchday, fixtures.matchday),
                utc_date = excluded.utc_date,
                status = excluded.status,
                home_team_id = excluded.home_team_id,
                away_team_id = excluded.away_team_id,
                venue = COALESCE(excluded.venue, fixtures.venue),
                referee = COALESCE(excluded.referee, fixtures.referee),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (
                str(fixture.external_id),
                fixture.provider,
                fixture.competition_id,
                fixture.season,
                fixture.matchday,
                fixture.utc_date,
                fixture.status,
                fixture.home_team_id,
                fixture.away_team_id,
                fixture.venue,
                fixture.referee,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_fixture_by_id(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Get fixture by internal ID with joined team and competition details."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 
                f.id, f.external_id, f.provider, f.utc_date, f.status, f.matchday, f.season, f.venue, f.referee,
                f.competition_id, f.home_team_id, f.away_team_id,
                c.name as competition_name, c.code as competition_code, c.country as competition_country, c.emblem_url as competition_emblem,
                ht.name as home_team_name, ht.short_name as home_team_short, ht.crest_url as home_team_crest,
                at.name as away_team_name, at.short_name as away_team_short, at.crest_url as away_team_crest,
                mr.winner, mr.full_time_home_score, mr.full_time_away_score,
                mr.half_time_home_score, mr.half_time_away_score
            FROM fixtures f
            JOIN competitions c ON f.competition_id = c.id
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            LEFT JOIN match_results mr ON f.id = mr.fixture_id
            WHERE f.id = ?
            LIMIT 1;
            """,
            (fixture_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row) if hasattr(row, "keys") else dict(zip([d[0] for d in cursor.description], row))
        from zola.utils.flags import get_country_flag_url
        res["home_team_flag"] = get_country_flag_url(res.get("home_team_name"))
        res["away_team_flag"] = get_country_flag_url(res.get("away_team_name"))
        res["competition_flag"] = get_country_flag_url(res.get("competition_country"))
        if not res.get("home_team_crest"):
            res["home_team_crest"] = res["home_team_flag"]
        if not res.get("away_team_crest"):
            res["away_team_crest"] = res["away_team_flag"]
        return res

    def get_fixture_by_external_id(self, provider: str, external_id: str) -> Optional[Fixture]:
        """Lookup fixture by external ID and provider."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM fixtures WHERE provider = ? AND external_id = ? LIMIT 1;",
            (provider, str(external_id)),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return Fixture(
            id=row["id"],
            external_id=row["external_id"],
            provider=row["provider"],
            competition_id=row["competition_id"],
            season=row["season"],
            matchday=row["matchday"],
            utc_date=row["utc_date"],
            status=row["status"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            venue=row["venue"],
            referee=row["referee"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_fixtures(
        self,
        competition_id: Optional[int] = None,
        status: Optional[str] = None,
        upcoming_only: bool = False,
        date: Optional[str] = None,
        gender: Optional[str] = None,
        limit: int = 100,
        priority_competition_ids: Optional[List[int]] = None,
        days_ahead: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query fixtures joined with competition and team names."""
        from datetime import timedelta
        cursor = self.conn.cursor()
        query = """
            SELECT 
                f.id, f.external_id, f.provider, f.utc_date, f.status, f.matchday, f.season, f.venue,
                f.competition_id, f.home_team_id, f.away_team_id,
                c.name as competition_name, c.code as competition_code, c.country as competition_country, c.emblem_url as competition_emblem,
                ht.name as home_team_name, ht.short_name as home_team_short, ht.crest_url as home_team_crest,
                at.name as away_team_name, at.short_name as away_team_short, at.crest_url as away_team_crest,
                mr.winner, mr.full_time_home_score, mr.full_time_away_score
            FROM fixtures f
            JOIN competitions c ON f.competition_id = c.id
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            LEFT JOIN match_results mr ON f.id = mr.fixture_id
            WHERE 1=1 AND f.status NOT LIKE 'INVALID%'
        """
        params: List[Any] = []
        if competition_id is not None:
            query += " AND f.competition_id = ?"
            params.append(competition_id)
        if priority_competition_ids is not None:
            unique_ids = sorted({int(value) for value in priority_competition_ids})
            if unique_ids:
                placeholders = ",".join("?" for _ in unique_ids)
                query += f" AND f.competition_id IN ({placeholders})"
                params.extend(unique_ids)
            else:
                query += " AND 1 = 0"
        if status is not None:
            query += " AND f.status = ?"
            params.append(status)
        if date:
            date_val = datetime.now(timezone.utc).strftime("%Y-%m-%d") if date.lower() == "today" else date
            query += " AND DATE(f.utc_date) = ?"
            params.append(date_val)
        if upcoming_only:
            now_iso = datetime.now(timezone.utc).isoformat()
            query += " AND f.status IN ('SCHEDULED', 'TIMED', 'TBD', 'NOT_STARTED') AND f.utc_date >= ?"
            params.append(now_iso)
            if days_ahead is not None:
                max_iso = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()
                query += " AND f.utc_date <= ?"
                params.append(max_iso)

        if gender and gender.lower() in ("men", "women"):
            women_markers = (
                "women", "woman", "female", "femenin", "femenil", "feminin", "feminine", "feminina", "feminino", "femminil", "femminile",
                "frauen", "ladies", "lady", "dames", "damen", "nwsl",
                "wsl", "damallsvenskan", "liga f",
            )
            women_clause = "(" + " OR ".join("LOWER(c.name) LIKE ?" for _ in women_markers) + ")"
            if gender.lower() == "women":
                query += f" AND {women_clause}"
            else:
                query += f" AND NOT {women_clause}"
            params.extend(f"%{marker}%" for marker in women_markers)

        if status == "FINISHED":
            query += " ORDER BY f.utc_date DESC LIMIT ?"
        else:
            query += " ORDER BY f.utc_date ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        from zola.utils.flags import get_country_flag_url
        res_list = []
        col_names = [col[0] for col in cursor.description] if cursor.description else []
        for r in rows:
            d = dict(r) if hasattr(r, "keys") else dict(zip(col_names, r))
            h_flag = get_country_flag_url(d.get("home_team_name"))
            a_flag = get_country_flag_url(d.get("away_team_name"))
            c_flag = get_country_flag_url(d.get("competition_country"))
            d["home_team_flag"] = h_flag
            d["away_team_flag"] = a_flag
            d["competition_flag"] = c_flag
            if not d.get("home_team_crest"):
                d["home_team_crest"] = h_flag
            if not d.get("away_team_crest"):
                d["away_team_crest"] = a_flag
            if not d.get("competition_emblem"):
                d["competition_emblem"] = c_flag
            res_list.append(d)
        return res_list

    # ---------------------------------------------------------
    # Match History Queries for Prediction Engine
    # ---------------------------------------------------------
    def get_team_matches(self, team_id: int, limit: int = 30) -> List[Dict[str, Any]]:
        """Fetch finished match history for a given team (home or away). Unifies across duplicate IDs by team name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM teams WHERE id = ? LIMIT 1;", (team_id,))
        row = cursor.fetchone()
        team_name = row["name"].strip() if row else None

        if team_name:
            cursor.execute(
                """
                SELECT 
                    f.id, f.utc_date, f.home_team_id, f.away_team_id, f.competition_id,
                    mr.full_time_home_score, mr.full_time_away_score, mr.winner,
                    ht.name as home_team_name, at.name as away_team_name,
                    CASE WHEN (f.home_team_id = ? OR LOWER(ht.name) = LOWER(?)) THEN 1 ELSE 0 END as is_home
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                WHERE (f.home_team_id = ? OR f.away_team_id = ? OR LOWER(ht.name) = LOWER(?) OR LOWER(at.name) = LOWER(?))
                  AND f.status = 'FINISHED'
                  AND mr.full_time_home_score IS NOT NULL
                ORDER BY f.utc_date DESC
                LIMIT ?;
                """,
                (team_id, team_name, team_id, team_id, team_name, team_name, limit),
            )
        else:
            cursor.execute(
                """
                SELECT 
                    f.id, f.utc_date, f.home_team_id, f.away_team_id, f.competition_id,
                    mr.full_time_home_score, mr.full_time_away_score, mr.winner,
                    ht.name as home_team_name, at.name as away_team_name,
                    CASE WHEN f.home_team_id = ? THEN 1 ELSE 0 END as is_home
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                WHERE (f.home_team_id = ? OR f.away_team_id = ?)
                  AND f.status = 'FINISHED'
                  AND mr.full_time_home_score IS NOT NULL
                ORDER BY f.utc_date DESC
                LIMIT ?;
                """,
                (team_id, team_id, team_id, limit),
            )
        return [dict(r) for r in cursor.fetchall()]

    def get_head_to_head_matches(self, team1_id: int, team2_id: int, limit: int = 15) -> List[Dict[str, Any]]:
        """Fetch historical head-to-head completed matches between two teams, matching across duplicate records by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, name FROM teams WHERE id IN (?, ?);", (team1_id, team2_id))
        team_rows = {r["id"]: r["name"].strip() for r in cursor.fetchall()}
        name1 = team_rows.get(team1_id)
        name2 = team_rows.get(team2_id)

        if name1 and name2:
            cursor.execute(
                """
                SELECT 
                    f.id, f.utc_date, f.home_team_id, f.away_team_id,
                    mr.full_time_home_score, mr.full_time_away_score, mr.winner,
                    ht.name as home_team_name, at.name as away_team_name
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                WHERE (
                    (f.home_team_id = ? AND f.away_team_id = ?) OR (f.home_team_id = ? AND f.away_team_id = ?)
                    OR (LOWER(ht.name) = LOWER(?) AND LOWER(at.name) = LOWER(?))
                    OR (LOWER(ht.name) = LOWER(?) AND LOWER(at.name) = LOWER(?))
                )
                  AND f.status = 'FINISHED'
                  AND mr.full_time_home_score IS NOT NULL
                ORDER BY f.utc_date DESC
                LIMIT ?;
                """,
                (team1_id, team2_id, team2_id, team1_id, name1, name2, name2, name1, limit),
            )
        else:
            cursor.execute(
                """
                SELECT 
                    f.id, f.utc_date, f.home_team_id, f.away_team_id,
                    mr.full_time_home_score, mr.full_time_away_score, mr.winner,
                    ht.name as home_team_name, at.name as away_team_name
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                WHERE ((f.home_team_id = ? AND f.away_team_id = ?) OR (f.home_team_id = ? AND f.away_team_id = ?))
                  AND f.status = 'FINISHED'
                  AND mr.full_time_home_score IS NOT NULL
                ORDER BY f.utc_date DESC
                LIMIT ?;
                """,
                (team1_id, team2_id, team2_id, team1_id, limit),
            )
        return [dict(r) for r in cursor.fetchall()]

    def get_competition_matches(self, competition_id: int, limit: int = 500) -> List[Dict[str, Any]]:
        """Fetch all completed matches in a competition for league averages."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, country FROM competitions WHERE id = ? LIMIT 1;", (competition_id,))
        comp_row = cursor.fetchone()
        comp_name = comp_row["name"].strip() if comp_row else None
        comp_country = comp_row["country"].strip() if comp_row and comp_row["country"] else None

        if comp_name:
            cursor.execute(
                """
                SELECT 
                    f.id, f.utc_date, f.home_team_id, f.away_team_id,
                    mr.full_time_home_score, mr.full_time_away_score, mr.winner
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                JOIN competitions c ON f.competition_id = c.id
                WHERE (f.competition_id = ? OR (LOWER(c.name) = LOWER(?) AND (c.country = ? OR ? IS NULL)))
                  AND f.status = 'FINISHED'
                  AND mr.full_time_home_score IS NOT NULL
                ORDER BY f.utc_date DESC
                LIMIT ?;
                """,
                (competition_id, comp_name, comp_country, comp_country, limit),
            )
        else:
            cursor.execute(
                """
                SELECT 
                    f.id, f.utc_date, f.home_team_id, f.away_team_id,
                    mr.full_time_home_score, mr.full_time_away_score, mr.winner
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                WHERE f.competition_id = ?
                  AND f.status = 'FINISHED'
                  AND mr.full_time_home_score IS NOT NULL
                ORDER BY f.utc_date DESC
                LIMIT ?;
                """,
                (competition_id, limit),
            )
        return [dict(r) for r in cursor.fetchall()]

    # ---------------------------------------------------------
    # Match Results
    # ---------------------------------------------------------
    def upsert_match_result(self, result: MatchResult) -> int:
        """Insert or update match result score. Returns match_result ID."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO match_results (
                fixture_id, winner, duration,
                full_time_home_score, full_time_away_score,
                half_time_home_score, half_time_away_score,
                extra_time_home_score, extra_time_away_score,
                penalty_home_score, penalty_away_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id) DO UPDATE SET
                winner = excluded.winner,
                duration = COALESCE(excluded.duration, match_results.duration),
                full_time_home_score = excluded.full_time_home_score,
                full_time_away_score = excluded.full_time_away_score,
                half_time_home_score = excluded.half_time_home_score,
                half_time_away_score = excluded.half_time_away_score,
                extra_time_home_score = excluded.extra_time_home_score,
                extra_time_away_score = excluded.extra_time_away_score,
                penalty_home_score = excluded.penalty_home_score,
                penalty_away_score = excluded.penalty_away_score,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (
                result.fixture_id,
                result.winner,
                result.duration,
                result.full_time_home_score,
                result.full_time_away_score,
                result.half_time_home_score,
                result.half_time_away_score,
                result.extra_time_home_score,
                result.extra_time_away_score,
                result.penalty_home_score,
                result.penalty_away_score,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_match_result_by_fixture_id(self, fixture_id: int) -> Optional[MatchResult]:
        """Retrieve match result for a specific fixture."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM match_results WHERE fixture_id = ? LIMIT 1;", (fixture_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return MatchResult(
            id=row["id"],
            fixture_id=row["fixture_id"],
            winner=row["winner"],
            duration=row["duration"],
            full_time_home_score=row["full_time_home_score"],
            full_time_away_score=row["full_time_away_score"],
            half_time_home_score=row["half_time_home_score"],
            half_time_away_score=row["half_time_away_score"],
            extra_time_home_score=row["extra_time_home_score"],
            extra_time_away_score=row["extra_time_away_score"],
            penalty_home_score=row["penalty_home_score"],
            penalty_away_score=row["penalty_away_score"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ---------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------
    def upsert_prediction(self, prediction: Prediction) -> int:
        """Insert or update a match prediction with immutability guarantees for settled forecasts."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                fixture_id, model_name, market, predicted_outcome, probability,
                probabilities_json, confidence, confidence_score,
                data_quality_notes, features_json, status, is_correct,
                settlement_status, model_version, odds_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fixture_id, model_name, market) DO UPDATE SET
                predicted_outcome = CASE WHEN predictions.status = 'PENDING' THEN excluded.predicted_outcome ELSE predictions.predicted_outcome END,
                probability = CASE WHEN predictions.status = 'PENDING' THEN excluded.probability ELSE predictions.probability END,
                probabilities_json = CASE WHEN predictions.status = 'PENDING' THEN excluded.probabilities_json ELSE predictions.probabilities_json END,
                confidence = CASE WHEN predictions.status = 'PENDING' THEN excluded.confidence ELSE predictions.confidence END,
                confidence_score = CASE WHEN predictions.status = 'PENDING' THEN excluded.confidence_score ELSE predictions.confidence_score END,
                data_quality_notes = CASE WHEN predictions.status = 'PENDING' THEN excluded.data_quality_notes ELSE predictions.data_quality_notes END,
                features_json = excluded.features_json,
                settlement_status = COALESCE(predictions.settlement_status, excluded.settlement_status),
                model_version = excluded.model_version,
                odds_json = COALESCE(excluded.odds_json, predictions.odds_json),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (
                prediction.fixture_id,
                prediction.model_name,
                prediction.market,
                prediction.predicted_outcome,
                prediction.probability,
                prediction.probabilities_json,
                prediction.confidence,
                prediction.confidence_score,
                prediction.data_quality_notes,
                prediction.features_json,
                prediction.status,
                prediction.is_correct,
                prediction.settlement_status or "PENDING",
                prediction.model_version or "v1.1.0",
                prediction.odds_json,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_predictions_for_fixture(self, fixture_id: int, model_name: Optional[str] = None) -> List[Prediction]:
        """Fetch predictions for a fixture."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM predictions WHERE fixture_id = ?"
        params: List[Any] = [fixture_id]
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        query += " ORDER BY model_name, market ASC;"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [
            Prediction(
                id=r["id"],
                fixture_id=r["fixture_id"],
                model_name=r["model_name"],
                market=r["market"],
                predicted_outcome=r["predicted_outcome"],
                probability=r["probability"],
                probabilities_json=r["probabilities_json"],
                confidence=r["confidence"],
                confidence_score=r["confidence_score"],
                data_quality_notes=r["data_quality_notes"],
                features_json=r["features_json"],
                status=r["status"],
                is_correct=r["is_correct"],
                settlement_status=r["settlement_status"] if "settlement_status" in r.keys() else "PENDING",
                model_version=r["model_version"] if "model_version" in r.keys() else "v1.1.0",
                odds_json=r["odds_json"] if "odds_json" in r.keys() else None,
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def evaluate_predictions(self, fixture_id: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate pending predictions against actual match results with WIN/LOSS/PUSH/VOID settlement."""
        from zola.prediction.calibration import brier_score, log_loss_score, expected_calibration_error
        cursor = self.conn.cursor()

        if fixture_id:
            cursor.execute("SELECT status FROM fixtures WHERE id = ?;", (fixture_id,))
            f_row = cursor.fetchone()
            if not f_row or str(f_row["status"]).startswith("INVALID") or str(f_row["status"]) == "INVALID_FIXTURE_IDENTITY":
                logger.warning("Settlement blocked for invalid fixture %s", fixture_id)
                return {
                    "evaluated_count": 0,
                    "correct_count": 0,
                    "accuracy_rate": 0.0,
                    "status": "BLOCKED",
                    "error": "INVALID FIXTURE — CANNOT BE SETTLED",
                }

        query = """
            SELECT p.*, mr.full_time_home_score, mr.full_time_away_score, mr.winner
            FROM predictions p
            JOIN fixtures f ON p.fixture_id = f.id
            JOIN match_results mr ON f.id = mr.fixture_id
            WHERE f.status = 'FINISHED'
              AND f.status NOT LIKE 'INVALID%'
              AND f.status != 'INVALID_FIXTURE_IDENTITY'
              AND p.status != 'INVALID'
              AND (p.settlement_status != 'VOID' OR p.settlement_status IS NULL)
              AND mr.full_time_home_score IS NOT NULL
        """
        params: List[Any] = []
        if fixture_id:
            query += " AND p.fixture_id = ?"
            params.append(fixture_id)
        else:
            query += " AND p.status = 'PENDING'"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        evaluated_count = 0
        correct_count = 0
        push_count = 0

        for r in rows:
            home_goals = int(r["full_time_home_score"])
            away_goals = int(r["full_time_away_score"])
            total_goals = home_goals + away_goals
            # Always compute canonical winner from verified final score if available
            if home_goals > away_goals:
                winner = "HOME_TEAM"
            elif away_goals > home_goals:
                winner = "AWAY_TEAM"
            elif home_goals == away_goals:
                winner = "DRAW"
            else:
                raw_w = str(r["winner"] or "").upper()
                if raw_w in ("HOME", "HOME_TEAM"):
                    winner = "HOME_TEAM"
                elif raw_w in ("AWAY", "AWAY_TEAM"):
                    winner = "AWAY_TEAM"
                else:
                    winner = "DRAW"

            market = r["market"]
            pred = r["predicted_outcome"]

            is_correct: Optional[bool] = False
            settlement = "LOSS"

            if market == "1X2":
                if winner == "HOME_TEAM" and pred == "HOME_WIN":
                    is_correct = True
                elif winner == "DRAW" and pred == "DRAW":
                    is_correct = True
                elif winner == "AWAY_TEAM" and pred == "AWAY_WIN":
                    is_correct = True
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "BTTS":
                both_scored = (home_goals > 0) and (away_goals > 0)
                is_correct = (pred == "BTTS_YES" and both_scored) or (pred == "BTTS_NO" and not both_scored)
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "OU05":
                over = total_goals > 0.5
                is_correct = (pred == "OVER_05" and over) or (pred == "UNDER_05" and not over)
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "OU15":
                over = total_goals > 1.5
                is_correct = (pred == "OVER_15" and over) or (pred == "UNDER_15" and not over)
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "OU25":
                over = total_goals > 2.5
                is_correct = (pred == "OVER_25" and over) or (pred == "UNDER_25" and not over)
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "OU35":
                over = total_goals > 3.5
                is_correct = (pred == "OVER_35" and over) or (pred == "UNDER_35" and not over)
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "DOUBLE_CHANCE":
                if pred == "1X" and winner in ("HOME_TEAM", "DRAW"):
                    is_correct = True
                elif pred == "X2" and winner in ("AWAY_TEAM", "DRAW"):
                    is_correct = True
                elif pred == "12" and winner in ("HOME_TEAM", "AWAY_TEAM"):
                    is_correct = True
                settlement = "WIN" if is_correct else "LOSS"

            elif market == "DNB":
                if winner == "DRAW":
                    # Draw No Bet is refunded/push
                    is_correct = None
                    settlement = "PUSH"
                elif winner == "HOME_TEAM" and pred == "1_DNB":
                    is_correct = True
                    settlement = "WIN"
                elif winner == "AWAY_TEAM" and pred == "2_DNB":
                    is_correct = True
                    settlement = "WIN"
                else:
                    is_correct = False
                    settlement = "LOSS"

            status_val = settlement
            is_correct_val = 1 if is_correct is True else (0 if is_correct is False else None)

            cursor.execute(
                """
                UPDATE predictions
                SET status = ?, settlement_status = ?, is_correct = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?;
                """,
                (status_val, settlement, is_correct_val, r["id"]),
            )
            evaluated_count += 1
            if is_correct is True:
                correct_count += 1
            elif is_correct is None:
                push_count += 1

        self.conn.commit()

        # Update accuracy_metrics table with Brier Score, Log Loss & ECE
        cursor.execute(
            """
            SELECT p.model_name, p.market,
                   COUNT(CASE WHEN p.is_correct IN (0, 1) THEN 1 END) as total,
                   SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct,
                   SUM(CASE WHEN p.settlement_status = 'PUSH' OR p.status = 'PUSH' THEN 1 ELSE 0 END) as pushes,
                   SUM(CASE WHEN p.settlement_status = 'VOID' OR p.status = 'VOID' THEN 1 ELSE 0 END) as voids
            FROM predictions p
            JOIN fixtures f ON p.fixture_id = f.id
            WHERE p.status != 'PENDING'
              AND p.status != 'INVALID'
              AND f.status NOT LIKE 'INVALID%'
              AND f.status != 'INVALID_FIXTURE_IDENTITY'
            GROUP BY p.model_name, p.market;
            """
        )
        metric_rows = cursor.fetchall()
        for mr_row in metric_rows:
            total = mr_row["total"] or 0
            correct = mr_row["correct"] or 0
            pushes = mr_row["pushes"] or 0
            voids = mr_row["voids"] or 0
            rate = round(correct / total, 4) if total > 0 else 0.0

            cursor.execute(
                """
                SELECT p.probability, p.is_correct
                FROM predictions p
                JOIN fixtures f ON p.fixture_id = f.id
                WHERE p.model_name = ? AND p.market = ? AND p.is_correct IN (0, 1)
                  AND p.status != 'INVALID'
                  AND f.status NOT LIKE 'INVALID%'
                  AND f.status != 'INVALID_FIXTURE_IDENTITY';
                """,
                (mr_row["model_name"], mr_row["market"]),
            )
            prob_rows = cursor.fetchall()
            y_true = [int(pr["is_correct"]) for pr in prob_rows]
            y_prob = [float(pr["probability"]) for pr in prob_rows]

            bs = brier_score(y_true, y_prob) if y_true else None
            ll = log_loss_score(y_true, y_prob) if y_true else None
            ece = expected_calibration_error(y_true, y_prob)[0] if y_true else None

            cursor.execute(
                """
                INSERT INTO accuracy_metrics (
                    model_name, market, total_predictions, correct_predictions,
                    accuracy_rate, brier_score, log_loss, ece,
                    push_predictions, void_predictions, last_evaluated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(model_name, market) DO UPDATE SET
                    total_predictions = excluded.total_predictions,
                    correct_predictions = excluded.correct_predictions,
                    accuracy_rate = excluded.accuracy_rate,
                    brier_score = excluded.brier_score,
                    log_loss = excluded.log_loss,
                    ece = excluded.ece,
                    push_predictions = excluded.push_predictions,
                    void_predictions = excluded.void_predictions,
                    last_evaluated_at = CURRENT_TIMESTAMP;
                """,
                (
                    mr_row["model_name"],
                    mr_row["market"],
                    total,
                    correct,
                    rate,
                    bs,
                    ll,
                    ece,
                    pushes,
                    voids,
                ),
            )
        self.conn.commit()

        return {
            "evaluated": evaluated_count,
            "correct": correct_count,
            "pushes": push_count,
            "accuracy": round(correct_count / (evaluated_count - push_count), 4) if (evaluated_count - push_count) > 0 else 0.0,
        }

    def settle_fixture(self, fixture_id: int, home_score: int, away_score: int) -> Dict[str, Any]:
        """Manually settle a fixture with confirmed final scores and evaluate its predictions."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT status FROM fixtures WHERE id = ?;", (fixture_id,))
        f_row = cursor.fetchone()
        if not f_row or str(f_row["status"]).startswith("INVALID") or str(f_row["status"]) == "INVALID_FIXTURE_IDENTITY":
            logger.warning("Manual settlement blocked for invalid fixture %s", fixture_id)
            return {
                "fixture_id": fixture_id,
                "status": "BLOCKED",
                "error": "INVALID FIXTURE — CANNOT BE SETTLED",
            }

        winner = "HOME_TEAM" if home_score > away_score else ("AWAY_TEAM" if away_score > home_score else "DRAW")
        mr = MatchResult(
            fixture_id=fixture_id,
            full_time_home_score=home_score,
            full_time_away_score=away_score,
            winner=winner,
        )
        self.upsert_match_result(mr)

        cursor.execute("UPDATE fixtures SET status = 'FINISHED' WHERE id = ?;", (fixture_id,))
        self.conn.commit()

        eval_res = self.evaluate_predictions(fixture_id=fixture_id)
        return {
            "fixture_id": fixture_id,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "evaluation": eval_res,
        }

    def get_accuracy_metrics(
        self,
        model_name: Optional[str] = None,
        market: Optional[str] = None,
        timeframe: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> List[AccuracyMetric]:
        """Fetch calculated accuracy performance stats with optional timeframe or confidence filters."""
        from zola.prediction.calibration import brier_score, log_loss_score, expected_calibration_error
        cursor = self.conn.cursor()

        # If timeframe or confidence filter is applied, compute dynamic stats from predictions
        if (timeframe and timeframe != "all") or (confidence and confidence != "all"):
            base_query = """
                SELECT p.model_name, p.market, p.probability, p.is_correct, p.settlement_status, p.confidence
                FROM predictions p
                JOIN fixtures f ON p.fixture_id = f.id
                WHERE p.status != 'PENDING'
                  AND p.status != 'INVALID'
                  AND f.status NOT LIKE 'INVALID%'
                  AND f.status != 'INVALID_FIXTURE_IDENTITY'
                  AND (p.settlement_status != 'VOID' OR p.settlement_status IS NULL)
            """
            params: List[Any] = []
            if model_name:
                base_query += " AND p.model_name = ?"
                params.append(model_name)
            if market:
                base_query += " AND p.market = ?"
                params.append(market)
            if confidence and confidence != "all":
                base_query += " AND p.confidence = ?"
                params.append(confidence)
            if timeframe == "7d":
                base_query += " AND f.utc_date >= datetime('now', '-7 days')"
            elif timeframe == "30d":
                base_query += " AND f.utc_date >= datetime('now', '-30 days')"
            elif timeframe == "90d":
                base_query += " AND f.utc_date >= datetime('now', '-90 days')"

            cursor.execute(base_query, params)
            rows = cursor.fetchall()

            grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            for r in rows:
                key = (r["model_name"], r["market"])
                grouped.setdefault(key, []).append(r)

            metrics_list = []
            for (m_name, m_kt), item_list in grouped.items():
                decided = [x for x in item_list if x["is_correct"] is not None]
                correct = sum(1 for x in decided if x["is_correct"] == 1)
                pushes = sum(1 for x in item_list if x["settlement_status"] == "PUSH")
                voids = sum(1 for x in item_list if x["settlement_status"] == "VOID")
                rate = round(correct / len(decided), 4) if decided else 0.0

                y_true = [int(x["is_correct"]) for x in decided]
                y_prob = [float(x["probability"]) for x in decided]
                bs = brier_score(y_true, y_prob) if y_true else None
                ll = log_loss_score(y_true, y_prob) if y_true else None
                ece = expected_calibration_error(y_true, y_prob)[0] if y_true else None

                metrics_list.append(
                    AccuracyMetric(
                        model_name=m_name,
                        market=m_kt,
                        total_predictions=len(decided),
                        correct_predictions=correct,
                        accuracy_rate=rate,
                        brier_score=bs,
                        log_loss=ll,
                        ece=ece,
                        push_predictions=pushes,
                        void_predictions=voids,
                        last_evaluated_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
            metrics_list.sort(key=lambda m: (m.accuracy_rate, m.total_predictions), reverse=True)
            return metrics_list

        # Standard unfiltered lookup from accuracy_metrics table
        query = "SELECT * FROM accuracy_metrics"
        params = []
        conditions = []
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        if market:
            conditions.append("market = ?")
            params.append(market)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY accuracy_rate DESC, total_predictions DESC;"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [
            AccuracyMetric(
                id=r["id"],
                model_name=r["model_name"],
                market=r["market"],
                total_predictions=r["total_predictions"],
                correct_predictions=r["correct_predictions"],
                accuracy_rate=r["accuracy_rate"],
                brier_score=r["brier_score"],
                log_loss=r["log_loss"] if "log_loss" in r.keys() else None,
                ece=r["ece"] if "ece" in r.keys() else None,
                push_predictions=r["push_predictions"] if "push_predictions" in r.keys() else 0,
                void_predictions=r["void_predictions"] if "void_predictions" in r.keys() else 0,
                last_evaluated_at=r["last_evaluated_at"],
            )
            for r in rows
        ]

    def get_prediction_history(
        self,
        limit: int = 40,
        offset: int = 0,
        outcome_filter: Optional[str] = None,
        market_filter: Optional[str] = None,
        search: Optional[str] = None,
        competition_id: Optional[int] = None,
        date: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch past completed matches with original Zola predictions, scores, and settlement outcomes."""
        cursor = self.conn.cursor()

        # Available dates query with count of settled matches having verified predictions
        cursor.execute(
            """
            SELECT substr(f.utc_date, 1, 10) as match_date, count(distinct f.id) as match_count
            FROM fixtures f
            JOIN match_results mr ON f.id = mr.fixture_id
            JOIN predictions p ON f.id = p.fixture_id
            WHERE mr.full_time_home_score IS NOT NULL
              AND p.model_name = 'ensemble'
              AND f.status NOT LIKE 'INVALID%'
              AND p.status != 'INVALID'
              AND (p.settlement_status != 'VOID' OR p.settlement_status IS NULL)
              AND f.status NOT LIKE 'INVALID%'
              AND p.status != 'INVALID'
              AND (p.settlement_status != 'VOID' OR p.settlement_status IS NULL)
            GROUP BY match_date
            ORDER BY match_date DESC
            LIMIT 50;
            """
        )
        available_dates = [
            {"date": r["match_date"], "count": r["match_count"]}
            for r in cursor.fetchall()
        ]

        is_date_active = bool(date and str(date).lower() not in ("all", "none", "*", ""))
        clean_date = str(date).strip() if is_date_active else None

        # Dynamic summary stats across settled matches matching active filters
        sum_query = """
            SELECT 
                COUNT(DISTINCT f.id) as total_fixtures,
                COUNT(p.id) as total_picks,
                SUM(CASE WHEN p.settlement_status = 'WIN' OR p.is_correct = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN p.settlement_status = 'LOSS' OR p.is_correct = 0 THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN p.settlement_status = 'PUSH' OR p.status = 'PUSH' THEN 1 ELSE 0 END) as pushes
            FROM fixtures f
            JOIN match_results mr ON f.id = mr.fixture_id
            JOIN predictions p ON f.id = p.fixture_id
            WHERE mr.full_time_home_score IS NOT NULL
              AND p.model_name = 'ensemble'
              AND f.status NOT LIKE 'INVALID%'
              AND p.status != 'INVALID'
              AND (p.settlement_status != 'VOID' OR p.settlement_status IS NULL)
        """
        sum_params: List[Any] = []
        if is_date_active:
            sum_query += " AND DATE(f.utc_date) = ?"
            sum_params.append(clean_date)
        elif date_from and date_to:
            sum_query += " AND DATE(f.utc_date) >= ? AND DATE(f.utc_date) <= ?"
            sum_params.extend([str(date_from).strip(), str(date_to).strip()])

        if competition_id:
            sum_query += " AND f.competition_id = ?"
            sum_params.append(competition_id)

        if market_filter and market_filter != "all":
            sum_query += " AND p.market = ?"
            sum_params.append(market_filter)

        if outcome_filter and outcome_filter != "all":
            if outcome_filter == "WIN":
                sum_query += " AND (p.settlement_status = 'WIN' OR p.is_correct = 1)"
            elif outcome_filter == "LOSS":
                sum_query += " AND (p.settlement_status = 'LOSS' OR p.is_correct = 0)"
            elif outcome_filter == "PUSH":
                sum_query += " AND (p.settlement_status = 'PUSH' OR p.status = 'PUSH')"

        cursor.execute(sum_query, sum_params)
        s_row = cursor.fetchone()
        tot_fixtures = s_row["total_fixtures"] if s_row else 0
        tot_picks = s_row["total_picks"] if s_row else 0
        tot_wins = s_row["wins"] or 0 if s_row else 0
        tot_losses = s_row["losses"] or 0 if s_row else 0
        tot_pushes = s_row["pushes"] or 0 if s_row else 0
        decided = tot_wins + tot_losses
        win_rate = round(tot_wins / decided, 4) if decided > 0 else 0.0

        summary = {
            "total_matches": tot_fixtures,
            "total_picks": tot_picks,
            "total_wins": tot_wins,
            "total_losses": tot_losses,
            "total_pushes": tot_pushes,
            "win_rate": win_rate,
            "is_date_filtered": is_date_active,
            "filtered_date": clean_date,
            "filtered_market": market_filter if market_filter and market_filter != "all" else None,
            "filtered_outcome": outcome_filter if outcome_filter and outcome_filter != "all" else None,
        }

        # Build query for fixtures
        base_query = """
            SELECT f.id as fixture_id, f.utc_date, f.status as fixture_status,
                   c.id as comp_id, c.name as comp_name, c.code as comp_code, c.emblem_url as comp_emblem,
                   ht.id as home_id, ht.name as home_name, ht.crest_url as home_crest,
                   at.id as away_id, at.name as away_name, at.crest_url as away_crest,
                   mr.full_time_home_score, mr.full_time_away_score, mr.winner
            FROM fixtures f
            JOIN competitions c ON f.competition_id = c.id
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            JOIN match_results mr ON f.id = mr.fixture_id
            WHERE mr.full_time_home_score IS NOT NULL
              AND f.status NOT LIKE 'INVALID%'
              AND (mr.winner != 'VOID' OR mr.winner IS NULL)
              AND EXISTS (
                  SELECT 1 FROM predictions p
                  WHERE p.fixture_id = f.id
                    AND p.status != 'INVALID'
                    AND (p.settlement_status != 'VOID' OR p.settlement_status IS NULL)
              )
        """
        params: List[Any] = []

        if is_date_active:
            base_query += " AND DATE(f.utc_date) = ?"
            params.append(str(date).strip())
        elif date_from and date_to:
            base_query += " AND DATE(f.utc_date) >= ? AND DATE(f.utc_date) <= ?"
            params.extend([str(date_from).strip(), str(date_to).strip()])

        if search:
            search_like = f"%{search.strip()}%"
            base_query += " AND (ht.name LIKE ? OR at.name LIKE ? OR c.name LIKE ?)"
            params.extend([search_like, search_like, search_like])

        if competition_id:
            base_query += " AND f.competition_id = ?"
            params.append(competition_id)

        if outcome_filter and outcome_filter != "all":
            if outcome_filter == "WIN":
                base_query += " AND EXISTS (SELECT 1 FROM predictions p WHERE p.fixture_id = f.id AND p.model_name = 'ensemble' AND (p.settlement_status = 'WIN' OR p.is_correct = 1))"
            elif outcome_filter == "LOSS":
                base_query += " AND EXISTS (SELECT 1 FROM predictions p WHERE p.fixture_id = f.id AND p.model_name = 'ensemble' AND (p.settlement_status = 'LOSS' OR p.is_correct = 0))"
            elif outcome_filter == "PUSH":
                base_query += " AND EXISTS (SELECT 1 FROM predictions p WHERE p.fixture_id = f.id AND p.model_name = 'ensemble' AND (p.settlement_status = 'PUSH' OR p.status = 'PUSH'))"

        if market_filter and market_filter != "all":
            base_query += " AND EXISTS (SELECT 1 FROM predictions p WHERE p.fixture_id = f.id AND p.model_name = 'ensemble' AND p.market = ?)"
            params.append(market_filter)

        count_query = f"SELECT COUNT(*) FROM ({base_query})"
        cursor.execute(count_query, params)
        total_matching = cursor.fetchone()[0]

        base_query += " ORDER BY f.utc_date DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        cursor.execute(base_query, params)
        rows = cursor.fetchall()

        history_items = []
        for r in rows:
            fid = r["fixture_id"]
            h_score = r["full_time_home_score"]
            a_score = r["full_time_away_score"]
            if h_score is not None and a_score is not None:
                winner = "HOME_TEAM" if h_score > a_score else ("AWAY_TEAM" if a_score > h_score else "DRAW")
            else:
                raw_w = str(r["winner"] or "").upper()
                winner = "HOME_TEAM" if raw_w in ("HOME", "HOME_TEAM") else ("AWAY_TEAM" if raw_w in ("AWAY", "AWAY_TEAM") else "DRAW")

            p_query = """
                SELECT market, predicted_outcome, probability, confidence, confidence_score,
                       settlement_status, is_correct, features_json
                FROM predictions
                WHERE fixture_id = ? AND model_name = 'ensemble'
                  AND status != 'INVALID'
                  AND (settlement_status != 'VOID' OR settlement_status IS NULL)
            """
            p_params: List[Any] = [fid]
            if market_filter and market_filter != "all":
                p_query += " AND market = ?"
                p_params.append(market_filter)
            if outcome_filter and outcome_filter != "all":
                if outcome_filter == "WIN":
                    p_query += " AND (settlement_status = 'WIN' OR is_correct = 1)"
                elif outcome_filter == "LOSS":
                    p_query += " AND (settlement_status = 'LOSS' OR is_correct = 0)"
                elif outcome_filter == "PUSH":
                    p_query += " AND (settlement_status = 'PUSH' OR status = 'PUSH')"
            p_query += """
                ORDER BY CASE market
                    WHEN '1X2' THEN 1
                    WHEN 'DOUBLE_CHANCE' THEN 2
                    WHEN 'DNB' THEN 3
                    WHEN 'BTTS' THEN 4
                    WHEN 'OU05' THEN 5
                    WHEN 'OU15' THEN 6
                    WHEN 'OU25' THEN 7
                    WHEN 'OU35' THEN 8
                    ELSE 9 END ASC;
            """
            cursor.execute(p_query, p_params)
            p_rows = cursor.fetchall()
            preds_list = []
            c_wins = 0
            c_losses = 0
            c_pushes = 0

            for pr in p_rows:
                settle = pr["settlement_status"]
                is_corr = pr["is_correct"]
                if not settle or settle == "PENDING":
                    if is_corr == 1:
                        settle = "WIN"
                    elif is_corr == 0:
                        settle = "LOSS"
                    elif pr["market"] == "DNB" and winner == "DRAW":
                        settle = "PUSH"
                    else:
                        settle = "VOID"

                if settle == "WIN":
                    c_wins += 1
                elif settle == "LOSS":
                    c_losses += 1
                elif settle == "PUSH":
                    c_pushes += 1

                preds_list.append({
                    "market": pr["market"],
                    "predicted_outcome": pr["predicted_outcome"],
                    "probability": round(float(pr["probability"]), 4),
                    "confidence": pr["confidence"],
                    "confidence_score": pr["confidence_score"],
                    "settlement_status": settle,
                    "is_correct": is_corr,
                })

            from zola.utils.flags import get_country_flag_url
            h_flag = get_country_flag_url(r["home_name"])
            a_flag = get_country_flag_url(r["away_name"])

            item = {
                "fixture_id": fid,
                "utc_date": r["utc_date"],
                "status": r["fixture_status"],
                "competition": {
                    "id": r["comp_id"],
                    "name": r["comp_name"],
                    "code": r["comp_code"],
                    "emblem_url": r["comp_emblem"],
                },
                "home_team": {
                    "id": r["home_id"],
                    "name": r["home_name"],
                    "crest_url": r["home_crest"] or h_flag,
                    "flag_url": h_flag,
                },
                "away_team": {
                    "id": r["away_id"],
                    "name": r["away_name"],
                    "crest_url": r["away_crest"] or a_flag,
                    "flag_url": a_flag,
                },
                "final_score": {
                    "home": h_score,
                    "away": a_score,
                    "winner": winner,
                    "display": f"{h_score} - {a_score}",
                },
                "ensemble_predictions": preds_list,
                "card_summary": {
                    "total_markets": len(preds_list),
                    "wins": c_wins,
                    "losses": c_losses,
                    "pushes": c_pushes,
                    "win_rate": round(c_wins / (c_wins + c_losses), 4) if (c_wins + c_losses) > 0 else 0.0,
                },
            }
            history_items.append(item)

        return {
            "total_count": total_matching,
            "summary": summary,
            "history": history_items,
            "available_dates": available_dates,
            "selected_date": str(date).strip() if is_date_active else None,
        }

    def run_walk_forward_backtest(
        self,
        sample_limit: int = 40,
        min_history: int = 4,
    ) -> Dict[str, Any]:
        """Perform chronological walk-forward validation across historical finished fixtures."""
        from zola.prediction.dixon_coles import DixonColesModel
        from zola.prediction.elo import EloRatingModel
        from zola.prediction.ensemble import ConsensusEnsembleModel
        from zola.prediction.form_momentum import FormMomentumModel
        from zola.prediction.head_to_head import HeadToHeadModel
        from zola.prediction.poisson import PoissonModel
        from zola.prediction.calibration import brier_score, log_loss_score, expected_calibration_error

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id, f.competition_id,
                   mr.full_time_home_score, mr.full_time_away_score, mr.winner
            FROM fixtures f
            JOIN match_results mr ON f.id = mr.fixture_id
            WHERE f.status = 'FINISHED'
              AND mr.full_time_home_score IS NOT NULL
            ORDER BY f.utc_date DESC
            LIMIT ?;
            """,
            (sample_limit,),
        )
        fixtures = cursor.fetchall()
        if not fixtures:
            return {"status": "EMPTY", "message": "No finished fixtures found for backtesting."}

        models = [
            PoissonModel(),
            DixonColesModel(),
            EloRatingModel(),
            FormMomentumModel(),
            HeadToHeadModel(),
        ]
        ensemble = ConsensusEnsembleModel()

        evaluated_matches = 0
        market_stats: Dict[str, Dict[str, Any]] = {
            m: {"total": 0, "correct": 0, "probs": [], "actuals": []}
            for m in ["1X2", "BTTS", "OU05", "OU15", "OU25", "OU35", "DOUBLE_CHANCE", "DNB"]
        }
        model_stats: Dict[str, Dict[str, Any]] = {
            m.name: {"total": 0, "correct": 0} for m in models
        }
        model_stats["ensemble"] = {"total": 0, "correct": 0}

        for fix in fixtures:
            fix_id = fix["id"]
            h_id = fix["home_team_id"]
            a_id = fix["away_team_id"]
            comp_id = fix["competition_id"]
            match_date = fix["utc_date"]
            h_score = fix["full_time_home_score"]
            a_score = fix["full_time_away_score"]
            tot_goals = h_score + a_score
            if h_score is not None and a_score is not None:
                winner = "HOME_TEAM" if h_score > a_score else ("AWAY_TEAM" if a_score > h_score else "DRAW")
            else:
                raw_w = str(fix["winner"] or "").upper()
                winner = "HOME_TEAM" if raw_w in ("HOME", "HOME_TEAM") else ("AWAY_TEAM" if raw_w in ("AWAY", "AWAY_TEAM") else "DRAW")

            # Walk-forward data retrieval: only matches strictly before match_date
            cursor.execute(
                """
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score,
                       CASE WHEN f.home_team_id = ? THEN 1 ELSE 0 END as is_home
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                WHERE (f.home_team_id = ? OR f.away_team_id = ?)
                  AND f.utc_date < ?
                ORDER BY f.utc_date DESC
                LIMIT 20;
                """,
                (h_id, h_id, h_id, match_date),
            )
            home_matches = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score,
                       CASE WHEN f.home_team_id = ? THEN 1 ELSE 0 END as is_home
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                WHERE (f.home_team_id = ? OR f.away_team_id = ?)
                  AND f.utc_date < ?
                ORDER BY f.utc_date DESC
                LIMIT 20;
                """,
                (a_id, a_id, a_id, match_date),
            )
            away_matches = [dict(r) for r in cursor.fetchall()]

            if len(home_matches) < min_history or len(away_matches) < min_history:
                continue

            cursor.execute(
                """
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                WHERE ((f.home_team_id = ? AND f.away_team_id = ?) OR (f.home_team_id = ? AND f.away_team_id = ?))
                  AND f.utc_date < ?
                ORDER BY f.utc_date DESC
                LIMIT 10;
                """,
                (h_id, a_id, a_id, h_id, match_date),
            )
            h2h_matches = [dict(r) for r in cursor.fetchall()]

            cursor.execute(
                """
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score
                FROM fixtures f
                JOIN match_results mr ON f.id = mr.fixture_id
                WHERE f.competition_id = ? AND f.utc_date < ?
                ORDER BY f.utc_date DESC
                LIMIT 50;
                """,
                (comp_id, match_date),
            )
            league_matches = [dict(r) for r in cursor.fetchall()]

            fix_dict = dict(fix)
            fix_preds: Dict[str, Dict[str, Any]] = {}
            for m in models:
                try:
                    preds = m.predict(fix_dict, home_matches, away_matches, h2h_matches, league_matches)
                    fix_preds[m.name] = preds
                except Exception:
                    pass

            if not fix_preds:
                continue

            ensemble_preds = ensemble.combine(fix_preds)
            fix_preds["ensemble"] = ensemble_preds
            evaluated_matches += 1

            # Settle each model on 1X2
            for m_name, preds_dict in fix_preds.items():
                if "1X2" in preds_dict:
                    pred_1x2 = preds_dict["1X2"].predicted_outcome
                    correct_1x2 = (
                        (winner == "HOME_TEAM" and pred_1x2 == "HOME_WIN") or
                        (winner == "DRAW" and pred_1x2 == "DRAW") or
                        (winner == "AWAY_TEAM" and pred_1x2 == "AWAY_WIN")
                    )
                    model_stats[m_name]["total"] += 1
                    if correct_1x2:
                        model_stats[m_name]["correct"] += 1

            # Settle ensemble across all 8 markets
            for mkt, pred_obj in ensemble_preds.items():
                pred_out = pred_obj.predicted_outcome
                is_win = False
                is_push = False

                if mkt == "1X2":
                    is_win = (
                        (winner == "HOME_TEAM" and pred_out == "HOME_WIN") or
                        (winner == "DRAW" and pred_out == "DRAW") or
                        (winner == "AWAY_TEAM" and pred_out == "AWAY_WIN")
                    )
                elif mkt == "BTTS":
                    b_scored = (h_score > 0 and a_score > 0)
                    is_win = (pred_out == "BTTS_YES" and b_scored) or (pred_out == "BTTS_NO" and not b_scored)
                elif mkt == "OU05":
                    is_win = (pred_out == "OVER_05" and tot_goals > 0.5) or (pred_out == "UNDER_05" and tot_goals <= 0.5)
                elif mkt == "OU15":
                    is_win = (pred_out == "OVER_15" and tot_goals > 1.5) or (pred_out == "UNDER_15" and tot_goals <= 1.5)
                elif mkt == "OU25":
                    is_win = (pred_out == "OVER_25" and tot_goals > 2.5) or (pred_out == "UNDER_25" and tot_goals <= 2.5)
                elif mkt == "OU35":
                    is_win = (pred_out == "OVER_35" and tot_goals > 3.5) or (pred_out == "UNDER_35" and tot_goals <= 3.5)
                elif mkt == "DOUBLE_CHANCE":
                    is_win = (
                        (pred_out == "1X" and winner in ("HOME_TEAM", "DRAW")) or
                        (pred_out == "X2" and winner in ("AWAY_TEAM", "DRAW")) or
                        (pred_out == "12" and winner in ("HOME_TEAM", "AWAY_TEAM"))
                    )
                elif mkt == "DNB":
                    if winner == "DRAW":
                        is_push = True
                    else:
                        is_win = (winner == "HOME_TEAM" and pred_out == "1_DNB") or (winner == "AWAY_TEAM" and pred_out == "2_DNB")

                if not is_push:
                    market_stats[mkt]["total"] += 1
                    if is_win:
                        market_stats[mkt]["correct"] += 1
                    market_stats[mkt]["probs"].append(pred_obj.probability)
                    market_stats[mkt]["actuals"].append(1 if is_win else 0)

        # Calculate final backtest results
        market_results = {}
        for mkt, s in market_stats.items():
            tot = s["total"]
            corr = s["correct"]
            rate = round(corr / tot, 4) if tot > 0 else 0.0
            bs = brier_score(s["actuals"], s["probs"]) if tot > 0 else None
            ll = log_loss_score(s["actuals"], s["probs"]) if tot > 0 else None
            ece = expected_calibration_error(s["actuals"], s["probs"])[0] if tot > 0 else None
            market_results[mkt] = {
                "total": tot,
                "correct": corr,
                "accuracy_rate": rate,
                "brier_score": bs,
                "log_loss": ll,
                "ece": ece,
            }

        model_results = {}
        for m_name, s in model_stats.items():
            tot = s["total"]
            corr = s["correct"]
            model_results[m_name] = {
                "total": tot,
                "correct": corr,
                "accuracy_rate": round(corr / tot, 4) if tot > 0 else 0.0,
            }

        return {
            "status": "SUCCESS",
            "evaluated_matches": evaluated_matches,
            "sample_limit": sample_limit,
            "markets": market_results,
            "models_1x2": model_results,
        }

    # ---------------------------------------------------------
    # Historical Data Imports Provenance
    # ---------------------------------------------------------
    def record_historical_import(self, imp: HistoricalDataImport) -> int:
        """Record provenance and metadata for imported CSV datasets."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO historical_data_imports (
                source_name, filename, file_hash, season, competition_code, rows_imported, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id;
            """,
            (
                imp.source_name,
                imp.filename,
                imp.file_hash,
                imp.season,
                imp.competition_code,
                imp.rows_imported,
                imp.notes,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_historical_imports(self) -> List[HistoricalDataImport]:
        """Fetch all recorded historical data imports."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM historical_data_imports ORDER BY imported_at DESC;")
        rows = cursor.fetchall()
        return [
            HistoricalDataImport(
                id=r["id"],
                source_name=r["source_name"],
                filename=r["filename"],
                file_hash=r["file_hash"],
                season=r["season"],
                competition_code=r["competition_code"],
                rows_imported=r["rows_imported"],
                notes=r["notes"],
                imported_at=r["imported_at"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------
    # Research Sources
    # ---------------------------------------------------------
    def upsert_research_source(self, source: ResearchSource) -> int:
        """Insert or update a research source."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO research_sources (name, source_type, url, status, notes)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                source_type = excluded.source_type,
                url = COALESCE(excluded.url, research_sources.url),
                status = excluded.status,
                notes = COALESCE(excluded.notes, research_sources.notes),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (source.name, source.source_type, source.url, source.status, source.notes),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_research_sources(self) -> List[ResearchSource]:
        """Return all research sources."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM research_sources ORDER BY id ASC;")
        rows = cursor.fetchall()
        return [
            ResearchSource(
                id=r["id"],
                name=r["name"],
                source_type=r["source_type"],
                url=r["url"],
                status=r["status"],
                notes=r["notes"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------
    # User-selected monitored sources (Preserved & Paused)
    # ---------------------------------------------------------
    def add_monitored_source(self, platform: str, name: str, url: str) -> int:
        """Add or reactivate a user-selected monitoring source."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO monitored_sources (platform, name, url, status)
            VALUES (?, ?, ?, 'ACTIVE')
            ON CONFLICT(url) DO UPDATE SET
                platform = excluded.platform,
                name = excluded.name,
                status = 'ACTIVE',
                updated_at = CURRENT_TIMESTAMP
            RETURNING id;
            """,
            (platform, name, url),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def list_monitored_sources(self, include_paused: bool = True) -> List[Dict[str, Any]]:
        """List user-selected sources without exposing credentials (none are stored here)."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM monitored_sources"
        if not include_paused:
            query += " WHERE status = 'ACTIVE'"
        query += " ORDER BY platform, name"
        cursor.execute(query)
        return [dict(row) for row in cursor.fetchall()]

    def set_monitored_source_status(self, source_id: int, status: str) -> bool:
        """Pause or reactivate a source; return whether a row was updated."""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE monitored_sources SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, source_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def save_source_item(self, item: Dict[str, Any]) -> int:
        """Store a discovered post idempotently for future collectors."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO source_items
                (source_id, external_id, title, url, published_at, content_text, content_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, external_id) DO UPDATE SET
                title = COALESCE(excluded.title, source_items.title),
                url = excluded.url,
                published_at = COALESCE(excluded.published_at, source_items.published_at),
                content_text = COALESCE(excluded.content_text, source_items.content_text),
                content_hash = COALESCE(excluded.content_hash, source_items.content_hash)
            RETURNING id;
            """,
            (
                item["source_id"], item["external_id"], item.get("title"), item["url"],
                item.get("published_at"), item.get("content_text"), item.get("content_hash"),
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_monitored_source_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Find an active monitored source by its canonical URL."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM monitored_sources WHERE url = ? AND status = 'ACTIVE' LIMIT 1", (url,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_recent_source_items(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the latest collected source items with their source labels."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT i.*, s.platform, s.name AS source_name
            FROM source_items i JOIN monitored_sources s ON s.id = i.source_id
            ORDER BY COALESCE(i.published_at, i.collected_at) DESC, i.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    # ---------------------------------------------------------
    # API Collection Logs
    # ---------------------------------------------------------
    def log_api_call(self, log_entry: ApiCollectionLog) -> int:
        """Record an API collection log entry."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO api_collection_logs (provider, endpoint, status_code, response_time_ms, cache_hit, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING id;
            """,
            (
                log_entry.provider,
                log_entry.endpoint,
                log_entry.status_code,
                log_entry.response_time_ms,
                1 if log_entry.cache_hit else 0,
                log_entry.error_message,
            ),
        )
        row = cursor.fetchone()
        self.conn.commit()
        return int(row["id"] if isinstance(row, sqlite3.Row) else row[0])

    def get_recent_api_logs(self, limit: int = 20) -> List[ApiCollectionLog]:
        """Fetch recent API collection logs."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM api_collection_logs ORDER BY id DESC LIMIT ?;",
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            ApiCollectionLog(
                id=r["id"],
                provider=r["provider"],
                endpoint=r["endpoint"],
                status_code=r["status_code"],
                response_time_ms=r["response_time_ms"],
                cache_hit=bool(r["cache_hit"]),
                error_message=r["error_message"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------
    # Stats Summary
    # ---------------------------------------------------------
    def get_database_stats(self) -> Dict[str, int]:
        """Return counts of all primary entities."""
        cursor = self.conn.cursor()
        stats: Dict[str, int] = {}
        tables = [
            "competitions", "teams", "fixtures", "match_results",
            "research_sources", "monitored_sources", "source_items",
            "api_collection_logs", "predictions", "historical_data_imports",
        ]
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            stats[table] = int(cursor.fetchone()[0])
        return stats

    # ---------------------------------------------------------
    # BeSoccer-Inspired Match Discovery & Prioritization System
    # ---------------------------------------------------------
    def get_discovery_fixtures(
        self,
        date: Optional[str] = None,
        category: str = "all",
        search: Optional[str] = None,
        country: Optional[str] = None,
        competition_id: Optional[int] = None,
        competition_name: Optional[str] = None,
        sort_by: str = "recommended",
        favorites_only: bool = False,
    ) -> Dict[str, Any]:
        """Fetch fixtures for a specific date, score with prioritization engine, and group by competition."""
        from datetime import timedelta
        from zola.services.prioritization import MatchPrioritizationEngine

        # 1. Resolve date
        if not date or date.lower() in ("today", "all", "upcoming"):
            target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif date.lower() == "yesterday":
            target_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        elif date.lower() == "tomorrow":
            target_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            target_date = date.strip()

        # 2. Get user favorites & prioritization engine
        fav_data = self.get_user_favorites()
        fav_team_ids = {str(item["id"]) for item in fav_data.get("teams", [])}
        fav_team_names = {item["name"].lower() for item in fav_data.get("teams", [])}
        fav_comp_ids = {str(item["id"]) for item in fav_data.get("competitions", [])}
        fav_comp_names = {item["name"].lower() for item in fav_data.get("competitions", [])}

        weights = self.get_prioritization_weights()
        engine = MatchPrioritizationEngine(weights=weights)

        is_upcoming_category = (category or "").lower().strip() == "upcoming"

        # 3. Query all fixtures for this date (or 10-day window if upcoming category)
        cursor = self.conn.cursor()
        if is_upcoming_category:
            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            start_date = min(target_date, now_str) if (not date or date.lower() in ("today", "all", "upcoming")) else target_date
            end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
            query = """
                SELECT 
                    f.id, f.external_id, f.provider, f.utc_date, f.status, f.matchday, f.season, f.venue, f.referee,
                    f.competition_id, f.home_team_id, f.away_team_id,
                    c.name as competition_name, c.code as competition_code, c.country as competition_country,
                    c.emblem_url as competition_logo, c.emblem_url as competition_emblem,
                    ht.name as home_team_name, ht.short_name as home_team_short,
                    ht.crest_url as home_team_logo, ht.crest_url as home_team_crest,
                    at.name as away_team_name, at.short_name as away_team_short,
                    at.crest_url as away_team_logo, at.crest_url as away_team_crest,
                    mr.winner, mr.full_time_home_score, mr.full_time_away_score,
                    mr.half_time_home_score, mr.half_time_away_score,
                    EXISTS(SELECT 1 FROM predictions p WHERE p.fixture_id = f.id) as has_prediction
                FROM fixtures f
                JOIN competitions c ON f.competition_id = c.id
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                LEFT JOIN match_results mr ON f.id = mr.fixture_id
                WHERE f.status IN ('SCHEDULED', 'TIMED', 'TBD', 'NOT_STARTED')
                  AND f.status NOT LIKE 'INVALID%'
                  AND f.status != 'INVALID_FIXTURE_IDENTITY'
                  AND DATE(f.utc_date) >= ? AND DATE(f.utc_date) <= ?
                ORDER BY f.utc_date ASC, f.id ASC;
            """
            cursor.execute(query, (start_date, end_date))
        else:
            query = """
                SELECT 
                    f.id, f.external_id, f.provider, f.utc_date, f.status, f.matchday, f.season, f.venue, f.referee,
                    f.competition_id, f.home_team_id, f.away_team_id,
                    c.name as competition_name, c.code as competition_code, c.country as competition_country,
                    c.emblem_url as competition_logo, c.emblem_url as competition_emblem,
                    ht.name as home_team_name, ht.short_name as home_team_short,
                    ht.crest_url as home_team_logo, ht.crest_url as home_team_crest,
                    at.name as away_team_name, at.short_name as away_team_short,
                    at.crest_url as away_team_logo, at.crest_url as away_team_crest,
                    mr.winner, mr.full_time_home_score, mr.full_time_away_score,
                    mr.half_time_home_score, mr.half_time_away_score,
                    EXISTS(SELECT 1 FROM predictions p WHERE p.fixture_id = f.id) as has_prediction
                FROM fixtures f
                JOIN competitions c ON f.competition_id = c.id
                JOIN teams ht ON f.home_team_id = ht.id
                JOIN teams at ON f.away_team_id = at.id
                LEFT JOIN match_results mr ON f.id = mr.fixture_id
                WHERE DATE(f.utc_date) = ?
                  AND f.status NOT LIKE 'INVALID%'
                  AND f.status != 'INVALID_FIXTURE_IDENTITY'
                ORDER BY f.utc_date ASC, f.id ASC;
            """
            cursor.execute(query, (target_date,))
        col_names = [col[0] for col in cursor.description] if cursor.description else []
        rows = [dict(r) if hasattr(r, "keys") else dict(zip(col_names, r)) for r in cursor.fetchall()]

        from zola.utils.flags import get_country_flag_url, get_team_monogram_svg

        # 4. Score all fixtures and collect category counts
        scored_fixtures: List[Dict[str, Any]] = []
        counts = {
            "all": len(rows),
            "live": 0,
            "upcoming": 0,
            "finished": 0,
            "postponed": 0,
            "cancelled": 0,
            "favorites": 0,
        }

        distinct_countries_set = set()
        distinct_comps_dict: Dict[int, Dict[str, Any]] = {}

        for f in rows:
            # Score mapping for UI compatibility
            f["home_score"] = f.get("full_time_home_score")
            f["away_score"] = f.get("full_time_away_score")

            # Country flags & monograms
            h_flag = get_country_flag_url(f.get("home_team_name"))
            a_flag = get_country_flag_url(f.get("away_team_name"))
            c_flag = get_country_flag_url(f.get("competition_country"))
            f["home_team_flag"] = h_flag
            f["away_team_flag"] = a_flag
            f["competition_flag"] = c_flag
            if not f.get("home_team_crest") or not f.get("home_team_logo"):
                f["home_team_crest"] = f.get("home_team_crest") or f.get("home_team_logo") or h_flag
                f["home_team_logo"] = f["home_team_crest"]
            if not f.get("away_team_crest") or not f.get("away_team_logo"):
                f["away_team_crest"] = f.get("away_team_crest") or f.get("away_team_logo") or a_flag
                f["away_team_logo"] = f["away_team_crest"]
            if not f.get("competition_logo") or not f.get("competition_emblem"):
                f["competition_logo"] = f.get("competition_logo") or f.get("competition_emblem") or c_flag
                f["competition_emblem"] = f["competition_logo"]
            f["home_team_fallback"] = h_flag or get_team_monogram_svg(f.get("home_team_name", ""))
            f["away_team_fallback"] = a_flag or get_team_monogram_svg(f.get("away_team_name", ""))
            st = (f.get("status") or "").upper()
            c_name = f.get("competition_name") or "Unknown Competition"
            c_country = f.get("competition_country") or "World"
            c_id = f.get("competition_id")

            if c_country:
                distinct_countries_set.add(c_country)
            if c_id not in distinct_comps_dict:
                distinct_comps_dict[c_id] = {
                    "id": c_id,
                    "competition_id": c_id,
                    "name": c_name,
                    "competition_name": c_name,
                    "country": c_country,
                    "competition_country": c_country,
                    "logo": f.get("competition_logo"),
                    "emblem_url": f.get("competition_logo"),
                }

            # Check favorites
            is_fav_team = (
                str(f["home_team_id"]) in fav_team_ids or
                f["home_team_name"].lower() in fav_team_names or
                str(f["away_team_id"]) in fav_team_ids or
                f["away_team_name"].lower() in fav_team_names
            )
            is_fav_comp = (
                str(c_id) in fav_comp_ids or
                c_name.lower() in fav_comp_names
            )
            is_fav = is_fav_team or is_fav_comp
            f["is_favorite"] = bool(is_fav)
            f["is_favorite_home"] = bool(str(f["home_team_id"]) in fav_team_ids or f["home_team_name"].lower() in fav_team_names)
            f["is_favorite_away"] = bool(str(f["away_team_id"]) in fav_team_ids or f["away_team_name"].lower() in fav_team_names)
            f["is_favorite_competition"] = bool(is_fav_comp)

            # Compute priority score
            score, cat, breakdown = engine.compute_priority_score(f)
            f["priority_score"] = score
            f["priority_category"] = cat
            f["priority_breakdown"] = breakdown

            # Tally counts
            if is_fav:
                counts["favorites"] += 1

            if st in ("IN_PLAY", "LIVE", "PAUSED", "1H", "2H", "HT", "ET", "P"):
                counts["live"] += 1
                f["category_type"] = "live"
            elif st in ("FINISHED", "FT", "AET", "PEN"):
                counts["finished"] += 1
                f["category_type"] = "finished"
            elif st in ("POSTPONED", "PST"):
                counts["postponed"] += 1
                f["category_type"] = "postponed"
            elif st in ("CANCELLED", "CANC", "ABD", "ABANDONED"):
                counts["cancelled"] += 1
                f["category_type"] = "cancelled"
            else:
                counts["upcoming"] += 1
                f["category_type"] = "upcoming"

            scored_fixtures.append(f)

        # Ensure upcoming count reflects all scheduled matches across the 10-day window
        try:
            up_end = (datetime.strptime(target_date, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
            cursor.execute(
                """
                SELECT COUNT(*) FROM fixtures
                WHERE status IN ('SCHEDULED', 'TIMED', 'TBD', 'NOT_STARTED')
                  AND status NOT LIKE 'INVALID%'
                  AND status != 'INVALID_FIXTURE_IDENTITY'
                  AND DATE(utc_date) >= ? AND DATE(utc_date) <= ?
                """,
                (target_date, up_end),
            )
            up_row = cursor.fetchone()
            if up_row and up_row[0] > 0:
                counts["upcoming"] = up_row[0]
        except Exception:
            pass

        # 5. Apply Category, Search, Country, Competition, and Favorites Filters
        filtered: List[Dict[str, Any]] = []
        cat_lower = (category or "all").lower().strip()
        search_lower = (search or "").lower().strip()
        country_lower = (country or "").lower().strip()
        comp_name_lower = (competition_name or "").lower().strip()

        for f in scored_fixtures:
            st = f.get("category_type", "upcoming")
            # Category filter
            if cat_lower == "live" and st != "live":
                continue
            elif cat_lower == "upcoming" and st != "upcoming":
                continue
            elif cat_lower == "finished" and st != "finished":
                continue
            elif cat_lower == "postponed" and st != "postponed":
                continue
            elif cat_lower == "cancelled" and st != "cancelled":
                continue
            elif cat_lower == "favorites" and not f.get("is_favorite"):
                continue

            if favorites_only and not f.get("is_favorite"):
                continue

            if competition_id and f.get("competition_id") != competition_id:
                continue

            if comp_name_lower and (f.get("competition_name") or "").lower().strip() != comp_name_lower:
                continue

            if country_lower and (f.get("competition_country") or "").lower().strip() != country_lower:
                continue

            if search_lower:
                match_text = f"{f.get('home_team_name', '')} {f.get('away_team_name', '')} {f.get('competition_name', '')} {f.get('competition_country', '')}".lower()
                if search_lower not in match_text:
                    continue

            filtered.append(f)

        # 6. Apply Sorting
        sort_mode = (sort_by or "recommended").lower().strip()
        if sort_mode == "kickoff":
            filtered.sort(key=lambda x: (x.get("utc_date", ""), -x.get("priority_score", 0)))
        elif sort_mode == "popularity":
            filtered.sort(key=lambda x: -x.get("priority_breakdown", {}).get("team_popularity", 0) - x.get("priority_breakdown", {}).get("competition_popularity", 0))
        elif sort_mode == "competition":
            filtered.sort(key=lambda x: (x.get("competition_country", ""), x.get("competition_name", ""), x.get("utc_date", "")))
        elif sort_mode == "live_first":
            def live_first_key(x):
                is_live = 0 if x.get("category_type") == "live" else 1
                return (is_live, -x.get("priority_score", 0), x.get("utc_date", ""))
            filtered.sort(key=live_first_key)
        else:  # recommended
            def recommended_key(x):
                is_live = 0 if x.get("category_type") == "live" else 1
                return (is_live, -x.get("priority_score", 0), x.get("utc_date", ""))
            filtered.sort(key=recommended_key)

        # 7. Extract Featured & Live Matches (strictly from the FILTERED set so filters are respected)
        # When filtered to a specific competition/country, feature top matches (>= 68.0); otherwise marquee threshold >= 75.0
        featured_threshold = 68.0 if (competition_id or comp_name_lower or country_lower) else 75.0
        featured_matches = [
            f for f in filtered
            if f.get("priority_score", 0) >= featured_threshold and f.get("category_type") in ("upcoming", "live")
        ]
        featured_matches.sort(key=lambda x: -x.get("priority_score", 0))
        featured_matches = featured_matches[:6]

        live_matches = [f for f in filtered if f.get("category_type") == "live"]
        live_matches.sort(key=lambda x: -x.get("priority_score", 0))

        # 8. Group matches by competition (BeSoccer style)
        comp_groups: Dict[int, Dict[str, Any]] = {}
        for f in filtered:
            cid = f["competition_id"]
            if cid not in comp_groups:
                c_country_name = f.get("competition_country") or "World"
                c_flag_url = get_country_flag_url(c_country_name)
                comp_groups[cid] = {
                    "id": cid,
                    "competition_id": cid,
                    "name": f.get("competition_name") or "Unknown Competition",
                    "competition_name": f.get("competition_name") or "Unknown Competition",
                    "code": f.get("competition_code"),
                    "country": c_country_name,
                    "competition_country": c_country_name,
                    "country_flag": c_flag_url,
                    "competition_flag": c_flag_url,
                    "logo": f.get("competition_logo"),
                    "emblem_url": f.get("competition_logo"),
                    "competition_logo": f.get("competition_logo"),
                    "is_favorite": bool(str(cid) in fav_comp_ids or (f.get("competition_name") or "").lower() in fav_comp_names),
                    "priority_score": f.get("priority_score", 50.0),
                    "matches": [],
                }
            comp_groups[cid]["matches"].append(f)
            # Track max priority score for the competition header
            if f.get("priority_score", 0) > comp_groups[cid]["priority_score"]:
                comp_groups[cid]["priority_score"] = f.get("priority_score", 0)

        # Convert groups to list and calculate match counts
        primary_competitions: List[Dict[str, Any]] = []
        other_competitions: List[Dict[str, Any]] = []

        for cid, group in comp_groups.items():
            group["match_count"] = len(group["matches"])
            # Within each competition, keep chronological order unless explicitly sorting by popularity
            if sort_mode not in ("popularity",):
                group["matches"].sort(key=lambda x: x.get("utc_date", ""))

            # Categorize into Primary vs Other Competitions (lower tier or score < 40)
            if group["priority_score"] >= 40.0:
                primary_competitions.append(group)
            else:
                other_competitions.append(group)

        # Sort competitions by priority score descending
        primary_competitions.sort(key=lambda x: (-x.get("priority_score", 0), x.get("country", ""), x.get("name", "")))
        other_competitions.sort(key=lambda x: (-x.get("priority_score", 0), x.get("country", ""), x.get("name", "")))

        return {
            "date": target_date,
            "counts": counts,
            "featured": featured_matches,
            "featured_matches": featured_matches,
            "live": live_matches,
            "live_matches": live_matches,
            "competitions": primary_competitions,
            "grouped_competitions": primary_competitions,
            "other_competitions": other_competitions,
            "total_matches": len(filtered),
            "all_fixtures": filtered,
            "available_countries": sorted(list(distinct_countries_set)),
            "available_competitions": sorted(list(distinct_comps_dict.values()), key=lambda x: x["name"]),
            "sort": sort_mode,
            "category": cat_lower,
        }

    # ---------------------------------------------------------
    # Favorites Management
    # ---------------------------------------------------------
    def get_user_favorites(self) -> Dict[str, List[Dict[str, str]]]:
        """Fetch all stored user favorites grouped by team and competition."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT item_type, item_id, item_name FROM user_favorites ORDER BY item_name ASC;")
        rows = cursor.fetchall()
        teams = []
        competitions = []
        for r in rows:
            item = {"id": r["item_id"], "name": r["item_name"]}
            if r["item_type"] == "team":
                teams.append(item)
            elif r["item_type"] == "competition":
                competitions.append(item)
        return {"teams": teams, "competitions": competitions}

    def toggle_user_favorite(self, item_type: str, item_id: str, item_name: str) -> bool:
        """Toggle favorite status. Returns True if added, False if removed."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM user_favorites WHERE item_type = ? AND item_id = ?;",
            (item_type, str(item_id)),
        )
        row = cursor.fetchone()
        if row:
            cursor.execute(
                "DELETE FROM user_favorites WHERE item_type = ? AND item_id = ?;",
                (item_type, str(item_id)),
            )
            self.conn.commit()
            return False
        else:
            cursor.execute(
                "INSERT INTO user_favorites (item_type, item_id, item_name) VALUES (?, ?, ?);",
                (item_type, str(item_id), item_name),
            )
            self.conn.commit()
            return True

    # ---------------------------------------------------------
    # Prioritization Settings Management
    # ---------------------------------------------------------
    def get_prioritization_weights(self) -> PrioritizationWeights:
        """Retrieve stored scoring weights or return default weights."""
        from zola.services.prioritization import PrioritizationWeights

        cursor = self.conn.cursor()
        cursor.execute("SELECT value_json FROM prioritization_settings WHERE key = 'weights' LIMIT 1;")
        row = cursor.fetchone()
        if row and row["value_json"]:
            try:
                data = json.loads(row["value_json"])
                return PrioritizationWeights(**data).normalize()
            except Exception:
                pass
        return PrioritizationWeights()

    def save_prioritization_weights(self, weights_dict: Dict[str, float]) -> PrioritizationWeights:
        """Persist updated prioritization scoring weights."""
        from zola.services.prioritization import PrioritizationWeights

        weights = PrioritizationWeights(
            competition_importance=float(weights_dict.get("competition_importance", 0.30)),
            competition_popularity=float(weights_dict.get("competition_popularity", 0.20)),
            team_popularity=float(weights_dict.get("team_popularity", 0.20)),
            match_relevance=float(weights_dict.get("match_relevance", 0.15)),
            kickoff_proximity=float(weights_dict.get("kickoff_proximity", 0.10)),
            data_completeness=float(weights_dict.get("data_completeness", 0.05)),
        ).normalize()

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO prioritization_settings (key, value_json, updated_at)
            VALUES ('weights', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP;
            """,
            (json.dumps(weights.to_dict()),),
        )
        self.conn.commit()
        return weights

    def get_setting(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve JSON setting by key from prioritization_settings."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT value_json, updated_at FROM prioritization_settings WHERE key = ? LIMIT 1;", (key,))
        row = cursor.fetchone()
        if row and row["value_json"]:
            try:
                data = json.loads(row["value_json"])
                if isinstance(data, dict):
                    data["_updated_at"] = str(row["updated_at"])
                return data
            except Exception:
                return None
        return None

    def save_setting(self, key: str, value: Dict[str, Any]) -> None:
        """Persist JSON setting by key into prioritization_settings."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO prioritization_settings (key, value_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = CURRENT_TIMESTAMP;
            """,
            (key, json.dumps(value)),
        )
        self.conn.commit()

    # ---------------------------------------------------------
    # Telegram Recipients Management
    # ---------------------------------------------------------
    def get_telegram_recipients(self) -> List[Dict[str, Any]]:
        """Retrieve all saved Telegram recipients ordered by default status and recency."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, chat_id, username, display_name, type, is_default, last_used_at, created_at
            FROM telegram_recipients
            ORDER BY is_default DESC, last_used_at DESC;
        """)
        return [dict(r) for r in cursor.fetchall()]

    def upsert_telegram_recipient(
        self,
        chat_id: str,
        username: Optional[str] = None,
        display_name: Optional[str] = None,
        recipient_type: str = "private",
        is_default: bool = False,
    ) -> Dict[str, Any]:
        """Save or update a Telegram recipient and optionally set as default."""
        cid = str(chat_id).strip()
        u = username.strip().lstrip("@") if username and username.strip() else None
        d = display_name.strip() if display_name and display_name.strip() else (f"@{u}" if u else f"Chat {cid}")

        cursor = self.conn.cursor()
        if is_default:
            cursor.execute("UPDATE telegram_recipients SET is_default = 0;")

        cursor.execute("""
            INSERT INTO telegram_recipients (chat_id, username, display_name, type, is_default, last_used_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = COALESCE(excluded.username, telegram_recipients.username),
                display_name = COALESCE(excluded.display_name, telegram_recipients.display_name),
                type = excluded.type,
                is_default = CASE WHEN excluded.is_default = 1 THEN 1 ELSE telegram_recipients.is_default END,
                last_used_at = CURRENT_TIMESTAMP;
        """, (cid, u, d, recipient_type, 1 if is_default else 0))
        self.conn.commit()

        cursor.execute("SELECT id, chat_id, username, display_name, type, is_default, last_used_at FROM telegram_recipients WHERE chat_id = ?;", (cid,))
        row = cursor.fetchone()
        return dict(row) if row else {"chat_id": cid, "username": u, "display_name": d, "is_default": is_default}

    def find_telegram_recipient(self, query: str) -> Optional[Dict[str, Any]]:
        """Find a recipient by numeric chat_id, username (@user or user), or display name."""
        if not query:
            return None
        q = str(query).strip()
        q_clean = q.lstrip("@").lower()

        cursor = self.conn.cursor()
        # 1. Exact chat_id match
        cursor.execute("SELECT id, chat_id, username, display_name, type, is_default FROM telegram_recipients WHERE chat_id = ? LIMIT 1;", (q,))
        row = cursor.fetchone()
        if row:
            return dict(row)

        # 2. Case-insensitive username match
        cursor.execute("SELECT id, chat_id, username, display_name, type, is_default FROM telegram_recipients WHERE LOWER(username) = ? LIMIT 1;", (q_clean,))
        row = cursor.fetchone()
        if row:
            return dict(row)

        # 3. Case-insensitive display name match
        cursor.execute("SELECT id, chat_id, username, display_name, type, is_default FROM telegram_recipients WHERE LOWER(display_name) = ? LIMIT 1;", (q.lower(),))
        row = cursor.fetchone()
        if row:
            return dict(row)

        return None

    def touch_telegram_recipient(self, chat_id: str) -> None:
        """Update last_used_at timestamp when a message is successfully delivered."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE telegram_recipients SET last_used_at = CURRENT_TIMESTAMP WHERE chat_id = ?;", (str(chat_id).strip(),))
        self.conn.commit()

    def delete_telegram_recipient(self, chat_id: str) -> bool:
        """Delete a recipient by chat_id."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM telegram_recipients WHERE chat_id = ?;", (str(chat_id).strip(),))
        self.conn.commit()
        return cursor.rowcount > 0
