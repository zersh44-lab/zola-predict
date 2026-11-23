"""Unit tests for automatic daily fixture synchronization and Vercel Cron endpoint."""

import json
import sqlite3
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from zola.config import Settings
from zola.database.models import Competition, Fixture, Team
from zola.database.repository import FootballRepository
from zola.client.base import BaseFootballClient, StandardCompetition, StandardFixture, StandardTeam, FootballApiError
from zola.services.collector import FootballCollectorService
from zola.dashboard.server import DashboardServer


class MockFootballClient(BaseFootballClient):
    """Mock client for testing fixture sync without external network calls."""

    def __init__(self, fixtures=None, should_fail=False):
        super().__init__(api_key="mock_key", cache=None, rate_limiter=None)
        self._fixtures = fixtures or []
        self.should_fail = should_fail

    @property
    def provider_name(self) -> str:
        return "api-football"

    @property
    def default_base_url(self) -> str:
        return "https://api-football.mock"

    def get_auth_headers(self):
        return {"x-apisports-key": "mock_key"}

    def get_historical_results(self, competition_code_or_id, season=None, limit=50):
        return [], False, 10

    def get_competitions(self):
        return [StandardCompetition(external_id="39", name="Premier League", code="PL", country="England")], False, 10

    def get_fixtures(self, competition_code_or_id=None, status=None, season=None, date=None, date_from=None, date_to=None):
        if self.should_fail:
            raise FootballApiError("Simulated upstream network failure")
        # Filter fixtures for the requested date if date is passed
        if date:
            filtered = [f for f in self._fixtures if f.utc_date.startswith(date)]
            return filtered, False, 15
        return self._fixtures, False, 15


@pytest.fixture
def sync_db(tmp_path: Path):
    """Create a temporary initialized SQLite database with standard test entities."""
    db_file = tmp_path / "sync_test.db"
    from zola.database.connection import init_db, get_connection
    init_db(db_file)
    conn = get_connection(db_file)
    repo = FootballRepository(conn)

    # Competitions
    c = Competition(external_id="39", name="Premier League", code="PL", country="England", provider="api-football")
    cid = repo.upsert_competition(c)

    # Teams
    t1 = Team(external_id="40", name="Liverpool", short_name="LIV", tla="LIV", provider="api-football")
    t2 = Team(external_id="33", name="Manchester United", short_name="MUN", tla="MUN", provider="api-football")
    t3 = Team(external_id="47", name="Tottenham", short_name="TOT", tla="TOT", provider="api-football")
    t4 = Team(external_id="57", name="Ipswich", short_name="IPS", tla="IPS", provider="api-football")
    t1_id = repo.upsert_team(t1)
    t2_id = repo.upsert_team(t2)
    t3_id = repo.upsert_team(t3)
    t4_id = repo.upsert_team(t4)

    conn.close()
    return db_file, cid, (t1_id, t2_id, t3_id, t4_id)


def test_cron_sync_idempotent_and_no_duplicates(sync_db):
    """Confirm sync fetches upcoming fixtures and creates zero duplicate rows on repeated runs."""
    db_file, cid, (t1, t2, t3, t4) = sync_db
    from zola.database.connection import get_connection

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tomorrow_str = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    sf1 = StandardFixture(
        external_id="1557991",
        competition_external_id="39",
        utc_date=f"{today_str}T15:00:00+00:00",
        status="SCHEDULED",
        home_team=StandardTeam(external_id="40", name="Liverpool"),
        away_team=StandardTeam(external_id="33", name="Manchester United"),
        season="2026",
    )
    sf2 = StandardFixture(
        external_id="1557992",
        competition_external_id="39",
        utc_date=f"{tomorrow_str}T15:00:00+00:00",
        status="SCHEDULED",
        home_team=StandardTeam(external_id="47", name="Tottenham"),
        away_team=StandardTeam(external_id="57", name="Ipswich"),
        season="2026",
    )

    client = MockFootballClient(fixtures=[sf1, sf2])

    conn = get_connection(db_file)
    repo = FootballRepository(conn)
    collector = FootballCollectorService(client=client, repository=repo)

    # First run
    res1 = collector.sync_upcoming_window(days_ahead=2)
    assert res1["fixtures_synced"] == 2

    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM fixtures WHERE external_id IN ('1557991', '1557992');")
    assert cur.fetchone()[0] == 2

    # Second run with same data (must be idempotent, 0 new rows created)
    res2 = collector.sync_upcoming_window(days_ahead=2)
    cur.execute("SELECT count(*) FROM fixtures WHERE external_id IN ('1557991', '1557992');")
    assert cur.fetchone()[0] == 2  # Still exactly 2
    conn.close()


def test_cron_sync_handles_postponed_and_rescheduled(sync_db):
    """Confirm sync updates status for postponed matches and changes date for rescheduled matches."""
    db_file, cid, (t1, t2, t3, t4) = sync_db
    from zola.database.connection import get_connection

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Initial scheduled fixture
    sf = StandardFixture(
        external_id="1557995",
        competition_external_id="39",
        utc_date=f"{today_str}T15:00:00+00:00",
        status="SCHEDULED",
        home_team=StandardTeam(external_id="40", name="Liverpool"),
        away_team=StandardTeam(external_id="33", name="Manchester United"),
        season="2026",
    )
    client = MockFootballClient(fixtures=[sf])

    conn = get_connection(db_file)
    repo = FootballRepository(conn)
    collector = FootballCollectorService(client=client, repository=repo)
    collector.sync_upcoming_window(days_ahead=1)

    cur = conn.cursor()
    cur.execute("SELECT utc_date, status FROM fixtures WHERE external_id = '1557995';")
    row = cur.fetchone()
    assert row["status"] == "SCHEDULED"

    # Match gets postponed and rescheduled to 3 days later
    new_date = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
    sf_updated = StandardFixture(
        external_id="1557995",
        competition_external_id="39",
        utc_date=f"{new_date}T20:00:00+00:00",
        status="POSTPONED",
        home_team=StandardTeam(external_id="40", name="Liverpool"),
        away_team=StandardTeam(external_id="33", name="Manchester United"),
        season="2026",
    )
    client_updated = MockFootballClient(fixtures=[sf_updated])
    collector_updated = FootballCollectorService(client=client_updated, repository=repo)
    collector_updated.sync_upcoming_window(days_ahead=4)

    cur.execute("SELECT utc_date, status FROM fixtures WHERE external_id = '1557995';")
    updated_row = cur.fetchone()
    assert updated_row["status"] == "POSTPONED"
    assert new_date in updated_row["utc_date"]
    conn.close()


def test_cron_sync_preserves_quarantined_fixtures(sync_db):
    """Confirm sync can never overwrite or unquarantine an INVALID_FIXTURE_IDENTITY record."""
    db_file, cid, (t1, t2, t3, t4) = sync_db
    from zola.database.connection import get_connection

    conn = get_connection(db_file)
    cur = conn.cursor()
    # Insert quarantined fixture
    cur.execute("""
        INSERT INTO fixtures (external_id, provider, competition_id, utc_date, status, home_team_id, away_team_id)
        VALUES ('1208946', 'api-football', ?, '2026-11-22 16:30:00', 'INVALID_FIXTURE_IDENTITY', ?, ?);
    """, (cid, t1, t2))
    conn.commit()

    repo = FootballRepository(conn)
    # Incoming payload with valid-looking details under the quarantined external_id
    f_attack = Fixture(
        external_id="1208946",
        provider="api-football",
        competition_id=cid,
        utc_date="2026-11-22 16:30:00",
        status="SCHEDULED",
        home_team_id=t1,
        away_team_id=t2,
    )
    fid = repo.upsert_fixture(f_attack)

    cur.execute("SELECT status FROM fixtures WHERE external_id = '1208946';")
    status = cur.fetchone()[0]
    assert status == "INVALID_FIXTURE_IDENTITY"  # Remains quarantined!
    conn.close()


def test_cron_sync_api_failure_preserves_existing_data(sync_db):
    """Confirm that when upstream API fails, existing database records are untouched."""
    db_file, cid, (t1, t2, t3, t4) = sync_db
    from zola.database.connection import get_connection

    conn = get_connection(db_file)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO fixtures (external_id, provider, competition_id, utc_date, status, home_team_id, away_team_id)
        VALUES ('pre_existing_1', 'api-football', ?, '2026-11-25 15:00:00', 'SCHEDULED', ?, ?);
    """, (cid, t1, t2))
    conn.commit()

    repo = FootballRepository(conn)
    failing_client = MockFootballClient(should_fail=True)
    collector = FootballCollectorService(client=failing_client, repository=repo)

    # Sync throws exception or handles failure gracefully
    collector.sync_upcoming_window(days_ahead=3)

    # Check that existing fixture still exists intact
    cur.execute("SELECT status, utc_date FROM fixtures WHERE external_id = 'pre_existing_1';")
    row = cur.fetchone()
    assert row is not None
    assert row["status"] == "SCHEDULED"
    conn.close()


def test_cron_endpoint_auth_and_metadata(sync_db):
    """Confirm /api/cron/sync enforces CRON_SECRET and logs execution metadata."""
    db_file, cid, _ = sync_db
    settings = Settings(
        football_api_key="mock_key",
        cron_secret="my_super_secret_cron_token",
        db_path=db_file,
    )

    from zola.dashboard.server import DashboardRequestHandler
    import io

    # 1. Reject without auth header or secret param
    h1 = DashboardRequestHandler.__new__(DashboardRequestHandler)
    h1.server = MagicMock(settings=settings)
    h1.headers = {}
    h1.path = "/api/cron/sync"
    h1.rfile = io.BytesIO()
    h1.wfile = io.BytesIO()
    
    with patch.object(h1, "_send_json") as mock_send:
        h1._handle_cron_sync(settings, str(db_file))
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "Unauthorized" in args[0].get("error", "")
        assert mock_send.call_args[0][1] == 401

    # 2. Accept with valid Bearer token
    h2 = DashboardRequestHandler.__new__(DashboardRequestHandler)
    h2.server = MagicMock(settings=settings)
    h2.headers = {"Authorization": "Bearer my_super_secret_cron_token"}
    h2.path = "/api/cron/sync?days_ahead=2"
    h2.rfile = io.BytesIO()
    h2.wfile = io.BytesIO()

    with patch("zola.services.collector.FootballCollectorService.sync_upcoming_window") as mock_sync, \
         patch("zola.prediction.engine.PredictionEngine.predict_upcoming_fixtures") as mock_pred, \
         patch.object(h2, "_send_json") as mock_send:
        mock_sync.return_value = {
            "start_date": "2026-11-23",
            "end_date": "2026-11-25",
            "days_ahead": 2,
            "dates": ["2026-11-23", "2026-11-24", "2026-11-25"],
            "fixtures_synced": 5,
        }
        mock_pred.return_value = 8

        h2._handle_cron_sync(settings, str(db_file))
        mock_send.assert_called_once()
        res_data = mock_send.call_args[0][0]
        assert res_data["success"] is True
        assert res_data["sync"]["fixtures_synced"] == 5
        assert res_data["sync"]["predictions_generated"] == 8
        assert "05:00 Ethiopia Time" in res_data["sync"]["schedule"]

    # 3. Verify metadata persisted in DB and retrievable
    from zola.database.connection import get_connection
    conn = get_connection(db_file)
    repo = FootballRepository(conn)
    meta = repo.get_setting("last_fixture_sync")
    assert meta is not None
    assert meta["status"] == "SUCCESS"
    assert meta["fixtures_synced"] == 5
    conn.close()
