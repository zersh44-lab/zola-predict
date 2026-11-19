"""Unit tests for the local web dashboard HTTP server and REST API endpoints."""

import json
import threading
import urllib.request
import pytest
from datetime import datetime
from pathlib import Path

from zola.config import Settings
from zola.database.models import Competition, Team, Fixture
from zola.dashboard.server import DashboardServer


@pytest.fixture
def test_dashboard_server(repo, tmp_path: Path):
    """Run DashboardServer on an ephemeral port with the test repository."""
    # Seed sample competition, teams, and fixture
    c = Competition(external_id="PL_TEST", name="Premier League", code="PL", provider="mock")
    c_id = repo.upsert_competition(c)

    t1 = Team(external_id="ARS_TEST", name="Arsenal", short_name="Arsenal", tla="ARS", provider="mock")
    t2 = Team(external_id="MCI_TEST", name="Man City", short_name="Man City", tla="MCI", provider="mock")
    t1_id = repo.upsert_team(t1)
    t2_id = repo.upsert_team(t2)

    f = Fixture(
        external_id="FIX_TEST_1",
        competition_id=c_id,
        home_team_id=t1_id,
        away_team_id=t2_id,
        season="2024",
        utc_date=datetime(2024, 11, 28, 20, 0, 0),
        status="SCHEDULED",
        provider="mock",
    )
    repo.upsert_fixture(f)

    # Use a file-based SQLite database for threaded server requests
    db_file = tmp_path / "test_dash.db"
    settings = Settings(
        football_api_key="mock_key",
        gemini_api_key="mock_gemini",
        telegram_bot_token="mock_token",
        db_path=db_file,
    )
    from zola.database.connection import init_db, get_connection
    init_db(db_file)
    f_conn = get_connection(db_file)
    from zola.database.repository import FootballRepository
    f_repo = FootballRepository(f_conn)
    c_id = f_repo.upsert_competition(c)
    t1_id = f_repo.upsert_team(t1)
    t2_id = f_repo.upsert_team(t2)
    f.competition_id = c_id
    f.home_team_id = t1_id
    f.away_team_id = t2_id
    f_repo.upsert_fixture(f)
    f_conn.close()

    # Port 0 lets OS pick a free available port
    server = DashboardServer(("127.0.0.1", 0), settings)
    host, port = server.server_address

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    base_url = f"http://{host}:{port}"
    yield base_url, server

    server.shutdown()
    server.server_close()


def test_dashboard_index_html(test_dashboard_server):
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "Zola Predict" in content
        assert "Consensus 1X2 Recommendation" in content


def test_api_status(test_dashboard_server):
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(f"{base_url}/api/status")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "online"
        assert data["credentials"]["FOOTBALL_API_KEY"] == "PRESENT"
        assert data["stats"]["fixtures"] >= 1


def test_api_fixtures(test_dashboard_server):
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(f"{base_url}/api/fixtures")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert len(data["fixtures"]) >= 1
        assert data["fixtures"][0]["home_team_name"] == "Arsenal"


def test_api_predict_fixture(test_dashboard_server):
    base_url, _ = test_dashboard_server
    # Predict fixture ID 1
    req = urllib.request.Request(f"{base_url}/api/predict?id=1")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "predictions" in data
        assert "home_form" in data
        assert "away_form" in data
        assert "h2h" in data
        assert "comparison_stats" in data
        preds = data["predictions"]
        assert "ensemble" in preds
        assert "poisson" in preds
        assert "1X2" in preds["ensemble"]
        assert "BTTS" in preds["ensemble"]


def test_api_evaluate_predictions(test_dashboard_server):
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(
        f"{base_url}/api/evaluate",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["success"] is True
        assert "stats" in data


def test_api_competitions_and_filters(test_dashboard_server):
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(f"{base_url}/api/competitions")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "competitions" in data
        assert len(data["competitions"]) >= 1
        comp = data["competitions"][0]
        assert "upcoming_count" in comp
        assert comp["name"] == "Premier League"

    # Test filtering with competition_id and gender=men
    req_f = urllib.request.Request(f"{base_url}/api/fixtures?competition_id=1&gender=men")
    with urllib.request.urlopen(req_f, timeout=5) as resp_f:
        assert resp_f.status == 200
        data_f = json.loads(resp_f.read().decode("utf-8"))
        assert len(data_f["fixtures"]) >= 1


def test_api_history(test_dashboard_server):
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(f"{base_url}/api/history?limit=10")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "history" in data
        assert "summary" in data
        assert "total_count" in data
        assert "available_dates" in data
        assert "selected_date" in data
        assert data["summary"]["is_date_filtered"] is False

    # Test date-specific query
    req_date = urllib.request.Request(f"{base_url}/api/history?date=2024-11-28")
    with urllib.request.urlopen(req_date, timeout=5) as resp_date:
        assert resp_date.status == 200
        data_date = json.loads(resp_date.read().decode("utf-8"))
        assert data_date["selected_date"] == "2024-11-28"
        assert data_date["summary"]["is_date_filtered"] is True
        assert data_date["summary"]["filtered_date"] == "2024-11-28"


def test_dashboard_history_calendar_elements(test_dashboard_server):
    """Verify that index.html contains the history calendar date ribbon, picker, and banner."""
    base_url, _ = test_dashboard_server
    req = urllib.request.Request(f"{base_url}/")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        content = resp.read().decode("utf-8")
        assert "history-date-ribbon" in content
        assert "history-date-picker" in content
        assert "btn-clear-history-date" in content
        assert "history-date-banner" in content
        assert "renderHistoryDateRibbon" in content
        assert "selectHistoryDate" in content



