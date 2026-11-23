"""Lightweight local HTTP dashboard server for Zola Predict."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import urllib.parse
import webbrowser
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from zola.config import Settings, load_settings
from zola.database.connection import get_connection, init_db
from zola.database.repository import FootballRepository
from zola.prediction.engine import PredictionEngine
from zola.research.ai import get_ai_provider
from zola.research.ai.providers import _serialize_predictions
from zola.telegram.reporter import TelegramReportSender
from zola.validation.fixture import validate_fixture_for_prediction

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for Zola Predict local dashboard and REST API."""

    server_settings: Settings

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stdout access logs to keep terminal output clean."""
        logger.debug("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    def _send_json(self, data: Any, status_code: int = 200) -> None:
        """Send JSON response with UTF-8 encoding and CORS headers."""
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_content: str, status_code: int = 200) -> None:
        """Send HTML document response with strict no-cache headers to prevent stale UI caching."""
        payload = html_content.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(payload)

    def _handle_history_sync(self, settings: Settings, db_path: str) -> None:
        """Fetch finished match results for recent matchdays, persist scores, and evaluate predictions."""
        from datetime import datetime, timedelta, timezone
        from zola.client.api_sports import ApiSportsClient
        from zola.database.models import MatchResult
        from zola.services.collector import CALENDAR_RESULTS

        conn = get_connection(db_path)
        synced_count = 0
        evaluated_result = {}
        try:
            repo = FootballRepository(conn)
            cursor = conn.cursor()

            # 1. Historical calendar reconciler permanently disabled for data integrity

            # 2. Fetch live finished fixtures across rolling 4-day window if API key available
            if settings.football_api_key:
                client = ApiSportsClient(api_key=settings.football_api_key)
                now_utc = datetime.now(timezone.utc)
                dates = [
                    (now_utc - timedelta(days=i)).strftime("%Y-%m-%d")
                    for i in range(4, -1, -1)
                ]
                for d in dates:
                    try:
                        fixtures, _, _ = client.get_fixtures(competition_code_or_id="all", date=d, status=None)
                        finished = [
                            f for f in fixtures
                            if f.status in ("FINISHED", "FT", "AET", "PEN")
                            or (f.result and f.result.full_time_home_score is not None)
                        ]
                        for sf in finished:
                            if not sf.result or sf.result.full_time_home_score is None:
                                continue
                            cursor.execute("SELECT id FROM fixtures WHERE external_id = ?", (str(sf.external_id),))
                            row = cursor.fetchone()
                            fid = row["id"] if row else None
                            if not fid and sf.home_team and sf.away_team:
                                cursor.execute(
                                    """
                                    SELECT f.id FROM fixtures f
                                    JOIN teams ht ON f.home_team_id = ht.id
                                    JOIN teams at ON f.away_team_id = at.id
                                    WHERE (ht.external_id = ? AND at.external_id = ?)
                                      AND substr(f.utc_date, 1, 10) = ?
                                    """,
                                    (str(sf.home_team.external_id), str(sf.away_team.external_id), d),
                                )
                                row2 = cursor.fetchone()
                                fid = row2["id"] if row2 else None
                            if not fid and sf.home_team and sf.away_team:
                                cursor.execute(
                                    """
                                    SELECT f.id FROM fixtures f
                                    JOIN teams ht ON f.home_team_id = ht.id
                                    JOIN teams at ON f.away_team_id = at.id
                                    WHERE (LOWER(ht.name) LIKE ? OR LOWER(?) LIKE LOWER(ht.name))
                                      AND (LOWER(at.name) LIKE ? OR LOWER(?) LIKE LOWER(at.name))
                                      AND substr(f.utc_date, 1, 10) = ?
                                    """,
                                    (f"%{sf.home_team.name.lower()}%", sf.home_team.name.lower(),
                                     f"%{sf.away_team.name.lower()}%", sf.away_team.name.lower(), d),
                                )
                                row3 = cursor.fetchone()
                                fid = row3["id"] if row3 else None
                            if fid:
                                cursor.execute("UPDATE fixtures SET status = 'FINISHED' WHERE id = ?", (fid,))
                                repo.upsert_match_result(
                                    MatchResult(
                                        fixture_id=fid,
                                        winner=sf.result.winner,
                                        duration=sf.result.duration or "REGULAR",
                                        full_time_home_score=sf.result.full_time_home_score,
                                        full_time_away_score=sf.result.full_time_away_score,
                                        half_time_home_score=sf.result.half_time_home_score,
                                        half_time_away_score=sf.result.half_time_away_score,
                                    )
                                )
                                synced_count += 1
                        conn.commit()
                    except Exception as e:
                        logger.warning("History sync failed for date %s: %s", d, e)

            evaluated_result = repo.evaluate_predictions()
            self._send_json({
                "status": "success",
                "synced_fixtures": synced_count,
                "evaluated_predictions": evaluated_result,
            })
        finally:
            conn.close()

    def do_OPTIONS(self) -> None:
        """Handle pre-flight CORS requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests for static UI and API endpoints."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        settings = getattr(self.server, "settings", load_settings())
        db_path = settings.db_path

        # 1. Static HTML Frontend & Brand Assets
        if path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            if not index_path.exists():
                alt_path = Path(__file__).resolve().parent.parent.parent / "public" / "index.html"
                if alt_path.exists():
                    index_path = alt_path

            if index_path.exists():
                self._send_html(index_path.read_text(encoding="utf-8"))
            else:
                self._send_json({"error": "Dashboard UI template not found"}, 404)
            return

        if path.startswith("/static/") or path.startswith("/brand/") or path in ("/favicon.ico", "/favicon.png", "/logo.png", "/emblem.png", "/app-icon.png"):
            rel_path = path
            if rel_path.startswith("/static/"):
                rel_path = rel_path[8:]
            elif rel_path.startswith("/"):
                rel_path = rel_path[1:]

            target_file = (STATIC_DIR / rel_path).resolve()
            if target_file.is_file() and str(target_file).startswith(str(STATIC_DIR.resolve())):
                content_type = "application/octet-stream"
                ext = target_file.suffix.lower()
                if ext == ".png":
                    content_type = "image/png"
                elif ext in (".jpg", ".jpeg"):
                    content_type = "image/jpeg"
                elif ext == ".svg":
                    content_type = "image/svg+xml"
                elif ext in (".ico", ".icon"):
                    content_type = "image/x-icon"
                elif ext == ".css":
                    content_type = "text/css"
                elif ext == ".js":
                    content_type = "application/javascript"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "public, max-age=86400")
                data = target_file.read_bytes()
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

        # 2. System Status & Summary Stats
        if path == "/api/status":
            import os
            from pathlib import Path
            root_cand = Path(__file__).resolve().parent.parent.parent / "data" / "zola_predict.db"
            tmp_p = Path("/tmp/zola_predict.db")
            diag = {
                "cwd": os.getcwd(),
                "file": __file__,
                "target_exists": Path("data/zola_predict.db").exists(),
                "target_size": Path("data/zola_predict.db").stat().st_size if Path("data/zola_predict.db").exists() else 0,
                "root_data_exists": root_cand.exists(),
                "root_data_size": root_cand.stat().st_size if root_cand.exists() else 0,
                "var_task_exists": Path("/var/task/data/zola_predict.db").exists(),
                "var_task_size": Path("/var/task/data/zola_predict.db").stat().st_size if Path("/var/task/data/zola_predict.db").exists() else 0,
                "tmp_exists": tmp_p.exists(),
                "tmp_size": tmp_p.stat().st_size if tmp_p.exists() else 0,
                "env_vercel": os.environ.get("VERCEL"),
            }

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                stats = repo.get_database_stats()
                recent_logs = [log.to_dict() for log in repo.get_recent_api_logs(limit=5)]
                cred_status = settings.get_credentials_status()

                self._send_json({
                    "status": "online",
                    "diagnostics": diag,
                    "football_api_provider": settings.football_api_provider,
                    "ai_provider": settings.ai_provider,
                    "ollama_base_url": settings.ollama_base_url,
                    "db_path": str(db_path),
                    "cache_ttl_minutes": settings.cache_ttl_minutes,
                    "rate_limit_rpm": settings.rate_limit_rpm,
                    "stats": stats,
                    "credentials": cred_status,
                    "recent_logs": recent_logs,
                })
            finally:
                conn.close()
            return

        # 3. List Competitions
        if path == "/api/competitions":
            conn = get_connection(db_path)
            try:
                cursor = conn.cursor()
                now_iso = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    """
                    SELECT c.id, c.external_id, c.name, c.code, c.country, c.provider,
                           COUNT(CASE WHEN f.status = 'SCHEDULED' AND f.utc_date >= ? THEN 1 END) as upcoming_count
                    FROM competitions c
                    LEFT JOIN fixtures f ON f.competition_id = c.id
                    GROUP BY c.id
                    ORDER BY c.name ASC
                    """,
                    (now_iso,),
                )
                comps = [dict(r) for r in cursor.fetchall()]
                self._send_json({"competitions": comps})
            finally:
                conn.close()
            return

        # 4. List Fixtures
        if path == "/api/fixtures":
            status = query.get("status", [None])[0]
            date = query.get("date", [None])[0]
            upcoming_only_raw = query.get("upcoming_only", ["false"])[0]
            upcoming_only = upcoming_only_raw.lower() in ("true", "1", "yes")

            if status and status.upper() == "TODAY":
                date = "today"
                status = None
            elif status and status.upper() == "UPCOMING":
                status = "SCHEDULED"
                upcoming_only = True
            comp_id_raw = query.get("competition_id", [None])[0]
            priority_ids_raw = query.get("competition_ids", [None])[0]
            priority_ids = None
            if priority_ids_raw is not None:
                priority_ids = [int(value) for value in priority_ids_raw.split(",") if value.strip().isdigit()]
            gender_raw = query.get("gender", [""])[0].lower()
            gender = gender_raw if gender_raw in ("men", "women") else None
            limit_raw = query.get("limit", ["100"])[0]

            days_ahead_raw = query.get("days_ahead", [None])[0]
            days_ahead = int(days_ahead_raw) if days_ahead_raw and days_ahead_raw.isdigit() else (10 if upcoming_only else None)
            comp_id = int(comp_id_raw) if comp_id_raw and comp_id_raw.isdigit() else None
            try:
                limit = max(1, min(int(limit_raw), 1000))
            except ValueError:
                limit = 100

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                fixtures = repo.get_fixtures(
                    competition_id=comp_id,
                    status=status,
                    upcoming_only=upcoming_only,
                    date=date,
                    gender=gender,
                    limit=limit,
                    priority_competition_ids=priority_ids,
                    days_ahead=days_ahead,
                )
                self._send_json({"fixtures": fixtures})
            finally:
                conn.close()
            return

        # 4b. BeSoccer-Inspired Match Discovery API
        if path == "/api/discovery":
            date = query.get("date", [None])[0]
            category = query.get("category", ["all"])[0]
            search = query.get("search", [None])[0]
            country = query.get("country", [None])[0]
            competition_param = query.get("competition", [None])[0] or query.get("competition_id", [None])[0]
            comp_id = int(competition_param) if competition_param and competition_param.strip().isdigit() else None
            comp_name = competition_param.strip() if competition_param and not competition_param.strip().isdigit() else None
            sort_by = query.get("sort", ["recommended"])[0]
            favs_raw = query.get("favorites_only", ["false"])[0]
            favorites_only = favs_raw.lower() in ("true", "1", "yes")

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                discovery_data = repo.get_discovery_fixtures(
                    date=date,
                    category=category,
                    search=search,
                    country=country,
                    competition_id=comp_id,
                    competition_name=comp_name,
                    sort_by=sort_by,
                    favorites_only=favorites_only,
                )
                # Aliases for frontend convenience
                discovery_data["featured_matches"] = discovery_data.get("featured", [])
                discovery_data["live_matches"] = discovery_data.get("live", [])
                discovery_data["grouped_competitions"] = discovery_data.get("competitions", [])
                if "all_fixtures" not in discovery_data:
                    all_fix = []
                    for c in discovery_data.get("competitions", []):
                        all_fix.extend(c.get("matches", []))
                    for oc in discovery_data.get("other_competitions", []):
                        all_fix.extend(oc.get("matches", []))
                    discovery_data["all_fixtures"] = all_fix
                self._send_json(discovery_data)
            finally:
                conn.close()
            return

        # 4c. Prioritization Weights Config
        if path == "/api/prioritization/config":
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                weights = repo.get_prioritization_weights()
                self._send_json({"weights": weights.to_dict()})
            finally:
                conn.close()
            return

        # 4d. Favorites List
        if path == "/api/favorites":
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                favs = repo.get_user_favorites()
                self._send_json({
                    "favorites": favs,
                    "teams": favs.get("teams", []),
                    "competitions": favs.get("competitions", []),
                })
            finally:
                conn.close()
            return

        # 5. Single Fixture Detail
        if path == "/api/fixture":
            fid_raw = query.get("id", [None])[0]
            if not fid_raw or not fid_raw.isdigit():
                self._send_json({"error": "Invalid or missing 'id' parameter"}, 400)
                return

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                fixture = repo.get_fixture_by_id(int(fid_raw))
                if not fixture:
                    self._send_json({"error": "Fixture not found"}, 404)
                    return

                # Strict Fixture Data Integrity Check
                is_valid, rej_reason = validate_fixture_for_prediction(fixture, enforce_future_kickoff=False)
                if not is_valid:
                    self._send_json({
                        "error": f"Fixture data integrity violation: {rej_reason}",
                        "is_skipped": True,
                        "skip_reason": rej_reason,
                    }, 400)
                    return
                self._send_json({"fixture": fixture})
            finally:
                conn.close()
            return

        # 6. Run & Fetch Predictions for Fixture
        if path == "/api/predict":
            fid_raw = query.get("id", [None])[0]
            if not fid_raw or not fid_raw.isdigit():
                self._send_json({"error": "Invalid or missing 'id' parameter"}, 400)
                return

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                fixture = repo.get_fixture_by_id(int(fid_raw))
                if not fixture:
                    self._send_json({"error": "Fixture not found"}, 404)
                    return

                # Strict Fixture Data Integrity Check
                is_valid, rej_reason = validate_fixture_for_prediction(fixture, enforce_future_kickoff=False)
                if not is_valid:
                    self._send_json({
                        "error": f"Fixture data integrity violation: {rej_reason}",
                        "is_skipped": True,
                        "skip_reason": rej_reason,
                    }, 400)
                    return

                odds_data = None
                raw_odds = query.get("odds", [None])[0]
                if raw_odds:
                    try:
                        odds_data = json.loads(raw_odds)
                    except Exception:
                        pass

                engine = PredictionEngine(repo)
                results = engine.predict_fixture(int(fid_raw), persist=True, odds_data=odds_data)
                serialized = _serialize_predictions(results)

                smart_routing = None
                if "ensemble" in results and "1X2" in results["ensemble"]:
                    feat = results["ensemble"]["1X2"].features or {}
                    smart_routing = feat.get("smart_routing")

                # Pre-Match Context & Comparison Statistics
                home_id = fixture.get("home_team_id")
                away_id = fixture.get("away_team_id")
                home_matches = repo.get_team_matches(home_id, limit=10) if home_id else []
                away_matches = repo.get_team_matches(away_id, limit=10) if away_id else []
                h2h_matches = repo.get_head_to_head_matches(home_id, away_id, limit=10) if (home_id and away_id) else []

                def _extract_team_stats(matches):
                    form = []
                    gf_tot = 0
                    ga_tot = 0
                    clean_sheets = 0
                    wins = 0
                    for m in matches[:5]:
                        w = m.get("winner")
                        is_h = m.get("is_home")
                        if w == "DRAW":
                            form.append("D")
                        elif (is_h == 1 and w == "HOME_TEAM") or (is_h == 0 and w == "AWAY_TEAM"):
                            form.append("W")
                            wins += 1
                        else:
                            form.append("L")

                    for m in matches:
                        is_h = m.get("is_home")
                        gf = m.get("full_time_home_score") if is_h else m.get("full_time_away_score")
                        ga = m.get("full_time_away_score") if is_h else m.get("full_time_home_score")
                        if gf is not None:
                            gf_tot += gf
                        if ga is not None:
                            ga_tot += ga
                            if ga == 0:
                                clean_sheets += 1

                    n = max(len(matches), 1)
                    return {
                        "form": form,
                        "avg_goals_scored": round(gf_tot / n, 2) if matches else None,
                        "avg_goals_conceded": round(ga_tot / n, 2) if matches else None,
                        "clean_sheet_pct": round((clean_sheets / n) * 100, 1) if matches else None,
                        "win_rate_pct": round((wins / max(len(matches[:5]), 1)) * 100, 1) if matches else None,
                    }

                h_stats = _extract_team_stats(home_matches)
                a_stats = _extract_team_stats(away_matches)

                h2h_summary = {"total": len(h2h_matches), "home_wins": 0, "draws": 0, "away_wins": 0}
                for m in h2h_matches:
                    w = m.get("winner")
                    m_hid = m.get("home_team_id")
                    if w == "DRAW":
                        h2h_summary["draws"] += 1
                    elif (w == "HOME_TEAM" and m_hid == home_id) or (w == "AWAY_TEAM" and m_hid != home_id):
                        h2h_summary["home_wins"] += 1
                    elif (w == "AWAY_TEAM" and m_hid == home_id) or (w == "HOME_TEAM" and m_hid != home_id):
                        h2h_summary["away_wins"] += 1

                poi_feat = results.get("poisson", {}).get("1X2", {}).features or {}

                comparison_stats = {
                    "home_xg": poi_feat.get("lambda_home"),
                    "away_xg": poi_feat.get("lambda_away"),
                    "home_goals_scored": h_stats.get("avg_goals_scored"),
                    "away_goals_scored": a_stats.get("avg_goals_scored"),
                    "home_goals_conceded": h_stats.get("avg_goals_conceded"),
                    "away_goals_conceded": a_stats.get("avg_goals_conceded"),
                    "home_clean_sheet_pct": h_stats.get("clean_sheet_pct"),
                    "away_clean_sheet_pct": a_stats.get("clean_sheet_pct"),
                    "home_win_rate_pct": h_stats.get("win_rate_pct"),
                    "away_win_rate_pct": a_stats.get("win_rate_pct"),
                }

                # High-Confidence Prediction System evaluation
                high_confidence_report = None
                ens = results.get("ensemble", {})
                for m_pick in ens.values():
                    if hasattr(m_pick, "features") and m_pick.features and "high_confidence_report" in m_pick.features:
                        high_confidence_report = m_pick.features["high_confidence_report"]
                        break
                if not high_confidence_report:
                    from zola.prediction.high_confidence import HighConfidenceSelector
                    high_confidence_report = HighConfidenceSelector.evaluate_fixture(
                        fixture=fixture,
                        ensemble_predictions=ens,
                        comparison_stats=comparison_stats,
                        home_form=h_stats.get("form", []),
                        away_form=a_stats.get("form", []),
                        h2h_summary=h2h_summary,
                        odds_data=odds_data,
                        require_valid_odds=True,
                    ).to_dict()

                self._send_json({
                    "fixture": fixture,
                    "predictions": serialized,
                    "smart_routing": smart_routing,
                    "high_confidence_report": high_confidence_report,
                    "home_form": h_stats.get("form", []),
                    "away_form": a_stats.get("form", []),
                    "h2h": h2h_summary,
                    "comparison_stats": comparison_stats,
                })
            finally:
                conn.close()
            return

        # 7. Match Research Summary (AI or Statistical Fallback)
        if path == "/api/research":
            fid_raw = query.get("id", [None])[0]
            if not fid_raw or not fid_raw.isdigit():
                self._send_json({"error": "Invalid or missing 'id' parameter"}, 400)
                return

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                fixture = repo.get_fixture_by_id(int(fid_raw))
                if not fixture:
                    self._send_json({"error": "Fixture not found"}, 404)
                    return

                # Strict Fixture Data Integrity Check
                is_valid, rej_reason = validate_fixture_for_prediction(fixture, enforce_future_kickoff=False)
                if not is_valid:
                    self._send_json({
                        "error": f"Fixture data integrity violation: {rej_reason}",
                        "is_skipped": True,
                        "skip_reason": rej_reason,
                    }, 400)
                    return

                engine = PredictionEngine(repo)
                preds = engine.predict_fixture(int(fid_raw), persist=True)

                ai_provider = get_ai_provider(settings)
                report = ai_provider.generate_research_summary(fixture, preds)

                self._send_json({
                    "fixture": fixture,
                    "provider": ai_provider.name,
                    "report": report,
                })
            finally:
                conn.close()
            return

        # 8. Accuracy Metrics Leaderboard & Comprehensive Performance
        if path == "/api/metrics":
            timeframe = query.get("timeframe", [None])[0]
            confidence = query.get("confidence", [None])[0]
            model_name = query.get("model", [None])[0]
            market = query.get("market", [None])[0]

            if timeframe == "all":
                timeframe = None
            if confidence == "all":
                confidence = None

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                metrics = repo.get_accuracy_metrics(
                    model_name=model_name,
                    market=market,
                    timeframe=timeframe,
                    confidence=confidence,
                )
                metrics_dicts = [m.to_dict() for m in metrics]

                tot_evaluated = sum(m.total_predictions for m in metrics)
                tot_correct = sum(m.correct_predictions for m in metrics)
                overall_acc = round(tot_correct / tot_evaluated, 4) if tot_evaluated > 0 else 0.0

                brier_vals = [m.brier_score for m in metrics if m.brier_score is not None]
                avg_brier = round(sum(brier_vals) / len(brier_vals), 4) if brier_vals else None

                ece_vals = [m.ece for m in metrics if m.ece is not None]
                avg_ece = round(sum(ece_vals) / len(ece_vals), 4) if ece_vals else None

                logloss_vals = [m.log_loss for m in metrics if m.log_loss is not None]
                avg_logloss = round(sum(logloss_vals) / len(logloss_vals), 4) if logloss_vals else None

                # Group by Market
                market_group: Dict[str, Dict[str, Any]] = {}
                for m in metrics:
                    mkt = m.market
                    if mkt not in market_group:
                        market_group[mkt] = {"total": 0, "correct": 0, "brier_scores": [], "eces": []}
                    market_group[mkt]["total"] += m.total_predictions
                    market_group[mkt]["correct"] += m.correct_predictions
                    if m.brier_score is not None:
                        market_group[mkt]["brier_scores"].append(m.brier_score)
                    if m.ece is not None:
                        market_group[mkt]["eces"].append(m.ece)

                market_summary = []
                for mkt, data in market_group.items():
                    t = data["total"]
                    c = data["correct"]
                    bs_list = data["brier_scores"]
                    ece_list = data["eces"]
                    market_summary.append({
                        "market": mkt,
                        "total_predictions": t,
                        "correct_predictions": c,
                        "accuracy_rate": round(c / t, 4) if t > 0 else 0.0,
                        "brier_score": round(sum(bs_list) / len(bs_list), 4) if bs_list else None,
                        "ece": round(sum(ece_list) / len(ece_list), 4) if ece_list else None,
                    })
                market_summary.sort(key=lambda x: (x["accuracy_rate"], x["total_predictions"]), reverse=True)

                # Group by Model
                model_group: Dict[str, Dict[str, Any]] = {}
                for m in metrics:
                    m_name = m.model_name
                    if m_name not in model_group:
                        model_group[m_name] = {"total": 0, "correct": 0, "brier_scores": []}
                    model_group[m_name]["total"] += m.total_predictions
                    model_group[m_name]["correct"] += m.correct_predictions
                    if m.brier_score is not None:
                        model_group[m_name]["brier_scores"].append(m.brier_score)

                model_summary = []
                for m_name, data in model_group.items():
                    t = data["total"]
                    c = data["correct"]
                    bs_list = data["brier_scores"]
                    model_summary.append({
                        "model_name": m_name,
                        "total_predictions": t,
                        "correct_predictions": c,
                        "accuracy_rate": round(c / t, 4) if t > 0 else 0.0,
                        "brier_score": round(sum(bs_list) / len(bs_list), 4) if bs_list else None,
                    })
                model_summary.sort(key=lambda x: (x["accuracy_rate"], x["total_predictions"]), reverse=True)

                # Best & Worst Performing Markets
                valid_mkts = [m for m in market_summary if m["total_predictions"] >= 5]
                best_markets = sorted(valid_mkts, key=lambda x: x["accuracy_rate"], reverse=True)[:3]
                worst_markets = sorted(valid_mkts, key=lambda x: x["accuracy_rate"])[:3]

                # Group by League / Competition (Ensemble)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.name as comp_name,
                           COUNT(p.id) as total,
                           SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct
                    FROM predictions p
                    JOIN fixtures f ON p.fixture_id = f.id
                    JOIN competitions c ON f.competition_id = c.id
                    WHERE p.status != 'PENDING' AND p.is_correct IS NOT NULL AND p.model_name = 'ensemble'
                    GROUP BY c.name
                    ORDER BY total DESC;
                """)
                league_summary = []
                for row in cursor.fetchall():
                    tot = row["total"]
                    corr = row["correct"] or 0
                    league_summary.append({
                        "competition": row["comp_name"],
                        "total_predictions": tot,
                        "correct_predictions": corr,
                        "accuracy_rate": round(corr / tot, 4) if tot > 0 else 0.0,
                    })

                # Group by Confidence Tier (Ensemble)
                cursor.execute("""
                    SELECT p.confidence,
                           COUNT(p.id) as total,
                           SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct
                    FROM predictions p
                    WHERE p.status != 'PENDING' AND p.is_correct IS NOT NULL AND p.model_name = 'ensemble'
                    GROUP BY p.confidence
                    ORDER BY total DESC;
                """)
                confidence_summary = []
                for row in cursor.fetchall():
                    tot = row["total"]
                    corr = row["correct"] or 0
                    confidence_summary.append({
                        "confidence": row["confidence"] or "MEDIUM",
                        "total_predictions": tot,
                        "correct_predictions": corr,
                        "accuracy_rate": round(corr / tot, 4) if tot > 0 else 0.0,
                    })

                self._send_json({
                    "metrics": metrics_dicts,
                    "summary": {
                        "total_evaluated": tot_evaluated,
                        "correct_predictions": tot_correct,
                        "overall_accuracy": overall_acc,
                        "average_brier": avg_brier,
                        "average_log_loss": avg_logloss,
                        "average_ece": avg_ece,
                        "timeframe": timeframe or "all",
                        "confidence": confidence or "all",
                    },
                    "by_market": market_summary,
                    "by_model": model_summary,
                    "by_league": league_summary,
                    "by_confidence": confidence_summary,
                    "best_markets": best_markets,
                    "worst_markets": worst_markets,
                })
            finally:
                conn.close()
            return

        # 8b. Calibration & Reliability Diagram Bins
        if path == "/api/calibration":
            conn = get_connection(db_path)
            try:
                from zola.prediction.calibration import expected_calibration_error
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT probability, is_correct
                    FROM predictions
                    WHERE status != 'PENDING' AND is_correct IN (0, 1);
                    """
                )
                rows = cursor.fetchall()
                y_true = [int(r["is_correct"]) for r in rows]
                y_prob = [float(r["probability"]) for r in rows]

                ece, bins = expected_calibration_error(
                    y_true,
                    y_prob,
                    bin_edges=[0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]
                )
                self._send_json({
                    "ece": ece,
                    "total_samples": len(y_true),
                    "bins": bins,
                })
            finally:
                conn.close()
            return

        # 8c. Prediction History & Final Game Results
        if path == "/api/history":
            limit = int(query.get("limit", [40])[0])
            offset = int(query.get("offset", [0])[0])
            outcome_filter = query.get("outcome", [None])[0]
            market_filter = query.get("market", [None])[0]
            search = query.get("search", [None])[0]
            date = query.get("date", [None])[0]
            date_from = query.get("date_from", [None])[0]
            date_to = query.get("date_to", [None])[0]
            comp_id_str = query.get("competition_id", [None])[0]
            comp_id = int(comp_id_str) if comp_id_str and comp_id_str.isdigit() else None

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                data = repo.get_prediction_history(
                    limit=limit,
                    offset=offset,
                    outcome_filter=outcome_filter,
                    market_filter=market_filter,
                    search=search,
                    competition_id=comp_id,
                    date=date,
                    date_from=date_from,
                    date_to=date_to,
                )
                self._send_json(data)
            finally:
                conn.close()
            return

        # 8b. Sync & Settle Recent Prediction History (GET/POST)
        if path in ("/api/history/sync", "/api/history/settle"):
            self._handle_history_sync(settings, db_path)
            return

        # 9. Telegram Info, Discovered Chats & Saved Recipients
        if path == "/api/telegram/info":
            sender = TelegramReportSender(settings.telegram_bot_token)
            configured = sender.is_configured()
            bot_info = sender.get_bot_info() if configured else None
            recent_chats = sender.get_recent_chat_ids() if configured else []

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                # Ensure default chat_id is in recipients if present
                if settings.telegram_chat_id:
                    repo.upsert_telegram_recipient(
                        chat_id=settings.telegram_chat_id,
                        username=None,
                        display_name="Default Recipient",
                        is_default=True,
                    )
                # Sync any discovered chats from bot updates
                for c in recent_chats:
                    cid = str(c.get("chat_id"))
                    if cid:
                        repo.upsert_telegram_recipient(
                            chat_id=cid,
                            username=c.get("username"),
                            display_name=c.get("username") or f"Chat {cid}",
                            recipient_type=c.get("type", "private"),
                        )
                recipients = repo.get_telegram_recipients()
            finally:
                conn.close()

            self._send_json({
                "configured": configured,
                "configured_chat_id": bool(settings.telegram_chat_id),
                "chat_id": settings.telegram_chat_id or "",
                "bot": bot_info,
                "recent_chats": recent_chats,
                "recipients": recipients,
            })
            return

        if path == "/api/telegram/recipients":
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                if settings.telegram_chat_id:
                    repo.upsert_telegram_recipient(
                        chat_id=settings.telegram_chat_id,
                        display_name="Default Recipient",
                        is_default=True,
                    )
                recipients = repo.get_telegram_recipients()
                self._send_json({"recipients": recipients})
            finally:
                conn.close()
            return

        # 10. Monitored Sources (Preserved & Paused)
        if path == "/api/sources":
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                sources = repo.list_monitored_sources()
                items = repo.get_recent_source_items(limit=10)
                self._send_json({"sources": sources, "recent_items": items})
            finally:
                conn.close()
            return

        self._send_json({"error": "Not Found", "path": path}, 404)

    def do_POST(self) -> None:
        """Handle POST requests for actions."""
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        settings = getattr(self.server, "settings", load_settings())
        db_path = settings.db_path

        # Read JSON body if present
        content_length = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_length > 0:
            try:
                raw_body = self.rfile.read(content_length).decode("utf-8")
                body = json.loads(raw_body)
            except Exception:
                body = {}

        # 0. Sync & Settle Recent Prediction History (POST)
        if path in ("/api/history/sync", "/api/history/settle"):
            self._handle_history_sync(settings, db_path)
            return

        # 0a. Run Walk-Forward Historical Backtest
        if path == "/api/backtest":
            limit = int(body.get("limit", 30))
            min_hist = int(body.get("min_history", 2))
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                result = repo.run_walk_forward_backtest(sample_limit=limit, min_history=min_hist)
                self._send_json({"success": True, "backtest": result})
            finally:
                conn.close()
            return

        # 0b. Settle a Specific Fixture with Official Final Scores
        if path == "/api/settle/fixture":
            fid = int(body.get("fixture_id", 0))
            h_score = int(body.get("home_score", 0))
            a_score = int(body.get("away_score", 0))
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                res = repo.settle_fixture(fixture_id=fid, home_score=h_score, away_score=a_score)
                self._send_json({"success": True, "result": res})
            finally:
                conn.close()
            return

        # 1. Run Evaluation of Predictions against Finished Scores
        if path == "/api/evaluate":
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                stats = repo.evaluate_predictions()
                metrics = [m.to_dict() for m in repo.get_accuracy_metrics()]
                self._send_json({
                    "success": True,
                    "stats": stats,
                    "metrics": metrics,
                })
            finally:
                conn.close()
            return

        # 2. Send Telegram Report or Test Message
        if path == "/api/telegram/send":
            user_target = body.get("user") or body.get("username") or body.get("chat_id") or settings.telegram_chat_id
            if not user_target:
                self._send_json({"error": "No Telegram user or chat_id provided and TELEGRAM_CHAT_ID is not configured in .env"}, 400)
                return

            sender = TelegramReportSender(settings.telegram_bot_token)
            if not sender.is_configured():
                self._send_json({"error": "Telegram Bot Token is not configured in .env"}, 400)
                return

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                resolved_chat_id, err = sender.resolve_chat_id(str(user_target), repository=repo)
                if not resolved_chat_id:
                    self._send_json({"error": err or f"Could not find Telegram user '{user_target}'"}, 400)
                    return

                if body.get("test"):
                    ok = sender.send_test_message(chat_id=resolved_chat_id, repository=repo)
                    if ok:
                        self._send_json({"success": True, "message": f"Test message delivered to {user_target} (Chat ID: {resolved_chat_id}).", "chat_id": resolved_chat_id})
                    else:
                        self._send_json({"error": f"Failed to deliver test message to Telegram user {user_target} (Chat ID: {resolved_chat_id})."}, 500)
                    return

                fid = body.get("fixture_id")
                if not fid:
                    self._send_json({"error": "Missing 'fixture_id'"}, 400)
                    return

                ok = sender.send_fixture_report(chat_id=resolved_chat_id, fixture_id=int(fid), repository=repo)
                if ok:
                    # Save as default recipient if requested
                    if body.get("save_as_default"):
                        raw_str = str(user_target).strip()
                        u = raw_str.lstrip("@") if not raw_str.lstrip("-").isdigit() else None
                        repo.upsert_telegram_recipient(
                            chat_id=resolved_chat_id,
                            username=u,
                            display_name=u or f"Chat {resolved_chat_id}",
                            is_default=True,
                        )
                    self._send_json({
                        "success": True,
                        "message": f"Prediction report for fixture #{fid} successfully delivered to {user_target} (Chat ID: {resolved_chat_id})!",
                        "chat_id": resolved_chat_id,
                    })
                else:
                    self._send_json({"error": f"Failed to deliver match report to Telegram user {user_target} (Chat ID: {resolved_chat_id})."}, 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            finally:
                conn.close()
            return

        # 2b. Add / Update Telegram Recipient
        if path == "/api/telegram/recipients":
            raw_target = str(body.get("chat_id") or body.get("user") or "").strip()
            if not raw_target:
                self._send_json({"error": "Missing 'chat_id' or Telegram username"}, 400)
                return
            username = body.get("username")
            display_name = body.get("display_name")
            rec_type = body.get("type") or ("channel" if raw_target.startswith("@") else "private")
            is_default = bool(body.get("is_default", False))

            if raw_target.startswith("@") and not username:
                username = raw_target.lstrip("@")
            if not display_name:
                display_name = f"@{username}" if username else f"Chat {raw_target}"

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                rec = repo.upsert_telegram_recipient(
                    chat_id=raw_target,
                    username=username,
                    display_name=display_name,
                    recipient_type=rec_type,
                    is_default=is_default,
                )

                if is_default:
                    # Update .env and in-memory default
                    env_path = Path(".env")
                    if env_path.exists():
                        lines = env_path.read_text(encoding="utf-8").splitlines()
                        found = False
                        new_lines = []
                        for line in lines:
                            if line.strip().startswith("TELEGRAM_CHAT_ID=") or line.strip() == "TELEGRAM_CHAT_ID":
                                new_lines.append(f"TELEGRAM_CHAT_ID={raw_target}")
                                found = True
                            else:
                                new_lines.append(line)
                        if not found:
                            new_lines.append(f"TELEGRAM_CHAT_ID={raw_target}")
                        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                    if hasattr(self.server, "settings"):
                        import dataclasses
                        self.server.settings = dataclasses.replace(self.server.settings, telegram_chat_id=raw_target)

                self._send_json({"success": True, "recipient": rec})
            finally:
                conn.close()
            return

        # 2c. Delete Telegram Recipient
        if path == "/api/telegram/recipients/delete":
            raw_target = str(body.get("chat_id") or "").strip()
            if not raw_target:
                self._send_json({"error": "Missing 'chat_id'"}, 400)
                return
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                ok = repo.delete_telegram_recipient(raw_target)
                self._send_json({"success": ok, "chat_id": raw_target})
            finally:
                conn.close()
            return

        # 3. Save Default Telegram Chat ID to .env and runtime settings
        if path == "/api/telegram/save-chat-id":
            chat_id = str(body.get("chat_id") or "").strip()
            if not chat_id:
                self._send_json({"error": "No chat_id provided"}, 400)
                return

            env_path = Path(".env")
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
                found = False
                new_lines = []
                for line in lines:
                    if line.strip().startswith("TELEGRAM_CHAT_ID=") or line.strip() == "TELEGRAM_CHAT_ID":
                        new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
                        found = True
                    else:
                        new_lines.append(line)
                if not found:
                    new_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")
                env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

            # Update settings in memory
            import dataclasses
            if hasattr(self.server, "settings"):
                self.server.settings = dataclasses.replace(self.server.settings, telegram_chat_id=chat_id)

            self._send_json({
                "success": True,
                "chat_id": chat_id,
                "message": f"Telegram Chat ID {chat_id} saved successfully as default.",
            })
            return

        # 4. Save/Update Prioritization Weights
        if path == "/api/prioritization/config":
            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                updated_weights = repo.save_prioritization_weights(body)
                self._send_json({"success": True, "weights": updated_weights.to_dict()})
            finally:
                conn.close()
            return

        # 5. Toggle Team or Competition Favorite
        if path == "/api/favorites/toggle":
            item_type = body.get("item_type") or body.get("target_type")
            item_id = str(body.get("item_id") or body.get("target_id") or "").strip()
            item_name = str(body.get("item_name") or body.get("target_name") or "Unknown").strip()
            if not item_type or not item_id:
                self._send_json({"error": "Missing item_type or item_id"}, 400)
                return

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                is_fav = repo.toggle_user_favorite(item_type, item_id, item_name)
                favs = repo.get_user_favorites()
                self._send_json({
                    "success": True,
                    "item_type": item_type,
                    "item_id": item_id,
                    "is_favorite": is_fav,
                    "favorites": favs,
                })
            finally:
                conn.close()
            return

        # 6. Synchronize Specific Date from API-Football
        if path == "/api/sync/date":
            target_date = str(body.get("date") or "today").strip()
            from datetime import timedelta
            from zola.client.api_sports import ApiSportsClient
            from zola.client.cache import RequestCache
            from zola.client.rate_limiter import RateLimiter
            from zola.services.collector import FootballCollectorService

            date_val = target_date
            if target_date.lower() == "today":
                date_val = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            elif target_date.lower() == "yesterday":
                date_val = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            elif target_date.lower() == "tomorrow":
                date_val = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                cache = RequestCache(conn)
                limiter = RateLimiter(max_rpm=10)
                client = ApiSportsClient(api_key=settings.football_api_key, cache=cache, rate_limiter=limiter)
                collector = FootballCollectorService(client=client, repository=repo)

                res = collector.sync_fixtures(competition_code_or_id="all", date=date_val)
                # Auto-generate predictions for newly synced fixtures
                pred_count = 0
                try:
                    engine = PredictionEngine(repo)
                    pred_count = engine.predict_upcoming_fixtures(days_ahead=2)
                except Exception:
                    pass

                self._send_json({
                    "success": True,
                    "date": date_val,
                    "fixtures_synced": res.get("fixtures_synced", 0),
                    "cache_hit": res.get("cache_hit", False),
                    "predictions_generated": pred_count,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            finally:
                conn.close()
            return

        # 7. Synchronize Upcoming Fixtures (7 to 10 days ahead) and generate predictions
        if path == "/api/sync/upcoming":
            days_ahead = int(body.get("days_ahead", 7))
            days_ahead = max(1, min(days_ahead, 14))
            competition = str(body.get("competition") or "all").strip()

            from zola.client.api_sports import ApiSportsClient
            from zola.client.football_data import FootballDataOrgClient
            from zola.client.cache import RequestCache
            from zola.client.rate_limiter import RateLimiter
            from zola.services.collector import FootballCollectorService

            conn = get_connection(db_path)
            try:
                repo = FootballRepository(conn)
                cache = RequestCache(conn)
                limiter = RateLimiter(max_rpm=10)

                fb_key = getattr(settings, "football_data_api_key", None)
                if not fb_key and getattr(settings, "football_api_provider", "") == "football-data":
                    fb_key = settings.football_api_key
                fb_client = FootballDataOrgClient(api_key=fb_key, cache=cache, rate_limiter=limiter) if fb_key else None

                api_sports_key = settings.football_api_key if getattr(settings, "football_api_provider", "") == "api-football" else None
                api_client = ApiSportsClient(api_key=api_sports_key, cache=cache, rate_limiter=limiter) if api_sports_key else None

                primary_client = fb_client or api_client or ApiSportsClient(api_key=settings.football_api_key, cache=cache, rate_limiter=limiter)
                collector = FootballCollectorService(client=primary_client, repository=repo)

                # Sync upcoming rolling window
                sync_res = collector.sync_upcoming_window(
                    competition_code_or_id=competition,
                    days_ahead=days_ahead,
                    football_data_client=fb_client,
                    secondary_client=api_client,
                )

                # Automatically generate predictions for upcoming matches in this window (10 days)
                engine = PredictionEngine(repo)
                pred_count = engine.predict_upcoming_fixtures(days_ahead=max(days_ahead, 10), limit=500)

                self._send_json({
                    "success": True,
                    "days_ahead": days_ahead,
                    "fixtures_synced": sync_res.get("fixtures_synced", 0),
                    "dates": sync_res.get("dates", []),
                    "predictions_generated": pred_count,
                })
            except Exception as e:
                logger.error("Error in /api/sync/upcoming: %s", e, exc_info=True)
                self._send_json({"error": str(e)}, 500)
            finally:
                conn.close()
            return

        self._send_json({"error": "Not Found", "path": path}, 404)


class DashboardServer(HTTPServer):
    """Custom HTTPServer maintaining project settings."""

    def __init__(self, server_address: tuple[str, int], settings: Settings):
        self.settings = settings
        super().__init__(server_address, DashboardRequestHandler)


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
    settings: Optional[Settings] = None,
) -> None:
    """Start local dashboard HTTP server and optionally open default browser."""
    current_settings = settings or load_settings()
    init_db(current_settings.db_path)

    # Attempt to bind to requested port, incrementing if already in use
    target_port = port
    server: Optional[DashboardServer] = None
    for offset in range(10):
        try:
            server = DashboardServer((host, target_port + offset), current_settings)
            target_port = target_port + offset
            break
        except OSError as e:
            if offset == 9:
                raise RuntimeError(f"Could not bind dashboard server to {host}:{port}-{port+9}: {e}")

    if not server:
        raise RuntimeError("Failed to initialize server.")

    url = f"http://{host}:{target_port}"
    print("\n=======================================================")
    print("             ZOLA PREDICT - LOCAL DASHBOARD            ")
    print("=======================================================")
    print(f"  Local Dashboard URL : {url}")
    print(f"  Database Path       : {current_settings.db_path}")
    print(f"  Live Data Provider  : {current_settings.football_api_provider}")
    print(f"  AI Research Provider: {current_settings.ai_provider}")
    print("-------------------------------------------------------")
    print("  Press Ctrl+C in this terminal to stop the dashboard.")
    print("=======================================================\n")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Stopping Zola Predict dashboard server...")
    finally:
        server.server_close()
        print("[+] Dashboard server stopped cleanly.")
