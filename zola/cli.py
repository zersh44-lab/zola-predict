"""Command Line Interface (CLI) for Zola Predict."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from zola.client import create_football_client
from zola.client.base import AuthenticationError, FootballApiError
from zola.config import load_settings
from zola.data_sources.csv_importer import HistoricalCsvImporter
from zola.database.connection import get_connection, init_db
from zola.database.repository import FootballRepository
from zola.prediction.engine import PredictionEngine
from zola.research.ai import get_ai_provider
from zola.services.collector import FootballCollectorService
from zola.telegram.reporter import TelegramReportSender
from zola.dashboard.server import run_dashboard


def cmd_check_env() -> int:
    """Check configuration and display variable presence without exposing values."""
    settings = load_settings()
    status = settings.get_credentials_status()

    print("\n=======================================================")
    print("           ZOLA PREDICT - CONFIGURATION STATUS         ")
    print("=======================================================")
    for var_name, state in status.items():
        state_formatted = f"[{state}]" if state == "PRESENT" else f"[{state}]"
        print(f"  {var_name:<22} : {state_formatted}")

    print("-------------------------------------------------------")
    print(f"  Live Data Provider     : {settings.football_api_provider}")
    print(f"  AI Research Provider   : {settings.ai_provider} (Ollama: {settings.ollama_base_url})")
    print(f"  Database Path          : {settings.db_path}")
    print(f"  Cache TTL (minutes)    : {settings.cache_ttl_minutes}")
    print(f"  Rate Limit (RPM)       : {settings.rate_limit_rpm} req/min")
    print("=======================================================\n")

    if status["FOOTBALL_API_KEY"] == "MISSING":
        print("[!] Note: FOOTBALL_API_KEY is not configured in .env.")
        print("    To enable live data collection, open .env and set:")
        print("    FOOTBALL_API_KEY=your_key_here\n")
        return 1
    return 0


def cmd_init_db() -> int:
    """Initialize database tables, schema, and initial entities."""
    settings = load_settings()
    print(f"[*] Initializing database at: {settings.db_path}...")
    init_db(settings.db_path)

    conn = get_connection(settings.db_path)
    repo = FootballRepository(conn)
    stats = repo.get_database_stats()
    conn.close()

    print("[+] Database initialized successfully.")
    print("[*] Current entity counts:")
    for entity, count in stats.items():
        print(f"    - {entity:<24}: {count}")
    return 0


def cmd_status() -> int:
    """Show database summary and API call statistics."""
    settings = load_settings()
    if not Path(settings.db_path).exists():
        print(f"[!] Database file does not exist at {settings.db_path}. Run 'init-db' first.")
        return 1

    # Apply additive schema changes to existing database
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    repo = FootballRepository(conn)
    stats = repo.get_database_stats()
    recent_logs = repo.get_recent_api_logs(limit=5)
    conn.close()

    print("\n=======================================================")
    print("            ZOLA PREDICT - LOCAL DATABASE STATUS       ")
    print("=======================================================")
    for entity, count in stats.items():
        print(f"  {entity:<24} : {count}")
    print("-------------------------------------------------------")
    print("  Recent API Collection Logs (last 5):")
    if not recent_logs:
        print("    (No API calls recorded yet)")
    else:
        for log in recent_logs:
            hit = "HIT" if log.cache_hit else "MISS"
            print(f"    - [{log.created_at}] {log.provider} {log.endpoint} -> {log.status_code} ({hit}, {log.response_time_ms}ms)")
    print("=======================================================\n")
    return 0


def cmd_sync_competitions() -> int:
    """Synchronize competitions from the configured football API."""
    settings = load_settings()
    conn = get_connection(settings.db_path)
    client = create_football_client(settings, conn=conn)
    repo = FootballRepository(conn)
    collector = FootballCollectorService(client, repo)

    print(f"[*] Fetching competitions from {client.provider_name}...")
    try:
        comps = collector.sync_competitions()
        print(f"[+] Successfully synchronized {len(comps)} competitions.")
        for c in comps[:10]:
            print(f"    - [{c.code or c.external_id}] {c.name} ({c.country or 'International'})")
        if len(comps) > 10:
            print(f"    ... and {len(comps) - 10} more.")
        return 0
    except AuthenticationError as e:
        print(f"[!] Authentication Error: {e}")
        return 1
    except FootballApiError as e:
        print(f"[!] API Error: {e}")
        return 1
    finally:
        conn.close()


def cmd_sync_fixtures(
    competition: Optional[str] = "39",
    status: Optional[str] = "SCHEDULED",
    date: Optional[str] = None,
    season: Optional[str] = None,
    sync_all: bool = False,
    days_ahead: Optional[int] = None,
) -> int:
    """Synchronize fixtures for a competition or date, handling duplicates idempotently."""
    settings = load_settings()
    conn = get_connection(settings.db_path)
    client = create_football_client(settings, conn=conn)
    repo = FootballRepository(conn)
    collector = FootballCollectorService(client, repo)

    # If --all flag is set, clear competition filter
    target_comp = None if sync_all or (competition and competition.lower() in ("all", "*", "none")) else competition

    if days_ahead is not None:
        print(f"[*] Synchronizing upcoming fixtures {days_ahead} days ahead for {target_comp or 'all'}...")
        try:
            res = collector.sync_upcoming_window(
                competition_code_or_id=target_comp,
                days_ahead=days_ahead,
            )
            count = res["fixtures_synced"]
            print(f"[+] Completed: {count} fixture(s) processed and stored across the next {days_ahead} days.")
            engine = PredictionEngine(repo)
            pred_count = engine.predict_upcoming_fixtures(days_ahead=days_ahead)
            if pred_count > 0:
                print(f"[+] Automatically generated predictions for {pred_count} upcoming scheduled fixture(s).")
            return 0
        except AuthenticationError as e:
            print(f"[!] Authentication Error: {e}")
            return 1
        except Exception as e:
            print(f"[!] Sync Error: {e}")
            return 1

    # Resolve date keywords if supplied
    target_date = date
    if date:
        date_lower = date.lower().strip()
        if date_lower == "today":
            target_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        elif date_lower == "tomorrow":
            target_date = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    display_target = f"date {target_date}" if target_date else f"competition '{target_comp or 'all'}'"
    print(f"[*] Synchronizing {status or 'all'} fixtures for {display_target}...")
    try:
        res = collector.sync_fixtures(
            competition_code_or_id=target_comp,
            status=status,
            season=season,
            date=target_date,
        )
        cache_indicator = "(Served from local cache)" if res["cache_hit"] else f"({res['response_time_ms']}ms)"
        count = res["fixtures_synced"]
        print(f"[+] Completed: {count} fixture(s) processed and stored locally {cache_indicator}.")
        if count == 0 and target_comp == "39":
            print("    [*] Note: Premier League (39) currently has 0 scheduled fixtures on selected date (international window).")
            print("    [*] Tip: Run 'sync-fixtures --all --date today' or 'sync-fixtures --date tomorrow' to collect active matches.")
        return 0
    except AuthenticationError as e:
        print(f"[!] Authentication Error: {e}")
        return 1
    except FootballApiError as e:
        print(f"[!] API Error: {e}")
        return 1
    finally:
        conn.close()


def cmd_sync_results(competition: str, limit: int = 20) -> int:
    """Synchronize completed match results and scores for a competition."""
    settings = load_settings()
    conn = get_connection(settings.db_path)
    client = create_football_client(settings, conn=conn)
    repo = FootballRepository(conn)
    collector = FootballCollectorService(client, repo)

    print(f"[*] Synchronizing recent {limit} results for competition '{competition}'...")
    try:
        res = collector.sync_historical_results(competition_code_or_id=competition, limit=limit)
        cache_indicator = "(Served from local cache)" if res["cache_hit"] else f"({res['response_time_ms']}ms)"
        print(f"[+] Completed: {res['historical_results_synced']} completed matches recorded {cache_indicator}.")
        return 0
    except AuthenticationError as e:
        print(f"[!] Authentication Error: {e}")
        return 1
    except FootballApiError as e:
        print(f"[!] API Error: {e}")
        return 1
    finally:
        conn.close()


def cmd_list_fixtures(
    competition: Optional[str] = None,
    status: Optional[str] = None,
    upcoming_only: bool = False,
    limit: int = 15,
) -> int:
    """List stored fixtures from the local database."""
    settings = load_settings()
    conn = get_connection(settings.db_path)
    repo = FootballRepository(conn)

    comp_id = None
    if competition:
        c = repo.get_competition_by_code_or_external_id(settings.football_api_provider, competition)
        if c:
            comp_id = c.id

    fixtures = repo.get_fixtures(competition_id=comp_id, status=status, upcoming_only=upcoming_only, limit=limit)
    conn.close()

    if not fixtures:
        print("[*] No fixtures found matching query. Run 'sync-fixtures' to collect data.")
        return 0

    mode_label = "Upcoming Scheduled" if upcoming_only else (status or "All")
    print(f"\nStored Fixtures ({mode_label}, showing up to {limit}):")
    print(f"{'ID':<6} {'Date (UTC)':<18} {'Status':<11} {'Competition':<22} {'Match':<36} {'Score':<8}")
    print("-" * 105)
    for f in fixtures:
        date_str = str(f["utc_date"])[:16].replace("T", " ")
        match_str = f"{f['home_team_name']} vs {f['away_team_name']}"
        score = f"{f['full_time_home_score']} - {f['full_time_away_score']}" if f["full_time_home_score"] is not None else "-:-"
        comp_name = f['competition_name'][:21]
        print(f"{f['id']:<6} {date_str:<18} {f['status']:<11} {comp_name:<22} {match_str[:35]:<36} {score:<8}")
    print("-" * 105 + "\n")
    return 0


# ---------------------------------------------------------------------------
# Prediction Engine CLI Commands
# ---------------------------------------------------------------------------
def cmd_predict_match(fixture_id: int) -> int:
    """Run all 6 independent statistical models on a single match fixture."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        fixture = repo.get_fixture_by_id(fixture_id)
        if not fixture:
            print(f"[!] Fixture ID {fixture_id} not found in database.")
            return 1

        print(f"\n==================================================================================")
        print(f"  PREDICTION REPORT: {fixture['home_team_name']} vs {fixture['away_team_name']}")
        print(f"  Competition: {fixture['competition_name']} | Date: {fixture['utc_date']}")
        print(f"==================================================================================")

        engine = PredictionEngine(repo)
        results = engine.predict_fixture(fixture_id, persist=True)

        # Show Table of Models across 1X2 Market
        print("\n[Market 1: Match Outcome (1X2) - Across All 6 Independent Models]")
        print(f"{'Model':<16} {'Predicted':<12} {'Confidence':<12} {'Prob (Home / Draw / Away)':<30}")
        print("-" * 75)
        for model_name, markets in results.items():
            pred = markets.get("1X2")
            if not pred:
                continue
            probs = pred.probabilities
            p_str = f"H: {probs.get('HOME_WIN', 0)*100:.1f}% | D: {probs.get('DRAW', 0)*100:.1f}% | A: {probs.get('AWAY_WIN', 0)*100:.1f}%"
            print(f"{model_name:<16} {pred.predicted_outcome:<12} {pred.confidence:<12} {p_str:<30}")

        # Show Consensus Ensemble Breakdown across All 6 Markets
        print("\n[Consensus Ensemble Predictions Across All 6 Prediction Markets]")
        print(f"{'Market':<18} {'Best Pick':<15} {'Probability':<14} {'Confidence':<12}")
        print("-" * 65)
        ensemble = results.get("ensemble", {})
        for market, p in ensemble.items():
            print(f"{market:<18} {p.predicted_outcome:<15} {p.probability*100:>5.1f}%{'':<8} {p.confidence:<12}")

        # Data Quality & Caveats
        ens_1x2 = ensemble.get("1X2")
        if ens_1x2 and ens_1x2.data_quality_notes:
            print("\n[Data Quality & Calibration Notes]")
            for note in ens_1x2.data_quality_notes:
                print(f"  * {note}")

        print("==================================================================================\n")
        return 0
    finally:
        conn.close()


def cmd_predict_fixtures(competition: Optional[str] = None, limit: int = 10) -> int:
    """Run prediction engine on scheduled fixtures."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        comp_id = None
        if competition:
            c = repo.get_competition_by_code_or_external_id(settings.football_api_provider, competition)
            if c:
                comp_id = c.id

        fixtures = repo.get_fixtures(competition_id=comp_id, status="SCHEDULED", limit=limit)
        if not fixtures:
            # Try any status other than finished
            fixtures = [f for f in repo.get_fixtures(competition_id=comp_id, limit=limit) if f["status"] != "FINISHED"]

        if not fixtures:
            print("[*] No upcoming fixtures found. Run 'sync-fixtures' first to fetch match schedules.")
            return 0

        print(f"\n[*] Generating predictions for {len(fixtures)} upcoming fixture(s)...")
        engine = PredictionEngine(repo)

        print(f"{'ID':<5} {'Match':<32} {'1X2 Pick':<12} {'Prob':<8} {'Over/Under 2.5':<16} {'BTTS':<10} {'Confidence':<10}")
        print("-" * 98)
        for f in fixtures:
            preds = engine.predict_fixture(f["id"], persist=True)
            ens = preds.get("ensemble", {})
            p_1x2 = ens.get("1X2")
            p_ou25 = ens.get("OU25")
            p_btts = ens.get("BTTS")

            match_label = f"{f['home_team_name'][:14]} vs {f['away_team_name'][:14]}"
            pick_1x2 = p_1x2.predicted_outcome if p_1x2 else "N/A"
            prob_1x2 = f"{p_1x2.probability*100:.0f}%" if p_1x2 else "-"
            pick_ou = p_ou25.predicted_outcome if p_ou25 else "-"
            pick_btts = p_btts.predicted_outcome if p_btts else "-"
            conf = p_1x2.confidence if p_1x2 else "LOW"

            print(f"{f['id']:<5} {match_label:<32} {pick_1x2:<12} {prob_1x2:<8} {pick_ou:<16} {pick_btts:<10} {conf:<10}")

        print("-" * 98)
        print("[+] All predictions stored in database. Run 'model-stats' to view performance metrics.\n")
        return 0
    finally:
        conn.close()


def cmd_evaluate_predictions() -> int:
    """Evaluate pending predictions against completed match results."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        print("[*] Evaluating pending predictions against finished match scores...")
        stats = repo.evaluate_predictions()
        print(f"[+] Evaluation Complete:")
        print(f"    - Evaluated Predictions: {stats['evaluated']}")
        print(f"    - Correct Predictions  : {stats['correct']}")
        print(f"    - Accuracy Rate        : {stats['accuracy']*100:.1f}%\n")
        return 0
    finally:
        conn.close()


def cmd_model_stats() -> int:
    """Display accuracy leaderboard per model and market."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        metrics = repo.get_accuracy_metrics()
        if not metrics:
            print("[*] No evaluated predictions recorded yet. Run predictions and 'evaluate-predictions' after matches complete.")
            return 0

        print("\n=======================================================")
        print("          ZOLA PREDICT - MODEL ACCURACY LEADERBOARD    ")
        print("=======================================================")
        print(f"{'Model':<16} {'Market':<15} {'Evaluated':<12} {'Correct':<10} {'Accuracy':<10}")
        print("-" * 65)
        for m in metrics:
            print(f"{m.model_name:<16} {m.market:<15} {m.total_predictions:<12} {m.correct_predictions:<10} {m.accuracy_rate*100:>6.1f}%")
        print("=======================================================\n")
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Historical CSV Importer Command
# ---------------------------------------------------------------------------
def cmd_import_csv(filepath: str, season: Optional[str] = None, competition: Optional[str] = None) -> int:
    """Import historical FootballData CSV with provenance tracking."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        importer = HistoricalCsvImporter(repo)
        print(f"[*] Importing historical CSV: {filepath}...")
        res = importer.import_csv(
            file_path_or_content=filepath,
            season=season,
            default_competition_code=competition,
        )
        print(f"[+] Import Successful:")
        print(f"    - Source Repository : {res['source_name']}")
        print(f"    - File Hash (SHA256): {res['file_hash']}")
        print(f"    - Matches Imported  : {res['rows_imported']}")
        print(f"    - Provider Tag      : {res['provider']} (isolated from live data)\n")
        return 0
    except Exception as e:
        print(f"[!] CSV Import Error: {e}")
        return 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# AI Research Command
# ---------------------------------------------------------------------------
def cmd_research_match(fixture_id: int) -> int:
    """Synthesize AI-assisted research breakdown for a match fixture."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        fixture = repo.get_fixture_by_id(fixture_id)
        if not fixture:
            print(f"[!] Fixture ID {fixture_id} not found.")
            return 1

        engine = PredictionEngine(repo)
        preds = engine.predict_fixture(fixture_id, persist=True)

        provider = get_ai_provider(settings)
        print(f"[*] Synthesizing research report using [{provider.name}] provider...\n")
        report = provider.generate_research_summary(fixture, preds)
        print(report)
        print()
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Source Registry Commands (Preserved & Paused)
# ---------------------------------------------------------------------------
def _valid_public_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def cmd_source_add(platform: str, name: str, url: str) -> int:
    """Register a user-selected channel/account/page for later monitoring."""
    if not _valid_public_url(url):
        print("[!] Source URL must be a public HTTPS URL.")
        return 2
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        source_id = repo.add_monitored_source(platform, name.strip(), url.strip())
        print(f"[+] Source saved (ID {source_id}, {platform}). Daily collection is not enabled yet.")
        return 0
    finally:
        conn.close()


def cmd_source_list() -> int:
    """List saved YouTube, TikTok, Telegram, and website sources."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        rows = FootballRepository(conn).list_monitored_sources()
        if not rows:
            print("[*] No user-selected sources saved yet.")
            return 0
        print(f"{'ID':<5} {'PLATFORM':<11} {'STATUS':<8} {'NAME':<24} URL")
        for row in rows:
            print(f"{row['id']:<5} {row['platform']:<11} {row['status']:<8} {row['name'][:23]:<24} {row['url']}")
        return 0
    finally:
        conn.close()


def cmd_source_pause(source_id: int, status: str) -> int:
    """Pause or reactivate a monitoring source."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        changed = FootballRepository(conn).set_monitored_source_status(source_id, status)
        if not changed:
            print(f"[!] No source found with ID {source_id}.")
            return 1
        print(f"[+] Source {source_id} set to {status}.")
        return 0
    finally:
        conn.close()


def cmd_source_items(limit: int = 20) -> int:
    """Display recent source posts saved locally."""
    settings = load_settings()
    init_db(settings.db_path)
    conn = get_connection(settings.db_path)
    try:
        rows = FootballRepository(conn).get_recent_source_items(max(1, min(limit, 100)))
        if not rows:
            print("[*] No source posts collected yet.")
            return 0
        for row in rows:
            when = row.get("published_at") or row.get("collected_at") or "time unknown"
            print(f"\n[{row['platform']}] {row['source_name']} — {when}")
            content = row.get("content_text") or row.get("title") or "(no text)"
            content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", content)
            print(content.replace("\r", " ").replace("\n", " ")[:500])
            print(row["url"])
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Telegram Bot Delivery & Local Dashboard Commands
# ---------------------------------------------------------------------------
def cmd_telegram_chats() -> int:
    """Discover recent chat IDs that have interacted with the Telegram bot."""
    settings = load_settings()
    sender = TelegramReportSender(settings.telegram_bot_token)
    if not sender.is_configured():
        print("[!] Telegram Bot Token is not configured in .env.")
        print("    Add TELEGRAM_BOT_TOKEN=your_token to .env to enable Telegram features.")
        return 1

    bot_info = sender.get_bot_info()
    if bot_info:
        print(f"\n[*] Telegram Bot: @{bot_info.get('username')} (ID: {bot_info.get('id')})")
    else:
        print("\n[!] Could not contact Telegram API. Check your bot token in .env or internet connection.")
        return 1

    print("[*] Checking for recent chat interactions...")
    chats = sender.get_recent_chat_ids()
    if not chats:
        print("[*] No recent chat interactions found.")
        print(f"    Open Telegram, search for @{bot_info.get('username')}, and send '/start' or a message.")
        print("    Then run this command again to see your Chat ID.\n")
        return 0

    print(f"\nFound {len(chats)} recent chat interaction(s):")
    print(f"{'Chat ID':<16} {'Username / Name':<28} {'Type':<12}")
    print("-" * 58)
    for c in chats:
        print(f"{str(c['chat_id']):<16} {str(c.get('username', 'User')):<28} {str(c.get('type', 'private')):<12}")
    print("-" * 58)
    print("\nTip: To set a default chat ID, add this to your .env file:")
    print(f"TELEGRAM_CHAT_ID={chats[0]['chat_id']}\n")
    return 0


def cmd_send_report(fixture_id: int, chat_id: Optional[str] = None) -> int:
    """Send formatted prediction match card to a Telegram chat."""
    settings = load_settings()
    init_db(settings.db_path)
    target_chat_id = chat_id or settings.telegram_chat_id

    if not target_chat_id:
        print("[!] Missing chat ID.")
        print("    Specify --chat-id <id> or set TELEGRAM_CHAT_ID in .env.")
        print("    Run 'telegram-chats' to find your chat ID.")
        return 1

    sender = TelegramReportSender(settings.telegram_bot_token)
    if not sender.is_configured():
        print("[!] Telegram Bot Token is not configured in .env.")
        return 1

    conn = get_connection(settings.db_path)
    try:
        repo = FootballRepository(conn)
        print(f"[*] Generating prediction and delivering report for fixture #{fixture_id} to chat {target_chat_id}...")
        ok = sender.send_fixture_report(chat_id=target_chat_id, fixture_id=fixture_id, repository=repo)
        if ok:
            print(f"[+] Match report successfully delivered to Telegram chat {target_chat_id}!\n")
            return 0
        else:
            print(f"[!] Failed to deliver match report to Telegram. Check bot permissions.\n")
            return 1
    except Exception as e:
        print(f"[!] Error: {e}")
        return 1
    finally:
        conn.close()


def cmd_test_telegram(chat_id: Optional[str] = None) -> int:
    """Send a test greeting to verify Telegram bot setup."""
    settings = load_settings()
    target_chat_id = chat_id or settings.telegram_chat_id

    if not target_chat_id:
        print("[!] Missing chat ID.")
        print("    Specify --chat-id <id> or set TELEGRAM_CHAT_ID in .env.")
        print("    Run 'telegram-chats' to find your chat ID.")
        return 1

    sender = TelegramReportSender(settings.telegram_bot_token)
    if not sender.is_configured():
        print("[!] Telegram Bot Token is not configured in .env.")
        return 1

    print(f"[*] Sending test greeting to Telegram chat {target_chat_id}...")
    if sender.send_test_message(chat_id=target_chat_id):
        print(f"[+] Test message delivered successfully to chat {target_chat_id}!\n")
        return 0
    else:
        print(f"[!] Failed to send test message. Check bot permissions.\n")
        return 1


def cmd_dashboard(host: str = "127.0.0.1", port: int = 8080, no_browser: bool = False) -> int:
    """Launch the local web dashboard."""
    settings = load_settings()
    try:
        run_dashboard(host=host, port=port, open_browser=not no_browser, settings=settings)
        return 0
    except Exception as e:
        print(f"[!] Dashboard Error: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="zola",
        description="Zola Predict - Local-first football research and prediction assistant (Windows 11)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Core System Commands
    subparsers.add_parser("check-env", help="Check environment variables safely (never exposes secrets)")
    subparsers.add_parser("init-db", help="Initialize SQLite database schema and tables")
    subparsers.add_parser("status", help="Show database summary statistics and recent API logs")

    # Football Data Synchronization
    subparsers.add_parser("sync-competitions", help="Fetch and store competitions from the football API")
    sync_fix = subparsers.add_parser("sync-fixtures", help="Fetch upcoming fixtures and store idempotently")
    sync_fix.add_argument("--competition", "-c", default="39", help="Competition code or ID, or 'all' (default: 39 for Premier League)")
    sync_fix.add_argument("--status", "-s", default="SCHEDULED", help="Fixture status filter (e.g. SCHEDULED)")
    sync_fix.add_argument("--date", "-d", default=None, help="Match date filter (YYYY-MM-DD, 'today', or 'tomorrow')")
    sync_fix.add_argument("--season", default=None, help="Season year filter (e.g. 2026)")
    sync_fix.add_argument("--all", "-a", action="store_true", help="Sync across all active competitions")
    sync_fix.add_argument("--days-ahead", type=int, default=None, help="Number of days ahead to sync (e.g. 7-10 days)")
    sync_res = subparsers.add_parser("sync-results", help="Fetch historical results and match scores")
    sync_res.add_argument("--competition", "-c", default="39", help="Competition code or ID (API-Football Premier League ID: 39)")
    sync_res.add_argument("--limit", "-l", type=int, default=20, help="Maximum number of finished matches to fetch")
    list_fix = subparsers.add_parser("list-fixtures", help="Display fixtures stored in local database")
    list_fix.add_argument("--competition", "-c", default=None, help="Filter by competition code or ID")
    list_fix.add_argument("--status", "-s", default=None, help="Filter by fixture status (SCHEDULED, FINISHED)")
    list_fix.add_argument("--upcoming", "-u", action="store_true", help="Show only genuine upcoming scheduled fixtures")
    list_fix.add_argument("--limit", "-l", type=int, default=15, help="Number of fixtures to display")

    # Prediction Engine Commands
    p_match = subparsers.add_parser("predict-match", help="Run 6 independent models and consensus on a fixture ID")
    p_match.add_argument("--id", type=int, required=True, help="Fixture database ID")
    p_fixs = subparsers.add_parser("predict-fixtures", help="Run prediction engine on upcoming matches")
    p_fixs.add_argument("--competition", "-c", default=None, help="Filter by competition code or ID")
    p_fixs.add_argument("--limit", "-l", type=int, default=10, help="Number of fixtures to predict")
    subparsers.add_parser("evaluate-predictions", help="Evaluate predictions against actual match results")
    subparsers.add_parser("model-stats", help="Show accuracy leaderboard across models and markets")

    # Historical CSV Data Import
    csv_cmd = subparsers.add_parser("import-csv", help="Import historical FootballData CSV with provenance")
    csv_cmd.add_argument("--file", "-f", required=True, help="Path to CSV file (e.g., E0.csv)")
    csv_cmd.add_argument("--season", "-s", default=None, help="Season label (e.g., 2023-2024)")
    csv_cmd.add_argument("--competition", "-c", default="E0", help="Division code (e.g., E0 for Premier League)")

    # AI Research Summary
    res_cmd = subparsers.add_parser("research-match", help="Generate AI-assisted football research report")
    res_cmd.add_argument("--id", type=int, required=True, help="Fixture database ID")

    # Generic Monitored Source Registry (Deferred features)
    source_add = subparsers.add_parser("source-add", help="Save a YouTube, TikTok, Telegram, or website source")
    source_add.add_argument("--platform", choices=["youtube", "tiktok", "telegram", "website"], required=True)
    source_add.add_argument("--name", required=True, help="A label for this source")
    source_add.add_argument("--url", required=True, help="Public HTTPS channel, account, or page URL")
    subparsers.add_parser("source-list", help="List saved user-selected research sources")
    source_state = subparsers.add_parser("source-status", help="Pause or reactivate a saved source")
    source_state.add_argument("--id", type=int, required=True)
    source_state.add_argument("--status", choices=["ACTIVE", "PAUSED"], required=True)
    feed = subparsers.add_parser("source-items", help="Display recently collected source posts")
    feed.add_argument("--limit", type=int, default=20, help="Maximum posts to display (1-100)")

    # Telegram Bot Delivery Commands
    subparsers.add_parser("telegram-chats", help="Discover recent chats interacting with @ZolaPredict_bot")
    tg_send = subparsers.add_parser("send-report", help="Send match prediction report to Telegram")
    tg_send.add_argument("--id", type=int, required=True, help="Fixture database ID")
    tg_send.add_argument("--chat-id", default=None, help="Telegram chat ID (or set TELEGRAM_CHAT_ID in .env)")
    tg_test = subparsers.add_parser("test-telegram", help="Send test greeting to Telegram chat")
    tg_test.add_argument("--chat-id", default=None, help="Telegram chat ID (or set TELEGRAM_CHAT_ID in .env)")

    # Local Web Dashboard
    dash_cmd = subparsers.add_parser("dashboard", help="Start local web dashboard on Windows 11")
    dash_cmd.add_argument("--host", default="127.0.0.1", help="Binding host address (default: 127.0.0.1)")
    dash_cmd.add_argument("--port", "-p", type=int, default=8080, help="Port to listen on (default: 8080)")
    dash_cmd.add_argument("--no-browser", action="store_true", help="Do not automatically launch web browser")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "check-env":
        return cmd_check_env()
    elif args.command == "init-db":
        return cmd_init_db()
    elif args.command == "status":
        return cmd_status()
    elif args.command == "sync-competitions":
        return cmd_sync_competitions()
    elif args.command == "sync-fixtures":
        return cmd_sync_fixtures(
            competition=args.competition,
            status=args.status,
            date=args.date,
            season=args.season,
            sync_all=getattr(args, "all", False),
            days_ahead=getattr(args, "days_ahead", None),
        )
    elif args.command == "sync-results":
        return cmd_sync_results(competition=args.competition, limit=args.limit)
    elif args.command == "list-fixtures":
        return cmd_list_fixtures(
            competition=args.competition,
            status=args.status,
            upcoming_only=getattr(args, "upcoming", False),
            limit=args.limit,
        )
    elif args.command == "predict-match":
        return cmd_predict_match(fixture_id=args.id)
    elif args.command == "predict-fixtures":
        return cmd_predict_fixtures(competition=args.competition, limit=args.limit)
    elif args.command == "evaluate-predictions":
        return cmd_evaluate_predictions()
    elif args.command == "model-stats":
        return cmd_model_stats()
    elif args.command == "import-csv":
        return cmd_import_csv(filepath=args.file, season=args.season, competition=args.competition)
    elif args.command == "research-match":
        return cmd_research_match(fixture_id=args.id)
    elif args.command == "telegram-chats":
        return cmd_telegram_chats()
    elif args.command == "send-report":
        return cmd_send_report(fixture_id=args.id, chat_id=args.chat_id)
    elif args.command == "test-telegram":
        return cmd_test_telegram(chat_id=args.chat_id)
    elif args.command == "dashboard":
        return cmd_dashboard(host=args.host, port=args.port, no_browser=args.no_browser)
    elif args.command == "source-add":
        return cmd_source_add(platform=args.platform, name=args.name, url=args.url)
    elif args.command == "source-list":
        return cmd_source_list()
    elif args.command == "source-status":
        return cmd_source_pause(source_id=args.id, status=args.status)
    elif args.command == "source-items":
        return cmd_source_items(limit=args.limit)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
