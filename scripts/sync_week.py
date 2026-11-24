import os, sys, time
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timezone, timedelta
from zola.config import load_settings
from zola.client.api_sports import ApiSportsClient
from zola.client.cache import RequestCache
from zola.client.rate_limiter import RateLimiter
from zola.database.connection import get_connection
from zola.database.repository import FootballRepository
from zola.services.collector import FootballCollectorService
from zola.prediction.engine import PredictionEngine

settings = load_settings()
conn = get_connection(settings.db_path)
repo = FootballRepository(conn)
cache = RequestCache(conn)
limiter = RateLimiter(max_rpm=10)
client = ApiSportsClient(api_key=settings.football_api_key, cache=cache, rate_limiter=limiter)
collector = FootballCollectorService(client=client, repository=repo)

start_dt = datetime.now(timezone.utc)
dates_to_sync = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(8)]
print(f"Target sync dates: {dates_to_sync}")

total_synced = 0
for dt in dates_to_sync:
    print(f"--- Syncing {dt} ---")
    try:
        res = collector.sync_fixtures(competition_code_or_id="all", date=dt, status=None)
        count = res.get("fixtures_synced", 0)
        total_synced += count
        print(f"Date {dt}: synced {count} fixtures (cache_hit={res.get('cache_hit')})")
    except Exception as e:
        print(f"Error syncing {dt}: {e}")

print(f"Total fixtures synced: {total_synced}")

# Now generate predictions for all upcoming fixtures
print("--- Generating Predictions for Upcoming Fixtures ---")
engine = PredictionEngine(repo)
pred_count = engine.predict_upcoming_fixtures(days_ahead=8, limit=500)
print(f"Total predictions generated: {pred_count}")

# Settle any finished fixtures
print("--- Evaluating / Settling Concluded Matches ---")
eval_res = repo.evaluate_predictions()
print(f"Evaluation: {eval_res.get('stats')}")

stats = repo.get_database_stats()
print(f"Final DB Stats: {stats}")
conn.close()
