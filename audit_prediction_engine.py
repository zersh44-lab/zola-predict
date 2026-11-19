import json
from zola.database.connection import get_connection

def run_audit():
    conn = get_connection("data/zola_predict.db")
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in c.fetchall()]
    print("=== TABLES IN DB ===")
    print(tables)
    print("\n=== ROW COUNTS ===")
    for t in ['fixtures', 'match_results', 'predictions', 'accuracy_metrics', 'teams', 'competitions']:
        if t in tables:
            cnt = c.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
            print(f"{t}: {cnt}")
            
    # Sample predictions
    c.execute("SELECT id, fixture_id, model_name, market, predicted_outcome, probability, confidence, confidence_score, status, is_correct FROM predictions LIMIT 10;")
    preds = [dict(r) for r in c.fetchall()]
    print("\n=== SAMPLE PREDICTIONS ===")
    print(json.dumps(preds, indent=2))
    
    # Check fixtures with results
    c.execute("""
        SELECT f.id, f.utc_date, ht.name as home, at.name as away, mr.full_time_home_score as fth, mr.full_time_away_score as fta
        FROM fixtures f
        JOIN teams ht ON f.home_team_id = ht.id
        JOIN teams at ON f.away_team_id = at.id
        JOIN match_results mr ON f.id = mr.fixture_id
        LIMIT 5;
    """)
    finished = [dict(r) for r in c.fetchall()]
    print("\n=== SAMPLE FINISHED FIXTURES WITH RESULTS ===")
    print(json.dumps(finished, indent=2))

    # Check accuracy metrics
    c.execute("SELECT * FROM accuracy_metrics;")
    metrics = [dict(r) for r in c.fetchall()]
    print("\n=== ACCURACY METRICS ===")
    print(json.dumps(metrics, indent=2))
    
    conn.close()

if __name__ == "__main__":
    run_audit()
