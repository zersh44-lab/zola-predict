"""Execute walk-forward backtest across historical fixtures, populating predictions and accuracy metrics."""

import json
import sqlite3
from datetime import datetime, timezone
from zola.database.connection import get_connection
from zola.database.repository import FootballRepository
from zola.database.models import Prediction
from zola.prediction.poisson import PoissonModel
from zola.prediction.dixon_coles import DixonColesModel
from zola.prediction.elo import EloRatingModel
from zola.prediction.form_momentum import FormMomentumModel
from zola.prediction.head_to_head import HeadToHeadModel
from zola.prediction.ensemble import ConsensusEnsembleModel


def run_comprehensive_backtest(matches_per_league: int = 30):
    conn = get_connection("data/zola_predict.db")
    repo = FootballRepository(conn)
    cursor = repo.conn.cursor()

    models = [
        PoissonModel(),
        DixonColesModel(),
        EloRatingModel(),
        FormMomentumModel(),
        HeadToHeadModel(),
    ]
    ensemble = ConsensusEnsembleModel()

    # Get target leagues
    cursor.execute("""
        SELECT c.id, c.name, COUNT(f.id) as match_count
        FROM competitions c
        JOIN fixtures f ON c.id = f.competition_id
        JOIN match_results mr ON f.id = mr.fixture_id
        WHERE mr.full_time_home_score IS NOT NULL
        GROUP BY c.id, c.name
        HAVING match_count >= 50
        ORDER BY match_count DESC;
    """)
    target_leagues = cursor.fetchall()
    print(f"Found {len(target_leagues)} major competitions with >= 50 finished matches:")
    for lg in target_leagues:
        print(f"  - {lg['name']} (ID: {lg['id']}): {lg['match_count']} finished matches")

    total_fixtures_processed = 0
    total_preds_created = 0

    for lg in target_leagues:
        c_id = lg["id"]
        c_name = lg["name"]
        print(f"\nProcessing {c_name} (target: {matches_per_league} matches)...")

        cursor.execute("""
            SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id, f.competition_id,
                   mr.full_time_home_score, mr.full_time_away_score, mr.winner,
                   ht.name as home_name, at.name as away_name
            FROM fixtures f
            JOIN match_results mr ON f.id = mr.fixture_id
            JOIN teams ht ON f.home_team_id = ht.id
            JOIN teams at ON f.away_team_id = at.id
            WHERE f.status = 'FINISHED'
              AND mr.full_time_home_score IS NOT NULL
              AND f.competition_id = ?
            ORDER BY f.utc_date DESC
            LIMIT ?;
        """, (c_id, matches_per_league * 2))
        candidate_fixtures = cursor.fetchall()

        league_processed = 0
        for fix in candidate_fixtures:
            if league_processed >= matches_per_league:
                break

            fix_id = fix["id"]
            h_id = fix["home_team_id"]
            a_id = fix["away_team_id"]
            m_date = fix["utc_date"]
            h_score = fix["full_time_home_score"]
            a_score = fix["full_time_away_score"]

            # Only pre-match history
            cursor.execute("""
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score,
                       CASE WHEN f.home_team_id = ? THEN 1 ELSE 0 END as is_home
                FROM fixtures f JOIN match_results mr ON f.id = mr.fixture_id
                WHERE (f.home_team_id = ? OR f.away_team_id = ?) AND f.utc_date < ?
                ORDER BY f.utc_date DESC LIMIT 20;
            """, (h_id, h_id, h_id, m_date))
            home_matches = [dict(r) for r in cursor.fetchall()]

            cursor.execute("""
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score,
                       CASE WHEN f.home_team_id = ? THEN 1 ELSE 0 END as is_home
                FROM fixtures f JOIN match_results mr ON f.id = mr.fixture_id
                WHERE (f.home_team_id = ? OR f.away_team_id = ?) AND f.utc_date < ?
                ORDER BY f.utc_date DESC LIMIT 20;
            """, (a_id, a_id, a_id, m_date))
            away_matches = [dict(r) for r in cursor.fetchall()]

            if len(home_matches) < 3 or len(away_matches) < 3:
                continue

            cursor.execute("""
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score
                FROM fixtures f JOIN match_results mr ON f.id = mr.fixture_id
                WHERE ((f.home_team_id = ? AND f.away_team_id = ?) OR (f.home_team_id = ? AND f.away_team_id = ?))
                  AND f.utc_date < ? ORDER BY f.utc_date DESC LIMIT 10;
            """, (h_id, a_id, a_id, h_id, m_date))
            h2h_matches = [dict(r) for r in cursor.fetchall()]

            cursor.execute("""
                SELECT f.id, f.utc_date, f.home_team_id, f.away_team_id,
                       mr.full_time_home_score, mr.full_time_away_score
                FROM fixtures f JOIN match_results mr ON f.id = mr.fixture_id
                WHERE f.competition_id = ? AND f.utc_date < ? ORDER BY f.utc_date DESC LIMIT 50;
            """, (c_id, m_date))
            league_matches = [dict(r) for r in cursor.fetchall()]

            fix_dict = dict(fix)
            fix_dict["competition_name"] = c_name

            fix_preds = {}
            for m in models:
                try:
                    fix_preds[m.name] = m.predict(fix_dict, home_matches, away_matches, h2h_matches, league_matches)
                except Exception:
                    pass

            if not fix_preds:
                continue

            ens_preds = ensemble.combine(fix_preds)
            fix_preds["ensemble"] = ens_preds

            # Save predictions with model_version = 'v1.1-upgraded'
            for m_name, market_dict in fix_preds.items():
                for mkt, pred_obj in market_dict.items():
                    p_model = Prediction(
                        id=None,
                        fixture_id=fix_id,
                        model_name=m_name,
                        market=mkt,
                        predicted_outcome=pred_obj.predicted_outcome,
                        probability=pred_obj.probability,
                        probabilities_json=json.dumps(pred_obj.probabilities),
                        confidence=pred_obj.confidence,
                        confidence_score=pred_obj.confidence_score,
                        data_quality_notes=json.dumps(pred_obj.data_quality_notes or []),
                        features_json=json.dumps(pred_obj.features or {}),
                        status="PENDING",
                        is_correct=None,
                        settlement_status="PENDING",
                        model_version="v1.1-upgraded",
                    )
                    repo.upsert_prediction(p_model)
                    total_preds_created += 1

            league_processed += 1
            total_fixtures_processed += 1

        print(f"  Processed {league_processed} matches for {c_name}.")

    print(f"\nDone generating predictions! Total fixtures: {total_fixtures_processed}, total predictions: {total_preds_created}")
    print("Evaluating all pending predictions against official final scores...")
    settlement_stats = repo.evaluate_predictions()
    print(f"Settlement results: {settlement_stats}")

    print("\nRetrieving updated accuracy metrics from repository...")
    metrics = repo.get_accuracy_metrics()
    print(f"Total metrics rows: {len(metrics)}")
    conn.close()


if __name__ == "__main__":
    run_comprehensive_backtest(matches_per_league=25)
