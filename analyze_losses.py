import sqlite3

conn = sqlite3.connect('data/zola_predict.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

# 0. Canonicalize match_results winner
c.execute("""
    UPDATE match_results
    SET winner = CASE
        WHEN full_time_home_score > full_time_away_score THEN 'HOME_TEAM'
        WHEN full_time_away_score > full_time_home_score THEN 'AWAY_TEAM'
        WHEN full_time_home_score = full_time_away_score THEN 'DRAW'
        WHEN UPPER(winner) IN ('HOME', 'HOME_TEAM') THEN 'HOME_TEAM'
        WHEN UPPER(winner) IN ('AWAY', 'AWAY_TEAM') THEN 'AWAY_TEAM'
        WHEN UPPER(winner) IN ('DRAW', 'TIE') THEN 'DRAW'
        ELSE winner
    END
    WHERE full_time_home_score IS NOT NULL;
""")
conn.commit()

# 1. Total Losses by Market across ALL models
c.execute("""
    SELECT market,
           COUNT(*) as total,
           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as losses,
           SUM(CASE WHEN settlement_status = 'PUSH' THEN 1 ELSE 0 END) as pushes
    FROM predictions
    WHERE is_correct IS NOT NULL
    GROUP BY market
    ORDER BY losses DESC;
""")
all_losses = c.fetchall()

print("=== TOTAL LOSSES BY MARKET (ALL MODELS COMBINED) ===")
print(f"{'Market':<15} | {'Losses':<6} | {'Wins':<6} | {'Loss Rate':<10} | {'Win Rate':<10} | {'Total':<6}")
print("-" * 65)
for r in all_losses:
    dec = r['wins'] + r['losses']
    loss_rate = (r['losses'] / dec * 100) if dec else 0
    win_rate = (r['wins'] / dec * 100) if dec else 0
    print(f"{r['market']:<15} | {r['losses']:<6} | {r['wins']:<6} | {loss_rate:>8.1f}% | {win_rate:>8.1f}% | {dec:<6}")

# 2. Losses by Market for ENSEMBLE
c.execute("""
    SELECT market,
           COUNT(*) as total,
           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as losses,
           SUM(CASE WHEN settlement_status = 'PUSH' THEN 1 ELSE 0 END) as pushes
    FROM predictions
    WHERE model_name = 'ensemble' AND is_correct IS NOT NULL
    GROUP BY market
    ORDER BY losses DESC;
""")
ens_losses = c.fetchall()

print("\n=== LOSSES BY MARKET (CONSENSUS ENSEMBLE ONLY) ===")
print(f"{'Market':<15} | {'Losses':<6} | {'Wins':<6} | {'Loss Rate':<10} | {'Win Rate':<10} | {'Total':<6}")
print("-" * 65)
for r in ens_losses:
    dec = r['wins'] + r['losses']
    loss_rate = (r['losses'] / dec * 100) if dec else 0
    win_rate = (r['wins'] / dec * 100) if dec else 0
    print(f"{r['market']:<15} | {r['losses']:<6} | {r['wins']:<6} | {loss_rate:>8.1f}% | {win_rate:>8.1f}% | {dec:<6}")

# 3. Deep Dive: Failure modes in 1X2
print("\n=== 1X2 MARKET LOSS BREAKDOWN (ENSEMBLE) ===")
c.execute("""
    SELECT p.predicted_outcome,
           mr.winner as actual_winner,
           COUNT(*) as count
    FROM predictions p
    JOIN fixtures f ON p.fixture_id = f.id
    JOIN match_results mr ON f.id = mr.fixture_id
    WHERE p.model_name = 'ensemble' AND p.market = '1X2' AND p.is_correct = 0
    GROUP BY p.predicted_outcome, mr.winner
    ORDER BY count DESC;
""")
for r in c.fetchall():
    print(f"Predicted: {r['predicted_outcome']:<10} -> But actual winner was: {r['actual_winner']:<10} (Lost {r['count']} times)")

# 4. BTTS breakdown
print("\n=== BTTS LOSS BREAKDOWN (ENSEMBLE) ===")
c.execute("""
    SELECT p.predicted_outcome,
           COUNT(*) as count
    FROM predictions p
    JOIN fixtures f ON p.fixture_id = f.id
    JOIN match_results mr ON f.id = mr.fixture_id
    WHERE p.model_name = 'ensemble' AND p.market = 'BTTS' AND p.is_correct = 0
    GROUP BY p.predicted_outcome
    ORDER BY count DESC;
""")
for r in c.fetchall():
    print(f"Predicted: {r['predicted_outcome']} -> Lost {r['count']} times")

# 5. OU25 breakdown
print("\n=== OU25 LOSS BREAKDOWN (ENSEMBLE) ===")
c.execute("""
    SELECT p.predicted_outcome,
           COUNT(*) as count
    FROM predictions p
    JOIN fixtures f ON p.fixture_id = f.id
    JOIN match_results mr ON f.id = mr.fixture_id
    WHERE p.model_name = 'ensemble' AND p.market = 'OU25' AND p.is_correct = 0
    GROUP BY p.predicted_outcome
    ORDER BY count DESC;
""")
for r in c.fetchall():
    print(f"Predicted: {r['predicted_outcome']} -> Lost {r['count']} times")

# 6. Confidence tier breakdown for losses
print("\n=== LOSSES BY CONFIDENCE TIER (ENSEMBLE) ===")
c.execute("""
    SELECT confidence,
           COUNT(*) as total,
           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as losses
    FROM predictions
    WHERE model_name = 'ensemble' AND is_correct IS NOT NULL
    GROUP BY confidence
    ORDER BY losses DESC;
""")
for r in c.fetchall():
    dec = r['wins'] + r['losses']
    l_rate = (r['losses'] / dec * 100) if dec else 0
    w_rate = (r['wins'] / dec * 100) if dec else 0
    print(f"Confidence: {r['confidence']:<8} | Losses: {r['losses']:<3} | Wins: {r['wins']:<3} | Loss Rate: {l_rate:.1f}% | Win Rate: {w_rate:.1f}%")

# 7. Leagues
print("\n=== LOSSES BY LEAGUE / COMPETITION (ENSEMBLE) ===")
c.execute("""
    SELECT c.name as comp_name,
           COUNT(*) as total,
           SUM(CASE WHEN p.is_correct = 0 THEN 1 ELSE 0 END) as losses,
           SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as wins
    FROM predictions p
    JOIN fixtures f ON p.fixture_id = f.id
    JOIN competitions c ON f.competition_id = c.id
    WHERE p.model_name = 'ensemble' AND p.is_correct IS NOT NULL
    GROUP BY c.name
    ORDER BY losses DESC;
""")
for r in c.fetchall():
    dec = r['wins'] + r['losses']
    l_rate = (r['losses'] / dec * 100) if dec else 0
    w_rate = (r['wins'] / dec * 100) if dec else 0
    print(f"{r['comp_name']:<35} | Losses: {r['losses']:<3} | Wins: {r['wins']:<3} | Loss Rate: {l_rate:.1f}%")
