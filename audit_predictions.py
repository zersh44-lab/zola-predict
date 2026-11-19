import sqlite3
import json

conn = sqlite3.connect('data/zola_predict.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute('SELECT COUNT(*) FROM fixtures;')
tot_fixtures = c.fetchone()[0]

c.execute('SELECT COUNT(*) FROM match_results WHERE full_time_home_score IS NOT NULL;')
tot_results = c.fetchone()[0]

c.execute('SELECT COUNT(*) FROM predictions;')
tot_preds = c.fetchone()[0]

c.execute('SELECT COUNT(*) FROM predictions WHERE is_correct IS NOT NULL;')
tot_evaluated = c.fetchone()[0]

print(f"Total fixtures: {tot_fixtures}")
print(f"Total match results: {tot_results}")
print(f"Total predictions: {tot_preds}")
print(f"Total evaluated predictions: {tot_evaluated}")

def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            widths[i] = max(widths[i], len(str(val)))
    header_str = " | ".join(f"{h:<{widths[i]}}" for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    print(header_str)
    print(sep_str)
    for r in rows:
        print(" | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(r)))

# 1. Performance by model and market
c.execute('''
    SELECT model_name, market,
           COUNT(*) as total,
           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
           SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect,
           SUM(CASE WHEN settlement_status = 'PUSH' THEN 1 ELSE 0 END) as pushes
    FROM predictions
    WHERE is_correct IS NOT NULL OR settlement_status = 'PUSH'
    GROUP BY model_name, market
    ORDER BY market, model_name;
''')
rows = [dict(r) for r in c.fetchall()]
table_data = []
for r in rows:
    decided = r["correct"] + r["incorrect"]
    acc = (r["correct"] / decided * 100) if decided > 0 else 0.0
    table_data.append([
        r["market"], r["model_name"], r["total"], r["correct"], r["incorrect"], r["pushes"], f"{acc:.1f}%"
    ])

print("\n=== PERFORMANCE BY MARKET & MODEL ===")
print_table(["Market", "Model", "Total", "Correct", "Incorrect", "Pushes", "Accuracy"], table_data)

# 2. Performance by Confidence Range (for Ensemble)
c.execute('''
    SELECT confidence,
           COUNT(*) as total,
           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
           SUM(CASE WHEN is_correct = 0 THEN 1 ELSE 0 END) as incorrect
    FROM predictions
    WHERE model_name = 'ensemble' AND is_correct IS NOT NULL
    GROUP BY confidence;
''')
conf_rows = [dict(r) for r in c.fetchall()]
conf_table = []
for r in conf_rows:
    dec = r["correct"] + r["incorrect"]
    acc = (r["correct"] / dec * 100) if dec > 0 else 0.0
    conf_table.append([r["confidence"], r["total"], r["correct"], r["incorrect"], f"{acc:.1f}%"])

print("\n=== ENSEMBLE PERFORMANCE BY CONFIDENCE TIER ===")
print_table(["Confidence", "Total", "Correct", "Incorrect", "Accuracy"], conf_table)

# 3. Performance by Competition (Ensemble)
c.execute('''
    SELECT c.name as comp_name,
           COUNT(p.id) as total,
           SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct,
           SUM(CASE WHEN p.is_correct = 0 THEN 1 ELSE 0 END) as incorrect
    FROM predictions p
    JOIN fixtures f ON p.fixture_id = f.id
    JOIN competitions c ON f.competition_id = c.id
    WHERE p.model_name = 'ensemble' AND p.is_correct IS NOT NULL
    GROUP BY c.name
    ORDER BY total DESC;
''')
comp_rows = [dict(r) for r in c.fetchall()]
comp_table = []
for r in comp_rows:
    dec = r["correct"] + r["incorrect"]
    acc = (r["correct"] / dec * 100) if dec > 0 else 0.0
    comp_table.append([r["comp_name"][:35], r["total"], r["correct"], r["incorrect"], f"{acc:.1f}%"])

print("\n=== ENSEMBLE PERFORMANCE BY COMPETITION ===")
print_table(["Competition", "Total", "Correct", "Incorrect", "Accuracy"], comp_table)

# 4. Draw Analysis & 1X2 Bias
c.execute('''
    SELECT mr.winner,
           COUNT(*) as count
    FROM fixtures f
    JOIN match_results mr ON f.id = mr.fixture_id
    WHERE mr.full_time_home_score IS NOT NULL
    GROUP BY mr.winner;
''')
actual_winners = {r["winner"]: r["count"] for r in c.fetchall()}
print("\n=== ACTUAL OUTCOMES IN EVALUATED MATCHES ===")
print(actual_winners)

c.execute('''
    SELECT predicted_outcome,
           COUNT(*) as total,
           SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct
    FROM predictions
    WHERE model_name = 'ensemble' AND market = '1X2' AND is_correct IS NOT NULL
    GROUP BY predicted_outcome;
''')
pred_1x2 = [dict(r) for r in c.fetchall()]
pred_table = []
for r in pred_1x2:
    acc = (r["correct"] / r["total"] * 100) if r["total"] > 0 else 0.0
    pred_table.append([r["predicted_outcome"], r["total"], r["correct"], f"{acc:.1f}%"])
print("\n=== ENSEMBLE 1X2 PREDICTED OUTCOMES BREAKDOWN ===")
print_table(["Outcome", "Times Predicted", "Correct", "Accuracy"], pred_table)

# 5. Over/Under Analysis (Total Goals Predicted vs Actual)
c.execute('''
    SELECT p.market, p.predicted_outcome,
           COUNT(*) as total,
           SUM(CASE WHEN p.is_correct = 1 THEN 1 ELSE 0 END) as correct
    FROM predictions p
    WHERE p.model_name = 'ensemble' AND p.market IN ('OU05', 'OU15', 'OU25', 'OU35') AND p.is_correct IS NOT NULL
    GROUP BY p.market, p.predicted_outcome;
''')
ou_rows = [dict(r) for r in c.fetchall()]
ou_table = []
for r in ou_rows:
    acc = (r["correct"] / r["total"] * 100) if r["total"] > 0 else 0.0
    ou_table.append([r["market"], r["predicted_outcome"], r["total"], r["correct"], f"{acc:.1f}%"])
print("\n=== ENSEMBLE OVER/UNDER BREAKDOWN ===")
print_table(["Market", "Pick", "Total", "Correct", "Accuracy"], ou_table)
