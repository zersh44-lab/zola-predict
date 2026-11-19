import sqlite3

conn = sqlite3.connect('data/zola_predict.db')
conn.row_factory = sqlite3.Row
c = conn.cursor()

c.execute('''
    SELECT c.name as comp_name, COUNT(f.id) as match_count, MIN(f.utc_date) as min_date, MAX(f.utc_date) as max_date
    FROM fixtures f
    JOIN match_results mr ON f.id = mr.fixture_id
    JOIN competitions c ON f.competition_id = c.id
    WHERE mr.full_time_home_score IS NOT NULL
    GROUP BY c.name
    ORDER BY match_count DESC;
''')
for r in c.fetchall():
    print(f"{r['comp_name'][:40]:<40} {r['match_count']:<6} ({r['min_date'][:10]} to {r['max_date'][:10]})")
