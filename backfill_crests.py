"""Backfill missing team crests and country flags in SQLite database."""

import sqlite3
from zola.utils.flags import COUNTRY_CODE_MAP, get_country_flag_url

conn = sqlite3.connect("data/zola_predict.db")
cursor = conn.cursor()

# 1. Total teams before
cursor.execute("SELECT COUNT(*), COUNT(CASE WHEN crest_url IS NOT NULL AND crest_url != '' THEN 1 END) FROM teams;")
total, with_crest = cursor.fetchone()
print(f"Before: {with_crest}/{total} teams have crests ({total - with_crest} missing)")

# 2. Backfill from duplicate teams with same name that have crest_url
cursor.execute("""
    UPDATE teams
    SET crest_url = (
        SELECT t2.crest_url
        FROM teams t2
        WHERE LOWER(TRIM(t2.name)) = LOWER(TRIM(teams.name))
          AND t2.crest_url IS NOT NULL
          AND t2.crest_url != ''
        LIMIT 1
    )
    WHERE (crest_url IS NULL OR crest_url = '')
      AND EXISTS (
        SELECT 1
        FROM teams t2
        WHERE LOWER(TRIM(t2.name)) = LOWER(TRIM(teams.name))
          AND t2.crest_url IS NOT NULL
          AND t2.crest_url != ''
      );
""")
print(f"Updated from duplicate team rows: {cursor.rowcount}")

# 3. Backfill national teams from COUNTRY_CODE_MAP
national_updated = 0
for country_name, code in COUNTRY_CODE_MAP.items():
    flag_url = f"https://flagcdn.com/w80/{code}.png"
    cursor.execute("""
        UPDATE teams
        SET crest_url = ?
        WHERE (crest_url IS NULL OR crest_url = '')
          AND LOWER(TRIM(name)) = ?;
    """, (flag_url, country_name))
    national_updated += cursor.rowcount

print(f"Updated national teams from COUNTRY_CODE_MAP: {national_updated}")

# 4. Check known club names that might be missing crests
COMMON_CLUB_CRESTS = {
    "arsenal": "https://media.api-sports.io/football/teams/42.png",
    "chelsea": "https://media.api-sports.io/football/teams/49.png",
    "liverpool": "https://media.api-sports.io/football/teams/40.png",
    "manchester city": "https://media.api-sports.io/football/teams/50.png",
    "man city": "https://media.api-sports.io/football/teams/50.png",
    "manchester united": "https://media.api-sports.io/football/teams/33.png",
    "man united": "https://media.api-sports.io/football/teams/33.png",
    "tottenham": "https://media.api-sports.io/football/teams/47.png",
    "tottenham hotspur": "https://media.api-sports.io/football/teams/47.png",
    "newcastle": "https://media.api-sports.io/football/teams/34.png",
    "newcastle united": "https://media.api-sports.io/football/teams/34.png",
    "aston villa": "https://media.api-sports.io/football/teams/66.png",
    "brighton": "https://media.api-sports.io/football/teams/51.png",
    "west ham": "https://media.api-sports.io/football/teams/48.png",
    "west ham united": "https://media.api-sports.io/football/teams/48.png",
    "everton": "https://media.api-sports.io/football/teams/45.png",
    "fulham": "https://media.api-sports.io/football/teams/36.png",
    "brentford": "https://media.api-sports.io/football/teams/55.png",
    "bournemouth": "https://media.api-sports.io/football/teams/35.png",
    "crystal palace": "https://media.api-sports.io/football/teams/52.png",
    "wolves": "https://media.api-sports.io/football/teams/39.png",
    "wolverhampton": "https://media.api-sports.io/football/teams/39.png",
    "wolverhampton wanderers": "https://media.api-sports.io/football/teams/39.png",
    "nottingham forest": "https://media.api-sports.io/football/teams/65.png",
    "leicester": "https://media.api-sports.io/football/teams/46.png",
    "leicester city": "https://media.api-sports.io/football/teams/46.png",
    "southampton": "https://media.api-sports.io/football/teams/41.png",
    "ipswich": "https://media.api-sports.io/football/teams/57.png",
    "ipswich town": "https://media.api-sports.io/football/teams/57.png",
    "luton": "https://media.api-sports.io/football/teams/1359.png",
    "luton town": "https://media.api-sports.io/football/teams/1359.png",
    "burnley": "https://media.api-sports.io/football/teams/44.png",
    "sheffield united": "https://media.api-sports.io/football/teams/62.png",
    "sheffield utd": "https://media.api-sports.io/football/teams/62.png",
    "leeds": "https://media.api-sports.io/football/teams/63.png",
    "leeds united": "https://media.api-sports.io/football/teams/63.png",
    "real madrid": "https://media.api-sports.io/football/teams/541.png",
    "barcelona": "https://media.api-sports.io/football/teams/529.png",
    "atletico madrid": "https://media.api-sports.io/football/teams/530.png",
    "bayern munich": "https://media.api-sports.io/football/teams/157.png",
    "borussia dortmund": "https://media.api-sports.io/football/teams/165.png",
    "dortmund": "https://media.api-sports.io/football/teams/165.png",
    "leverkusen": "https://media.api-sports.io/football/teams/168.png",
    "bayer leverkusen": "https://media.api-sports.io/football/teams/168.png",
    "psg": "https://media.api-sports.io/football/teams/85.png",
    "paris saint germain": "https://media.api-sports.io/football/teams/85.png",
    "juventus": "https://media.api-sports.io/football/teams/496.png",
    "inter": "https://media.api-sports.io/football/teams/505.png",
    "inter milan": "https://media.api-sports.io/football/teams/505.png",
    "milan": "https://media.api-sports.io/football/teams/489.png",
    "ac milan": "https://media.api-sports.io/football/teams/489.png",
    "napoli": "https://media.api-sports.io/football/teams/492.png",
    "roma": "https://media.api-sports.io/football/teams/497.png",
    "as roma": "https://media.api-sports.io/football/teams/497.png",
}

club_updated = 0
for club_name, crest in COMMON_CLUB_CRESTS.items():
    cursor.execute("""
        UPDATE teams
        SET crest_url = ?
        WHERE (crest_url IS NULL OR crest_url = '')
          AND LOWER(TRIM(name)) = ?;
    """, (crest, club_name))
    club_updated += cursor.rowcount

print(f"Updated major clubs from common crests: {club_updated}")

conn.commit()

# 4b. Match fuzzy/partial names for European clubs from api-football
cursor.execute("SELECT id, name FROM teams WHERE (crest_url IS NULL OR crest_url = '');")
missing_teams = cursor.fetchall()

cursor.execute("SELECT name, crest_url FROM teams WHERE crest_url IS NOT NULL AND crest_url != '';")
all_with_crest = cursor.fetchall()

alias_map = {
    "nott'm forest": "nottingham",
    "vallecano": "rayo vallecano",
    "sociedad": "real sociedad",
    "ath bilbao": "athletic club",
    "ath madrid": "atletico madrid",
    "celta": "celta vigo",
    "betis": "real betis",
    "m'gladbach": "monchengladbach",
    "ein frankfurt": "eintracht frankfurt",
    "fc koln": "koln",
    "verona": "hellas verona",
    "inter": "internazionale",
    "sp gijon": "sporting gijon",
    "le havre": "le havre",
    "clermont": "clermont foot",
    "paris fc": "paris fc",
}

fuzzy_updated = 0
for tid, tname in missing_teams:
    norm = tname.lower().strip()
    target_crest = None

    # Check alias map
    for alias_k, alias_v in alias_map.items():
        if alias_k in norm:
            for cname, crest in all_with_crest:
                if alias_v in cname.lower():
                    target_crest = crest
                    break
        if target_crest:
            break

    # Check substring match
    if not target_crest and len(norm) >= 4:
        # Clean common prefixes/suffixes
        clean = norm.replace("fc", "").replace("cf", "").strip()
        if len(clean) >= 4:
            for cname, crest in all_with_crest:
                c_clean = cname.lower().replace("fc", "").replace("cf", "").strip()
                if (clean in c_clean or c_clean in clean) and len(c_clean) >= 4:
                    target_crest = crest
                    break

    if target_crest:
        cursor.execute("UPDATE teams SET crest_url = ? WHERE id = ?;", (target_crest, tid))
        fuzzy_updated += 1

print(f"Updated European clubs via fuzzy matching: {fuzzy_updated}")
conn.commit()

EUROPEAN_TOP_CLUBS = {
    "atalanta": "https://media.api-sports.io/football/teams/499.png",
    "fiorentina": "https://media.api-sports.io/football/teams/502.png",
    "lazio": "https://media.api-sports.io/football/teams/487.png",
    "torino": "https://media.api-sports.io/football/teams/503.png",
    "bologna": "https://media.api-sports.io/football/teams/500.png",
    "cagliari": "https://media.api-sports.io/football/teams/490.png",
    "udinese": "https://media.api-sports.io/football/teams/494.png",
    "empoli": "https://media.api-sports.io/football/teams/511.png",
    "genoa": "https://media.api-sports.io/football/teams/495.png",
    "monza": "https://media.api-sports.io/football/teams/1579.png",
    "lecce": "https://media.api-sports.io/football/teams/867.png",
    "parma": "https://media.api-sports.io/football/teams/523.png",
    "venezia": "https://media.api-sports.io/football/teams/517.png",
    "verona": "https://media.api-sports.io/football/teams/504.png",
    "frosinone": "https://media.api-sports.io/football/teams/512.png",
    "sassuolo": "https://media.api-sports.io/football/teams/488.png",
    "girona": "https://media.api-sports.io/football/teams/547.png",
    "osasuna": "https://media.api-sports.io/football/teams/727.png",
    "alaves": "https://media.api-sports.io/football/teams/263.png",
    "almeria": "https://media.api-sports.io/football/teams/723.png",
    "celta vigo": "https://media.api-sports.io/football/teams/538.png",
    "werder bremen": "https://media.api-sports.io/football/teams/162.png",
    "hoffenheim": "https://media.api-sports.io/football/teams/167.png",
    "freiburg": "https://media.api-sports.io/football/teams/160.png",
    "wolfsburg": "https://media.api-sports.io/football/teams/161.png",
    "stuttgart": "https://media.api-sports.io/football/teams/172.png",
    "vfb stuttgart": "https://media.api-sports.io/football/teams/172.png",
    "bochum": "https://media.api-sports.io/football/teams/176.png",
    "vfl bochum": "https://media.api-sports.io/football/teams/176.png",
    "heidenheim": "https://media.api-sports.io/football/teams/180.png",
    "st pauli": "https://media.api-sports.io/football/teams/186.png",
    "holstein kiel": "https://media.api-sports.io/football/teams/191.png",
    "lille": "https://media.api-sports.io/football/teams/79.png",
    "paris sg": "https://media.api-sports.io/football/teams/85.png",
    "rennes": "https://media.api-sports.io/football/teams/94.png",
    "brest": "https://media.api-sports.io/football/teams/1063.png",
    "strasbourg": "https://media.api-sports.io/football/teams/95.png",
    "montpellier": "https://media.api-sports.io/football/teams/82.png",
    "lorient": "https://media.api-sports.io/football/teams/97.png",
    "st etienne": "https://media.api-sports.io/football/teams/1063.png",
    "auxerre": "https://media.api-sports.io/football/teams/108.png",
    "reims": "https://media.api-sports.io/football/teams/93.png",
    "nice": "https://media.api-sports.io/football/teams/84.png",
    "lens": "https://media.api-sports.io/football/teams/116.png",
    "monaco": "https://media.api-sports.io/football/teams/91.png",
    "marseille": "https://media.api-sports.io/football/teams/81.png",
    "lyon": "https://media.api-sports.io/football/teams/80.png",
    "nantes": "https://media.api-sports.io/football/teams/83.png",
    "toulouse": "https://media.api-sports.io/football/teams/96.png",
    "le havre": "https://media.api-sports.io/football/teams/99.png",
    "angers": "https://media.api-sports.io/football/teams/77.png",
}

for name, crest in EUROPEAN_TOP_CLUBS.items():
    cursor.execute("""
        UPDATE teams
        SET crest_url = ?
        WHERE (crest_url IS NULL OR crest_url = '')
          AND (LOWER(TRIM(name)) = ? OR LOWER(TRIM(name)) LIKE ?);
    """, (crest, name, f"%{name}%"))

conn.commit()

# 5. Total teams after
cursor.execute("SELECT COUNT(*), COUNT(CASE WHEN crest_url IS NOT NULL AND crest_url != '' THEN 1 END) FROM teams;")
total, with_crest = cursor.fetchone()
print(f"After: {with_crest}/{total} teams have crests ({total - with_crest} missing)")

cursor.execute("SELECT name, provider, external_id FROM teams WHERE crest_url IS NULL OR crest_url = '' LIMIT 40;")
print("Remaining teams missing crests:")
for r in cursor.fetchall():
    print("  ", r)

conn.close()
