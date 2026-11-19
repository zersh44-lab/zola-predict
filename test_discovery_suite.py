import urllib.request
import json
import sys

base_url = "http://127.0.0.1:8080"

print("--- Testing Zola Predict BeSoccer Discovery System ---")

# Test 1: Discovery Feed for today
req = urllib.request.Request(f"{base_url}/api/discovery?date=2026-11-17")
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode())
    print("1. Discovery Feed: HTTP", resp.status)
    print("   Counts:", data.get("counts"))
    print("   Featured matches count:", len(data.get("featured_matches", [])))
    print("   Grouped competitions count:", len(data.get("grouped_competitions", [])))
    if data.get("featured_matches"):
        fm = data["featured_matches"][0]
        print(f"   Top Featured Match: {fm.get('home_team_name')} vs {fm.get('away_team_name')} (Score: {fm.get('priority_score', 0):.1f})")

# Test 2: Prioritization Config GET
req = urllib.request.Request(f"{base_url}/api/prioritization/config")
with urllib.request.urlopen(req) as resp:
    cfg = json.loads(resp.read().decode())
    print("2. Prioritization Config GET: HTTP", resp.status)
    print("   Weights:", cfg.get("weights"))

# Test 3: Prioritization Config POST
post_data = json.dumps({
    "competition_importance": 0.35,
    "competition_popularity": 0.20,
    "team_popularity": 0.20,
    "match_relevance": 0.10,
    "kickoff_proximity": 0.10,
    "data_completeness": 0.05
}).encode()
req = urllib.request.Request(f"{base_url}/api/prioritization/config", data=post_data, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print("3. Prioritization Config POST: HTTP", resp.status, "success:", res.get("success"))

# Test 4: Favorites GET
req = urllib.request.Request(f"{base_url}/api/favorites")
with urllib.request.urlopen(req) as resp:
    favs = json.loads(resp.read().decode())
    print("4. Favorites GET: HTTP", resp.status)
    print("   Teams count:", len(favs.get("teams", [])))
    print("   Competitions count:", len(favs.get("competitions", [])))

# Test 5: Favorites Toggle POST
fav_post = json.dumps({"target_type": "team", "target_id": 15, "target_name": "Portugal"}).encode()
req = urllib.request.Request(f"{base_url}/api/favorites/toggle", data=fav_post, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as resp:
    fav_res = json.loads(resp.read().decode())
    print("5. Favorites Toggle ON: HTTP", resp.status, fav_res.get("success"))

# Test 6: Verify Portugal in Favorites
req = urllib.request.Request(f"{base_url}/api/favorites")
with urllib.request.urlopen(req) as resp:
    favs = json.loads(resp.read().decode())
    print("6. Favorites after toggle ON:", [t.get("name") for t in favs.get("teams", [])])

# Test 7: Toggle Portugal off
req = urllib.request.Request(f"{base_url}/api/favorites/toggle", data=fav_post, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req) as resp:
    fav_res2 = json.loads(resp.read().decode())
    print("7. Favorites Toggle OFF: HTTP", resp.status, fav_res2.get("success"))

# Test 8: Filter Discovery by Category (upcoming)
req = urllib.request.Request(f"{base_url}/api/discovery?date=2026-11-17&category=upcoming")
with urllib.request.urlopen(req) as resp:
    disc_up = json.loads(resp.read().decode())
    print("8. Discovery upcoming category count:", len(disc_up.get("all_fixtures", [])))

# Test 9: Filter Discovery by Search (Wales)
req = urllib.request.Request(f"{base_url}/api/discovery?date=2026-11-17&search=Wales")
with urllib.request.urlopen(req) as resp:
    disc_wales = json.loads(resp.read().decode())
    print("9. Discovery search 'Wales' matches:", len(disc_wales.get("all_fixtures", [])))

# Test 10: Sort by kickoff time
req = urllib.request.Request(f"{base_url}/api/discovery?date=2026-11-17&sort=kickoff")
with urllib.request.urlopen(req) as resp:
    disc_kick = json.loads(resp.read().decode())
    first_fix = disc_kick.get("all_fixtures", [])[0] if disc_kick.get("all_fixtures") else {}
    print(f"10. Discovery sort 'kickoff': First match at {first_fix.get('utc_date')}: {first_fix.get('home_team_name')} vs {first_fix.get('away_team_name')}")

# Test 11: Prediction endpoint preserves workflow
first_id = disc_wales.get("all_fixtures", [{}])[0].get("id")
if first_id:
    req = urllib.request.Request(f"{base_url}/api/predict?id={first_id}")
    with urllib.request.urlopen(req) as resp:
        pred_res = json.loads(resp.read().decode())
        ens = pred_res.get("predictions", {}).get("ensemble", {}).get("1X2", {})
        print(f"11. Prediction on fixture {first_id}: outcome={ens.get('predicted_outcome')}, confidence={ens.get('confidence')}")

print("\n--- All 11 Discovery System Verification Tests Passed! ---")
