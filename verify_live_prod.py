import json
import urllib.parse
import urllib.request
import sys

base = "https://zolapredicts.vercel.app"

print("==================================================================")
print("1. TEST LIVE /api/predict ON MATCH WITHOUT ODDS")
print("==================================================================")
url1 = f"{base}/api/predict?id=48"
req1 = urllib.request.Request(url1, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req1, timeout=20) as resp:
        data1 = json.loads(resp.read().decode("utf-8"))
        hc1 = data1.get("high_confidence_report", {})
        print("HTTP Status:", resp.status)
        print("is_skipped:", hc1.get("is_skipped"))
        print("skip_reason:", hc1.get("skip_reason"))
        print("total_picks:", hc1.get("total_picks"))
        print("picks count:", len(hc1.get("picks", [])))
        print("skip_rationale:", hc1.get("skip_rationale"))
        
        # Verify no synthetic odds generated
        ens = data1.get("predictions", {}).get("ensemble", {})
        for m, pred in ens.items():
            va = pred.get("value_analysis")
            if va:
                print(f"Market {m} value_analysis status: {va.get('status')}")
except Exception as e:
    print("Error in Test 1:", e)

print("\n==================================================================")
print("2. TEST LIVE /api/predict WITH VALID PRE-MATCH ODDS (OU15)")
print("==================================================================")
valid_odds = {"OU15": {"OVER_15": 1.45, "is_synthetic": False}}
odds_param = urllib.parse.quote(json.dumps(valid_odds))
url2 = f"{base}/api/predict?id=48&odds={odds_param}"
req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req2, timeout=20) as resp:
        data2 = json.loads(resp.read().decode("utf-8"))
        hc2 = data2.get("high_confidence_report", {})
        print("HTTP Status:", resp.status)
        print("is_skipped:", hc2.get("is_skipped"))
        print("skip_reason:", hc2.get("skip_reason"))
        print("total_picks:", hc2.get("total_picks"))
        picks = hc2.get("picks", [])
        for p in picks:
            print(f"  • Pick #{p.get('rank')}: {p.get('market')} ({p.get('outcome_display')})")
            print(f"    Probability: {p.get('probability_pct')} | Odd: @ {p.get('market_odd')}")
            print(f"    Expected Value: {p.get('expected_value_pct')} | Status: {p.get('betting_value_status')}")
            assert p.get("market") in ("OU05", "OU15"), f"Invalid market selected: {p.get('market')}"
except Exception as e:
    print("Error in Test 2:", e)

print("\n==================================================================")
print("3. TEST LIVE /api/predict WITH DOUBLE CHANCE ODDS (SHOULD BE BLOCKED)")
print("==================================================================")
dc_odds = {"DOUBLE_CHANCE": {"1X": 1.60, "X2": 2.10}}
dc_param = urllib.parse.quote(json.dumps(dc_odds))
url3 = f"{base}/api/predict?id=48&odds={dc_param}"
req3 = urllib.request.Request(url3, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req3, timeout=20) as resp:
        data3 = json.loads(resp.read().decode("utf-8"))
        hc3 = data3.get("high_confidence_report", {})
        print("HTTP Status:", resp.status)
        print("is_skipped:", hc3.get("is_skipped"))
        print("skip_reason:", hc3.get("skip_reason"))
        print("total_picks:", hc3.get("total_picks"))
        picks3 = hc3.get("picks", [])
        print("Picks count:", len(picks3))
        assert hc3.get("is_skipped") is True, "Double Chance was not skipped!"
        assert len(picks3) == 0, "Double Chance generated picks!"
        print("Double Chance is strictly BLOCKED as required!")
except Exception as e:
    print("Error in Test 3:", e)

print("\n==================================================================")
print("4. TEST LIVE /api/predict WITH SYNTHETIC ODDS (SHOULD BE SKIPPED)")
print("==================================================================")
synth_odds = {"OU15": {"OVER_15": 1.45, "is_synthetic": True, "source": "synthetic"}}
synth_param = urllib.parse.quote(json.dumps(synth_odds))
url4 = f"{base}/api/predict?id=48&odds={synth_param}"
req4 = urllib.request.Request(url4, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req4, timeout=20) as resp:
        data4 = json.loads(resp.read().decode("utf-8"))
        hc4 = data4.get("high_confidence_report", {})
        print("HTTP Status:", resp.status)
        print("is_skipped:", hc4.get("is_skipped"))
        print("skip_reason:", hc4.get("skip_reason"))
        print("total_picks:", hc4.get("total_picks"))
        assert hc4.get("is_skipped") is True, "Synthetic odds were not skipped!"
        print("Synthetic odds successfully REJECTED as required!")
except Exception as e:
    print("Error in Test 4:", e)
