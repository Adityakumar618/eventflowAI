"""
Mappls API — Final working format verification for all 6 APIs.
Known working:
  - Route ADV: REST key in URL path + bearer token header
  - Rev Geocode: bearer token header only
"""
import requests, json

CLIENT_ID     = '96dHZVzsAuuxSJ-tDn4LaehwjEHq2s15wC2BcQ1r3yn-ONX77cea2jiKPlFyXxo4xHfe6Zi5eSYRzS-a0LGrCxid__jG67F8'
CLIENT_SECRET = 'lrFxI-iSEg_c57tGZPkQjTH0FFqxvORR7OXwhKG46D0ZNalQVCITc82Kv4igU6EqRPMLN0RUWXnFcdPNaFS_BwIJvXqd_KR9EvXtV6Ws7Y0='
REST_KEY      = '99fcdc1f089f4dfaf2470df871d8c741'
BASE          = f"https://apis.mappls.com/advancedmaps/v1/{REST_KEY}"

# Get token
token_resp = requests.post(
    "https://outpost.mappls.com/api/security/oauth/token",
    data={"grant_type":"client_credentials","client_id":CLIENT_ID,"client_secret":CLIENT_SECRET},
    headers={"Content-Type":"application/x-www-form-urlencoded"}, timeout=10
)
token = token_resp.json()["access_token"]
hdrs  = {"Authorization": f"bearer {token}"}
print(f"Token OK: {token[:20]}...\n")
print("="*62)

# ── 1. Route ADV (CONFIRMED: REST key in URL + bearer header) ─────
print("[1] Route ADV — Silk Board to Marathahalli")
r1 = requests.get(
    f"{BASE}/route_adv/driving/77.6228,12.9177;77.6972,12.9591",
    headers=hdrs, timeout=10
)
print(f"  Status: {r1.status_code}")
if r1.status_code == 200:
    d = r1.json()["routes"][0]
    dist_km  = d["distance"] / 1000
    dur_min  = d["duration"] / 60
    traf_min = d.get("duration_in_traffic", d["duration"]) / 60
    print(f"  Distance:  {dist_km:.1f} km")
    print(f"  Free-flow: {dur_min:.1f} min")
    print(f"  Traffic:   {traf_min:.1f} min")
    print(f"  Delay:     {traf_min - dur_min:.1f} min")
    print("  [WORKING]")
print()

# ── 2. Rev Geocode (CONFIRMED: bearer header only) ─────────────────
print("[2] Reverse Geocode — Silk Board")
r2 = requests.get(
    f"{BASE}/rev_geocode",
    params={"lat": "12.9177", "lng": "77.6228"},
    headers=hdrs, timeout=10
)
print(f"  Status: {r2.status_code}")
if r2.status_code == 200:
    res = r2.json().get("results", [{}])
    if res:
        print(f"  Address: {res[0].get('formatted_address','')[:80]}")
    print("  [WORKING]")
print()

# ── 3. Geocode (try with REST key in URL) ─────────────────────────
print("[3] Geocode — try with REST key in URL")
r3 = requests.get(
    f"{BASE}/geocode",
    params={"address": "Silk Board Junction, Bengaluru"},
    headers=hdrs, timeout=10
)
print(f"  Status: {r3.status_code} | {r3.text[:150]}")
print()

# ── 4. Nearby (try with REST key in URL) ─────────────────────────
print("[4] Nearby — Police stations near Silk Board")
r4 = requests.get(
    f"{BASE}/nearby",
    params={"keywords": "police station", "refLocation": "12.9177,77.6228", "radius": 2000},
    headers=hdrs, timeout=10
)
print(f"  Status: {r4.status_code} | {r4.text[:200]}")
if r4.status_code == 200:
    locs = r4.json().get("suggestedLocations", [])
    print(f"  Found {len(locs)} locations")
    for loc in locs[:3]:
        print(f"    - {loc.get('placeName','')}")
print()

# ── 5. Distance Matrix ETA (try with REST key in URL) ─────────────
print("[5] Distance Matrix ETA — Yelahanka to Hebbal")
r5 = requests.get(
    f"{BASE}/distance_matrix_eta/driving/",
    params={"origins": "13.1007,77.5963", "destinations": "13.0354,77.5910"},
    headers=hdrs, timeout=10
)
print(f"  Status: {r5.status_code} | {r5.text[:200]}")
# Also try different origins format
r5b = requests.get(
    f"{BASE}/distance_matrix_eta/driving/",
    params={"origins": "77.5963,13.1007", "destinations": "77.5910,13.0354"},
    headers=hdrs, timeout=10
)
print(f"  Status (lng,lat format): {r5b.status_code} | {r5b.text[:200]}")
print()

# ── 6. Snap to Road ───────────────────────────────────────────────
print("[6] Snap to Road")
r6 = requests.get(
    f"{BASE}/snap_to_road",
    params={"pts": "12.9177,77.6228"},
    headers=hdrs, timeout=10
)
print(f"  Status: {r6.status_code} | {r6.text[:200]}")
print()

# ── Summary ───────────────────────────────────────────────────────
print("="*62)
print("SUMMARY:")
for name, code in [("Route ADV", r1.status_code), ("Rev Geocode", r2.status_code),
                    ("Geocode", r3.status_code), ("Nearby", r4.status_code),
                    ("Distance Matrix", r5.status_code), ("Snap to Road", r6.status_code)]:
    icon = "WORKING" if code == 200 else f"status={code}"
    print(f"  {name:<18} {icon}")
