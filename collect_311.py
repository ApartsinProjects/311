"""
collect_311.py — Assemble a unified multi-city 311 FREE-TEXT corpus for the
cross-jurisdiction classification benchmark.

Pulls citizen-narrative service requests from 7 verified free-text portals into a
single normalized schema:

    city, source, native_category, native_parent, text, address, lat, lon,
    created_at, status, raw_id

- 6 cities via the Socrata resource API (stable :id ordering, offset paging).
- San Francisco via the Open311 GeoReport v2 API (page paging; bulk export strips text).

Usage:
    python collect_311.py --per-city 2000          # fast proof run (default)
    python collect_311.py --per-city 50000         # fuller pull
    python collect_311.py --cities Gainesville,Honolulu
    python collect_311.py --app-token XXXX         # optional Socrata token (higher rate limit)

Writes one CSV per city to data/raw/<city>.csv and a combined data/raw/all_cities.csv,
and prints a per-city summary (rows kept, non-empty-text %, #distinct categories).
"""
import argparse, csv, json, os, ssl, sys, time, urllib.request, urllib.error, urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "data", "raw")
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"

UNIFIED_COLS = ["city","source","native_category","native_parent","text",
                "address","lat","lon","created_at","status","raw_id"]

# Per-city field mapping. fn = row -> unified dict. Socrata unless open311=True.
SOCRATA_CITIES = {
    "BatonRouge":  dict(domain="data.brla.gov",               did="7ixm-mnvx",
                        text="comments", cat="typename", parent="parenttype",
                        addr="streetname", lat="latitude", lon="longitude",
                        created="datetimeinit", status="requeststatus", rid="requestid"),
    "Bloomington": dict(domain="data.bloomington.in.gov",      did="aw6y-t4ix",
                        text="description", cat="service_name", parent=None,
                        addr="address", lat="lat", lon="long",
                        created="requested_datetime", status="status", rid="service_request_id"),
    "Richmond":    dict(domain="data.richmondgov.com",         did="vgg4-hjn8",
                        text="description", cat="category", parent=None,
                        addr="street_address", lat="lat", lon="lng",
                        created="createdat", status="status", rid="id"),
    "Auburn_WA":   dict(domain="data.auburnwa.gov",            did="bduj-5afh",
                        text="description", cat="service_name", parent=None,
                        addr="address", lat="lat", lon="long",
                        created="requested_datetime", status="status", rid="service_request_id"),
    "Gainesville": dict(domain="data.cityofgainesville.org",   did="78uv-94ar",
                        text="description", cat="request_type", parent=None,
                        addr="location_detail", lat="latitude", lon="longitude",
                        created="created", status="status", rid="id"),
    "Honolulu":    dict(domain="data.honolulu.gov",            did="jdy7-ftwe",
                        text="description", cat="request_type", parent=None,
                        addr="address", lat="latitude", lon="longitude",
                        created="created_at", status="status", rid="service_request_id"),
}
SF = dict(url="https://mobile311.sfgov.org/open311/v2/requests.json",
          text="description", cat="service_name", addr="address",
          lat="lat", lon="long", created="requested_datetime",
          status="status", rid="service_request_id")


def http_json(url, timeout=60, tries=3):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 522) and a < tries-1:
                time.sleep(2*(a+1)); continue
            print(f"    HTTP {e.code} on {url[:90]}"); return None
        except Exception as e:
            if a < tries-1:
                time.sleep(2*(a+1)); continue
            print(f"    ERR {type(e).__name__}: {e}"); return None
    return None


def g(row, key):
    if not key: return ""
    v = row.get(key)
    if isinstance(v, dict):  # Socrata location/point objects
        return json.dumps(v, ensure_ascii=False)
    return "" if v is None else str(v).strip()


def is_real_text(t):
    t = (t or "").strip()
    return bool(t) and bool(set(t) - set("- .")) and len(t) >= 3


def collect_socrata(city, cfg, cap, token):
    dom, did = cfg["domain"], cfg["did"]
    base = f"https://{dom}/resource/{did}.json"
    tok = f"&$$app_token={token}" if token else ""
    kept, offset, page = [], 0, 1000
    while len(kept) < cap:
        url = f"{base}?$limit={page}&$offset={offset}&$order=:id{tok}"
        rows = http_json(url)
        if not rows:
            break
        for row in rows:
            text = g(row, cfg["text"])
            if not is_real_text(text):
                continue
            kept.append({
                "city": city, "source": "socrata",
                "native_category": g(row, cfg["cat"]),
                "native_parent": g(row, cfg.get("parent")),
                "text": text,
                "address": g(row, cfg.get("addr")),
                "lat": g(row, cfg.get("lat")), "lon": g(row, cfg.get("lon")),
                "created_at": g(row, cfg.get("created")),
                "status": g(row, cfg.get("status")),
                "raw_id": g(row, cfg.get("rid")),
            })
            if len(kept) >= cap:
                break
        offset += page
        if len(rows) < page:
            break
        time.sleep(0.1)
    return kept


def collect_sf(cap):
    kept, page = [], 1
    while len(kept) < cap and page <= 200:
        url = f"{SF['url']}?page={page}&per_page=100"
        rows = http_json(url)
        if not rows:
            break
        seen_new = 0
        for row in rows:
            text = g(row, SF["text"])
            if not is_real_text(text):
                continue
            kept.append({
                "city": "SanFrancisco", "source": "open311",
                "native_category": g(row, SF["cat"]), "native_parent": "",
                "text": text, "address": g(row, SF["addr"]),
                "lat": g(row, SF["lat"]), "lon": g(row, SF["lon"]),
                "created_at": g(row, SF["created"]), "status": g(row, SF["status"]),
                "raw_id": g(row, SF["rid"]),
            })
            seen_new += 1
            if len(kept) >= cap:
                break
        if seen_new == 0:
            break
        page += 1
        time.sleep(0.2)
    return kept


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=UNIFIED_COLS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-city", type=int, default=2000)
    ap.add_argument("--cities", default="all")
    ap.add_argument("--app-token", default=os.environ.get("SOCRATA_APP_TOKEN", ""))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    want = set(SOCRATA_CITIES) | {"SanFrancisco"}
    if args.cities != "all":
        want = {c.strip() for c in args.cities.split(",")}

    combined, summary = [], []
    for city, cfg in SOCRATA_CITIES.items():
        if city not in want:
            continue
        print(f"[{city}] pulling up to {args.per_city} ...")
        rows = collect_socrata(city, cfg, args.per_city, args.app_token)
        write_csv(os.path.join(OUT_DIR, f"{city}.csv"), rows)
        cats = {r["native_category"] for r in rows if r["native_category"]}
        summary.append((city, len(rows), len(cats)))
        combined += rows

    if "SanFrancisco" in want:
        print(f"[SanFrancisco] pulling up to {args.per_city} via Open311 API ...")
        rows = collect_sf(args.per_city)
        write_csv(os.path.join(OUT_DIR, "SanFrancisco.csv"), rows)
        cats = {r["native_category"] for r in rows if r["native_category"]}
        summary.append(("SanFrancisco", len(rows), len(cats)))
        combined += rows

    write_csv(os.path.join(OUT_DIR, "all_cities.csv"), combined)

    print("\n==== SUMMARY ====")
    print(f"{'city':16s} {'rows':>8s} {'#categories':>12s}")
    for city, n, ncat in summary:
        print(f"{city:16s} {n:8d} {ncat:12d}")
    print(f"{'TOTAL':16s} {len(combined):8d}")
    print(f"\nwrote {OUT_DIR}\\*.csv  (combined: all_cities.csv)")


if __name__ == "__main__":
    main()
