"""
temporal_split.py -- W7 temporal robustness on ALL 7 cities.
Pulls each city's (created_date, text, native_category) fresh from its Socrata source with the
CORRECT date field (fixing the earlier collector mis-mapping for Baton Rouge / Richmond / Honolulu),
plus San Francisco from the assembled corpus. For each city, split by date (earliest 80% train,
latest 20% test) and compare in-city TF-IDF macro-F1 under temporal vs a random 80/20 split on the
same rows. temporal ~ random => random splitting does not inflate in-city performance.
"""
import csv, os, json, ssl, urllib.request, urllib.error
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from filter_labels import is_informative
csv.field_size_limit(10**7)
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# city -> (domain, dataset_id, text_col, cat_col, date_col)
SRC = {
 "BatonRouge":  ("data.brla.gov", "7ixm-mnvx", "comments", "typename", "createdate"),
 "Bloomington": ("data.bloomington.in.gov", "aw6y-t4ix", "description", "service_name", "requested_datetime"),
 "Richmond":    ("data.richmondgov.com", "vgg4-hjn8", "description", "category", "createddate"),
 "Auburn_WA":   ("data.auburnwa.gov", "bduj-5afh", "description", "service_name", "requested_datetime"),
 "Gainesville": ("data.cityofgainesville.org", "78uv-94ar", "description", "request_type", "created"),
 "Honolulu":    ("data.honolulu.gov", "jdy7-ftwe", "description", "request_type", "date_created"),
}
PULL = 20000


def pull(city):
    dom, did, tc, cc, dc = SRC[city]
    url = f"https://{dom}/resource/{did}.json?$select={dc}+as+d,{tc}+as+t,{cc}+as+c&$order={dc}&$limit={PULL}"
    r = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(r, timeout=90, context=CTX) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def load_sf():
    rows = []
    with open("data/raw/all_cities.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["city"] == "SanFrancisco" and (r.get("created_at") or "").strip():
                rows.append({"d": r["created_at"], "t": r["text"], "c": r["native_category"]})
    return rows


def harmonize_filter(city, raw, mp):
    out = []
    for r in raw:
        d = (r.get("d") or "").strip(); t = (r.get("t") or "").strip()
        if not d or len(t) < 3 or not is_informative(t):
            continue
        sup = mp.get(city, {}).get(r.get("c", ""), "General_Admin_Other")
        if sup == "General_Admin_Other":
            continue
        out.append((d, t, sup))
    return out


def vec():
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000, strip_accents="unicode")
    c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000)
    return w, c


def fit_score(train, test):
    if len(train) > 8000:
        idx = np.random.RandomState(0).permutation(len(train))[:8000]; train = [train[i] for i in idx]
    tw = [t for t, _ in train]; ty = [y for _, y in train]; ew = [t for t, _ in test]; ey = [y for _, y in test]
    w, c = vec(); Xtr = hstack([w.fit_transform(tw), c.fit_transform(tw)]).tocsr(); Xte = hstack([w.transform(ew), c.transform(ew)]).tocsr()
    clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", solver="saga", tol=1e-3); clf.fit(Xtr, ty)
    return f1_score(ey, clf.predict(Xte), average="macro", zero_division=0)


def main():
    mp = json.load(open("data/harmonization_map.json", encoding="utf-8"))
    cities = list(SRC) + ["SanFrancisco"]
    out = {}
    print(f"{'city':14s}{'n':>7s}{'temporal':>10s}{'random':>9s}{'delta':>8s}")
    for city in cities:
        try:
            raw = load_sf() if city == "SanFrancisco" else pull(city)
        except Exception as e:
            print(f"{city:14s}  pull failed: {type(e).__name__}"); continue
        rows = harmonize_filter(city, raw, mp)
        if len(rows) < 200:
            print(f"{city:14s}{len(rows):7d}  too few"); continue
        rows.sort(key=lambda x: x[0])
        cut = int(0.8 * len(rows))
        tr_t = [(t, y) for _, t, y in rows[:cut]]; te_t = [(t, y) for _, t, y in rows[cut:]]
        idx = np.random.RandomState(0).permutation(len(rows))
        tr_r = [(rows[i][1], rows[i][2]) for i in idx[:cut]]; te_r = [(rows[i][1], rows[i][2]) for i in idx[cut:]]
        ft, fr = fit_score(tr_t, te_t), fit_score(tr_r, te_r)
        out[city] = {"n": len(rows), "temporal": round(ft, 4), "random": round(fr, 4)}
        print(f"{city:14s}{len(rows):7d}{ft:10.3f}{fr:9.3f}{ft-fr:+8.3f}")
    if out:
        mt = np.mean([v["temporal"] for v in out.values()]); mr = np.mean([v["random"] for v in out.values()])
        print(f"\nmean temporal={mt:.3f}  random={mr:.3f}  delta={mt-mr:+.3f}  (n_cities={len(out)})")
    json.dump(out, open("results/temporal_split.json", "w"), indent=2)
    print("wrote results/temporal_split.json")


if __name__ == "__main__":
    main()
