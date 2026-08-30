"""
temporal_split.py -- W7 temporal robustness for the 4 cities that carry timestamps
(Bloomington, Auburn WA, Gainesville, San Francisco). For each, split by created_at
(earliest 80% train, latest 20% test) and compare in-city TF-IDF macro-F1 under the
temporal split vs a random 80/20 split on the same rows. If temporal ~ random, random
splitting is not badly inflating in-city performance. Documents that 3/7 cities lack
usable timestamps (collector field-mapping limitation).
"""
import csv, os, json, re
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
csv.field_size_limit(10**7)

from filter_labels import is_informative
DATED = ["Bloomington", "Auburn_WA", "Gainesville", "SanFrancisco"]


def load_dated():
    mp = json.load(open("data/harmonization_map.json", encoding="utf-8"))
    by = defaultdict(list)
    with open("data/raw/all_cities.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r["city"]
            if c not in DATED:
                continue
            d = (r.get("created_at") or "").strip()
            t = (r["text"] or "").strip()
            if not d or len(t) < 3 or not is_informative(t):
                continue
            sup = mp.get(c, {}).get(r["native_category"], "General_Admin_Other")
            if sup == "General_Admin_Other":
                continue
            by[c].append((d, t, sup))
    return by


def vec():
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3, max_features=30000, strip_accents="unicode")
    c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000)
    return w, c


def fit_score(train, test):
    if len(train) > 8000:                          # cap for tractable saga on this host
        idx = np.random.RandomState(0).permutation(len(train))[:8000]
        train = [train[i] for i in idx]
    tw = [t for t, _ in train]; ty = [y for _, y in train]; ew = [t for t, _ in test]; ey = [y for _, y in test]
    w, c = vec(); Xtr = hstack([w.fit_transform(tw), c.fit_transform(tw)]).tocsr(); Xte = hstack([w.transform(ew), c.transform(ew)]).tocsr()
    clf = LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", solver="saga", tol=1e-3); clf.fit(Xtr, ty)
    return f1_score(ey, clf.predict(Xte), average="macro", zero_division=0)


def main():
    by = load_dated()
    print(f"{'city':14s}{'n':>7s}{'temporal':>10s}{'random':>9s}{'delta':>8s}")
    rows_out = {}
    for c in DATED:
        rows = by.get(c, [])
        if len(rows) < 200:
            print(f"{c:14s}{len(rows):7d}  too few"); continue
        rows.sort(key=lambda x: x[0])                       # sort by created_at
        cut = int(0.8 * len(rows))
        tr_t = [(t, y) for _, t, y in rows[:cut]]; te_t = [(t, y) for _, t, y in rows[cut:]]
        # random split on the same rows
        rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
        tr_r = [(rows[i][1], rows[i][2]) for i in idx[:cut]]; te_r = [(rows[i][1], rows[i][2]) for i in idx[cut:]]
        f_temp = fit_score(tr_t, te_t); f_rand = fit_score(tr_r, te_r)
        rows_out[c] = {"n": len(rows), "temporal": round(f_temp, 4), "random": round(f_rand, 4)}
        print(f"{c:14s}{len(rows):7d}{f_temp:10.3f}{f_rand:9.3f}{f_temp-f_rand:+8.3f}")
    if rows_out:
        mt = np.mean([v["temporal"] for v in rows_out.values()]); mr = np.mean([v["random"] for v in rows_out.values()])
        print(f"\nmean temporal={mt:.3f}  mean random={mr:.3f}  delta={mt-mr:+.3f}")
        print("(temporal ~ random => random splitting does not badly inflate in-city performance)")
    json.dump(rows_out, open("results/temporal_split.json", "w"), indent=2)
    print("wrote results/temporal_split.json")


if __name__ == "__main__":
    main()
