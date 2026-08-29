"""
baseline_tfidf.py  --  First cross-jurisdiction baseline for the multi-city 311 benchmark.

Task: predict the harmonized super-class from the citizen free text.
Two protocols:
  (A) IN-CITY      : stratified 80/20 split within each city (upper bound / no domain shift).
  (B) LOCO         : Leave-One-City-Out -- train on all other cities, test on the held-out city
                     (the real cross-jurisdiction transfer number).

Model: TF-IDF (word 1-2grams + char 3-5grams) + multinomial Logistic Regression. CPU, local.

Sanity invariant (enforced & printed): for each city, IN-CITY macro-F1 must exceed its LOCO
macro-F1. If it doesn't, something is wrong (leakage, degenerate labels) -- flagged loudly.

Usage:
  python baseline_tfidf.py                      # all data in data/raw/all_cities.csv
  python baseline_tfidf.py --cap-per-city 8000  # balance city sizes
  python baseline_tfidf.py --drop-other         # (default) exclude General_Admin_Other
"""
import argparse, csv, json, os, sys
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
csv.field_size_limit(10**7)


def load(cap, drop_other):
    mp = json.load(open(os.path.join(DATA, "harmonization_map.json"), encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(DATA, "raw", "all_cities.csv"), encoding="utf-8")))
    by_city = defaultdict(list)
    for r in rows:
        city, text, nat = r["city"], (r["text"] or "").strip(), r["native_category"]
        if len(text) < 3:
            continue
        sup = mp.get(city, {}).get(nat, "General_Admin_Other")
        if drop_other and sup == "General_Admin_Other":
            continue
        by_city[city].append((text, sup))
    if cap:
        for c in by_city:
            if len(by_city[c]) > cap:
                idx = np.random.RandomState(0).permutation(len(by_city[c]))[:cap]
                by_city[c] = [by_city[c][i] for i in idx]
    return by_city


def make_vectorizers():
    word = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=3,
                           max_features=30000, strip_accents="unicode", lowercase=True)
    char = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5),
                           min_df=3, max_features=30000, lowercase=True)
    return word, char


def fit_transform(train_txt, test_txt):
    word, char = make_vectorizers()
    Xtr = hstack([word.fit_transform(train_txt), char.fit_transform(train_txt)]).tocsr()
    Xte = hstack([word.transform(test_txt), char.transform(test_txt)]).tocsr()
    return Xtr, Xte


def train_eval(train, test):
    tr_txt, tr_y = zip(*train); te_txt, te_y = zip(*test)
    Xtr, Xte = fit_transform(list(tr_txt), list(te_txt))
    clf = LogisticRegression(max_iter=1500, C=4.0, class_weight="balanced",
                             solver="saga", tol=1e-3)
    clf.fit(Xtr, tr_y)
    pred = clf.predict(Xte)
    macro = f1_score(te_y, pred, average="macro", zero_division=0)
    acc = accuracy_score(te_y, pred)
    maj = Counter(te_y).most_common(1)[0][1] / len(te_y)
    return macro, acc, maj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-city", type=int, default=0)
    ap.add_argument("--drop-other", action="store_true", default=True)
    ap.add_argument("--keep-other", dest="drop_other", action="store_false")
    args = ap.parse_args()

    by_city = load(args.cap_per_city, args.drop_other)
    cities = sorted(by_city, key=lambda c: -len(by_city[c]))
    print("dataset (rows per city, after harmonize" + (", drop Other" if args.drop_other else "") + "):")
    for c in cities:
        nlab = len(set(y for _, y in by_city[c]))
        print(f"  {c:14s} {len(by_city[c]):>7d} rows  {nlab:>2d} classes")
    total = sum(len(v) for v in by_city.values())
    print(f"  TOTAL          {total:>7d} rows\n")

    incity = {}
    print("=== (A) IN-CITY  (stratified 80/20) ===")
    print(f"{'city':14s}{'macroF1':>9s}{'acc':>7s}{'major':>7s}")
    for c in cities:
        data = by_city[c]
        ys = [y for _, y in data]
        keep = {k for k, v in Counter(ys).items() if v >= 5}     # need >=5 to split
        data = [(t, y) for t, y in data if y in keep]
        if len(set(y for _, y in data)) < 2:
            print(f"{c:14s}   n/a (single class)"); continue
        tr, te = train_test_split(data, test_size=0.2, random_state=0,
                                  stratify=[y for _, y in data])
        macro, acc, maj = train_eval(tr, te)
        incity[c] = macro
        print(f"{c:14s}{macro:9.3f}{acc:7.3f}{maj:7.3f}")

    print("\n=== (B) LEAVE-ONE-CITY-OUT (train on others, test on held-out) ===")
    print(f"{'held-out city':14s}{'macroF1':>9s}{'acc':>7s}{'major':>7s}{'  vs in-city (gap)':>20s}")
    loco = {}
    for c in cities:
        train = [d for oc in cities if oc != c for d in by_city[oc]]
        test = by_city[c]
        # restrict to labels seen in training
        train_labels = set(y for _, y in train)
        test_f = [(t, y) for t, y in test if y in train_labels]
        if len(test_f) < 20 or len(set(y for _, y in test_f)) < 2:
            print(f"{c:14s}   n/a"); continue
        macro, acc, maj = train_eval(train, test_f)
        loco[c] = macro
        gap = (incity.get(c, float('nan')) - macro)
        flag = "  <-- INVARIANT VIOLATION" if (c in incity and macro >= incity[c]) else ""
        print(f"{c:14s}{macro:9.3f}{acc:7.3f}{maj:7.3f}{incity.get(c,float('nan')):12.3f}{gap:+8.3f}{flag}")

    if incity and loco:
        mi = np.mean([incity[c] for c in loco if c in incity])
        ml = np.mean([loco[c] for c in loco])
        print(f"\nSUMMARY  mean IN-CITY macroF1 = {mi:.3f}   mean LOCO macroF1 = {ml:.3f}"
              f"   transfer gap = {mi-ml:+.3f}")
        viol = [c for c in loco if c in incity and loco[c] >= incity[c]]
        print("SANITY: in-city beats LOCO for every city." if not viol
              else f"SANITY VIOLATION for: {viol} -- investigate before trusting numbers.")
        try:
            from results_log import save_result
            save_result("tfidf_baseline",
                        {"mean_incity": round(float(mi), 4), "mean_loco": round(float(ml), 4),
                         "transfer_gap": round(float(mi - ml), 4),
                         "per_city_incity": {c: round(v, 4) for c, v in incity.items()},
                         "per_city_loco": {c: round(v, 4) for c, v in loco.items()}},
                        config={"cap_per_city": args.cap_per_city, "drop_other": args.drop_other},
                        note="TF-IDF(word1-2+char3-5)+LogReg")
        except Exception as e:
            print(f"[results_log] skipped: {e}")


if __name__ == "__main__":
    main()
