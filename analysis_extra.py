"""Reviewer-requested quantitative additions: per-city token stats, filter rates,
and the same-platform Richmond<->Auburn transfer cell (both SeeClickFix)."""
import csv, os
from collections import defaultdict
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from eval_common import load_split
csv.field_size_limit(10**7)
DATA = "data"

# ---- per-city token length stats on the benchmark ----
by = defaultdict(list)
with open(os.path.join(DATA, "harmonized_filtered.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        by[r["city"]].append(len(r["text"].split()))
print("=== per-city token length (whitespace tokens) ===")
print(f"{'city':13s}{'median':>7s}{'p90':>6s}{'%>64':>7s}{'%>128':>7s}")
for c, L in by.items():
    a = np.array(L)
    print(f"{c:13s}{np.median(a):7.0f}{np.percentile(a,90):6.0f}{100*(a>64).mean():6.1f}%{100*(a>128).mean():6.1f}%")

# ---- same-platform transfer: Richmond <-> Auburn (both SeeClickFix) ----
def vec():
    w = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=3, max_features=30000, strip_accents="unicode")
    c = TfidfVectorizer(sublinear_tf=True, analyzer="char_wb", ngram_range=(3,5), min_df=3, max_features=30000)
    return w, c
def cap(rows, n=8000, seed=0):
    return rows if len(rows)<=n else [rows[i] for i in np.random.RandomState(seed).permutation(len(rows))[:n]]
def tp(train, test):
    tw=[t for t,_ in train]; ty=[y for _,y in train]; ew=[t for t,_ in test]; ey=[y for _,y in test]
    w,c=vec(); Xtr=hstack([w.fit_transform(tw),c.fit_transform(tw)]).tocsr(); Xte=hstack([w.transform(ew),c.transform(ew)]).tocsr()
    clf=LogisticRegression(max_iter=1000,C=4.0,class_weight="balanced",solver="saga",tol=1e-3); clf.fit(Xtr,ty)
    return f1_score(ey, clf.predict(Xte), average="macro", zero_division=0)

sp = load_split()
print("\n=== same-platform vs LOCO transfer (TF-IDF, macro-F1) ===")
r_tr, r_te = cap(sp["train"]["Richmond"]), sp["test"]["Richmond"]
a_tr, a_te = cap(sp["train"]["Auburn_WA"]), sp["test"]["Auburn_WA"]
print(f"Richmond->Auburn (same platform): {tp(r_tr, a_te):.3f}   (Auburn LOCO was 0.434)")
print(f"Auburn->Richmond (same platform): {tp(a_tr, r_te):.3f}   (Richmond LOCO was 0.477)")
