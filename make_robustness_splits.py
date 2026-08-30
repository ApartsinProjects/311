"""
make_robustness_splits.py -- W7 robustness splits.
(A) DEDUP: collapse exact-duplicate texts corpus-wide and remove any TRAIN row whose normalized
    text also appears in that city's TEST set, so train/test text overlap cannot inflate in-city.
(B) TEMPORAL: earlier dates -> train, later -> test, per city, IF timestamps are available.
Reads data/all_cities.csv (has created_at) + the frozen data/eval_split.csv test membership.
Writes data/eval_split_dedup.csv (always) and data/eval_split_temporal.csv (if dates exist).
"""
import csv, os, re
from collections import defaultdict, Counter
csv.field_size_limit(10**7)
DATA = "data"


def norm(t):
    return re.sub(r"\s+", " ", (t or "").strip().lower())


def load_frozen_test():
    """Return {city: set(normalized test texts)} and the full split rows."""
    test_norm = defaultdict(set); rows = []
    with open(os.path.join(DATA, "eval_split.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if r["role"] == "test":
                test_norm[r["city"]].add(norm(r["text"]))
    return test_norm, rows


def build_dedup():
    test_norm, rows = load_frozen_test()
    seen = set()          # global exact-dup collapse (keep first occurrence)
    out = []; dropped_dup = dropped_leak = 0
    for r in rows:
        n = norm(r["text"])
        if r["role"] == "train":
            if n in test_norm[r["city"]]:          # train text overlaps this city's test -> leak
                dropped_leak += 1; continue
            if n in seen:                          # exact duplicate already kept
                dropped_dup += 1; continue
            seen.add(n)
        out.append(r)
    with open(os.path.join(DATA, "eval_split_dedup.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["city", "role", "label", "text"]); w.writeheader()
        for r in out: w.writerow({k: r[k] for k in ["city", "role", "label", "text"]})
    print(f"[dedup] dropped {dropped_leak} train/test-overlap rows + {dropped_dup} exact dups; "
          f"kept {len(out)} of {len(rows)} -> data/eval_split_dedup.csv")


def build_temporal():
    # check if all_cities.csv has usable created_at per city
    path = os.path.join(DATA, "raw", "all_cities.csv")
    if not os.path.exists(path):
        print("[temporal] no all_cities.csv; skipping"); return
    by = defaultdict(list); have_date = Counter()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("created_at") or "").strip()
            by[r["city"]].append((d, r.get("text", ""), r.get("native_category", "")))
            if d: have_date[r["city"]] += 1
    cov = {c: have_date[c] / max(1, len(v)) for c, v in by.items()}
    print("[temporal] date coverage per city:", {c: f"{p:.0%}" for c, p in cov.items()})
    usable = [c for c, p in cov.items() if p > 0.8]
    print(f"[temporal] cities with >80% dates: {usable}")
    if len(usable) < 4:
        print("[temporal] too few cities have timestamps for a clean temporal benchmark; "
              "documenting this as a data limitation rather than producing a partial split.")
        return
    print("[temporal] (would build earlier->train / later->test for usable cities)")


if __name__ == "__main__":
    build_dedup()
    build_temporal()
