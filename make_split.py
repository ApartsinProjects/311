"""
make_split.py -- freeze ONE canonical benchmark split shared by every arm.

For each city we hold out a fixed, seeded, stratified TEST set (up to --test-size rows).
Every model (TF-IDF, DistilBERT, LLM) predicts on exactly these rows, so in-city and
leave-one-city-out are directly comparable and the three arms are one comparison.

Canonical benchmark uses the informative subset (data/harmonized_filtered.csv), which also
removes 'Test'/shorthand hygiene junk. Writes data/eval_split.csv with columns:
  city, role (train|test), label, text
Train rows are the per-city remainder (used for in-city training and pooled for LOCO).
"""
import csv, os, sys, json
from collections import defaultdict, Counter
import numpy as np

csv.field_size_limit(10**7)
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEED = 42


def main():
    test_size = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    src = os.path.join(DATA, "harmonized_filtered.csv")
    by_city = defaultdict(list)
    with open(src, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_city[r["city"]].append((r["text"], r["label"]))

    rng = np.random.RandomState(SEED)
    rows_out = []
    print(f"{'city':14s}{'total':>8s}{'test':>7s}{'train':>8s}{'#cls(test)':>11s}")
    for city, rows in by_city.items():
        # stratified test: sample proportionally per label, capped so test ~ test_size
        by_lab = defaultdict(list)
        for i, (t, y) in enumerate(rows):
            by_lab[y].append(i)
        n = len(rows)
        frac = min(1.0, test_size / n)
        test_idx = set()
        for y, idxs in by_lab.items():
            k = max(1, int(round(len(idxs) * frac))) if len(idxs) >= 2 else 0
            if k:
                pick = rng.choice(idxs, size=min(k, len(idxs)), replace=False)
                test_idx.update(int(i) for i in pick)
        for i, (t, y) in enumerate(rows):
            role = "test" if i in test_idx else "train"
            rows_out.append((city, role, y, t))
        ntest = len(test_idx)
        ncls = len({rows[i][1] for i in test_idx})
        print(f"{city:14s}{n:8d}{ntest:7d}{n-ntest:8d}{ncls:11d}")

    out = os.path.join(DATA, "eval_split.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["city", "role", "label", "text"])
        w.writerows(rows_out)
    ntest = sum(1 for r in rows_out if r[1] == "test")
    print(f"\nwrote {out}  ({len(rows_out)} rows, {ntest} test)  seed={SEED}")


if __name__ == "__main__":
    main()
