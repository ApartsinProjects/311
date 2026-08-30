"""
predictors.py -- Exp D: what predicts city-to-city transfer difficulty? For each ordered city
pair, compute divergence features (label distribution, vocabulary, text length, platform match)
and correlate them with the 7x7 TF-IDF transfer score. Exploratory (7 cities, 42 off-diagonal
pairs), reported with Spearman rho.
"""
import json, re
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from eval_common import load_split

SEECLICKFIX = {"Richmond", "Auburn_WA"}


def city_stats(train):
    stats = {}
    for c, rows in train.items():
        labels = Counter(y for _, y in rows)
        vocab = Counter()
        lens = []
        for t, _ in rows[:8000]:
            toks = re.findall(r"[a-z]+", t.lower())
            vocab.update(toks); lens.append(len(t.split()))
        stats[c] = {"labels": labels, "vocab": vocab, "meanlen": np.mean(lens)}
    return stats


def dist(counter, keys):
    tot = sum(counter.values()) or 1
    return np.array([counter.get(k, 0) / tot for k in keys]) + 1e-9


def js(ca, cb):
    keys = set(ca) | set(cb)
    return float(jensenshannon(dist(ca, keys), dist(cb, keys)))


def main():
    M = json.load(open("results/transfer_matrix.json")); cities = M["cities"]; mat = np.array(M["matrix"])
    st = city_stats(load_split()["train"])
    # top-2000 vocab per city for tractable JS
    for c in st:
        st[c]["vocab"] = Counter(dict(st[c]["vocab"].most_common(2000)))
    rows = []
    for i, a in enumerate(cities):
        for j, b in enumerate(cities):
            if i == j:
                continue
            rows.append({
                "transfer": mat[i, j],
                "label_js": js(st[a]["labels"], st[b]["labels"]),
                "vocab_js": js(st[a]["vocab"], st[b]["vocab"]),
                "len_diff": abs(st[a]["meanlen"] - st[b]["meanlen"]),
                "same_platform": 1.0 if {a, b} <= SEECLICKFIX else 0.0,
            })
    y = np.array([r["transfer"] for r in rows])
    print(f"n_pairs={len(rows)}  (exploratory: 7 cities)")
    print(f"{'feature':16s}{'Spearman rho':>14s}{'p':>10s}")
    for f in ["label_js", "vocab_js", "len_diff", "same_platform"]:
        x = np.array([r[f] for r in rows])
        rho, p = spearmanr(x, y)
        print(f"{f:16s}{rho:14.3f}{p:10.3f}")
    print("\n(negative rho for a divergence feature => more divergence, lower transfer)")


if __name__ == "__main__":
    main()
