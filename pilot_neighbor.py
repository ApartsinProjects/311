"""Direct test of the user's question: do SIMILAR texts from DIFFERENT cities get DIFFERENT labels?
Uses embedding similarity as the semantic control (no lossy hand-schema). If, at matched text
similarity, cross-city pairs disagree on the harmonized label MORE than within-city pairs, that is
policy shift P(Y|S,A) != P(Y|S) measured model-free.

  python pilot_neighbor.py submit    # sample rows/city, embed (cache)
  python pilot_neighbor.py analyze    # similarity-binned cross- vs within-city label agreement + examples
"""
import sys, os, json, time
import numpy as np
from collections import defaultdict, Counter
from eval_common import load_split

CITIES = ["BatonRouge", "Bloomington", "Gainesville"]
N = 1200
EMB_F = "results/pilot_nbr_emb.json"
POOL_F = "results/pilot_nbr_pool.json"
EMB_MODEL = "text-embedding-3-small"


def submit():
    os.makedirs("results", exist_ok=True)
    sp = load_split()
    rng = np.random.RandomState(0)
    pool = []
    for c in CITIES:
        rows = sp["train"][c]
        idx = rng.permutation(len(rows))[:N]
        pool += [(rows[i][0], rows[i][1], c) for i in idx]
    json.dump(pool, open(POOL_F, "w"), ensure_ascii=False)
    from openai_batch import client
    embs = []
    txts = [t for t, y, c in pool]
    for i in range(0, len(txts), 256):
        chunk = [t[:1200] for t in txts[i:i+256]]
        for a in range(4):
            try:
                r = client.embeddings.create(model=EMB_MODEL, input=chunk)
                embs.extend([d.embedding for d in r.data]); break
            except Exception:
                if a == 3:
                    raise
                time.sleep(2*(a+1))
        print(f"  embedded {min(i+256,len(txts))}/{len(txts)}"); sys.stdout.flush()
    json.dump(embs, open(EMB_F, "w"))
    print(f"pool={len(pool)} by_city={Counter(c for _,_,c in pool)}; wrote {EMB_F}")


def analyze():
    pool = [tuple(x) for x in json.load(open(POOL_F, encoding="utf-8"))]
    E = np.array(json.load(open(EMB_F, encoding="utf-8")), dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    y = np.array([p[1] for p in pool]); city = np.array([p[2] for p in pool])
    n = len(pool)
    S = E @ E.T
    np.fill_diagonal(S, -1.0)                       # exclude self
    same = city[:, None] == city[None, :]

    # For each row, its best WITHIN-city and best CROSS-city neighbor: (sim, label_agree)
    bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    rec = {"within": defaultdict(lambda: [0, 0]), "cross": defaultdict(lambda: [0, 0])}  # bin -> [agree, total]

    def tally(mask_row, kind):
        for i in range(n):
            sims = np.where(mask_row[i], S[i], -1.0)
            j = int(np.argmax(sims))
            s = sims[j]
            if s < 0.5:
                continue
            b = next(k for k, (lo, hi) in enumerate(bins) if lo <= s < hi)
            rec[kind][b][1] += 1
            rec[kind][b][0] += int(y[i] == y[j])

    tally(same, "within")
    tally(~same, "cross")

    print(f"n={n} per-city={dict(Counter(city.tolist()))}  labels in harmonized 14-class space")
    print(f"{'cosine bin':>12s}{'within-city agree':>20s}{'cross-city agree':>20s}{'gap':>8s}")
    out_bins = []
    for k, (lo, hi) in enumerate(bins):
        wa, wt = rec["within"][k]; ca, ct = rec["cross"][k]
        wr = wa/wt if wt else float("nan"); cr = ca/ct if ct else float("nan")
        gap = (wr - cr) if (wt and ct) else float("nan")
        print(f"  [{lo:.1f},{hi:.1f}){wa:>6d}/{wt:<6d}={wr:>5.2f}{ca:>10d}/{ct:<6d}={cr:>5.2f}{gap:>8.2f}")
        out_bins.append({"bin": [lo, hi], "within": [wa, wt], "cross": [ca, ct],
                         "within_agree": None if not wt else round(wr, 4),
                         "cross_agree": None if not ct else round(cr, 4)})

    # concrete high-similarity cross-city DISAGREEMENTS (the user's exact scenario)
    print("\nHigh-similarity (cos>0.85) cross-city pairs with DIFFERENT labels:")
    examples = []
    seen = set()
    order = np.argsort(-S, axis=1)
    for i in range(n):
        for j in order[i][:1]:
            if same[i, j] or S[i, j] < 0.85 or y[i] == y[j]:
                continue
            key = tuple(sorted([i, int(j)]))
            if key in seen:
                continue
            seen.add(key)
            examples.append({"cos": round(float(S[i, j]), 3),
                             "a": {"city": pool[i][2], "label": pool[i][1], "text": pool[i][0][:160]},
                             "b": {"city": pool[int(j)][2], "label": pool[int(j)][1], "text": pool[int(j)][0][:160]}})
    examples.sort(key=lambda e: -e["cos"])
    for e in examples[:12]:
        print(f"  cos={e['cos']}  {e['a']['city']}:{e['a']['label']}  vs  {e['b']['city']}:{e['b']['label']}")
        print(f"     A: {e['a']['text']}")
        print(f"     B: {e['b']['text']}")
    # aggregate: among cos>0.85 cross-city best-neighbor pairs, disagreement rate
    ca, ct = rec["cross"][4]  # bin [0.9,1.01)
    ca3, ct3 = rec["cross"][3]  # [0.8,0.9)
    json.dump({"bins": out_bins, "n_examples_cos>0.85_crosscity_diff": len(examples),
               "examples": examples[:40]}, open("results/pilot_neighbor.json", "w"), indent=2)
    print(f"\n{len(examples)} distinct cross-city near-paraphrase pairs (cos>0.85) disagree on label.")
    print("wrote results/pilot_neighbor.json")
    print("\n[read] policy shift is supported IF cross-city agreement is well below within-city at HIGH cosine.")


if __name__ == "__main__":
    {"submit": submit, "analyze": analyze}[sys.argv[1] if len(sys.argv) > 1 else "submit"]()
