"""B1 (ambiguity-driven taxonomy redesign w/ random-merge controls) + B2 (noise-corrected leaderboard).
Everything scores against already-frozen artifacts; no retraining. Prints invariants for verification."""
import json, glob, os, numpy as np
from eval_common import load_split

sp = load_split(); cities = list(sp["test"])
gold = {c: [y for _, y in sp["test"][c]] for c in cities}
LABELS = sorted({y for v in gold.values() for y in v}); Lidx = {l: i for i, l in enumerate(LABELS)}
accG = json.load(open("results/preds/acceptable_sets.json"))
accGem = json.load(open("results/preds/acceptable_sets_gemini25flash.json"))


def flat(d):  # {city:[..]} -> flat list in city order
    return [x for c in cities for x in d[c]]


GOLD = flat(gold)
ACC = [set(s) for c in cities for s in accG[c]]
ACC2 = [set(s) for c in cities for s in accGem[c]]


def fast_macro(g, p):
    g = np.asarray(g); p = np.asarray(p)
    labs = np.unique(np.concatenate([g, p])); f = []
    for l in labs:
        tp = np.sum((p == l) & (g == l)); fp = np.sum((p == l) & (g != l)); fn = np.sum((p != l) & (g == l))
        d = 2 * tp + fp + fn; f.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(f))


def load_arm(name, proto):
    P = json.load(open(f"results/preds/{name}.json"))[proto]
    return flat(P)


# ---------------- B2: noise-corrected leaderboard ----------------
ARMS = [("tfidf", "loco"), ("distilbert", "loco"), ("roberta", "loco"),
        ("llm_gpt4omini", "zeroshot"), ("llm_gpt4o", "zeroshot"),
        ("llm_gpt4omini_namesonly", "zeroshot"), ("llm_gpt4omini_fewshot", "zeroshot"),
        ("sbert", "zeroshot"), ("nli", "zeroshot")]
print("=== B2: strict vs defensibility-adjusted leaderboard (cross-city) ===")
print(f"{'arm':30s}{'strictF1':>9s}{'strictAcc':>10s}{'lenAcc':>8s}{'lenCons':>9s}")
rows = []
for name, proto in ARMS:
    p = load_arm(name, proto)
    sf1 = fast_macro(GOLD, p)
    sacc = np.mean([p[i] == GOLD[i] for i in range(len(GOLD))])
    lacc = np.mean([p[i] == GOLD[i] or p[i] in ACC[i] for i in range(len(GOLD))])
    lcons = np.mean([p[i] == GOLD[i] or (p[i] in ACC[i] and p[i] in ACC2[i]) for i in range(len(GOLD))])
    rows.append((name, sf1, sacc, lacc, lcons))
    print(f"{name:30s}{sf1:9.4f}{sacc:10.4f}{lacc:8.4f}{lcons:9.4f}")
    assert lacc + 1e-9 >= sacc, f"INVARIANT FAIL: lenient<strict for {name}"
print("[inv] lenient>=strict for all arms: OK")


def ranking(rows, k):
    return [r[0] for r in sorted(rows, key=lambda r: -r[k])]


rs, rl = ranking(rows, 2), ranking(rows, 3)  # strictAcc vs lenAcc
print("strict-acc order :", " > ".join(a.replace('llm_gpt4omini', 'mini').replace('_', '') for a in rs))
print("lenient-acc order:", " > ".join(a.replace('llm_gpt4omini', 'mini').replace('_', '') for a in rl))
flips = [(rs[i], rs[j]) for i in range(len(rs)) for j in range(i + 1, len(rs))
         if rl.index(rs[i]) > rl.index(rs[j])]
print(f"pairwise order flips strict->lenient: {len(flips)}", flips[:6])

# ---------------- B1: ambiguity-driven taxonomy ----------------
print("\n=== B1: co-acceptability graph + agglomerative merge ===")
C = np.zeros((14, 14))
for s in ACC:
    ids = [Lidx[x] for x in s if x in Lidx]
    for a in ids:
        for b in ids:
            if a != b: C[a, b] += 1


def agglomerate(sim, K):
    clusters = [[i] for i in range(14)]
    while len(clusters) > K:
        best = (-1, None)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = np.mean([sim[a, b] for a in clusters[i] for b in clusters[j]])
                if s > best[0]: best = (s, (i, j))
        i, j = best[1]; clusters[i] += clusters[j]; del clusters[j]
    return clusters


def merge_map(clusters):
    m = {}
    for cid, cl in enumerate(clusters):
        for i in cl: m[LABELS[i]] = cid
    return m


def apply_map(labels, m):
    return [m.get(x, x) for x in labels]


def loco_incity(name, m):
    lo = fast_macro(apply_map(GOLD, m), apply_map(load_arm(name, "loco" if name != "llm_gpt4omini" else "zeroshot"), m))
    return lo


# in-city preds only exist for encoders
def encoder_scores(name, m):
    P = json.load(open(f"results/preds/{name}.json"))
    inc = fast_macro(apply_map(GOLD, m), apply_map(flat(P["incity"]), m))
    loc = fast_macro(apply_map(GOLD, m), apply_map(flat(P["loco"]), m))
    return inc, loc


rng = np.random.RandomState(0)
print(f"{'K':>3s}{'sizes':>16s}{'DB_incity':>10s}{'DB_loco':>9s}{'gap':>7s}{'rand_loco_mean':>15s}{'pctile':>8s}")
merges_out = {}
for K in [14, 13, 12, 11, 10, 9, 8]:
    cl = agglomerate(C, K) if K < 14 else [[i] for i in range(14)]
    m = merge_map(cl); sizes = sorted(len(c) for c in cl)
    inc, loc = encoder_scores("distilbert", m)
    if K == 14:
        base_inc, base_loc = inc, loc
        print(f"[inv] K=14 DistilBERT incity={inc:.4f} loco={loc:.4f} (published 0.8258/0.5573)")
    # random control: same size multiset
    rand_loco = []
    for _ in range(200):
        perm = list(rng.permutation(14)); rc = []; k = 0
        for sz in sizes:
            rc.append(perm[k:k + sz]); k += sz
        rm = merge_map(rc)
        rand_loco.append(fast_macro(apply_map(GOLD, rm), apply_map(flat(json.load(open('results/preds/distilbert.json'))['loco']), rm)))
    rand_loco = np.array(rand_loco); pct = (loc > rand_loco).mean()
    groups = ["+".join(sorted(LABELS[i][:4] for i in c)) for c in cl if len(c) > 1]
    merges_out[K] = {"loco": round(loc, 4), "incity": round(inc, 4), "gap": round(inc - loc, 4),
                     "rand_loco_mean": round(float(rand_loco.mean()), 4), "pctile_vs_random": round(float(pct), 3),
                     "merged_groups": groups}
    print(f"{K:3d}{str(sizes):>16s}{inc:10.4f}{loc:9.4f}{inc-loc:7.3f}{rand_loco.mean():15.4f}{pct:8.2f}")

# ---- (a) coincidence of annotator-ambiguity and model-confusion ----
from scipy.stats import spearmanr
M = np.zeros((14, 14))
db = load_arm("distilbert", "loco")
for g, p in zip(GOLD, db):
    if g in Lidx and p in Lidx and g != p: M[Lidx[g], Lidx[p]] += 1
Cs = (C + C.T) / 2; Ms = (M + M.T) / 2; iu = np.triu_indices(14, 1)
rho_cc, p_cc = spearmanr(Cs[iu], Ms[iu])
print(f"\n(a) co-acceptability vs model-confusion: Spearman rho={rho_cc:.3f} p={p_cc:.2e}")


# ---- (b) residual multi-acceptable rows: ambiguity merge vs random ----
def resid(m):
    return np.mean([len({m[x] for x in s if x in m}) > 1 for s in ACC])


ambred = {"baseline_K14": round(float(resid({l: l for l in LABELS})), 4)}
rng2 = np.random.RandomState(0)
for K in [12, 10, 8]:
    cl = agglomerate(C, K); m = merge_map(cl); sizes = sorted(len(c) for c in cl)
    amb = resid(m); rr = []
    for _ in range(200):
        perm = list(rng2.permutation(14)); rc = []; k = 0
        for sz in sizes: rc.append(perm[k:k + sz]); k += sz
        rr.append(resid(merge_map(rc)))
    rr = np.array(rr)
    ambred[f"K{K}"] = {"ambiguity_resid": round(amb, 4), "random_mean": round(float(rr.mean()), 4),
                       "beats_frac_random": round(float((amb < rr).mean()), 3)}

out = {"leaderboard": {n: {"strictF1": round(sf1, 4), "strictAcc": round(sa, 4),
                          "lenAcc": round(la, 4), "lenCons": round(lc, 4)}
                       for n, sf1, sa, la, lc in rows},
       "leaderboard_flips_strict_to_lenient": flips,
       "coincidence_coaccept_vs_confusion": {"spearman_rho": round(float(rho_cc), 3), "p": float(p_cc)},
       "ambiguity_reduction": ambred,
       "macroF1_gap_control_NULL": {K: merges_out[K] for K in merges_out},
       "note": "macroF1 gap reduction is NOT better than random (documented null); the supported "
               "claims are the coincidence rho and the ambiguity-reduction control."}
json.dump(out, open("results/taxonomy_redesign.json", "w"), indent=2)
print("wrote results/taxonomy_redesign.json (coincidence + ambiguity-reduction + documented null)")
