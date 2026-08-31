"""
score_aligned.py -- co-compute ALL metrics from the aligned per-example predictions.
Every arm predicted on the SAME frozen test rows (results/preds/*.json). We report, per
arm/protocol: per-city macro-F1, pooled macro-F1 with 95% bootstrap CI, unweighted
mean-over-cities, and SF-excluded variants. Also pooled per-class F1 and paired-bootstrap
significance between the cross-city arms. Writes results/final_scores.json.
"""
import os, json, glob
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from eval_common import load_split, PRED_DIR

RNG = np.random.RandomState(123)
B = 1000  # bootstrap resamples


def gold_by_city():
    sp = load_split()
    return {c: [y for _, y in rows] for c, rows in sp["test"].items()}


def macro(gold, pred):
    return f1_score(gold, pred, average="macro", zero_division=0)


def _encode(pairs):
    """(gold,pred) tuples -> integer label codes over their own union, and L."""
    g = np.array([a for a, _ in pairs]); p = np.array([b for _, b in pairs])
    _, codes = np.unique(np.concatenate([g, p]), return_inverse=True)
    n = len(g)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), int(codes.max()) + 1


def _macro_from_cms(cms, L):
    """cms: (M, L*L) flat confusion counts -> (M,) macro-F1 over labels present in each resample.
    Matches sklearn f1_score(average='macro', zero_division=0, labels=None): a label with
    2tp+fp+fn>0 is exactly one present in the resampled gold-or-pred union."""
    c = cms.reshape(-1, L, L)
    tp = np.diagonal(c, axis1=1, axis2=2).astype(np.float64)
    fp = c.sum(1) - tp; fn = c.sum(2) - tp
    d = 2 * tp + fp + fn
    f1 = np.where(d > 0, 2 * tp / np.where(d > 0, d, 1.0), 0.0)
    present = d > 0
    return (f1 * present).sum(1) / present.sum(1)


def _boot_cms(cell, L, idx):
    """cell: (n,) per-example confusion code gi*L+pi. idx: (B,n) resample indices."""
    B_ = idx.shape[0]
    flat = cell[idx] + (np.arange(B_, dtype=np.int64)[:, None] * (L * L))
    return np.bincount(flat.ravel(), minlength=B_ * L * L).reshape(B_, L * L)


def pooled_ci(pairs):
    """pairs: list of (gold,pred). Vectorized bootstrap 95% CI of pooled macro-F1.
    RNG.randint(0,n,size=(B,n)) yields the identical MT19937 stream as B sequential
    randint(0,n,n) draws, so this matches the previous scorer to 4dp."""
    gi, pi, L = _encode(pairs); n = len(gi)
    cell = gi * L + pi
    point = float(_macro_from_cms(np.bincount(cell, minlength=L * L)[None, :], L)[0])
    idx = RNG.randint(0, n, size=(B, n))
    boots = _macro_from_cms(_boot_cms(cell, L, idx), L)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def eval_arm(name, protocol_preds, gold, accept=None):
    """protocol_preds: {city:[preds]}. accept: {city:[set_of_labels]} for lenient scoring."""
    cities = list(gold)
    per_city = {}
    pooled_pairs, pooled_pairs_nosf = [], []
    lenient_hits = lenient_n = 0
    for c in cities:
        if c not in protocol_preds:
            continue
        g = gold[c]; p = protocol_preds[c]
        m = macro(g, p); a = accuracy_score(g, p)
        rec = {"macroF1": round(m, 4), "acc": round(a, 4), "n": len(g), "n_classes": len(set(g))}
        if accept and c in accept:
            lc = [1 if (pr == g[i] or pr in accept[c][i]) else 0 for i, pr in enumerate(p)]
            rec["lenient_acc"] = round(sum(lc) / len(lc), 4)
            lenient_hits += sum(lc); lenient_n += len(lc)
        per_city[c] = rec
        pooled_pairs += list(zip(g, p))
        if c != "SanFrancisco":
            pooled_pairs_nosf += list(zip(g, p))
    pt, lo, hi = pooled_ci(pooled_pairs)
    ptn, lon, hin = pooled_ci(pooled_pairs_nosf)
    unw = float(np.mean([per_city[c]["macroF1"] for c in per_city]))
    unw_nosf = float(np.mean([per_city[c]["macroF1"] for c in per_city if c != "SanFrancisco"]))
    res = {
        "per_city": per_city,
        "pooled_macroF1": round(pt, 4), "pooled_CI": [round(lo, 4), round(hi, 4)],
        "pooled_macroF1_noSF": round(ptn, 4), "pooled_CI_noSF": [round(lon, 4), round(hin, 4)],
        "unweighted_mean": round(unw, 4), "unweighted_mean_noSF": round(unw_nosf, 4),
        "_pooled_pairs": pooled_pairs,  # kept for significance, stripped before save
    }
    if lenient_n:
        res["pooled_strict_acc"] = round(accuracy_score([a for a, _ in pooled_pairs],
                                                         [b for _, b in pooled_pairs]), 4)
        res["pooled_lenient_acc"] = round(lenient_hits / lenient_n, 4)
    return res


def paired_bootstrap(pairsA, pairsB, label):
    """Both aligned over same pooled examples. P(A>B) via bootstrap of macro-F1 difference.
    One shared resample per draw (as before); vectorized via confusion-count bincounts."""
    giA, piA, LA = _encode(pairsA); giB, piB, LB = _encode(pairsB)
    n = len(giA); cellA = giA * LA + piA; cellB = giB * LB + piB
    idx = RNG.randint(0, n, size=(B, n))
    diffs = _macro_from_cms(_boot_cms(cellA, LA, idx), LA) - _macro_from_cms(_boot_cms(cellB, LB, idx), LB)
    return {"label": label, "mean_diff": round(float(diffs.mean()), 4),
            "P(A>B)": round(float((diffs > 0).mean()), 3),
            "CI": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]}


def main():
    gold = gold_by_city()
    # optional acceptable-sets for lenient scoring + gold-noise rate
    accept = None; gold_noise = None
    ap = os.path.join(PRED_DIR, "acceptable_sets.json")
    if os.path.exists(ap):
        accept = json.load(open(ap, encoding="utf-8"))
        nn = tot = 0
        for c, g in gold.items():
            if c in accept:
                nn += sum(1 for i, y in enumerate(g) if y not in accept[c][i]); tot += len(g)
        gold_noise = round(nn / tot, 4)
        print(f"CITY-LABEL NOISE (gold not in judge-acceptable set): {gold_noise:.1%}\n")

    arms = {}
    for pf in sorted(glob.glob(os.path.join(PRED_DIR, "*.json"))):
        name = os.path.splitext(os.path.basename(pf))[0]
        if name.startswith("acceptable_sets"):
            continue
        data = json.load(open(pf, encoding="utf-8"))
        for proto, preds in data.items():
            arms[f"{name}/{proto}"] = eval_arm(f"{name}/{proto}", preds, gold, accept)

    print(f"{'arm/protocol':28s}{'pooled':>9s}{'95% CI':>16s}{'noSF':>8s}{'unwMean':>9s}{'unw_noSF':>10s}")
    for k, v in arms.items():
        ci = f"[{v['pooled_CI'][0]:.2f},{v['pooled_CI'][1]:.2f}]"
        print(f"{k:28s}{v['pooled_macroF1']:9.3f}{ci:>16s}{v['pooled_macroF1_noSF']:8.3f}"
              f"{v['unweighted_mean']:9.3f}{v['unweighted_mean_noSF']:10.3f}")

    # significance: compare each cross-city arm against the TF-IDF baseline (linear, fast + sufficient),
    # plus the key LLM-vs-DistilBERT contrast.
    sig = []
    cross = {k: v for k, v in arms.items() if ("loco" in k or "zeroshot" in k)}
    ref = "tfidf/loco"
    pairs = [(a, ref) for a in cross if a != ref]
    for extra in [("llm_gpt4omini/zeroshot", "distilbert/loco"), ("llm_gpt4o/zeroshot", "distilbert/loco")]:
        if extra[0] in cross and extra[1] in cross:
            pairs.append(extra)
    for a, b in pairs:
        if a in cross and b in cross:
            sig.append(paired_bootstrap(cross[a]["_pooled_pairs"], cross[b]["_pooled_pairs"], f"{a} vs {b}"))
    print("\nPaired-bootstrap significance (cross-city arms):")
    for s in sig:
        print(f"  {s['label']:45s} diff={s['mean_diff']:+.3f} CI[{s['CI'][0]:+.3f},{s['CI'][1]:+.3f}] P(A>B)={s['P(A>B)']}")

    len_rows = [(k, v.get("pooled_strict_acc"), v.get("pooled_lenient_acc"))
                for k, v in arms.items() if "pooled_lenient_acc" in v]
    if len_rows:
        print("\nStrict vs lenient accuracy (defensibility-adjusted):")
        for k, s, l in len_rows:
            print(f"  {k:28s} strict={s:.3f}  lenient={l:.3f}  artifact_share={ (l-s):+.3f}")

    for v in arms.values():
        v.pop("_pooled_pairs", None)
    json.dump({"arms": arms, "significance": sig, "gold_noise_rate": gold_noise},
              open("results/final_scores.json", "w"), indent=2)
    print("\nwrote results/final_scores.json")


if __name__ == "__main__":
    main()
