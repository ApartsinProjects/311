"""
judge_agreement.py -- compare two independent LLM judges (gpt-4o-mini vs gemini-2.5-flash) on the
acceptable-set task, and recompute defensibility under each judge + consensus. Directly addresses
the judge-circularity concern (W1): if an independent-family judge yields similar lenient gains,
the effect is not a same-family artifact. Self-contained (no sklearn).
"""
import csv, os, json
from collections import defaultdict
csv.field_size_limit(10**7)
PRED = os.path.join("results", "preds")


def gold_by_city():
    g = defaultdict(list)
    with open(os.path.join("data", "eval_split.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["role"] == "test":
                g[r["city"]].append(r["label"])
    return g


def load(name):
    p = os.path.join(PRED, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def main():
    gold = gold_by_city()
    jA = load("acceptable_sets.json")                 # gpt-4o-mini judge
    jB = load("acceptable_sets_gemini25flash.json")   # gemini judge (independent family)
    if not (jA and jB):
        print("need both acceptable_sets.json and acceptable_sets_gemini25flash.json"); return

    # ---- judge agreement + per-judge label-noise ----
    exact = tot = 0; jac = 0.0
    noiseA = noiseB = noiseCons = 0
    for c, gs in gold.items():
        for i, g in enumerate(gs):
            a, b = jA[c][i], jB[c][i]
            exact += (set(a) == set(b)); jac += jaccard(a, b); tot += 1
            noiseA += g not in a; noiseB += g not in b
            noiseCons += (g not in a) and (g not in b)   # consensus: both judges reject the gold
    print(f"JUDGE AGREEMENT (gpt-4o-mini vs gemini-2.5-flash), n={tot}")
    print(f"  exact-set match: {exact/tot:.1%}   mean Jaccard: {jac/tot:.3f}")
    print(f"  city-label-noise: judgeA(gpt4omini)={noiseA/tot:.1%}  judgeB(gemini)={noiseB/tot:.1%}  "
          f"consensus(both reject)={noiseCons/tot:.1%}\n")

    # ---- lenient accuracy per arm under each judge + consensus (union of acceptable sets) ----
    arms = {"tfidf": "loco", "distilbert": "loco", "llm_gpt4omini": "zeroshot"}
    print(f"{'arm':22s}{'strict':>8s}{'lenA':>8s}{'lenB':>8s}{'lenUnion':>9s}")
    for name, proto in arms.items():
        preds = load(f"{name}.json")
        if not preds:
            continue
        p = preds[proto]
        s = la = lb = lu = n = 0
        for c, gs in gold.items():
            for i, g in enumerate(gs):
                pr = p[c][i]; a, b = jA[c][i], jB[c][i]
                s += pr == g; n += 1
                la += (pr == g) or (pr in a)
                lb += (pr == g) or (pr in b)
                lu += (pr == g) or (pr in a) or (pr in b)
        print(f"{name+'/'+proto:22s}{s/n:8.3f}{la/n:8.3f}{lb/n:8.3f}{lu/n:9.3f}")
    print("\nlenA=gpt4omini judge, lenB=gemini judge, lenUnion=either judge accepts")
    json.dump({"exact_match": round(exact/tot,4), "mean_jaccard": round(jac/tot,4),
               "noise_gpt4omini": round(noiseA/tot,4), "noise_gemini": round(noiseB/tot,4),
               "noise_consensus": round(noiseCons/tot,4)},
              open("results/judge_agreement.json","w"), indent=2)
    print("wrote results/judge_agreement.json")


if __name__ == "__main__":
    main()
