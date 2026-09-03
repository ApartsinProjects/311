"""Does mining scale with more labeled data? b=2000 vs 8000, same fixed test, cached + optimized."""
import json, sys
import numpy as np
from collections import defaultdict
import semclf, triggers, oaillm
from semclf import TASKS, score, stratified_budget, paired_test

T = TASKS["bloom"]
test = T.test + T.test_dup
txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
semclf.set_trace("results/bigbudget_trace.jsonl")
zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
rag_cache = {}
print(f"BIG BUDGET  test={len(test)}  zero-shot={azs:.4f}")
out = {"zero_shot": azs}
for b in [2000, 8000]:
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = int(0.7 * b)
    art = f"results/big_art_{b}.json"
    import os
    if os.path.exists(art):
        D = json.load(open(art, encoding="utf-8"))["D"]
    else:
        D, g = semclf.mine_rulebook(T, bud[:nm], bud[nm:])
        json.dump({"D": D, "gate": g}, open(art, "w"), indent=2, ensure_ascii=False)
    base = semclf._desc_classify(T, txt, D); a0, _, _ = score(T, base, gold)
    # triggers mined at this budget
    tf = f"results/big_trig_{b}.json"
    if os.path.exists(tf):
        rules = json.load(open(tf, encoding="utf-8"))["rules"]
    else:
        mtxt = [r["text"] for r in bud[:nm]][:1500]; mgold = [r["label"] for r in bud[:nm]][:1500]
        vtxt = [r["text"] for r in bud[nm:]][:400]; vgold = [r["label"] for r in bud[nm:]][:400]
        mb = semclf._desc_classify(T, mtxt, D); vb = semclf._desc_classify(T, vtxt, D)
        rules, _ = triggers.mine_triggers(T, mtxt, mgold, mb, vtxt, vgold, vb)
        json.dump({"rules": rules}, open(tf, "w"), indent=2, ensure_ascii=False)
    final = triggers.apply_triggers(T, txt, base, rules)
    a1, ci1, _ = score(T, final, gold)
    ptz = paired_test(final, zs, gold)
    rag = semclf.lexical_rag(T, txt); ar, _, _ = score(T, rag, gold)
    ptr = paired_test(rag, final, gold)
    out[b] = {"rulebook": a0, "final": a1, "rag": ar, "n_rules": len(rules),
              "vs_zs_p": ptz["p_mcnemar"], "rag_vs_ours_p": ptr["p_mcnemar"]}
    print(f"  b={b:5d}: rulebook={a0:.4f}  +triggers={a1:.4f} ({len(rules)} rules) CI=({ci1[0]:.3f},{ci1[1]:.3f})")
    print(f"          vs zero-shot {ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e} | RAG={ar:.4f} gap={ar-a1:+.4f} p={ptr['p_mcnemar']:.3f}")
    sys.stdout.flush()
json.dump(out, open("results/bigbudget.json", "w"), indent=2)
print(f"cache: {oaillm.cache_stats()}")
