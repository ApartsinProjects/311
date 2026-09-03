"""Single-seed sanity run: does mining now behave (no degradation as budget grows)?"""
import json, sys
import numpy as np
from collections import defaultdict
import semclf
from semclf import TASKS, score, stratified_budget, paired_test

T = TASKS["bloom"]
test = T.test + T.test_dup
txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
semclf.set_trace("results/mono_trace.jsonl")
zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
print(f"MONOTONICITY CHECK (seed 0)  test={len(test)}  zero-shot={azs:.4f}")
out = {}
for b in [200, 1000, 2000]:
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = max(int(0.7*b), 20)
    D, gate = semclf.mine_rulebook(T, bud[:nm], bud[nm:])
    preds = semclf._desc_classify(T, txt, D)
    a, ci, _ = score(T, preds, gold)
    pt = paired_test(preds, zs, gold)
    out[b] = {"acc": a, "gate": gate, "vs_zs": pt["delta"], "p": pt["p_mcnemar"]}
    print(f"  b={b:5d}: gate={gate:.3f}  TEST={a:.4f} CI=({ci[0]:.3f},{ci[1]:.3f})  "
          f"vs_zs={pt['delta']:+.4f} p={pt['p_mcnemar']:.1e}{' *' if pt['significant'] else ''}")
    json.dump({"D": D}, open(f"results/mono_art_{b}.json","w"), indent=2, ensure_ascii=False)
    sys.stdout.flush()
accs=[out[b]["acc"] for b in [200,1000,2000]]
print(f"\n  monotone (no degradation with more data)? {all(accs[i]<=accs[i+1]+0.01 for i in range(len(accs)-1))}")
print(f"  all >= zero-shot? {all(a>=azs for a in accs)}")
json.dump({"zero_shot": azs, "by_budget": out}, open("results/mono.json","w"), indent=2)
