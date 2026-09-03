"""ENSEMBLE of mined rulebooks: mining is high-variance (0.667-0.854 across runs), so mine several
rulebooks from different stratified samples and majority-vote their predictions, then apply the
trigger rules on top of the voted base.

Cost note: the extra cost is in MINING (offline, one-time) and in inference calls proportional to the
number of rulebooks. Reported per member so the accuracy/cost tradeoff is explicit.

  python ensemble.py [budget] [n_members]
"""
import sys, json, os
import numpy as np
from collections import defaultdict, Counter
import semclf, triggers
from semclf import TASKS, score, paired_test, stratified_budget

def member_artifact(T, b, seed):
    p = f"results/ens_art_{b}_{seed}.json"
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))["D"]
    bud = stratified_budget(T.pool, b, seed=seed)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = max(int(0.7 * b), 20)
    D, gate = semclf.mine_rulebook(T, bud[:nm], bud[nm:], verbose=True)
    json.dump({"D": D, "gate": gate, "seed": seed}, open(p, "w"), indent=2, ensure_ascii=False)
    return D


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    T = TASKS["bloom"]
    semclf.set_trace(f"results/ensemble_trace_{b}.jsonl")
    test = T.test + T.test_dup
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
    print(f"ENSEMBLE b={b} members={M} test={len(test)}  zero-shot={azs:.4f}")

    preds_list = []
    for s in range(M):
        # seed 0 reuses the already-mined artifact so we do not pay for it twice
        art = f"results/mono_art_{b}.json"
        if s == 0 and os.path.exists(art):
            D = json.load(open(art, encoding="utf-8"))["D"]
        else:
            print(f"  [member {s}] mining ...")
            D = member_artifact(T, b, s)
        bud = stratified_budget(T.pool, b, seed=s)
        T.budget = bud; T.by = defaultdict(list)
        for r in bud: T.by[r["label"]].append(r["text"])
        p = semclf._desc_classify(T, txt, D)
        a, _, _ = score(T, p, gold)
        preds_list.append(p)
        print(f"  [member {s}] test={a:.4f}")
        sys.stdout.flush()

    # majority vote; ties -> member 0 (the reference rulebook)
    voted = []
    for i in range(len(test)):
        c = Counter(preds_list[m][i] for m in range(M))
        top, n = c.most_common(1)[0]
        voted.append(top if n > 1 else preds_list[0][i])
    av, civ, _ = score(T, voted, gold)
    pt0 = paired_test(voted, preds_list[0], gold)
    print(f"\n  VOTED base   {av:.4f} CI=({civ[0]:.3f},{civ[1]:.3f})  vs member0 {pt0['delta']:+.4f} "
          f"p={pt0['p_mcnemar']:.1e}{' *' if pt0['significant'] else ''}")

    # triggers on top of the voted base
    rules = json.load(open(f"results/triggers_{b}.json", encoding="utf-8"))["rules"]
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    final = triggers.apply_triggers(T, txt, voted, rules)
    af, cif, _ = score(T, final, gold)
    ptz = paired_test(final, zs, gold)
    rag = semclf.lexical_rag(T, txt); arag, _, _ = score(T, rag, gold)
    ptr = paired_test(rag, final, gold)
    print(f"  VOTED + triggers {af:.4f} CI=({cif[0]:.3f},{cif[1]:.3f})")
    print(f"     vs zero-shot {ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e}{' *' if ptz['significant'] else ''}")
    print(f"     vs RAG({arag:.4f}) {-ptr['delta']:+.4f} p={ptr['p_mcnemar']:.3f} "
          f"RAG_significantly_better={ptr['significant']}")
    print(f"  cost: {M} classify calls + trigger checks per item (mining is offline)")
    json.dump({"budget": b, "members": M, "member_accs": [float(score(T, p, gold)[0]) for p in preds_list],
               "voted": av, "voted_triggers": af, "zero_shot": azs, "rag": arag,
               "vs_zs": ptz, "rag_vs_ours": ptr},
              open(f"results/ensemble_{b}.json", "w"), indent=2)
    print(f"wrote results/ensemble_{b}.json")


if __name__ == "__main__":
    main()
