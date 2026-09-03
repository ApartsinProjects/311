"""FINAL TABLE: all methods x budgets on the fixed test, with PAIRED significance vs zero-shot.
Uses the response cache (reruns are free) and the stratified budgets + mined artifacts from bench.prepare.

  python finaltable.py
"""
import json, os
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import semclf, oaillm
from semclf import TASKS, score, paired_test, stratified_budget

BUDGETS = [200, 1000, 2000]


def main():
    T = TASKS["bloom"]
    test = T.test + T.test_dup
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    ref = set(semclf.norm(r["text"]) for r in stratified_budget(T.pool, 2000))
    is_dup = [semclf.norm(t) in ref for t in txt]
    maj = Counter(gold).most_common(1)[0][1] / len(gold)
    print(f"FINAL TABLE  test={len(test)} labels={len(T.LBL)} majority_baseline={maj:.4f}")
    rows = []

    def add(name, budget, preds, ref_preds=None):
        acc, ci, unp = score(T, preds, gold)
        nd = [i for i in range(len(test)) if not is_dup[i]]; dd = [i for i in range(len(test)) if is_dup[i]]
        nov = float(np.mean([preds[i] == gold[i] for i in nd]))
        dup = float(np.mean([preds[i] == gold[i] for i in dd]))
        pt = paired_test(preds, ref_preds, gold) if ref_preds is not None else None
        rows.append({"method": name, "budget": budget, "acc": acc, "ci": ci, "novel": nov, "dup": dup,
                     "unparsed": unp, "vs_zs": pt})
        sig = ""
        if pt: sig = f"  vs_zs={pt['delta']:+.4f} p={pt['p_mcnemar']:.1e}{' *' if pt['significant'] else ''}"
        print(f"  {name:22s} b={str(budget):>5s} acc={acc:.4f} CI=({ci[0]:.3f},{ci[1]:.3f}) "
              f"nov={nov:.3f} dup={dup:.3f}{sig}")

    # zero-shot: budget-independent reference
    zs = semclf.zero_shot(T, txt)
    add("zero-shot", "-", zs)
    for b in BUDGETS:
        bud = stratified_budget(T.pool, b)
        T.budget = bud; T.by = defaultdict(list)
        for r in bud: T.by[r["label"]].append(r["text"])
        add("RAG", b, semclf.lexical_rag(T, txt), zs)
        add("k-shot/class", b, semclf.kshot_per_class(T, txt), zs)
        p = f"results/bench_art_{b}.json"
        if os.path.exists(p):
            D = json.load(open(p, encoding="utf-8"))["D"]
            add("mined-rulebook", b, semclf._desc_classify(T, txt, D), zs)
        # fine-tuned (local, free)
        v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
        X = v.fit_transform([r["text"] for r in bud])
        clf = LogisticRegression(max_iter=600, C=6.0).fit(X, [r["label"] for r in bud])
        add("fine-tuned", b, list(clf.predict(v.transform(txt))), zs)
    json.dump({"majority": maj, "rows": rows}, open("results/final_table.json", "w"), indent=2)
    print(f"\ncache: {oaillm.cache_stats()}")
    print("wrote results/final_table.json")


if __name__ == "__main__":
    main()
