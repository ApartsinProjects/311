"""Reference baselines for a task on the fixed test: zero-shot, diverse k-shot, RAG, fine-tuned.
Reports lift over zero-shot with paired significance. Inference = gpt-4o-mini (fair); fine-tuned is
local sklearn (no LLM). These are the cheap reference points; the TUNED prompt-only baselines
(val-optimized k-shot = kopt.py, MIPROv2 = miprov2.py) and our method are run separately.

  python baselines.py [task]
"""
import sys, json
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import semclf
from semclf import TASKS, score, paired_test, stratified_budget


def run(task):
    T = TASKS[task]
    bud = stratified_budget(T.pool, 2000, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    test = T.test + T.test_dup
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    is_dup = set(semclf.norm(r["text"]) for r in bud)
    maj = Counter(gold).most_common(1)[0][1] / len(gold)
    print(f"BASELINES {task}: classes={len(T.LBL)} test={len(test)} majority={maj:.4f}")

    zs = semclf.zero_shot(T, txt); azs, ciz, _ = score(T, zs, gold)
    rows = {"majority": maj, "zero_shot": azs}
    print(f"  {'majority':22s} {maj:.4f}")
    print(f"  {'zero-shot':22s} {azs:.4f}  CI=({ciz[0]:.3f},{ciz[1]:.3f})")

    def add(name, preds):
        a, ci, unp = score(T, preds, gold)
        pt = paired_test(preds, zs, gold)
        rows[name] = {"acc": a, "ci": ci, "unparsed": unp, "vs_zs": pt["delta"], "p": pt["p_mcnemar"]}
        sig = "*" if pt["significant"] else " "
        print(f"  {name:22s} {a:.4f}  CI=({ci[0]:.3f},{ci[1]:.3f})  lift={pt['delta']:+.4f} p={pt['p_mcnemar']:.1e}{sig} unp={unp:.3f}")

    add("k-shot diverse", semclf.kshot_per_class(T, txt, k=2, select="diverse"))
    add("RAG (needs store)", semclf.lexical_rag(T, txt))
    # fine-tuned: local TF-IDF + LR (no LLM)
    v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
    X = v.fit_transform([r["text"] for r in bud])
    clf = LogisticRegression(max_iter=700, C=6.0).fit(X, [r["label"] for r in bud])
    add("fine-tuned (needs train)", list(clf.predict(v.transform(txt))))
    json.dump(rows, open(f"results/baselines_{task}.json", "w"), indent=2)
    print(f"  wrote results/baselines_{task}.json")


if __name__ == "__main__":
    tasks = sys.argv[1:] or ["bloom", "hupd", "mimic"]
    for t in tasks:
        run(t); print()
