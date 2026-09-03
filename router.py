"""EASY/HARD ROUTER + hard-subset method comparison.

Stage 1: learn a binary router that predicts whether ZERO-SHOT will be correct on an item
         ("easy") or not ("hard"). Two routers, both trained on the same budget:
           (a) mined-rules router  -- convention-style rules mined for the binary easy/hard task
           (b) TF-IDF+LR router    -- free, non-LLM reference
Stage 2: compare every method ON THE HARD SUBSET (where methods actually differ), reported two ways:
           * ORACLE-hard  = the true zero-shot failures (clean method comparison, router-independent)
           * ROUTER-hard  = what the router actually flags (realistic)
Stage 3: cascade overall = zero-shot on easy + method on hard  (depends on router quality)

  python router.py [budget]
"""
import sys, json, re
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import semclf
from semclf import TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, _freq_examples, _traced_call, chat_many
import corrections

MAXR = 12


# ---------------- router (a): mined binary rules ----------------
def mine_router_rules(T, mine_txt, mine_hard, rounds=3, per_round=5):
    """Mine rules that say when the general classifier is UNRELIABLE on this kind of request."""
    rules = []
    hard_ex = [mine_txt[i] for i in range(len(mine_txt)) if mine_hard[i]]
    easy_ex = [mine_txt[i] for i in range(len(mine_txt)) if not mine_hard[i]]
    sysm = (f"You write rules that flag which {T.item}s a general-purpose classifier will get WRONG, "
            f"because the organization files them by convention rather than by their literal wording. "
            f"Write rules that identify RISKY/ambiguous requests -- not the correct category. "
            f"Each rule <=25 words, concrete and checkable. No preamble.")
    for rnd in range(rounds):
        msg = (f"Requests the classifier got WRONG (HARD):\n{_freq_examples(hard_ex, 12)}\n\n"
               f"Requests it got RIGHT (EASY):\n{_freq_examples(easy_ex, 8)}\n\n"
               + (f"Existing flags (do not duplicate):\n" + "\n".join(f"- {r}" for r in rules) + "\n\n" if rules else "")
               + f"Write {per_round} NEW rules, one per line, each describing a kind of request that is HARD.")
        o = _traced_call("router_rules", [{"role": "system", "content": sysm}, {"role": "user", "content": msg}],
                         MINE_MODEL, 400, round=rnd)
        for line in (o or "").splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if 10 < len(line) < 200: rules.append(line)
        rules = rules[:MAXR]
        if len(rules) >= MAXR: break
    return rules


def apply_router_rules(T, texts, rules):
    block = "\n".join(f"- {r}" for r in rules)
    sysm = (f"Decide whether a general-purpose classifier is LIKELY TO FAIL on this {T.item} "
            f"(because of the organization's filing conventions). Reply ONLY HARD or EASY.")
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"A {T.item} is HARD if any of these apply:\n{block}\n\n"
              f"{T.item.capitalize()}: {t[:TRUNC]}\nHARD or EASY:"}] for t in texts]
    return [(o or "").strip().upper().startswith("H") for o in chat_many(msgs, model=CLF_MODEL, max_tokens=4)]


# ---------------- router (b): TF-IDF + LR (free) ----------------
def lr_router(mine_txt, mine_hard, test_txt):
    v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
    X = v.fit_transform(mine_txt)
    clf = LogisticRegression(max_iter=600, C=2.0, class_weight="balanced").fit(X, mine_hard)
    p = clf.predict_proba(v.transform(test_txt))[:, list(clf.classes_).index(True)]
    return p


def router_quality(name, flag, true_hard):
    tp = sum(1 for i in range(len(flag)) if flag[i] and true_hard[i])
    fp = sum(1 for i in range(len(flag)) if flag[i] and not true_hard[i])
    fn = sum(1 for i in range(len(flag)) if not flag[i] and true_hard[i])
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"  router[{name:9s}] flagged={sum(flag):4d}/{len(flag)}  precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}")
    return {"flagged": int(sum(flag)), "precision": prec, "recall": rec, "f1": f1}


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    semclf.set_trace(f"results/router_trace_{b}.jsonl")
    bud = semclf.stratified_budget(T.pool, b); T.budget = bud
    T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    n_mine = int(0.7 * b); mine, val = bud[:n_mine], bud[n_mine:]
    test = T.test + T.test_dup
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    t_txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"ROUTER  budget={b} mine={len(mine)} test={len(test)}")

    # --- ground truth for routing: where does zero-shot fail? ---
    m_zs = semclf.zero_shot(T, m_txt)
    m_hard = [m_zs[i] != m_gold[i] for i in range(len(mine))]
    t_zs = semclf.zero_shot(T, t_txt)
    t_hard = [t_zs[i] != gold[i] for i in range(len(test))]
    zs_acc, zs_ci, _ = score(T, t_zs, gold)
    print(f"  zero-shot overall={zs_acc:.4f}; hard rate: mine={np.mean(m_hard):.3f} test={np.mean(t_hard):.3f}")

    # --- routers ---
    rules = mine_router_rules(T, m_txt, m_hard)
    print(f"\n=== {len(rules)} mined HARD-flag rules ===")
    for r in rules[:8]: print("  -", r[:140])
    flag_rules = apply_router_rules(T, t_txt, rules)
    q_rules = router_quality("mined", flag_rules, t_hard)
    p_lr = lr_router(m_txt, m_hard, t_txt)
    thr = np.quantile(p_lr, 1 - np.mean(m_hard))       # flag same rate as observed on mine
    flag_lr = list(p_lr >= thr)
    q_lr = router_quality("tfidf-LR", flag_lr, t_hard)

    # --- methods to compare on the hard subsets ---
    corr = json.load(open(f"results/corrections_{b}.json", encoding="utf-8"))["rules"] \
        if __import__("os").path.exists(f"results/corrections_{b}.json") else []
    art_p = f"results/bench_art_{b}.json"
    D = json.load(open(art_p, encoding="utf-8"))["D"] if __import__("os").path.exists(art_p) else None
    methods = {"zero_shot": lambda tx: semclf.zero_shot(T, tx),
               "RAG": lambda tx: semclf.lexical_rag(T, tx)}
    if corr: methods["corrections"] = lambda tx: corrections.classify(T, tx, corr)
    if D: methods["mined_flat"] = lambda tx: semclf._desc_classify(T, tx, D)

    out = {"budget": b, "zero_shot_overall": zs_acc, "router_mined": q_rules, "router_lr": q_lr,
           "hard_rate_test": float(np.mean(t_hard)), "on_hard": {}, "cascade": {}}
    for hname, hflag in [("ORACLE-hard", t_hard), ("ROUTER-hard(mined)", flag_rules), ("ROUTER-hard(LR)", flag_lr)]:
        idx = [i for i in range(len(test)) if hflag[i]]
        if not idx: continue
        print(f"\n--- {hname}: n={len(idx)} ---")
        for mname, fn in methods.items():
            preds = fn([t_txt[i] for i in idx])
            acc = float(np.mean([preds[j] == gold[idx[j]] for j in range(len(idx))]))
            # cascade overall: zero-shot on the rest, this method on the flagged set
            full = list(t_zs)
            for j, i in enumerate(idx): full[i] = preds[j]
            casc = float(np.mean([full[i] == gold[i] for i in range(len(test))]))
            out["on_hard"].setdefault(hname, {})[mname] = acc
            out["cascade"].setdefault(hname, {})[mname] = casc
            print(f"    {mname:12s} on-hard={acc:.4f}   cascade-overall={casc:.4f}")
    json.dump(out, open(f"results/router_{b}.json", "w"), indent=2)
    print(f"\nwrote results/router_{b}.json")


if __name__ == "__main__":
    main()
