"""SEED VARIANCE of rule mining: is the mining process stable, or is a good rulebook a lucky draw?
Re-mines the artifact at each budget with several stratified-sample seeds and evaluates every one on
the fixed test. Reports mean/spread per budget, plus the val->test relationship (does the gate select
the good rulebooks?).

  python seedvar.py [seeds] [budgets]
"""
import sys, json, os, copy
import numpy as np
from collections import defaultdict, Counter
import semclf, flows
from semclf import TASKS, score, stratified_budget, paired_test

SEEDS = [0, 1, 2]
BUDGETS = [200, 1000, 2000]


def mine_one(T, b, seed):
    """Mine a rulebook from a stratified budget drawn with `seed`. Returns (D, val_acc)."""
    bud = stratified_budget(T.pool, b, seed=seed)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    n_mine = max(int(0.7 * b), 20)
    mine, val = bud[:n_mine], bud[n_mine:]
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    D = semclf._seed(T); tried = set()
    def vacc(DD):
        vp = semclf._desc_classify(T, v_txt, DD)
        return float(np.mean([vp[i] == v_gold[i] for i in range(len(val))]))
    best_D, best_v = copy.deepcopy(D), vacc(D)
    for rnd in range(5):
        m_pred = semclf._desc_classify(T, m_txt, D)
        conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine))
                       if m_pred[i] != m_gold[i] and m_pred[i] != "UNPARSED")
        bt = [(g, p) for (g, p), _ in conf.most_common() if (g, p) not in tried and conf[(g, p)] >= 2][:8]
        if not bt: break
        base = {c: len(T.by[c]) for c in T.LBL}
        for gt, pr in bt:
            exs = [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and m_pred[i] == pr]
            contrast = [m_txt[i] for i in range(len(mine)) if m_gold[i] == pr and m_pred[i] == pr]
            diag = semclf._diagnose(T, gt, pr, exs, contrast, D, n_err=conf[(gt, pr)])
            upd = semclf._refine(T, gt, pr, exs, D, contrast=contrast, base_rates=base,
                                 n_err=conf[(gt, pr)], diag=diag)
            semclf.apply_update(D, upd); tried.add((gt, pr))
        v = vacc(D)
        if v >= best_v: best_v, best_D = v, copy.deepcopy(D)
        else: D = copy.deepcopy(best_D); tried -= set(bt)
    return best_D, best_v


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else SEEDS
    budgets = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else BUDGETS
    T = TASKS["bloom"]
    test = T.test + T.test_dup
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    semclf.set_trace("results/seedvar_trace.jsonl")
    zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
    print(f"SEED VARIANCE  test={len(test)}  zero-shot={azs:.4f}  seeds={seeds} budgets={budgets}")
    out = {}
    for b in budgets:
        accs = []
        for s in seeds:
            p = f"results/seedart_{b}_{s}.json"
            if os.path.exists(p):
                D, v = json.load(open(p, encoding="utf-8"))["D"], json.load(open(p, encoding="utf-8"))["val"]
            else:
                D, v = mine_one(T, b, s)
                json.dump({"D": D, "val": v}, open(p, "w"), indent=2, ensure_ascii=False)
            # restore the same budget for classification context
            bud = stratified_budget(T.pool, b, seed=s)
            T.budget = bud; T.by = defaultdict(list)
            for r in bud: T.by[r["label"]].append(r["text"])
            preds = semclf._desc_classify(T, txt, D)
            a, ci, _ = score(T, preds, gold)
            pt = paired_test(preds, zs, gold)
            accs.append(a)
            print(f"  b={b:5d} seed={s}: val={v:.3f}  TEST={a:.4f}  vs_zs={pt['delta']:+.4f} "
                  f"p={pt['p_mcnemar']:.1e}{' *' if pt['significant'] else ''}")
            sys.stdout.flush()
        out[b] = {"accs": accs, "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                  "min": float(min(accs)), "max": float(max(accs))}
        print(f"  --> b={b}: mean={np.mean(accs):.4f} sd={np.std(accs):.4f} "
              f"range=[{min(accs):.4f},{max(accs):.4f}]  (zero-shot={azs:.4f})")
    json.dump({"zero_shot": azs, "by_budget": out}, open("results/seedvar.json", "w"), indent=2)
    print("\nwrote results/seedvar.json")


if __name__ == "__main__":
    main()
