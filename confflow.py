"""CONFUSION-DRIVEN decomposition: mine the multiclass structure from the error matrix instead of
designing it a priori.

Why: the exhaustive/semantic decompositions failed for diagnosable reasons.
  * hierarchical (0.747): groups were semantic, so confusable classes sat in DIFFERENT groups and a
    router mistake was unrecoverable.
  * ECOC (0.708): random bits forced unrelated classes into one side, so each binary question was
    incoherent.
Fix: build the structure FROM the confusion matrix.
  * grouping: classes that are confused with each other go in the SAME group, so the router only has
    to make easy distinctions and the hard decision happens inside the group with focused rules.
  * ECOC: each bit splits along a real confusion boundary, so every binary question is answerable.

  python confflow.py [budget]
"""
import sys, json
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import (TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, paired_test, stratified_budget,
                    _dline, chat_many)


def _pick(o, n):
    import re
    m = re.search(r"\d+", o or "")
    return (int(m.group(0)) - 1) if (m and 1 <= int(m.group(0)) <= n) else -1


def confusion_groups(labels, conf, max_group=6):
    """Agglomerative grouping: merge the two classes most confused with each other, repeatedly.
    Symmetric confusion weight = errors(a->b) + errors(b->a)."""
    groups = [[l] for l in labels]
    def w(g1, g2):
        return sum(conf.get((a, b), 0) + conf.get((b, a), 0) for a in g1 for b in g2)
    while True:
        best, bi, bj = 0, -1, -1
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if len(groups[i]) + len(groups[j]) > max_group: continue
                s = w(groups[i], groups[j])
                if s > best: best, bi, bj = s, i, j
        if bi < 0 or best == 0: break
        groups[bi] = groups[bi] + groups[bj]; groups.pop(bj)
    return groups


def route(T, texts, groups, gnames):
    book = "\n".join(f"{i+1}. {gnames[i]}: " + ", ".join(g) for i, g in enumerate(groups))
    sysm = (f"Choose which GROUP the {T.item} belongs to. Groups collect {T.label}s that are easy to "
            f"confuse, so you only need the broad distinction. Reply ONLY the group number.")
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"Groups:\n{book}\n\n{T.item.capitalize()}: {t[:TRUNC]}\nGroup:"}]
            for t in texts]
    out = []
    for o in chat_many(msgs, model=CLF_MODEL, max_tokens=6):
        k = _pick(o, len(groups)); out.append(k if k >= 0 else 0)
    return out


def resolve(T, texts, cands, D):
    """Pick within the group, using the mined rulebook entries for just those classes."""
    msgs = []
    for i, c in enumerate(cands):
        book = "\n".join(f"{j+1}. {_dline(T, x, D)}" for j, x in enumerate(c))
        msgs.append([{"role": "system", "content":
                      f"Pick the single best {T.label} from the candidates. These are the organization's "
                      f"filing conventions and may contradict the literal wording; follow them. Reply ONLY the number."},
                     {"role": "user", "content": f"Candidates:\n{book}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\nNumber:"}])
    out = []
    for o, c in zip(chat_many(msgs, model=CLF_MODEL, max_tokens=6), cands):
        k = _pick(o, len(c)); out.append(c[k] if k >= 0 else c[0])
    return out


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    semclf.set_trace(f"results/confflow_trace_{b}.jsonl")
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = int(0.7 * b); mine = bud[:nm]
    test = T.test + T.test_dup
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    D = json.load(open(f"results/mono_art_{b}.json", encoding="utf-8"))["D"]
    print(f"CONFUSION-DRIVEN FLOW b={b} test={len(test)}")

    # 1) confusion matrix of the base (mined rulebook) on the mine set
    m_pred = semclf._desc_classify(T, m_txt, D)
    conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine)) if m_pred[i] != m_gold[i])
    print(f"  base mine-acc={np.mean([m_pred[i]==m_gold[i] for i in range(len(mine))]):.3f}, "
          f"{len(conf)} distinct confusions")

    # 2) group confusable classes TOGETHER
    groups = confusion_groups(T.LBL, conf)
    gnames = [" / ".join(g[:3]) + ("..." if len(g) > 3 else "") for g in groups]
    print(f"  {len(groups)} confusion-driven groups:")
    for g in groups: print(f"     {g}")

    # 3) route + resolve on test
    base = semclf._desc_classify(T, txt, D); a0, _, _ = score(T, base, gold)
    gi = route(T, txt, groups, gnames)
    cands = [groups[k] for k in gi]
    rec = np.mean([gold[i] in cands[i] for i in range(len(test))])
    single = [i for i in range(len(test)) if len(cands[i]) == 1]
    preds = resolve(T, txt, cands, D)
    a1, ci1, _ = score(T, preds, gold)
    pt = paired_test(preds, base, gold)
    zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
    print(f"\n=== RESULTS (n={len(test)}) ===")
    print(f"  zero-shot                    {azs:.4f}")
    print(f"  flat mined rulebook (base)   {a0:.4f}")
    print(f"  confusion-driven hierarchical{a1:.4f} CI=({ci1[0]:.3f},{ci1[1]:.3f})  "
          f"vs base {pt['delta']:+.4f} p={pt['p_mcnemar']:.1e}{' *' if pt['significant'] else ''}")
    print(f"  router recall (gold in chosen group) = {rec:.3f}   groups resolved trivially: {len(single)}")
    json.dump({"budget": b, "groups": groups, "zero_shot": azs, "base": a0, "flow": a1,
               "router_recall": float(rec), "vs_base": pt},
              open(f"results/confflow_{b}.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote results/confflow_{b}.json")


if __name__ == "__main__":
    main()
