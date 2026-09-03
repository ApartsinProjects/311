"""MAJORITY FLOW (all-LLM): gate -> zero-shot for reliable items -> mined rules for hard items.

Scope: the 11 majority classes (>=400 examples, 97.7% of data). Minority classes are routed out by a
separate gate and handled by their own flow (later).

Pipeline, entirely LLM calls over mined text:
  1. GATE: mined rules flag items where zero-shot is unreliable        (1 call)
  2. EASY  -> plain zero-shot                                          (1 call)
  3. HARD  -> rules mined ONLY on hard examples (focused, undiluted)   (1 call)

Mining hygiene: the budget is split so the gate and the hard-rules are mined on data held out from
each other, and the hard-rules are mined on the REAL router-flagged distribution (with its false
alarms), not on oracle-hard.

  python majflow.py [budget]
"""
import sys, json
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import (TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, paired_test, stratified_budget,
                    _freq_examples, _traced_call, chat_many, _dline)

MINCOUNT = 400


def majority_classes(T):
    c = Counter(r["label"] for r in T.pool)
    return sorted([l for l, n in c.items() if n >= MINCOUNT])


# ---------------- GATE: mined rules that flag zero-shot-unreliable items ----------------
def mine_gate(T, txt, hard, rounds=3, per_round=5, gate_val=None):
    rules = []
    hx = [txt[i] for i in range(len(txt)) if hard[i]]
    ex = [txt[i] for i in range(len(txt)) if not hard[i]]
    sysm = (f"You write rules that flag which {T.item}s a general-purpose classifier will get WRONG, "
            f"because this organization files them by convention rather than by their literal wording. "
            f"Flag RISK, do not name the category. Each rule <=25 words, concrete and checkable.")
    for rnd in range(rounds):
        msg = (f"HARD (classifier got these wrong):\n{_freq_examples(hx, 12)}\n\n"
               f"EASY (it got these right):\n{_freq_examples(ex, 8)}\n\n"
               + (f"Existing flags (do not duplicate):\n" + "\n".join(f"- {r}" for r in rules) + "\n\n" if rules else "")
               + f"Write {per_round} NEW rules, one per line, each describing a kind of {T.item} that is HARD.")
        o = _traced_call("gate_rules", [{"role": "system", "content": sysm}, {"role": "user", "content": msg}],
                         MINE_MODEL, 400, round=rnd)
        for line in (o or "").splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if 10 < len(line) < 200: rules.append(line)
        if len(rules) >= 12: break
    return rules[:12]


def apply_gate(T, texts, rules):
    block = "\n".join(f"- {r}" for r in rules)
    sysm = (f"Decide whether a general-purpose classifier is LIKELY TO FAIL on this {T.item} because of "
            f"the organization's filing conventions. Reply ONLY HARD or EASY.")
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"A {T.item} is HARD if any of these apply:\n{block}\n\n"
              f"{T.item.capitalize()}: {t[:TRUNC]}\nHARD or EASY:"}] for t in texts]
    return [(o or "").strip().upper().startswith("H") for o in chat_many(msgs, model=CLF_MODEL, max_tokens=4)]


# ---------------- HARD-ITEM CLASSIFIER: rules mined only on flagged items ----------------
def classify_hard(T, texts, rules):
    block = "\n".join(f"- {r}" for r in rules)
    cats = f"{T.label.capitalize()} options:\n" + "\n".join(f"- {l}" for l in T.LBL)
    sysm = (f"Classify the {T.item}. It was flagged as one this organization files by CONVENTION, so the "
            f"rules below override the literal wording. Reply with ONLY one {T.label} name copied verbatim.")
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"{cats}\n\nFILING CONVENTIONS (these override the obvious reading):\n{block}\n\n"
              f"{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}] for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def mine_hard_rules(T, txt, gold, pred, val_txt, val_gold, max_rules=12):
    """Mine convention rules for the flagged set, PER-RULE gated on a held-out slice of flagged items."""
    import corrections
    rules = []
    def gacc(rs):
        p = classify_hard(T, val_txt, rs) if rs else semclf.zero_shot(T, val_txt)
        return float(np.mean([p[i] == val_gold[i] for i in range(len(val_gold))]))
    cur = gacc([])
    print(f"  [hard] gate-slice baseline (zero-shot) = {cur:.3f}")
    conf = Counter((gold[i], pred[i]) for i in range(len(txt)) if pred[i] != gold[i] and pred[i] != "UNPARSED")
    for (gt, pr), n in conf.most_common(max_rules * 2):
        if n < 2 or len(rules) >= max_rules: continue
        exs = [txt[i] for i in range(len(txt)) if gold[i] == gt and pred[i] == pr]
        contrast = [txt[i] for i in range(len(txt)) if gold[i] == pr and pred[i] == pr]
        r = corrections.one_exception(T, gt, pr, exs, contrast, rules, n)
        if not r: continue
        v = gacc(rules + [r])
        if v >= cur:
            rules.append(r); cur = v
            print(f"    [rule+] {gt[:16]}<-{pr[:16]} gate={v:.3f}: {r[:70]}")
        else:
            print(f"    [rule-] {gt[:16]}<-{pr[:16]} gate={v:.3f} (drop)")
    return rules, cur


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    semclf.set_trace(f"results/majflow_trace_{b}.jsonl")
    MAJ = majority_classes(T)
    T.LBL = MAJ                                   # majority-only label space for this flow
    bud = [r for r in stratified_budget(T.pool, int(b * 1.1), seed=0) if r["label"] in MAJ][:b]
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    test = [r for r in (T.test + T.test_dup) if r["label"] in MAJ]
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"MAJORITY FLOW b={b}: classes={len(MAJ)} budget={len(bud)} test={len(test)}")

    # split budget: A = gate mining, B = hard-rule mining (held out from A), V = gate slice
    na = int(0.45 * len(bud)); nb = int(0.85 * len(bud))
    A, B, V = bud[:na], bud[na:nb], bud[nb:]
    a_txt = [r["text"] for r in A]; a_gold = [r["label"] for r in A]
    b_txt = [r["text"] for r in B]; b_gold = [r["label"] for r in B]
    v_txt = [r["text"] for r in V]; v_gold = [r["label"] for r in V]

    # 1) where does zero-shot fail (on A)?
    a_zs = semclf.zero_shot(T, a_txt)
    a_hard = [a_zs[i] != a_gold[i] for i in range(len(A))]
    print(f"  zero-shot on A: acc={1-np.mean(a_hard):.4f} (hard rate {np.mean(a_hard):.3f})")

    # 2) mine the gate on A, apply to B
    g_rules = mine_gate(T, a_txt, a_hard)
    print(f"  [gate] {len(g_rules)} rules")
    b_zs = semclf.zero_shot(T, b_txt)
    b_hard_true = [b_zs[i] != b_gold[i] for i in range(len(B))]
    b_flag = apply_gate(T, b_txt, g_rules)
    tp = sum(1 for i in range(len(B)) if b_flag[i] and b_hard_true[i])
    prec = tp / max(sum(b_flag), 1); rec = tp / max(sum(b_hard_true), 1)
    print(f"  [gate] on B: flagged {sum(b_flag)}/{len(B)} precision={prec:.3f} recall={rec:.3f}")

    # 3) mine hard rules on the REAL flagged subset of B (includes false alarms)
    fi = [i for i in range(len(B)) if b_flag[i]]
    v_flag = apply_gate(T, v_txt, g_rules)
    vi = [i for i in range(len(V)) if v_flag[i]] or list(range(min(60, len(V))))
    h_rules, hgate = mine_hard_rules(T, [b_txt[i] for i in fi], [b_gold[i] for i in fi], [b_zs[i] for i in fi],
                                     [v_txt[i] for i in vi], [v_gold[i] for i in vi])
    print(f"  [hard] {len(h_rules)} rules (gate slice {hgate:.3f})")

    # 4) INFERENCE on test
    zs = semclf.zero_shot(T, txt); a0, ci0, _ = score(T, zs, gold)
    flag = apply_gate(T, txt, g_rules)
    idx = [i for i in range(len(test)) if flag[i]]
    final = list(zs)
    if idx and h_rules:
        hp = classify_hard(T, [txt[i] for i in idx], h_rules)
        for j, i in enumerate(idx): final[i] = hp[j]
    a1, ci1, _ = score(T, final, gold)
    pt = paired_test(final, zs, gold)
    hard_true = [zs[i] != gold[i] for i in range(len(test))]
    tp = sum(1 for i in idx if hard_true[i])
    fixed = sum(1 for i in idx if zs[i] != gold[i] and final[i] == gold[i])
    broke = sum(1 for i in idx if zs[i] == gold[i] and final[i] != gold[i])
    print(f"\n=== MAJORITY FLOW RESULTS (test n={len(test)}) ===")
    print(f"  zero-shot only     {a0:.4f} CI=({ci0[0]:.3f},{ci0[1]:.3f})")
    print(f"  gate+zs+hardrules  {a1:.4f} CI=({ci1[0]:.3f},{ci1[1]:.3f})  "
          f"delta={pt['delta']:+.4f} p={pt['p_mcnemar']:.1e}{' *' if pt['significant'] else ''}")
    print(f"  gate flagged {len(idx)}/{len(test)} ({len(idx)/len(test)*100:.0f}%), "
          f"precision={tp/max(len(idx),1):.3f} recall={tp/max(sum(hard_true),1):.3f}")
    print(f"  stage2 FIXED {fixed}  BROKE {broke}  net {fixed-broke:+d}")
    print(f"  calls/item = {1 + len(idx)/len(test) + 1:.2f} (gate + zs/hard)")
    json.dump({"budget": b, "classes": MAJ, "zero_shot": a0, "flow": a1, "paired": pt,
               "gate_rules": g_rules, "hard_rules": h_rules,
               "flagged": len(idx), "fixed": fixed, "broke": broke},
              open(f"results/majflow_{b}.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote results/majflow_{b}.json")


if __name__ == "__main__":
    main()
