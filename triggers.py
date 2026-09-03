"""TRIGGER-GATED RULES: apply a mined convention only when a focused binary check confirms it fires.

Diagnosis this fixes: a rule written into the classifier prompt does not apply surgically. The measured
convention "missed pickup + recycling -> Trash" (89% supported) changed 130 predictions and netted ZERO
(59 fixed / 59 broken) because the model fired it on any recycling mention and even perturbed unrelated
items. A rule must be a CONDITIONAL REMAP, not a suggestion.

A rule = (trigger question, from_class, to_class), applied as:
  1. base prediction (zero-shot or mined rulebook)
  2. for items whose prediction == from_class, ask the trigger question (binary LLM call)
  3. remap to to_class only on YES
This cannot touch items outside from_class, so damage is bounded by construction.
Still all-LLM: no trained model, no store, no embeddings.

  python triggers.py [budget]
"""
import sys, json
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import (TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, paired_test, stratified_budget,
                    _freq_examples, _traced_call, chat_many)


def write_trigger(T, gt, pred, exs, contrast, n_err):
    """One YES/NO question that fires on the convention cases and NOT on the contrast cases."""
    sysm = (f"You write ONE yes/no question that identifies when a {T.item} currently classified as "
            f"'{pred}' should actually be filed as '{gt}' by this organization.\n"
            f"The question must be answerable from the {T.item} text alone, must be TRUE for the misfiled "
            f"examples, and FALSE for the correctly-classified contrast examples. Be specific and narrow: "
            f"a question that is too broad will misfire. Output ONLY the question, max 25 words.")
    msg = (f"Misfiled as '{pred}' but truly '{gt}' ({n_err} cases):\n{_freq_examples(exs, 8)}\n\n"
           f"Correctly '{pred}' (the question must be NO for these):\n{_freq_examples(contrast, 6)}\n\n"
           f"Write the yes/no question:")
    q = _traced_call("trigger", [{"role": "system", "content": sysm}, {"role": "user", "content": msg}],
                     MINE_MODEL, 60, gt=gt, pred=pred, n_err=n_err)
    q = (q or "").strip().strip('"').replace("\n", " ")
    return q if 10 < len(q) < 220 else ""


def write_triggers_many(T, specs):
    """Batch-write one yes/no question per (gt,pred,exs,contrast,n) spec in a SINGLE chat_many call.
    Questions do not depend on acceptance order, so they can all be generated at once."""
    msgs = []
    for (gt, pred, exs, contrast, n) in specs:
        sysm = (f"You write ONE yes/no question that identifies when a {T.item} currently classified as "
                f"'{pred}' should actually be filed as '{gt}' by this organization.\n"
                f"The question must be answerable from the {T.item} text alone, must be TRUE for the misfiled "
                f"examples, and FALSE for the correctly-classified contrast examples. Be specific and narrow: "
                f"a question that is too broad will misfire. Output ONLY the question, max 25 words.")
        msg = (f"Misfiled as '{pred}' but truly '{gt}' ({n} cases):\n{_freq_examples(exs, 8)}\n\n"
               f"Correctly '{pred}' (the question must be NO for these):\n{_freq_examples(contrast, 6)}\n\n"
               f"Write the yes/no question:")
        msgs.append([{"role": "system", "content": sysm}, {"role": "user", "content": msg}])
    outs = chat_many(msgs, model=MINE_MODEL, max_tokens=60)
    qs = []
    for o in outs:
        q = (o or "").strip().strip('"').replace("\n", " ")
        qs.append(q if 10 < len(q) < 220 else "")
    return qs


def ask_trigger(T, texts, q):
    sysm = "Answer the question about the text with ONLY YES or NO."
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"{T.item.capitalize()}: {t[:TRUNC]}\n\nQuestion: {q}\nAnswer:"}]
            for t in texts]
    return [(o or "").strip().upper().startswith("Y") for o in chat_many(msgs, model=CLF_MODEL, max_tokens=4)]


def apply_triggers(T, texts, base, rules):
    """Surgical: each rule only sees items currently predicted as its from_class."""
    preds = list(base)
    for r in rules:
        idx = [i for i in range(len(texts)) if preds[i] == r["from"]]
        if not idx: continue
        fire = ask_trigger(T, [texts[i] for i in idx], r["q"])
        for j, i in enumerate(idx):
            if fire[j]: preds[i] = r["to"]
    return preds


def mine_triggers(T, m_txt, m_gold, m_base, v_txt, v_gold, v_base,
                  max_rules=24, min_err=2, rounds=3):
    """ITERATIVE per-rule mining. After each accepted batch the base is re-classified so the NEW
    confusion structure is mined next. A rule is kept if it improves val, OR if it fixes something and
    breaks nothing (a strictly-greater test wrongly rejected zero-damage rules on ties)."""
    rules = []
    cur_pred = list(v_base)
    cur = float(np.mean([cur_pred[i] == v_gold[i] for i in range(len(v_gold))]))
    print(f"  [trig] base val={cur:.4f}")
    tried = set()
    m_cur = list(m_base)
    for rnd in range(rounds):
        conf = Counter((m_gold[i], m_cur[i]) for i in range(len(m_txt))
                       if m_cur[i] != m_gold[i] and m_cur[i] != "UNPARSED")
        cands = [(p, n) for p, n in conf.most_common() if p not in tried and n >= min_err]
        if not cands:
            print(f"  [trig] round {rnd+1}: no new confusions"); break
        added = 0
        # PRE-WRITE all candidate questions for this round in ONE batch (no per-candidate single call)
        specs = []
        for (gt, pr), n in cands:
            exs = [m_txt[i] for i in range(len(m_txt)) if m_gold[i] == gt and m_cur[i] == pr]
            contrast = [m_txt[i] for i in range(len(m_txt)) if m_gold[i] == pr and m_cur[i] == pr]
            specs.append((gt, pr, exs, contrast, n))
        qs = write_triggers_many(T, specs)
        for ((gt, pr), n), q in zip(cands, qs):
            if len(rules) >= max_rules: break
            tried.add((gt, pr))
            if not q: continue
            cand = rules + [{"q": q, "from": pr, "to": gt}]
            vp = apply_triggers(T, v_txt, v_base, cand)
            v = float(np.mean([vp[i] == v_gold[i] for i in range(len(v_gold))]))
            # marginal effect of THIS rule on top of the accepted set
            fixed = sum(1 for i in range(len(v_gold)) if cur_pred[i] != v_gold[i] and vp[i] == v_gold[i])
            broke = sum(1 for i in range(len(v_gold)) if cur_pred[i] == v_gold[i] and vp[i] != v_gold[i])
            keep = (v > cur) or (broke == 0 and fixed > 0)      # zero-damage rules are always worth keeping
            print(f"    [{'KEEP' if keep else 'drop'}] {gt[:16]}<-{pr[:16]} val={v:.4f} "
                  f"(marg fix{fixed}/brk{broke}) {q[:60]}")
            if keep:
                rules, cur, cur_pred, added = cand, max(v, cur), vp, added + 1
        print(f"  [trig] round {rnd+1}: +{added} rules ({len(rules)} total), val={cur:.4f}")
        if added == 0: break
        m_cur = apply_triggers(T, m_txt, m_base, rules)          # re-classify -> new confusions
    return rules, cur


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    semclf.set_trace(f"results/triggers_trace_{b}.jsonl")
    bud = stratified_budget(T.pool, b, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    nm = int(0.7 * b); mine, val = bud[:nm], bud[nm:]
    test = T.test + T.test_dup
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"TRIGGERS b={b}: mine={len(mine)} val={len(val)} test={len(test)}")

    # base = the mined rulebook (our best all-LLM method); fall back to zero-shot if absent
    import os
    art = f"results/mono_art_{b}.json"
    if os.path.exists(art):
        D = json.load(open(art, encoding="utf-8"))["D"]
        base_fn = lambda tx: semclf._desc_classify(T, tx, D); base_name = "mined-rulebook"
    else:
        base_fn = lambda tx: semclf.zero_shot(T, tx); base_name = "zero-shot"
    m_base = base_fn(m_txt); v_base = base_fn(v_txt); t_base = base_fn(txt)
    a0, ci0, _ = score(T, t_base, gold)
    print(f"  base = {base_name}: test={a0:.4f}")

    rules, vfinal = mine_triggers(T, m_txt, m_gold, m_base, v_txt, v_gold, v_base)
    print(f"\n  {len(rules)} trigger rules kept (val {vfinal:.4f})")
    for r in rules: print(f"   [{r['from']} -> {r['to']}] {r['q'][:110]}")

    t_final = apply_triggers(T, txt, t_base, rules)
    a1, ci1, _ = score(T, t_final, gold)
    pt = paired_test(t_final, t_base, gold)
    zs = semclf.zero_shot(T, txt); azs, _, _ = score(T, zs, gold)
    ptz = paired_test(t_final, zs, gold)
    fired = sum(1 for i in range(len(test)) if t_final[i] != t_base[i])
    fixed = sum(1 for i in range(len(test)) if t_base[i] != gold[i] and t_final[i] == gold[i])
    broke = sum(1 for i in range(len(test)) if t_base[i] == gold[i] and t_final[i] != gold[i])
    print(f"\n=== RESULTS (test n={len(test)}) ===")
    print(f"  zero-shot            {azs:.4f}")
    print(f"  base ({base_name:14s}) {a0:.4f}")
    print(f"  + trigger rules      {a1:.4f} CI=({ci1[0]:.3f},{ci1[1]:.3f})")
    print(f"     vs base: {pt['delta']:+.4f} p={pt['p_mcnemar']:.1e}{' *' if pt['significant'] else ''}")
    print(f"     vs zero-shot: {ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e}{' *' if ptz['significant'] else ''}")
    print(f"  triggers changed {fired} preds: FIXED {fixed} BROKE {broke} net {fixed-broke:+d}")
    json.dump({"budget": b, "base": base_name, "base_acc": a0, "final": a1, "zero_shot": azs,
               "rules": rules, "vs_base": pt, "vs_zs": ptz, "fixed": fixed, "broke": broke},
              open(f"results/triggers_{b}.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote results/triggers_{b}.json")


if __name__ == "__main__":
    main()
