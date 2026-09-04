"""VALIDATION-OPTIMIZED fixed k-shot: the strongest fair prompt-only demo baseline.

Fixed k-shot with demos chosen to MAXIMIZE held-out accuracy (greedy forward selection), instead of
by frequency (degenerate on unique text) or diversity alone. This is the demo analog of our rule
mining: both tune a single fixed prompt on the same val set with the same budget; only the prompt
CONTENT differs (instances vs mined rules). Inference uses the cheap CLF_MODEL, like every method.

  candidate pool = a diverse shortlist per class (keeps the search tractable)
  greedy step    = add the (class, demo) whose inclusion most improves val; stop when no add helps
  evaluation     = one batched classification of val per candidate set (cache-aware)

  python kopt.py [task] [k_per_class]
"""
import sys, json
import numpy as np
from collections import defaultdict
import semclf
from semclf import TASKS, CLF_MODEL, TRUNC, score, paired_test, stratified_budget, chat_many, norm


def _classify_with(T, texts, demos_by):
    block = "\n".join(f"- \"{ex[:100]}\" -> {l}" for l in T.LBL for ex in demos_by.get(l, []))
    sys = (f"Classify the {T.item} into its single best {T.label} using the labeled examples. "
           f"Reply with ONLY one {T.label} name.")
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"Examples:\n{block}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}]
            for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def optimize(T, val_txt, val_gold, k=2, pool_per_class=6, rounds=None):
    """Greedy forward selection of demos to maximize val accuracy, capped at k per class."""
    cand = {l: semclf._diverse_demos(T.by[l], pool_per_class) for l in T.LBL}  # diverse shortlist
    chosen = defaultdict(list)
    def acc(dby):
        p = _classify_with(T, val_txt, dby)
        return float(np.mean([p[i] == val_gold[i] for i in range(len(val_gold))]))
    cur = acc(chosen)
    print(f"  [kopt] start (0 demos) val={cur:.4f}")
    total_slots = k * len(T.LBL)
    for step in range(rounds or total_slots):
        # evaluate every legal (class, demo) addition in ONE batch
        options = []
        for l in T.LBL:
            if len(chosen[l]) >= k: continue
            for ex in cand[l]:
                if ex in chosen[l]: continue
                options.append((l, ex))
        if not options: break
        big, spans = [], []
        for l, ex in options:
            trial = {c: list(chosen[c]) for c in T.LBL}; trial[l].append(ex)
            block = "\n".join(f"- \"{e[:100]}\" -> {c}" for c in T.LBL for e in trial.get(c, []))
            s0 = len(big)
            sysm = (f"Classify the {T.item} into its single best {T.label} using the labeled examples. "
                    f"Reply with ONLY one {T.label} name.")
            for t in val_txt:
                big.append([{"role": "system", "content": sysm},
                            {"role": "user", "content": f"Examples:\n{block}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}])
            spans.append((l, ex, s0, len(big)))
        outs = chat_many(big, model=CLF_MODEL, max_tokens=24)
        best = (cur, None)
        for l, ex, s0, s1 in spans:
            preds = [T.parse(o) for o in outs[s0:s1]]
            a = np.mean([preds[i] == val_gold[i] for i in range(len(val_gold))])
            if a > best[0]: best = (a, (l, ex))
        if best[1] is None: print(f"  [kopt] step {step+1}: no add improves val -> stop"); break
        l, ex = best[1]; chosen[l].append(ex); cur = best[0]
        print(f"  [kopt] step {step+1}: +demo[{l}] val={cur:.4f}  ({sum(len(v) for v in chosen.values())} demos)")
    return dict(chosen), cur


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "bloom"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    T = TASKS[task]
    semclf.set_trace(f"results/kopt_trace_{task}.jsonl")
    bud = stratified_budget(T.pool, 2000, seed=0)
    T.budget = bud; T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    n_val = 400
    val = bud[-n_val:]                                   # held-out slice for selection
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    test = T.test + T.test_dup; t_txt = [r["text"] for r in test]; t_gold = [r["label"] for r in test]
    print(f"KOPT {task} k={k}: val={len(val)} test={len(test)} classes={len(T.LBL)}")

    demos, vfin = optimize(T, v_txt, v_gold, k=k)
    preds = _classify_with(T, t_txt, demos); a, ci, _ = score(T, preds, t_gold)
    # references
    zs = semclf.zero_shot(T, t_txt); azs, _, _ = score(T, zs, t_gold)
    div = semclf.kshot_per_class(T, t_txt, k=k, select="diverse"); ad, _, _ = score(T, div, t_gold)
    ptz = paired_test(preds, zs, t_gold)
    print(f"\n=== KOPT RESULTS ({task}) ===")
    print(f"  zero-shot            {azs:.4f}")
    print(f"  k-shot diverse       {ad:.4f}")
    print(f"  k-shot VAL-OPTIMIZED {a:.4f} CI=({ci[0]:.3f},{ci[1]:.3f})  vs zero-shot {ptz['delta']:+.4f} p={ptz['p_mcnemar']:.1e}")
    json.dump({"task": task, "k": k, "zero_shot": azs, "kshot_diverse": ad, "kshot_valopt": a,
               "demos": demos, "vs_zs": ptz}, open(f"results/kopt_{task}.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote results/kopt_{task}.json")


if __name__ == "__main__":
    main()
