"""CORRECTIONS (boosting-style): mine rules ONLY on the residual -- the items zero-shot gets wrong --
and append them to the plain zero-shot prompt as a compact "exceptions" block.

Rationale: zero-shot already handles most classes well; a full 18-class rulebook is mostly redundant and
can OVERRIDE correct zero-shot priors (the description-collision failure). Mining only the residual
yields focused, discriminative corrections and costs less budget.

Variants:
  corrections      : zero-shot prompt + mined exception rules (1 call/item, no router)
  cascade          : router decides hard/easy; easy -> zero-shot, hard -> corrections (cost play)

  python corrections.py [budget]
"""
import sys, json, re, copy
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, _freq_examples, _traced_call, chat_many

MAXRULES = 14


def zs_prompt(T, exceptions=None):
    """The plain zero-shot prompt, optionally + a compact corrections block."""
    base = f"{T.label.capitalize()} options:\n" + "\n".join(f"- {l}" for l in T.LBL)
    if exceptions:
        base += ("\n\nEXCEPTIONS -- this organization's filing conventions that override the obvious reading "
                 "(apply only when one matches):\n" + "\n".join(f"- {r}" for r in exceptions))
    return base


def classify(T, texts, exceptions=None):
    sysm = (f"Classify the {T.item} into its single best {T.label}. "
            + (f"Some {T.label}s follow filing conventions that contradict the literal wording; when an "
               f"EXCEPTION matches, follow it over your intuition. " if exceptions else "")
            + f"Reply with ONLY one {T.label} name copied verbatim.")
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"{zs_prompt(T, exceptions)}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}]
            for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def mine_corrections(T, mine, val, rounds=4, per_round=6):
    """Iteratively mine exception rules for the residual of the CURRENT prompt (zero-shot + rules)."""
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    rules = []
    def vacc(rs):
        vp = classify(T, v_txt, rs or None)
        return float(np.mean([vp[i] == v_gold[i] for i in range(len(val))]))
    best, best_v = [], vacc([])
    print(f"  [corr] zero-shot val={best_v:.3f}")
    tried = set()
    for rnd in range(rounds):
        m_pred = classify(T, m_txt, rules or None)
        conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine))
                       if m_pred[i] != m_gold[i] and m_pred[i] != "UNPARSED")
        bt = [(g, p) for (g, p), _ in conf.most_common() if (g, p) not in tried and conf[(g, p)] >= 2][:per_round]
        if not bt:
            print("  [corr] no recurring residual confusions left"); break
        # PER-RULE validation: test each candidate on its own, keep only those that help.
        # (Batch-validating poisons good rules when one candidate over-triggers.)
        for gt, pr in bt:
            exs = [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and m_pred[i] == pr]
            contrast = [m_txt[i] for i in range(len(mine)) if m_gold[i] == pr and m_pred[i] == pr]
            r = one_exception(T, gt, pr, exs, contrast, rules, conf[(gt, pr)])
            tried.add((gt, pr))
            if not r: continue
            v = vacc(rules + [r])
            keep = v >= best_v          # ties allowed: a neutral rule may still help elsewhere
            print(f"    [rule] {gt[:14]}<-{pr[:14]} val={v:.3f} {'KEEP' if keep else 'drop'}: {r[:80]}")
            if keep:
                rules = (rules + [r])[-MAXRULES:]; best_v = v; best = list(rules)
        print(f"  [corr] round {rnd+1}: kept {len(best)} rules, val={best_v:.3f}")
        rules = list(best)
    return best, best_v


def one_exception(T, gt, pred, exs, contrast, rules, n_err):
    """One compact exception rule for a residual confusion (diagnose+write in a single focused call)."""
    sysm = (f"You write ONE exception rule for a {T.label} classifier. A general-purpose classifier just "
            f"made a systematic mistake because this organization files by convention, not by the literal "
            f"wording. State the exception as a short trigger->{T.label} rule that a classifier can apply. "
            f"It must FIX the errors WITHOUT breaking the contrast items. Max 28 words. No preamble.")
    msg = (f"{n_err} {T.item}s that truly belong to '{gt}' were classified as '{pred}'.\n\n"
           f"MISFILED (really '{gt}'):\n{_freq_examples(exs)}\n\n"
           f"CORRECTLY '{pred}' (must not break):\n{_freq_examples(contrast, 5)}\n\n"
           + (f"Existing exceptions (do not duplicate):\n" + "\n".join(f"- {r}" for r in rules) + "\n\n" if rules else "")
           + f"Write the single exception rule:")
    o = _traced_call("exception", [{"role": "system", "content": sysm}, {"role": "user", "content": msg}],
                     MINE_MODEL, 80, gt=gt, pred=pred, n_err=n_err)
    o = (o or "").strip().strip('"').replace("\n", " ")
    return o[:200] if len(o) > 10 else ""


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    semclf.set_trace(f"results/corrections_trace_{b}.jsonl")
    bud = semclf.stratified_budget(T.pool, b); T.budget = bud
    T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    n_mine = int(0.7 * b); mine, val = bud[:n_mine], bud[n_mine:]
    test = T.test + T.test_dup                    # full fixed 1500
    ttx = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"CORRECTIONS  budget={b} mine={len(mine)} val={len(val)} test={len(test)} labels={len(T.LBL)}")
    rules, vbest = mine_corrections(T, mine, val)
    print(f"\n=== {len(rules)} mined exception rules (val={vbest:.3f}) ===")
    for r in rules: print("  -", r[:150])
    zs = classify(T, ttx, None); a0, ci0, _ = score(T, zs, gold)
    co = classify(T, ttx, rules or None); a1, ci1, _ = score(T, co, gold)
    print(f"\nzero-shot     overall={a0:.4f} CI=({ci0[0]:.3f},{ci0[1]:.3f})")
    print(f"corrections   overall={a1:.4f} CI=({ci1[0]:.3f},{ci1[1]:.3f})   delta={a1-a0:+.4f}")
    json.dump({"budget": b, "n_rules": len(rules), "rules": rules,
               "zero_shot": a0, "corrections": a1, "delta": a1 - a0},
              open(f"results/corrections_{b}.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote results/corrections_{b}.json")


if __name__ == "__main__":
    main()
