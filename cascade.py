"""CONVENTION CASCADE -- a classification flow built ENTIRELY from LLM calls (no RAG, no trained model,
no embeddings at inference). Everything the classifier "knows" is mined text.

Offline (gpt-4.1), budget split in half to avoid stage-1 overfitting its own errors:
  half A -> STAGE 1: mine an exceptions rulebook (diagnose -> edit, validated per round)
  half B -> run stage 1 (held out from its mining) to get its REALISTIC errors, then:
              ROUTER: mine HARD-flag rules that predict STAGE-1 failure (not zero-shot failure)
              apply the router to half B -> the realistic flagged set (with its false positives,
              and missing the hards the router misses)
              STAGE 2: mine focused rules on THAT flagged set -- the distribution stage 2 truly sees

Inference (gpt-4o-mini, prompts only):
  call 1: class list + exceptions -> label
  call 2 (only if the router flags it): focused re-decision with stage-2 rules

  python cascade.py [budget]
"""
import sys, json, os
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, _freq_examples, _traced_call, chat_many
import corrections
import router as R


def stage2_classify(T, texts, s1_rules, s2_rules):
    """Focused re-decision for flagged items: exceptions + the stage-2 rules mined on flagged data."""
    block = "\n".join(f"- {r}" for r in (s1_rules + s2_rules))
    sysm = (f"Classify the {T.item} into its single best {T.label}. This one was flagged as likely to be "
            f"misfiled, so weigh the organization's filing conventions carefully; they override the "
            f"literal wording. Reply with ONLY one {T.label} name copied verbatim.")
    cats = f"{T.label.capitalize()} options:\n" + "\n".join(f"- {l}" for l in T.LBL)
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"{cats}\n\nFILING CONVENTIONS:\n{block}\n\n"
              f"{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}] for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)]


def mine_stage2(T, txt, gold, pred, rounds=3, per_round=5):
    """Mine rules for the confusions that remain ON THE FLAGGED SET (realistic hard distribution)."""
    rules = []; tried = set()
    conf = Counter((gold[i], pred[i]) for i in range(len(txt)) if pred[i] != gold[i] and pred[i] != "UNPARSED")
    for (gt, pr), n in conf.most_common(rounds * per_round):
        if n < 2 or (gt, pr) in tried: continue
        exs = [txt[i] for i in range(len(txt)) if gold[i] == gt and pred[i] == pr]
        contrast = [txt[i] for i in range(len(txt)) if gold[i] == pr and pred[i] == pr]
        r = corrections.one_exception(T, gt, pr, exs, contrast, rules, n)
        if r: rules.append(r)
        tried.add((gt, pr))
        if len(rules) >= rounds * per_round: break
    return rules


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    semclf.set_trace(f"results/cascade_trace_{b}.jsonl")
    bud = semclf.stratified_budget(T.pool, b); T.budget = bud
    T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    half = len(bud) // 2
    A, B = bud[:half], bud[half:]                       # A: stage-1 mining; B: realistic downstream
    a_mine, a_val = A[:int(0.75 * len(A))], A[int(0.75 * len(A)):]
    test = T.test + T.test_dup
    t_txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"CASCADE budget={b}: A(stage1)={len(A)} B(router+stage2)={len(B)} test={len(test)}")

    # ---------- STAGE 1: exceptions rulebook, mined on A ----------
    s1_rules, v1 = corrections.mine_corrections(T, a_mine, a_val)
    print(f"[stage1] {len(s1_rules)} exception rules (val={v1:.3f})")

    # ---------- realistic errors of stage 1 on held-out B ----------
    b_txt = [r["text"] for r in B]; b_gold = [r["label"] for r in B]
    b_pred = corrections.classify(T, b_txt, s1_rules or None)
    b_hard = [b_pred[i] != b_gold[i] for i in range(len(B))]
    print(f"[stage1] accuracy on held-out B = {1-np.mean(b_hard):.4f}  (hard rate {np.mean(b_hard):.3f})")

    # ---------- ROUTER: mined rules predicting STAGE-1 failure ----------
    r_rules = R.mine_router_rules(T, b_txt, b_hard)
    print(f"[router] {len(r_rules)} HARD-flag rules")
    b_flag = R.apply_router_rules(T, b_txt, r_rules)
    q = R.router_quality("mined/B", b_flag, b_hard)

    # ---------- STAGE 2: mined on the REALISTIC flagged subset of B ----------
    fidx = [i for i in range(len(B)) if b_flag[i]]
    print(f"[stage2] mining on {len(fidx)} router-flagged items "
          f"({sum(1 for i in fidx if b_hard[i])} truly hard, {sum(1 for i in fidx if not b_hard[i])} false alarms)")
    s2_rules = mine_stage2(T, [b_txt[i] for i in fidx], [b_gold[i] for i in fidx], [b_pred[i] for i in fidx]) if fidx else []
    print(f"[stage2] {len(s2_rules)} focused rules")

    # ---------- INFERENCE on the fixed test ----------
    zs = semclf.zero_shot(T, t_txt); a_zs, ci_zs, _ = score(T, zs, gold)
    p1 = corrections.classify(T, t_txt, s1_rules or None); a1, ci1, _ = score(T, p1, gold)
    flag = R.apply_router_rules(T, t_txt, r_rules)
    t_hard_true = [p1[i] != gold[i] for i in range(len(test))]
    qt = R.router_quality("mined/test", flag, t_hard_true)
    idx = [i for i in range(len(test)) if flag[i]]
    final = list(p1)
    if idx and s2_rules:
        p2 = stage2_classify(T, [t_txt[i] for i in idx], s1_rules, s2_rules)
        for j, i in enumerate(idx): final[i] = p2[j]
    a2, ci2, _ = score(T, final, gold)
    on_hard_1 = float(np.mean([p1[i] == gold[i] for i in idx])) if idx else float("nan")
    on_hard_2 = float(np.mean([final[i] == gold[i] for i in idx])) if idx else float("nan")
    # ---- per-stage attribution: WHERE does the cascade gain/lose? ----
    fixed = sum(1 for i in idx if p1[i] != gold[i] and final[i] == gold[i])   # stage2 rescued
    broke = sum(1 for i in idx if p1[i] == gold[i] and final[i] != gold[i])   # stage2 damaged
    missed = sum(1 for i in range(len(test)) if not flag[i] and p1[i] != gold[i])  # router let errors through
    fp = sum(1 for i in idx if p1[i] == gold[i])                              # flagged though already right
    print(f"\n=== STAGE ATTRIBUTION ===")
    print(f"  stage1 errors total            {sum(1 for i in range(len(test)) if p1[i]!=gold[i])}")
    print(f"  router CAUGHT (flagged+wrong)  {len(idx)-fp}   MISSED (unflagged+wrong) {missed}")
    print(f"  router FALSE ALARMS (flagged but already right) {fp}")
    print(f"  stage2 FIXED {fixed}   BROKE {broke}   net {fixed-broke:+d}")
    print(f"  -> ceiling if stage2 were perfect on flagged: "
          f"{(sum(1 for i in range(len(test)) if p1[i]==gold[i] or flag[i]))/len(test):.4f}")
    print(f"\n=== RESULTS (test n={len(test)}) ===")
    print(f"  zero-shot            {a_zs:.4f}  CI=({ci_zs[0]:.3f},{ci_zs[1]:.3f})")
    print(f"  stage1 (exceptions)  {a1:.4f}  CI=({ci1[0]:.3f},{ci1[1]:.3f})   delta_vs_zs={a1-a_zs:+.4f}")
    print(f"  CASCADE (s1+router+s2) {a2:.4f}  CI=({ci2[0]:.3f},{ci2[1]:.3f})   delta_vs_s1={a2-a1:+.4f}")
    print(f"  flagged {len(idx)}/{len(test)} ({len(idx)/len(test)*100:.0f}%): on-flagged stage1={on_hard_1:.3f} -> cascade={on_hard_2:.3f}")
    print(f"  calls/item = 1 + {len(idx)/len(test):.2f} = {1+len(idx)/len(test):.2f}")
    json.dump({"budget": b, "zero_shot": a_zs, "stage1": a1, "cascade": a2,
               "s1_rules": s1_rules, "router_rules": r_rules, "s2_rules": s2_rules,
               "router_quality_B": q, "router_quality_test": qt,
               "flagged_frac": len(idx)/len(test), "on_flagged_s1": on_hard_1, "on_flagged_cascade": on_hard_2,
               "attribution": {"stage2_fixed": fixed, "stage2_broke": broke,
                               "router_missed_errors": missed, "router_false_alarms": fp,
                               "calls_per_item": 1 + len(idx)/len(test)}},
              open(f"results/cascade_{b}.json", "w"), indent=2, ensure_ascii=False)
    print(f"wrote results/cascade_{b}.json")


if __name__ == "__main__":
    main()
