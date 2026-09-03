"""How much of RAG's advantage can a FIXED (cacheable, store-free) prompt capture?
Scale the per-class artifact: k demos/class for k = 2,5,10,20 (+ optional mined rules), all in one
static prompt. Compares against zero-shot (floor) and RAG (per-query retrieval, the target).
Everything at inference is a single LLM call with a constant prefix -- no store, no trained model.

  python kscale.py [budget]
"""
import sys, json
import numpy as np
from collections import defaultdict, Counter
import semclf
from semclf import TASKS, CLF_MODEL, TRUNC, score, norm, chat_many

KS = [2, 5, 10, 20]


def select(texts, k, mode="diverse"):
    """Offline demo selection (no inference-time cost). frequency = most common templates;
    diverse = distinct templates spread over the class."""
    c = Counter(norm(x) for x in texts); reps = {}
    for x in texts:
        n = norm(x)
        if n not in reps: reps[n] = x
    order = [reps[n] for n, _ in c.most_common()]
    if mode == "frequency": return order[:k]
    # diverse: walk the distinct templates with a stride so we don't take k near-identical ones
    if len(order) <= k: return order
    step = max(1, len(order) // k)
    return [order[i] for i in range(0, len(order), step)][:k]


def run(T, texts, k, mode):
    block = []
    for l in T.LBL:
        for ex in select(T.by[l], k, mode): block.append(f"- \"{ex[:110]}\" -> {l}")
    demo = "\n".join(block)
    sysm = (f"Classify the {T.item} into its single best {T.label} using the labeled examples, which cover "
            f"every {T.label}. Some categories follow filing conventions that contradict the literal wording; "
            f"match the request against the examples. Reply with ONLY one {T.label} name.")
    msgs = [[{"role": "system", "content": sysm},
             {"role": "user", "content": f"Examples:\n{demo}\n\n{T.item.capitalize()}: {t[:TRUNC]}\n{T.label.capitalize()}:"}]
            for t in texts]
    return [T.parse(o) for o in chat_many(msgs, model=CLF_MODEL, max_tokens=24)], len(demo.split())


def main():
    b = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    T = TASKS["bloom"]
    bud = semclf.stratified_budget(T.pool, b); T.budget = bud
    T.by = defaultdict(list)
    for r in bud: T.by[r["label"]].append(r["text"])
    test = T.test + T.test_dup
    txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    print(f"K-SCALE budget={b} test={len(test)} labels={len(T.LBL)}")
    zs = semclf.zero_shot(T, txt); a, ci, _ = score(T, zs, gold)
    print(f"  zero-shot (floor)        acc={a:.4f} CI=({ci[0]:.3f},{ci[1]:.3f})")
    out = {"zero_shot": a}
    for mode in ["frequency", "diverse"]:
        for k in KS:
            preds, ntok = run(T, txt, k, mode)
            a2, ci2, unp = score(T, preds, gold)
            out[f"{mode}_k{k}"] = {"acc": a2, "ci": ci2, "prompt_words": ntok}
            print(f"  k={k:2d} ({mode:9s}) acc={a2:.4f} CI=({ci2[0]:.3f},{ci2[1]:.3f}) "
                  f"prompt~{ntok}w unp={unp:.3f}")
            sys.stdout.flush()
    rag = semclf.lexical_rag(T, txt); a3, ci3, _ = score(T, rag, gold)
    out["rag"] = {"acc": a3, "ci": ci3}
    print(f"  RAG (target, needs store) acc={a3:.4f} CI=({ci3[0]:.3f},{ci3[1]:.3f})")
    json.dump(out, open(f"results/kscale_{b}.json", "w"), indent=2)
    print(f"wrote results/kscale_{b}.json")


if __name__ == "__main__":
    main()
