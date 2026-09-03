"""Mined semantic per-class descriptions (positive + negative), error-driven refinement, on Bloomington
with gemini-2.5-flash (OpenRouter), embedding-free. No train, no store: a one-time mined description
artifact classifies via API. Tracks train-err + test-acc each round.

  python bloom_desc.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from br_native_compare import labelset
from oaillm import chat_many, _call

N_MINE = 800; N_TEST = 1000; ROUNDS = 6; BATCHES = 10
SPLIT = "results/bloom_split.json"


def parse(o, LBL):
    o = (o or "").strip().strip('".').lower()
    for l in LBL:
        if o == l.lower(): return l
    best = None
    for l in LBL:
        if l.lower() in o and (best is None or len(l) > len(best)): best = l
    return best or "UNPARSED"


def desc_line(c, D):
    pos = "; ".join(D[c]["pos"][:4]) or c.lower()
    neg = "; ".join(D[c]["neg"][:3])
    return f"{c}: {pos}" + (f" | NOT: {neg}" if neg else "")


def classify(texts, LBL, D):
    book = "\n".join(desc_line(c, D) for c in LBL)
    sys = ("Route the 311 request to this city's EXACT service category using the descriptions "
           "(what belongs, and what does NOT). Reply with ONLY the category name.")
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"Categories:\n{book}\n\nRequest: {t[:450]}\nCategory:"}] for t in texts]
    return [parse(o, LBL) for o in chat_many(msgs, max_tokens=24)]


def seed(LBL, by):
    D = {c: {"pos": [], "neg": []} for c in LBL}
    msgs = []
    for c in LBL:
        exs = "\n".join(f"- {t[:120]}" for t in by[c][:12])
        msgs.append([{"role": "system", "content": "You write a one-sentence operational definition of a 311 category."},
                     {"role": "user", "content": f"Category: {c}\nExample requests:\n{exs}\n\nOne-sentence definition (<=25 words):"}])
    outs = chat_many(msgs, max_tokens=60)
    for c, o in zip(LBL, outs):
        if o.strip(): D[c]["pos"].append(o.strip()[:150])
    return D


def refine(gt, pred, exs, D):
    sample = "\n".join(f"- {t[:150]}" for t in exs[:12])
    msg = (f"Requests were misrouted to '{pred}' but truly belong to '{gt}'.\n"
           f"'{gt}': {desc_line(gt, D)}\n'{pred}': {desc_line(pred, D)}\n\nRequests (true={gt}):\n{sample}\n\n"
           "Output STRICT JSON: {\"gt_positive\":\"<short clause of what DOES belong to " + gt + ">\", "
           "\"pred_negative\":\"<short clause: these belong to " + gt + ", not " + pred + ">\"}. Only JSON.")
    o = _call([{"role": "system", "content": "You refine class descriptions from misclassified examples."},
               {"role": "user", "content": msg}], max_tokens=200)
    m = re.search(r"\{.*\}", o, re.S)
    try:
        j = json.loads(m.group(0)); return j.get("gt_positive", "").strip(), j.get("pred_negative", "").strip()
    except Exception:
        return "", ""


def main():
    import copy
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"][:N_TEST]
    LBL = labelset(pool, d["test"]); budget = pool[:2000]
    mine = budget[:N_MINE]; val = budget[N_MINE:N_MINE+300]      # held-out validation for the gate
    by = defaultdict(list)
    for r in budget: by[r["label"]].append(r["text"])
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    t_txt = [r["text"] for r in test]; t_gold = [r["label"] for r in test]
    D = seed(LBL, by); tried = set()
    print(f"Bloomington desc (gpt-4o-mini): labels={len(LBL)} mine={len(mine)} val={len(val)} test={len(test)}")
    def vacc(DD):
        vp = classify(v_txt, LBL, DD); return float(np.mean([vp[i] == v_gold[i] for i in range(len(val))]))
    best_D = copy.deepcopy(D); best_v = vacc(D)
    print(f"round 0 (seed): val={best_v:.4f}")
    for rnd in range(1, ROUNDS + 1):
        m_pred = classify(m_txt, LBL, D)
        conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine))
                       if m_pred[i] != "UNPARSED" and m_pred[i] != m_gold[i])
        batches = [(g, p) for (g, p), _ in conf.most_common() if (g, p) not in tried and conf[(g, p)] >= 2][:BATCHES]
        if not batches:
            print("no more confusion batches"); break
        for gt, pr in batches:
            exs = [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and m_pred[i] == pr]
            gp, pn = refine(gt, pr, exs, D)
            if gp: D[gt]["pos"].append(gp)
            if pn: D[pr]["neg"].append(pn)
            tried.add((gt, pr))
        v = vacc(D)
        keep = v >= best_v
        print(f"round {rnd}: val={v:.4f} {'KEEP(best)' if keep else 'revert-to-best'}")
        if keep:
            best_v = v; best_D = copy.deepcopy(D)
        else:
            D = copy.deepcopy(best_D)      # revert; try different batches next round
    # final eval with best-val descriptions
    t_pred = classify(t_txt, LBL, best_D)
    t_acc = float(np.mean([t_pred[i] == t_gold[i] for i in range(len(test))]))
    print(f"\nFINAL (best-val) desc TEST_acc={t_acc:.4f}  (RAG=0.888, zero-shot=0.811, fine-tuned@1k=0.844)")
    conf = Counter((t_gold[i], t_pred[i]) for i in range(len(test)) if t_pred[i] != t_gold[i])
    print("top residual confusions (true -> pred):")
    for (g, p), c in conf.most_common(12): print(f"   {c:3d}  {g[:26]:26s} -> {p[:26]}")
    json.dump({"test_acc": t_acc, "val": best_v, "descriptions": best_D,
               "residual_confusions": [[f"{g}->{p}", c] for (g, p), c in conf.most_common(20)]},
              open("results/bloom_desc.json", "w"), indent=2, ensure_ascii=False)
    print("wrote results/bloom_desc.json")


if __name__ == "__main__":
    main()
