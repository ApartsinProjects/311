"""Error-driven semantic instruction mining (no embeddings).
Per class we maintain a semantic description: POSITIVE descriptors (what belongs) + NEGATIVE descriptors
(what does not -> belongs elsewhere). Loop:
  1. classify with current descriptions (pure LLM);
  2. collect mislabeled batches keyed by (GT, predicted);
  3. a refiner LLM reads a batch + both classes' current descriptions, explains the mismatch, and returns
     an addition to the GT class's POSITIVE rules and to the predicted class's NEGATIVE rules;
  4. merge; repeat.
Track train error and test accuracy each round.

  python br_native_desc.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset

import os
MODEL = "gpt-4o-mini"
N_MINE = int(os.environ.get("DESC_MINE", "500"))
ROUNDS = int(os.environ.get("DESC_ROUNDS", "6"))
BATCHES_PER_ROUND = 10
SPLIT = os.environ.get("DESC_SPLIT", "results/br_split.json")
SEEDCB = os.environ.get("DESC_SEEDCB", "results/br_codebook_2000.json")
OUT = os.environ.get("DESC_OUT", "results/br_native_desc.json")


def llm(sys_u_list, max_tokens=8):
    out = [None]*len(sys_u_list)
    def one(i):
        s, u = sys_u_list[i]
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": s}, {"role": "user", "content": u}])
                return r.choices[0].message.content.strip()
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return ""
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(sys_u_list))}
        for f in as_completed(futs): out[futs[f]] = f.result()
    return out


def desc_line(c, D):
    pos = "; ".join(D[c]["pos"][:4]) or c.lower()
    neg = "; ".join(D[c]["neg"][:3])
    return f"{c}: {pos}" + (f" | NOT: {neg}" if neg else "")


def render(cats, D):
    return "\n".join(f"{i+1}. {desc_line(c, D)}" for i, c in enumerate(cats))


def pick(o, n):
    m = re.search(r"\d+", o or "")
    return (int(m.group(0))-1) if (m and 1 <= int(m.group(0)) <= n) else -1


def classify(texts, cats, D):
    book = render(cats, D)
    s = ("Route the 311 request to this city's EXACT service category using the category descriptions "
         "(what belongs, and what does NOT). Reply with ONLY the category number.")
    prompts = [(s, f"Categories:\n{book}\n\nRequest: {t[:400]}\nNumber:") for t in texts]
    outs = llm(prompts)
    return [cats[pick(o, len(cats))] if pick(o, len(cats)) >= 0 else "UNPARSED" for o in outs]


def refine_batch(gt, pred, exs, D):
    """One LLM call: explain the GT-vs-pred mismatch, return additions to gt.pos and pred.neg as JSON."""
    sample = "\n".join(f"- {t[:160]}" for t in exs[:12])
    msg = (f"A classifier put these requests in '{pred}' but they truly belong to '{gt}'.\n"
           f"Current '{gt}' description: {desc_line(gt, D)}\n"
           f"Current '{pred}' description: {desc_line(pred, D)}\n\n"
           f"Requests (true={gt}):\n{sample}\n\n"
           "Explain briefly the distinction, then output STRICT JSON: "
           '{\"gt_positive\": \"<one short clause of what DOES belong to ' + gt + '>\", '
           '\"pred_negative\": \"<one short clause: these belong to ' + gt + ', not ' + pred + '>\"}. '
           "Only the JSON.")
    r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=160,
        messages=[{"role": "system", "content": "You refine class descriptions from misclassified examples."},
                  {"role": "user", "content": msg}])
    m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)
    try:
        j = json.loads(m.group(0)); return j.get("gt_positive", "").strip(), j.get("pred_negative", "").strip()
    except Exception:
        return "", ""


def seed(cats):
    D = {c: {"pos": [], "neg": []} for c in cats}
    try:
        cb = json.load(open(SEEDCB, encoding="utf-8"))["codebook"]
        for c in cats:
            e = cb.get(c, "")
            e = re.sub(r"(?i)operational definition:\s*", "", e).split("Cues:")[0].strip()
            if e: D[c]["pos"].append(e[:140])
    except Exception:
        pass
    return D


def main():
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"]
    cats = labelset(pool, test); budget = pool[:2000]
    mine = budget[:N_MINE]; m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    te_txt = [r["text"] for r in test]; te_gold = [r["label"] for r in test]
    D = seed(cats); tried = set()
    for rnd in range(ROUNDS):
        m_pred = classify(m_txt, cats, D)
        m_err = np.mean([m_pred[i] != m_gold[i] for i in range(len(mine))])
        t_pred = classify(te_txt, cats, D)
        t_acc = np.mean([t_pred[i] == te_gold[i] for i in range(len(test))])
        print(f"round {rnd}: train_err={m_err:.3f} TEST_acc={t_acc:.4f} "
              f"(desc sizes pos={sum(len(v['pos']) for v in D.values())} neg={sum(len(v['neg']) for v in D.values())})")
        conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine))
                       if m_pred[i] != "UNPARSED" and m_pred[i] != m_gold[i])
        batches = [(gt, pr) for (gt, pr), _ in conf.most_common() if (gt, pr) not in tried and conf[(gt, pr)] >= 2][:BATCHES_PER_ROUND]
        if not batches:
            print("no more confusion batches"); break
        for gt, pr in batches:
            exs = [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and m_pred[i] == pr]
            gp, pn = refine_batch(gt, pr, exs, D)
            if gp: D[gt]["pos"].append(gp)
            if pn: D[pr]["neg"].append(pn)
            tried.add((gt, pr))
    t_pred = classify(te_txt, cats, D)
    t_acc = np.mean([t_pred[i] == te_gold[i] for i in range(len(test))])
    m_pred = classify(m_txt, cats, D); m_acc = np.mean([m_pred[i] == m_gold[i] for i in range(len(mine))])
    print(f"\nFINAL: TRAIN_acc={m_acc:.4f} TEST_acc={t_acc:.4f}  (RAG test=0.756, flat-codebook=0.657)")
    json.dump({"train_acc": float(m_acc), "test_acc": float(t_acc),
               "descriptions": D}, open(OUT, "w"), indent=2, ensure_ascii=False)
    print("wrote results/br_native_desc.json")


if __name__ == "__main__":
    main()
