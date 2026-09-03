"""Multiclass-decomposition LLM classification (novel angle): instead of one flat 80-way prompt, use
classical ONE-VS-ONE. Narrow to top-K candidates (lexical, no embedding), then run a pairwise
TOURNAMENT: each step an LLM decides 'A or B?' using a MINED semantic description of the A-vs-B
DIFFERENCE. Winner survives. K-1 LLM calls per item. Compares to flat codebook + RAG on the same subset.

  python br_native_ovo.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset

import os
MODEL = "gpt-4o-mini"
N_TEST = int(os.environ.get("OVO_TEST", "500"))
K = int(os.environ.get("OVO_K", "5"))
SPLIT = os.environ.get("OVO_SPLIT", "results/br_split.json")
OUT = os.environ.get("OVO_OUT", "results/br_native_ovo.json")


def norm(t):
    t = re.sub(r"\d+", "#", t.lower()); t = re.sub(r"[^a-z# ]", " ", t); return re.sub(r"\s+", " ", t).strip()


def toks(s): return set(s.split())


def jacc(a, b): return len(a & b)/len(a | b) if (a and b) else 0.0


def build(budget):
    by = defaultdict(list); ntok = defaultdict(dict)
    for r in budget: by[r["label"]].append(r["text"])
    for c, texts in by.items():
        for t in texts: ntok[c][norm(t)] = toks(norm(t))
    return by, ntok


_disc = {}
def discriminator(A, B, by):
    key = tuple(sorted([A, B]))
    if key in _disc: return _disc[key]
    exA = "\n".join(f"- {t[:110]}" for t in by[A][:8]); exB = "\n".join(f"- {t[:110]}" for t in by[B][:8])
    msg = (f"Two city 311 categories:\nA = {A}\nB = {B}\n\nExample requests for A:\n{exA}\n\n"
           f"Example requests for B:\n{exB}\n\nIn <=30 words, state the KEY distinguishing rule: how to tell "
           f"an A request from a B request.")
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=70,
            messages=[{"role": "system", "content": "You write one crisp rule distinguishing two categories."},
                      {"role": "user", "content": msg}])
        d = r.choices[0].message.content.strip()
    except Exception:
        d = ""
    _disc[key] = d; return d


def duel(text, A, B, by):
    d = discriminator(A, B, by)
    msg = (f"Distinguishing rule:\n{d}\n\nRequest: {text[:300]}\n\nWhich category fits better?\n"
           f"1. {A}\n2. {B}\nReply with ONLY 1 or 2.")
    for a in range(3):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=4,
                messages=[{"role": "system", "content": "You pick the better of two categories for the request."},
                          {"role": "user", "content": msg}])
            m = re.search(r"[12]", r.choices[0].message.content or "")
            return A if (m and m.group(0) == "1") else B
        except Exception:
            import time; time.sleep(1.2*(a+1))
    return A


def lexical_topk(qt, cats, ntok, k):
    sc = []
    for c in cats:
        best = max((jacc(qt, tt) for tt in ntok[c].values()), default=0.0)
        sc.append((best, c))
    sc.sort(reverse=True); return [c for _, c in sc[:k]]


def classify_item(text, cats, ntok, by):
    qt = toks(norm(text)); cand = lexical_topk(qt, cats, ntok, K)
    if len(cand) == 1: return cand[0], cand
    win = cand[0]
    for c in cand[1:]:
        win = duel(text, win, c, by)
    return win, cand


def main():
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"]
    cats = labelset(pool, test); budget = pool[:2000]
    by, ntok = build(budget)
    rng = np.random.RandomState(3); idx = rng.permutation(len(test))[:N_TEST]
    sub = [test[i] for i in idx]; gold = [r["label"] for r in sub]
    preds = [None]*len(sub); cands = [None]*len(sub)
    def work(i):
        p, c = classify_item(sub[i]["text"], cats, ntok, by); return i, p, c
    with ThreadPoolExecutor(max_workers=10) as ex:
        for f in as_completed([ex.submit(work, i) for i in range(len(sub))]):
            i, p, c = f.result(); preds[i] = p; cands[i] = c
    acc = np.mean([preds[i] == gold[i] for i in range(len(sub))])
    rec = np.mean([gold[i] in cands[i] for i in range(len(sub))])
    print(f"[OvO tournament, n={N_TEST}, K={K}] acc={acc:.4f}  lex-recall@{K}={rec:.3f}  #discriminators={len(_disc)}")
    print(f"  reference on full test: RAG=0.756, flat-codebook=0.657, fine-tuned=0.684, 1-NN=0.680")
    json.dump({"acc": float(acc), "recall": float(rec), "K": K, "n": N_TEST, "n_disc": len(_disc)},
              open(OUT, "w"), indent=2)


if __name__ == "__main__":
    main()
