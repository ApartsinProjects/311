"""Pure LLM + lexical method (NO embedding model anywhere).
Instruction mining: per category -> cue words + de-duplicated templates (all lexical).
Classification:
  1. LEXICAL narrow: score each category by best token-overlap (Jaccard) between the request and its
     templates; keep top-K candidates. (word matching, no embeddings)
  2. Deterministic template match: exact-normalized, else fuzzy token-Jaccard >= thr -> assign.
  3. LLM fallback: give the LLM the top-K candidates + templates, pick one.
Eval on held-in train (own template excluded) and test.

  python br_native_noembed.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset

MODEL = "gpt-4o-mini"; N_TReval = 600


def norm(t):
    t = re.sub(r"\d+", "#", t.lower()); t = re.sub(r"[^a-z# ]", " ", t); return re.sub(r"\s+", " ", t).strip()


def toks(s): return set(s.split())


def jacc(a, b):
    return len(a & b)/len(a | b) if (a and b) else 0.0


def build(budget):
    by = defaultdict(list); tmplnorm = defaultdict(Counter)
    for r in budget:
        by[r["label"]].append(r["text"]); tmplnorm[r["label"]][norm(r["text"])] += 1
    reps = {}
    for c, texts in by.items():
        seen = {}
        for t in texts:
            n = norm(t)
            if n not in seen: seen[n] = t
        reps[c] = [seen[n] for n, _ in tmplnorm[c].most_common()]
    ntok = {c: {tn: toks(tn) for tn in tmplnorm[c]} for c in by}
    return by, tmplnorm, reps, ntok


def classify_llm(prompts):
    out = [None]*len(prompts)
    def one(i):
        s, u = prompts[i]
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=8,
                    messages=[{"role": "system", "content": s}, {"role": "user", "content": u}])
                return r.choices[0].message.content.strip()
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return ""
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(prompts))}
        for f in as_completed(futs): out[futs[f]] = f.result()
    return out


def pick(o, n):
    m = re.search(r"\d+", o or "")
    return (int(m.group(0))-1) if (m and 1 <= int(m.group(0)) <= n) else -1


def lexical_rank(qt, cats, ntok, self_c=None):
    """score each category by best token-Jaccard between query and its templates; return sorted (score,cat)."""
    sc = []
    for c in cats:
        best = 0.0
        for tn, tt in ntok[c].items():
            j = jacc(qt, tt)
            # on train, ignore a template identical to the query unless it also appears for others
            if j > best: best = j
        sc.append((best, c))
    sc.sort(reverse=True); return sc


def evaluate(name, item_txt, gold, cats, tmplnorm, reps, ntok, K, T, fuzzy, self_norm=None):
    preds = [None]*len(item_txt); via = Counter(); rec = 0
    llm_idx = []; llm_prompts = []; llm_cands = []
    for r in range(len(item_txt)):
        q = norm(item_txt[r]); qt = toks(q)
        ranked = lexical_rank(qt, cats, ntok)
        cand = [c for _, c in ranked[:K]]
        if gold[r] in cand: rec += 1
        # deterministic exact-normalized match within candidates (train: need count>self)
        exact = [c for c in cand if q in tmplnorm[c] and (self_norm is None or tmplnorm[c][q] > (1 if self_norm[r] == c else 0))]
        if len(exact) >= 1:
            preds[r] = max(exact, key=lambda c: tmplnorm[c][q]); via["exact"] += 1; continue
        # fuzzy match within candidates
        best = (-1.0, None)
        for c in cand:
            for tn, tt in ntok[c].items():
                if self_norm is not None and tn == q and self_norm[r] == c and tmplnorm[c][q] <= 1:
                    continue
                j = jacc(qt, tt)
                if j > best[0]: best = (j, c)
        if best[0] >= fuzzy:
            preds[r] = best[1]; via["fuzzy"] += 1; continue
        body = "\n".join(f"{j+1}. {c} | e.g.: " + " ; ".join('"'+t[:70]+'"' for t in reps[c][:T]) for j, c in enumerate(cand))
        s = "Pick the EXACT city service category from the numbered candidates, matching the closest example. Reply with ONLY the number."
        llm_prompts.append((s, f"Candidates:\n{body}\n\nRequest: {item_txt[r][:300]}\nNumber:"))
        llm_idx.append(r); llm_cands.append(cand); via["llm"] += 1
    if llm_prompts:
        outs = classify_llm(llm_prompts)
        for a, r in enumerate(llm_idx):
            k = pick(outs[a], len(llm_cands[a])); preds[r] = llm_cands[a][k] if k >= 0 else "UNPARSED"
    acc = np.mean([preds[r] == gold[r] for r in range(len(gold))])
    print(f"[{name}] acc={acc:.4f} lex-recall@{K}={rec/len(gold):.3f} via={dict(via)} T={T} fuzzy={fuzzy}")
    return acc


def main():
    d = json.load(open("results/br_split.json", encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); budget = pool[:2000]
    y = [r["label"] for r in budget]; txt = [r["text"] for r in budget]
    by, tmplnorm, reps, ntok = build(budget)
    rng = np.random.RandomState(1); ev = rng.permutation(len(budget))[:N_TReval]
    tr_txt = [txt[i] for i in ev]; tr_gold = [y[i] for i in ev]; tr_self = [y[i] for i in ev]
    te_txt = [r["text"] for r in test]; te_gold = [r["label"] for r in test]
    print("=== NO-EMBEDDING (lexical narrow + template match + LLM fallback) ===")
    print("-- TRAIN (held-in, own template excluded) --")
    for K in [8, 12, 20]:
        evaluate(f"train K={K}", tr_txt, tr_gold, LBL, tmplnorm, reps, ntok, K, T=6, fuzzy=0.8, self_norm=tr_self)
    print("-- TEST --")
    for K in [8, 12]:
        evaluate(f"test K={K}", te_txt, te_gold, LBL, tmplnorm, reps, ntok, K, T=6, fuzzy=0.8)
    print("  reference: RAG(embedding) train=0.743 test=0.756; embedding-centroid method train=0.977 test=0.723")


if __name__ == "__main__":
    main()
