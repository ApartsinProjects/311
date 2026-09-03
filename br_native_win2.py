"""Push instruction method to >=90% on TRAIN: centroid narrowing + DETERMINISTIC template matching
(apply the template rules directly: exact-normalized then fuzzy token-Jaccard) with LLM fallback only
for requests no template covers. Eval on held-in train and on test; report match-coverage.

  python br_native_win2.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset, parse_label

MODEL = "gpt-4o-mini"; N_TReval = 600


def norm(t):
    t = re.sub(r"\d+", "#", t.lower()); t = re.sub(r"[^a-z# ]", " ", t); return re.sub(r"\s+", " ", t).strip()


def toks(s): return set(s.split())


def jacc(a, b):
    if not a or not b: return 0.0
    return len(a & b)/len(a | b)


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
        reps[c] = [seen[n] for n, _ in tmplnorm[c].most_common()]  # ordered distinct templates
    return by, tmplnorm, reps


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
    m = re.search(r"\d+", o or "");
    if not m: return -1
    v = int(m.group(0)); return v-1 if 1 <= v <= n else -1


def evaluate(name, items, item_txt, gold, cent, cats, tmplnorm, reps, K, T, fuzzy=0.80, self_norm=None):
    csim = items @ cent.T
    ntok = {c: {tn: toks(tn) for tn in tmplnorm[c]} for c in cats}
    preds = [None]*len(item_txt); via = Counter(); llm_idx = []; llm_prompts = []; llm_cands = []
    for r in range(len(item_txt)):
        top = np.argsort(-csim[r])[:K]; cand = [cats[k] for k in top]
        q = norm(item_txt[r]); qt = toks(q)
        # deterministic exact-normalized match within candidates (exclude self on train)
        exact = [c for c in cand if q in tmplnorm[c] and (self_norm is None or tmplnorm[c][q] > (1 if self_norm[r] == c else 0))]
        if len(exact) == 1:
            preds[r] = exact[0]; via["exact"] += 1; continue
        if len(exact) > 1:
            preds[r] = max(exact, key=lambda c: tmplnorm[c][q]); via["exact_multi"] += 1; continue
        # fuzzy: best token-Jaccard template among candidates
        best = (-1, None)
        for c in cand:
            for tn in ntok[c]:
                j = jacc(qt, ntok[c][tn])
                if j > best[0]: best = (j, c)
        if best[0] >= fuzzy:
            preds[r] = best[1]; via["fuzzy"] += 1; continue
        # LLM fallback
        body = "\n".join(f"{j+1}. {c} | e.g.: " + " ; ".join('"'+t[:70]+'"' for t in reps[c][:T]) for j, c in enumerate(cand))
        s = "Pick the EXACT city service category from the numbered candidates, matching the closest example. Reply with ONLY the number."
        llm_prompts.append((s, f"Candidates:\n{body}\n\nRequest: {item_txt[r][:300]}\nNumber:"))
        llm_idx.append(r); llm_cands.append(cand); via["llm"] += 1
    if llm_prompts:
        outs = classify_llm(llm_prompts)
        for a, r in enumerate(llm_idx):
            k = pick(outs[a], len(llm_cands[a])); preds[r] = llm_cands[a][k] if k >= 0 else "UNPARSED"
    acc = np.mean([preds[r] == gold[r] for r in range(len(gold))])
    print(f"[{name}] acc={acc:.4f}  via={dict(via)}  K={K} T={T} fuzzy={fuzzy}")
    return acc


def main():
    d = json.load(open("results/br_split.json", encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); budget = pool[:2000]
    E = np.array(json.load(open("results/br_emb.json", encoding="utf-8"))["emb"], dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True)+1e-9); Eb = E[:2000]; Et = E[2000:]
    y = [r["label"] for r in budget]; txt = [r["text"] for r in budget]
    cats = LBL; cidx = {c: i for i, c in enumerate(cats)}
    cent = np.zeros((len(cats), E.shape[1]), dtype=np.float32); cnt = np.zeros(len(cats))
    for i in range(len(budget)): cent[cidx[y[i]]] += Eb[i]; cnt[cidx[y[i]]] += 1
    cent /= (cnt[:, None]+1e-9); cent /= (np.linalg.norm(cent, axis=1, keepdims=True)+1e-9)
    by, tmplnorm, reps = build(budget)
    rng = np.random.RandomState(1); ev = rng.permutation(len(budget))[:N_TReval]
    tr_items = Eb[ev]; tr_txt = [txt[i] for i in ev]; tr_gold = [y[i] for i in ev]
    tr_selfnorm = [y[i] for i in ev]  # each train item contributes its own template to its own category
    te_gold = [r["label"] for r in test]; te_txt = [r["text"] for r in test]
    print("=== TRAIN (held-in) — deterministic template match + LLM fallback ===")
    for K in [8, 12, 20]:
        evaluate(f"train K={K}", tr_items, tr_txt, tr_gold, cent, cats, tmplnorm, reps, K, T=6,
                 self_norm=tr_selfnorm)
    print("=== TEST — same method ===")
    for K in [8, 12]:
        evaluate(f"test K={K}", Et, te_txt, te_gold, cent, cats, tmplnorm, reps, K, T=6)
    print("  reference: RAG train=0.743 test=0.756; fine-tuned test=0.684")


if __name__ == "__main__":
    main()
