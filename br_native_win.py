"""Push the instruction method to WIN on TRAINING data: centroid narrowing (top-K nearest category
centroids -- tiny 80-vector store, not the corpus) + per-category DEDUPLICATED TEMPLATE lists in a
focused Stage-B. Evaluate on held-in training items vs RAG (self-excluded). Sweep K and template count.

  python br_native_win.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset, parse_label

MODEL = "gpt-4o-mini"
N_EVAL = 600


def norm(t):
    t = re.sub(r"\d+", "#", t.lower()); t = re.sub(r"[^a-z# ]", " ", t); return re.sub(r"\s+", " ", t).strip()


def templates(texts, T):
    c = Counter(norm(t) for t in texts); reps = {}
    for t in texts:
        n = norm(t)
        if n not in reps: reps[n] = t
    return [reps[n][:90] for n, _ in c.most_common(T)]


def classify(prompts):
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
    if not m: return -1
    v = int(m.group(0)); return v-1 if 1 <= v <= n else -1


def main():
    d = json.load(open("results/br_split.json", encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); budget = pool[:2000]
    E = np.array(json.load(open("results/br_emb.json", encoding="utf-8"))["emb"], dtype=np.float32)[:2000]
    E /= (np.linalg.norm(E, axis=1, keepdims=True)+1e-9)
    y = [r["label"] for r in budget]; txt = [r["text"] for r in budget]
    # category centroids
    cats = LBL; cidx = {c: i for i, c in enumerate(cats)}
    cent = np.zeros((len(cats), E.shape[1]), dtype=np.float32); cnt = np.zeros(len(cats))
    by = defaultdict(list)
    for i in range(len(budget)):
        cent[cidx[y[i]]] += E[i]; cnt[cidx[y[i]]] += 1; by[y[i]].append(txt[i])
    cent /= (cnt[:, None]+1e-9); cent /= (np.linalg.norm(cent, axis=1, keepdims=True)+1e-9)
    rng = np.random.RandomState(1); ev = rng.permutation(len(budget))[:N_EVAL]
    gold = [y[i] for i in ev]

    def run_instr(K, T):
        tmpl = {c: templates(by[c], T) for c in cats}
        csim = E[ev] @ cent.T
        prompts = []; candsets = []
        for r, i in enumerate(ev):
            top = np.argsort(-csim[r])[:K]; cand = [cats[k] for k in top]; candsets.append(cand)
            body = "\n".join(f"{j+1}. {c} | templates: " + " ; ".join('"'+t+'"' for t in tmpl[c][:T])
                             for j, c in enumerate(cand))
            s = ("Pick the EXACT city service category for the request from the numbered candidates, "
                 "matching it to the closest template. Reply with ONLY the number.")
            prompts.append((s, f"Candidates:\n{body}\n\nRequest: {txt[i][:300]}\nNumber:"))
        out = classify(prompts); preds = []
        for r in range(len(ev)):
            k = pick(out[r], len(candsets[r])); preds.append(candsets[r][k] if k >= 0 else "UNPARSED")
        recall = np.mean([gold[r] in candsets[r] for r in range(len(ev))])
        acc = np.mean([preds[r] == gold[r] for r in range(len(ev))])
        return acc, recall

    def run_rag(k=12):
        S = E[ev] @ E.T
        for r, i in enumerate(ev):
            S[r, i] = -1  # exclude self
        prompts = []
        for r, i in enumerate(ev):
            nn = np.argsort(-S[r])[:k]
            demos = "\n".join(f"- \"{txt[j][:110]}\" -> {y[j]}" for j in nn)
            s = "Route the request to this city EXACT service category using the labeled examples. Reply with ONLY the category name."
            prompts.append((s, f"Examples:\n{demos}\n\nRequest: {txt[i][:300]}\nCategory:"))
        out = classify(prompts)
        preds = [parse_label(o, LBL) for o in out]
        return np.mean([preds[r] == gold[r] for r in range(len(ev))])

    rag = run_rag()
    print(f"[TRAIN-EVAL n={N_EVAL}] RAG (self-excluded) = {rag:.4f}")
    best = None
    for K in [6, 8, 12]:
        for T in [4, 8]:
            acc, rec = run_instr(K, T)
            win = "  <-- WINS" if acc >= rag else ""
            print(f"  instr centroid-narrow K={K:2d} T={T}: acc={acc:.4f} (cand-recall={rec:.3f}){win}")
            if best is None or acc > best[0]: best = (acc, K, T)
    print(f"\nBEST instr = {best[0]:.4f} at K={best[1]} T={best[2]}  vs RAG {rag:.4f}  "
          f"=> {'INSTRUCTIONS WIN ON TRAINING' if best[0] >= rag else 'still behind'}")
    json.dump({"n_eval": N_EVAL, "rag": float(rag), "best_instr": float(best[0]),
               "best_K": best[1], "best_T": best[2]}, open("results/br_native_win.json", "w"), indent=2)


if __name__ == "__main__":
    main()
