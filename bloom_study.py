"""No-train classification comparison on Bloomington (18 classes), single model = gemini-2.5-flash via
OpenRouter, embedding-free (TF-IDF lexical retrieval). Establishes the frontier:
  zero-shot | lexical-RAG (per-query demos) | k-shot-per-class (static demos) .
Fine-tuned (trained) reference from the TF-IDF+LR curve. Our mined-instruction methods run separately.

  python bloom_study.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from br_native_compare import labelset
from oaillm import chat_many

N_TEST = 1000
BUDGET = 2000
KRAG = 12
SPLIT = "results/bloom_split.json"


def parse(o, LBL):
    o = (o or "").strip().strip('".').lower()
    for l in LBL:
        if o == l.lower(): return l
    best = None
    for l in LBL:
        if l.lower() in o and (best is None or len(l) > len(best)): best = l
    if best: return best
    for l in sorted(LBL, key=len):
        if len(o) > 4 and o in l.lower(): return l
    return "UNPARSED"


def acc(preds, gold): return float(np.mean([preds[i] == gold[i] for i in range(len(gold))]))


def main():
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"][:N_TEST]
    LBL = labelset(pool, d["test"]); budget = pool[:BUDGET]
    gold = [r["label"] for r in test]; ttxt = [r["text"] for r in test]
    catlist = "\n".join(f"- {l}" for l in LBL)
    print(f"Bloomington: labels={len(LBL)} budget={len(budget)} test={len(test)} model=gemini-2.5-flash")

    # ---- zero-shot ----
    SYS = ("Route the municipal 311 request to this city's EXACT service category. "
           "Reply with ONLY one category name copied verbatim.")
    zs_msgs = [[{"role": "system", "content": SYS},
                {"role": "user", "content": f"Categories:\n{catlist}\n\nRequest: {t[:500]}\nCategory:"}] for t in ttxt]
    zs = [parse(o, LBL) for o in chat_many(zs_msgs)]
    print(f"  zero-shot         acc={acc(zs, gold):.4f}")

    # ---- lexical retriever (TF-IDF, no embedding) ----
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=40000)
    Xb = vec.fit_transform([r["text"] for r in budget]); Xt = vec.transform(ttxt)
    sims = (Xt @ Xb.T).toarray()

    # ---- lexical-RAG (per-query top-KRAG demos) ----
    rag_msgs = []
    for i in range(len(test)):
        nn = np.argsort(-sims[i])[:KRAG]
        demos = "\n".join(f"- \"{budget[j]['text'][:120]}\" -> {budget[j]['label']}" for j in nn)
        rag_msgs.append([{"role": "system", "content": "Route the request to the EXACT category using the labeled examples. Reply with ONLY the category name."},
                         {"role": "user", "content": f"Examples:\n{demos}\n\nRequest: {ttxt[i][:500]}\nCategory:"}])
    rag = [parse(o, LBL) for o in chat_many(rag_msgs)]
    print(f"  lexical-RAG (k={KRAG}) acc={acc(rag, gold):.4f}")

    # ---- k-shot-per-class (static): frequency-template selection ----
    def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", re.sub(r"\d+", "#", t.lower()))).strip()
    by = defaultdict(list)
    for r in budget: by[r["label"]].append(r["text"])
    def top_templates(texts, k):
        c = Counter(norm(t) for t in texts); reps = {}
        for t in texts:
            n = norm(t)
            if n not in reps: reps[n] = t
        return [reps[n] for n, _ in c.most_common(k)]
    for KS in [2, 3]:
        demo_block = []
        for l in LBL:
            for ex in top_templates(by[l], KS):
                demo_block.append(f"- \"{ex[:100]}\" -> {l}")
        block = "\n".join(demo_block)
        ks_msgs = [[{"role": "system", "content": "Route the request to the EXACT category using the labeled examples (one block covering every category). Reply with ONLY the category name."},
                    {"role": "user", "content": f"Examples:\n{block}\n\nRequest: {t[:500]}\nCategory:"}] for t in ttxt]
        ks = [parse(o, LBL) for o in chat_many(ks_msgs)]
        print(f"  k-shot/class (k={KS}, freq) acc={acc(ks, gold):.4f}  (demos={len(demo_block)})")

    print(f"  [ref] fine-tuned TF-IDF+LR: 0.844 @1k, 0.909 @full (trained baseline)")
    json.dump({"zero_shot": acc(zs, gold), "lexical_rag": acc(rag, gold)},
              open("results/bloom_study.json", "w"), indent=2)
    print("wrote results/bloom_study.json")


if __name__ == "__main__":
    main()
