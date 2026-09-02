"""Baton Rouge NATIVE-category (80 org-specific labels) method comparison.
zero-shot LLM (given the category list) vs RAG-few-shot. Fine-tuned curve computed separately.
Instruction mining added in br_native_instruct.py.

  python br_native_compare.py embed
  python br_native_compare.py submit
  python br_native_compare.py report
"""
import sys, os, json
import numpy as np
from collections import Counter
from openai_batch import client, submit_chat_batch, collect_chat_batch

DATA = "data/br_native.json"
EMB_F = "results/br_emb.json"
SPLIT_F = "results/br_split.json"
N_TEST = 2000
BUDGET = 2000
K = 12
EMB_MODEL = "text-embedding-3-small"
MODEL = "gpt-4o-mini"


def load():
    rows = json.load(open(DATA, encoding="utf-8"))
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    return [rows[i] for i in idx[N_TEST:]], [rows[i] for i in idx[:N_TEST]]


def labelset(pool, test):
    return sorted(set(r["label"] for r in pool + test))


def split():
    pool, test = load(); json.dump({"pool": pool, "test": test}, open(SPLIT_F, "w"), ensure_ascii=False)
    return pool, test


def embed():
    pool, test = split(); budget = pool[:BUDGET]
    texts = [r["text"] for r in budget] + [r["text"] for r in test]; out = []
    for i in range(0, len(texts), 256):
        for a in range(4):
            try:
                r = client.embeddings.create(model=EMB_MODEL, input=[t[:500] for t in texts[i:i+256]])
                out.extend([d.embedding for d in r.data]); break
            except Exception:
                if a == 3: raise
                import time; time.sleep(2*(a+1))
        print(f"  embedded {min(i+256,len(texts))}/{len(texts)}"); sys.stdout.flush()
    json.dump({"n_budget": len(budget), "emb": out}, open(EMB_F, "w")); print(f"wrote {EMB_F}")


def parse_label(o, LBL):
    o = (o or "").strip().upper()
    for l in LBL:
        if o == l.upper():
            return l
    for l in LBL:                                  # substring both directions
        if l.upper() in o or o in l.upper() and len(o) > 6:
            return l
    return "UNPARSED"


def submit():
    d = json.load(open(SPLIT_F, encoding="utf-8")); test = d["test"]; pool = d["pool"]
    LBL = labelset(pool, test); budget = pool[:BUDGET]
    catlist = "\n".join(f"- {l}" for l in LBL)
    E = np.array(json.load(open(EMB_F, encoding="utf-8"))["emb"], dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Eb = E[:len(budget)]; Et = E[len(budget):]
    zs = [(f"zs:{i}", {"messages": [
        {"role": "system", "content": "You route a municipal 311 request to this city's EXACT service "
         "category. Reply with ONLY one category name copied verbatim from the list."},
        {"role": "user", "content": f"Categories:\n{catlist}\n\nRequest: {test[i]['text'][:600]}\nCategory:"}],
        "temperature": 0, "max_tokens": 20}) for i in range(len(test))]
    submit_chat_batch(MODEL, zs, tag="br_zs")
    sims = Et @ Eb.T; rag = []
    for i in range(len(test)):
        nn = np.argsort(-sims[i])[:K]
        demos = "\n".join(f"- \"{budget[j]['text'][:120]}\" -> {budget[j]['label']}" for j in nn)
        rag.append((f"rag:{i}", {"messages": [
            {"role": "system", "content": "Route the request to this city's EXACT service category. "
             "Use the labeled examples. Reply with ONLY one category name."},
            {"role": "user", "content": f"Examples:\n{demos}\n\nRequest: {test[i]['text'][:600]}\nCategory:"}],
            "temperature": 0, "max_tokens": 20}))
    submit_chat_batch(MODEL, rag, tag="br_rag")
    print(f"submitted zero-shot + RAG at budget={BUDGET} K={K} labels={len(LBL)}")


def report():
    d = json.load(open(SPLIT_F, encoding="utf-8")); test = d["test"]; pool = d["pool"]
    LBL = labelset(pool, test); gold = [r["label"] for r in test]; out = {}
    for tag, pre in (("br_zs", "zs"), ("br_rag", "rag")):
        res = collect_chat_batch(tag=tag, verbose=False)
        if res is None:
            print(f"{tag} not ready"); return
        pred = [parse_label(res.get(f"{pre}:{i}", ""), LBL) for i in range(len(test))]
        acc = np.mean([pred[i] == gold[i] for i in range(len(test))])
        unp = np.mean([pred[i] == "UNPARSED" for i in range(len(test))])
        out[pre] = {"acc": round(float(acc), 4), "unparsed": round(float(unp), 3)}
        print(f"  {pre:5s} acc={acc:.4f} unparsed={unp:.3f}")
    json.dump(out, open("results/br_native_compare.json", "w"), indent=2)
    print("wrote results/br_native_compare.json")
    print("(fine-tuned TF-IDF+LR: 200->0.507, 2000->0.684, full->0.796)")


if __name__ == "__main__":
    {"embed": embed, "submit": submit, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "embed"]()
