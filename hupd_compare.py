"""HUPD many-label (539 IPC subclasses) method comparison, low-data regime.
Methods: zero-shot LLM, RAG-few-shot (retrieve nearest labeled titles as demos). Fine-tuned (TF-IDF+LR)
curve computed separately. Instruction-mining added in hupd_instruct.py.

  python hupd_compare.py embed     # cache embeddings for pool+test (OpenAI)
  python hupd_compare.py submit     # zero-shot + RAG batches at budget B
  python hupd_compare.py report
"""
import sys, os, json, re
import numpy as np
from collections import Counter
from openai_batch import client, submit_chat_batch, collect_chat_batch

DATA = "data/hupd_title.json"
EMB_F = "results/hupd_emb.json"
SPLIT_F = "results/hupd_split.json"
N_TEST = 2000
BUDGET = 2000          # labeled examples available (RAG demo source)
K = 12                 # RAG demos per query
EMB_MODEL = "text-embedding-3-small"
MODEL = "gpt-4o-mini"
CODE_RE = re.compile(r"[A-H]\d{2}[A-Z]")


def load():
    rows = json.load(open(DATA, encoding="utf-8"))
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    test = [rows[i] for i in idx[:N_TEST]]; pool = [rows[i] for i in idx[N_TEST:]]
    return pool, test


def split():
    pool, test = load()
    json.dump({"pool": pool, "test": test}, open(SPLIT_F, "w"), ensure_ascii=False)
    return pool, test


def embed():
    pool, test = split()
    budget = pool[:BUDGET]
    texts = [r["text"] for r in budget] + [r["text"] for r in test]
    out = []
    for i in range(0, len(texts), 256):
        for a in range(4):
            try:
                r = client.embeddings.create(model=EMB_MODEL, input=[t[:500] for t in texts[i:i+256]])
                out.extend([d.embedding for d in r.data]); break
            except Exception:
                if a == 3: raise
                import time; time.sleep(2*(a+1))
        print(f"  embedded {min(i+256,len(texts))}/{len(texts)}"); sys.stdout.flush()
    json.dump({"n_budget": len(budget), "emb": out}, open(EMB_F, "w"))
    print(f"wrote {EMB_F}")


def parse_code(o):
    m = CODE_RE.search((o or "").upper().replace(" ", ""))
    return m.group(0) if m else "UNPARSED"


def submit():
    d = json.load(open(SPLIT_F, encoding="utf-8")); test = d["test"]; pool = d["pool"]
    budget = pool[:BUDGET]
    E = np.array(json.load(open(EMB_F, encoding="utf-8"))["emb"], dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Eb = E[:len(budget)]; Et = E[len(budget):]
    # zero-shot
    zs = [(f"zs:{i}", {"messages": [
        {"role": "system", "content": "You are a patent examiner. Output ONLY the 4-character IPC "
         "subclass code (e.g. G06F, A61K) for the patent title."},
        {"role": "user", "content": f"Title: {test[i]['text']}\nIPC subclass code:"}],
        "temperature": 0, "max_tokens": 8}) for i in range(len(test))]
    submit_chat_batch(MODEL, zs, tag="hupd_zs")
    # RAG: nearest K budget titles as demos
    sims = Et @ Eb.T
    rag = []
    for i in range(len(test)):
        nn = np.argsort(-sims[i])[:K]
        demos = "\n".join(f"- \"{budget[j]['text']}\" -> {budget[j]['label']}" for j in nn)
        rag.append((f"rag:{i}", {"messages": [
            {"role": "system", "content": "Classify the patent title into its 4-character IPC subclass "
             "code. Use the labeled examples as a guide. Output ONLY the code."},
            {"role": "user", "content": f"Labeled examples:\n{demos}\n\nTitle: {test[i]['text']}\nIPC subclass code:"}],
            "temperature": 0, "max_tokens": 8}))
    submit_chat_batch(MODEL, rag, tag="hupd_rag")
    print(f"submitted zero-shot ({len(zs)}) + RAG ({len(rag)}) at budget={BUDGET}, K={K}")


def report():
    d = json.load(open(SPLIT_F, encoding="utf-8")); test = d["test"]
    gold = [r["label"] for r in test]
    out = {}
    for tag in ("hupd_zs", "hupd_rag"):
        res = collect_chat_batch(tag=tag, verbose=False)
        if res is None:
            print(f"{tag} not ready"); return
        pre = "zs" if tag == "hupd_zs" else "rag"
        pred = [parse_code(res.get(f"{pre}:{i}", "")) for i in range(len(test))]
        acc = np.mean([pred[i] == gold[i] for i in range(len(test))])
        unp = np.mean([pred[i] == "UNPARSED" for i in range(len(test))])
        out[pre] = {"acc": round(float(acc), 4), "unparsed": round(float(unp), 3)}
        print(f"  {pre:6s} acc={acc:.4f}  unparsed={unp:.3f}")
    json.dump(out, open("results/hupd_compare.json", "w"), indent=2)
    print("wrote results/hupd_compare.json")
    print("(fine-tuned TF-IDF+LR at budget=2000: acc 0.270; at 24768: 0.463)")


if __name__ == "__main__":
    {"embed": embed, "submit": submit, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "embed"]()
