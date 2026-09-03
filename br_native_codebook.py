"""Codebook mining = the RIGHT instruction-based replacement for RAG.
Instead of retrieving examples per query, MINE a per-category codebook once (a learned definition +
distinguishing cues for EVERY category, summarized from that category's labeled examples), and put the
whole codebook in a FIXED, cacheable prompt. No retrieval, no store at inference.

Root cause of the earlier rulebook underperformance: it only wrote disambiguation rules for the top ~8
confusion PAIRS, but errors are spread across 44+ categories, and the prompt gave only bare category
NAMES (which mean nothing to the LLM, e.g. 'MISSED HANDPILE SERVICE'). RAG's advantage is showing what
text maps to each category; the codebook encodes exactly that, compiled once.

  python br_native_codebook.py build 2000      # mine codebook from budget B (sync, ~80 LLM calls)
  python br_native_codebook.py classify 2000    # fixed-prompt classify test; compare to RAG
"""
import sys, os, json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset

MODEL = "gpt-4o-mini"
SPLIT_F = "results/br_split.json"


def cb_path(B): return f"results/br_codebook_{B}.json"


def build(B):
    d = json.load(open(SPLIT_F, encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); budget = pool[:B]
    by = defaultdict(list)
    for r in budget:
        by[r["label"]].append(r["text"])
    def entry(label):
        exs = by.get(label, [])
        if not exs:
            return label, f"(no examples in budget) requests of type: {label.lower()}"
        rng = np.random.RandomState(0); idx = rng.permutation(len(exs))[:25]
        sample = "\n".join(f"- {exs[i][:160]}" for i in idx)
        msg = (f"City 311 category: \"{label}\"\nHere are real requests filed under it:\n{sample}\n\n"
               "Write a ONE-sentence operational definition of what requests go in this category, then "
               "'Cues:' followed by 4-8 short trigger phrases/words that distinguish it from similar "
               "categories. Be specific to how THIS city files things. <=40 words total.")
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=110,
                    messages=[{"role": "system", "content": "You write concise category codebook entries."},
                              {"role": "user", "content": msg}])
                return label, r.choices[0].message.content.strip().replace("\n", " ")
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return label, label.lower()
    cb = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(entry, l) for l in LBL]):
            l, e = f.result(); cb[l] = e
    json.dump({"budget": B, "codebook": cb}, open(cb_path(B), "w"), indent=2, ensure_ascii=False)
    print(f"[build] codebook for {len(cb)} categories from budget {B} -> {cb_path(B)}")
    for l in LBL[:4]:
        print(f"   {l}: {cb[l][:110]}")


def parse(o, LBL, alias):
    o = (o or "").strip().strip('".').upper()
    if o in alias:
        return alias[o]
    # longest category name contained in output
    best = None
    for l in LBL:
        if l.upper() in o and (best is None or len(l) > len(best)):
            best = l
    if best:
        return best
    # output contained in a category name (partial), pick longest overlap
    for l in sorted(LBL, key=len):
        if len(o) > 6 and o in l.upper():
            return l
    return "UNPARSED"


def classify(B):
    d = json.load(open(SPLIT_F, encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); gold = [r["label"] for r in test]
    cb = json.load(open(cb_path(B), encoding="utf-8"))["codebook"]
    alias = {l.upper(): l for l in LBL}
    book = "\n".join(f"- {l}: {cb.get(l,'')}" for l in LBL)
    sys_msg = ("You route a municipal 311 request to this city's EXACT service category, using the "
               "codebook of category definitions. Reply with ONLY one category name copied verbatim.")
    def one(i):
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=20,
                    messages=[{"role": "system", "content": sys_msg},
                              {"role": "user", "content": f"Codebook:\n{book}\n\nRequest: {test[i]['text'][:600]}\nCategory:"}])
                return parse(r.choices[0].message.content, LBL, alias)
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return "UNPARSED"
    preds = [None]*len(test)
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(test))}
        for f in as_completed(futs):
            preds[futs[f]] = f.result()
    json.dump(preds, open(f"results/br_codebook_preds_{B}.json", "w"))
    acc = np.mean([preds[i] == gold[i] for i in range(len(test))])
    unp = np.mean([p == "UNPARSED" for p in preds])
    print(f"[classify B={B}] codebook acc={acc:.4f}  unparsed={unp:.3f}  (RAG@2000=0.7565, fine-tuned@{B} from curve)")
    json.dump({"budget": B, "codebook_acc": float(acc), "unparsed": float(unp)},
              open(f"results/br_codebook_result_{B}.json", "w"), indent=2)
    print(f"wrote results/br_codebook_result_{B}.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    {"build": build, "classify": classify}[cmd](B)
