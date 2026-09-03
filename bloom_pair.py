"""Two-stage refinement to fix sibling confusions: flat mined-description classifier proposes its TOP-2
categories, then a focused pairwise DUEL between them decides, using a mined A-vs-B difference rule.
Reuses the best descriptions from bloom_desc.json. Targets the waste/veg/water sibling errors that cost
us vs RAG. gpt-4o-mini, embedding-free.

  python bloom_pair.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from br_native_compare import labelset
from oaillm import chat_many, _call

N_TEST = 1000
SPLIT = "results/bloom_split.json"


def desc_line(c, D):
    pos = "; ".join(D[c]["pos"][:4]) or c.lower()
    neg = "; ".join(D[c]["neg"][:3])
    return f"{c}: {pos}" + (f" | NOT: {neg}" if neg else "")


def top2(texts, LBL, D):
    book = "\n".join(f"{i+1}. {desc_line(c, D)}" for i, c in enumerate(LBL))
    sys = ("Route the 311 request to this city's service category. Give the TWO most likely category "
           "NUMBERS, best first, comma-separated. Only the two numbers.")
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"Categories:\n{book}\n\nRequest: {t[:450]}\nTwo numbers:"}] for t in texts]
    outs = chat_many(msgs, max_tokens=12)
    res = []
    for o in outs:
        nums = [int(x)-1 for x in re.findall(r"\d+", o or "")][:2]
        cand = [LBL[k] for k in nums if 0 <= k < len(LBL)]
        res.append(cand or [LBL[0]])
    return res


_disc = {}
def discriminator(A, B, by):
    key = tuple(sorted([A, B]))
    if key in _disc: return _disc[key]
    exA = "\n".join(f"- {t[:110]}" for t in by[A][:8]); exB = "\n".join(f"- {t[:110]}" for t in by[B][:8])
    o = _call([{"role": "system", "content": "You write one crisp rule distinguishing two 311 categories."},
               {"role": "user", "content": f"A={A}\nB={B}\n\nExamples of A:\n{exA}\n\nExamples of B:\n{exB}\n\n"
                f"In <=30 words: how to tell an A request from a B request?"}], max_tokens=70)
    _disc[key] = o.strip(); return _disc[key]


def main():
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"][:N_TEST]
    LBL = labelset(pool, d["test"]); budget = pool[:2000]
    by = defaultdict(list)
    for r in budget: by[r["label"]].append(r["text"])
    D = json.load(open("results/bloom_desc.json", encoding="utf-8"))["descriptions"]
    t_txt = [r["text"] for r in test]; gold = [r["label"] for r in test]
    cands = top2(t_txt, LBL, D)
    stage1 = [c[0] for c in cands]
    acc1 = np.mean([stage1[i] == gold[i] for i in range(len(test))])
    print(f"stage-1 (top-1 of flat) acc={acc1:.4f}")
    # build duels for items whose top-2 differ
    duel_msgs = []; duel_idx = []
    for i in range(len(test)):
        if len(cands[i]) < 2 or cands[i][0] == cands[i][1]:
            continue
        A, B = cands[i][0], cands[i][1]; rule = discriminator(A, B, by)
        duel_msgs.append([{"role": "system", "content": "Pick the better of two 311 categories for the request."},
                          {"role": "user", "content": f"Rule:\n{rule}\n\nRequest: {t_txt[i][:300]}\n\n1. {A}\n2. {B}\nReply ONLY 1 or 2."}])
        duel_idx.append(i)
    outs = chat_many(duel_msgs, max_tokens=4)
    final = list(stage1)
    for a, i in enumerate(duel_idx):
        m = re.search(r"[12]", outs[a] or "")
        final[i] = cands[i][0] if (not m or m.group(0) == "1") else cands[i][1]
    acc2 = np.mean([final[i] == gold[i] for i in range(len(test))])
    # top-2 recall (ceiling of this method)
    rec2 = np.mean([gold[i] in cands[i] for i in range(len(test))])
    print(f"stage-2 (top-2 + duel) acc={acc2:.4f}  top-2 recall={rec2:.3f}  duels={len(duel_idx)}  #disc={len(_disc)}")
    print(f"  refs: flat-desc=0.829, zero-shot=0.811, fine-tuned@1k=0.844, RAG=0.888")
    conf = Counter((gold[i], final[i]) for i in range(len(test)) if final[i] != gold[i])
    print("residual after duel (true->pred):")
    for (g, p), c in conf.most_common(10): print(f"   {c:3d}  {g[:24]:24s} -> {p[:24]}")
    json.dump({"stage1": float(acc1), "stage2": float(acc2), "top2_recall": float(rec2)},
              open("results/bloom_pair.json", "w"), indent=2)


if __name__ == "__main__":
    main()
