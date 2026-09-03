"""Pattern-augmented mined instructions: each class description = semantic definition + MINED
DISCRIMINATIVE PATTERNS (high-precision phrase->class associations, incl. non-semantic ones like
'sticker attached'->Trash). Store-free: a small mined pattern list per class, applied by the LLM.
Tests whether adding learnable patterns closes the gap to RAG. gpt-4o-mini, embedding-free.

  python bloom_pattern.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from br_native_compare import labelset
from oaillm import chat_many

N_TEST = 1000
SPLIT = "results/bloom_split.json"
MINPREC = 0.80; MINSUP = 8; TOPPAT = 8


def grams(t):
    toks = re.findall(r"[a-z]{2,}", t.lower()); g = []
    g += toks
    for i in range(len(toks)-1): g.append(toks[i]+" "+toks[i+1])
    for i in range(len(toks)-2): g.append(toks[i]+" "+toks[i+1]+" "+toks[i+2])
    return set(g)


def mine_patterns(budget, LBL):
    """high-precision discriminative n-grams: dominant class precision>=MINPREC, support>=MINSUP."""
    gc = defaultdict(Counter)      # ngram -> class counts
    for r in budget:
        for g in grams(r["text"]): gc[g][r["label"]] += 1
    per_class = defaultdict(list)  # class -> [(precision, support, ngram)]
    for g, cc in gc.items():
        tot = sum(cc.values())
        if tot < MINSUP: continue
        cls, cnt = cc.most_common(1)[0]; prec = cnt/tot
        if prec >= MINPREC and len(g) >= 4:
            per_class[cls].append((prec, tot, g))
    for c in per_class:
        # prefer high precision then high support; drop redundant substrings
        per_class[c].sort(key=lambda x: (-x[0], -x[1]))
    return per_class


def parse(o, LBL):
    o = (o or "").strip().strip('".').lower()
    for l in LBL:
        if o == l.lower(): return l
    best = None
    for l in LBL:
        if l.lower() in o and (best is None or len(l) > len(best)): best = l
    return best or "UNPARSED"


def seed_defs(LBL, by):
    msgs = []
    for c in LBL:
        exs = "\n".join(f"- {t[:110]}" for t in by[c][:10])
        msgs.append([{"role": "system", "content": "You write a one-sentence operational definition of a municipal service category."},
                     {"role": "user", "content": f"Category: {c}\nExamples:\n{exs}\n\nOne sentence (<=22 words):"}])
    return {c: o.strip()[:140] for c, o in zip(LBL, chat_many(msgs, max_tokens=55))}


def main():
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"][:N_TEST]
    LBL = labelset(pool, d["test"]); budget = pool[:2000]
    by = defaultdict(list)
    for r in budget: by[r["label"]].append(r["text"])
    defs = seed_defs(LBL, by)
    pats = mine_patterns(budget, LBL)
    npat = sum(min(len(pats[c]), TOPPAT) for c in LBL)
    print(f"mined {npat} high-precision patterns across {sum(1 for c in LBL if pats[c])} classes")
    for c in LBL[:4]:
        print(f"   {c}: {[g for _,_,g in pats[c][:5]]}")
    # build book: definition + mined characteristic phrases
    lines = []
    for c in LBL:
        ph = "; ".join(f'"{g}"' for _, _, g in pats[c][:TOPPAT])
        lines.append(f"- {c}: {defs[c]}" + (f" | characteristic phrases: {ph}" if ph else ""))
    book = "\n".join(lines)
    gold = [r["label"] for r in test]; ttxt = [r["text"] for r in test]
    sys = ("Classify the municipal service request into its single best service category using the "
           "definitions and the characteristic phrases (phrases that, in this city's data, strongly "
           "indicate a category even if not obvious). Reply with ONLY one category name.")
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"Categories:\n{book}\n\nRequest: {t[:450]}\nCategory:"}] for t in ttxt]
    preds = [parse(o, LBL) for o in chat_many(msgs, max_tokens=24)]
    corr = [preds[i] == gold[i] for i in range(len(test))]
    acc = float(np.mean(corr))
    c = np.array(corr); rng = np.random.RandomState(0); bs = c[rng.randint(0, len(c), (2000, len(c)))].mean(1)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    print(f"\nPATTERN-augmented acc={acc:.4f}  95%CI=({ci[0]:.3f},{ci[1]:.3f})")
    print(f"  refs: zero-shot=0.811, flat-desc~0.83, fine-tuned@1k=0.844, RAG=0.888")
    conf = Counter((gold[i], preds[i]) for i in range(len(test)) if preds[i] != gold[i])
    print("residual (true->pred):")
    for (g, p), n in conf.most_common(8): print(f"   {n:3d}  {g[:22]:22s} -> {p[:22]}")
    json.dump({"acc": acc, "ci": ci, "n_patterns": npat}, open("results/bloom_pattern.json", "w"), indent=2)


if __name__ == "__main__":
    main()
