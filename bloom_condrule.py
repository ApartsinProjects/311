"""Conditional disambiguation rules (user's form): for each confusable pair, mine an IF-THEN rule from
grouped examples ("'sticker attached' alone -> Trash; with yard/brush -> Yard Waste"). Prompt = semantic
definitions + a shared 'disambiguation rules' section. Applies with reasoning; store-free.

  python bloom_condrule.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from br_native_compare import labelset
from oaillm import chat_many, _call

N_TEST = 1000; N_MINE = 800; MAXRULES = 20
SPLIT = "results/bloom_split.json"


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


def cond_rule(A, B, exA, exB):
    sA = "\n".join(f"- {t[:130]}" for t in exA[:10]); sB = "\n".join(f"- {t[:130]}" for t in exB[:10])
    default = A if len(exA) >= len(exB) else B     # base-rate default for generic requests
    msg = (f"Two often-confused municipal categories: '{A}' and '{B}'.\n\nRequests that are '{A}':\n{sA}\n\n"
           f"Requests that are '{B}':\n{sB}\n\nWrite ONE conditional rule (<=40 words) to decide between them. "
           f"Use the EXACT category names '{A}' and '{B}' (never the letters A/B). "
           f"Cover the generic case with: 'if only generic wording, choose {default}'. "
           f"Form: \"'{A}' vs '{B}': if <condition> choose '{A}'; if <condition> choose '{B}'; if generic, choose {default}.\"")
    r = _call([{"role": "system", "content": "You write one crisp conditional disambiguation rule using the exact category names given."},
               {"role": "user", "content": msg}], max_tokens=110).strip()
    # safety: if the model still emitted bare A/B placeholders, prefix the pair so it's usable
    if (" A" in r or "->A" in r or " B" in r) and A not in r:
        r = f"'{A}' vs '{B}': " + r
    return r


def classify(texts, LBL, defs, rules):
    book = "\n".join(f"- {c}: {defs[c]}" for c in LBL)
    rblock = ("\nDisambiguation rules:\n" + "\n".join(f"- {r}" for r in rules)) if rules else ""
    sys = ("Classify the municipal service request into its single best service category using the "
           "definitions and disambiguation rules. Reply with ONLY one category name.")
    msgs = [[{"role": "system", "content": sys},
             {"role": "user", "content": f"Categories:\n{book}{rblock}\n\nRequest: {t[:450]}\nCategory:"}] for t in texts]
    return [parse(o, LBL) for o in chat_many(msgs, max_tokens=24)]


def main():
    d = json.load(open(SPLIT, encoding="utf-8")); pool = d["pool"]; test = d["test"][:N_TEST]
    LBL = labelset(pool, d["test"]); budget = pool[:2000]; mine = budget[:N_MINE]
    by = defaultdict(list)
    for r in budget: by[r["label"]].append(r["text"])
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    gold = [r["label"] for r in test]; ttxt = [r["text"] for r in test]
    defs = seed_defs(LBL, by)
    # find confusable pairs via a flat-def classify on mine
    m_pred = classify(m_txt, LBL, defs, [])
    conf = Counter()
    for i in range(len(mine)):
        if m_pred[i] != "UNPARSED" and m_pred[i] != m_gold[i]:
            conf[tuple(sorted([m_gold[i], m_pred[i]]))] += 1
    pairs = [p for p, c in conf.most_common(MAXRULES) if c >= 3]
    print(f"confusable pairs mined: {len(pairs)}")
    rules = []
    for A, B in pairs:
        r = cond_rule(A, B, by[A], by[B])
        if r: rules.append(r)
    for r in rules[:6]: print("   RULE:", r[:130])
    preds = classify(ttxt, LBL, defs, rules)
    corr = [preds[i] == gold[i] for i in range(len(test))]; acc = float(np.mean(corr))
    c = np.array(corr); rng = np.random.RandomState(0); bs = c[rng.randint(0, len(c), (2000, len(c)))].mean(1)
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    print(f"\nCONDITIONAL-RULE acc={acc:.4f}  95%CI=({ci[0]:.3f},{ci[1]:.3f})  #rules={len(rules)}")
    print(f"  refs: zero-shot=0.811, flat-desc~0.83, flat-pattern=0.811, fine-tuned@1k=0.844, RAG=0.888")
    resid = Counter((gold[i], preds[i]) for i in range(len(test)) if preds[i] != gold[i])
    print("residual (true->pred):")
    for (g, p), n in resid.most_common(8): print(f"   {n:3d}  {g[:22]:22s} -> {p[:22]}")
    json.dump({"acc": acc, "ci": ci, "n_rules": len(rules), "rules": rules},
              open("results/bloom_condrule.json", "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
