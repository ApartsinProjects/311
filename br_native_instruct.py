"""Rulebook-as-RAG-replacement on Baton Rouge native categories (80 labels).
Mine a validated instruction rulebook from budget B, apply it in a FIXED (cacheable) prompt with NO
retrieval, and compare accuracy + prompt cost to RAG at the same B.

Each candidate instruction is kept only if adding it improves a held-out validation slice (re-classified
with the accumulated rulebook). Sync + threaded. Writes results/br_native_instruct.json.

  python br_native_instruct.py 2000
"""
import sys, os, json
import numpy as np
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client

MODEL = "gpt-4o-mini"
SPLIT_F = "results/br_split.json"


def labelset(pool, test):
    return sorted(set(r["label"] for r in pool + test))


def clsprompt(text, LBL, rules):
    p = "Categories:\n" + "\n".join(f"- {l}" for l in LBL) + "\n\n"
    if rules:
        p += "Filing rules (this city's conventions; apply when they match):\n" + \
             "\n".join(f"- {r}" for r in rules) + "\n\n"
    return p + f"Request: {text[:600]}\nReply with ONLY one category name.\nCategory:"


def parse(o, LBL):
    o = (o or "").strip().upper()
    for l in LBL:
        if o == l.upper():
            return l
    for l in LBL:
        if l.upper() in o or (o in l.upper() and len(o) > 6):
            return l
    return "UNPARSED"


def classify_all(texts, LBL, rules):
    out = [None] * len(texts)
    def one(i):
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=20,
                    messages=[{"role": "system", "content": "You route a municipal 311 request to this "
                               "city's EXACT service category."},
                              {"role": "user", "content": clsprompt(texts[i], LBL, rules)}])
                return parse(r.choices[0].message.content, LBL)
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return "UNPARSED"
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(texts))}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
    return out


def write_rule(X, Y, texts):
    sample = "\n".join(f"- {t[:160]}" for t in texts[:18])
    msg = (f"A classifier routed these requests to '{X}', but this city files them as '{Y}'. Write ONE "
           f"concise rule (max 25 words) telling when to choose '{Y}' over '{X}'. Name the textual cue "
           f"and the target category.\n\nExamples:\n{sample}\n\nRule:")
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=60,
            messages=[{"role": "system", "content": "You write concise, general routing rules."},
                      {"role": "user", "content": msg}])
        return r.choices[0].message.content.strip().strip('"')
    except Exception:
        return ""


def acc(pred, gold):
    return float(np.mean([pred[i] == gold[i] for i in range(len(gold))]))


def run(B):
    d = json.load(open(SPLIT_F, encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test)
    budget = pool[:B]
    rng = np.random.RandomState(0); pm = rng.permutation(len(budget)); cut = int(0.75*len(budget))
    mine = [budget[i] for i in pm[:cut]]; val = [budget[i] for i in pm[cut:]][:400]
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    print(f"[B={B}] mine={len(mine)} val={len(val)} test={len(test)} labels={len(LBL)}; base-classifying...")
    m_base = classify_all(m_txt, LBL, [])
    v_base = classify_all(v_txt, LBL, [])
    rules = []; v_acc = acc(v_base, v_gold); tried = set()
    print(f"  base val acc={v_acc:.4f}")
    for it in range(30):
        m_cur = classify_all(m_txt, LBL, rules) if rules else m_base
        conf = Counter((m_cur[i], m_gold[i]) for i in range(len(mine))
                       if m_cur[i] != "UNPARSED" and m_cur[i] != m_gold[i])
        pair = next((p for p, _ in conf.most_common() if p not in tried and conf[p] >= 4), None)
        if pair is None:
            break
        tried.add(pair); X, Y = pair
        exs = [m_txt[i] for i in range(len(mine)) if m_cur[i] == X and m_gold[i] == Y]
        instr = write_rule(X, Y, exs)
        if not instr:
            continue
        v_try = classify_all(v_txt, LBL, rules + [instr]); a_try = acc(v_try, v_gold)
        keep = a_try > v_acc + 0.002
        print(f"  it{it} {X[:18]}->{Y[:18]} n={conf[pair]} val {v_acc:.4f}->{a_try:.4f} {'KEEP' if keep else 'drop'}")
        if keep:
            rules.append(instr); v_acc = a_try
    # final: fixed-rulebook classification of test (NO retrieval)
    t_txt = [r["text"] for r in test]; t_gold = [r["label"] for r in test]
    t_base = classify_all(t_txt, LBL, [])
    t_rule = classify_all(t_txt, LBL, rules)
    a_base, a_rule = acc(t_base, t_gold), acc(t_rule, t_gold)
    print(f"\n[B={B}] TEST zero-shot={a_base:.4f}  rulebook={a_rule:.4f}  (#rules={len(rules)})")
    print(f"  RAG@2000 (for reference) = 0.7565 ; fine-tuned@{B} approx from curve")
    for r in rules:
        print("   *", r)
    out = {"budget": B, "n_rules": len(rules), "zero_shot": a_base, "rulebook": a_rule,
           "rules": rules, "labels": len(LBL)}
    json.dump(out, open(f"results/br_native_instruct_{B}.json", "w"), indent=2)
    print(f"wrote results/br_native_instruct_{B}.json")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
