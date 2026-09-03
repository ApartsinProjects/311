"""Iteratively EXTEND the instruction codebook to capture the patterns behind residual errors
(including near-duplicate template clusters), until training error is negligible. Tracks mine-error
and test-accuracy each round to see whether extension closes the RAG gap or overfits.

Start = comprehensive cue-codebook. Each round: find top confusion clusters (pred X, gold Y) on the
mine set, have the LLM write a discriminating rule for each, append to a growing 'Disambiguation rules'
section (fixed, cacheable), re-classify. Stop when few clusters remain or max rounds.

  python br_native_extend.py
"""
import json, re
import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai_batch import client
from br_native_compare import labelset, parse_label

MODEL = "gpt-4o-mini"
N_MINE = 800
ROUNDS = 10
CL_PER_ROUND = 6


def grams(t):
    toks = re.findall(r"[a-z]{2,}", t.lower()); g = set(toks)
    for i in range(len(toks)-1): g.add(toks[i]+" "+toks[i+1])
    return g


def cue_codebook(budget, LBL):
    docf = Counter(); catf = defaultdict(Counter); catN = Counter()
    for r in budget:
        gs = grams(r["text"]); catN[r["label"]] += 1
        for g in gs: docf[g] += 1; catf[r["label"]][g] += 1
    N = len(budget); lines = {}
    for l in LBL:
        n = catN[l] or 1; scored = []
        for g, c in catf[l].items():
            if c < 3: continue
            insh = c/n; ov = docf[g]/N
            if insh >= 0.15 and insh/(ov+1e-9) >= 2: scored.append((insh/(ov+1e-9)*insh, g))
        scored.sort(reverse=True); lines[l] = ", ".join(g for _, g in scored[:10])
    return lines


def build_prompt(cues, LBL, rules):
    book = "\n".join(f"- {l}: cues= {cues.get(l,'')}" for l in LBL)
    p = f"Categories and cue phrases:\n{book}\n"
    if rules:
        p += "\nDisambiguation rules:\n" + "\n".join(f"- {r}" for r in rules) + "\n"
    return p


def classify(texts, prompt_body, LBL):
    sysm = ("Route the municipal 311 request to this city EXACT service category using the cue codebook "
            "and disambiguation rules. Reply with ONLY one category name verbatim.")
    out = [None]*len(texts)
    def one(i):
        for a in range(3):
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=20,
                    messages=[{"role": "system", "content": sysm},
                              {"role": "user", "content": f"{prompt_body}\nRequest: {texts[i][:600]}\nCategory:"}])
                return parse_label(r.choices[0].message.content, LBL)
            except Exception:
                import time; time.sleep(1.5*(a+1))
        return "UNPARSED"
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(one, i): i for i in range(len(texts))}
        for f in as_completed(futs): out[futs[f]] = f.result()
    return out


def make_rule(X, Y, exsY):
    sample = "\n".join(f"- {t[:160]}" for t in exsY[:12])
    msg = (f"These requests were misrouted to '{X}' but actually belong to '{Y}'. State the pattern that "
           f"identifies '{Y}' vs '{X}' in <=25 words as: \"If <pattern>, it's {Y} not {X}.\"\n\n"
           f"Examples of {Y}:\n{sample}")
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=60,
            messages=[{"role": "system", "content": "You write one concise disambiguation rule."},
                      {"role": "user", "content": msg}])
        return r.choices[0].message.content.strip().strip('"')
    except Exception:
        return ""


def main():
    d = json.load(open("results/br_split.json", encoding="utf-8")); pool = d["pool"]; test = d["test"]
    LBL = labelset(pool, test); budget = pool[:2000]
    mine = budget[:N_MINE]; m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    t_txt = [r["text"] for r in test]; t_gold = [r["label"] for r in test]
    cues = cue_codebook(budget, LBL)
    rules = []; tried = set()
    E = np.array(json.load(open("results/br_emb.json", encoding="utf-8"))["emb"], dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True)+1e-9); nnsim = (E[2000:]@E[:2000].T).max(1)
    def stratified(preds):
        r = {}
        for name, f in [("nov", lambda s: s < 0.7), ("mid", lambda s: 0.7 <= s < 0.9), ("dup", lambda s: s >= 0.9)]:
            idx = [i for i in range(len(test)) if f(nnsim[i])]
            r[name] = round(float(np.mean([preds[i] == t_gold[i] for i in idx])), 3)
        return r
    for rnd in range(ROUNDS):
        body = build_prompt(cues, LBL, rules)
        m_pred = classify(m_txt, body, LBL)
        m_err = np.mean([m_pred[i] != m_gold[i] for i in range(len(mine))])
        conf = Counter((m_pred[i], m_gold[i]) for i in range(len(mine))
                       if m_pred[i] != "UNPARSED" and m_pred[i] != m_gold[i])
        pairs = [p for p, _ in conf.most_common() if p not in tried and conf[p] >= 3][:CL_PER_ROUND]
        # measure test each round
        t_pred = classify(t_txt, body, LBL)
        t_acc = np.mean([t_pred[i] == t_gold[i] for i in range(len(test))])
        print(f"round {rnd}: #rules={len(rules)} mine_err={m_err:.3f} TEST_acc={t_acc:.4f} strat={stratified(t_pred)}")
        if not pairs:
            print("no more recurring clusters"); break
        new = []
        for (X, Y) in pairs:
            exsY = [m_txt[i] for i in range(len(mine)) if m_gold[i] == Y and m_pred[i] == X]
            r = make_rule(X, Y, exsY)
            if r: new.append(r)
            tried.add((X, Y))
        rules += new
    body = build_prompt(cues, LBL, rules)
    t_pred = classify(t_txt, body, LBL)
    t_acc = np.mean([t_pred[i] == t_gold[i] for i in range(len(test))])
    print(f"\nFINAL: #rules={len(rules)} TEST_acc={t_acc:.4f} strat={stratified(t_pred)}  (RAG=0.756)")
    json.dump({"n_rules": len(rules), "test_acc": float(t_acc), "rules": rules},
              open("results/br_native_extend.json", "w"), indent=2)
    print("wrote results/br_native_extend.json")


if __name__ == "__main__":
    main()
