"""Axis-2 (FLOW) comparison: the SAME mined instruction artifact classified via 5 classical multiclass
schemes -- flat, one-vs-all, one-vs-one, hierarchical, ECOC -- against the 3 baselines
(zero-shot = lower bound, RAG = target to beat, k-shot/class).

Everything runs on the SAME items with the SAME 2k labeling budget, so the only variable is the flow.
The mined descriptions are mined ONCE (convention-aware, gpt-4.1) and cached to disk.
classify=gpt-4o-mini. No embeddings anywhere.

  python flows.py [n_test]
"""
import sys, json, re, os
import numpy as np
from collections import defaultdict, Counter
from oaillm import chat_many, _call
import semclf
from semclf import Task, TASKS, CLF_MODEL, MINE_MODEL, TRUNC, score, _dline, _seed, _refine, _diagnose, zero_shot, lexical_rag, kshot_per_class

DCACHE = "results/flows_desc.json"


# ---------------- shared: mine the artifact once ----------------
def mine_artifact(T, mine, val, rounds=5, batches=8):
    if os.path.exists(DCACHE):
        return json.load(open(DCACHE, encoding="utf-8"))
    import copy
    m_txt = [r["text"] for r in mine]; m_gold = [r["label"] for r in mine]
    v_txt = [r["text"] for r in val]; v_gold = [r["label"] for r in val]
    D = _seed(T); tried = set()
    def vacc(DD):
        vp = semclf._desc_classify(T, v_txt, DD)
        return float(np.mean([vp[i] == v_gold[i] for i in range(len(val))]))
    best_D, best_v = copy.deepcopy(D), vacc(D)
    print(f"  [mine] seed val={best_v:.3f}")
    for rnd in range(rounds):
        m_pred = semclf._desc_classify(T, m_txt, D)
        conf = Counter((m_gold[i], m_pred[i]) for i in range(len(mine))
                       if m_pred[i] != m_gold[i] and m_pred[i] != "UNPARSED")
        bt = [(g, p) for (g, p), _ in conf.most_common() if (g, p) not in tried and conf[(g, p)] >= 2][:batches]
        if not bt: break
        base = {c: len(T.by[c]) for c in T.LBL}
        for gt, pr in bt:
            exs = [m_txt[i] for i in range(len(mine)) if m_gold[i] == gt and m_pred[i] == pr]
            contrast = [m_txt[i] for i in range(len(mine)) if m_gold[i] == pr and m_pred[i] == pr]
            diag = _diagnose(T, gt, pr, exs, contrast, D, n_err=conf[(gt, pr)])
            upd = _refine(T, gt, pr, exs, D, contrast=contrast, base_rates=base,
                          n_err=conf[(gt, pr)], diag=diag)
            semclf.apply_update(D, upd)
            tried.add((gt, pr))
        v = vacc(D)
        print(f"  [mine] round {rnd+1} val={v:.3f} {'keep' if v >= best_v else 'revert'}")
        if v >= best_v: best_v, best_D = v, copy.deepcopy(D)
        else: D = copy.deepcopy(best_D); tried -= set(bt)
    json.dump(best_D, open(DCACHE, "w"), indent=2, ensure_ascii=False)
    return best_D


def _num(o, n):
    m = re.search(r"\d+", o or "")
    return (int(m.group(0)) - 1) if (m and 1 <= int(m.group(0)) <= n) else -1


CONV = ("These are the organization's FILING CONVENTIONS, which sometimes contradict common sense: "
        "when a description states a convention, FOLLOW IT over your own intuition about the wording.")


# ---------------- FLOW 1: flat ----------------
def flow_flat(T, texts, D):
    return semclf._desc_classify(T, texts, D)


# ---------------- FLOW 2: one-vs-all ----------------
def flow_ova(T, texts, D):
    """Per class: binary 'does it belong here?'. Ties broken by a flat call over the YES set."""
    msgs = []; idx = []
    for i, t in enumerate(texts):
        for c in T.LBL:
            msgs.append([{"role": "system", "content": f"Answer YES or NO only. {CONV}"},
                         {"role": "user", "content": f"{T.label.capitalize()} '{c}': {_dline(T, c, D)}\n\n"
                          f"{T.item.capitalize()}: {t[:TRUNC]}\n\nIs this {T.item} filed under '{c}'? YES or NO:"}])
            idx.append((i, c))
    outs = chat_many(msgs, model=CLF_MODEL, max_tokens=4)
    yes = defaultdict(list)
    for (i, c), o in zip(idx, outs):
        if (o or "").strip().upper().startswith("Y"): yes[i].append(c)
    preds = [None] * len(texts); tie_i = []; tie_msgs = []
    for i in range(len(texts)):
        y = yes.get(i, [])
        if len(y) == 1: preds[i] = y[0]
        else:
            cand = y if len(y) > 1 else T.LBL
            book = "\n".join(f"{j+1}. {_dline(T, c, D)}" for j, c in enumerate(cand))
            tie_msgs.append([{"role": "system", "content": f"Pick the single best {T.label}. Reply ONLY the number. {CONV}"},
                             {"role": "user", "content": f"Candidates:\n{book}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\nNumber:"}])
            tie_i.append((i, cand))
    if tie_msgs:
        for (i, cand), o in zip(tie_i, chat_many(tie_msgs, model=CLF_MODEL, max_tokens=8)):
            k = _num(o, len(cand)); preds[i] = cand[k] if k >= 0 else cand[0]
    return preds


# ---------------- FLOW 3: one-vs-one (top-3 shortlist, round-robin, canonical orientation) ----------------
_disc = {}
def _pair_rule(T, A, B, by):
    A, B = sorted([A, B])                       # CANONICAL orientation (fixes the earlier bug)
    if (A, B) in _disc: return (A, B), _disc[(A, B)]
    exA = "\n".join(f"- {t[:100]}" for t in by[A][:6]); exB = "\n".join(f"- {t[:100]}" for t in by[B][:6])
    o = _call([{"role": "system", "content": f"You write one crisp rule distinguishing two {T.label}s. "
                f"These are filing conventions and may contradict common sense; describe actual behavior. "
                f"Use the EXACT names, never letters."},
               {"role": "user", "content": f"'{A}': {_dline(T, A, {A: {'pos': [], 'neg': []}}) if False else ''}\n"
                f"Examples of '{A}':\n{exA}\n\nExamples of '{B}':\n{exB}\n\n"
                f"In <=35 words: how to tell '{A}' from '{B}'? Name both exactly."}],
              model=MINE_MODEL, max_tokens=90)
    _disc[(A, B)] = o.strip(); return (A, B), _disc[(A, B)]


def flow_ovo(T, texts, D, by, K=3):
    # shortlist via one flat top-K call
    book = "\n".join(f"{i+1}. {_dline(T, c, D)}" for i, c in enumerate(T.LBL))
    sl = [[{"role": "system", "content": f"List the {K} most likely {T.label} NUMBERS, best first, comma-separated. {CONV}"},
           {"role": "user", "content": f"{T.label.capitalize()}s:\n{book}\n\n{T.item.capitalize()}: {t[:TRUNC]}\nNumbers:"}] for t in texts]
    cands = []
    for o in chat_many(sl, model=CLF_MODEL, max_tokens=14):
        nums = [int(x) - 1 for x in re.findall(r"\d+", o or "")][:K]
        c = [T.LBL[k] for k in nums if 0 <= k < len(T.LBL)]
        cands.append(list(dict.fromkeys(c)) or [T.LBL[0]])
    # mine rules for all needed pairs (canonical), then round-robin duels
    need = set()
    for c in cands:
        for a in range(len(c)):
            for b in range(a + 1, len(c)): need.add(tuple(sorted([c[a], c[b]])))
    for A, B in need: _pair_rule(T, A, B, by)
    msgs = []; meta = []
    for i, c in enumerate(cands):
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                X, Y = sorted([c[a], c[b]]); rule = _disc.get((X, Y), "")
                msgs.append([{"role": "system", "content": f"Pick the better of two {T.label}s. Reply ONLY 1 or 2. {CONV}"},
                             {"role": "user", "content": f"Rule:\n{rule}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\n\n1. {X}\n2. {Y}\nAnswer:"}])
                meta.append((i, X, Y))
    votes = defaultdict(Counter)
    for (i, X, Y), o in zip(meta, chat_many(msgs, model=CLF_MODEL, max_tokens=4)):
        m = re.search(r"[12]", o or "")
        votes[i][X if (not m or m.group(0) == "1") else Y] += 1   # parse-fail -> first (canonical), never challenger
    preds = []
    for i, c in enumerate(cands):
        v = votes.get(i)
        preds.append(v.most_common(1)[0][0] if v else c[0])
    return preds


# ---------------- FLOW 4: hierarchical (mined groups, top-2 then within) ----------------
def mine_groups(T, D):
    lines = "\n".join(f"- {c}" for c in T.LBL)
    o = _call([{"role": "system", "content": f"You group {T.label}s into coherent higher-level groups for a "
                f"two-stage classifier. Groups should be BALANCED (3-5 members each) and easy to tell apart."},
               {"role": "user", "content": f"{T.label.capitalize()}s:\n{lines}\n\nOutput JSON: "
                "{\"<group name>\": [\"<exact name>\", ...], ...}. Every name exactly once. Only JSON."}],
              model=MINE_MODEL, max_tokens=800)
    m = re.search(r"\{.*\}", o, re.S)
    try:
        g = json.loads(m.group(0))
        seen = set(); clean = {}
        for k, v in g.items():
            mem = [x for x in v if x in T.LBL and x not in seen]
            for x in mem: seen.add(x)
            if mem: clean[k] = mem
        miss = [c for c in T.LBL if c not in seen]
        if miss: clean.setdefault("Other", []).extend(miss)
        return clean
    except Exception:
        return {"All": list(T.LBL)}


def flow_hier(T, texts, D, groups):
    gn = list(groups)
    gbook = "\n".join(f"{i+1}. {g}: " + "; ".join(groups[g][:6]) for i, g in enumerate(gn))
    m1 = [[{"role": "system", "content": f"Choose the TWO most likely groups. Reply with two numbers, comma-separated. {CONV}"},
           {"role": "user", "content": f"Groups:\n{gbook}\n\n{T.item.capitalize()}: {t[:TRUNC]}\nNumbers:"}] for t in texts]
    cands = []
    for o in chat_many(m1, model=CLF_MODEL, max_tokens=12):
        nums = [int(x) - 1 for x in re.findall(r"\d+", o or "")][:2]
        gs = [gn[k] for k in nums if 0 <= k < len(gn)] or [gn[0]]
        cs = []
        for g in gs: cs += groups[g]
        cands.append(list(dict.fromkeys(cs)) or list(T.LBL))
    m2 = []
    for i, c in enumerate(cands):
        book = "\n".join(f"{j+1}. {_dline(T, x, D)}" for j, x in enumerate(c))
        m2.append([{"role": "system", "content": f"Pick the single best {T.label}. Reply ONLY the number. {CONV}"},
                   {"role": "user", "content": f"Candidates:\n{book}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\nNumber:"}])
    preds = []
    for (o, c) in zip(chat_many(m2, model=CLF_MODEL, max_tokens=8), cands):
        k = _num(o, len(c)); preds.append(c[k] if k >= 0 else c[0])
    return preds


# ---------------- FLOW 5: ECOC ----------------
def make_code(T, M=12, seed=0):
    rng = np.random.RandomState(seed); n = len(T.LBL)
    while True:
        C = rng.randint(0, 2, (n, M))
        if len(set(map(tuple, C))) == n and all(0 < C[:, j].sum() < n for j in range(M)):
            return C


def flow_ecoc(T, texts, D, C):
    M = C.shape[1]; bits = []
    for j in range(M):
        side0 = [T.LBL[i] for i in range(len(T.LBL)) if C[i, j] == 0]
        side1 = [T.LBL[i] for i in range(len(T.LBL)) if C[i, j] == 1]
        d0 = "; ".join(_dline(T, c, D)[:90] for c in side0[:6]); d1 = "; ".join(_dline(T, c, D)[:90] for c in side1[:6])
        msgs = [[{"role": "system", "content": f"Decide which GROUP the {T.item} belongs to. Reply ONLY 1 or 2. {CONV}"},
                 {"role": "user", "content": f"GROUP 1 ({', '.join(side0)}): {d0}\n\nGROUP 2 ({', '.join(side1)}): {d1}\n\n"
                  f"{T.item.capitalize()}: {t[:TRUNC]}\nGroup:"}] for t in texts]
        outs = chat_many(msgs, model=CLF_MODEL, max_tokens=4)
        b = []
        for o in outs:
            m = re.search(r"[12]", o or "")
            b.append(0 if (m and m.group(0) == "1") else 1)
        bits.append(b)
        print(f"    [ecoc] bit {j+1}/{M} done"); sys.stdout.flush()
    B = np.array(bits).T                      # (n_items, M)
    preds = []
    for i in range(len(texts)):
        d = (C != B[i]).sum(1)                # Hamming distance to each codeword
        preds.append(T.LBL[int(np.argmin(d))])
    return preds


def flow_ovo_from_cands(T, texts, cands, by):
    """Round-2 of OvO: candidate shortlists already computed in a batch; run the duels."""
    need = set()
    for c in cands:
        for a in range(len(c)):
            for b in range(a + 1, len(c)): need.add(tuple(sorted([c[a], c[b]])))
    for A, B in need: _pair_rule(T, A, B, by)
    msgs = []; meta = []
    for i, c in enumerate(cands):
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                X, Y = sorted([c[a], c[b]]); rule = _disc.get((X, Y), "")
                msgs.append([{"role": "system", "content": f"Pick the better of two {T.label}s. Reply ONLY 1 or 2. {CONV}"},
                             {"role": "user", "content": f"Rule:\n{rule}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\n\n1. {X}\n2. {Y}\nAnswer:"}])
                meta.append((i, X, Y))
    votes = defaultdict(Counter)
    for (i, X, Y), o in zip(meta, chat_many(msgs, model=CLF_MODEL, max_tokens=4)):
        m = re.search(r"[12]", o or "")
        votes[i][X if (not m or m.group(0) == "1") else Y] += 1
    return [votes[i].most_common(1)[0][0] if votes.get(i) else cands[i][0] for i in range(len(cands))]


def flow_stage2_from_cands(T, texts, cands, D):
    """Round-2 of hierarchical: groups already chosen in a batch; pick within the candidate set."""
    msgs = []
    for i, c in enumerate(cands):
        book = "\n".join(f"{j+1}. {_dline(T, x, D)}" for j, x in enumerate(c))
        msgs.append([{"role": "system", "content": f"Pick the single best {T.label}. Reply ONLY the number. {CONV}"},
                     {"role": "user", "content": f"Candidates:\n{book}\n\n{T.item.capitalize()}: {texts[i][:TRUNC]}\nNumber:"}])
    out = []
    for o, c in zip(chat_many(msgs, model=CLF_MODEL, max_tokens=8), cands):
        k = _num(o, len(c)); out.append(c[k] if k >= 0 else c[0])
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    T = TASKS["bloom"]
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(T.test))[:n]
    test = [T.test[i] for i in idx]
    texts = [r["text"] for r in test]; gold = [r["label"] for r in test]
    mine = T.budget[:800]; val = T.budget[800:1100]
    by = defaultdict(list)
    for r in T.budget: by[r["label"]].append(r["text"])
    print(f"FLOW COMPARISON  labels={len(T.LBL)}  n_test={len(test)} (novel slice)  budget={len(T.budget)}")
    print(f"models: mine={MINE_MODEL} classify={CLF_MODEL}\n[mining shared artifact]")
    D = mine_artifact(T, mine, val)
    groups = mine_groups(T, D); C = make_code(T, M=12)
    print(f"  groups={ {k: len(v) for k, v in groups.items()} }  ecoc bits={C.shape[1]}\n")
    res = {}
    runs = [
        ("zero_shot  (lower bound)", lambda: zero_shot(T, texts)),
        ("RAG        (to beat)", lambda: lexical_rag(T, texts)),
        ("kshot/class", lambda: kshot_per_class(T, texts)),
        ("mined>flat", lambda: flow_flat(T, texts, D)),
        ("mined>one-vs-all", lambda: flow_ova(T, texts, D)),
        ("mined>one-vs-one", lambda: flow_ovo(T, texts, D, by)),
        ("mined>hierarchical", lambda: flow_hier(T, texts, D, groups)),
        ("mined>ECOC", lambda: flow_ecoc(T, texts, D, C)),
    ]
    for name, fn in runs:
        preds = fn()
        acc, ci, unp = score(T, preds, gold)
        res[name] = {"acc": acc, "ci": ci, "unparsed": unp}
        print(f"  {name:26s} acc={acc:.4f}  95%CI=({ci[0]:.3f},{ci[1]:.3f})  unparsed={unp:.3f}")
        sys.stdout.flush()
    json.dump(res, open("results/flows.json", "w"), indent=2)
    print("\nwrote results/flows.json")


if __name__ == "__main__":
    main()
