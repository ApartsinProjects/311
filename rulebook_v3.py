"""Rulebook v3: ITERATIVE LLM rule mining with a statistical validation gate (per user's design).

Loop: find the largest uncovered confusion (base=X, gold=Y); show the LLM a batch of those error
texts; the LLM names a concise distinguishing pattern + keywords; gate the candidate on a held-out
VALIDATION split (keep only if relabeling X->Y where it fires is net-positive); remove covered errors;
repeat until no confusion has enough uncovered errors or no candidate validates.

Evaluated two ways to separate 'does a convention exist' from 'does it transfer':
  within  = held-out slice of the SAME source cities (in-distribution)
  cross   = the held-out organization (transfer)
Placebo control: shuffle the 'to' labels of the kept rules -> must not help.

  python rulebook_v3.py 311
  python rulebook_v3.py cfpb
"""
import sys, os, json, re
import numpy as np
from collections import defaultdict, Counter
from openai_batch import client
from rulebook_v2 import _load_311, _load_cfpb, fires, apply_rules, validate, score, kw

MODEL = "gpt-4o-mini"


def llm_pattern(X, Y, texts):
    """Ask the LLM for keywords that distinguish 'file as Y though the text reads like X'."""
    sample = "\n".join(f"- {t[:200]}" for t in texts[:25])
    msg = (f"A text-only classifier labeled these municipal/administrative requests as '{X}', but the "
           f"organization actually filed them as '{Y}'. Find the COMMON textual pattern that signals "
           f"'{Y}' here.\n\nExamples:\n{sample}\n\n"
           "Reply with 3-7 lowercase keyword/phrase triggers (single words or short bigrams) that best "
           "capture this pattern, as a JSON list of strings. Only the JSON list.")
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=120,
            messages=[{"role": "system", "content": "You find concise, general textual triggers."},
                      {"role": "user", "content": msg}])
        m = re.search(r"\[.*\]", r.choices[0].message.content, re.S)
        kws = json.loads(m.group(0)) if m else []
        return [str(k).lower().strip() for k in kws if isinstance(k, str) and 2 <= len(str(k)) <= 30][:7]
    except Exception:
        return []


def iterative_mine(mine_rows, mine_base, val_rows, val_base, min_pair=6, max_rules=30):
    err = defaultdict(list)                          # (X,Y) -> mine indices, base=X gold=Y!=X
    for i, (t, y) in enumerate(mine_rows):
        if mine_base[i] not in ("UNPARSED",) and mine_base[i] != y:
            err[(mine_base[i], y)].append(i)
    covered = set()                                  # mine indices already explained
    kept = []
    attempted = set()
    while len(kept) < max_rules:
        # largest uncovered confusion
        ranked = sorted(((p, [i for i in idx if i not in covered]) for p, idx in err.items()),
                        key=lambda kv: -len(kv[1]))
        prog = False
        for (X, Y), idx in ranked:
            if (X, Y) in attempted or len(idx) < min_pair:
                continue
            attempted.add((X, Y))
            kws = llm_pattern(X, Y, [mine_rows[i][0] for i in idx])
            if not kws:
                continue
            cand = [{"from": X, "to": Y, "kw": kws}]
            good = validate(cand, val_rows, val_base, min_support=4, margin=0.15)
            if good:
                kept.append(good[0])
                for i in idx:                        # mark covered
                    if any(k in mine_rows[i][0].lower() for k in kws):
                        covered.add(i)
                prog = True
                break                                 # recompute ranking after each accepted rule
        if not prog:
            break
    return kept


def run(domain):
    grp, LBL = _load_311() if domain == "311" else _load_cfpb()
    src_rows, src_base = grp["src"]
    rng = np.random.RandomState(0); perm = rng.permutation(len(src_rows))
    n = len(src_rows); a, b = int(0.5*n), int(0.75*n)
    mine_i, val_i, wtest_i = perm[:a], perm[a:b], perm[b:]      # mine / validate / within-test
    def sub(ix): return [src_rows[i] for i in ix], [src_base[i] for i in ix]
    m_rows, m_base = sub(mine_i); v_rows, v_base = sub(val_i); w_rows, w_base = sub(wtest_i)
    kept = iterative_mine(m_rows, m_base, v_rows, v_base)
    print(f"[{domain}] validated rules = {len(kept)}")
    for r in kept[:14]:
        print(f"   {r['from']:>22s} -> {r['to']:<22s} kw={r['kw'][:5]}")
    tos = [r["to"] for r in kept]; rng.shuffle(tos)
    placebo = [dict(r, to=tos[i]) for i, r in enumerate(kept)]
    t_rows, t_base = grp["test"]
    res = {}
    for tag, (rows, base) in [("within", (w_rows, w_base)), ("cross", (t_rows, t_base))]:
        res[tag] = {"no_rules": score(base, rows, base),
                    "rules": score(apply_rules(kept, rows, base), rows, base),
                    "placebo": score(apply_rules(placebo, rows, base), rows, base)}
    print(f"\n{'setting':10s}{'arm':10s}{'acc_all':>9s}{'acc_hard':>9s}{'recovered':>11s}{'broke':>7s}")
    for tag in ("within", "cross"):
        for arm in ("no_rules", "rules", "placebo"):
            s = res[tag][arm]
            print(f"{tag:10s}{arm:10s}{s['acc_all']:9.4f}{s['acc_hard']:9.4f}{s['recovered']:11d}{s['broke']:7d}")
    json.dump({"domain": domain, "n_rules": len(kept), "results": res, "rules": kept},
              open(f"results/rulebook_v3_{domain}.json", "w"), indent=2)
    print("\n[invariants]")
    for tag in ("within", "cross"):
        r, nr, pl = res[tag]["rules"], res[tag]["no_rules"], res[tag]["placebo"]
        print(f"  {tag}: rules>{{no_rules,placebo}}? {r['acc_all']} vs {nr['acc_all']}/{pl['acc_all']}  "
              f"{'PASS' if r['acc_all']>nr['acc_all'] and r['acc_all']>=pl['acc_all'] else 'FAIL'}")
    print(f"wrote results/rulebook_v3_{domain}.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "311")
