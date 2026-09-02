"""Rulebook v2: VALIDATED, GATED convention remaps (fixes the v1 failure where free LLM rule-rewriting
corrupted correct predictions).

A rule = (from_class X, keyword pattern K, to_class Y): 'when the text-only classifier says X and the
text matches K, this institution files it as Y.' Rules are:
  * MINED only from RECURRING source disagreements (base=X, gold=Y), keyword-characterized;
  * VALIDATED on a held-out source split -- kept only if, where the rule fires, relabeling X->Y is
    net-positive (gold=Y beats gold=X by a margin with enough support);
  * APPLIED DETERMINISTICALLY to the held-out org's base predictions (can't override unmatched rows).
Placebo control: same fired rows, but Y targets shuffled -> must NOT help.

Reuses the base classifications already computed (BASE_TAG batches). No new LLM calls.

  python rulebook_v2.py 311
  python rulebook_v2.py cfpb
"""
import sys, os, re, json
import numpy as np
from collections import defaultdict, Counter
from openai_batch import collect_chat_batch

STOP = set("the a an and or of to in on for with is are was were be been being this that it its at by "
           "from as our we i you they he she my me not no there here up out over under near please "
           "have has had will would can could should about into your their his her them can't cannot".split())


def kw(text):
    toks = re.findall(r"[a-z]{3,}", text.lower())
    return [t for t in toks if t not in STOP]


def _load_311():
    from pilot_rulebook import parse_label, LABELS
    pool = json.load(open("results/rb_pool.json", encoding="utf-8"))
    res = collect_chat_batch(tag="rb_base", verbose=False)
    grp = {}
    for g in ("src", "test", "update"):
        rows = [(pool[g][i][0], pool[g][i][1]) for i in range(len(pool[g]))]   # (text, gold)
        base = [parse_label(res.get(f"{g}:{i}", "")) for i in range(len(pool[g]))]
        grp[g] = (rows, base)
    return grp, LABELS


def _load_cfpb():
    from cfpb_rulebook import parse_label
    pool = json.load(open("results/cfpb_rb_pool.json", encoding="utf-8")); LBL = pool["labels"]
    res = collect_chat_batch(tag="cfpb_rb_base", verbose=False)
    grp = {}
    for g in ("src", "test", "update"):
        rows = [(pool[g][i]["t"], pool[g][i]["p"]) for i in range(len(pool[g]))]
        base = [parse_label(res.get(f"{g}:{i}", ""), LBL) for i in range(len(pool[g]))]
        grp[g] = (rows, base)
    return grp, LBL


def mine_rules(rows, base, min_pair=6, min_kw=4, top_kw=6):
    """Candidate rules from recurring (base=X, gold=Y) disagreements, characterized by keywords."""
    by_pair = defaultdict(list)                       # (X,Y) -> row indices where base=X, gold=Y!=X
    for i, (t, y) in enumerate(rows):
        if base[i] not in ("UNPARSED",) and base[i] != y:
            by_pair[(base[i], y)].append(i)
    # background keyword freq (for distinctiveness)
    bg = Counter()
    for t, y in rows:
        bg.update(set(kw(t)))
    N = len(rows)
    rules = []
    for (X, Y), idxs in by_pair.items():
        if len(idxs) < min_pair:
            continue
        kc = Counter()
        for i in idxs:
            kc.update(set(kw(rows[i][0])))
        # distinctive keywords: frequent in this pair, not ubiquitous overall
        cand = [(k, c) for k, c in kc.items() if c >= min_kw and bg[k] < 0.30 * N]
        cand.sort(key=lambda kv: -kv[1])
        kws = [k for k, c in cand[:top_kw]]
        if kws:
            rules.append({"from": X, "to": Y, "kw": kws, "support_mine": len(idxs)})
    return rules


def fires(rule, text, base_pred):
    return base_pred == rule["from"] and any(k in text.lower() for k in rule["kw"])


def validate(rules, rows, base, min_support=4, margin=0.15):
    """Keep rules that, on this (validation) split, relabel X->Y net-positively where they fire."""
    kept = []
    for r in rules:
        fired = [i for i, (t, y) in enumerate(rows) if fires(r, t, base[i])]
        if len(fired) < min_support:
            continue
        gy = np.mean([rows[i][1] == r["to"] for i in fired])     # would-be-correct after remap
        gx = np.mean([rows[i][1] == r["from"] for i in fired])   # correct if left as base
        if gy - gx >= margin:
            r2 = dict(r); r2["val_fire"] = len(fired); r2["val_gain"] = round(float(gy - gx), 3)
            kept.append(r2)
    return kept


def apply_rules(rules, rows, base):
    pred = list(base)
    for i, (t, y) in enumerate(rows):
        for r in rules:                      # first matching rule wins
            if fires(r, t, pred[i] if pred[i] == r["from"] else base[i]) and base[i] == r["from"]:
                pred[i] = r["to"]; break
    return pred


def score(pred, rows, base):
    gold = [y for t, y in rows]; n = len(gold)
    hard = [i for i in range(n) if base[i] != gold[i]]
    easy = [i for i in range(n) if base[i] == gold[i]]
    acc = np.mean([pred[i] == gold[i] for i in range(n)])
    ah = np.mean([pred[i] == gold[i] for i in hard]) if hard else float("nan")
    rec = sum(1 for i in hard if pred[i] == gold[i]); broke = sum(1 for i in easy if pred[i] != gold[i])
    return {"acc_all": round(float(acc), 4), "acc_hard": round(float(ah), 4),
            "recovered": int(rec), "broke": int(broke), "n_hard": len(hard)}


def run(domain):
    grp, LBL = _load_311() if domain == "311" else _load_cfpb()
    (src_rows, src_base) = grp["src"]
    # split source into mine / validation
    rng = np.random.RandomState(0); perm = rng.permutation(len(src_rows))
    cut = int(0.6 * len(src_rows))
    mine_idx, val_idx = perm[:cut], perm[cut:]
    m_rows = [src_rows[i] for i in mine_idx]; m_base = [src_base[i] for i in mine_idx]
    v_rows = [src_rows[i] for i in val_idx]; v_base = [src_base[i] for i in val_idx]
    cand = mine_rules(m_rows, m_base)
    kept = validate(cand, v_rows, v_base)
    print(f"[{domain}] source={len(src_rows)} candidate_rules={len(cand)} validated_rules={len(kept)}")
    for r in sorted(kept, key=lambda r: -r["val_gain"])[:12]:
        print(f"   {r['from']:>22s} -> {r['to']:<22s} kw={r['kw'][:4]} val_gain={r['val_gain']} fire={r['val_fire']}")
    # placebo: shuffle the 'to' targets among kept rules
    tos = [r["to"] for r in kept]; rng.shuffle(tos)
    placebo = [dict(r, to=tos[i]) for i, r in enumerate(kept)]
    # transfer to held-out test
    t_rows, t_base = grp["test"]
    base_s = score(t_base, t_rows, t_base)
    rule_s = score(apply_rules(kept, t_rows, t_base), t_rows, t_base)
    plac_s = score(apply_rules(placebo, t_rows, t_base), t_rows, t_base)
    # + few-new-city update: mine+validate also on the held-out update split, add net-positive rules
    u_rows, u_base = grp["update"]
    upd_cand = mine_rules(u_rows, u_base, min_pair=3, min_kw=2)
    upd_kept = validate(upd_cand, u_rows, u_base, min_support=2, margin=0.0)
    updated = kept + upd_kept
    upd_s = score(apply_rules(updated, t_rows, t_base), t_rows, t_base)
    print(f"\n{'arm':16s}{'acc_all':>9s}{'acc_hard':>9s}{'recovered':>11s}{'broke':>7s}")
    for name, s in [("no_rules", base_s), ("source_rules", rule_s), ("placebo", plac_s), ("updated", upd_s)]:
        print(f"{name:16s}{s['acc_all']:9.4f}{s['acc_hard']:9.4f}{s['recovered']:11d}{s['broke']:7d}")
    out = {"domain": domain, "held_out_n": len(t_rows), "n_validated_rules": len(kept),
           "arms": {"no_rules": base_s, "source_rules": rule_s, "placebo": plac_s, "updated": upd_s},
           "rules": kept}
    json.dump(out, open(f"results/rulebook_v2_{domain}.json", "w"), indent=2)
    print("\n[invariants]")
    print(f"  rules beat no_rules? {rule_s['acc_all']} vs {base_s['acc_all']}  "
          f"{'PASS' if rule_s['acc_all']>base_s['acc_all'] else 'FAIL'}")
    print(f"  rules beat placebo? {rule_s['acc_all']} vs {plac_s['acc_all']}  "
          f"{'PASS' if rule_s['acc_all']>plac_s['acc_all'] else 'FAIL'}")
    print(f"  net non-destructive? broke={rule_s['broke']} recovered={rule_s['recovered']}  "
          f"{'PASS' if rule_s['recovered']>=rule_s['broke'] else 'CHECK'}")
    print(f"wrote results/rulebook_v2_{domain}.json")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "311")
